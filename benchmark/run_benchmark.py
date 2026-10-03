"""
Benchmark Harness
=================
Runs the pipeline on all benchmark datasets and computes accuracy metrics
against ground truth measurements (laser/tape measured values).

Usage:
  python benchmark/run_benchmark.py --ground-truth benchmark/ground_truth.json

Outputs:
  benchmark/results/benchmark_report.json
  benchmark/results/benchmark_report.md
  benchmark/results/accuracy_table.csv
"""

import argparse
import json
import sys
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RoomScanPipeline

# Configurable safe frame skips per dataset to prevent OOM
DEFAULT_FRAME_SKIPS = {
    "single_room": 20,
    "single_scan_floor_only": 40,
    "single_scan_with_ceiling": 80,
}
DEFAULT_FALLBACK_SKIP = 20


def run_benchmark(
    ground_truth_path: Path,
    datasets_path: Path,
    output_path: Path,
    tiers: List[str] = ["lidar"],
    frame_skip: Optional[int] = None,
) -> Dict:
    """
    Run benchmark across all datasets and tiers.
    
    Returns benchmark report dict.
    """
    print("=" * 70)
    print("  BRYNZ BENCHMARK HARNESS")
    print("=" * 70)
    
    # Load ground truth
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    
    results = {
        "tiers": {},
        "gates": {},
        "repeatability": {},
        "head_to_head": {},
    }
    
    for tier in tiers:
        print(f"\n[TIER: {tier.upper()}]")
        tier_results = []
        
        for scan_name, gt in ground_truth.get("scans", {}).items():
            scan_path = datasets_path / scan_name
            if not scan_path.exists():
                print(f"  Skipping {scan_name}: not found")
                continue
            
            skip = frame_skip if frame_skip is not None else DEFAULT_FRAME_SKIPS.get(scan_name, DEFAULT_FALLBACK_SKIP)
            print(f"  Processing: {scan_name} (frame_skip={skip})")
            
            out_dir = output_path / tier / scan_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                pipeline = RoomScanPipeline(
                    input_path=scan_path,
                    output_path=out_dir,
                    tier=tier,
                    frame_skip=skip,
                    room_id=scan_name,
                    enable_damage=True,
                    verbose=False,
                )
                result = pipeline.run()
                
                # Compare against ground truth
                metrics = compare_to_ground_truth(result, gt)
                tier_results.append({
                    "scan": scan_name,
                    "metrics": metrics,
                    "output_path": str(out_dir),
                })
                
                ch_err = metrics.get("ceiling_height_error_cm")
                if ch_err is not None:
                    print(f"    Ceiling height error: {ch_err:.1f}cm (GT: {gt.get('ceiling_height_m')}m)")
                else:
                    print("    Ceiling height error: not_scored (ground truth unavailable)")

                fa_err = metrics.get("floor_area_error_pct")
                if fa_err is not None:
                    print(f"    Floor area error: {fa_err:.1f}%")
                else:
                    print("    Floor area error: not_scored (ground truth unavailable)")
                
            except Exception as e:
                print(f"    ERROR: {type(e).__name__}: {e}")
                tier_results.append({
                    "scan": scan_name,
                    "error": f"{type(e).__name__}: {e}",
                    "metrics": None,
                })
        
        results["tiers"][tier] = tier_results
    
    # Compute gate pass/fail
    results["gates"] = compute_gates(results["tiers"])
    
    # Save report
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate markdown report
    md_path = output_path / "benchmark_report.md"
    generate_markdown_report(results, ground_truth, md_path)
    
    print(f"\nBenchmark complete. Report: {report_path}")
    return results


def compare_to_ground_truth(result: Dict, gt: Dict) -> Dict:
    """
    Compare pipeline output against ground truth measurements.
    Gracefully handles null or missing ground truth values.
    Returns dict of error metrics.
    """
    metrics = {
        "ceiling_height_error_cm": None,
        "ceiling_height_pass": None,
        "floor_area_error_pct": None,
        "wall_length_error_cm_mean": None,
        "wall_length_error_cm_max": None,
        "opening_width_error_cm_mean": None,
        "opening_width_pct_pass_2cm": None,
        "opening_width_pass": None,
        "status": "not_scored",
    }
    
    scored_items = 0

    for room_id, room_data in result.get("rooms", {}).items():
        geom = room_data.get("geometry", {})
        
        # Ceiling height error
        pred_height = geom.get("ceiling_height_m", 0)
        gt_height = gt.get("ceiling_height_m")
        if gt_height is not None:
            err_m = abs(pred_height - gt_height)
            metrics["ceiling_height_error_cm"] = err_m * 100
            metrics["ceiling_height_pred_m"] = pred_height
            metrics["ceiling_height_gt_m"] = gt_height
            metrics["ceiling_height_pass"] = err_m <= 0.015  # <=1.5cm gate
            scored_items += 1
        
        # Floor area error
        pred_area = geom.get("floor_area_m2", 0)
        gt_area = gt.get("floor_area_m2")
        if gt_area is not None and gt_area > 0:
            err_pct = abs(pred_area - gt_area) / gt_area * 100
            metrics["floor_area_error_pct"] = err_pct
            metrics["floor_area_pred_m2"] = pred_area
            metrics["floor_area_gt_m2"] = gt_area
            scored_items += 1
        
        # Wall length errors
        wall_errors = []
        for wall_gt in gt.get("walls", []):
            gt_length = wall_gt.get("length_m")
            if gt_length is None:
                continue
            
            # Find matching predicted wall (closest length)
            pred_walls = geom.get("walls", [])
            if pred_walls:
                pred_lengths = [w.get("length_m", 0) for w in pred_walls]
                closest = min(pred_lengths, key=lambda x: abs(x - gt_length))
                err_cm = abs(closest - gt_length) * 100
                wall_errors.append(err_cm)
        
        if wall_errors:
            metrics["wall_length_error_cm_mean"] = float(np.mean(wall_errors))
            metrics["wall_length_error_cm_max"] = float(np.max(wall_errors))
            scored_items += 1
        
        # Opening widths
        opening_errors = []
        for opening_gt in gt.get("openings", []):
            gt_width = opening_gt.get("width_m")
            if gt_width is None:
                continue
            pred_openings = geom.get("openings", [])
            if pred_openings:
                pred_widths = [o.get("width_m", 0) for o in pred_openings]
                closest = min(pred_widths, key=lambda x: abs(x - gt_width))
                err_cm = abs(closest - gt_width) * 100
                opening_errors.append(err_cm)
        
        if opening_errors:
            metrics["opening_width_error_cm_mean"] = float(np.mean(opening_errors))
            pct_pass = sum(1 for e in opening_errors if e <= 2.0) / len(opening_errors)
            metrics["opening_width_pct_pass_2cm"] = pct_pass
            metrics["opening_width_pass"] = pct_pass >= 0.85  # >=85% within 2cm gate
            scored_items += 1
    
    if scored_items > 0:
        metrics["status"] = "scored"

    return metrics


def compute_gates(tier_results: Dict) -> Dict:
    """Compute pass/fail for each benchmark gate."""
    gates = {}
    
    for tier, scans in tier_results.items():
        valid_scans = [s for s in scans if s.get("metrics")]
        
        if not valid_scans:
            gates[tier] = {
                "status": "NO_DATA",
                "ceiling_height_gate_pass": None,
                "ceiling_height_worst_cm": None,
                "opening_width_gate_pass": None,
                "n_scans_evaluated": 0,
            }
            continue
        
        scored_ch = [
            s["metrics"]["ceiling_height_error_cm"]
            for s in valid_scans
            if s["metrics"].get("ceiling_height_error_cm") is not None
        ]
        scored_ow = [
            s["metrics"]["opening_width_pass"]
            for s in valid_scans
            if s["metrics"].get("opening_width_pass") is not None
        ]
        
        if not scored_ch and not scored_ow:
            gates[tier] = {
                "status": "NOT_SCORED",
                "ceiling_height_gate_pass": None,
                "ceiling_height_worst_cm": None,
                "opening_width_gate_pass": None,
                "n_scans_evaluated": len(valid_scans),
                "note": "Ground truth null for all evaluated metrics",
            }
            continue

        gates[tier] = {
            "status": "EVALUATED",
            "ceiling_height_gate_pass": all(e <= 1.5 for e in scored_ch) if scored_ch else None,
            "ceiling_height_worst_cm": max(scored_ch) if scored_ch else None,
            "opening_width_gate_pass": all(scored_ow) if scored_ow else None,
            "n_scans_evaluated": len(valid_scans),
        }
    
    return gates


def generate_markdown_report(
    results: Dict,
    ground_truth: Dict,
    output_path: Path,
):
    """Generate a markdown benchmark report safely handling missing values."""
    lines = [
        "# Brynz Pipeline — Benchmark Report\n",
        "## Gate Summary\n",
        "| Tier | Ceiling Height Gate (≤1.5cm) | Opening Width Gate (≥85%@2cm) | Pass? |",
        "|------|------------------------------|-------------------------------|-------|",
    ]
    
    for tier, gate in results.get("gates", {}).items():
        status = gate.get("status", "UNKNOWN")
        ch_pass_val = gate.get("ceiling_height_gate_pass")
        ow_pass_val = gate.get("opening_width_gate_pass")
        worst_cm = gate.get("ceiling_height_worst_cm")

        if ch_pass_val is True:
            ch_pass = "✅"
        elif ch_pass_val is False:
            ch_pass = "❌"
        else:
            ch_pass = "N/A"

        if ow_pass_val is True:
            ow_pass = "✅"
        elif ow_pass_val is False:
            ow_pass = "❌"
        else:
            ow_pass = "N/A"

        if isinstance(worst_cm, (int, float)):
            ch_err = f"{worst_cm:.1f}cm"
        else:
            ch_err = "N/A"

        if status == "NOT_SCORED":
            overall = "⚠️ NOT SCORED (GT null)"
        elif status == "NO_DATA":
            overall = "❌ NO DATA"
        elif ch_pass_val and ow_pass_val:
            overall = "✅ PASS"
        elif ch_pass_val is False or ow_pass_val is False:
            overall = "❌ FAIL"
        else:
            overall = "N/A"

        lines.append(f"| {tier} | {ch_pass} ({ch_err}) | {ow_pass} | {overall} |")
    
    lines += [
        "\n## Per-Scan Results\n",
    ]
    
    for tier, scans in results.get("tiers", {}).items():
        lines.append(f"### Tier: {tier.upper()}\n")
        lines.append("| Scan | Ceiling H Error | Floor Area Error | Wall Length Error | Status |")
        lines.append("|------|-----------------|------------------|-------------------|--------|")
        
        for scan in scans:
            name = scan.get("scan", "unknown")
            if scan.get("error"):
                lines.append(f"| {name} | ERROR | ERROR | ERROR | ❌ {scan.get('error')} |")
                continue
            m = scan.get("metrics", {}) or {}
            ch_val = m.get("ceiling_height_error_cm")
            fa_val = m.get("floor_area_error_pct")
            wl_val = m.get("wall_length_error_cm_mean")

            ch = f"{ch_val:.1f}cm" if isinstance(ch_val, (int, float)) else "not_scored"
            fa = f"{fa_val:.1f}%" if isinstance(fa_val, (int, float)) else "not_scored"
            wl = f"{wl_val:.1f}cm" if isinstance(wl_val, (int, float)) else "not_scored"
            st = m.get("status", "unknown")
            lines.append(f"| {name} | {ch} | {fa} | {wl} | {st} |")
        lines.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Brynz benchmark")
    parser.add_argument("--ground-truth", required=True, help="Ground truth JSON file")
    parser.add_argument("--datasets", default=".", help="Path containing scan folders")
    parser.add_argument("--output", default="benchmark/results", help="Output directory")
    parser.add_argument("--tier", nargs="+", default=["lidar"], 
                       choices=["lidar", "video", "photo"],
                       help="Tiers to benchmark")
    parser.add_argument("--frame-skip", type=int, default=None,
                       help="Override frame skip for all datasets (default: dataset-adaptive)")
    
    args = parser.parse_args()
    
    run_benchmark(
        ground_truth_path=Path(args.ground_truth),
        datasets_path=Path(args.datasets),
        output_path=Path(args.output),
        tiers=args.tier,
        frame_skip=args.frame_skip,
    )
