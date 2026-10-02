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
from typing import Dict, List, Any
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RoomScanPipeline


def run_benchmark(
    ground_truth_path: Path,
    datasets_path: Path,
    output_path: Path,
    tiers: List[str] = ["lidar", "video", "photo"],
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
            
            print(f"  Processing: {scan_name}")
            
            out_dir = output_path / tier / scan_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                pipeline = RoomScanPipeline(
                    input_path=scan_path,
                    output_path=out_dir,
                    tier=tier,
                    frame_skip=5,
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
                
                print(f"    Ceiling height error: {metrics['ceiling_height_error_cm']:.1f}cm "
                      f"(GT: {gt.get('ceiling_height_m', '?')}m)")
                print(f"    Floor area error: {metrics['floor_area_error_pct']:.1f}%")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                tier_results.append({
                    "scan": scan_name,
                    "error": str(e),
                    "metrics": None,
                })
        
        results["tiers"][tier] = tier_results
    
    # Compute gate pass/fail
    results["gates"] = compute_gates(results["tiers"])
    
    # Save report
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
    Returns dict of error metrics.
    """
    metrics = {}
    
    for room_id, room_data in result.get("rooms", {}).items():
        geom = room_data.get("geometry", {})
        
        # Ceiling height error
        pred_height = geom.get("ceiling_height_m", 0)
        gt_height = gt.get("ceiling_height_m")
        if gt_height:
            err_m = abs(pred_height - gt_height)
            metrics["ceiling_height_error_cm"] = err_m * 100
            metrics["ceiling_height_pred_m"] = pred_height
            metrics["ceiling_height_gt_m"] = gt_height
            metrics["ceiling_height_pass"] = err_m <= 0.015  # ≤1.5cm gate
        
        # Floor area error
        pred_area = geom.get("floor_area_m2", 0)
        gt_area = gt.get("floor_area_m2")
        if gt_area and gt_area > 0:
            err_pct = abs(pred_area - gt_area) / gt_area * 100
            metrics["floor_area_error_pct"] = err_pct
            metrics["floor_area_pred_m2"] = pred_area
            metrics["floor_area_gt_m2"] = gt_area
        
        # Wall length errors
        wall_errors = []
        for wall_gt in gt.get("walls", []):
            wall_id = wall_gt.get("id")
            gt_length = wall_gt.get("length_m")
            
            # Find matching predicted wall (closest length)
            pred_walls = geom.get("walls", [])
            if pred_walls and gt_length:
                pred_lengths = [w.get("length_m", 0) for w in pred_walls]
                closest = min(pred_lengths, key=lambda x: abs(x - gt_length))
                err_cm = abs(closest - gt_length) * 100
                wall_errors.append(err_cm)
        
        if wall_errors:
            metrics["wall_length_error_cm_mean"] = float(np.mean(wall_errors))
            metrics["wall_length_error_cm_max"] = float(np.max(wall_errors))
        
        # Opening widths
        opening_errors = []
        for opening_gt in gt.get("openings", []):
            gt_width = opening_gt.get("width_m")
            if not gt_width:
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
            metrics["opening_width_pass"] = pct_pass >= 0.85  # ≥85% within 2cm gate
    
    return metrics


def compute_gates(tier_results: Dict) -> Dict:
    """Compute pass/fail for each benchmark gate."""
    gates = {}
    
    for tier, scans in tier_results.items():
        valid_scans = [s for s in scans if s.get("metrics")]
        
        if not valid_scans:
            gates[tier] = {"status": "NO_DATA"}
            continue
        
        ceiling_errs = [
            s["metrics"].get("ceiling_height_error_cm", 999)
            for s in valid_scans
        ]
        opening_pass = [
            s["metrics"].get("opening_width_pass", False)
            for s in valid_scans
        ]
        
        gates[tier] = {
            # Gate: ceiling height ≤1.5cm
            "ceiling_height_gate_pass": all(e <= 1.5 for e in ceiling_errs),
            "ceiling_height_worst_cm": max(ceiling_errs) if ceiling_errs else 999,
            # Gate: opening widths ≥85% within 2cm
            "opening_width_gate_pass": all(opening_pass),
            "n_scans_evaluated": len(valid_scans),
        }
    
    return gates


def generate_markdown_report(
    results: Dict,
    ground_truth: Dict,
    output_path: Path,
):
    """Generate a markdown benchmark report."""
    lines = [
        "# Brynz Pipeline — Benchmark Report\n",
        "## Gate Summary\n",
        "| Tier | Ceiling Height Gate (≤1.5cm) | Opening Width Gate (≥85%@2cm) | Pass? |",
        "|------|------------------------------|-------------------------------|-------|",
    ]
    
    for tier, gate in results.get("gates", {}).items():
        ch_pass = "✅" if gate.get("ceiling_height_gate_pass") else "❌"
        ow_pass = "✅" if gate.get("opening_width_gate_pass") else "❌"
        ch_err = f"{gate.get('ceiling_height_worst_cm', '?'):.1f}cm"
        overall = "✅ PASS" if (gate.get("ceiling_height_gate_pass") and gate.get("opening_width_gate_pass")) else "❌ FAIL"
        lines.append(f"| {tier} | {ch_pass} ({ch_err}) | {ow_pass} | {overall} |")
    
    lines += [
        "\n## Per-Scan Results\n",
    ]
    
    for tier, scans in results.get("tiers", {}).items():
        lines.append(f"### Tier: {tier.upper()}\n")
        lines.append("| Scan | Ceiling H Error | Floor Area Error | Wall Length Error |")
        lines.append("|------|-----------------|------------------|-------------------|")
        
        for scan in scans:
            m = scan.get("metrics", {}) or {}
            name = scan.get("scan", "unknown")
            ch = f"{m.get('ceiling_height_error_cm', '?'):.1f}cm" if "ceiling_height_error_cm" in m else "N/A"
            fa = f"{m.get('floor_area_error_pct', '?'):.1f}%" if "floor_area_error_pct" in m else "N/A"
            wl = f"{m.get('wall_length_error_cm_mean', '?'):.1f}cm" if "wall_length_error_cm_mean" in m else "N/A"
            lines.append(f"| {name} | {ch} | {fa} | {wl} |")
        lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Brynz benchmark")
    parser.add_argument("--ground-truth", required=True, help="Ground truth JSON file")
    parser.add_argument("--datasets", default=".", help="Path containing scan folders")
    parser.add_argument("--output", default="benchmark/results", help="Output directory")
    parser.add_argument("--tier", nargs="+", default=["lidar"], 
                       choices=["lidar", "video", "photo"],
                       help="Tiers to benchmark")
    
    args = parser.parse_args()
    
    run_benchmark(
        ground_truth_path=Path(args.ground_truth),
        datasets_path=Path(args.datasets),
        output_path=Path(args.output),
        tiers=args.tier,
    )
