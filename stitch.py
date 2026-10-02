"""
Multi-room stitching command.
Stitches multiple processed room outputs into a single floor plan.

Usage:
  python stitch.py --rooms <room1_folder> <room2_folder> ... --output <output_folder>
  
  OR load from previously saved report.json files:
  python stitch.py --reports room1_output/report.json room2_output/report.json --output stitched
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.stitcher import MultiRoomStitcher
from src.floor_plan import render_stitched_plan


def main():
    parser = argparse.ArgumentParser(description="Stitch multiple rooms into one floor plan")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--rooms", nargs="+",
        help="Room scan folders (each processed first with run.py)"
    )
    group.add_argument(
        "--reports", nargs="+",
        help="Paths to report.json files from already-processed rooms"
    )
    
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for stitched plan"
    )
    parser.add_argument(
        "--no-drift-correction",
        action="store_true",
        help="Disable drift correction (for ablation study)"
    )
    
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load room results
    room_results = {}
    
    if args.reports:
        for report_path in args.reports:
            with open(report_path) as f:
                data = json.load(f)
            for room_id, room_data in data.get("rooms", {}).items():
                room_results[room_id] = room_data
    else:
        for room_folder in args.rooms:
            room_path = Path(room_folder)
            report_file = room_path / "report.json"
            
            if not report_file.exists():
                # Try with _output suffix
                report_file = Path(str(room_path) + "_output") / "report.json"
            
            if not report_file.exists():
                print(f"WARNING: No report.json found for {room_folder}. "
                      f"Run: python run.py --input {room_folder} first")
                continue
            
            with open(report_file) as f:
                data = json.load(f)
            for room_id, room_data in data.get("rooms", {}).items():
                room_results[room_id] = room_data
    
    if not room_results:
        print("ERROR: No room data found. Process rooms first with run.py")
        sys.exit(1)
    
    print(f"Stitching {len(room_results)} rooms: {list(room_results.keys())}")
    
    # Extract geometries
    room_geometries = {rid: rdata.get("geometry", {}) for rid, rdata in room_results.items()}
    damage_by_room = {rid: rdata.get("damage", []) for rid, rdata in room_results.items()}
    
    # Stitch
    stitcher = MultiRoomStitcher(output_path, verbose=True)
    stitched = stitcher.stitch(
        room_results={rid: {"geometry": g} for rid, g in room_geometries.items()},
        apply_drift_correction=not args.no_drift_correction,
    )
    
    # Render stitched plan
    print("Rendering stitched floor plan...")
    stitched_geoms = {
        rid: rdata.get("geometry", {})
        for rid, rdata in stitched["rooms"].items()
    }
    
    png_path = render_stitched_plan(stitched_geoms, output_path, damage_by_room)
    print(f"Stitched plan: {png_path}")
    
    # Save stitched JSON
    out_json = output_path / "stitched_report.json"
    with open(out_json, "w") as f:
        json.dump(stitched, f, indent=2, default=str)
    print(f"Stitched report: {out_json}")
    
    total_area = stitched.get("total_floor_area_m2", 0)
    print(f"\nTotal property area: {total_area:.2f} m²")
    print(f"Drift correction: {'Applied' if stitched.get('drift_correction_applied') else 'Not applied'}")


if __name__ == "__main__":
    main()
