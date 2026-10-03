"""
Brynz Room Scanner Pipeline
============================
One command per capture:
  python run.py --input <scan_folder> [--tier lidar|video|photo] [--output <output_dir>]

Produces:
  - floor_plan.png         : rendered dimensioned floor plan
  - report.json            : full output schema (measurements, damage, confidence intervals)
  - floor_plan.svg         : vector floor plan
  - scope_items.csv        : repair scope line items
"""

import argparse
import sys
from pathlib import Path
from src.pipeline import RoomScanPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Brynz Room Scanner — LiDAR/Video/Photo to Floor Plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to scan folder (containing depth/, confidence/, odometry.csv, rgb.mp4, camera_matrix.csv)"
    )
    parser.add_argument(
        "--tier",
        choices=["lidar", "video", "photo"],
        default="lidar",
        help="Input tier (default: lidar). photo mode: --input is a folder of images."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: <input>_output)"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame for speed (default: 5)"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Minimum LiDAR confidence level to include (0=low,1=medium,2=high). Default: 1"
    )
    parser.add_argument(
        "--room-id",
        default="room_001",
        help="Room identifier for multi-room stitching (default: room_001)"
    )
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Stitch with previously processed rooms in output directory"
    )
    parser.add_argument(
        "--damage",
        action="store_true",
        default=True,
        help="Enable damage detection (default: True)"
    )
    parser.add_argument(
        "--no-damage",
        action="store_false",
        dest="damage",
        help="Disable damage detection"
    )
    parser.add_argument(
        "--drift-correction",
        action="store_true",
        default=True,
        help="Enable trajectory drift correction / loop closure (default: True)"
    )
    parser.add_argument(
        "--no-drift-correction",
        action="store_false",
        dest="drift_correction",
        help="Disable trajectory drift correction (ablation: open-loop odometry)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.name}_output"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Brynz Room Scanner Pipeline")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")
    print(f"  Tier:   {args.tier}")
    print(f"  Output: {output_path}")
    print(f"  Drift:  {'ON (loop closure enabled)' if args.drift_correction else 'OFF (open-loop odometry)'}")
    print(f"{'='*60}\n")

    pipeline = RoomScanPipeline(
        input_path=input_path,
        output_path=output_path,
        tier=args.tier,
        frame_skip=args.frame_skip,
        confidence_threshold=args.confidence_threshold,
        room_id=args.room_id,
        enable_damage=args.damage,
        apply_drift_correction=args.drift_correction,
        verbose=args.verbose,
    )

    result = pipeline.run()

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for room_id, room_data in result["rooms"].items():
        geom = room_data["geometry"]
        print(f"  Room: {room_id}")
        print(f"    Floor area:     {geom['floor_area_m2']:.2f} ± {geom['floor_area_ci_m2']:.2f} m²")
        print(f"    Ceiling height: {geom['ceiling_height_m']:.3f} ± {geom['ceiling_height_ci_m']:.3f} m")
        print(f"    Walls:          {len(geom['walls'])}")
        print(f"    Openings:       {len(geom['openings'])}")
        if "damage" in room_data and room_data["damage"]:
            print(f"    Damage regions: {len(room_data['damage'])}")
    print(f"\n  Output written to: {output_path}")
    print(f"  Floor plan:  {output_path}/floor_plan.png")
    print(f"  JSON report: {output_path}/report.json")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
