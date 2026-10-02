"""
Core pipeline orchestrator for Brynz Room Scanner.
Handles all three tiers: LiDAR, Video, Photo.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from .lidar_processor import LiDARProcessor
from .geometry import GeometryExtractor
from .floor_plan import FloorPlanRenderer
from .damage_detector import DamageDetector
from .stitcher import MultiRoomStitcher
from .output_schema import OutputSchema

logger = logging.getLogger(__name__)


class RoomScanPipeline:
    """
    Main pipeline that orchestrates all processing steps.
    
    Steps:
      1. Load sensor data (tier-dependent)
      2. Build 3D point cloud
      3. Extract room geometry (walls, floor, ceiling, openings)
      4. Generate floor plan
      5. Detect damage (optional)
      6. Stitch with other rooms (optional)
      7. Write outputs (JSON, PNG, SVG, CSV)
    """

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        tier: str = "lidar",
        frame_skip: int = 5,
        confidence_threshold: int = 1,
        room_id: str = "room_001",
        enable_damage: bool = True,
        verbose: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.tier = tier
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.room_id = room_id
        self.enable_damage = enable_damage
        self.verbose = verbose

        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            level=level
        )

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        
        print(f"[1/6] Loading {self.tier} data...")
        point_cloud, frames_rgb = self._load_data()
        print(f"      Point cloud: {len(point_cloud['xyz'])} points")

        print("[2/6] Extracting room geometry...")
        geom_extractor = GeometryExtractor(verbose=self.verbose)
        geometry = geom_extractor.extract(point_cloud)
        self._print_geometry(geometry)

        print("[3/6] Rendering floor plan...")
        renderer = FloorPlanRenderer(output_path=self.output_path, room_id=self.room_id)
        floor_plan_path = renderer.render(geometry)
        print(f"      Saved: {floor_plan_path}")

        damage_regions = []
        if self.enable_damage and frames_rgb:
            print("[4/6] Detecting damage...")
            detector = DamageDetector(verbose=self.verbose)
            damage_regions = detector.detect(frames_rgb, geometry, point_cloud)
            print(f"      Found {len(damage_regions)} damage regions")
        else:
            print("[4/6] Damage detection skipped")

        print("[5/6] Building output schema...")
        schema = OutputSchema()
        result = schema.build(
            room_id=self.room_id,
            geometry=geometry,
            damage_regions=damage_regions,
            tier=self.tier,
            processing_time_s=time.time() - t0,
        )

        print("[6/6] Writing outputs...")
        self._write_outputs(result)

        elapsed = time.time() - t0
        print(f"\n  Total processing time: {elapsed:.1f}s")
        return result

    def _load_data(self):
        """Load data based on tier."""
        if self.tier == "lidar":
            processor = LiDARProcessor(
                input_path=self.input_path,
                frame_skip=self.frame_skip,
                confidence_threshold=self.confidence_threshold,
                verbose=self.verbose,
            )
            return processor.load()
        elif self.tier == "video":
            from .tiers.video_processor import VideoProcessor
            processor = VideoProcessor(
                input_path=self.input_path,
                frame_skip=self.frame_skip,
                verbose=self.verbose,
            )
            return processor.load()
        elif self.tier == "photo":
            from .tiers.photo_processor import PhotoProcessor
            processor = PhotoProcessor(
                input_path=self.input_path,
                verbose=self.verbose,
            )
            return processor.load()
        else:
            raise ValueError(f"Unknown tier: {self.tier}")

    def _print_geometry(self, geometry: Dict):
        g = geometry
        print(f"      Floor area:     {g['floor_area_m2']:.2f} ± {g['floor_area_ci_m2']:.2f} m²")
        print(f"      Ceiling height: {g['ceiling_height_m']:.3f} ± {g['ceiling_height_ci_m']:.3f} m")
        print(f"      Walls detected: {len(g['walls'])}")
        print(f"      Openings:       {len(g['openings'])}")

    def _write_outputs(self, result: Dict):
        # JSON report
        report_path = self.output_path / "report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"      report.json → {report_path}")

        # Scope CSV
        scope_path = self.output_path / "scope_items.csv"
        self._write_scope_csv(result, scope_path)
        print(f"      scope_items.csv → {scope_path}")

    def _write_scope_csv(self, result: Dict, path: Path):
        import csv
        items = []
        for room_id, room_data in result.get("rooms", {}).items():
            for item in room_data.get("scope_items", []):
                items.append({
                    "room_id": room_id,
                    "surface": item.get("surface", ""),
                    "item": item.get("description", ""),
                    "quantity": item.get("quantity", ""),
                    "unit": item.get("unit", ""),
                    "damage_class": item.get("damage_class", ""),
                })
        with open(path, "w", newline="") as f:
            if items:
                writer = csv.DictWriter(f, fieldnames=items[0].keys())
                writer.writeheader()
                writer.writerows(items)
