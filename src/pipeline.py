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
        apply_drift_correction: bool = True,
        verbose: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.tier = tier
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.room_id = room_id
        self.enable_damage = enable_damage
        self.apply_drift_correction = apply_drift_correction
        self.verbose = verbose
        self.drift_info = None
        self.depth_info = None

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

        # Connected space segmentation (CP4)
        connected_spaces_graph = None
        try:
            from .connected_spaces import ConnectedSpaceSegmenter
            segmenter = ConnectedSpaceSegmenter()
            cs_graph  = segmenter.segment(
                geometry, xyz=point_cloud["xyz"], session_id=self.input_path.name
            )
            connected_spaces_graph = cs_graph.to_dict()
            status = connected_spaces_graph.get("segmentation_status", "?")
            n_sp   = connected_spaces_graph.get("n_spaces", 0)
            print(f"      Spaces:         {n_sp} ({status})")
        except Exception as _e:
            logger.debug(f"Connected space segmentation skipped: {_e}")

        # Debug visualizations (best-effort)
        try:
            from .debug_visualizer import render_debug
            from .lidar_processor import LiDARProcessor
            _proc = LiDARProcessor(self.input_path, frame_skip=self.frame_skip)
            _poses = _proc._load_odometry()
            render_debug(point_cloud["xyz"], geometry, _poses,
                         self.output_path / "debug",
                         tag=self.input_path.name)
        except Exception as _e:
            logger.debug(f"Debug visualizer skipped: {_e}")

        print("[3/6] Rendering floor plan...")
        renderer = FloorPlanRenderer(output_path=self.output_path, room_id=self.room_id)
        if (
            connected_spaces_graph
            and connected_spaces_graph.get("n_spaces", 0) > 1
            and connected_spaces_graph.get("segmentation_status") == "complete"
        ):
            try:
                from .floor_plan import render_multi_space_plan
                multi_spaces = connected_spaces_graph.get("spaces", {})
                floor_plan_path = render_multi_space_plan(
                    multi_spaces,
                    output_path=self.output_path,
                    title=f"Floor Plan — {self.room_id} ({len(multi_spaces)} connected spaces)",
                )
            except Exception as _me:
                logger.debug(f"Multi-space floor plan render fallback: {_me}")
                floor_plan_path = renderer.render(geometry)
        else:
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
            connected_spaces_graph=connected_spaces_graph,
            drift_info=self.drift_info,
            depth_info=self.depth_info,
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
                apply_drift_correction=self.apply_drift_correction,
                verbose=self.verbose,
            )
            data = processor.load()
            self.drift_info = getattr(processor, "drift_info", None)
            return data
        elif self.tier == "video":
            from .tiers.video_processor import VideoProcessor
            processor = VideoProcessor(
                input_path=self.input_path,
                frame_skip=self.frame_skip,
                verbose=self.verbose,
            )
            data = processor.load()
            self.depth_info = getattr(processor, "depth_info", None)
            return data
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
        src   = g.get('floor_area_source', '?')
        ratio = g.get('debug', {}).get('bbox_polygon_ratio', '?')
        multi = ' [MULTI-SPACE?]' if isinstance(ratio, float) and ratio > 3.0 else ''
        print(f"      Floor area:     {g['floor_area_m2']:.2f} +/- {g['floor_area_ci_m2']:.2f} m2  (source: {src})")
        print(f"      Ceiling height: {g['ceiling_height_m']:.3f} +/- {g['ceiling_height_ci_m']:.3f} m  (reliable: {g.get('ceiling_detection_reliable','?')})")
        print(f"      Room polygon:   {g.get('room_perimeter_m',0):.2f} m perimeter")
        print(f"      Walls detected: {len(g['walls'])}")
        print(f"      Openings:       {len(g['openings'])} (conservative, unverified)")
        print(f"      Bbox/polygon:   {ratio}{multi}")


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
