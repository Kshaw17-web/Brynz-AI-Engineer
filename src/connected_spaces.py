"""
Connected-Space / Multi-Room Geometry
=======================================
CP4: Adds the data structures needed to represent multiple connected spaces
     discovered in a single scan (e.g. single_scan_floor_only may contain
     multiple rooms or corridors).

Schema:
  SpaceGeometry: geometry for one space (room, corridor, etc.)
  ConnectedSpaceGraph: full graph of spaces + adjacency

If reliable segmentation cannot be established from the data, the
segmentation status is marked as 'incomplete' rather than fabricating rooms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpaceGeometry:
    """
    Geometry for one detected space (room, corridor, alcove, etc.).
    All lengths in metres. Areas in m².
    Coordinate system: X-Z is the horizontal plane (Y is up).
    """
    space_id:            str
    space_type:          str = "unknown"   # "room", "corridor", "open_plan", etc.

    # Footprint
    polygon:             List[List[float]] = field(default_factory=list)
    area_m2:             float = 0.0
    area_ci_m2:          float = 0.0
    perimeter_m:         float = 0.0
    polygon_source:      str = "unknown"   # "wall_intersections" | "convex_hull_fallback"

    # Vertical
    floor_y:             float = 0.0
    ceiling_y:           float = 0.0
    ceiling_height_m:    float = 0.0
    ceiling_height_ci_m: float = 0.0
    ceiling_reliable:    bool = False

    # Walls belonging to this space
    walls:               List[Dict] = field(default_factory=list)

    # Openings (doors/windows) on boundary walls
    openings:            List[Dict] = field(default_factory=list)

    # Boundary wall IDs that are shared with an adjacent space
    shared_wall_ids:     List[str] = field(default_factory=list)

    # Confidence / assumptions
    assumptions:         List[str] = field(default_factory=list)
    confidence:          str = "low"   # "low" | "medium" | "high"

    # Debug
    debug:               Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class SpaceAdjacency:
    """An adjacency relationship between two spaces."""
    space_a:       str
    space_b:       str
    shared_wall_ids: List[str] = field(default_factory=list)
    opening_ids:   List[str] = field(default_factory=list)
    connection_type: str = "wall"   # "wall" | "opening" | "inferred"
    confidence:    str = "low"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConnectedSpaceGraph:
    """
    Full graph of spaces discovered from one scan session.

    segmentation_status:
      "complete"   — reliable space boundaries established
      "partial"    — some spaces identified but boundaries uncertain
      "incomplete" — insufficient evidence; single-space assumption used
    """
    session_id:          str = "scan_001"
    segmentation_status: str = "incomplete"
    segmentation_method: str = "none"
    spaces:              Dict[str, SpaceGeometry] = field(default_factory=dict)
    adjacency:           List[SpaceAdjacency] = field(default_factory=list)
    notes:               List[str] = field(default_factory=list)

    def add_space(self, space: SpaceGeometry):
        self.spaces[space.space_id] = space

    def add_adjacency(self, adj: SpaceAdjacency):
        self.adjacency.append(adj)

    def total_area_m2(self) -> float:
        return sum(s.area_m2 for s in self.spaces.values())

    def to_dict(self) -> Dict:
        return {
            "session_id":          self.session_id,
            "segmentation_status": self.segmentation_status,
            "segmentation_method": self.segmentation_method,
            "n_spaces":            len(self.spaces),
            "total_area_m2":       round(self.total_area_m2(), 4),
            "spaces": {sid: s.to_dict() for sid, s in self.spaces.items()},
            "adjacency":           [a.to_dict() for a in self.adjacency],
            "notes":               self.notes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-space segmenter
# ─────────────────────────────────────────────────────────────────────────────

class ConnectedSpaceSegmenter:
    """
    Attempts to split a point cloud / geometry into connected spaces.

    Strategy:
    1. Start from the full geometry (walls, polygon) of a single scan.
    2. Compute the ratio of floor-point bounding box to polygon area.
       If ratio > MULTI_SPACE_RATIO, flag as possible multi-space.
    3. Attempt to partition the wall set into groups by spatial proximity.
    4. For each group, re-run the polygon construction.
    5. If partitioning is unreliable (e.g. walls too sparse, no clear gaps),
       return a single-space graph with status='incomplete'.

    This deliberately avoids fabricating room boundaries.
    """

    MULTI_SPACE_RATIO = 3.0   # floor_bbox / polygon_area > this → likely multi-space
    MIN_SPACE_AREA_M2 = 2.0   # minimum plausible space area
    MAX_WALL_GAP_M    = 2.5   # max gap between wall clusters to form separate spaces

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def segment(
        self,
        geometry: Dict,
        xyz: Optional[np.ndarray] = None,
        session_id: str = "scan_001",
    ) -> ConnectedSpaceGraph:
        """
        Attempt segmentation. Returns a ConnectedSpaceGraph.
        If segmentation is not reliable, returns a graph with one space
        and segmentation_status='incomplete'.
        """
        graph = ConnectedSpaceGraph(session_id=session_id)
        walls  = geometry.get("walls", [])
        polygon = geometry.get("room_polygon") or geometry.get("footprint_polygon", [])

        if not walls or len(polygon) < 3:
            graph.segmentation_status = "incomplete"
            graph.notes.append("Insufficient geometry for segmentation")
            space = self._geometry_to_space("space_000", geometry, "unknown")
            graph.add_space(space)
            return graph

        # ── Check multi-space ratio ──────────────────────────────────────────
        ratio = self._bbox_polygon_ratio(xyz, np.array(polygon), geometry)
        logger.info(f"  Multi-space check: bbox/polygon ratio = {ratio:.2f}")

        n_walls  = len(walls)
        n_az_fam = len(geometry.get("debug", {}).get("orthogonal_families_deg", []))

        multi_space_evidence = (
            ratio > self.MULTI_SPACE_RATIO or
            n_walls > 10
        )

        if not multi_space_evidence:
            # Single space — reliable
            graph.segmentation_status = "complete"
            graph.segmentation_method = "single_space"
            space = self._geometry_to_space("space_000", geometry, self._infer_type(geometry))
            space.confidence = "medium"
            graph.add_space(space)
            graph.notes.append(
                f"Single space detected (bbox/polygon={ratio:.1f}, walls={n_walls})"
            )
            return graph

        # ── Attempt wall-cluster partitioning ────────────────────────────────
        logger.info(f"  Multi-space evidence (ratio={ratio:.1f}, walls={n_walls}). Attempting partition.")
        spaces_geom = self._partition_walls(walls, geometry)

        if spaces_geom is None or len(spaces_geom) < 2:
            graph.segmentation_status = "incomplete"
            graph.segmentation_method = "partition_failed"
            space = self._geometry_to_space("space_000", geometry, "open_plan_or_multi_room")
            space.confidence = "low"
            space.assumptions.append(
                f"bbox/polygon ratio={ratio:.1f} suggests multiple connected spaces "
                f"but partition was unreliable. Treated as single geometry."
            )
            graph.add_space(space)
            graph.notes.append(
                f"Multi-space evidence found (ratio={ratio:.1f}) but segmentation incomplete. "
                f"Manual inspection recommended."
            )
            return graph

        # ── Multiple spaces found ────────────────────────────────────────────
        graph.segmentation_status = "partial"
        graph.segmentation_method = "wall_cluster_partition"
        graph.notes.append(
            f"Partitioned {n_walls} walls into {len(spaces_geom)} space candidates"
        )
        for i, sg in enumerate(spaces_geom):
            space = self._geometry_to_space(f"space_{i:03d}", sg, "room")
            space.confidence = "low"
            space.assumptions.append(
                "Space boundary derived from wall cluster partition. "
                "Verify against physical layout."
            )
            graph.add_space(space)

        return graph

    def _bbox_polygon_ratio(
        self, xyz: Optional[np.ndarray],
        polygon: np.ndarray,
        geometry: Dict,
    ) -> float:
        """Ratio of floor-point bounding box area to polygon area."""
        from src.geometry import GeometryExtractor
        floor_y = geometry.get("floor_y", 0.0)

        if xyz is not None:
            floor_pts = xyz[xyz[:, 1] < floor_y + 0.15][:, [0, 2]]
            if len(floor_pts) > 10:
                bbox_area = float(
                    (floor_pts[:, 0].max() - floor_pts[:, 0].min()) *
                    (floor_pts[:, 1].max() - floor_pts[:, 1].min())
                )
            else:
                bbox_area = 1.0
        else:
            bbox_area = geometry.get("debug", {}).get("wall_zone_pts", 1.0) / 100

        poly_area = geometry.get("floor_area_m2", 1.0)
        return float(bbox_area) / max(0.1, float(poly_area))

    def _partition_walls(self, walls: List[Dict], geometry: Dict) -> Optional[List[Dict]]:
        """
        Try to partition walls into spatial clusters.
        Returns list of mini-geometry dicts, or None if unreliable.
        """
        if len(walls) < 4:
            return None

        # Group walls by centroid (midpoint of start/end)
        centres = []
        for w in walls:
            s = np.array(w.get("start_xz", [0, 0]))
            e = np.array(w.get("end_xz", [0, 0]))
            centres.append((s + e) / 2)

        centres = np.array(centres)

        # Simple distance-based clustering
        from scipy.cluster.hierarchy import fcluster, linkage
        if len(centres) < 4:
            return None

        try:
            Z = linkage(centres, method="ward")
            # Cut at MAX_WALL_GAP_M distance
            labels = fcluster(Z, t=self.MAX_WALL_GAP_M * 2, criterion="distance")
            unique_labels = np.unique(labels)
        except Exception as e:
            logger.warning(f"  Wall clustering failed: {e}")
            return None

        if len(unique_labels) < 2:
            return None

        spaces = []
        for lab in unique_labels:
            group_walls = [w for w, l in zip(walls, labels) if l == lab]
            if len(group_walls) < 2:
                continue

            # Build minimal geometry for this group
            mini_geom = {k: v for k, v in geometry.items()}
            mini_geom["walls"] = group_walls
            # Rebuild polygon from these walls only
            try:
                from src.geometry import GeometryExtractor
                ge = GeometryExtractor()
                floor_y = geometry.get("floor_y", 0.0)
                result = ge._build_room_polygon(
                    group_walls, floor_y,
                    np.zeros((5, 3), dtype=np.float32)
                )
                if result["area_m2"] >= self.MIN_SPACE_AREA_M2:
                    mini_geom.update(result)
                    mini_geom["floor_area_m2"] = result["area_m2"]
                    spaces.append(mini_geom)
            except Exception as e:
                logger.debug(f"  Mini-geometry failed for cluster {lab}: {e}")
                continue

        return spaces if len(spaces) >= 2 else None

    def _geometry_to_space(
        self, space_id: str, geometry: Dict, space_type: str
    ) -> SpaceGeometry:
        polygon = geometry.get("room_polygon") or geometry.get("footprint_polygon", [])
        return SpaceGeometry(
            space_id=space_id,
            space_type=space_type,
            polygon=polygon,
            area_m2=float(geometry.get("floor_area_m2", 0)),
            area_ci_m2=float(geometry.get("floor_area_ci_m2", 0)),
            perimeter_m=float(geometry.get("room_perimeter_m", 0)),
            polygon_source=str(geometry.get("floor_area_source", "unknown")),
            floor_y=float(geometry.get("floor_y", 0)),
            ceiling_y=float(geometry.get("ceiling_y", 0)),
            ceiling_height_m=float(geometry.get("ceiling_height_m", 0)),
            ceiling_height_ci_m=float(geometry.get("ceiling_height_ci_m", 0)),
            ceiling_reliable=bool(geometry.get("ceiling_detection_reliable", False)),
            walls=geometry.get("walls", []),
            openings=geometry.get("openings", []),
            assumptions=["depth_scale=0.001m/unit [ASSUMPTION]",
                         "RGB intrinsics scaled by depth_res/rgb_res [ASSUMPTION]"],
            debug=geometry.get("debug", {}),
        )

    def _infer_type(self, geometry: Dict) -> str:
        h = geometry.get("ceiling_height_m", 0)
        a = geometry.get("floor_area_m2", 0)
        if a < 5:
            return "small_room"
        if a > 25:
            return "large_room_or_open_plan"
        return "room"
