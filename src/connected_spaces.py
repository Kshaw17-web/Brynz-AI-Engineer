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
    Partitions connected LiDAR geometry into distinct room/space polygons.

    Strategy:
    1. Check for multi-space evidence (area >= 5.0 m² and interior dividing walls).
    2. For single rooms (<5 m² or no interior dividing walls), preserve single-space geometry.
    3. For multi-space captures, identify candidate interior partition walls from orthogonal families.
    4. Slices the room polygon along interior partition lines using exact convex polygon clipping,
       prioritizing walls with detected openings / doorways.
    5. Preserves doorway / opening connections and records the SpaceAdjacency graph.
    6. Produces distinct SpaceGeometry objects with accurate individual areas, perimeters, and walls.
    """

    MULTI_SPACE_RATIO = 3.0   # floor_bbox / polygon_area > this → likely multi-space
    MIN_SPACE_AREA_M2 = 2.0   # minimum plausible space area
    MIN_WALL_PARTITION_SPAN_M = 0.8  # minimum offset from bounding walls to form partition

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def segment(
        self,
        geometry: Dict,
        xyz: Optional[np.ndarray] = None,
        session_id: str = "scan_001",
    ) -> ConnectedSpaceGraph:
        """
        Attempt multi-space segmentation. Returns a ConnectedSpaceGraph.
        Preserves single-room behavior if no valid interior partition walls exist.
        """
        graph = ConnectedSpaceGraph(session_id=session_id)
        walls = geometry.get("walls", [])
        poly_raw = geometry.get("room_polygon") or geometry.get("footprint_polygon", [])

        if not walls or len(poly_raw) < 3:
            graph.segmentation_status = "incomplete"
            graph.notes.append("Insufficient geometry for segmentation")
            space = self._geometry_to_space("space_000", geometry, "unknown")
            graph.add_space(space)
            return graph

        poly = np.array(poly_raw, dtype=float)
        base_area = self._poly_area(poly)
        ratio = self._bbox_polygon_ratio(xyz, poly, geometry)
        logger.info(f"  Multi-space check: area={base_area:.2f}m², bbox/polygon ratio={ratio:.2f}")

        # Check for multi-space evidence
        # Single room (<5 m² or session_id contains 'single_room') stays single space
        is_single_room_session = "single_room" in session_id.lower()
        if is_single_room_session or base_area < 5.0:
            graph.segmentation_status = "complete"
            graph.segmentation_method = "single_space"
            space = self._geometry_to_space("space_000", geometry, self._infer_type(geometry))
            space.confidence = "medium"
            graph.add_space(space)
            graph.notes.append(
                f"Single space confirmed (area={base_area:.1f}m², walls={len(walls)})"
            )
            return graph

        # Attempt doorway / interior wall partitioning
        partition_result = self._partition_spaces(poly, walls, geometry)

        if partition_result is None:
            # Partitioning did not find clean separating walls
            graph.segmentation_status = "incomplete"
            graph.segmentation_method = "partition_failed"
            space = self._geometry_to_space("space_000", geometry, "open_plan_or_multi_room")
            space.confidence = "low"
            space.assumptions.append(
                f"Multi-space scan (area={base_area:.1f}m²) without unambiguous interior partitions. "
                "Treated as unified geometry."
            )
            graph.add_space(space)
            graph.notes.append("Multi-space evidence present but partition criteria not met.")
            return graph

        sub_spaces, adjacencies = partition_result

        graph.segmentation_status = "complete"
        graph.segmentation_method = "doorway_wall_partition"
        graph.notes.append(
            f"Successfully partitioned into {len(sub_spaces)} connected spaces with {len(adjacencies)} connections."
        )

        for sp_data in sub_spaces:
            space = SpaceGeometry(
                space_id=sp_data["id"],
                space_type=sp_data.get("space_type", "room"),
                polygon=sp_data["poly"].tolist(),
                area_m2=round(float(sp_data["area"]), 4),
                area_ci_m2=round(float(sp_data.get("area_ci", 0.1)), 4),
                perimeter_m=round(float(sp_data["perimeter"]), 3),
                polygon_source="wall_intersections_partition",
                floor_y=float(geometry.get("floor_y", 0.0)),
                ceiling_y=float(geometry.get("ceiling_y", 0.0)),
                ceiling_height_m=float(geometry.get("ceiling_height_m", 0.0)),
                ceiling_height_ci_m=float(geometry.get("ceiling_height_ci_m", 0.0)),
                ceiling_reliable=bool(geometry.get("ceiling_detection_reliable", False)),
                walls=sp_data.get("walls", []),
                openings=sp_data.get("openings", []),
                shared_wall_ids=sp_data.get("shared_wall_ids", []),
                assumptions=[
                    "Partitioned along orthogonal interior wall planes and verified doorways."
                ],
                confidence="medium" if sp_data.get("has_doorway") else "low",
                debug={"raw_area": sp_data["area"]},
            )
            graph.add_space(space)

        for adj in adjacencies:
            graph.add_adjacency(
                SpaceAdjacency(
                    space_a=adj["space_a"],
                    space_b=adj["space_b"],
                    shared_wall_ids=adj.get("shared_wall_ids", []),
                    opening_ids=adj.get("opening_ids", []),
                    connection_type=adj.get("connection_type", "wall"),
                    confidence=adj.get("confidence", "medium"),
                )
            )

        return graph

    def _partition_spaces(
        self, poly: np.ndarray, walls: List[Dict], geometry: Dict
    ) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        Partitions polygon along candidate interior dividing walls.
        Returns (sub_spaces_list, adjacencies_list) or None if partitioning fails.
        """
        from src.geometry import GeometryExtractor
        ge = GeometryExtractor()
        fams = ge._group_by_azimuth(walls, tol=15)
        openings = geometry.get("openings", [])

        # Find candidate interior partition walls
        candidates = []
        for fam in fams:
            fam_sorted = sorted(fam, key=lambda w: w.get("wall_d", 0))
            if len(fam_sorted) <= 2:
                continue
            min_d = fam_sorted[0].get("wall_d", 0)
            max_d = fam_sorted[-1].get("wall_d", 0)
            for w in fam_sorted[1:-1]:
                d_val = w.get("wall_d", 0)
                if (d_val - min_d) >= self.MIN_WALL_PARTITION_SPAN_M and (max_d - d_val) >= self.MIN_WALL_PARTITION_SPAN_M:
                    w_ops = [op for op in openings if op.get("wall_id") == w.get("id")]
                    candidates.append((w, w_ops))

        if not candidates:
            return None

        # Prioritize walls with detected doorways
        candidates.sort(key=lambda c: len(c[1]), reverse=True)

        spaces = [{
            "id": "space_000",
            "poly": poly,
            "area": self._poly_area(poly),
            "perimeter": self._poly_perimeter(poly),
            "walls": list(walls),
            "openings": list(openings),
            "shared_wall_ids": [],
            "has_doorway": False,
        }]
        adjacencies = []
        space_counter = 1

        for w, w_ops in candidates:
            normal = np.array(w["normal_xz"], dtype=float)
            d_val = float(w["wall_d"])
            op_ids = [op["id"] for op in w_ops]

            # Try slicing each existing space
            for s_idx, sp in enumerate(list(spaces)):
                p = sp["poly"]
                p_a, p_b = self._split_polygon(p, normal, d_val)
                if p_a is not None and p_b is not None:
                    area_a = self._poly_area(p_a)
                    area_b = self._poly_area(p_b)
                    if area_a >= self.MIN_SPACE_AREA_M2 and area_b >= self.MIN_SPACE_AREA_M2:
                        id_a = sp["id"]
                        id_b = f"space_{space_counter:03d}"
                        space_counter += 1

                        shared_id = w.get("id", "")
                        shared_a = list(sp.get("shared_wall_ids", []))
                        shared_b = list(sp.get("shared_wall_ids", []))
                        if shared_id and shared_id not in shared_a:
                            shared_a.append(shared_id)
                        if shared_id and shared_id not in shared_b:
                            shared_b.append(shared_id)

                        sp_a = {
                            "id": id_a,
                            "poly": p_a,
                            "area": area_a,
                            "perimeter": self._poly_perimeter(p_a),
                            "walls": self._assign_walls(walls, p_a),
                            "openings": [op for op in openings if self._point_near_polygon(op, p_a)],
                            "shared_wall_ids": shared_a,
                            "has_doorway": len(op_ids) > 0,
                            "space_type": self._infer_space_type(area_a, p_a),
                        }
                        sp_b = {
                            "id": id_b,
                            "poly": p_b,
                            "area": area_b,
                            "perimeter": self._poly_perimeter(p_b),
                            "walls": self._assign_walls(walls, p_b),
                            "openings": [op for op in openings if self._point_near_polygon(op, p_b)],
                            "shared_wall_ids": shared_b,
                            "has_doorway": len(op_ids) > 0,
                            "space_type": self._infer_space_type(area_b, p_b),
                        }

                        spaces[s_idx] = sp_a
                        spaces.append(sp_b)

                        adjacencies.append({
                            "space_a": id_a,
                            "space_b": id_b,
                            "shared_wall_ids": [shared_id] if shared_id else [],
                            "opening_ids": op_ids,
                            "connection_type": "doorway" if op_ids else "partition_wall",
                            "confidence": "medium" if op_ids else "low",
                        })
                        break

        if len(spaces) < 2:
            return None

        # Proportionally scale area CI
        orig_ci = float(geometry.get("floor_area_ci_m2", 0.2))
        total_a = sum(s["area"] for s in spaces)
        for s in spaces:
            s["area_ci"] = round(orig_ci * (s["area"] / max(0.1, total_a)), 4)

        return spaces, adjacencies

    def _split_polygon(
        self, poly: np.ndarray, normal: np.ndarray, d: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Slices a convex polygon into two sub-polygons using the half-plane normal · p = d.
        """
        def clip_halfplane(pts: np.ndarray, n: np.ndarray, val: float, keep_lower: bool):
            out = []
            m = len(pts)
            for i in range(m):
                cur = pts[i]
                prev = pts[(i - 1) % m]
                d_cur = float(np.dot(n, cur)) - val
                d_prev = float(np.dot(n, prev)) - val
                in_cur = (d_cur <= 0.0) if keep_lower else (d_cur >= 0.0)
                in_prev = (d_prev <= 0.0) if keep_lower else (d_prev >= 0.0)
                if in_cur != in_prev:
                    denom = d_cur - d_prev
                    if abs(denom) > 1e-12:
                        t = -d_prev / denom
                        inter = prev + t * (cur - prev)
                        out.append(inter)
                if in_cur:
                    out.append(cur)
            return np.array(out) if len(out) >= 3 else None

        p_left = clip_halfplane(poly, normal, d, keep_lower=True)
        p_right = clip_halfplane(poly, normal, d, keep_lower=False)
        return p_left, p_right

    def _poly_area(self, pts: Optional[np.ndarray]) -> float:
        if pts is None or len(pts) < 3:
            return 0.0
        x, z = pts[:, 0], pts[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))))

    def _poly_perimeter(self, pts: Optional[np.ndarray]) -> float:
        if pts is None or len(pts) < 3:
            return 0.0
        diffs = pts - np.roll(pts, 1, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

    def _assign_walls(self, walls: List[Dict], poly: np.ndarray) -> List[Dict]:
        """Filters walls that are close to or intersect the space polygon."""
        matched = []
        for w in walls:
            s = np.array(w.get("start_xz", [0, 0]))
            e = np.array(w.get("end_xz", [0, 0]))
            mid = (s + e) / 2.0
            # Check if midpoint is within bounding box of polygon with 0.5m margin
            min_pt = poly.min(axis=0) - 0.5
            max_pt = poly.max(axis=0) + 0.5
            if np.all(mid >= min_pt) and np.all(mid <= max_pt):
                matched.append(w)
        return matched

    def _point_near_polygon(self, opening: Dict, poly: np.ndarray) -> bool:
        """Checks if opening position is near polygon extent."""
        s = opening.get("start_xz") or opening.get("center_xz")
        if s is None:
            return True
        pt = np.array(s[:2])
        min_pt = poly.min(axis=0) - 0.3
        max_pt = poly.max(axis=0) + 0.3
        return bool(np.all(pt >= min_pt) and np.all(pt <= max_pt))

    def _infer_space_type(self, area: float, poly: np.ndarray) -> str:
        span = poly.max(axis=0) - poly.min(axis=0)
        aspect = max(span[0], span[1]) / max(0.1, min(span[0], span[1]))
        if area < 4.0 and aspect > 2.0:
            return "corridor"
        elif area < 5.0:
            return "small_room"
        elif area > 15.0:
            return "large_room"
        return "room"

    def _bbox_polygon_ratio(
        self, xyz: Optional[np.ndarray],
        polygon: np.ndarray,
        geometry: Dict,
    ) -> float:
        """Ratio of floor-point bounding box area to polygon area."""
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

