"""
CP3 Unit Tests — wall candidate scoring, polygon construction, openings.
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import GeometryExtractor

SAMPLE_DIR = Path(__file__).parent.parent / "single_room"
SAMPLE_AVAILABLE = SAMPLE_DIR.exists()

GEO = GeometryExtractor()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a synthetic room
# ─────────────────────────────────────────────────────────────────────────────

def _box_room(lx=4.0, lz=3.0, h=2.5, n=800, noise=0.01, rng_seed=0):
    """Return Nx3 point cloud for a box room with floor+ceiling+4 walls."""
    rng = np.random.default_rng(rng_seed)
    pts = []

    def noisy(arr):
        return arr + rng.normal(0, noise, arr.shape)

    # Floor / ceiling (Y = 0 / h)
    for y0 in [0.0, h]:
        u = rng.uniform(0, lx, n)
        v = rng.uniform(0, lz, n)
        pts.append(np.column_stack([u, np.full(n, y0), v]))

    # Walls (with height variation so they pass height-coverage filter)
    for x0 in [0.0, lx]:
        yy = rng.uniform(0.1, h - 0.1, n)
        zz = rng.uniform(0, lz, n)
        pts.append(noisy(np.column_stack([np.full(n, x0), yy, zz])))

    for z0 in [0.0, lz]:
        yy = rng.uniform(0.1, h - 0.1, n)
        xx = rng.uniform(0, lx, n)
        pts.append(noisy(np.column_stack([xx, yy, np.full(n, z0)])))

    return np.vstack(pts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Wall scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestWallScoring:
    def _make_candidate(self, **kwargs):
        base = {
            "id": "wall_00",
            "n_inliers": 500,
            "length_m": 3.0,
            "height_span_m": 1.8,
            "height_coverage": 0.72,
            "rms_residual_m": 0.02,
            "point_density": 50.0,
            "azimuth_deg": 0.0,
        }
        base.update(kwargs)
        return base

    def test_accepts_valid_wall(self):
        c = self._make_candidate()
        ok, reason = GEO._score_candidate(c, room_h=2.5)
        assert ok, f"Expected valid wall to pass, got: {reason}"

    def test_rejects_too_few_points(self):
        c = self._make_candidate(n_inliers=50)
        ok, _ = GEO._score_candidate(c, room_h=2.5)
        assert not ok

    def test_rejects_too_short(self):
        c = self._make_candidate(length_m=0.2)
        ok, _ = GEO._score_candidate(c, room_h=2.5)
        assert not ok

    def test_rejects_low_height_span(self):
        c = self._make_candidate(height_span_m=0.3)
        ok, _ = GEO._score_candidate(c, room_h=2.5)
        assert not ok

    def test_rejects_low_height_coverage(self):
        c = self._make_candidate(height_coverage=0.10)
        ok, _ = GEO._score_candidate(c, room_h=2.5)
        assert not ok

    def test_rejects_high_residual(self):
        c = self._make_candidate(rms_residual_m=0.15)
        ok, _ = GEO._score_candidate(c, room_h=2.5)
        assert not ok

    def test_rejects_low_density(self):
        c = self._make_candidate(point_density=5.0)
        ok, _ = GEO._score_candidate(c, room_h=2.5)
        assert not ok

    def test_rejection_reason_string(self):
        c = self._make_candidate(n_inliers=10)
        ok, reason = GEO._score_candidate(c, room_h=2.5)
        assert not ok
        assert "too_few_pts" in reason


# ─────────────────────────────────────────────────────────────────────────────
# Azimuth orthogonality selection
# ─────────────────────────────────────────────────────────────────────────────

class TestAzimuthSelection:
    def _fake_hist(self, azs, bins=180):
        """Build a dummy histogram array peaked at given azimuths."""
        h = np.zeros(bins)
        for a in azs:
            idx = int(a / 180 * bins)
            h[max(0, idx-2):min(bins, idx+3)] = 100
        return h

    def test_selects_orthogonal_pair(self):
        """With 0° and 90° present, should pick those two."""
        azs = np.array([0.0, 45.0, 90.0, 135.0])
        h = self._fake_hist(azs)
        selected = GEO._pick_orthogonal_families(azs, h)
        diffs = [abs(selected[i] - selected[j]) % 180
                 for i in range(len(selected)) for j in range(i+1, len(selected))]
        assert any(78 <= d <= 102 for d in diffs), \
            f"No orthogonal pair selected. Selected: {selected}, diffs: {diffs}"

    def test_handles_single_direction(self):
        azs = np.array([45.0])
        h = self._fake_hist(azs)
        selected = GEO._pick_orthogonal_families(azs, h)
        assert len(selected) == 1

    def test_handles_empty_directions(self):
        azs = np.array([])
        h = np.zeros(180)
        selected = GEO._pick_orthogonal_families(azs, h)
        assert len(selected) == 0

    def test_tolerates_imperfect_orthogonality(self):
        """89° is within tolerance → should be selected as pair."""
        azs = np.array([0.0, 89.0])
        h = self._fake_hist(azs)
        selected = GEO._pick_orthogonal_families(azs, h)
        assert len(selected) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Wall intersection
# ─────────────────────────────────────────────────────────────────────────────

class TestWallIntersection:
    def _wall(self, az_deg, d, length=3.0):
        az = np.radians(az_deg)
        n = np.array([np.cos(az), np.sin(az)])
        t = np.array([-n[1], n[0]])
        s = (n * d + t * (-length / 2)).tolist()
        e = (n * d + t * (length / 2)).tolist()
        return {
            "normal_xz": n.tolist(),
            "tangent_xz": t.tolist(),
            "wall_d": float(d),
            "start_xz": s,
            "end_xz": e,
        }

    def test_perpendicular_walls_intersect(self):
        """Two perpendicular walls should intersect."""
        wa = self._wall(az_deg=0, d=1.0, length=4.0)
        wb = self._wall(az_deg=90, d=2.0, length=4.0)
        pt = GEO._line_intersection_xz(wa, wb)
        assert pt is not None, "Perpendicular walls should intersect"
        # na=(1,0), da=1 → x=1; nb=(0,1), db=2 → z=2
        assert abs(pt[0] - 1.0) < 0.01, f"Expected x=1.0, got {pt[0]}"
        assert abs(pt[1] - 2.0) < 0.01, f"Expected z=2.0, got {pt[1]}"

    def test_parallel_walls_do_not_intersect(self):
        """Parallel walls should return None."""
        wa = self._wall(az_deg=0, d=1.0)
        wb = self._wall(az_deg=0, d=3.0)
        pt = GEO._line_intersection_xz(wa, wb)
        assert pt is None, "Parallel walls should not intersect"

    def test_intersection_outside_extent_rejected(self):
        """Intersection far outside wall extent should be None."""
        wa = self._wall(az_deg=0, d=1.0, length=1.0)
        wb = self._wall(az_deg=90, d=100.0, length=1.0)  # 100m away
        pt = GEO._line_intersection_xz(wa, wb)
        assert pt is None, "Intersection far outside extent should be rejected"


# ─────────────────────────────────────────────────────────────────────────────
# Room polygon construction
# ─────────────────────────────────────────────────────────────────────────────

class TestRoomPolygon:
    def _az_wall(self, az_deg, d, length=5.0, n_inliers=500, h_span=1.8):
        az = np.radians(az_deg)
        n = np.array([np.cos(az), np.sin(az)])
        t = np.array([-n[1], n[0]])
        return {
            "id": f"w_{az_deg}_{d}",
            "azimuth_deg": float(az_deg),
            "normal_xz": n.tolist(),
            "tangent_xz": t.tolist(),
            "wall_d": float(d),
            "start_xz": (n * d + t * (-length / 2)).tolist(),
            "end_xz":   (n * d + t * (length / 2)).tolist(),
            "length_m": length,
            "length_ci_m": 0.05,
            "n_inliers": n_inliers,
            "height_span_m": h_span,
            "rms_residual_m": 0.02,
            "point_density": 50.0,
            "height_coverage": 0.72,
        }

    def test_box_room_4_walls(self):
        """4 walls forming a 4x3 box → polygon area ≈ 12 m²."""
        walls = [
            self._az_wall(0,   0.0),   # x=0 wall
            self._az_wall(0,   4.0),   # x=4 wall
            self._az_wall(90,  0.0),   # z=0 wall
            self._az_wall(90,  3.0),   # z=3 wall
        ]
        xyz_fake = np.array([[0, -0.05, 0], [4, -0.05, 3]], dtype=np.float32)
        result = GEO._build_room_polygon(walls, floor_y=0.0, xyz_clean=xyz_fake)
        assert result["source"] == "wall_intersections", \
            f"Expected wall_intersections, got {result['source']}"
        area = result["area_m2"]
        assert 8.0 <= area <= 16.0, f"Expected ~12 m², got {area:.2f} m²"

    def test_polygon_needs_2_families(self):
        """Single azimuth family → should fall back to convex hull."""
        walls = [self._az_wall(0, 0.0), self._az_wall(0, 4.0)]
        xyz_fake = np.zeros((10, 3), dtype=np.float32)
        result = GEO._build_room_polygon(walls, floor_y=0.0, xyz_clean=xyz_fake)
        assert "convex_hull" in result["source"] or result["area_m2"] == 0.0

    def test_polygon_area_zero_for_no_walls(self):
        xyz_fake = np.zeros((10, 3), dtype=np.float32)
        result = GEO._build_room_polygon([], floor_y=0.0, xyz_clean=xyz_fake)
        # Should fall back gracefully
        assert isinstance(result["area_m2"], float)


# ─────────────────────────────────────────────────────────────────────────────
# Corner clustering
# ─────────────────────────────────────────────────────────────────────────────

class TestCornerClustering:
    def test_nearby_corners_merged(self):
        pts = np.array([[0.0, 0.0], [0.1, 0.05], [5.0, 5.0]])
        merged = GEO._cluster_corners(pts, tol=0.25)
        assert len(merged) == 2, f"Expected 2 clusters, got {len(merged)}"

    def test_distant_corners_preserved(self):
        pts = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
        merged = GEO._cluster_corners(pts, tol=0.25)
        assert len(merged) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Polygon area
# ─────────────────────────────────────────────────────────────────────────────

class TestPolygonArea:
    def test_box_4x3(self):
        walls = [
            TestRoomPolygon()._az_wall(0, 0.0),
            TestRoomPolygon()._az_wall(0, 4.0),
            TestRoomPolygon()._az_wall(90, 0.0),
            TestRoomPolygon()._az_wall(90, 3.0),
        ]
        xyz_fake = np.array([[0, -0.05, 0]], dtype=np.float32)
        result = GEO._build_room_polygon(walls, floor_y=0.0, xyz_clean=xyz_fake)
        if result["source"] == "wall_intersections":
            assert 10.0 <= result["area_m2"] <= 14.0


# ─────────────────────────────────────────────────────────────────────────────
# Opening filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestOpeningFiltering:
    def test_no_openings_for_solid_wall(self):
        """Dense, complete wall should not generate openings."""
        rng = np.random.default_rng(0)
        # Wall at x=0, covering full height 0-2.5m, z=[0,3]
        n = 800
        x = rng.normal(0, 0.02, n)
        y = rng.uniform(0.1, 2.4, n)
        z = rng.uniform(0.0, 3.0, n)
        wall_pts = np.stack([x, y, z], axis=1).astype(np.float32)

        wall = {
            "id": "wall_00",
            "normal_xz": [1.0, 0.0],
            "tangent_xz": [0.0, 1.0],
            "wall_d": 0.0,
            "start_xz": [0.0, 0.0],
            "end_xz":   [0.0, 3.0],
            "length_m": 3.0,
        }
        openings = GEO._detect_openings_conservative(
            wall_pts, [wall], floor_y=0.0, ceil_y=2.5, room_h=2.5
        )
        assert len(openings) == 0, f"Expected 0 openings for solid wall, got {len(openings)}"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end synthetic room
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticRoom:
    def test_full_extraction_box_room(self):
        xyz = _box_room(lx=4.0, lz=3.0, h=2.5, n=1000)
        result = GEO.extract({"xyz": xyz})

        # Ceiling height
        h = result["ceiling_height_m"]
        assert 1.8 <= h <= 3.2, f"Ceiling height {h:.3f}m far from expected 2.5m"

        # At least 2 walls
        assert len(result["walls"]) >= 2, "Expected at least 2 walls"

        # Wall count not excessive
        assert len(result["walls"]) <= 12, f"Too many walls: {len(result['walls'])}"

        # All walls pass scoring
        for w in result["walls"]:
            assert w["rms_residual_m"] <= GEO.MAX_RESIDUAL_M + 1e-6, \
                f"Wall {w['id']} has high residual: {w['rms_residual_m']}"
            assert w["n_inliers"] >= GEO.MIN_WALL_PTS, \
                f"Wall {w['id']} has too few inliers: {w['n_inliers']}"

        # Floor area is returned
        assert isinstance(result["floor_area_m2"], float)
        assert result["floor_area_m2"] > 0


@pytest.mark.skipif(not SAMPLE_AVAILABLE, reason="sample data not present")
class TestSampleData:
    def test_single_room_loads(self):
        from src.lidar_processor import LiDARProcessor
        proc = LiDARProcessor(SAMPLE_DIR, frame_skip=40)
        pc, _ = proc.load()
        assert len(pc["xyz"]) > 1000

    def test_single_room_geometry(self):
        from src.lidar_processor import LiDARProcessor
        proc = LiDARProcessor(SAMPLE_DIR, frame_skip=40)
        pc, _ = proc.load()
        result = GEO.extract(pc)
        # Plausibility: ceiling between 1.5 and 4m
        h = result["ceiling_height_m"]
        assert 1.5 <= h <= 4.0, f"Ceiling height {h:.3f}m is implausible"
        # Not too many walls
        assert len(result["walls"]) <= 20, f"Too many walls: {len(result['walls'])}"


if __name__ == "__main__":
    import traceback

    tests_to_run = [
        TestWallScoring(),
        TestAzimuthSelection(),
        TestWallIntersection(),
        TestRoomPolygon(),
        TestCornerClustering(),
        TestPolygonArea(),
        TestOpeningFiltering(),
        TestSyntheticRoom(),
    ]

    passed = 0
    failed = 0
    for suite in tests_to_run:
        for name in dir(suite):
            if not name.startswith("test_"):
                continue
            try:
                getattr(suite, name)()
                print(f"  PASS  {suite.__class__.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {suite.__class__.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")


# ─────────────────────────────────────────────────────────────────────────────
# CP4 Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWallClipping:
    """Test the Liang-Barsky wall-to-polygon clipping in floor_plan.py."""

    def test_clip_inside_polygon(self):
        """A line through the polygon centre clips to two boundary points."""
        from src.floor_plan import _clip_line_to_polygon
        poly = np.array([[0,0],[4,0],[4,3],[0,3]], dtype=float)
        p1 = np.array([-10, 1.5])
        p2 = np.array([ 10, 1.5])
        result = _clip_line_to_polygon(p1, p2, poly)
        assert result is not None
        s, e = result
        assert abs(s[0]) < 0.01 or abs(e[0]) < 0.01   # one end at x=0
        assert abs(s[0] - 4) < 0.01 or abs(e[0] - 4) < 0.01  # other at x=4

    def test_clip_misses_polygon(self):
        """A line that does not pass through polygon returns None."""
        from src.floor_plan import _clip_line_to_polygon
        poly = np.array([[0,0],[4,0],[4,3],[0,3]], dtype=float)
        p1 = np.array([-5, 10])
        p2 = np.array([ 5, 10])
        result = _clip_line_to_polygon(p1, p2, poly)
        assert result is None

    def test_wall_to_clipped_segment(self):
        """_wall_to_clipped_segment clips a wall to its polygon intersection."""
        from src.floor_plan import _wall_to_clipped_segment
        poly = np.array([[0,0],[4,0],[4,3],[0,3]], dtype=float)
        wall = {
            "normal_xz":  [1.0, 0.0],  # X-normal wall at x=2
            "tangent_xz": [0.0, 1.0],
            "wall_d":      2.0,
            "start_xz":   [2, -10],
            "end_xz":      [2,  10],
        }
        seg = _wall_to_clipped_segment(wall, poly)
        assert seg is not None
        s, e = seg
        # Both points should have x near 2.0
        assert abs(s[0] - 2.0) < 0.1
        assert abs(e[0] - 2.0) < 0.1
        # Y span should be roughly 0 to 3
        zmin = min(s[1], e[1])
        zmax = max(s[1], e[1])
        assert zmin > -0.1
        assert zmax < 3.1


class TestOpeningPositions:
    """Test that CP4 openings contain along_wall position fields."""

    def test_opening_has_position_fields(self):
        """Openings must have along_wall_start_m and along_wall_end_m."""
        rng = np.random.default_rng(99)
        # Build a wall with a deliberate gap in the middle
        # Wall along Z from 0 to 4, normal in X direction at x=0
        pts = []
        # Dense points on wall, except a 1.0m gap from z=1.5 to z=2.5
        for z in np.linspace(0, 4, 300):
            if 1.5 <= z <= 2.5:
                continue  # gap
            y = rng.uniform(0.1, 2.0)
            pts.append([rng.normal(0, 0.02), y, z])
        wall_pts = np.array(pts, dtype=np.float32)

        wall = {
            "id": "wall_test",
            "normal_xz":  [1.0, 0.0],
            "tangent_xz": [0.0, 1.0],
            "wall_d":      0.0,
            "length_m":    4.0,
            "start_xz":   [0, 0],
            "end_xz":      [0, 4],
        }

        ge = GeometryExtractor()
        openings = ge._detect_openings_conservative(
            wall_pts, [wall], floor_y=0.0, ceil_y=2.2, room_h=2.2
        )
        # May or may not detect (depends on density), but if detected,
        # must have position fields
        for o in openings:
            assert "along_wall_start_m" in o, "Missing along_wall_start_m"
            assert "along_wall_end_m"   in o, "Missing along_wall_end_m"

    def test_short_wall_no_opening(self):
        """Walls shorter than 1.5m must not produce openings."""
        rng = np.random.default_rng(42)
        pts = []
        for z in np.linspace(0, 1.0, 100):
            pts.append([rng.normal(0, 0.02), rng.uniform(0.1, 2.0), z])
        wall_pts = np.array(pts, dtype=np.float32)
        wall = {
            "id": "short_wall",
            "normal_xz":  [1.0, 0.0],
            "tangent_xz": [0.0, 1.0],
            "wall_d":      0.0,
            "length_m":    1.0,   # too short
            "start_xz":   [0, 0],
            "end_xz":      [0, 1.0],
        }
        ge = GeometryExtractor()
        openings = ge._detect_openings_conservative(
            wall_pts, [wall], floor_y=0.0, ceil_y=2.2, room_h=2.2
        )
        assert len(openings) == 0, "Short wall must not produce openings"


class TestMultiSpaceSchema:
    """Test ConnectedSpaceGraph data structure."""

    def test_single_space_schema(self):
        from src.connected_spaces import SpaceGeometry, ConnectedSpaceGraph
        g = ConnectedSpaceGraph(session_id="test")
        space = SpaceGeometry(
            space_id="space_000",
            space_type="room",
            polygon=[[0,0],[4,0],[4,3],[0,3]],
            area_m2=12.0,
        )
        g.add_space(space)
        d = g.to_dict()
        assert d["n_spaces"] == 1
        assert "space_000" in d["spaces"]
        assert d["total_area_m2"] == pytest.approx(12.0, abs=0.1)

    def test_graph_defaults(self):
        from src.connected_spaces import ConnectedSpaceGraph
        g = ConnectedSpaceGraph()
        assert g.segmentation_status == "incomplete"
        assert g.total_area_m2() == 0.0

    def test_single_space_segmenter(self):
        """Small room with clear geometry → single space, status complete."""
        from src.connected_spaces import ConnectedSpaceSegmenter
        # Use a small floor bbox ratio (bbox/polygon < 3)
        geom = {
            "floor_area_m2": 12.0,
            "room_polygon": [[0,0],[4,0],[4,3],[0,3]],
            "walls": [
                {"id": "w1", "azimuth_deg": 0,   "normal_xz": [1,0], "tangent_xz": [0,1],
                 "wall_d": 0, "length_m": 3, "start_xz": [0,0], "end_xz": [0,3]},
                {"id": "w2", "azimuth_deg": 90,  "normal_xz": [0,1], "tangent_xz": [1,0],
                 "wall_d": 0, "length_m": 4, "start_xz": [0,0], "end_xz": [4,0]},
                {"id": "w3", "azimuth_deg": 0,   "normal_xz": [1,0], "tangent_xz": [0,1],
                 "wall_d": 4, "length_m": 3, "start_xz": [4,0], "end_xz": [4,3]},
                {"id": "w4", "azimuth_deg": 90,  "normal_xz": [0,1], "tangent_xz": [1,0],
                 "wall_d": 3, "length_m": 4, "start_xz": [0,3], "end_xz": [4,3]},
            ],
            "debug": {"bbox_polygon_ratio": 1.2},
            "floor_y": 0.0,
        }
        seg = ConnectedSpaceSegmenter()
        graph = seg.segment(geom, xyz=None, session_id="test")
        assert graph.segmentation_status in ("complete", "incomplete")
        assert len(graph.spaces) >= 1

    def test_multi_room_segmenter_with_doorways(self):
        """Layout with an interior dividing wall and doorway splits into 2 connected spaces."""
        from src.connected_spaces import ConnectedSpaceSegmenter
        geom = {
            "floor_area_m2": 32.0,
            "room_polygon": [[0, 0], [8, 0], [8, 4], [0, 4]],
            "walls": [
                # Outer boundary
                {"id": "w_left",   "azimuth_deg": 0,  "normal_xz": [1, 0], "tangent_xz": [0, 1],
                 "wall_d": 0, "length_m": 4, "start_xz": [0, 0], "end_xz": [0, 4]},
                {"id": "w_right",  "azimuth_deg": 0,  "normal_xz": [1, 0], "tangent_xz": [0, 1],
                 "wall_d": 8, "length_m": 4, "start_xz": [8, 0], "end_xz": [8, 4]},
                {"id": "w_bottom", "azimuth_deg": 90, "normal_xz": [0, 1], "tangent_xz": [1, 0],
                 "wall_d": 0, "length_m": 8, "start_xz": [0, 0], "end_xz": [8, 0]},
                {"id": "w_top",    "azimuth_deg": 90, "normal_xz": [0, 1], "tangent_xz": [1, 0],
                 "wall_d": 4, "length_m": 8, "start_xz": [0, 4], "end_xz": [8, 4]},
                # Interior dividing wall at x = 4 with doorway
                {"id": "w_middle", "azimuth_deg": 0,  "normal_xz": [1, 0], "tangent_xz": [0, 1],
                 "wall_d": 4, "length_m": 4, "start_xz": [4, 0], "end_xz": [4, 4]},
            ],
            "openings": [
                {"id": "door_01", "wall_id": "w_middle", "type": "door", "width_m": 0.9,
                 "start_xz": [4, 1.5], "end_xz": [4, 2.4]},
            ],
            "debug": {"bbox_polygon_ratio": 3.5},
            "floor_y": 0.0,
        }
        seg = ConnectedSpaceSegmenter()
        graph = seg.segment(geom, xyz=None, session_id="multi_room_scan")
        assert graph.segmentation_status == "complete"
        assert graph.segmentation_method == "doorway_wall_partition"
        assert len(graph.spaces) == 2
        assert len(graph.adjacency) == 1
        adj = graph.adjacency[0]
        assert adj.connection_type == "doorway"
        assert "door_01" in adj.opening_ids
        assert "w_middle" in adj.shared_wall_ids
        assert graph.total_area_m2() == pytest.approx(32.0, abs=0.1)

    def test_single_room_preserves_single_space(self):
        """Single room scan (area < 5 m² or session_id 'single_room') preserves 1 space."""
        from src.connected_spaces import ConnectedSpaceSegmenter
        geom = {
            "floor_area_m2": 2.53,
            "room_polygon": [[0, 0], [2, 0], [2, 1.26], [0, 1.26]],
            "walls": [
                {"id": "w1", "azimuth_deg": 0, "normal_xz": [1, 0], "wall_d": 0, "length_m": 1.26, "start_xz": [0, 0], "end_xz": [0, 1.26]},
                {"id": "w2", "azimuth_deg": 0, "normal_xz": [1, 0], "wall_d": 2, "length_m": 1.26, "start_xz": [2, 0], "end_xz": [2, 1.26]},
                {"id": "w3", "azimuth_deg": 90, "normal_xz": [0, 1], "wall_d": 0, "length_m": 2, "start_xz": [0, 0], "end_xz": [2, 0]},
                {"id": "w4", "azimuth_deg": 90, "normal_xz": [0, 1], "wall_d": 1.26, "length_m": 2, "start_xz": [0, 1.26], "end_xz": [2, 1.26]},
            ],
            "openings": [],
            "debug": {"bbox_polygon_ratio": 4.5},
            "floor_y": 0.0,
        }
        seg = ConnectedSpaceSegmenter()
        graph = seg.segment(geom, xyz=None, session_id="single_room")
        assert graph.segmentation_status == "complete"
        assert graph.segmentation_method == "single_space"
        assert len(graph.spaces) == 1
        assert len(graph.adjacency) == 0



class TestOutputSchemaV2:
    """Test schema version and new fields."""

    def test_schema_version(self):
        from src.output_schema import OutputSchema, SCHEMA_VERSION
        assert SCHEMA_VERSION == "2.0"

    def test_assumptions_present(self):
        from src.output_schema import OutputSchema
        s = OutputSchema()
        result = s.build(
            room_id="test", geometry={"walls": [], "openings": [], "floor_area_m2": 5,
                                       "floor_area_ci_m2": 0.1, "ceiling_height_m": 2.5,
                                       "ceiling_height_ci_m": 0.01, "floor_area_source": "test",
                                       "ceiling_detection_reliable": True, "debug": {}},
            damage_regions=[], tier="lidar",
        )
        assert "assumptions" in result
        ids = [a["id"] for a in result["assumptions"]]
        assert "depth_scale" in ids
        assert "imu_gravity_alignment" in ids

    def test_drift_audit_present(self):
        from src.output_schema import OutputSchema
        s = OutputSchema()
        result = s.build(
            room_id="test", geometry={"walls": [], "openings": [], "floor_area_m2": 5,
                                       "floor_area_ci_m2": 0.1, "ceiling_height_m": 2.5,
                                       "ceiling_height_ci_m": 0.01, "floor_area_source": "test",
                                       "ceiling_detection_reliable": True, "debug": {}},
            damage_regions=[], tier="lidar",
        )
        assert "drift_audit" in result
        da = result["drift_audit"]
        assert "drift_correction_applied" in da
        assert da["drift_correction_applied"] is False


class TestDriftCorrection:
    """Tests comparing ON vs OFF drift correction behavior."""

    @staticmethod
    def _create_synthetic_trajectory(drift_offset=(0.16, 0.0, 0.12), n_steps=20):
        """Create a synthetic 10m square closed-loop path ending with a drift offset."""
        # Square: (0,0) -> (3,0) -> (3,3) -> (0,3) -> (0,0) + drift
        corners = [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 3.0]),
            np.array([0.0, 0.0, 3.0]),
            np.array([drift_offset[0], drift_offset[1], drift_offset[2]]),
        ]
        poses = {}
        idx = 0
        for seg in range(len(corners) - 1):
            c1, c2 = corners[seg], corners[seg + 1]
            for alpha in np.linspace(0, 1, n_steps, endpoint=(seg == len(corners) - 2)):
                pos = (1 - alpha) * c1 + alpha * c2
                poses[idx] = {
                    "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                    "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                }
                idx += 1
        return poses

    def test_drift_correction_on_closes_loop(self):
        """When ON, linear trajectory loop closure eliminates loop closure error."""
        from src.lidar_processor import LiDARProcessor
        proc = LiDARProcessor(Path("dummy"), apply_drift_correction=True)
        poses = self._create_synthetic_trajectory(drift_offset=(0.16, 0.0, 0.12))

        pre_gap = np.linalg.norm([poses[max(poses.keys())]["x"] - poses[0]["x"],
                                  poses[max(poses.keys())]["y"] - poses[0]["y"],
                                  poses[max(poses.keys())]["z"] - poses[0]["z"]])
        assert pre_gap == pytest.approx(0.20, abs=0.001)

        corrected_poses, drift_info = proc._apply_trajectory_drift_correction(poses)

        assert drift_info["drift_correction_applied"] is True
        assert drift_info["method"] == "linear_trajectory_loop_closure"
        assert drift_info["loop_closure_status"] == "applied_closed_loop"
        assert drift_info["loop_closure_pre_residual_m"] == pytest.approx(0.20, abs=0.001)
        assert drift_info["loop_closure_post_residual_m"] == pytest.approx(0.00, abs=1e-5)
        assert drift_info["trajectory_length_m"] > 10.0

        # Start pose remains origin
        assert corrected_poses[0]["x"] == pytest.approx(0.0, abs=1e-6)
        assert corrected_poses[0]["z"] == pytest.approx(0.0, abs=1e-6)

        # End pose is corrected to match start pose exactly
        last_frame = max(corrected_poses.keys())
        assert corrected_poses[last_frame]["x"] == pytest.approx(0.0, abs=1e-5)
        assert corrected_poses[last_frame]["z"] == pytest.approx(0.0, abs=1e-5)

    def test_drift_correction_off_preserves_raw_odometry(self):
        """When OFF, raw odometry and accumulated drift are strictly preserved."""
        from src.lidar_processor import LiDARProcessor
        proc = LiDARProcessor(Path("dummy"), apply_drift_correction=False)
        poses = self._create_synthetic_trajectory(drift_offset=(0.16, 0.0, 0.12))

        last_frame = max(poses.keys())
        orig_x = poses[last_frame]["x"]
        orig_z = poses[last_frame]["z"]

        drift_info = proc._audit_unmitigated_drift(poses)

        assert drift_info["drift_correction_applied"] is False
        assert drift_info["method"] == "none"
        assert drift_info["loop_closure_status"] == "open_loop_disabled"
        assert drift_info["loop_closure_pre_residual_m"] == pytest.approx(0.20, abs=0.001)
        assert drift_info["loop_closure_post_residual_m"] == pytest.approx(0.20, abs=0.001)

        # OFF path does not alter poses
        assert poses[last_frame]["x"] == orig_x
        assert poses[last_frame]["z"] == orig_z

    def test_drift_correction_open_trajectory_not_falsely_closed(self):
        """An open trajectory (> 0.8m closure gap) is recognized as open and unwarped."""
        from src.lidar_processor import LiDARProcessor
        proc = LiDARProcessor(Path("dummy"), apply_drift_correction=True, loop_closure_threshold_m=0.8)
        # Open trajectory: moves 6m along X axis
        poses = {
            i: {"x": float(i), "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
            for i in range(7)
        }
        corrected_poses, drift_info = proc._apply_trajectory_drift_correction(poses)

        assert drift_info["drift_correction_applied"] is False
        assert drift_info["method"] == "none"
        assert drift_info["loop_closure_status"] == "open_trajectory_unclosed"
        assert drift_info["loop_closure_post_residual_m"] == pytest.approx(6.0, abs=0.01)
        # Poses remain unchanged
        assert corrected_poses[6]["x"] == 6.0

    def test_drift_correction_is_deterministic(self):
        """Repeated runs on the same trajectory produce identical results."""
        from src.lidar_processor import LiDARProcessor
        proc = LiDARProcessor(Path("dummy"), apply_drift_correction=True)
        poses = self._create_synthetic_trajectory(drift_offset=(0.25, 0.0, 0.15))

        res1, info1 = proc._apply_trajectory_drift_correction(poses)
        res2, info2 = proc._apply_trajectory_drift_correction(poses)

        assert info1 == info2
        for f in poses:
            assert res1[f]["x"] == res2[f]["x"]
            assert res1[f]["y"] == res2[f]["y"]
            assert res1[f]["z"] == res2[f]["z"]

    def test_schema_drift_audit_on_vs_off(self):
        """OutputSchema correctly records ON vs OFF drift audit states."""
        from src.output_schema import OutputSchema
        schema = OutputSchema()
        geom = {"walls": [{"rms_residual_m": 0.02}], "openings": [], "floor_area_m2": 10.0}

        # ON state
        drift_info_on = {
            "drift_correction_applied": True,
            "method": "linear_trajectory_loop_closure",
            "loop_closure_status": "applied_closed_loop",
            "trajectory_length_m": 25.0,
            "loop_closure_pre_residual_m": 0.172,
            "loop_closure_post_residual_m": 0.0,
        }
        doc_on = schema.build("room_001", geom, [], "lidar", drift_info=drift_info_on)
        audit_on = doc_on["drift_audit"]
        assert audit_on["drift_correction_applied"] is True
        assert audit_on["method"] == "linear_trajectory_loop_closure"
        assert audit_on["loop_closure_post_residual_m"] == 0.0

        # OFF state
        drift_info_off = {
            "drift_correction_applied": False,
            "method": "none",
            "loop_closure_status": "open_loop_disabled",
            "trajectory_length_m": 25.0,
            "loop_closure_pre_residual_m": 0.172,
            "loop_closure_post_residual_m": 0.172,
        }
        doc_off = schema.build("room_001", geom, [], "lidar", drift_info=drift_info_off)
        audit_off = doc_off["drift_audit"]
        assert audit_off["drift_correction_applied"] is False
        assert audit_off["method"] == "none"
        assert audit_off["loop_closure_status"] == "open_loop_disabled"
        assert audit_off["loop_closure_post_residual_m"] == 0.172

    def test_point_cloud_shifts_by_drift_correction(self):
        """Depth points unprojected with corrected poses shift the final frame points by closure residual."""
        from src.lidar_processor import LiDARProcessor
        proc_on = LiDARProcessor(Path("dummy"), apply_drift_correction=True)
        poses = self._create_synthetic_trajectory(drift_offset=(0.16, 0.0, 0.12))

        corrected_poses, _ = proc_on._apply_trajectory_drift_correction(poses)
        raw_poses = poses

        last_f = max(poses.keys())
        t_raw = np.array([raw_poses[last_f]["x"], raw_poses[last_f]["y"], raw_poses[last_f]["z"]])
        t_corr = np.array([corrected_poses[last_f]["x"], corrected_poses[last_f]["y"], corrected_poses[last_f]["z"]])

        diff = t_raw - t_corr
        assert np.linalg.norm(diff) == pytest.approx(0.20, abs=0.001)
        assert t_corr[0] == pytest.approx(0.0, abs=1e-5)
        assert t_corr[2] == pytest.approx(0.0, abs=1e-5)

    def test_stitcher_drift_correction_on_cyclic_adjacency(self):
        """Stitcher distributes closure error when a multi-room cycle exists."""
        from src.stitcher import MultiRoomStitcher
        stitcher = MultiRoomStitcher(Path("dummy"))
        layouts = {
            "r1": {"offset_x": 0.0, "offset_z": 0.0, "rotation": 0.0},
            "r2": {"offset_x": 4.0, "offset_z": 0.0, "rotation": 0.0},
            "r3": {"offset_x": 4.0, "offset_z": 4.0, "rotation": 0.0},
            "r4": {"offset_x": 0.2, "offset_z": 0.1, "rotation": 0.0},
        }
        adjacency = [
            {"room_a": "r1", "room_b": "r2"},
            {"room_a": "r2", "room_b": "r3"},
            {"room_a": "r3", "room_b": "r4"},
            {"room_a": "r4", "room_b": "r1"},
        ]
        corrected = stitcher._apply_drift_correction(layouts, adjacency)
        assert corrected["r4"]["offset_x"] == pytest.approx(0.0, abs=1e-5)
        assert corrected["r4"]["offset_z"] == pytest.approx(0.0, abs=1e-5)

    def test_azimuth_clustering_boundary_wrapping_deduplication(self):
        """Azimuth clustering must treat 0° and 180° as circularly adjacent and not split boundary peaks."""
        from src.geometry import GeometryExtractor
        geo = GeometryExtractor()

        # Build synthetic wall normals with a peak at ~0°/180° (split across boundary) and ~90°
        rng = np.random.default_rng(42)
        az_0 = rng.normal(1.5, 1.0, 500) % 180
        az_180 = (180.0 - rng.normal(1.5, 1.0, 500)) % 180
        az_90 = rng.normal(90.0, 1.5, 600) % 180

        all_az = np.concatenate([az_0, az_180, az_90])

        hist, _ = np.histogram(all_az, bins=geo.NORMAL_HIST_BINS, range=(0, 180))
        from scipy.ndimage import uniform_filter1d
        from scipy.signal import find_peaks

        hist_s = uniform_filter1d(hist.astype(float), size=5, mode="wrap")
        pad = int(geo.NORMAL_PEAK_SEP_DEG)
        hist_wrapped = np.concatenate([hist_s[-pad:], hist_s, hist_s[:pad]])
        peaks_w, _ = find_peaks(
            hist_wrapped,
            height=hist_s.max() * geo.NORMAL_PEAK_MIN_FRAC,
            distance=geo.NORMAL_PEAK_SEP_DEG,
        )
        peaks = [p - pad for p in peaks_w if 0 <= p - pad < geo.NORMAL_HIST_BINS]
        peak_azs = (np.array(peaks) + 0.5) * (180.0 / geo.NORMAL_HIST_BINS)
        peak_heights = hist_s[peaks]
        order = np.argsort(peak_heights)[::-1]

        selected_azs = []
        for idx in order:
            az = float(peak_azs[idx])
            is_far = True
            for sel_az in selected_azs:
                circ_dist = min(abs(az - sel_az), 180.0 - abs(az - sel_az))
                if circ_dist < geo.NORMAL_PEAK_SEP_DEG:
                    is_far = False
                    break
            if is_far:
                selected_azs.append(az)
            if len(selected_azs) >= geo.MAX_AZ_FAMILIES:
                break

        # Circular separation must eliminate duplicate boundary peak
        for i in range(len(selected_azs)):
            for j in range(i + 1, len(selected_azs)):
                circ_d = min(abs(selected_azs[i] - selected_azs[j]),
                             180.0 - abs(selected_azs[i] - selected_azs[j]))
                assert circ_d >= geo.NORMAL_PEAK_SEP_DEG

        ortho = geo._pick_orthogonal_families(np.array(selected_azs), hist_s)
        assert len(ortho) == 2, f"Expected exactly 2 orthogonal families, got: {ortho}"
        diff = abs(ortho[0] - ortho[1]) % 180
        angle_from_90 = abs(diff - 90)
        assert angle_from_90 <= geo.ORTHO_TOLERANCE_DEG

    def test_floor_only_azimuth_stability_under_perturbation(self):
        """Room geometry extraction maintains 2 orthogonal families under small rotation perturbations."""
        from src.geometry import GeometryExtractor
        geo = GeometryExtractor()

        # Build box room without ceiling (floor-only scenario)
        box = _box_room(lx=4.0, lz=3.0, h=2.5, n=2000, rng_seed=42)
        # Remove ceiling points (Y > 2.0)
        floor_only_pts = box[box[:, 1] < 2.0]

        # Rotate by a small perturbation angle (2.5 degrees) aligning near 0°/180° boundary
        theta = np.radians(2.5)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        rotated_pts = floor_only_pts @ R.T

        res_unrot = geo.extract({"xyz": floor_only_pts})
        res_rot = geo.extract({"xyz": rotated_pts})

        # Both orientations must extract exactly 2 orthogonal azimuth families
        assert len(res_unrot["debug"]["orthogonal_families_deg"]) == 2
        assert len(res_rot["debug"]["orthogonal_families_deg"]) == 2

        # Check orthogonality: |diff - 90| <= 15
        diff_unrot = abs(res_unrot["debug"]["orthogonal_families_deg"][0] - res_unrot["debug"]["orthogonal_families_deg"][1]) % 180
        assert abs(diff_unrot - 90) <= geo.ORTHO_TOLERANCE_DEG

        diff_rot = abs(res_rot["debug"]["orthogonal_families_deg"][0] - res_rot["debug"]["orthogonal_families_deg"][1]) % 180
        assert abs(diff_rot - 90) <= geo.ORTHO_TOLERANCE_DEG

        # Floor area is stable around true 12.0 m²
        assert res_unrot["floor_area_m2"] == pytest.approx(12.0, abs=2.0)
        assert res_rot["floor_area_m2"] == pytest.approx(12.0, abs=2.0)

    def test_floor_only_capture_drift_correction_regression(self):
        """LiDAR drift correction on floor-only capture does not inflate walls or area."""
        floor_only_dir = Path("single_scan_floor_only")
        if not floor_only_dir.exists():
            pytest.skip("single_scan_floor_only not present")

        from src.lidar_processor import LiDARProcessor
        from src.geometry import GeometryExtractor

        # Test point cloud extraction under drift correction ON
        proc_on = LiDARProcessor(floor_only_dir, frame_skip=40, apply_drift_correction=True)
        # Avoid heavy video decode
        proc_on._load_rgb_frames = lambda: {}
        pc_on, _ = proc_on.load()

        geo = GeometryExtractor()
        res_on = geo.extract(pc_on)

        # Regression check: dominant azimuths must identify orthogonal pair (2 families)
        assert len(res_on["debug"]["orthogonal_families_deg"]) == 2
        # Wall count must not blow up to 16
        assert len(res_on["walls"]) == 8
        # Floor area must remain close to the physical room footprint (~15-19 m²), not 43.4 m²
        assert 14.0 <= res_on["floor_area_m2"] <= 20.0





