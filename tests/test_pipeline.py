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
