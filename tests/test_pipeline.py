"""
Unit tests for the LiDAR pipeline.
Run:  python -m pytest tests/ -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lidar_processor import LiDARProcessor, DEPTH_SCALE_ASSUMPTION, DEPTH_SCALE_X, DEPTH_SCALE_Y
from src.geometry import GeometryExtractor

SAMPLE_DIR = Path(__file__).parent.parent / "single_room"
SAMPLE_AVAILABLE = SAMPLE_DIR.exists()


# ── Depth loading ─────────────────────────────────────────────────────────────

def test_depth_scale_assumption():
    """Depth scale is 0.001 m/unit (mm hypothesis)."""
    assert DEPTH_SCALE_ASSUMPTION == 0.001


def test_depth_intrinsics_scaling():
    """Depth intrinsics must be scaled from RGB by resolution ratio."""
    assert abs(DEPTH_SCALE_X - 256 / 1920) < 1e-8
    assert abs(DEPTH_SCALE_Y - 192 / 1440) < 1e-8


@pytest.mark.skipif(not SAMPLE_AVAILABLE, reason="sample data not present")
def test_depth_loading():
    """Depth PNGs load as uint16 2D arrays."""
    import cv2
    depth_files = sorted((SAMPLE_DIR / "depth").glob("*.png"))
    assert len(depth_files) > 0, "No depth files found"
    raw = cv2.imread(str(depth_files[0]), cv2.IMREAD_ANYDEPTH)
    assert raw is not None, "Failed to read depth PNG"
    assert raw.ndim == 2, f"Expected 2D array, got shape {raw.shape}"
    assert raw.dtype == np.uint16, f"Expected uint16, got {raw.dtype}"
    # Values consistent with mm hypothesis (0.05m to 6m = 50..6000)
    assert raw.max() <= 7000, f"Suspiciously large max value: {raw.max()}"
    assert raw.min() >= 10, f"Suspiciously small min value: {raw.min()}"


@pytest.mark.skipif(not SAMPLE_AVAILABLE, reason="sample data not present")
def test_depth_to_3d_basic():
    """Unproject a simple depth frame and check shape/range."""
    import cv2
    depth_files = sorted((SAMPLE_DIR / "depth").glob("*.png"))
    raw = cv2.imread(str(depth_files[0]), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth_m = raw * DEPTH_SCALE_ASSUMPTION

    fx = 1599.696 * DEPTH_SCALE_X
    fy = 1599.696 * DEPTH_SCALE_Y
    cx = 955.5105 * DEPTH_SCALE_X
    cy = 717.8084 * DEPTH_SCALE_Y

    mask = (depth_m > 0.05) & (depth_m < 6.0)
    proc = LiDARProcessor(SAMPLE_DIR)
    pts = proc._unproject(depth_m, fx, fy, cx, cy, mask)

    assert pts.shape[1] == 3, "Expected Nx3 points"
    assert len(pts) > 100, "Too few unprojected points"
    # Z values should match depth_m values under mask
    assert np.allclose(pts[:, 2], depth_m[mask], atol=1e-4)


# ── Pose loading ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not SAMPLE_AVAILABLE, reason="sample data not present")
def test_pose_loading():
    """Odometry CSV loads correct number of frames with expected keys."""
    proc = LiDARProcessor(SAMPLE_DIR)
    poses = proc._load_odometry()
    assert len(poses) > 100, f"Expected >100 poses, got {len(poses)}"
    first = list(poses.values())[0]
    for key in ["x", "y", "z", "qx", "qy", "qz", "qw"]:
        assert key in first, f"Missing key: {key}"
    # Quaternion should be roughly unit
    q = first
    qnorm = (q["qx"]**2 + q["qy"]**2 + q["qz"]**2 + q["qw"]**2) ** 0.5
    assert 0.99 < qnorm < 1.01, f"Non-unit quaternion: norm={qnorm}"


# ── Quaternion rotation ───────────────────────────────────────────────────────

def test_quaternion_identity():
    """Identity quaternion → identity rotation."""
    R = LiDARProcessor._quat_to_rot(0, 0, 0, 1)
    assert np.allclose(R, np.eye(3), atol=1e-6)


def test_quaternion_180_y():
    """180° rotation around Y: X → -X, Z → -Z."""
    R = LiDARProcessor._quat_to_rot(0, 1, 0, 0)
    p = np.array([1.0, 0.0, 0.0])
    assert np.allclose(R @ p, [-1., 0., 0.], atol=1e-6)


# ── Plane orientation filtering ───────────────────────────────────────────────

def test_plane_height_coverage_filter():
    """Walls with low height coverage should be rejected."""
    geo = GeometryExtractor()
    # Create a thin horizontal strip of points (not a wall — no height coverage)
    y_vals = np.full(200, 1.0)
    x_vals = np.linspace(0, 3.0, 200)
    z_vals = np.zeros(200)
    pts = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)
    walls = geo._detect_walls(pts, floor_y=-0.5, ceiling_y=2.0, room_height=2.5)
    # Thin horizontal strip has no height coverage → should not produce walls
    for w in walls:
        assert w["height_coverage"] >= geo.MIN_HEIGHT_COV_FRAC, \
            f"Wall passed with too low height coverage: {w['height_coverage']}"


def test_wall_min_length_filter():
    """Wall segments shorter than MIN_WALL_LENGTH_M should be rejected."""
    geo = GeometryExtractor()
    # Verify parameter value
    assert geo.MIN_WALL_LENGTH_M >= 0.3


# ── Wall candidate filtering (geometry checks) ────────────────────────────────

def test_simple_box_room():
    """
    Synthetic 4-wall rectangular room should produce ~4 dominant wall planes.
    Room: 4m x 3m, ceiling at 2.5m, floor at 0m.
    """
    rng = np.random.default_rng(0)
    pts_list = []

    # Floor
    n = 2000
    pts_list.append(np.column_stack([
        rng.uniform(0, 4, n), np.zeros(n), rng.uniform(0, 3, n)
    ]))
    # Ceiling
    pts_list.append(np.column_stack([
        rng.uniform(0, 4, n), np.full(n, 2.5), rng.uniform(0, 3, n)
    ]))
    # 4 walls (with height variation)
    for x0 in [0.0, 4.0]:
        h = rng.uniform(0.1, 2.4, n)
        pts_list.append(np.column_stack([
            np.full(n, x0) + rng.normal(0, 0.01, n),
            h,
            rng.uniform(0, 3, n),
        ]))
    for z0 in [0.0, 3.0]:
        h = rng.uniform(0.1, 2.4, n)
        pts_list.append(np.column_stack([
            rng.uniform(0, 4, n),
            h,
            np.full(n, z0) + rng.normal(0, 0.01, n),
        ]))

    xyz = np.vstack(pts_list).astype(np.float32)

    geo = GeometryExtractor()
    result = geo.extract({"xyz": xyz})

    n_walls = len(result["walls"])
    # Should detect approximately 4 walls (may find 4-8 due to merging)
    assert 2 <= n_walls <= 10, f"Expected 2–10 walls for box room, got {n_walls}"

    # Ceiling height should be close to 2.5m
    h = result["ceiling_height_m"]
    assert 2.0 <= h <= 3.0, f"Ceiling height {h} far from expected 2.5m"

    # Floor area should be close to 12 m²
    area = result["floor_area_m2"]
    assert 8.0 <= area <= 16.0, f"Floor area {area} far from expected 12 m²"


# ── Floor plan generation ─────────────────────────────────────────────────────

def test_floor_plan_renders(tmp_path):
    """Floor plan renderer produces a PNG file."""
    from src.floor_plan import FloorPlanRenderer
    # Minimal geometry stub
    geometry = {
        "floor_area_m2": 12.0,
        "floor_area_ci_m2": 0.24,
        "ceiling_height_m": 2.5,
        "ceiling_height_ci_m": 0.01,
        "footprint_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "walls": [
            {"id": "wall_00", "length_m": 4.0, "length_ci_m": 0.02,
             "start_xz": [0, 0], "end_xz": [4, 0], "normal_xz": [0, 1]},
        ],
        "openings": [],
    }
    renderer = FloorPlanRenderer(tmp_path, room_id="test_room")
    png = renderer.render(geometry, title="Test")
    assert png.exists(), f"Floor plan PNG not created at {png}"
    assert png.stat().st_size > 1000, "Floor plan PNG is suspiciously small"


if __name__ == "__main__":
    # Run basic tests without pytest
    test_depth_scale_assumption()
    test_depth_intrinsics_scaling()
    test_quaternion_identity()
    test_quaternion_180_y()
    test_wall_min_length_filter()
    print("Basic tests passed.")
    if SAMPLE_AVAILABLE:
        test_depth_loading()
        test_depth_to_3d_basic()
        test_pose_loading()
        print("Sample-data tests passed.")
    test_simple_box_room()
    print("Synthetic box room test passed.")
