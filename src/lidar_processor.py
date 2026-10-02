"""
LiDAR Processor — revised
==========================
Key fixes vs previous version:
  1. Depth camera intrinsics are NOT the same as RGB intrinsics.
     RGB is 1920x1440; depth is 256x192.
     We scale the RGB intrinsics by (256/1920, 192/1440) to get depth intrinsics.
     This is documented as an assumption — the data does not contain a separate
     depth calibration file, so we derive it from the ratio of resolutions.
     If a future dataset provides explicit depth intrinsics, pass them via config.

  2. Depth unit assumption: 0.001 m per raw uint16 unit (millimetres hypothesis).
     Max observed raw value is ~5852 => 5.852 m, consistent with iPhone LiDAR range.
     This is labelled as an assumption (DEPTH_SCALE_ASSUMPTION) throughout.

  3. Odometry CSV has leading spaces in header — strip all keys.

  4. Confidence PNGs load as (H,W,1) sometimes — always squeeze to 2D.

  5. Per-frame intrinsics (fx, fy, cx, cy) are present in odometry.csv.
     We prefer those over the static camera_matrix.csv, scaled to depth resolution.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Calibration assumptions (all clearly labeled) ───────────────────────────
# RGB camera resolution (source of intrinsics in camera_matrix.csv / odometry)
RGB_WIDTH = 1920
RGB_HEIGHT = 1440

# Depth sensor resolution
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192

# Scale factors to convert RGB intrinsics → depth intrinsics
DEPTH_SCALE_X = DEPTH_WIDTH / RGB_WIDTH    # = 0.13333...
DEPTH_SCALE_Y = DEPTH_HEIGHT / RGB_HEIGHT  # = 0.13333...

# Depth unit assumption: raw uint16 → metres
# Evidence: max observed raw value ~5852, which at 0.001 m/unit = 5.852 m
# This is consistent with iPhone-class LiDAR range (≤ 5 m typical).
DEPTH_SCALE_ASSUMPTION = 0.001  # metres per raw unit (ASSUMPTION, not verified)

# Configurable via constructor
DEFAULT_CONFIDENCE_THRESHOLD = 1   # 0=low,1=medium,2=high; filter ≤ 0 by default
DEFAULT_MAX_DEPTH_M = 5.5          # metres — clip obviously-far points


class LiDARProcessor:
    """
    Loads and processes the LiDAR scan data from a single-room directory.

    Directory layout expected:
        <input_path>/
            depth/          — 16-bit PNG depth frames (uint16, raw units)
            confidence/     — 8-bit PNG confidence frames (0/1/2)
            rgb.mp4         — RGB video (1920x1440, 60fps) — optional
            odometry.csv    — per-frame camera poses + per-frame intrinsics
            camera_matrix.csv — static camera matrix (fallback)
            imu.csv         — IMU data (not used for geometry)
    """

    def __init__(
        self,
        input_path: Path,
        frame_skip: int = 10,
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
        max_depth_m: float = DEFAULT_MAX_DEPTH_M,
        depth_scale: float = DEPTH_SCALE_ASSUMPTION,
        verbose: bool = False,
    ):
        self.input_path = Path(input_path)
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.max_depth_m = max_depth_m
        self.depth_scale = depth_scale  # metres per raw unit
        self.verbose = verbose

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self) -> Tuple[Dict, List]:
        """
        Load the scan and return:
          point_cloud: {"xyz": Nx3 float32, "rgb": Nx3 uint8}
          frames_rgb:  [(frame_idx, rgb_np_array), ...]  — for damage detection
        """
        logger.info("Loading camera matrix...")
        K_rgb = self._load_camera_matrix()

        logger.info("Loading odometry...")
        poses = self._load_odometry()

        logger.info("Loading depth + confidence frames...")
        depth_dir = self.input_path / "depth"
        conf_dir = self.input_path / "confidence"

        depth_files = sorted(depth_dir.glob("*.png"))
        logger.info(f"Found {len(depth_files)} depth frames")

        # Load RGB video for colour (best-effort; skipped if not present)
        rgb_frames = self._load_rgb_frames(len(depth_files))

        all_pts_world = []
        all_rgb = []
        frames_rgb = []

        selected = depth_files[:: self.frame_skip]
        logger.info(f"Processing {len(selected)} frames (skip={self.frame_skip})")

        for i, depth_file in enumerate(selected):
            if i % 50 == 0:
                logger.info(f"  Frame {i}/{len(selected)}...")

            # Frame index from filename stem (e.g. "000042" → 42)
            try:
                frame_idx = int(depth_file.stem)
            except ValueError:
                continue

            if frame_idx not in poses:
                continue

            pose = poses[frame_idx]

            # ── Depth ────────────────────────────────────────────────────
            depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            if depth_raw is None:
                continue
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[:, :, 0]

            depth_m = depth_raw.astype(np.float32) * self.depth_scale

            # ── Confidence ───────────────────────────────────────────────
            conf_file = conf_dir / depth_file.name
            if conf_file.exists():
                conf = cv2.imread(str(conf_file), cv2.IMREAD_UNCHANGED)
                if conf is None:
                    conf = cv2.imread(str(conf_file), cv2.IMREAD_GRAYSCALE)
                if conf is not None and conf.ndim == 3:
                    conf = conf[:, :, 0]
                if conf is not None:
                    mask = (
                        (conf >= self.confidence_threshold)
                        & (depth_m > 0.05)
                        & (depth_m < self.max_depth_m)
                    )
                else:
                    mask = (depth_m > 0.05) & (depth_m < self.max_depth_m)
            else:
                mask = (depth_m > 0.05) & (depth_m < self.max_depth_m)

            if mask.sum() < 10:
                continue

            # ── Intrinsics — prefer per-frame values from odometry ────────
            # odometry columns: fx, fy, cx, cy are in RGB pixel space
            # → scale to depth resolution
            if "fx" in pose:
                fx = pose["fx"] * DEPTH_SCALE_X
                fy = pose["fy"] * DEPTH_SCALE_Y
                cx = pose["cx"] * DEPTH_SCALE_X
                cy = pose["cy"] * DEPTH_SCALE_Y
            else:
                fx = K_rgb[0, 0] * DEPTH_SCALE_X
                fy = K_rgb[1, 1] * DEPTH_SCALE_Y
                cx = K_rgb[0, 2] * DEPTH_SCALE_X
                cy = K_rgb[1, 2] * DEPTH_SCALE_Y

            # ── Unproject depth → camera-space 3D ────────────────────────
            pts_cam = self._unproject(depth_m, fx, fy, cx, cy, mask)

            # ── Transform to world space via pose ─────────────────────────
            R = self._quat_to_rot(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
            t = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
            pts_world = (R @ pts_cam.T).T + t

            all_pts_world.append(pts_world.astype(np.float32))

            # ── Colour ────────────────────────────────────────────────────
            if frame_idx in rgb_frames:
                rgb_full = rgb_frames[frame_idx]
                # Sample colour at depth pixel positions
                h, w = depth_raw.shape
                uu, vv = np.meshgrid(np.arange(w), np.arange(h))
                # Map depth pixel → RGB pixel (scale up)
                u_rgb = (uu[mask] * RGB_WIDTH / DEPTH_WIDTH).astype(int).clip(0, RGB_WIDTH - 1)
                v_rgb = (vv[mask] * RGB_HEIGHT / DEPTH_HEIGHT).astype(int).clip(0, RGB_HEIGHT - 1)
                rgb_pts = rgb_full[v_rgb, u_rgb]
            else:
                rgb_pts = np.full((pts_world.shape[0], 3), 128, dtype=np.uint8)

            all_rgb.append(rgb_pts)

            # Collect RGB frame for damage detection (1 in 5 selected frames)
            if frame_idx in rgb_frames and i % 5 == 0:
                frames_rgb.append((frame_idx, rgb_frames[frame_idx]))

        if not all_pts_world:
            raise RuntimeError("No valid frames loaded. Check depth directory and poses.")

        xyz = np.vstack(all_pts_world)
        rgb = np.vstack(all_rgb) if all_rgb else np.zeros((len(xyz), 3), dtype=np.uint8)

        logger.info(
            f"Built point cloud: {len(xyz)} points "
            f"(depth_scale={self.depth_scale} m/unit [ASSUMPTION])"
        )
        return {"xyz": xyz, "rgb": rgb.astype(np.uint8)}, frames_rgb

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_camera_matrix(self) -> np.ndarray:
        """Load static RGB camera matrix from camera_matrix.csv (fallback)."""
        # Default: use known values from the data description
        K = np.array([
            [1599.696, 0.0,     955.5105],
            [0.0,     1599.696, 717.8084],
            [0.0,      0.0,       1.0   ],
        ], dtype=np.float64)

        path = self.input_path / "camera_matrix.csv"
        if not path.exists():
            # Try xlsx variant
            path_xlsx = self.input_path / "camera_matrix.xlsx"
            if not path_xlsx.exists():
                logger.warning("camera_matrix.csv not found; using hardcoded values")
                return K

        try:
            vals = []
            with open(path) as f:
                for line in f:
                    row = [v.strip() for v in line.split(",") if v.strip()]
                    if row:
                        vals.extend([float(v) for v in row if v])
            if len(vals) >= 4:
                K = np.array([
                    [vals[0], 0,      vals[2]],
                    [0,       vals[1], vals[3]],
                    [0,       0,       1      ],
                ], dtype=np.float64)
        except Exception as e:
            logger.warning(f"Could not parse camera_matrix.csv: {e}")

        return K

    def _load_odometry(self) -> Dict[int, Dict]:
        """
        Load per-frame poses from odometry.csv.
        Returns dict: frame_idx -> {x, y, z, qx, qy, qz, qw, fx, fy, cx, cy}

        Note: CSV header has spaces after commas (e.g. "timestamp, frame, x …")
        so DictReader keys include a leading space. We strip all keys/values.
        """
        path = self.input_path / "odometry.csv"
        poses = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [k.strip() for k in reader.fieldnames]
            for row in reader:
                row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                       for k, v in row.items()}
                try:
                    frame_idx = int(row["frame"])
                    p = {
                        "x":  float(row["x"]),
                        "y":  float(row["y"]),
                        "z":  float(row["z"]),
                        "qx": float(row["qx"]),
                        "qy": float(row["qy"]),
                        "qz": float(row["qz"]),
                        "qw": float(row["qw"]),
                    }
                    # Per-frame intrinsics (RGB space) — present in this dataset
                    for key in ["fx", "fy", "cx", "cy"]:
                        val_str = row.get(key, "").strip()
                        if val_str:
                            p[key] = float(val_str)
                    poses[frame_idx] = p
                except (KeyError, ValueError) as e:
                    logger.debug(f"Skipping odometry row: {e}")
        logger.debug(f"Loaded {len(poses)} poses")
        return poses

    def _load_rgb_frames(self, total_frames: int) -> Dict[int, np.ndarray]:
        """Load RGB frames from video file (best-effort)."""
        video_path = self.input_path / "rgb.mp4"
        frames = {}
        if not video_path.exists():
            return frames
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
        cap.release()
        logger.debug(f"Loaded {len(frames)} RGB frames from video")
        return frames

    def _unproject(
        self,
        depth_m: np.ndarray,
        fx: float, fy: float, cx: float, cy: float,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Unproject masked depth pixels → camera-space 3D points.
        Pinhole model: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy, Z=depth
        """
        h, w = depth_m.shape[:2]
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        z = depth_m[mask]
        x = (u[mask] - cx) * z / fx
        y = (v[mask] - cy) * z / fy
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Unit quaternion → 3x3 rotation matrix."""
        n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n < 1e-10:
            return np.eye(3)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        return np.array([
            [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
        ], dtype=np.float64)
