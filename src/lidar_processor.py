"""
LiDAR Processor — with gravity alignment
==========================================
Key calibration assumptions (all labelled):
  1. Depth intrinsics scaled from RGB: depth_fx = rgb_fx * (256/1920)
     This is an assumption — no separate depth calibration file found.
  2. Depth unit: 0.001 m/raw_unit (millimetre hypothesis).
     Max observed raw value ~5852 => 5.852 m, consistent with LiDAR range.
  3. Gravity alignment: derived from IMU mean accelerometer over the session.
     This corrects the ~31-degree tilt between the world frame (set at device
     startup) and the gravity-aligned frame needed for floor/ceiling/wall detection.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Calibration assumptions
RGB_WIDTH  = 1920
RGB_HEIGHT = 1440
DEPTH_WIDTH  = 256
DEPTH_HEIGHT = 192
DEPTH_SCALE_X = DEPTH_WIDTH  / RGB_WIDTH    # 0.13333
DEPTH_SCALE_Y = DEPTH_HEIGHT / RGB_HEIGHT   # 0.13333

DEPTH_SCALE_ASSUMPTION = 0.001   # metres per raw uint16 unit (ASSUMPTION)
DEFAULT_CONFIDENCE_THRESHOLD = 1
DEFAULT_MAX_DEPTH_M = 5.5


class LiDARProcessor:
    """
    Loads LiDAR scan data and returns a gravity-aligned 3D point cloud.

    Directory layout:
        <path>/depth/       — uint16 PNG depth frames (256x192)
        <path>/confidence/  — uint8 PNG confidence frames
        <path>/rgb.mp4      — RGB video (1920x1440, optional)
        <path>/odometry.csv — poses + per-frame intrinsics (RGB space)
        <path>/imu.csv      — accelerometer data for gravity estimation
        <path>/camera_matrix.csv — static RGB intrinsics (fallback)
    """

    def __init__(
        self,
        input_path: Path,
        frame_skip: int = 10,
        confidence_threshold: int = DEFAULT_CONFIDENCE_THRESHOLD,
        max_depth_m: float = DEFAULT_MAX_DEPTH_M,
        depth_scale: float = DEPTH_SCALE_ASSUMPTION,
        apply_drift_correction: bool = True,
        loop_closure_threshold_m: float = 0.8,
        min_loop_length_m: float = 3.0,
        verbose: bool = False,
    ):
        self.input_path = Path(input_path)
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.max_depth_m = max_depth_m
        self.depth_scale = depth_scale
        self.apply_drift_correction = apply_drift_correction
        self.loop_closure_threshold_m = loop_closure_threshold_m
        self.min_loop_length_m = min_loop_length_m
        self.verbose = verbose
        self.drift_info: Dict[str, Any] = {
            "drift_correction_applied": False,
            "method": "none",
            "loop_closure_status": "not_run",
            "trajectory_length_m": 0.0,
            "loop_closure_pre_residual_m": 0.0,
            "loop_closure_post_residual_m": 0.0,
        }

    def _apply_trajectory_drift_correction(
        self, poses: Dict[int, Dict]
    ) -> Tuple[Dict[int, Dict], Dict[str, Any]]:
        """
        Deterministic linear trajectory loop closure (traverse closure / Bowditch rule).

        If the trajectory forms a closed loop (start-to-end distance <= loop_closure_threshold_m
        and total path length >= min_loop_length_m), the closure gap:
            delta_t = t_end - t_start
        is distributed linearly along the cumulative traveled distance:
            t'_k = t_k - (s_k / total_length) * delta_t

        This guarantees t'_0 == t_0 and t'_{end} == t_0, eliminating accumulated
        translational drift across the loop before unprojection into world coordinates.
        """
        if len(poses) < 2:
            return poses, {
                "drift_correction_applied": False,
                "method": "none",
                "loop_closure_status": "insufficient_poses",
                "trajectory_length_m": 0.0,
                "loop_closure_pre_residual_m": 0.0,
                "loop_closure_post_residual_m": 0.0,
            }

        sorted_frames = sorted(poses.keys())
        translations = np.array(
            [[poses[f]["x"], poses[f]["y"], poses[f]["z"]] for f in sorted_frames],
            dtype=np.float64,
        )

        deltas = np.diff(translations, axis=0)
        step_dists = np.linalg.norm(deltas, axis=1)
        cum_dists = np.concatenate([[0.0], np.cumsum(step_dists)])
        total_length = float(cum_dists[-1])

        t_start = translations[0]
        t_end = translations[-1]
        closure_gap_vec = t_end - t_start
        pre_residual = float(np.linalg.norm(closure_gap_vec))

        if pre_residual <= self.loop_closure_threshold_m and total_length >= self.min_loop_length_m:
            factors = (cum_dists / max(total_length, 1e-6))[:, np.newaxis]
            corrected_translations = translations - factors * closure_gap_vec
            post_residual = float(
                np.linalg.norm(corrected_translations[-1] - corrected_translations[0])
            )

            corrected_poses = {}
            for i, f in enumerate(sorted_frames):
                new_pose = dict(poses[f])
                new_pose["x"] = float(corrected_translations[i, 0])
                new_pose["y"] = float(corrected_translations[i, 1])
                new_pose["z"] = float(corrected_translations[i, 2])
                corrected_poses[f] = new_pose

            drift_info = {
                "drift_correction_applied": True,
                "method": "linear_trajectory_loop_closure",
                "loop_closure_status": "applied_closed_loop",
                "trajectory_length_m": round(total_length, 3),
                "loop_closure_pre_residual_m": round(pre_residual, 4),
                "loop_closure_post_residual_m": round(post_residual, 4),
                "n_poses_corrected": len(sorted_frames),
            }
            logger.info(
                f"Trajectory loop closure applied: {pre_residual:.3f} m drift corrected to "
                f"{post_residual:.4f} m over {total_length:.2f} m path ({len(sorted_frames)} poses)"
            )
            return corrected_poses, drift_info
        else:
            status = (
                "open_trajectory_unclosed"
                if pre_residual > self.loop_closure_threshold_m
                else "path_too_short"
            )
            drift_info = {
                "drift_correction_applied": False,
                "method": "none",
                "loop_closure_status": status,
                "trajectory_length_m": round(total_length, 3),
                "loop_closure_pre_residual_m": round(pre_residual, 4),
                "loop_closure_post_residual_m": round(pre_residual, 4),
                "n_poses_corrected": 0,
            }
            logger.info(
                f"Trajectory drift correction skipped ({status}): gap={pre_residual:.3f} m, "
                f"length={total_length:.2f} m"
            )
            return poses, drift_info

    def _audit_unmitigated_drift(
        self, poses: Dict[int, Dict]
    ) -> Dict[str, Any]:
        """Audit trajectory drift without applying correction (OFF path)."""
        if len(poses) < 2:
            return {
                "drift_correction_applied": False,
                "method": "none",
                "loop_closure_status": "insufficient_poses",
                "trajectory_length_m": 0.0,
                "loop_closure_pre_residual_m": 0.0,
                "loop_closure_post_residual_m": 0.0,
            }

        sorted_frames = sorted(poses.keys())
        translations = np.array(
            [[poses[f]["x"], poses[f]["y"], poses[f]["z"]] for f in sorted_frames],
            dtype=np.float64,
        )
        deltas = np.diff(translations, axis=0)
        total_length = float(np.sum(np.linalg.norm(deltas, axis=1)))
        closure_gap = float(np.linalg.norm(translations[-1] - translations[0]))

        return {
            "drift_correction_applied": False,
            "method": "none",
            "loop_closure_status": "open_loop_disabled",
            "trajectory_length_m": round(total_length, 3),
            "loop_closure_pre_residual_m": round(closure_gap, 4),
            "loop_closure_post_residual_m": round(closure_gap, 4),
            "n_poses_corrected": 0,
        }

    def load(self) -> Tuple[Dict, List]:
        """
        Load scan and return:
          point_cloud: {"xyz": Nx3 float32 (gravity-aligned), "rgb": Nx3 uint8}
          frames_rgb:  [(frame_idx, rgb_array), ...]
        """
        logger.info("Loading camera matrix...")
        K_rgb = self._load_camera_matrix()

        logger.info("Loading odometry...")
        poses = self._load_odometry()

        if self.apply_drift_correction:
            poses, self.drift_info = self._apply_trajectory_drift_correction(poses)
        else:
            self.drift_info = self._audit_unmitigated_drift(poses)

        logger.info("Loading depth + confidence frames...")
        depth_dir  = self.input_path / "depth"
        conf_dir   = self.input_path / "confidence"
        depth_files = sorted(depth_dir.glob("*.png"))
        logger.info(f"Found {len(depth_files)} depth frames")

        rgb_frames = self._load_rgb_frames()
        all_pts, all_rgb, frames_rgb = [], [], []
        selected = depth_files[:: self.frame_skip]
        logger.info(f"Processing {len(selected)} frames (skip={self.frame_skip})")

        for i, depth_file in enumerate(selected):
            if i % 50 == 0:
                logger.info(f"  Frame {i}/{len(selected)}...")

            try:
                frame_idx = int(depth_file.stem)
            except ValueError:
                continue
            if frame_idx not in poses:
                continue

            pose = poses[frame_idx]

            # Depth
            depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            if depth_raw is None:
                continue
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[:, :, 0]
            depth_m = depth_raw.astype(np.float32) * self.depth_scale

            # Confidence
            mask = self._load_confidence_mask(depth_m, conf_dir / depth_file.name)
            if mask.sum() < 10:
                continue

            # Intrinsics — prefer per-frame from odometry (in RGB space) scaled to depth
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

            # Unproject + transform to world (use float32 throughout to limit RAM)
            pts_cam = self._unproject(depth_m, fx, fy, cx, cy, mask)
            R = self._quat_to_rot(pose["qx"], pose["qy"], pose["qz"], pose["qw"]).astype(np.float32)
            t = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32)
            pts_world = pts_cam @ R.T + t   # (N,3) @ (3,3) in float32, no transpose copy
            all_pts.append(pts_world)

            # Colour
            if frame_idx in rgb_frames:
                rgb_full = rgb_frames[frame_idx]
                h, w = depth_raw.shape
                uu, vv = np.meshgrid(np.arange(w), np.arange(h))
                u_rgb = (uu[mask] * RGB_WIDTH  / DEPTH_WIDTH).astype(int).clip(0, RGB_WIDTH - 1)
                v_rgb = (vv[mask] * RGB_HEIGHT / DEPTH_HEIGHT).astype(int).clip(0, RGB_HEIGHT - 1)
                all_rgb.append(rgb_full[v_rgb, u_rgb])
                if i % 5 == 0:
                    frames_rgb.append((frame_idx, rgb_full))
            else:
                all_rgb.append(np.full((pts_world.shape[0], 3), 128, dtype=np.uint8))

        if not all_pts:
            raise RuntimeError("No valid frames loaded.")

        xyz = np.vstack(all_pts)
        rgb = np.vstack(all_rgb).astype(np.uint8)
        logger.info(
            f"Built point cloud: {len(xyz)} points "
            f"(depth_scale={self.depth_scale} m/unit [ASSUMPTION])"
        )
        logger.info(f"  World-frame Y: [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}] m")

        # Gravity alignment from IMU
        R_grav = self._compute_gravity_alignment()
        if R_grav is not None:
            # Use float32 rotation (avoids ~350MB float64 intermediate on large clouds)
            R32 = R_grav.astype(np.float32)
            chunk = 1_000_000
            for i in range(0, len(xyz), chunk):
                xyz[i:i+chunk] = (R32 @ xyz[i:i+chunk].T).T
            logger.info(
                f"  After gravity alignment Y: [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}] m"
            )

        return {"xyz": xyz, "rgb": rgb}, frames_rgb

    def _load_confidence_mask(self, depth_m: np.ndarray, conf_path: Path) -> np.ndarray:
        """Load confidence PNG and build validity mask."""
        base = (depth_m > 0.05) & (depth_m < self.max_depth_m)
        if not conf_path.exists():
            return base
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        if conf is None:
            return base
        if conf.ndim == 3:
            conf = conf[:, :, 0]
        return base & (conf >= self.confidence_threshold)

    def _compute_gravity_alignment(self) -> Optional[np.ndarray]:
        """
        Estimate gravity from IMU and build a rotation that maps
        world-space 'up' to +Y.
        """
        imu_path = self.input_path / "imu.csv"
        if not imu_path.exists():
            logger.warning("imu.csv not found; gravity alignment skipped")
            return None

        accels = []
        with open(imu_path, newline="") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [k.strip() for k in reader.fieldnames]
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    ax = float(row.get("a_x", row.get("ax", "0")) or "0")
                    ay = float(row.get("a_y", row.get("ay", "0")) or "0")
                    az = float(row.get("a_z", row.get("az", "0")) or "0")
                    accels.append([ax, ay, az])
                except (ValueError, KeyError):
                    continue

        if len(accels) < 10:
            logger.warning("Too few IMU samples; gravity alignment skipped")
            return None

        grav = np.mean(accels, axis=0)
        norm = float(np.linalg.norm(grav))
        if norm < 0.1:
            logger.warning(f"IMU accel too small ({norm:.3f}); skipping gravity alignment")
            return None

        grav /= norm  # unit vector pointing DOWN in world space
        up_world = -grav
        target = np.array([0., 1., 0.])

        tilt_deg = float(np.degrees(np.arccos(np.clip(np.dot(up_world, target), -1, 1))))
        logger.info(
            f"  IMU gravity: {grav.round(4)}  tilt={tilt_deg:.1f} deg from world-Y"
        )

        v = np.cross(up_world, target)
        s = float(np.linalg.norm(v))
        c = float(np.dot(up_world, target))
        if s < 1e-8:
            R = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
        else:
            Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + Kx + Kx @ Kx * ((1 - c) / (s * s))
        return R.astype(np.float64)

    def _load_camera_matrix(self) -> np.ndarray:
        """Load RGB camera matrix from CSV (fallback to known values)."""
        K = np.array([
            [1599.696, 0.,      955.5105],
            [0.,      1599.696, 717.8084],
            [0.,       0.,        1.    ],
        ])
        path = self.input_path / "camera_matrix.csv"
        if not path.exists():
            return K
        try:
            vals = []
            with open(path) as f:
                for line in f:
                    row = [v.strip() for v in line.split(",") if v.strip()]
                    vals.extend([float(v) for v in row if v])
            if len(vals) >= 4:
                K = np.array([
                    [vals[0], 0, vals[2]],
                    [0, vals[1], vals[3]],
                    [0, 0, 1],
                ])
        except Exception as e:
            logger.warning(f"camera_matrix.csv parse error: {e}")
        return K

    def _load_odometry(self) -> Dict[int, Dict]:
        """Load per-frame poses from odometry.csv. Strips whitespace from headers."""
        path = self.input_path / "odometry.csv"
        poses = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [k.strip() for k in reader.fieldnames]
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    fi = int(row["frame"])
                    p = {k: float(row[k]) for k in ["x","y","z","qx","qy","qz","qw"]}
                    for k in ["fx","fy","cx","cy"]:
                        if row.get(k):
                            p[k] = float(row[k])
                    poses[fi] = p
                except (KeyError, ValueError):
                    continue
        logger.debug(f"Loaded {len(poses)} poses")
        return poses

    def _load_rgb_frames(self, max_frames: int = 200) -> Dict[int, np.ndarray]:
        """
        Load RGB frames from video with memory cap.
        Only loads 1 in 5 of the already-subsampled frames, up to max_frames total.
        This prevents OOM on long scans (e.g., 5251-frame datasets).
        """
        frames = {}
        vp = self.input_path / "rgb.mp4"
        if not vp.exists():
            return frames
        try:
            cap = cv2.VideoCapture(str(vp))
            fi = 0
            rgb_skip = max(self.frame_skip * 5, 1)  # load 1 per 5 depth frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if fi % rgb_skip == 0 and len(frames) < max_frames:
                    frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fi += 1
            cap.release()
        except Exception as e:
            logger.warning(f"RGB video load failed: {e}. Continuing without colour.")
            frames = {}
        logger.debug(f"Loaded {len(frames)} RGB frames (cap={max_frames})")
        return frames

    def _unproject(self, depth_m, fx, fy, cx, cy, mask) -> np.ndarray:
        """Pinhole unproject: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy, Z=depth."""
        h, w = depth_m.shape[:2]
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        z = depth_m[mask]
        return np.stack([(u[mask] - cx) * z / fx,
                         (v[mask] - cy) * z / fy,
                         z], axis=-1)

    @staticmethod
    def _quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
        """Unit quaternion to 3x3 rotation matrix."""
        n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
        if n < 1e-10:
            return np.eye(3)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        return np.array([
            [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz),  2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),  1-2*(qx*qx+qy*qy)],
        ])
