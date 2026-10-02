"""
LiDAR Data Processor
====================
Reads iPhone LiDAR data:
  - depth/*.png  : 16-bit depth maps (in millimetres, divide by 1000 for metres)
  - confidence/*.png : confidence maps (0=low, 1=medium, 2=high)
  - odometry.csv : camera poses (x, y, z, qx, qy, qz, qw) per frame
  - camera_matrix.csv : 3x3 intrinsic matrix

Outputs a fused point cloud (xyz + rgb) in world coordinates.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import csv

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class LiDARProcessor:
    """
    Fuses iPhone LiDAR depth frames into a single 3D point cloud.
    
    Key decisions:
    - Depth PNGs are 16-bit. Apple's Record3D/LiDAR depth is in mm 
      (divide by 1000 to get metres). We verify this by checking the 
      max depth value range.
    - Confidence filter removes noisy edge pixels.
    - We subsample frames (frame_skip) for speed; every 5th frame 
      still gives excellent density for room-scale scenes.
    - Poses from odometry.csv are in ARKit world frame (right-hand, Y-up).
    """

    # Apple LiDAR depth scale: 1 unit = 1mm (so /1000 = metres)
    DEPTH_SCALE = 1.0 / 1000.0

    def __init__(
        self,
        input_path: Path,
        frame_skip: int = 5,
        confidence_threshold: int = 1,
        max_depth_m: float = 8.0,
        verbose: bool = False,
    ):
        self.input_path = Path(input_path)
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.max_depth_m = max_depth_m
        self.verbose = verbose

    def load(self) -> Tuple[Dict, Optional[List]]:
        """
        Returns:
            point_cloud: dict with keys 'xyz' (Nx3 float32), 'rgb' (Nx3 uint8)
            frames_rgb: list of (frame_idx, rgb_frame_np) for damage detection
        """
        logger.info("Loading camera matrix...")
        K = self._load_camera_matrix()
        
        logger.info("Loading odometry...")
        poses = self._load_odometry()
        
        logger.info("Loading depth + confidence frames...")
        depth_dir = self.input_path / "depth"
        conf_dir = self.input_path / "confidence"
        
        depth_files = sorted(depth_dir.glob("*.png"))
        logger.info(f"Found {len(depth_files)} depth frames")
        
        # Load RGB video for colour
        rgb_frames = self._load_rgb_frames(len(depth_files))
        
        all_xyz = []
        all_rgb = []
        sampled_rgb_frames = []
        
        selected = depth_files[::self.frame_skip]
        logger.info(f"Processing {len(selected)} frames (skip={self.frame_skip})")
        
        for i, depth_file in enumerate(selected):
            if i % 50 == 0:
                logger.info(f"  Frame {i}/{len(selected)}...")
            
            frame_idx = int(depth_file.stem)
            
            # Skip if no pose available
            if frame_idx not in poses:
                continue
            
            pose = poses[frame_idx]
            
            # Load depth — uint16 PNG (16-bit), values in mm
            depth_raw = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            if depth_raw is None:
                continue
            
            # Safety: ensure 2D (should already be for these PNGs)
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[:, :, 0]
            
            depth_m = self._decode_depth(depth_raw)
            
            # Load confidence
            conf_file = conf_dir / depth_file.name
            if conf_file.exists():
                conf = cv2.imread(str(conf_file), cv2.IMREAD_GRAYSCALE)
                if conf is None:
                    conf = cv2.imread(str(conf_file), cv2.IMREAD_UNCHANGED)
                if conf is not None and conf.ndim == 3:
                    conf = conf[:, :, 0]  # squeeze (H,W,1) → (H,W)
                if conf is not None:
                    mask = (conf >= self.confidence_threshold) & (depth_m > 0.1) & (depth_m < self.max_depth_m)
                else:
                    mask = (depth_m > 0.1) & (depth_m < self.max_depth_m)
            else:
                mask = (depth_m > 0.1) & (depth_m < self.max_depth_m)
            
            # Unproject depth to 3D camera-space points
            pts_cam = self._unproject(depth_m, K, mask)
            
            if len(pts_cam) == 0:
                continue
            
            # Transform to world space
            pts_world = self._transform_to_world(pts_cam, pose)
            all_xyz.append(pts_world)
            
            # Colour from RGB frame
            if frame_idx in rgb_frames:
                rgb = rgb_frames[frame_idx]
                # Resize to depth resolution
                rgb_resized = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]))
                colours = rgb_resized[mask]
            else:
                colours = np.full((len(pts_world), 3), 180, dtype=np.uint8)
            all_rgb.append(colours)
            
            # Keep sampled RGB frames for damage detection
            if i % 10 == 0 and frame_idx in rgb_frames:
                sampled_rgb_frames.append((frame_idx, rgb_frames[frame_idx]))
        
        if not all_xyz:
            raise RuntimeError("No valid depth frames found. Check confidence_threshold and input data.")
        
        xyz = np.vstack(all_xyz).astype(np.float32)
        rgb = np.vstack(all_rgb).astype(np.uint8)
        
        logger.info(f"Built point cloud: {len(xyz)} points")
        
        return {"xyz": xyz, "rgb": rgb}, sampled_rgb_frames

    def _decode_depth(self, depth_raw: np.ndarray) -> np.ndarray:
        """
        Decode raw depth PNG to metres.
        Apple LiDAR depth from Record3D is stored in mm as uint16.
        """
        max_val = depth_raw.max()
        if max_val > 100:
            # Assume mm encoding (Apple standard)
            return depth_raw.astype(np.float32) * self.DEPTH_SCALE
        else:
            # Already in metres (some apps export float)
            return depth_raw.astype(np.float32)

    def _unproject(self, depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Unproject 2D depth pixels to 3D camera-space points.
        Uses the pinhole camera model: X = (u - cx) * Z / fx
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        h, w = depth_m.shape[:2]  # safe even if 3D array slips through
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        z = depth_m[mask]
        x = (u[mask] - cx) * z / fx
        y = (v[mask] - cy) * z / fy
        
        return np.stack([x, y, z], axis=-1)

    def _transform_to_world(self, pts_cam: np.ndarray, pose: Dict) -> np.ndarray:
        """
        Transform camera-space points to world space using pose (quaternion + translation).
        """
        R = self._quat_to_rot(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
        t = np.array([pose["x"], pose["y"], pose["z"]])
        
        # pts_world = R @ pts_cam.T + t[:, None]
        pts_world = (R @ pts_cam.T).T + t
        return pts_world

    def _quat_to_rot(self, qx, qy, qz, qw) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        # Normalize
        n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        return R

    def _load_camera_matrix(self) -> np.ndarray:
        """Load 3x3 intrinsic matrix from camera_matrix.csv."""
        path = self.input_path / "camera_matrix.csv"
        K = []
        with open(path) as f:
            for row in csv.reader(f):
                K.append([float(v.strip()) for v in row])
        return np.array(K, dtype=np.float64)

    def _load_odometry(self) -> Dict[int, Dict]:
        """
        Load per-frame poses from odometry.csv.
        Returns dict: frame_idx -> {x, y, z, qx, qy, qz, qw}
        
        Note: The CSV has spaces after commas in the header row
        (e.g. "timestamp, frame, x, y, ..."), so DictReader keys
        include a leading space. We strip all keys and values.
        """
        path = self.input_path / "odometry.csv"
        poses = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            # Strip spaces from fieldnames so " frame" → "frame"
            reader.fieldnames = [k.strip() for k in reader.fieldnames]
            for row in reader:
                # Strip spaces from values too
                row = {k.strip(): v.strip() if isinstance(v, str) else v 
                       for k, v in row.items()}
                try:
                    frame_idx = int(row["frame"])
                    poses[frame_idx] = {
                        "x":  float(row["x"]),
                        "y":  float(row["y"]),
                        "z":  float(row["z"]),
                        "qx": float(row["qx"]),
                        "qy": float(row["qy"]),
                        "qz": float(row["qz"]),
                        "qw": float(row["qw"]),
                    }
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping odometry row: {e}")
                    continue
        logger.debug(f"Loaded {len(poses)} poses")
        return poses

    def _load_rgb_frames(self, n_frames: int) -> Dict[int, np.ndarray]:
        """
        Load RGB frames from rgb.mp4.
        Only loads every frame_skip-th frame for efficiency.
        Returns dict: frame_idx -> np.ndarray (H, W, 3) BGR
        """
        video_path = self.input_path / "rgb.mp4"
        if not video_path.exists():
            logger.warning("rgb.mp4 not found, skipping RGB colour")
            return {}
        
        cap = cv2.VideoCapture(str(video_path))
        rgb_frames = {}
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                # Convert BGR to RGB
                rgb_frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_idx += 1
        
        cap.release()
        logger.debug(f"Loaded {len(rgb_frames)} RGB frames from video")
        return rgb_frames
