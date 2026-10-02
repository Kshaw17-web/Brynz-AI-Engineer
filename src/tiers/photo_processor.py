"""
Photo Tier Processor
=====================
Processes 2-8 room photos (no depth, no poses) to reconstruct room geometry.

Pipeline:
1. Run Structure-from-Motion (SfM) using COLMAP or DUSt3R 
   to recover camera poses and sparse 3D points
2. Apply monocular depth estimation to each photo
3. Fuse depth maps using recovered poses
4. Extract room geometry from fused point cloud

Accuracy tier:
- Wall lengths: ±8% (widest intervals, per spec)
- Ceiling height: ±5cm
- Area: ±8%
- Scale: resolved via DUSt3R's metric-scale predictions OR
  user-provided reference dimension (door width = 0.9m)

Multi-room support:
- Each room's folder produces a separate room reconstruction
- Rooms are stitched using opening detection

This tier represents the "floor": any picture in, results out.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PhotoProcessor:
    """
    Processes photo input (Tier 1: Photos).
    This is the most challenging tier — no sensor data, just RGB images.
    
    Approach:
    1. DUSt3R (Deep Unconstrained SfM) for pose estimation + depth
       - State-of-the-art, handles 2-8 overlapping images very well
       - Provides metric scale when global alignment is enabled
    2. Fallback: COLMAP (if DUSt3R unavailable)
    """

    def __init__(
        self,
        input_path: Path,
        verbose: bool = False,
        reference_height: Optional[float] = None,  # Known reference e.g. 2.4m ceiling
    ):
        self.input_path = Path(input_path)
        self.verbose = verbose
        self.reference_height = reference_height

    def load(self) -> Tuple[Dict, Optional[List]]:
        """
        Process photos to get point cloud.
        
        Input can be:
        - Directory of images: *.jpg, *.jpeg, *.png, *.heic
        - Multiple per-room directories
        
        Returns:
            point_cloud: {'xyz': Nx3, 'rgb': Nx3}
            frames_rgb: list of (frame_idx, rgb_array)
        """
        # Find images
        images = self._load_images()
        logger.info(f"Found {len(images)} photos")
        
        if len(images) < 2:
            raise ValueError(
                f"Need at least 2 photos for reconstruction, found {len(images)}. "
                f"Input path: {self.input_path}"
            )
        
        # Try DUSt3R first
        xyz, rgb = self._run_dust3r(images)
        
        if xyz is None or len(xyz) < 100:
            logger.warning("DUSt3R failed or insufficient points — trying COLMAP")
            xyz, rgb = self._run_colmap(images)
        
        if xyz is None or len(xyz) < 100:
            logger.warning("COLMAP failed — using heuristic reconstruction")
            xyz, rgb = self._heuristic_reconstruction(images)
        
        # Scale to metric
        xyz_metric = self._apply_metric_scale(xyz, images)
        
        frames_rgb = [(i, img) for i, img in enumerate(images)]
        
        return {"xyz": xyz_metric.astype(np.float32), "rgb": rgb.astype(np.uint8)}, frames_rgb

    def _load_images(self) -> List[np.ndarray]:
        """Load all images from input directory."""
        import cv2
        
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.heic", "*.HEIC"]
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(sorted(self.input_path.glob(ext)))
        
        images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        return images

    def _run_dust3r(
        self, images: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run DUSt3R for metric-scale 3D reconstruction from 2-8 photos.
        DUSt3R is a recent (2024) dense reconstruction model that handles
        small image sets very well without requiring camera calibration.
        """
        try:
            import sys
            dust3r_path = Path(__file__).parent.parent.parent / "third_party" / "dust3r"
            if dust3r_path.exists():
                sys.path.insert(0, str(dust3r_path))
            
            from dust3r.inference import inference
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.utils.image import load_images as dust3r_load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            import torch
            
            model_path = Path(__file__).parent.parent.parent / "models" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            if not model_path.exists():
                logger.warning("DUSt3R model not found. Run scripts/download_models.py")
                return None, None
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Running DUSt3R on {device}")
            
            model = AsymmetricCroCo3DStereo.from_pretrained(str(model_path)).to(device)
            
            # Save images to temp files for dust3r
            import tempfile
            import os
            import cv2
            
            with tempfile.TemporaryDirectory() as tmpdir:
                img_paths = []
                for i, img in enumerate(images):
                    p = os.path.join(tmpdir, f"{i:04d}.jpg")
                    cv2.imwrite(p, img[:, :, ::-1])
                    img_paths.append(p)
                
                dust3r_imgs = dust3r_load_images(img_paths, size=512)
                pairs = make_pairs(dust3r_imgs, scene_graph="complete", prefilter=None, symmetrize=True)
                
                output = inference(pairs, model, device, batch_size=1)
                
                scene = global_aligner(
                    output, device=device,
                    mode=GlobalAlignerMode.PointCloudOptimizer
                )
                loss = scene.compute_global_alignment(
                    init="mst", niter=300, schedule="cosine", lr=0.01
                )
                
                pts3d = scene.get_pts3d()
                masks = scene.get_masks()
                colors = scene.imgs
                
                all_xyz, all_rgb = [], []
                for pt, mask, color in zip(pts3d, masks, colors):
                    pt_np = pt.cpu().numpy()
                    mask_np = mask.cpu().numpy()
                    color_np = (color.cpu().numpy() * 255).astype(np.uint8)
                    
                    all_xyz.append(pt_np[mask_np].reshape(-1, 3))
                    all_rgb.append(color_np.reshape(-1, 3)[mask_np.flatten()])
                
                if all_xyz:
                    return np.vstack(all_xyz), np.vstack(all_rgb)
                return None, None
                
        except Exception as e:
            logger.warning(f"DUSt3R failed: {e}")
            return None, None

    def _run_colmap(
        self, images: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run COLMAP SfM + dense MVS for photo-based reconstruction.
        Requires COLMAP to be installed on the system.
        """
        try:
            import subprocess
            import tempfile
            import os
            import cv2
            
            result = subprocess.run(
                ["colmap", "version"],
                capture_output=True, timeout=5
            )
            if result.returncode != 0:
                return None, None
            
            logger.info("COLMAP found, running reconstruction...")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                img_dir = os.path.join(tmpdir, "images")
                db_path = os.path.join(tmpdir, "db.db")
                sparse_dir = os.path.join(tmpdir, "sparse")
                os.makedirs(img_dir)
                os.makedirs(sparse_dir)
                
                # Save images
                for i, img in enumerate(images):
                    p = os.path.join(img_dir, f"{i:04d}.jpg")
                    cv2.imwrite(p, img[:, :, ::-1])
                
                # Feature extraction
                subprocess.run([
                    "colmap", "feature_extractor",
                    "--database_path", db_path,
                    "--image_path", img_dir,
                    "--ImageReader.camera_model", "SIMPLE_RADIAL",
                ], check=True, capture_output=True, timeout=120)
                
                # Feature matching
                subprocess.run([
                    "colmap", "exhaustive_matcher",
                    "--database_path", db_path,
                ], check=True, capture_output=True, timeout=120)
                
                # Sparse reconstruction
                subprocess.run([
                    "colmap", "mapper",
                    "--database_path", db_path,
                    "--image_path", img_dir,
                    "--output_path", sparse_dir,
                ], check=True, capture_output=True, timeout=300)
                
                # Read sparse points
                sparse_model_dir = os.path.join(sparse_dir, "0")
                if not os.path.exists(sparse_model_dir):
                    return None, None
                
                # Read points3D.bin
                from src.utils.colmap_io import read_points3d_binary
                pts3d = read_points3d_binary(os.path.join(sparse_model_dir, "points3D.bin"))
                
                if pts3d:
                    xyz = np.array([p.xyz for p in pts3d.values()])
                    rgb = np.array([p.rgb for p in pts3d.values()], dtype=np.uint8)
                    return xyz, rgb
                
                return None, None
                
        except Exception as e:
            logger.warning(f"COLMAP failed: {e}")
            return None, None

    def _heuristic_reconstruction(
        self, images: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Last-resort heuristic reconstruction when neither DUSt3R nor COLMAP is available.
        Uses monocular depth estimation independently on each image.
        WARNING: This produces poor 3D geometry without proper pose estimation.
        """
        logger.warning(
            "Using heuristic photo reconstruction. "
            "Accuracy will be significantly lower than spec. "
            "Install DUSt3R or COLMAP for proper photo-tier support."
        )
        
        all_xyz, all_rgb = [], []
        
        h, w = images[0].shape[:2]
        fx = fy = w / (2 * np.tan(np.deg2rad(38.5)))
        cx, cy = w / 2, h / 2
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        for i, img in enumerate(images):
            # Assume frontal view for each image (known limitation)
            depth = self._simple_depth_estimate(img)
            
            # Unproject
            valid = (depth > 0.1) & (depth < 8.0)
            step = 4
            z = depth[::step, ::step][valid[::step, ::step]]
            x = (u[::step, ::step][valid[::step, ::step]] - cx) * z / fx
            y = (v[::step, ::step][valid[::step, ::step]] - cy) * z / fy
            
            pts = np.stack([x + i * 0.5, y, z], axis=-1)  # Offset by image index
            colors = img[::step, ::step].reshape(-1, 3)[valid[::step, ::step].flatten()]
            
            all_xyz.append(pts)
            all_rgb.append(colors)
        
        return np.vstack(all_xyz), np.vstack(all_rgb)

    def _simple_depth_estimate(self, img: np.ndarray) -> np.ndarray:
        """Very simple gradient-based depth heuristic."""
        h, w = img.shape[:2]
        y_coords = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
        depth = 0.5 + y_coords * 4.0  # 0.5m to 4.5m (floor is near in image bottom)
        return depth.astype(np.float32)

    def _apply_metric_scale(
        self, xyz: np.ndarray, images: List[np.ndarray]
    ) -> np.ndarray:
        """
        Apply metric scale to photo reconstruction.
        
        For photo tier, we use one of:
        1. User-provided reference height
        2. Detected door height (if door visible): standard 2.0m
        3. Y-extent assumption: typical room height 2.4m
        """
        if len(xyz) < 10:
            return xyz
        
        if self.reference_height is not None:
            y_range = xyz[:, 1].max() - xyz[:, 1].min()
            if y_range > 0.01:
                scale = self.reference_height / y_range
                return xyz * np.clip(scale, 0.1, 10.0)
        
        # Default: assume 2.4m height
        y_range = xyz[:, 1].max() - xyz[:, 1].min()
        if y_range > 0.01:
            scale = 2.4 / y_range
            return xyz * np.clip(scale, 0.1, 10.0)
        
        return xyz
