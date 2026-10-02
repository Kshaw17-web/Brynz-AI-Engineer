"""
Room Geometry Extractor
========================
Takes a 3D point cloud and extracts:
  - Floor plane (RANSAC)
  - Ceiling plane (RANSAC)
  - Wall planes (iterative RANSAC on residual points)
  - Room polygon (2D convex hull / alpha shape of floor projection)
  - Wall lengths with confidence intervals
  - Ceiling height with confidence interval
  - Opening detection (doors, windows) via gap analysis in wall point density

Key design choices:
  - RANSAC plane detection is robust to clutter and furniture
  - We project to floor plane to get the 2D footprint
  - Wall openings are detected by scanning for vertical gaps in point density
  - Confidence intervals are computed from point density and measurement repeatability
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
from scipy.spatial import ConvexHull
from scipy import stats

logger = logging.getLogger(__name__)

# Try to import open3d; fall back to pure numpy implementations
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("open3d not found — using numpy-only RANSAC (slower)")


class GeometryExtractor:
    """
    Extracts room geometry from a 3D point cloud.
    
    Coordinate system: ARKit world frame (Y-up, right-hand).
    - Y axis: vertical (up)
    - XZ plane: horizontal
    
    We detect:
      1. Floor plane: lowest RANSAC plane (normal ~ [0,1,0])
      2. Ceiling plane: highest RANSAC plane (normal ~ [0,-1,0])  
      3. Wall planes: vertical RANSAC planes (normal ⊥ Y axis)
      4. Room footprint: 2D polygon in XZ plane
      5. Openings: gaps in wall point density
    """

    # RANSAC parameters
    RANSAC_DISTANCE_THRESH = 0.05  # 5 cm inlier threshold
    RANSAC_N_ITERS = 1000
    MIN_PLANE_POINTS = 500

    # Opening detection
    MIN_OPENING_WIDTH_M = 0.5   # Minimum door/window width
    MAX_OPENING_WIDTH_M = 3.0   # Maximum opening (wider = likely a missing wall)
    MIN_OPENING_HEIGHT_M = 1.0  # Minimum opening height (window sill + height)

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def extract(self, point_cloud: Dict) -> Dict[str, Any]:
        """
        Main extraction routine.
        
        Returns:
            geometry dict with all measurements and confidence intervals
        """
        xyz = point_cloud["xyz"]
        
        logger.info(f"Extracting geometry from {len(xyz)} points")

        # Step 1: Voxel downsample for speed
        xyz_ds = self._voxel_downsample(xyz, voxel_size=0.02)  # 2cm voxels
        logger.info(f"Downsampled to {len(xyz_ds)} points")

        # Step 2: Remove outliers
        xyz_clean = self._remove_outliers(xyz_ds)
        logger.info(f"After outlier removal: {len(xyz_clean)} points")

        # Step 3: Detect floor plane
        floor_result = self._detect_horizontal_plane(xyz_clean, find_floor=True)
        floor_z = floor_result["height"]  # Y coordinate (up = +Y in ARKit)
        floor_normal = floor_result["normal"]
        
        # Step 4: Detect ceiling plane
        ceiling_result = self._detect_horizontal_plane(xyz_clean, find_floor=False)
        ceiling_z = ceiling_result["height"]
        
        # Step 5: Ceiling height
        ceiling_height_m = abs(ceiling_z - floor_z)
        
        # Step 6: Extract points in the "room layer" (between floor+5cm and ceiling-5cm)
        room_mask = (xyz_clean[:, 1] > floor_z + 0.05) & (xyz_clean[:, 1] < ceiling_z - 0.05)
        room_pts = xyz_clean[room_mask]
        
        # Step 7: Project to XZ plane (floor view) and get footprint
        xz = room_pts[:, [0, 2]]
        floor_pts_xz = xyz_clean[(xyz_clean[:, 1] < floor_z + 0.10)][:, [0, 2]]
        
        # Step 8: Detect walls
        walls = self._detect_walls(xyz_clean, floor_z, ceiling_z)
        
        # Step 9: Compute floor area and polygon
        footprint = self._compute_footprint(floor_pts_xz if len(floor_pts_xz) > 10 else xz)
        floor_area_m2 = footprint["area_m2"]
        
        # Step 10: Detect openings in walls
        openings = self._detect_openings(xyz_clean, walls, floor_z, ceiling_z)
        
        # Step 11: Compute confidence intervals
        ci = self._compute_confidence_intervals(
            xyz_clean, walls, floor_z, ceiling_z, footprint
        )

        result = {
            # Core measurements
            "ceiling_height_m": float(ceiling_height_m),
            "ceiling_height_ci_m": float(ci["ceiling_height_ci"]),
            "floor_area_m2": float(floor_area_m2),
            "floor_area_ci_m2": float(ci["floor_area_ci"]),
            "floor_z": float(floor_z),
            "ceiling_z": float(ceiling_z),
            
            # Walls
            "walls": walls,
            
            # Openings
            "openings": openings,
            
            # 2D footprint polygon (list of [x, z] points)
            "footprint_polygon": footprint["polygon"].tolist(),
            
            # Plane normals for reference
            "floor_normal": floor_normal.tolist(),
            
            # Stats
            "n_points_total": len(xyz),
            "n_points_used": len(xyz_clean),
        }
        return result

    def _voxel_downsample(self, xyz: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Voxel grid downsampling: keep one point per voxel (centroid).
        Uses numpy for speed; open3d if available.
        """
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_ds = pcd.voxel_down_sample(voxel_size)
            return np.asarray(pcd_ds.points, dtype=np.float32)
        
        # Numpy fallback
        indices = np.floor(xyz / voxel_size).astype(np.int32)
        _, unique_idx = np.unique(indices, axis=0, return_index=True)
        return xyz[unique_idx]

    def _remove_outliers(self, xyz: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        """
        Statistical outlier removal.
        Removes points that are further than std_ratio standard deviations
        from the mean distance to their nb_neighbors neighbours.
        """
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
            return np.asarray(pcd_clean.points, dtype=np.float32)
        
        # Simple bounding box clip as fallback
        q1 = np.percentile(xyz, 1, axis=0)
        q99 = np.percentile(xyz, 99, axis=0)
        mask = np.all((xyz >= q1) & (xyz <= q99), axis=1)
        return xyz[mask]

    def _detect_horizontal_plane(
        self, xyz: np.ndarray, find_floor: bool = True
    ) -> Dict:
        """
        Detect the dominant horizontal plane (floor or ceiling) using RANSAC.
        
        ARKit Y-axis is up, so:
        - Floor = lowest Y-valued dominant plane
        - Ceiling = highest Y-valued dominant plane
        
        Returns: {"height": float, "normal": np.ndarray, "n_inliers": int}
        """
        # Work in bins of Y values to find candidate heights
        y_vals = xyz[:, 1]
        
        if find_floor:
            # Floor: look in bottom 20% of point cloud height
            y_thresh = np.percentile(y_vals, 20)
            candidates = xyz[y_vals <= y_thresh]
        else:
            # Ceiling: look in top 20%
            y_thresh = np.percentile(y_vals, 80)
            candidates = xyz[y_vals >= y_thresh]
        
        if len(candidates) < self.MIN_PLANE_POINTS:
            # Fall back to using percentile directly
            height = np.percentile(y_vals, 2 if find_floor else 98)
            return {
                "height": float(height),
                "normal": np.array([0.0, 1.0, 0.0]),
                "n_inliers": 0,
            }
        
        best_height, best_normal, best_inliers = self._ransac_horizontal_plane(
            candidates, find_floor=find_floor
        )
        
        return {
            "height": float(best_height),
            "normal": best_normal,
            "n_inliers": best_inliers,
        }

    def _ransac_horizontal_plane(
        self, xyz: np.ndarray, find_floor: bool = True
    ) -> Tuple[float, np.ndarray, int]:
        """RANSAC to find horizontal plane in the point cloud."""
        best_inliers = 0
        best_height = 0.0
        
        n = len(xyz)
        rng = np.random.default_rng(42)
        
        for _ in range(min(self.RANSAC_N_ITERS, 200)):
            # Sample 3 random points
            idx = rng.choice(n, 3, replace=False)
            sample = xyz[idx]
            
            # Fit a horizontal plane (we constrain to horizontal by just using Y=c)
            h = np.mean(sample[:, 1])
            
            # Count inliers
            dist = np.abs(xyz[:, 1] - h)
            inliers = np.sum(dist < self.RANSAC_DISTANCE_THRESH)
            
            if inliers > best_inliers:
                best_inliers = inliers
                # Refine height with inlier mean
                inlier_pts = xyz[dist < self.RANSAC_DISTANCE_THRESH]
                best_height = float(np.median(inlier_pts[:, 1]))
        
        return best_height, np.array([0.0, 1.0, 0.0]), best_inliers

    def _detect_walls(
        self, xyz: np.ndarray, floor_z: float, ceiling_z: float
    ) -> List[Dict]:
        """
        Detect wall planes using iterative RANSAC on vertical planes.
        
        Wall plane normal is approximately horizontal (perpendicular to Y axis).
        We extract up to 8 walls (most rooms have 4-6).
        
        Each wall returned as:
        {
            "id": "wall_N",
            "length_m": float,
            "length_ci_m": float,
            "start_xz": [float, float],
            "end_xz": [float, float],
            "normal_xz": [float, float],
            "height_coverage": float  # fraction of wall height with points
        }
        """
        # Only use points in the wall zone (above floor, below ceiling)
        mask = (xyz[:, 1] > floor_z + 0.1) & (xyz[:, 1] < ceiling_z - 0.1)
        wall_pts = xyz[mask]
        
        if len(wall_pts) < self.MIN_PLANE_POINTS:
            return []
        
        walls = []
        remaining = wall_pts.copy()
        wall_id = 0
        
        for _ in range(12):  # max 12 wall planes
            if len(remaining) < self.MIN_PLANE_POINTS:
                break
            
            wall, inlier_mask = self._ransac_vertical_plane(remaining)
            if wall is None:
                break
            
            # Remove inliers from remaining
            remaining = remaining[~inlier_mask]
            
            # Compute wall extent (length)
            inlier_pts = wall_pts[~np.ones(len(wall_pts), dtype=bool)]  # all False
            # Re-find inliers in original wall_pts for measurement
            normal_xz = np.array([wall["a"], wall["c"]])
            normal_xz /= np.linalg.norm(normal_xz)
            
            dist = np.abs(
                wall_pts[:, 0] * wall["a"] + 
                wall_pts[:, 2] * wall["c"] + 
                wall["d"]
            )
            inliers = wall_pts[dist < self.RANSAC_DISTANCE_THRESH]
            
            if len(inliers) < 100:
                continue
            
            # Project inliers onto wall plane to find extent
            # Tangent direction = perpendicular to normal in XZ
            tangent = np.array([-normal_xz[1], normal_xz[0]])
            proj = inliers[:, [0, 2]] @ tangent
            
            p_min, p_max = proj.min(), proj.max()
            length_m = float(p_max - p_min)
            
            if length_m < 0.3:  # Skip tiny wall fragments
                continue
            
            # Find start/end points
            start_pt = proj.min() * tangent + normal_xz * (-wall["d"] / (wall["a"]**2 + wall["c"]**2)**0.5)
            end_pt = proj.max() * tangent + normal_xz * (-wall["d"] / (wall["a"]**2 + wall["c"]**2)**0.5)
            
            # Confidence interval: based on point density spread
            length_ci = float(np.std(proj) / np.sqrt(len(proj)) * 1.96 + 0.01)
            
            # Height coverage
            h_range = inliers[:, 1].max() - inliers[:, 1].min()
            h_coverage = min(1.0, h_range / max(0.1, ceiling_z - floor_z))
            
            walls.append({
                "id": f"wall_{wall_id:02d}",
                "length_m": round(length_m, 4),
                "length_ci_m": round(length_ci, 4),
                "start_xz": start_pt.tolist(),
                "end_xz": end_pt.tolist(),
                "normal_xz": normal_xz.tolist(),
                "n_inliers": int(len(inliers)),
                "height_coverage": round(float(h_coverage), 3),
            })
            wall_id += 1
        
        logger.info(f"Detected {len(walls)} walls")
        return walls

    def _ransac_vertical_plane(
        self, xyz: np.ndarray
    ) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """
        RANSAC to find a vertical plane (ax + cz + d = 0, no y term).
        Returns (plane_params_dict, inlier_bool_mask).
        """
        n = len(xyz)
        best_inliers = 0
        best_plane = None
        best_mask = None
        
        rng = np.random.default_rng()
        
        for _ in range(200):
            # Sample 2 points (defines a vertical plane normal direction)
            idx = rng.choice(n, 2, replace=False)
            p1, p2 = xyz[idx[0], [0, 2]], xyz[idx[1], [0, 2]]
            
            # Normal in XZ plane (perpendicular to the line p1->p2)
            diff = p2 - p1
            if np.linalg.norm(diff) < 1e-6:
                continue
            normal = np.array([-diff[1], diff[0]])  # rotate 90°
            normal /= np.linalg.norm(normal)
            
            # Plane: normal[0]*x + normal[1]*z = d
            d = -(normal[0] * p1[0] + normal[1] * p1[1])
            
            # Distance of all points
            dist = np.abs(xyz[:, 0] * normal[0] + xyz[:, 2] * normal[1] + d)
            mask = dist < self.RANSAC_DISTANCE_THRESH
            n_inliers = mask.sum()
            
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                # Refine with all inliers
                inlier_xz = xyz[mask][:, [0, 2]]
                # Least-squares fit
                A = np.column_stack([inlier_xz[:, 0], inlier_xz[:, 1]])
                b = -np.ones(len(inlier_xz))
                # Solve Ax = b => [a, c] where a*x + c*z = -1 => ax + cz + 1 = 0
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    a, c = coeffs
                    norm_len = np.sqrt(a**2 + c**2)
                    best_plane = {"a": a/norm_len, "c": c/norm_len, "d": 1.0/norm_len}
                    # Recompute mask with refined plane
                    dist2 = np.abs(xyz[:, 0]*a + xyz[:, 2]*c + 1.0) / norm_len
                    best_mask = dist2 < self.RANSAC_DISTANCE_THRESH
                except Exception:
                    best_plane = {"a": normal[0], "c": normal[1], "d": d}
                    best_mask = mask
        
        if best_inliers < self.MIN_PLANE_POINTS // 2:
            return None, None
        
        return best_plane, best_mask

    def _compute_footprint(self, xz: np.ndarray) -> Dict:
        """
        Compute 2D floor footprint polygon and area.
        Uses convex hull; for more complex rooms, alpha shape would be better
        (future: use shapely.alpha_shape).
        """
        if len(xz) < 3:
            return {"area_m2": 0.0, "polygon": np.array([[0,0],[1,0],[1,1],[0,1]])}
        
        try:
            hull = ConvexHull(xz)
            area_m2 = float(hull.volume)  # ConvexHull.volume = area for 2D
            polygon = xz[hull.vertices]
        except Exception as e:
            logger.warning(f"ConvexHull failed: {e}")
            area_m2 = 0.0
            polygon = xz[:4] if len(xz) >= 4 else xz
        
        return {"area_m2": area_m2, "polygon": polygon}

    def _detect_openings(
        self, xyz: np.ndarray, walls: List[Dict], 
        floor_z: float, ceiling_z: float
    ) -> List[Dict]:
        """
        Detect openings (doors and windows) by scanning each wall for
        vertical gaps in point density.
        
        Algorithm:
          For each wall:
            1. Find all points within 10cm of the wall plane
            2. Project to (horizontal_along_wall, vertical) 2D grid
            3. Look for vertical "gaps" (columns with low point density)
            4. A gap that spans floor-to-some-height → door
            5. A gap that spans mid-height → window
        """
        openings = []
        opening_id = 0
        
        for wall in walls:
            normal = np.array(wall["normal_xz"])
            d = -(normal[0] * wall["start_xz"][0] + normal[1] * wall["start_xz"][1])
            
            # Points near this wall
            dist = np.abs(xyz[:, 0] * normal[0] + xyz[:, 2] * normal[1] + d)
            near_mask = (dist < 0.15) & (xyz[:, 1] > floor_z) & (xyz[:, 1] < ceiling_z)
            near_pts = xyz[near_mask]
            
            if len(near_pts) < 50:
                continue
            
            # Project to (along-wall, vertical)
            tangent = np.array([-normal[1], normal[0]])
            along = near_pts[:, [0, 2]] @ tangent
            vert = near_pts[:, 1]
            
            # Create a 2D density histogram
            n_bins_h = 50
            n_bins_v = 20
            
            h_range = (along.min(), along.max())
            v_range = (floor_z, ceiling_z)
            
            if h_range[1] - h_range[0] < 0.3:
                continue
            
            H, xedges, yedges = np.histogram2d(
                along, vert,
                bins=[n_bins_h, n_bins_v],
                range=[h_range, v_range]
            )
            
            # Find columns (horizontal positions) with low density
            col_density = H.sum(axis=1)
            max_density = col_density.max()
            
            if max_density == 0:
                continue
            
            # A "gap" = column density < 10% of max
            gap_threshold = max_density * 0.10
            is_gap = col_density < gap_threshold
            
            # Find contiguous gap regions
            gap_groups = self._find_contiguous_groups(is_gap)
            
            bin_width = (h_range[1] - h_range[0]) / n_bins_h
            
            for start_bin, end_bin in gap_groups:
                width_m = (end_bin - start_bin + 1) * bin_width
                
                if not (self.MIN_OPENING_WIDTH_M <= width_m <= self.MAX_OPENING_WIDTH_M):
                    continue
                
                # Determine opening type from vertical extent
                gap_col = H[start_bin:end_bin+1, :].sum(axis=0)
                gap_is_empty = gap_col < gap_threshold
                
                # Find vertical extent of opening
                filled_v = ~gap_is_empty
                if filled_v.any():
                    # Height range that IS filled in the gap (inverted logic)
                    # Empty rows in the gap = the opening
                    empty_rows = np.where(gap_is_empty)[0]
                    if len(empty_rows) == 0:
                        continue
                    v_min_bin = empty_rows[0]
                    v_max_bin = empty_rows[-1]
                    v_min = float(v_range[0] + v_min_bin * (v_range[1] - v_range[0]) / n_bins_v)
                    v_max = float(v_range[0] + (v_max_bin + 1) * (v_range[1] - v_range[0]) / n_bins_v)
                else:
                    v_min = floor_z
                    v_max = ceiling_z
                
                opening_height = v_max - v_min
                if opening_height < self.MIN_OPENING_HEIGHT_M:
                    continue
                
                # Classify: door starts near floor, window starts higher
                opens_at_floor = (v_min - floor_z) < 0.2
                opening_type = "door" if opens_at_floor else "window"
                
                # Position along wall
                along_start = h_range[0] + start_bin * bin_width
                along_end = h_range[0] + (end_bin + 1) * bin_width
                
                # Width CI: half a bin width
                width_ci = bin_width / 2
                
                openings.append({
                    "id": f"opening_{opening_id:02d}",
                    "type": opening_type,
                    "wall_id": wall["id"],
                    "width_m": round(width_m, 3),
                    "width_ci_m": round(width_ci, 3),
                    "height_m": round(opening_height, 3),
                    "sill_height_m": round(max(0.0, v_min - floor_z), 3),
                    "along_wall_start_m": round(float(along_start), 3),
                    "along_wall_end_m": round(float(along_end), 3),
                })
                opening_id += 1
        
        logger.info(f"Detected {len(openings)} openings")
        return openings

    def _find_contiguous_groups(self, bool_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find start/end indices of contiguous True regions in a boolean mask."""
        groups = []
        in_group = False
        start = 0
        for i, val in enumerate(bool_mask):
            if val and not in_group:
                start = i
                in_group = True
            elif not val and in_group:
                groups.append((start, i - 1))
                in_group = False
        if in_group:
            groups.append((start, len(bool_mask) - 1))
        return groups

    def _compute_confidence_intervals(
        self, xyz: np.ndarray, walls: List[Dict],
        floor_z: float, ceiling_z: float, footprint: Dict
    ) -> Dict:
        """
        Compute 95% confidence intervals for each measurement.
        
        Methodology:
        - Bootstrap resampling on the point cloud
        - For ceiling height: std of Y values of ceiling inliers
        - For floor area: uncertainty from convex hull with perturbed points
        - For wall lengths: computed in wall detection above
        
        We use simplified closed-form estimates here:
        - Ceiling height CI: based on point spread near ceiling plane
        - Floor area CI: 2% of area (typical LiDAR accuracy)
        """
        # Ceiling height CI
        near_ceiling = xyz[xyz[:, 1] > ceiling_z - 0.1, 1]
        if len(near_ceiling) > 10:
            ceiling_std = np.std(near_ceiling)
            ceiling_ci = float(1.96 * ceiling_std / np.sqrt(len(near_ceiling)) + 0.005)
        else:
            ceiling_ci = 0.015  # Default 1.5cm if not enough points
        
        # Floor area CI (2% of area for LiDAR, wider for other tiers)
        area_ci = footprint["area_m2"] * 0.02
        
        return {
            "ceiling_height_ci": max(0.005, ceiling_ci),
            "floor_area_ci": max(0.05, area_ci),
        }
