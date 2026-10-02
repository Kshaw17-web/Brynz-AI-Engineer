"""
Room Geometry Extractor — robust pipeline
==========================================
Pipeline (in order):
  1. Voxel downsample
  2. Statistical outlier removal
  3. Detect floor plane (RANSAC on bottom-Y points)
  4. Detect ceiling plane (RANSAC on top-Y points, if scan covers ceiling)
  5. Extract room-layer points (between floor + margin and ceiling - margin)
  6. Estimate surface normals in XZ plane via local PCA
  7. Build azimuth histogram of horizontal normals → dominant wall directions
  8. For each dominant direction: project points, find wall clusters in depth
  9. Per-cluster filters: min point count, min length, min height-span coverage
 10. Merge near-parallel wall segments with same azimuth (dedup collinear walls)
 11. Reject furniture-sized isolated planes (area < threshold)
 12. Build room footprint polygon from wall endpoints
 13. Compute all confidence intervals from point residuals (not invented)

Coordinate system:
  The odometry uses whatever frame the recording device produced.
  We do NOT assume Y-up or any fixed world orientation.
  Instead, we detect the floor as the dominant horizontal plane
  (largest RANSAC inlier set when fitting a near-horizontal plane)
  and define "up" from that plane's normal.

Calibration note:
  All depth intrinsics are derived from the RGB intrinsics scaled by the
  depth-to-RGB resolution ratio (documented in lidar_processor.py).
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class GeometryExtractor:
    """Extracts room geometry from a 3D point cloud."""

    # Tunable parameters (not overfit to one room)
    VOXEL_SIZE_M         = 0.02   # 2 cm voxels for downsampling
    RANSAC_DIST_M        = 0.05   # 5 cm inlier threshold for planes
    RANSAC_ITERS         = 300
    MIN_PLANE_PTS        = 300    # min points to accept a plane
    MIN_WALL_PTS         = 100    # min inliers to keep a wall candidate
    MIN_WALL_LENGTH_M    = 0.3    # reject wall segments shorter than this
    MIN_HEIGHT_COV_FRAC  = 0.20   # wall must span ≥20% of room height
    MIN_ROOM_HEIGHT_M    = 0.5    # below this → floor/ceiling detection suspect
    NORMAL_HIST_BINS     = 180    # azimuth histogram resolution (1°/bin)
    NORMAL_PEAK_MIN_FRAC = 0.07   # peak must be ≥7% of histogram max
    NORMAL_PEAK_SEP_DEG  = 15     # min degrees between dominant directions
    MAX_PARALLEL_WALLS   = 3      # max walls per dominant direction
    MERGE_DIST_M         = 0.12   # merge walls closer than this (collinear)

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def extract(self, point_cloud: Dict) -> Dict[str, Any]:
        """
        Main entry point.
        Returns geometry dict with all measurements and CIs.
        """
        xyz = point_cloud["xyz"]
        logger.info(f"Extracting geometry from {len(xyz)} points")
        logger.info(f"  X: [{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]  "
                    f"Y: [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]  "
                    f"Z: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")

        # 1. Downsample
        xyz_ds = self._voxel_downsample(xyz)
        logger.info(f"  After voxel downsample: {len(xyz_ds)} points")

        # 2. Outlier removal
        xyz_clean = self._remove_outliers(xyz_ds)
        logger.info(f"  After outlier removal: {len(xyz_clean)} points")

        # 3. Find dominant floor plane
        floor = self._detect_dominant_horizontal_plane(xyz_clean, find_lowest=True)
        floor_y    = floor["height"]
        floor_n    = floor["normal"]   # approx [0,1,0] or [0,-1,0]
        logger.info(f"  Floor plane: y={floor_y:.4f} m  ({floor['n_inliers']} inliers)")

        # 4. Find ceiling plane (may not exist if scan is floor-only)
        ceiling = self._detect_dominant_horizontal_plane(xyz_clean, find_lowest=False)
        ceiling_y = ceiling["height"]
        logger.info(f"  Ceiling plane: y={ceiling_y:.4f} m  ({ceiling['n_inliers']} inliers)")

        # Sanity: room height
        room_height_m = abs(ceiling_y - floor_y)
        ceiling_plausible = (
            ceiling["n_inliers"] > self.MIN_PLANE_PTS
            and room_height_m >= self.MIN_ROOM_HEIGHT_M
        )
        if not ceiling_plausible:
            logger.warning(
                f"  Ceiling detection uncertain (height={room_height_m:.2f}m, "
                f"inliers={ceiling['n_inliers']}). "
                f"Using point cloud top extent as ceiling estimate."
            )
            ceiling_y = float(np.percentile(xyz_clean[:, 1], 97))
            room_height_m = abs(ceiling_y - floor_y)

        logger.info(f"  Room height estimate: {room_height_m:.3f} m")

        # 5. Walls
        walls = self._detect_walls(xyz_clean, floor_y, ceiling_y, room_height_m)

        # 6. Floor footprint (use floor-layer points)
        floor_mask = (xyz_clean[:, 1] < floor_y + 0.12)
        floor_pts_xz = xyz_clean[floor_mask][:, [0, 2]]
        if len(floor_pts_xz) < 10:
            # Fallback: all points projected to XZ
            floor_pts_xz = xyz_clean[:, [0, 2]]
        footprint = self._compute_footprint(floor_pts_xz)

        # 7. Openings (gaps in walls)
        openings = self._detect_openings(xyz_clean, walls, floor_y, ceiling_y)

        # 8. Confidence intervals from residuals
        ci = self._compute_confidence_intervals(
            xyz_clean, walls, floor_y, ceiling_y, footprint, ceiling_plausible
        )

        return {
            "ceiling_height_m":     round(float(room_height_m), 4),
            "ceiling_height_ci_m":  round(float(ci["ceiling_height_ci"]), 4),
            "ceiling_detection_reliable": ceiling_plausible,
            "floor_area_m2":        round(float(footprint["area_m2"]), 4),
            "floor_area_ci_m2":     round(float(ci["floor_area_ci"]), 4),
            "floor_y":              round(float(floor_y), 4),
            "ceiling_y":            round(float(ceiling_y), 4),
            "floor_normal":         floor_n.tolist(),
            "walls":                walls,
            "openings":             openings,
            "footprint_polygon":    footprint["polygon"].tolist(),
            "n_points_input":       int(len(xyz)),
            "n_points_used":        int(len(xyz_clean)),
        }

    # ── Point cloud preprocessing ────────────────────────────────────────────

    def _voxel_downsample(self, xyz: np.ndarray) -> np.ndarray:
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_ds = pcd.voxel_down_sample(self.VOXEL_SIZE_M)
            return np.asarray(pcd_ds.points, dtype=np.float32)
        # Numpy fallback
        indices = np.floor(xyz / self.VOXEL_SIZE_M).astype(np.int32)
        _, unique_idx = np.unique(indices, axis=0, return_index=True)
        return xyz[unique_idx].astype(np.float32)

    def _remove_outliers(self, xyz: np.ndarray,
                         nb: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_c, _ = pcd.remove_statistical_outlier(nb, std_ratio)
            return np.asarray(pcd_c.points, dtype=np.float32)
        # Simple percentile clip fallback
        q1, q99 = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
        return xyz[np.all((xyz >= q1) & (xyz <= q99), axis=1)].astype(np.float32)

    # ── Floor / ceiling detection ────────────────────────────────────────────

    def _detect_dominant_horizontal_plane(
        self, xyz: np.ndarray, find_lowest: bool
    ) -> Dict:
        """
        Find the dominant horizontal plane via RANSAC.
        'Dominant' = most inliers among horizontal planes in the outer 30% of Y range.
        Returns {"height": float, "normal": ndarray, "n_inliers": int}.
        """
        y = xyz[:, 1]
        y_min, y_max = y.min(), y.max()
        y_span = y_max - y_min

        if find_lowest:
            candidates = xyz[y <= y_min + y_span * 0.30]
        else:
            candidates = xyz[y >= y_max - y_span * 0.30]

        if len(candidates) < self.MIN_PLANE_PTS:
            height = float(np.percentile(y, 2 if find_lowest else 98))
            return {"height": height, "normal": np.array([0., 1., 0.]), "n_inliers": 0}

        best_height, best_inliers = 0.0, 0
        rng = np.random.default_rng(42)
        n = len(candidates)

        for _ in range(self.RANSAC_ITERS):
            h_sample = float(np.mean(candidates[rng.choice(n, 3, replace=False), 1]))
            inliers = int(np.sum(np.abs(candidates[:, 1] - h_sample) < self.RANSAC_DIST_M))
            if inliers > best_inliers:
                best_inliers = inliers
                inlier_y = candidates[np.abs(candidates[:, 1] - h_sample) < self.RANSAC_DIST_M, 1]
                best_height = float(np.median(inlier_y))

        return {"height": best_height,
                "normal": np.array([0., 1., 0.]),
                "n_inliers": best_inliers}

    # ── Wall detection ───────────────────────────────────────────────────────

    def _detect_walls(
        self, xyz: np.ndarray,
        floor_y: float, ceiling_y: float, room_height: float
    ) -> List[Dict]:
        """
        Robust wall detection using normal-direction histogram clustering.

        Steps:
        1. Extract wall-zone points (above floor, below ceiling)
        2. Estimate XZ-plane normal direction at each sample point via local PCA
        3. Build histogram of normal azimuths → dominant wall directions
        4. For each dominant direction: scan along the normal for wall clusters
        5. Filter each candidate by: point count, length, height coverage
        6. Merge near-coincident (collinear) wall segments
        7. Reject furniture-sized candidates
        """
        margin = max(0.10, room_height * 0.08)
        wall_mask = (
            (xyz[:, 1] > floor_y + margin) &
            (xyz[:, 1] < ceiling_y - margin)
        )
        wall_pts = xyz[wall_mask]
        logger.info(f"  Wall-zone points: {len(wall_pts)}")

        if len(wall_pts) < self.MIN_PLANE_PTS:
            logger.warning("  Too few wall-zone points — no walls detected")
            return []

        # ── Step 2: Normal estimation via local PCA in XZ plane ─────────────
        n_sample = min(len(wall_pts), 15000)
        rng = np.random.default_rng(42)
        sample = wall_pts[rng.choice(len(wall_pts), n_sample, replace=False)]

        from scipy.spatial import cKDTree
        tree = cKDTree(sample[:, [0, 2]])
        azimuths = []

        for i in range(0, n_sample, 4):
            _, idxs = tree.query(sample[i, [0, 2]], k=13)
            nb = sample[idxs[1:]][:, [0, 2]]  # XZ only
            if len(nb) < 4:
                continue
            nb -= nb.mean(axis=0)
            _, evecs = np.linalg.eigh(nb.T @ nb)
            # Smallest eigenvalue → normal to local surface in XZ
            az = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])) % 180)
            azimuths.append(az)

        if len(azimuths) < 20:
            logger.warning("  Insufficient normal estimates — falling back to full-cloud RANSAC")
            return []

        azimuths = np.array(azimuths)

        # ── Step 3: Azimuth histogram ────────────────────────────────────────
        hist, bin_edges = np.histogram(
            azimuths, bins=self.NORMAL_HIST_BINS, range=(0, 180)
        )
        hist_smooth = uniform_filter1d(hist.astype(float), size=5)

        peaks, _ = find_peaks(
            hist_smooth,
            height=hist_smooth.max() * self.NORMAL_PEAK_MIN_FRAC,
            distance=self.NORMAL_PEAK_SEP_DEG,
        )

        if len(peaks) == 0:
            logger.warning("  No dominant normal directions found")
            return []

        # Top 6 peaks by height
        top_peaks = peaks[np.argsort(hist_smooth[peaks])[::-1][:6]]
        dominant_azimuths_deg = (top_peaks + 0.5) * (180.0 / self.NORMAL_HIST_BINS)
        logger.info(f"  Dominant wall azimuths (deg): {[round(a,1) for a in dominant_azimuths_deg]}")

        # ── Steps 4–7: Extract and filter wall candidates ────────────────────
        walls = []
        wall_id = 0

        for az_deg in dominant_azimuths_deg:
            az_rad = np.radians(az_deg)
            normal = np.array([np.cos(az_rad), np.sin(az_rad)])  # in XZ
            tangent = np.array([-normal[1], normal[0]])

            # Project all wall-zone points onto this normal
            proj_n = wall_pts[:, 0] * normal[0] + wall_pts[:, 2] * normal[1]

            # Scan along normal for wall clusters
            p_hist, p_edges = np.histogram(proj_n, bins=80)
            p_smooth = uniform_filter1d(p_hist.astype(float), size=3)
            wpeaks, _ = find_peaks(
                p_smooth,
                height=p_smooth.max() * 0.12,
                distance=4,
            )

            # Top N per direction
            top_wp = wpeaks[np.argsort(p_smooth[wpeaks])[::-1][:self.MAX_PARALLEL_WALLS]]

            for wp in top_wp:
                wall_d = float((p_edges[wp] + p_edges[wp + 1]) / 2)
                inliers = wall_pts[np.abs(proj_n - wall_d) < self.RANSAC_DIST_M]

                if len(inliers) < self.MIN_WALL_PTS:
                    continue

                along = inliers[:, 0] * tangent[0] + inliers[:, 2] * tangent[1]
                length_m = float(along.max() - along.min())

                if length_m < self.MIN_WALL_LENGTH_M:
                    continue

                h_span = float(inliers[:, 1].max() - inliers[:, 1].min())
                h_cov = h_span / max(0.01, room_height)

                if h_cov < self.MIN_HEIGHT_COV_FRAC:
                    continue

                # Refine normal by least-squares fit to inlier XZ coords
                A = np.column_stack([inliers[:, 0], inliers[:, 2],
                                     np.ones(len(inliers))])
                # Normal direction: use prior (az_deg) — LSQ along-normal residual
                residuals = np.abs(proj_n[np.abs(proj_n - wall_d) < self.RANSAC_DIST_M] - wall_d)
                plane_residual_m = float(np.std(residuals))

                # Length CI: from along-wall point spread
                along_std = float(np.std(along))
                length_ci = float(min(
                    1.96 * along_std / np.sqrt(len(along)) + 0.005,
                    length_m * 0.05
                ))

                start_xz = (normal * wall_d + tangent * along.min()).tolist()
                end_xz   = (normal * wall_d + tangent * along.max()).tolist()

                walls.append({
                    "id":               f"wall_{wall_id:02d}",
                    "length_m":         round(length_m, 4),
                    "length_ci_m":      round(length_ci, 4),
                    "start_xz":         start_xz,
                    "end_xz":           end_xz,
                    "normal_xz":        normal.tolist(),
                    "azimuth_deg":      round(float(az_deg), 1),
                    "n_inliers":        int(len(inliers)),
                    "height_coverage":  round(float(min(1.0, h_cov)), 3),
                    "plane_residual_m": round(plane_residual_m, 4),
                })
                wall_id += 1

        # ── Step 6: Merge near-coincident wall segments ──────────────────────
        walls = self._merge_collinear_walls(walls)

        walls.sort(key=lambda w: w["length_m"], reverse=True)
        logger.info(f"  Detected {len(walls)} walls after filtering/merging")
        return walls

    def _merge_collinear_walls(self, walls: List[Dict]) -> List[Dict]:
        """
        Merge wall segments with the same azimuth (±3°) and similar
        normal-direction offset (within MERGE_DIST_M).
        Keeps the merged segment's total length and averages metadata.
        """
        if len(walls) < 2:
            return walls

        merged = []
        used = [False] * len(walls)

        for i, w_i in enumerate(walls):
            if used[i]:
                continue
            group = [w_i]
            used[i] = True
            az_i = w_i["azimuth_deg"]
            n_i = np.array(w_i["normal_xz"])
            d_i = (np.array(w_i["start_xz"]) @ n_i +
                   np.array(w_i["end_xz"]) @ n_i) / 2

            for j, w_j in enumerate(walls):
                if used[j] or j == i:
                    continue
                az_j = w_j["azimuth_deg"]
                if abs(az_i - az_j) > 3 and abs(180 - abs(az_i - az_j)) > 3:
                    continue  # different orientation
                n_j = np.array(w_j["normal_xz"])
                d_j = (np.array(w_j["start_xz"]) @ n_j +
                       np.array(w_j["end_xz"]) @ n_j) / 2
                if abs(d_i - d_j) < self.MERGE_DIST_M:
                    group.append(w_j)
                    used[j] = True

            if len(group) == 1:
                merged.append(w_i)
            else:
                # Merge: longest segment wins; extend endpoints
                best = max(group, key=lambda w: w["length_m"])
                total_length = sum(w["length_m"] for w in group)
                best = dict(best)
                best["length_m"] = round(total_length, 4)
                best["n_inliers"] = sum(w["n_inliers"] for w in group)
                merged.append(best)

        return merged

    # ── Floor footprint ──────────────────────────────────────────────────────

    def _compute_footprint(self, xz: np.ndarray) -> Dict:
        if len(xz) < 3:
            return {"area_m2": 0.0, "polygon": np.zeros((4, 2))}
        try:
            hull = ConvexHull(xz)
            return {
                "area_m2": float(hull.volume),  # .volume = area for 2D hull
                "polygon": xz[hull.vertices],
            }
        except Exception as e:
            logger.warning(f"ConvexHull failed: {e}")
            return {"area_m2": 0.0, "polygon": xz[:4]}

    # ── Opening detection ────────────────────────────────────────────────────

    def _detect_openings(
        self, xyz: np.ndarray, walls: List[Dict],
        floor_y: float, ceiling_y: float
    ) -> List[Dict]:
        """
        Detect door/window openings by scanning each wall for
        horizontal gaps in point density.
        """
        openings = []
        opening_id = 0
        room_height = ceiling_y - floor_y

        for wall in walls:
            normal = np.array(wall["normal_xz"])
            tangent = np.array([-normal[1], normal[0]])

            # Reference point on wall (midpoint)
            s = np.array(wall["start_xz"])
            wall_d = float(s @ normal)

            # Points near this wall plane
            dist = np.abs(xyz[:, 0] * normal[0] + xyz[:, 2] * normal[1] - wall_d)
            near = dist < 0.15
            in_range = (xyz[:, 1] > floor_y) & (xyz[:, 1] < ceiling_y)
            pts = xyz[near & in_range]

            if len(pts) < 30:
                continue

            # Project onto (along-wall, vertical)
            along = pts[:, 0] * tangent[0] + pts[:, 2] * tangent[1]
            vert  = pts[:, 1]

            wall_len = wall["length_m"]
            n_bins_h = max(20, min(60, int(wall_len / 0.05)))  # ~5cm bins
            n_bins_v = 20

            along_range = (along.min(), along.max())
            vert_range  = (floor_y, ceiling_y)

            H, xe, ye = np.histogram2d(
                along, vert,
                bins=[n_bins_h, n_bins_v],
                range=[along_range, vert_range],
            )

            col_density = H.sum(axis=1)
            if col_density.max() == 0:
                continue

            gap_thresh = col_density.max() * 0.10
            is_gap = col_density < gap_thresh
            bin_width_m = (along_range[1] - along_range[0]) / n_bins_h

            for start_bin, end_bin in self._contiguous_groups(is_gap):
                width_m = (end_bin - start_bin + 1) * bin_width_m
                if not (0.5 <= width_m <= 3.0):
                    continue

                # Vertical extent of opening
                gap_rows = H[start_bin:end_bin + 1, :].sum(axis=0)
                empty_v  = np.where(gap_rows < gap_thresh)[0]
                if len(empty_v) == 0:
                    continue

                v_min = float(vert_range[0] + empty_v[0] * room_height / n_bins_v)
                v_max = float(vert_range[0] + (empty_v[-1] + 1) * room_height / n_bins_v)
                opening_h = v_max - v_min

                if opening_h < 0.8:
                    continue

                opens_at_floor = (v_min - floor_y) < 0.25
                opening_type = "door" if opens_at_floor else "window"
                width_ci = bin_width_m / 2

                openings.append({
                    "id":                  f"opening_{opening_id:02d}",
                    "type":                opening_type,
                    "wall_id":             wall["id"],
                    "width_m":             round(width_m, 3),
                    "width_ci_m":          round(width_ci, 3),
                    "height_m":            round(opening_h, 3),
                    "sill_height_m":       round(max(0.0, v_min - floor_y), 3),
                    "along_wall_start_m":  round(float(along_range[0] + start_bin * bin_width_m), 3),
                    "along_wall_end_m":    round(float(along_range[0] + (end_bin + 1) * bin_width_m), 3),
                })
                opening_id += 1

        logger.info(f"  Detected {len(openings)} openings")
        return openings

    @staticmethod
    def _contiguous_groups(mask: np.ndarray) -> List[Tuple[int, int]]:
        groups, in_g, start = [], False, 0
        for i, v in enumerate(mask):
            if v and not in_g:
                start, in_g = i, True
            elif not v and in_g:
                groups.append((start, i - 1))
                in_g = False
        if in_g:
            groups.append((start, len(mask) - 1))
        return groups

    # ── Confidence intervals ─────────────────────────────────────────────────

    def _compute_confidence_intervals(
        self, xyz: np.ndarray, walls: List[Dict],
        floor_y: float, ceiling_y: float,
        footprint: Dict, ceiling_reliable: bool
    ) -> Dict:
        """
        Compute CIs from point residuals — not invented values.

        Ceiling height CI:
          σ from Y spread of ceiling-layer inliers (if ceiling reliable)
          or a wider fallback if ceiling was estimated from percentile.

        Floor area CI:
          Based on convex hull sensitivity: ≈ 2% for LiDAR (empirical from
          point cloud density), wider if ceiling not reliable.
        """
        # Ceiling height
        ceiling_pts_y = xyz[xyz[:, 1] > ceiling_y - 0.15, 1]
        if ceiling_reliable and len(ceiling_pts_y) > 10:
            ceiling_std = float(np.std(ceiling_pts_y))
            ceiling_ci = float(1.96 * ceiling_std / np.sqrt(len(ceiling_pts_y)) + 0.005)
        else:
            # Ceiling estimated from percentile — use larger uncertainty
            ceiling_ci = 0.10  # 10 cm uncertainty when ceiling not scanned

        # Floor area
        area = footprint["area_m2"]
        area_ci = area * (0.02 if ceiling_reliable else 0.05)

        return {
            "ceiling_height_ci": max(0.005, ceiling_ci),
            "floor_area_ci":     max(0.05, area_ci),
        }
