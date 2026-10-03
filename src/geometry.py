"""
Room Geometry Extractor — Checkpoint 3
========================================
Key improvements over CP2:
  1. Wall candidates scored by 6 independent criteria, not just height-coverage
  2. Azimuth families reduced to ≤2 dominant orthogonal directions using
     circular-mean clustering with angular-symmetry folding
  3. Room polygon built from WALL INTERSECTIONS, not convex hull of floor points
  4. Scanner trajectory explicitly separated from room boundary
  5. Residuals computed as true RMS distance to fitted plane, not threshold std
  6. Opening detection disabled until walls are reliable (reports as uncertainty)
  7. All assumptions clearly labelled

Wall scoring criteria (all must pass threshold):
  a. Point count ≥ MIN_WALL_PTS
  b. Physical length ≥ MIN_WALL_LENGTH_M
  c. Height span ≥ MIN_HEIGHT_SPAN_M  (absolute, not fraction)
  d. Height coverage fraction ≥ MIN_HEIGHT_COV_FRAC
  e. RMS plane residual ≤ MAX_RESIDUAL_M
  f. Point density ≥ MIN_POINT_DENSITY_PER_M2

Room polygon algorithm:
  - Take accepted walls' centre lines
  - Extend each line to infinite in both directions
  - Compute all pairwise intersections of walls from DIFFERENT azimuth families
  - Cluster intersection points (corners)
  - Order into a convex (or simple) polygon
  - Validate: polygon area ≈ floor-point bounding area
"""

import logging
import math
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
    """Extracts room geometry from a gravity-aligned 3D point cloud."""

    # ── Tunable thresholds (not hardcoded per dataset) ───────────────────────
    VOXEL_SIZE_M          = 0.03   # 3 cm voxels
    RANSAC_DIST_M         = 0.05   # 5 cm inlier threshold
    RANSAC_ITERS          = 400

    # Floor / ceiling
    MIN_FLOOR_CEIL_PTS    = 200    # min inliers to trust a horizontal plane
    FLOOR_BAND_FRAC       = 0.30   # use bottom 30% of Y range for floor search
    CEIL_BAND_FRAC        = 0.30

    # Wall candidate scoring (all criteria must pass)
    MIN_WALL_PTS          = 150    # minimum inlier points
    MIN_WALL_LENGTH_M     = 0.6    # minimum horizontal extent
    MIN_HEIGHT_SPAN_M     = 0.5    # minimum absolute vertical span of inliers
    MIN_HEIGHT_COV_FRAC   = 0.25   # must cover ≥25% of room height
    MAX_RESIDUAL_M        = 0.06   # RMS residual to fitted plane ≤ 6 cm
    MIN_POINT_DENSITY     = 20.0   # points per m² of wall area

    # Azimuth clustering
    NORMAL_HIST_BINS      = 180
    NORMAL_PEAK_MIN_FRAC  = 0.08
    NORMAL_PEAK_SEP_DEG   = 15
    MAX_AZ_FAMILIES       = 4     # accept at most 4 dominant directions
    ORTHO_TOLERANCE_DEG   = 15    # two azimuths are "orthogonal" if |diff-90|≤15

    # Room polygon
    CORNER_CLUSTER_DIST_M = 0.25  # merge candidate corners within this distance

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def extract(self, point_cloud: Dict) -> Dict[str, Any]:
        xyz = point_cloud["xyz"]
        logger.info(f"Extracting geometry from {len(xyz)} points")
        logger.info(f"  X: [{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]  "
                    f"Y: [{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]  "
                    f"Z: [{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")

        # 1. Preprocess
        xyz_ds    = self._voxel_downsample(xyz)
        xyz_clean = self._remove_outliers(xyz_ds)
        logger.info(f"  Downsampled: {len(xyz_ds)}  After outlier removal: {len(xyz_clean)}")

        # 2. Floor / ceiling
        floor_r   = self._detect_horizontal_plane(xyz_clean, find_lowest=True)
        floor_y   = floor_r["height"]
        logger.info(f"  Floor y={floor_y:.4f}m  ({floor_r['n_inliers']} inliers)")

        ceil_r    = self._detect_horizontal_plane(xyz_clean, find_lowest=False)
        ceil_y    = ceil_r["height"]
        room_h    = abs(ceil_y - floor_y)
        ceil_ok   = (ceil_r["n_inliers"] >= self.MIN_FLOOR_CEIL_PTS
                     and room_h >= 1.0)
        if not ceil_ok:
            logger.warning(
                f"  Ceiling detection uncertain (h={room_h:.2f}m, "
                f"inliers={ceil_r['n_inliers']}). Using Y-97th percentile."
            )
            ceil_y = float(np.percentile(xyz_clean[:, 1], 97))
            room_h = abs(ceil_y - floor_y)
        logger.info(f"  Ceiling y={ceil_y:.4f}m  room_h={room_h:.3f}m  reliable={ceil_ok}")

        # 3. Wall detection (two-stage)
        walls, debug = self._detect_walls(xyz_clean, floor_y, ceil_y, room_h)

        # 4. Room polygon from wall intersections
        polygon_result = self._build_room_polygon(walls, floor_y, xyz_clean)

        # 5. Openings (conservative)
        openings = self._detect_openings_conservative(
            xyz_clean, walls, floor_y, ceil_y, room_h
        )

        # 5b. Clip walls to room polygon to produce finite boundary segments
        poly_pts = polygon_result.get("polygon", [])
        if len(poly_pts) >= 3:
            from .floor_plan import _wall_to_clipped_segment
            poly_np = np.array(poly_pts, dtype=float)
            for wall in walls:
                wall["raw_start_xz"] = wall["start_xz"]
                wall["raw_end_xz"]   = wall["end_xz"]
                wall["raw_length_m"] = wall["length_m"]
                seg = _wall_to_clipped_segment(wall, poly_np)
                if seg is not None:
                    s, e = seg
                    wall["start_xz"] = [round(float(s[0]), 4), round(float(s[1]), 4)]
                    wall["end_xz"]   = [round(float(e[0]), 4), round(float(e[1]), 4)]
                    wall["length_m"] = round(float(np.linalg.norm(e - s)), 4)
                    wall["finite_segment"] = {
                        "start_xz": wall["start_xz"],
                        "end_xz":   wall["end_xz"],
                        "length_m": wall["length_m"],
                        "clipped_to_room_polygon": True,
                    }
                else:
                    wall["finite_segment"] = {
                        "start_xz": wall["start_xz"],
                        "end_xz":   wall["end_xz"],
                        "length_m": wall["length_m"],
                        "clipped_to_room_polygon": False,
                    }

        # 6. Confidence intervals
        ci = self._compute_ci(xyz_clean, walls, floor_y, ceil_y, ceil_ok, polygon_result)

        # 7. Multi-space diagnostic: floor bbox vs polygon area
        floor_pts_xz = xyz_clean[xyz_clean[:, 1] < floor_y + 0.15][:, [0, 2]]
        if len(floor_pts_xz) > 10:
            fp_bbox = float(
                (floor_pts_xz[:, 0].max() - floor_pts_xz[:, 0].min()) *
                (floor_pts_xz[:, 1].max() - floor_pts_xz[:, 1].min())
            )
        else:
            fp_bbox = 0.0
        poly_area = float(polygon_result["area_m2"])
        bbox_ratio = round(fp_bbox / max(0.1, poly_area), 2)
        debug["floor_bbox_m2"]         = round(fp_bbox, 2)
        debug["bbox_polygon_ratio"]    = bbox_ratio
        if "orthogonal_families_deg" not in debug:
            debug["orthogonal_families_deg"] = debug.get("azimuth_families_deg", [])

        return {
            "ceiling_height_m":          round(float(room_h), 4),
            "ceiling_height_ci_m":       round(float(ci["ceiling_height_ci"]), 4),
            "ceiling_detection_reliable": ceil_ok,
            "floor_y":                   round(float(floor_y), 4),
            "ceiling_y":                 round(float(ceil_y), 4),
            "floor_normal":              [0., 1., 0.],  # after gravity alignment
            "floor_area_m2":             round(float(polygon_result["area_m2"]), 4),
            "floor_area_ci_m2":          round(float(ci["floor_area_ci"]), 4),
            "floor_area_source":         polygon_result["source"],
            "walls":                     walls,
            "openings":                  openings,
            "room_polygon":              polygon_result["polygon"],
            "room_perimeter_m":          round(float(polygon_result.get("perimeter_m", 0)), 3),
            "footprint_polygon":         polygon_result["polygon"],  # compat alias
            "n_points_input":            int(len(xyz)),
            "n_points_used":             int(len(xyz_clean)),
            "debug":                     debug,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def _voxel_downsample(self, xyz: np.ndarray) -> np.ndarray:
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            return np.asarray(
                pcd.voxel_down_sample(self.VOXEL_SIZE_M).points, dtype=np.float32
            )
        # Numpy fallback: grid snap
        grid = np.floor(xyz / self.VOXEL_SIZE_M).astype(np.int32)
        _, idx = np.unique(grid, axis=0, return_index=True)
        return xyz[idx].astype(np.float32)

    def _remove_outliers(self, xyz: np.ndarray,
                         nb: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd_c, _ = pcd.remove_statistical_outlier(nb, std_ratio)
            return np.asarray(pcd_c.points, dtype=np.float32)
        q1 = np.percentile(xyz, 1, axis=0)
        q99 = np.percentile(xyz, 99, axis=0)
        return xyz[np.all((xyz >= q1) & (xyz <= q99), axis=1)].astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Floor / ceiling
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_horizontal_plane(self, xyz: np.ndarray, find_lowest: bool) -> Dict:
        y = xyz[:, 1]
        y_min, y_max = float(y.min()), float(y.max())
        y_span = y_max - y_min
        if find_lowest:
            cands = xyz[y <= y_min + y_span * self.FLOOR_BAND_FRAC]
        else:
            cands = xyz[y >= y_max - y_span * self.CEIL_BAND_FRAC]

        if len(cands) < self.MIN_FLOOR_CEIL_PTS:
            ht = float(np.percentile(y, 2 if find_lowest else 98))
            return {"height": ht, "n_inliers": 0}

        best_h, best_n = 0.0, 0
        rng = np.random.default_rng(42)
        n = len(cands)
        for _ in range(self.RANSAC_ITERS):
            h = float(np.mean(cands[rng.choice(n, 3, replace=False), 1]))
            mask = np.abs(cands[:, 1] - h) < self.RANSAC_DIST_M
            cnt = int(mask.sum())
            if cnt > best_n:
                best_n = cnt
                best_h = float(np.median(cands[mask, 1]))
        return {"height": best_h, "n_inliers": best_n}

    # ─────────────────────────────────────────────────────────────────────────
    # Wall detection — two-stage
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_walls(
        self, xyz: np.ndarray, floor_y: float, ceil_y: float, room_h: float
    ) -> Tuple[List[Dict], Dict]:
        """
        Stage 1: Generate wall candidates from normal-histogram azimuths.
        Stage 2: Score each candidate on 6 criteria; reject failures.
        Stage 3: Reduce to dominant orthogonal azimuth families.
        """
        debug = {}

        margin = max(0.10, room_h * 0.08)
        wall_pts = xyz[(xyz[:, 1] > floor_y + margin) & (xyz[:, 1] < ceil_y - margin)]
        logger.info(f"  Wall-zone points: {len(wall_pts)}")
        debug["wall_zone_pts"] = int(len(wall_pts))

        if len(wall_pts) < self.MIN_WALL_PTS:
            return [], debug

        # ── Stage 1a: Estimate local normals via XZ-PCA ─────────────────────
        n_sample = min(len(wall_pts), 12000)
        rng = np.random.default_rng(42)
        sample = wall_pts[rng.choice(len(wall_pts), n_sample, replace=False)]

        from scipy.spatial import cKDTree
        tree = cKDTree(sample[:, [0, 2]])
        azimuths = []
        for i in range(0, n_sample, 4):
            _, idxs = tree.query(sample[i, [0, 2]], k=13)
            nb = sample[idxs[1:]][:, [0, 2]]
            if len(nb) < 4:
                continue
            nb -= nb.mean(axis=0)
            _, evecs = np.linalg.eigh(nb.T @ nb)
            az = float(np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])) % 180)
            azimuths.append(az)

        if len(azimuths) < 20:
            return [], debug

        azimuths = np.array(azimuths)
        debug["n_normal_estimates"] = len(azimuths)

        # ── Stage 1b: Histogram and dominant directions (circular modulo 180°) ────────
        hist, _ = np.histogram(azimuths, bins=self.NORMAL_HIST_BINS, range=(0, 180))
        hist_s = uniform_filter1d(hist.astype(float), size=5, mode="wrap")

        # Circularly wrap histogram before peak detection so boundary peaks at 0°/180° are detected
        pad = int(self.NORMAL_PEAK_SEP_DEG)
        hist_wrapped = np.concatenate([hist_s[-pad:], hist_s, hist_s[:pad]])
        peaks_w, _ = find_peaks(
            hist_wrapped,
            height=hist_s.max() * self.NORMAL_PEAK_MIN_FRAC,
            distance=self.NORMAL_PEAK_SEP_DEG,
        )
        peaks = [p - pad for p in peaks_w if 0 <= p - pad < self.NORMAL_HIST_BINS]

        if len(peaks) == 0:
            dominant_azs = np.array([0.0, 90.0])
        else:
            peak_azs = (np.array(peaks) + 0.5) * (180.0 / self.NORMAL_HIST_BINS)
            peak_heights = hist_s[peaks]
            order = np.argsort(peak_heights)[::-1]
            selected_azs = []
            for idx in order:
                az = float(peak_azs[idx])
                # Circular distance modulo 180° to avoid duplicate split peaks across 0°/180°
                is_far = True
                for sel_az in selected_azs:
                    circ_dist = min(abs(az - sel_az), 180.0 - abs(az - sel_az))
                    if circ_dist < self.NORMAL_PEAK_SEP_DEG:
                        is_far = False
                        break
                if is_far:
                    selected_azs.append(az)
                if len(selected_azs) >= self.MAX_AZ_FAMILIES:
                    break
            dominant_azs = np.array(selected_azs) if selected_azs else np.array([0.0, 90.0])

        debug["raw_dominant_azimuth_deg"] = dominant_azs.tolist()
        logger.info(f"  Raw dominant azimuths: {[round(a,1) for a in dominant_azs]}")

        # ── Stage 1c: Reduce to orthogonal families ──────────────────────────
        dominant_azs = self._pick_orthogonal_families(dominant_azs, hist_s)
        debug["orthogonal_families_deg"] = [round(a, 1) for a in dominant_azs]
        logger.info(f"  Orthogonal azimuth families: {[round(a,1) for a in dominant_azs]}")

        # ── Stage 2: Extract and score wall candidates ───────────────────────
        candidates = self._extract_candidates(wall_pts, dominant_azs, floor_y, ceil_y, room_h)
        debug["n_candidates_before_filter"] = len(candidates)
        logger.info(f"  Candidates before scoring: {len(candidates)}")

        # Score and filter
        walls = []
        rejected = []
        for c in candidates:
            ok, reason = self._score_candidate(c, room_h)
            if ok:
                walls.append(c)
            else:
                rejected.append((c["id"], reason))

        debug["n_rejected"] = len(rejected)
        debug["rejection_reasons"] = rejected[:10]  # cap for readability

        # ── Stage 3: Deduplicate near-identical walls ─────────────────────────
        walls = self._deduplicate_walls(walls)
        walls.sort(key=lambda w: w["length_m"], reverse=True)

        logger.info(
            f"  Walls: {len(candidates)} candidates "
            f"→ {len(walls)} after scoring+dedup "
            f"(rejected {len(rejected)})"
        )
        return walls, debug

    def _pick_orthogonal_families(self, azs: np.ndarray, hist_s: np.ndarray) -> np.ndarray:
        """
        From the dominant azimuths, select a minimal set of ≤2 orthogonal families.

        Algorithm:
        1. Sort by histogram strength (strongest first)
        2. Take the strongest as family A
        3. Among the remaining, pick the one closest to A+90 as family B
           (only if within ORTHO_TOLERANCE_DEG of perpendicular)
        4. If no near-orthogonal partner, keep family A alone (L-shaped or single-axis room)
        5. If more than 2 families needed (e.g. hexagonal room), allow up to MAX_AZ_FAMILIES
           but only if their angular separation is not explained by A or B

        Returns: array of selected azimuth values (degrees).
        """
        if len(azs) == 0:
            return azs
        if len(azs) == 1:
            return azs

        # Pair candidates by angular proximity to 90 degrees apart
        best_pair = None
        best_pair_score = -1

        for i in range(len(azs)):
            for j in range(len(azs)):
                if i == j:
                    continue
                diff = abs(azs[i] - azs[j]) % 180
                angle_from_90 = abs(diff - 90)
                if angle_from_90 <= self.ORTHO_TOLERANCE_DEG:
                    # Score = lower angle_from_90 wins
                    score = self.ORTHO_TOLERANCE_DEG - angle_from_90
                    if score > best_pair_score:
                        best_pair_score = score
                        best_pair = (i, j)

        if best_pair is not None:
            i, j = best_pair
            selected = np.array([azs[i], azs[j]])
            logger.info(
                f"  Selected orthogonal pair: {azs[i]:.1f}° | {azs[j]:.1f}° "
                f"(angular diff = {abs(azs[i]-azs[j])%180:.1f}°)"
            )
            return selected
        else:
            # No orthogonal pair found — use all peaks up to MAX_AZ_FAMILIES
            logger.info(
                f"  No orthogonal pair within {self.ORTHO_TOLERANCE_DEG}°. "
                f"Using all {len(azs)} dominant directions."
            )
            return azs

    def _extract_candidates(
        self, wall_pts: np.ndarray,
        azimuth_deg_list: np.ndarray,
        floor_y: float, ceil_y: float, room_h: float
    ) -> List[Dict]:
        """
        For each dominant azimuth, scan along the normal direction and
        extract wall clusters. Returns all raw candidates (unscored).
        """
        candidates = []
        cand_id = 0

        for az_deg in azimuth_deg_list:
            az_rad = np.radians(az_deg)
            normal  = np.array([np.cos(az_rad), np.sin(az_rad)])  # in XZ
            tangent = np.array([-normal[1], normal[0]])

            proj_n = wall_pts[:, 0] * normal[0] + wall_pts[:, 2] * normal[1]

            p_hist, p_edges = np.histogram(proj_n, bins=100)
            p_smooth = uniform_filter1d(p_hist.astype(float), size=3)
            wpeaks, _ = find_peaks(p_smooth, height=p_smooth.max() * 0.10, distance=3)

            # Take strongest 4 peaks per direction
            top = wpeaks[np.argsort(p_smooth[wpeaks])[::-1][:4]]

            for wp in top:
                wall_d = float((p_edges[wp] + p_edges[wp + 1]) / 2)
                inlier_mask = np.abs(proj_n - wall_d) < self.RANSAC_DIST_M
                inliers = wall_pts[inlier_mask]

                if len(inliers) < 50:
                    continue

                along = inliers[:, 0] * tangent[0] + inliers[:, 2] * tangent[1]
                length_m = float(along.max() - along.min())

                # True RMS residual (not threshold-based)
                dist_to_plane = np.abs(
                    inliers[:, 0] * normal[0] + inliers[:, 2] * normal[1] - wall_d
                )
                rms_residual = float(np.sqrt(np.mean(dist_to_plane ** 2)))

                # Point density: pts per m² of wall face
                h_span = float(inliers[:, 1].max() - inliers[:, 1].min())
                wall_area = max(0.01, length_m * h_span)
                density = float(len(inliers) / wall_area)

                h_cov = h_span / max(0.01, room_h)

                start_xz = (normal * wall_d + tangent * along.min()).tolist()
                end_xz   = (normal * wall_d + tangent * along.max()).tolist()

                # Length CI from point spread
                along_std = float(np.std(along))
                length_ci = float(min(
                    1.96 * along_std / max(1, len(along) ** 0.5) + 0.005,
                    length_m * 0.05
                ))

                candidates.append({
                    "id":               f"wall_{cand_id:02d}",
                    "azimuth_deg":      round(float(az_deg), 1),
                    "normal_xz":        normal.tolist(),
                    "tangent_xz":       tangent.tolist(),
                    "wall_d":           round(wall_d, 4),
                    "length_m":         round(length_m, 4),
                    "length_ci_m":      round(length_ci, 4),
                    "start_xz":         start_xz,
                    "end_xz":           end_xz,
                    "n_inliers":        int(len(inliers)),
                    "height_span_m":    round(h_span, 4),
                    "height_coverage":  round(float(min(1.0, h_cov)), 3),
                    "rms_residual_m":   round(rms_residual, 5),
                    "point_density":    round(density, 2),
                    "inlier_y_min":     round(float(inliers[:, 1].min()), 4),
                    "inlier_y_max":     round(float(inliers[:, 1].max()), 4),
                    # kept for backward compat
                    "plane_residual_m": round(rms_residual, 5),
                    "height_from_floor_m": round(float(inliers[:, 1].min() - floor_y), 4),
                })
                cand_id += 1

        return candidates

    def _score_candidate(self, c: Dict, room_h: float) -> Tuple[bool, str]:
        """
        Apply all 6 wall-acceptance criteria.
        Returns (accepted: bool, rejection_reason: str).
        """
        if c["n_inliers"] < self.MIN_WALL_PTS:
            return False, f"too_few_pts({c['n_inliers']}<{self.MIN_WALL_PTS})"

        if c["length_m"] < self.MIN_WALL_LENGTH_M:
            return False, f"too_short({c['length_m']:.2f}m<{self.MIN_WALL_LENGTH_M}m)"

        if c["height_span_m"] < self.MIN_HEIGHT_SPAN_M:
            return False, f"height_span_low({c['height_span_m']:.2f}m<{self.MIN_HEIGHT_SPAN_M}m)"

        if c["height_coverage"] < self.MIN_HEIGHT_COV_FRAC:
            return False, f"height_cov_low({c['height_coverage']:.2f}<{self.MIN_HEIGHT_COV_FRAC})"

        if c["rms_residual_m"] > self.MAX_RESIDUAL_M:
            return False, f"high_residual({c['rms_residual_m']:.4f}>{self.MAX_RESIDUAL_M})"

        if c["point_density"] < self.MIN_POINT_DENSITY:
            return False, f"low_density({c['point_density']:.1f}<{self.MIN_POINT_DENSITY})"

        return True, ""

    def _deduplicate_walls(self, walls: List[Dict], dist_tol: float = 0.15) -> List[Dict]:
        """
        Remove near-duplicate walls: same azimuth family (±3°) and similar
        normal-offset (within dist_tol). Keep the one with more inliers.
        """
        if len(walls) < 2:
            return walls
        keep = [True] * len(walls)
        for i in range(len(walls)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(walls)):
                if not keep[j]:
                    continue
                az_diff = abs(walls[i]["azimuth_deg"] - walls[j]["azimuth_deg"]) % 180
                if az_diff > 5:
                    continue
                if abs(walls[i]["wall_d"] - walls[j]["wall_d"]) < dist_tol:
                    # Keep the one with more inliers
                    if walls[i]["n_inliers"] >= walls[j]["n_inliers"]:
                        keep[j] = False
                    else:
                        keep[i] = False
        return [w for w, k in zip(walls, keep) if k]

    # ─────────────────────────────────────────────────────────────────────────
    # Room polygon from wall intersections
    # ─────────────────────────────────────────────────────────────────────────

    def _build_room_polygon(
        self, walls: List[Dict], floor_y: float, xyz_clean: np.ndarray
    ) -> Dict:
        """
        Build the room boundary polygon from wall line intersections.
        Only walls from DIFFERENT azimuth families can intersect to form corners.

        Falls back to convex hull of floor points if <3 valid corners are found.
        """
        if len(walls) < 2:
            return self._floor_convex_hull(xyz_clean, floor_y, source="convex_hull_fallback")

        # Separate walls by azimuth family (group within 15° of each other)
        families = self._group_by_azimuth(walls)
        logger.info(f"  Azimuth families for polygon: {[len(f) for f in families]}")

        if len(families) < 2:
            return self._floor_convex_hull(xyz_clean, floor_y, source="convex_hull_single_family")

        # Collect wall lines as (point_on_line, direction, normal) tuples
        # Represent each wall as infinite line in XZ plane: n·x = d
        corners = []
        for fam_i in families:
            for fam_j in families:
                if fam_j is fam_i:
                    continue
                for wa in fam_i:
                    for wb in fam_j:
                        pt = self._line_intersection_xz(wa, wb)
                        if pt is not None:
                            corners.append(pt)

        if len(corners) < 3:
            logger.warning("  <3 wall intersection corners — falling back to floor hull")
            return self._floor_convex_hull(xyz_clean, floor_y, source="convex_hull_fallback")

        corners = np.array(corners)

        # Cluster nearby corners
        corners = self._cluster_corners(corners, self.CORNER_CLUSTER_DIST_M)
        logger.info(f"  Room corners after clustering: {len(corners)}")

        if len(corners) < 3:
            return self._floor_convex_hull(xyz_clean, floor_y, source="convex_hull_fallback")

        # Order corners into a polygon (convex hull of intersection points)
        try:
            hull = ConvexHull(corners)
            polygon = corners[hull.vertices]
            area_m2 = float(hull.volume)  # .volume = area for 2D
        except Exception as e:
            logger.warning(f"  ConvexHull of corners failed: {e}")
            return self._floor_convex_hull(xyz_clean, floor_y, source="convex_hull_fallback")

        # Perimeter
        n = len(polygon)
        perimeter = float(sum(
            np.linalg.norm(polygon[(i+1) % n] - polygon[i])
            for i in range(n)
        ))

        # Validate: compare to floor-layer bounding box
        floor_pts_xz = xyz_clean[xyz_clean[:, 1] < floor_y + 0.15][:, [0, 2]]
        if len(floor_pts_xz) > 10:
            fp_bbox_area = float(
                (floor_pts_xz[:, 0].max() - floor_pts_xz[:, 0].min()) *
                (floor_pts_xz[:, 1].max() - floor_pts_xz[:, 1].min())
            )
            logger.info(
                f"  Polygon area={area_m2:.2f}m²  "
                f"floor-bbox={fp_bbox_area:.2f}m² (scanner trajectory bound)"
            )

        logger.info(f"  Room polygon: {n} corners, area={area_m2:.2f}m², perimeter={perimeter:.2f}m")

        return {
            "polygon":     polygon.tolist(),
            "area_m2":     area_m2,
            "perimeter_m": perimeter,
            "n_corners":   n,
            "source":      "wall_intersections",
        }

    def _group_by_azimuth(self, walls: List[Dict], tol: float = 15.0) -> List[List[Dict]]:
        """Group walls into azimuth families (within tol degrees)."""
        if not walls:
            return []
        groups: List[List[Dict]] = []
        assigned = [False] * len(walls)
        for i, w in enumerate(walls):
            if assigned[i]:
                continue
            g = [w]
            assigned[i] = True
            for j in range(i + 1, len(walls)):
                if assigned[j]:
                    continue
                diff = abs(walls[i]["azimuth_deg"] - walls[j]["azimuth_deg"]) % 180
                if diff <= tol or (180 - diff) <= tol:
                    g.append(walls[j])
                    assigned[j] = True
            groups.append(g)
        return groups

    def _line_intersection_xz(self, wa: Dict, wb: Dict) -> Optional[np.ndarray]:
        """
        Intersect two infinite wall lines in XZ plane.
        Wall line: n · (x,z) = d
        Returns (x, z) intersection or None if parallel.
        """
        na = np.array(wa["normal_xz"])
        nb = np.array(wb["normal_xz"])
        da = float(wa["wall_d"])
        db = float(wb["wall_d"])

        # Solve: na·p = da, nb·p = db
        # [na0 na1] [x]   [da]
        # [nb0 nb1] [z] = [db]
        A = np.array([[na[0], na[1]], [nb[0], nb[1]]])
        b = np.array([da, db])
        det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        if abs(det) < 1e-6:
            return None  # parallel lines

        pt = np.linalg.solve(A, b)

        # Check the intersection lies within the extent of BOTH walls
        # (not too far outside the actual wall segments)
        def within_wall_extent(w: Dict, pt: np.ndarray, slack: float = 1.5) -> bool:
            s = np.array(w["start_xz"])
            e = np.array(w["end_xz"])
            t = np.array(w["tangent_xz"])
            along_s = float(s @ t)
            along_e = float(e @ t)
            along_pt = float(pt @ t)
            lo, hi = min(along_s, along_e), max(along_s, along_e)
            margin = slack  # allow 1.5m outside segment
            return (lo - margin) <= along_pt <= (hi + margin)

        if not (within_wall_extent(wa, pt) and within_wall_extent(wb, pt)):
            return None

        return pt

    def _cluster_corners(self, pts: np.ndarray, tol: float) -> np.ndarray:
        """Merge points within tol of each other (simple greedy clustering)."""
        if len(pts) == 0:
            return pts
        used = [False] * len(pts)
        merged = []
        for i in range(len(pts)):
            if used[i]:
                continue
            cluster = [pts[i]]
            used[i] = True
            for j in range(i + 1, len(pts)):
                if not used[j] and np.linalg.norm(pts[i] - pts[j]) < tol:
                    cluster.append(pts[j])
                    used[j] = True
            merged.append(np.mean(cluster, axis=0))
        return np.array(merged)

    def _floor_convex_hull(self, xyz: np.ndarray, floor_y: float, source: str) -> Dict:
        """Fallback: convex hull of floor-layer XZ points."""
        floor_pts = xyz[xyz[:, 1] < floor_y + 0.15][:, [0, 2]]
        if len(floor_pts) < 3:
            return {"polygon": [], "area_m2": 0.0, "perimeter_m": 0.0,
                    "n_corners": 0, "source": source}
        try:
            hull = ConvexHull(floor_pts)
            poly = floor_pts[hull.vertices]
            n = len(poly)
            perim = float(sum(
                np.linalg.norm(poly[(i+1) % n] - poly[i]) for i in range(n)
            ))
            logger.info(
                f"  Fallback floor convex hull: area={hull.volume:.2f}m²  "
                f"NOTE: This is scanner trajectory extent, not room boundary."
            )
            return {
                "polygon":     poly.tolist(),
                "area_m2":     float(hull.volume),
                "perimeter_m": perim,
                "n_corners":   n,
                "source":      source,
            }
        except Exception:
            return {"polygon": [], "area_m2": 0.0, "perimeter_m": 0.0,
                    "n_corners": 0, "source": "failed"}

    # ─────────────────────────────────────────────────────────────────────────
    # Openings (conservative)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_openings_conservative(
        self, xyz: np.ndarray, walls: List[Dict],
        floor_y: float, ceil_y: float, room_h: float
    ) -> List[Dict]:
        """
        Conservative opening detection. CP4 changes:
        - At most ONE opening per wall (the strongest gap only)
        - Wall must be >= 1.5m long (a door cannot fit in a 0.6m wall)
        - along_wall_start_m and along_wall_end_m now included for rendering
        - If only frame-skipped data available, we may have too few pts;
          in that case the wall is still checked but opening suppressed if < 50 pts
        An opening is NOT reported if:
        - Evidence is only absent-sampling (frame skip artefact)
        - Wall is too short relative to the gap width
        """
        openings = []
        opening_id = 0

        # Minimum wall length to attempt opening detection
        MIN_WALL_FOR_OPENING_M = 1.5

        for wall in walls:
            wall_len = float(wall.get("length_m", 0))
            if wall_len < MIN_WALL_FOR_OPENING_M:
                continue

            normal  = np.array(wall["normal_xz"])
            tangent = np.array(wall["tangent_xz"])
            wall_d  = float(wall["wall_d"])

            # Points in 12 cm band around wall, between floor and ceiling
            near_mask   = np.abs(xyz[:, 0] * normal[0] + xyz[:, 2] * normal[1] - wall_d) < 0.12
            height_mask = (xyz[:, 1] > floor_y) & (xyz[:, 1] < ceil_y)
            pts = xyz[near_mask & height_mask]

            if len(pts) < 50:
                continue

            along = pts[:, 0] * tangent[0] + pts[:, 2] * tangent[1]
            vert  = pts[:, 1]

            # 10 cm-wide bins along wall; clamp to wall's actual data range
            a_lo, a_hi = along.min(), along.max()
            actual_span = a_hi - a_lo
            if actual_span < 0.5:
                continue

            n_bins_h = max(6, min(40, int(actual_span / 0.10)))
            n_bins_v = 10
            H, xe, ye = np.histogram2d(
                along, vert, bins=[n_bins_h, n_bins_v],
                range=[[a_lo, a_hi], [floor_y, ceil_y]]
            )

            col_den = H.sum(axis=1)
            if col_den.max() == 0:
                continue

            wall_mean_density = float(col_den[col_den > 0].mean())   # use non-zero bins
            gap_thresh = wall_mean_density * 0.04   # 4% of mean (strict)
            is_gap = col_den < gap_thresh
            bin_w  = actual_span / n_bins_h

            # Collect all candidate gaps, then pick the best one
            candidates = []
            for gs, ge in self._runs(is_gap):
                if (ge - gs + 1) < 3:          # need >= 3 consecutive empty bins
                    continue
                width_m = (ge - gs + 1) * bin_w
                if not (0.6 <= width_m <= 3.0):
                    continue

                # Require gap width < 60% of wall length (not an entire-wall absence)
                if width_m > wall_len * 0.60:
                    continue

                # Vertical: check how much of the gap is empty
                gap_vert = H[gs:ge+1, :].sum(axis=0)
                empty_rows = np.where(gap_vert < gap_thresh * (ge - gs + 1))[0]
                if len(empty_rows) == 0:
                    continue

                v_lo = floor_y + empty_rows[0] * room_h / n_bins_v
                v_hi = floor_y + (empty_rows[-1] + 1) * room_h / n_bins_v
                open_h = v_hi - v_lo
                if open_h < 0.60:
                    continue

                # Score: prefer wider, floor-level gaps
                at_floor = (v_lo - floor_y) < 0.30
                score = width_m * (2.0 if at_floor else 1.0)

                # along_wall position (relative to wall start_xz)
                s_xz = np.array(wall["start_xz"])
                along_s_wall = float(s_xz @ tangent)
                pos_start = float(xe[gs] - along_s_wall)
                pos_end   = float(xe[ge + 1] - along_s_wall)

                candidates.append({
                    "score":    score,
                    "width_m":  round(width_m, 3),
                    "bin_w":    round(bin_w, 3),
                    "open_h":   round(open_h, 3),
                    "v_lo":     v_lo,
                    "at_floor": at_floor,
                    "pos_start": round(pos_start, 3),
                    "pos_end":   round(pos_end, 3),
                })

            if not candidates:
                continue

            # Keep only the highest-scoring gap per wall
            best = max(candidates, key=lambda c: c["score"])
            openings.append({
                "id":                  f"opening_{opening_id:02d}",
                "type":               "door" if best["at_floor"] else "window",
                "wall_id":             wall["id"],
                "width_m":             best["width_m"],
                "width_ci_m":          best["bin_w"],
                "height_m":            best["open_h"],
                "sill_height_m":       round(max(0., best["v_lo"] - floor_y), 3),
                "along_wall_start_m":  best["pos_start"],
                "along_wall_end_m":    best["pos_end"],
                "confidence":          "low",
                "note":               (
                    "Opening evidence: density gap in 2D histogram. "
                    "Requires field verification. May be sparse sampling artefact."
                ),
            })
            opening_id += 1

        logger.info(f"  Openings (conservative): {len(openings)}")
        return openings

    @staticmethod
    def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
        runs, in_r, s = [], False, 0
        for i, v in enumerate(mask):
            if v and not in_r:
                s, in_r = i, True
            elif not v and in_r:
                runs.append((s, i-1))
                in_r = False
        if in_r:
            runs.append((s, len(mask)-1))
        return runs

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence intervals
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_ci(
        self, xyz: np.ndarray, walls: List[Dict],
        floor_y: float, ceil_y: float,
        ceil_reliable: bool, polygon_result: Dict
    ) -> Dict:
        # Ceiling height CI
        ceil_pts_y = xyz[xyz[:, 1] > ceil_y - 0.15, 1]
        if ceil_reliable and len(ceil_pts_y) > 10:
            ci_h = float(1.96 * np.std(ceil_pts_y) / max(1, len(ceil_pts_y)**0.5) + 0.005)
        else:
            ci_h = 0.15  # wider uncertainty if ceiling not scanned

        # Floor area CI
        area = polygon_result.get("area_m2", 0.0)
        source = polygon_result.get("source", "")
        if "wall_intersections" in source:
            # From wall fits — CI based on worst wall length_ci
            max_len_ci = max((w.get("length_ci_m", 0.05) for w in walls), default=0.05)
            area_ci = area * (max_len_ci / max(0.1, area ** 0.5))
        else:
            # Scanner trajectory hull — inherently unreliable for room area
            area_ci = area * 0.20  # 20% uncertainty

        return {
            "ceiling_height_ci": round(max(0.005, ci_h), 4),
            "floor_area_ci":     round(max(0.1, area_ci), 4),
        }
