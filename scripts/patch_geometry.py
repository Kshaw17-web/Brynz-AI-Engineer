"""Script to patch _detect_walls in geometry.py with histogram-based approach."""
import re

NEW_METHOD = '''    def _detect_walls(self, xyz, floor_z, ceiling_z):
        """Detect walls via normal-histogram clustering (replaces iterative RANSAC)."""
        import numpy as np
        from scipy.spatial import cKDTree
        from scipy.ndimage import uniform_filter1d
        from scipy.signal import find_peaks
        import logging
        _log = logging.getLogger(__name__)

        room_height = ceiling_z - floor_z
        margin = max(0.05, room_height * 0.05)
        mask = (xyz[:, 1] > floor_z + margin) & (xyz[:, 1] < ceiling_z - margin)
        wall_pts = xyz[mask]
        if len(wall_pts) < self.MIN_PLANE_POINTS:
            return []

        n_sample = min(len(wall_pts), 12000)
        rng = np.random.default_rng(42)
        sample = wall_pts[rng.choice(len(wall_pts), n_sample, replace=False)]
        tree = cKDTree(sample[:, [0, 2]])
        azimuths = []
        for i in range(0, n_sample, 4):
            _, idxs = tree.query(sample[i, [0, 2]], k=13)
            nb = sample[idxs[1:], :]
            if len(nb) < 4:
                continue
            xz = nb[:, [0, 2]] - nb[:, [0, 2]].mean(axis=0)
            _, evecs = np.linalg.eigh(xz.T @ xz)
            az = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])) % 180
            azimuths.append(az)
        if len(azimuths) < 10:
            return []

        hist, _ = np.histogram(azimuths, bins=180, range=(0, 180))
        hist_s = uniform_filter1d(hist.astype(float), size=5)
        peaks, _ = find_peaks(hist_s, height=hist_s.max() * 0.08, distance=15)
        if len(peaks) == 0:
            peaks = np.array([0, 90])
        top_peaks = peaks[np.argsort(hist_s[peaks])[::-1][:6]]
        dominant_azimuths = top_peaks + 0.5

        walls = []
        wall_id = 0
        for az_deg in dominant_azimuths:
            az_rad = np.radians(az_deg)
            normal_xz = np.array([np.cos(az_rad), np.sin(az_rad)])
            tangent_xz = np.array([-normal_xz[1], normal_xz[0]])
            proj_n = wall_pts[:, 0] * normal_xz[0] + wall_pts[:, 2] * normal_xz[1]
            p_hist, p_edges = np.histogram(proj_n, bins=80)
            p_hs = uniform_filter1d(p_hist.astype(float), size=3)
            wps, _ = find_peaks(p_hs, height=p_hs.max() * 0.12, distance=4)
            for wp in wps[np.argsort(p_hs[wps])[::-1][:3]]:
                wall_d = (p_edges[wp] + p_edges[wp + 1]) / 2
                inliers = wall_pts[np.abs(proj_n - wall_d) < self.RANSAC_DISTANCE_THRESH]
                if len(inliers) < self.MIN_PLANE_POINTS // 4:
                    continue
                along = inliers[:, 0] * tangent_xz[0] + inliers[:, 2] * tangent_xz[1]
                length_m = float(along.max() - along.min())
                if length_m < 0.5:
                    continue
                h_cov = (inliers[:, 1].max() - inliers[:, 1].min()) / max(0.01, room_height)
                if h_cov < 0.15:
                    continue
                walls.append({
                    "id": f"wall_{wall_id:02d}",
                    "length_m": round(length_m, 4),
                    "length_ci_m": round(float(min(np.std(along)/np.sqrt(len(along))*1.96+0.01, length_m*0.05)), 4),
                    "start_xz": (normal_xz * wall_d + tangent_xz * along.min()).tolist(),
                    "end_xz": (normal_xz * wall_d + tangent_xz * along.max()).tolist(),
                    "normal_xz": normal_xz.tolist(),
                    "n_inliers": int(len(inliers)),
                    "height_coverage": round(float(min(1.0, h_cov)), 3),
                    "azimuth_deg": round(float(az_deg), 1),
                })
                wall_id += 1

        walls.sort(key=lambda w: w["length_m"], reverse=True)
        _log.info(f"Detected {len(walls)} walls")
        return walls

'''

content = open('src/geometry.py', 'r').read()
start_idx = content.find('    def _detect_walls(')
end_idx = content.find('    def _ransac_vertical_plane(')
assert start_idx != -1, "Could not find _detect_walls"
assert end_idx != -1, "Could not find _ransac_vertical_plane"

new_content = content[:start_idx] + NEW_METHOD + content[end_idx:]
open('src/geometry.py', 'w').write(new_content)
print(f"OK: replaced _detect_walls ({end_idx - start_idx} bytes → {len(NEW_METHOD)} bytes)")
print(f"New file: {len(new_content)} bytes")
