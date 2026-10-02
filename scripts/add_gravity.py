"""Add gravity alignment to LiDARProcessor via string patching."""
import re

content = open('src/lidar_processor.py', 'r').read()

# ── New methods to insert ────────────────────────────────────────────────────
GRAVITY_METHODS = '''
    def _load_imu_gravity(self) -> "np.ndarray | None":
        """
        Estimate the gravity direction in world coordinates from IMU accelerometer.
        Mean linear acceleration over the session ~ gravity
        (valid when device is not under sustained non-gravitational acceleration).
        Returns: unit vector pointing DOWN (direction of gravity), or None if unavailable.
        """
        imu_path = self.input_path / "imu.csv"
        if not imu_path.exists():
            logger.warning("imu.csv not found; gravity alignment disabled")
            return None
        accels = []
        with open(imu_path, newline="") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [k.strip() for k in reader.fieldnames]
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    ax = float(row.get("a_x", row.get("ax", "0")))
                    ay = float(row.get("a_y", row.get("ay", "0")))
                    az = float(row.get("a_z", row.get("az", "0")))
                    accels.append([ax, ay, az])
                except (ValueError, KeyError):
                    continue
        if len(accels) < 10:
            logger.warning("Too few IMU samples for gravity estimation")
            return None
        mean_accel = np.mean(accels, axis=0)
        norm = float(np.linalg.norm(mean_accel))
        if norm < 0.1:
            logger.warning(f"IMU mean accel too small ({norm:.3f}); gravity alignment disabled")
            return None
        grav = mean_accel / norm
        logger.info(f"Gravity direction from IMU (world space): {grav.round(4)}")
        return grav.astype(np.float64)

    @staticmethod
    def _gravity_alignment_rotation(gravity_world: "np.ndarray") -> "np.ndarray":
        """
        Build 3x3 rotation R so that R @ gravity_world = [0, -1, 0].
        After this, +Y is vertical up and -Y is down.
        Uses Rodrigues formula.
        """
        g = gravity_world / np.linalg.norm(gravity_world)
        up_world = -g
        target = np.array([0., 1., 0.])
        v = np.cross(up_world, target)
        s = float(np.linalg.norm(v))
        c = float(np.dot(up_world, target))
        if s < 1e-8:
            return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
        Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + Kx + Kx @ Kx * ((1 - c) / (s * s))

'''

# Insert methods before _load_camera_matrix
marker = '    # ── Private helpers'
assert marker in content, f"Marker not found: {marker}"
content = content.replace(marker, GRAVITY_METHODS + marker, 1)

# ── Add gravity computation before the empty-check ──────────────────────────
old_empty_check = '        if not all_pts_world:'
new_empty_check = '''        # Apply gravity alignment so that +Y = vertical up
        gravity = self._load_imu_gravity()
        if gravity is not None:
            R_grav = self._gravity_alignment_rotation(gravity)
        else:
            R_grav = None

        if not all_pts_world:'''
assert old_empty_check in content, "Could not find empty-check marker"
content = content.replace(old_empty_check, new_empty_check, 1)

# ── Rotate point cloud after stacking ───────────────────────────────────────
old_stack = '        xyz = np.vstack(all_pts_world)\n        rgb = np.vstack(all_rgb)'
new_stack = '''        xyz = np.vstack(all_pts_world)
        if R_grav is not None:
            xyz = (R_grav @ xyz.T).T.astype(np.float32)
            logger.info(
                f"Gravity-aligned point cloud Y: "
                f"[{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}] m"
            )
        rgb = np.vstack(all_rgb)'''
assert old_stack in content, "Could not find stack marker"
content = content.replace(old_stack, new_stack, 1)

open('src/lidar_processor.py', 'w').write(content)
print("Done. File size:", len(content), "bytes")
print("Gravity methods present:", "_load_imu_gravity" in content)
print("Rotation applied:", "R_grav @ xyz.T" in content)
