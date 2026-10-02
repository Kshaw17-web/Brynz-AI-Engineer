"""CP3 audit and scientific checks."""
import json, shutil, csv, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Save CP3 BEFORE ──────────────────────────────────────────────────────────
b3 = Path('benchmark/checkpoint3_before')
b3.mkdir(parents=True, exist_ok=True)
for f in ['report.json', 'floor_plan.png']:
    src = Path('single_room_output') / f
    if src.exists(): shutil.copy(src, b3 / f)
print("CP3 BEFORE saved to benchmark/checkpoint3_before/")

# ── Wall audit ───────────────────────────────────────────────────────────────
report = json.load(open('single_room_output/report.json'))
room   = list(report['rooms'].values())[0]['geometry']
walls  = room.get('walls', [])
openings = room.get('openings', [])

print(f"\n=== CP3 BEFORE — single_room ===")
print(f"n_walls={len(walls)}  n_openings={len(openings)}")
print(f"ceiling_height={room['ceiling_height_m']:.3f}m  "
      f"floor_area={room['floor_area_m2']:.2f}m2")
print(f"ceiling_reliable={room.get('ceiling_detection_reliable','?')}")
print()
print(f"{'ID':<10} {'Len(m)':<8} {'Az(deg)':<8} {'HtCov':<7} {'Inliers':<8} {'Residual(m)':<12}")
for w in sorted(walls, key=lambda x: x['length_m'], reverse=True):
    print(f"{w['id']:<10} {w['length_m']:<8.3f} {w.get('azimuth_deg',0):<8.1f} "
          f"{w['height_coverage']:<7.2f} {w['n_inliers']:<8} {w.get('plane_residual_m',0):<12.4f}")

azs = sorted([w['azimuth_deg'] for w in walls])
print(f"\nAzimuths: {[round(a,1) for a in azs]}")
print("Orthogonal pairs (within 12 deg of 90 apart):")
found = False
for i,a in enumerate(azs):
    for j,b in enumerate(azs):
        if i >= j: continue
        diff = abs(a - b) % 180
        if 78 <= diff <= 102:
            print(f"  {a:.1f} deg | {b:.1f} deg  diff={diff:.1f}")
            found = True
if not found:
    print("  None found — confirms non-orthogonal azimuths")

# ── Scientific check 1: Depth scale from odometry ───────────────────────────
print("\n=== SCIENTIFIC CHECK 1: Depth scale evidence ===")
with open('single_room/odometry.csv', newline='') as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [k.strip() for k in reader.fieldnames]
    rows = list(reader)
row0 = {k.strip(): v.strip() for k,v in rows[0].items()}
row1 = {k.strip(): v.strip() for k,v in rows[1].items()}
# Position delta between consecutive frames
dx = float(row1['x']) - float(row0['x'])
dy = float(row1['y']) - float(row0['y'])
dz = float(row1['z']) - float(row0['z'])
displacement = (dx**2 + dy**2 + dz**2)**0.5
print(f"Frame 0 pos: ({row0['x']}, {row0['y']}, {row0['z']})")
print(f"Frame 1 pos: ({row1['x']}, {row1['y']}, {row1['z']})")
print(f"Displacement frame 0→1: {displacement:.5f} m")
print(f"At 60fps, this is {displacement*60:.3f} m/s scanner speed")
print(f"fx in odometry row0: {row0.get('fx','not present')}")
print(f"NOTE: If odometry is in metres, scanner moves ~{displacement*60:.3f} m/s")
print(f"      This is {'plausible' if displacement*60 < 2.0 else 'IMPLAUSIBLY FAST — odometry may be in different units'}")

# ── Scientific check 2: Gravity vector in odometry frame ────────────────────
print("\n=== SCIENTIFIC CHECK 2: Gravity vector frame ===")
# IMU a_x/a_y/a_z: in which frame are these?
# The first odometry quaternion tells us the device orientation at t=0
qx,qy,qz,qw = [float(row0[k]) for k in ['qx','qy','qz','qw']]
norm_q = (qx**2+qy**2+qz**2+qw**2)**0.5
print(f"Frame-0 quaternion: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})  norm={norm_q:.4f}")
# Rotate [0,0,1] (z-axis in device) to world
# R[2] column (world Z in device space)
R = np.array([
    [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
    [  2*(qx*qy+qz*qw),1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
    [  2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),1-2*(qx**2+qy**2)],
]) / norm_q**2
print(f"Device Z in world space: {(R @ [0,0,1]).round(4)}")
print(f"Device Y in world space: {(R @ [0,1,0]).round(4)}")
print("NOTE: IMU accelerometer data is typically in device body frame,")
print("      but mean over long session approximates gravity in world frame")
print("      IF the device was not under sustained acceleration. This is an assumption.")

# ── Scientific check 3: Depth intrinsics ────────────────────────────────────
print("\n=== SCIENTIFIC CHECK 3: Depth intrinsics ===")
fx_odo = float(row0.get('fx', 0))
print(f"odometry fx at frame 0: {fx_odo}")
print(f"camera_matrix.csv fx: 1599.696 (RGB resolution 1920x1440)")
print(f"Depth resolution: 256x192")
print(f"Scale factor: 256/1920 = {256/1920:.6f}")
print(f"Scaled depth fx: {1599.696 * 256/1920:.4f}")
print(f"Scaled depth cx: {955.5105 * 256/1920:.4f}")
print(f"NOTE: No separate depth calibration file. RGB intrinsics scaled by")
print(f"      resolution ratio is an assumption. Could be wrong if sensors")
print(f"      have different FOVs (common in multi-sensor devices).")
print(f"      This remains UNVERIFIED without ground-truth geometry.")
