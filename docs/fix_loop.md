# Fix Loop

## 1. Problem / weakest observed behavior
In the initial Checkpoint 3 baseline (`benchmark/checkpoint3_before/report.json`), the pipeline exhibited severe geometric inconsistency in room boundary definition, wall extraction, and opening detection:

1. **Trajectory-Hull Contamination of Room Area**: Floor area was reported as **20.77 m²** (and up to 38.89 m² in earlier uncalibrated passes), derived from the 2D convex hull of all floor returns. This captured the operator's entire walking path across the room rather than the physical room perimeter.
2. **Proliferation of Phantom Wall Planes**: 18 independent wall candidates were detected across 6 arbitrary azimuth angles (1.5°, 18.5°, 45.5°, 61.5°, 98.5°, 114.5°). Non-orthogonal planes caused by scanner motion sweeps and noise clusters were treated as physical walls.
3. **Out-of-Bounds and Spurious Openings**: 24 openings were detected. 10 of them had negative along-wall start coordinates (e.g. `along_wall_start_m: -1.814` on `wall_11`) because gap detection operated along unconstrained line segments extending beyond physical corners.
4. **Infinite Wall Line Bleed**: Technical floor plan rendering drew unbounded lines across the bounding box, resulting in overlapping infinite lines crossing room boundaries.

Citation:
- Baseline metrics: [benchmark/checkpoint3_before/report.json](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/benchmark/checkpoint3_before/report.json)
- Baseline floor plan: [benchmark/checkpoint3_before/floor_plan.png](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/benchmark/checkpoint3_before/floor_plan.png)
- Baseline diff: [benchmark/diff/comparison.json](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/benchmark/diff/comparison.json)

---

## 2. Evidence
Direct quantitative evidence extracted from the existing baseline artifact `benchmark/checkpoint3_before/report.json`:

- **Detected walls**: 18 candidates
- **Wall azimuths**: `[1.5°, 18.5°, 45.5°, 61.5°, 98.5°, 114.5°]` (scattered, lacking orthogonal structure)
- **Reported wall residuals**: `min: 0.0142 m, max: 0.0146 m` (flat, reflecting distance inlier thresholding rather than true RMS distance to plane)
- **Detected openings**: 24 openings across 10 wall segments
- **Negative opening coordinates**: Openings on `wall_00`, `wall_01`, `wall_02`, `wall_04`, `wall_07`, `wall_08`, `wall_09`, `wall_11`, `wall_12`, and `wall_13` all contained negative start positions (e.g. `along_wall_start_m = -1.814 m`)
- **Floor area**: 20.7678 m² (CI: ±0.4154 m²)
- **Room polygon**: Absent / null (only `footprint_polygon` existed, containing a 31-vertex convex hull of 134,730 scattered floor returns)
- **Processing time**: 43.77 s

---

## 3. Root cause
Detailed technical inspection of `src/geometry.py` and git history (`commit 4b9b171`, `commit 776e9b1`) revealed four concrete root causes:

1. **Convex Hull on Cumulative Floor Returns**: `src/geometry.py` computed floor area by taking `scipy.spatial.ConvexHull` over all projected 2D floor points. In handheld mobile LiDAR, floor points accumulate along the surveyor's entire walking path. The convex hull inevitably expands with the operator's translation and camera tilt, measuring scanner trajectory span rather than the architectural room boundary.
2. **Absence of Orthogonal Filtering**: Wall detection performed RANSAC on point normals without orientation clustering. Density variations caused by slow movement or pause points generated high inlier counts for non-orthogonal planes (e.g., 45.5° and 61.5°), and created multiple redundant slices of the same structural wall.
3. **Unbounded 1D Projection for Gap Detection**: `_detect_openings()` projected point clouds onto wall tangent lines across the full span of detected inliers without clipping to corner intersections. Edge points from adjacent walls appeared as gaps along the projected axis, creating phantom doors with negative offsets.
4. **Unclipped Line Rendering**: `src/floor_plan.py` lacked parametric boundary clipping against the room polygon, extending lines across the entire image canvas.

---

## 4. Shipped fix
The fix was implemented and committed in `commit 4b9b171` (Checkpoint 3) and `commit 776e9b1` (Checkpoint 4):

1. **Dominant Azimuth & Orthogonal Family Filtering (`src/geometry.py`)**:
   - Computes an azimuth histogram of wall candidate normals.
   - Identifies orthogonal normal pairs ($|\theta_1 - \theta_2| \approx 90^\circ \pm 12^\circ$).
   - Enforces membership in the primary orthogonal families (`orthogonal_families_deg: [20.5°, 113.5°]`), rejecting diagonal and motion-induced phantom planes.
2. **6-Criterion Geometric Wall Rejection (`src/geometry.py`)**:
   - Rejects candidates failing: height coverage ($\ge 25\%$), point density threshold, true RMS plane residual threshold, minimum inlier count, and parallel slice deduplication ($< 0.3\,\text{m}$ offset sharing azimuth).
   - Replaces hardcoded distance thresholds with true RMS distance calculation:
     $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mathbf{n} \cdot \mathbf{p}_i - d)^2}$$
3. **Analytical Wall-Intersection Room Polygon (`src/geometry.py`)**:
   - Computes analytical 2D line intersections between adjacent orthogonal walls.
   - Solves corner coordinates and orders them into a closed, non-self-intersecting `room_polygon`.
   - Calculates `floor_area_m2` using the Shoelace formula on the wall-intersection polygon, tagged with `floor_area_source: "wall_intersections"`.
4. **Bounded Opening Validation (`src/geometry.py`)**:
   - Clips opening detection intervals strictly to $[0, L_{\text{wall}}]$, eliminating negative along-wall coordinates.
   - Requires verified empty bins across the vertical extent (floor to lintel).
   - Flags uncertain opening detections with low confidence and explicit verification notes.
5. **Parametric Finite Segment Clipping (`src/geometry.py`, `src/floor_plan.py`)**:
   - Implemented Liang-Barsky parametric line clipping of wall segments against the bounding room polygon.
   - Stores explicit `finite_segment` coordinates in the output schema and renders bounded segments on the floor plan without canvas bleed.

---

## 5. Prediction
Based on the design intent documented in git history and `scripts/cp3_diff.py` prior to final validation:

- **Wall count**: Expected to reduce from 18 unconstrained planes to 4–8 structural boundaries.
- **Wall azimuths**: Expected to collapse from 6 scattered angles into 2 orthogonal families separated by ~90°.
- **Floor area**: Expected to shrink from the trajectory-inflated ~20.77 m² to an enclosed structural polygon (~4.5–7.0 m²).
- **Openings**: Expected to drop from 24 unconstrained openings to conservative candidates with zero negative coordinates.
- **Residuals**: Expected to reflect true physical point-to-plane RMS values (~2–3 cm) rather than artificial constant threshold values.

*Chronology note*: The repository history records the engineering objectives above, but does not contain a timestamped pre-execution quantitative prediction (such as an exact numerical percentage improvement) prior to code execution. In accordance with strict engineering standards, no fabricated numerical prediction is claimed.

---

## 6. After result
Direct metrics from the post-fix artifacts `benchmark/checkpoint3_after/report.json` and `single_room_output/report.json`:

- **Detected walls**: Reduced to **7** (CP3 after) / **8** (CP4 single_room_output)
- **Wall azimuths**: Clustered strictly into orthogonal pairs: `[20.5°, 113.5°]` (CP3 after) / `[17.5°, 102.5°]` (CP4)
- **Wall RMS residuals**: True RMS measured at **0.0279 m – 0.0293 m** across all retained walls
- **Detected openings**: Reduced to **7** conservative candidates with low-confidence tags (CP3 after) / **0** after strict opening validation (CP4)
- **Negative coordinates**: **0** (all opening coordinates are strictly positive and within wall lengths)
- **Room polygon**: Successfully extracted with **4** corners; perimeter = **9.689 m** (CP3) / **7.922 m** (CP4)
- **Floor area**: **4.9969 m²** (CI: ±0.1753 m², source: `wall_intersections`) in CP3 after; **2.5277 m²** in CP4 video run
- **Floor plan rendering**: Walls rendered as clipped finite segments within the room polygon boundary

Artifact locations:
- [benchmark/checkpoint3_after/report.json](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/benchmark/checkpoint3_after/report.json)
- [benchmark/checkpoint3_after/floor_plan.png](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/benchmark/checkpoint3_after/floor_plan.png)
- [benchmark/checkpoint3_after/single_room_debug_panels.png](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/benchmark/checkpoint3_after/single_room_debug_panels.png)
- [benchmark/checkpoint3_after/single_room_wall_table.png](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/benchmark/checkpoint3_after/single_room_wall_table.png)
- [single_room_output/report.json](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/single_room_output/report.json)
- [single_room_output/floor_plan.png](file:///c:/Users/ksr20/OneDrive/Desktop/Brynz/single_room_output/floor_plan.png)

---

## 7. Before vs after

| Metric | BEFORE (CP3 Baseline) | AFTER (CP3 Fix) | AFTER (CP4 Hardened) | Status / Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Wall Count** | 18 | 7 | 8 | Filtered 11 phantom/duplicate planes |
| **Wall Azimuths** | 6 angles (1.5°–114.5°) | 2 orthogonal families (20.5°, 113.5°) | 2 orthogonal families (17.5°, 102.5°) | Orthogonal structure restored |
| **Wall Residual Metric** | Inlier threshold (0.014 m) | True RMS (0.028–0.029 m) | True RMS (0.028–0.029 m) | Statistically valid RMS |
| **Floor Area** | 20.77 m² | 4.9969 m² | 2.5277 m² | Bounded by walls, not walking path |
| **Floor Area Source** | Trajectory convex hull | `wall_intersections` | `wall_intersections` | Closed analytical polygon |
| **Room Polygon Corners** | 0 (none) | 4 | 4 | Closed corner polygon |
| **Room Perimeter** | Uncomputed | 9.69 m | 7.92 m | Extracted boundary perimeter |
| **Openings Count** | 24 | 7 (conservative) | 0 (strict) | Eliminated unconstrained false positives |
| **Negative Opening Offsets**| 10 | 0 | 0 | Boundary violations eliminated |
| **Wall Segment Rendering** | Infinite unbounded lines | Bounded line segments | Liang-Barsky clipped finite segments | Clean architectural floor plan |
| **Physical Accuracy Gate** | **NOT SCORED** | **NOT SCORED** | **NOT SCORED** | **No physical ground truth exists** |

*Important Benchmark Disclaimer*: Neither the before nor the after run passes or fails the assignment's numerical accuracy gate (e.g. $\pm 2\,\text{cm}$ or $\pm 5\%$). The repository's `benchmark/ground_truth.json` contains `null` for all physical dimensions because no physical tape or laser measurements were recorded. The changes represent verified engineering and geometric improvements, not physical ground-truth validation.

---

## 8. Remaining limitation
While the fix resolved internal geometric consistency and boundary extraction, several fundamental physical and algorithmic limitations remain:

1. **Unverified Physical Dimensions**: Because no independent laser or tape-measured ground truth exists in the repository, the actual physical accuracy of the 4.99 m² or 2.53 m² area and wall dimensions cannot be verified.
2. **Depth Scaling Assumption**: The depth image scaling factor ($0.001\,\text{m}/\text{unit}$) remains an unverified assumption based on standard Apple Record3D conventions.
3. **Focal Length Scaling Assumption**: Depth intrinsics ($f_x = 213.3\,\text{px}$) are scaled from RGB intrinsics ($f_x = 1599.7\,\text{px}$) via the resolution ratio ($256/1920$). Any field-of-view difference between the RGB and LiDAR modules introduces an uncalibrated scale bias.
4. **Open-Loop Drift**: Odometry is integrated without loop-closure optimization (pose graph or ICP). While wall RMS residuals (~2.8 cm) indicate short-range local consistency, cumulative drift across longer multi-room trajectories is uncorrected.
5. **Manhattan-World Bias**: The orthogonal azimuth clustering assumes rooms with perpendicular walls. Non-rectangular, slanted, or curved architectural geometries require more generalized boundary partitioning.

---

## 9. Reproduction
The before, after, and comparison artifacts can be reproduced using established repository scripts and commands:

1. **Inspect Before vs After Report Diff**:
   ```powershell
   python -X utf8 scripts/cp3_diff.py
   ```
2. **Run Pipeline on Single Room Dataset**:
   ```powershell
   python run.py --data sample_data/single_room --tier lidar --output single_room_output
   ```
3. **Run Full Benchmark Harness** (verifies null ground truth handling without crash):
   ```powershell
   python benchmark/run_benchmark.py
   ```
4. **Run Unit and Regression Test Suite**:
   ```powershell
   pytest tests/test_pipeline.py
   ```
