# Benchmark Audit

**Date:** October 3, 2026  
**Status:** Factual Audit of Existing Repository Evidence  
**Scope:** Applied AI Engineering Assessment — Take-Home Deliverables

---

## 1. Benchmark Datasets

The repository currently contains three raw LiDAR captures provided as sample data. All quantitative dimensions listed below are **pipeline reconstruction outputs**, NOT verified physical ground truth.

### A. `single_room`
- **Tier:** LiDAR
- **Pipeline Execution:** Completed successfully (exit code 0; ~31–60s runtime depending on damage evaluation).
- **Points Processed:** 4,144,214 points (subsampled to 86 frames @ frame_skip=20).
- **Reconstruction Outputs:**
  - Walls Detected: 7 finite clipped segments
  - Openings: 3 conservative gap detections (with along-wall coordinates, marked confidence: low)
  - Reconstructed Floor Area: 6.80 ± 0.19 m² (source: `wall_intersections`)
  - Reconstructed Ceiling Height: 2.127 ± 0.009 m (reliable: True, gravity-aligned)
- **Known Limitations:**
  - True physical room dimensions unmeasured (no tape/laser ground truth).
  - Openings are inferred solely from density gaps in 2D projection, not verified by field measurement.
  - Bbox/polygon ratio is 4.67 due to scanner trajectory entering/exiting doorway; multi-space segmentation is flagged as incomplete.

### B. `single_scan_floor_only`
- **Tier:** LiDAR
- **Pipeline Execution:** Completed successfully (exit code 0; ~75s runtime @ frame_skip=40).
- **Points Processed:** 6,215,251 points (subsampled to 122 frames @ frame_skip=40).
- **Reconstruction Outputs:**
  - Walls Detected: 8 finite clipped segments
  - Openings: 3 conservative detections
  - Reconstructed Floor Area: 18.39 ± 0.36 m² (source: `wall_intersections`)
  - Reconstructed Ceiling Height: 3.935 ± 0.010 m (reliable: True, but indicates missing ceiling scan or double-height space)
- **Known Limitations:**
  - Odometry trajectory spans an 8.51m × 8.67m area over a 53.8m closed loop across multiple rooms/corridors (3 distinct orientation families: ~24.5°, ~151.5°, ~178.5°).
  - Single bounding room polygon cannot represent the multi-room topology without manual doorway partitioning; segmentation status is marked as `incomplete`.
  - Ceiling was not directly scanned (camera pointed primarily at floor), resulting in higher uncertainty for ceiling height.

### C. `single_scan_with_ceiling`
- **Tier:** LiDAR
- **Pipeline Execution:** Completed successfully (exit code 0; ~130s runtime @ frame_skip=80).
- **Points Processed:** 5,760,443 points (subsampled to 122 frames @ frame_skip=80).
- **Reconstruction Outputs:**
  - Walls Detected: 8 finite clipped segments
  - Openings: 7 conservative detections
  - Reconstructed Floor Area: 20.37 ± 0.36 m² (source: `wall_intersections`)
  - Reconstructed Ceiling Height: 2.126 ± 0.006 m (reliable: True, gravity-aligned)
- **Known Limitations:**
  - High frame count (9,745 raw frames) requires frame subsampling (frame_skip ≥ 60) to avoid memory allocation exhaustion on standard machines.
  - Wall orientation families (28.5° / 117.5°) and ceiling height (2.126m) match `single_room` (2.127m), indicating capture within the same physical building, but covering a larger footprint (18.1m perimeter).

---

## 2. Ground-Truth Status

- **Physical Ground Truth Absence:** No physical measurements (laser distance meter or manual tape measurements) exist anywhere in the repository, datasets, or documentation.
- **Template Status:** `benchmark/ground_truth.json` is an unpopulated schema template where all fields (`ceiling_height_m`, `floor_area_m2`, `walls`, `openings`) for all three scans are explicitly set to `null`.
- **Integrity Rule:** No absolute accuracy gates (e.g. ceiling height error ≤ 1.5 cm, opening width error ≤ 2.0 cm) can legitimately be scored or claimed as "passed" at this time. Doing so without physical measurements would require fabricating ground truth.

---

## 3. Gates That Can Be Evidenced Without Physical Ground Truth

| Evaluation Criterion | Status | Evidence Present in Repository |
| :--- | :--- | :--- |
| **Pipeline Execution Across Captures** | **EVIDENCED** | All three supplied LiDAR datasets run to completion via `run.py` and `benchmark/run_benchmark.py`, producing dimensioned floor plans, SVG vectors, JSON reports, and CSV scope items. |
| **Output Schema Validity (v2.0)** | **EVIDENCED** | Output conforms to Schema 2.0: room IDs, finite clipped wall segments, bounding polygon, ceiling height with CI, uncertainty provenance, assumptions block, and multi-space graph. |
| **Test Suite Verification** | **EVIDENCED** | 36 unit/integration tests pass (`pytest tests/test_pipeline.py -v`) covering plane fitting, wall scoring, azimuth selection, polygon clipping, and schema structure. |
| **Drift Audit / Trajectory Evidence** | **EVIDENCED** | Odometry audit recorded in `report.json` and diagnostic scripts (`scripts/diagnose_floor_only.py`). Closed-loop return drift is 0.17m over a 53.8m traverse (0.3% error). Open-loop status and lack of ICP loop-closure are explicitly documented without unsupported claims. |
| **Before / After Fix-Loop Evidence** | **EVIDENCED** | Full diff bundle exists in `benchmark/diff/comparison.json` and checkpoint artifacts (`benchmark/before/`, `benchmark/after/`, `benchmark/checkpoint3_before/`, `benchmark/checkpoint3_after/`). |
| **Benchmark Harness Functionality** | **EVIDENCED** | `benchmark/run_benchmark.py` runs across all captures with failure isolation and dataset-adaptive memory management, correctly reporting unmeasured metrics as `not_scored` rather than crashing. |
| **Self-Consistency (Ceiling Repeatability)** | **PARTIALLY EVIDENCED** | `single_room` reconstructed ceiling height is 2.127 ± 0.009 m; `single_scan_with_ceiling` is 2.126 ± 0.006 m (difference: 0.001 m across independent captures of the same building). However, formal two-capture repeatability of identical room boundaries is not fully demonstrated. |
| **Multi-Tier Processing (Video/Photo)** | **NOT EVIDENCED** | Video and Photo tier processors have placeholders/skeletons, but no verified benchmark runs or outputs across Video/Photo tiers exist on the supplied datasets. |
| **Multi-Room Topological Adjacency** | **PARTIALLY EVIDENCED** | Data structures for `ConnectedSpaceGraph` and adjacency exist in `src/connected_spaces.py`, and multi-space flag triggers on large trajectory ratios. However, automated doorway cut partitioning remains incomplete on complex layouts. |

---

## 4. Assignment Requirements Not Yet Evidenced

1. **Laser / Tape Physical Ground Truth:**
   - No independent physical measurements exist to benchmark absolute dimensional accuracy against the target gates.
2. **Three-Tier Same-Space Capture:**
   - The assignment requires the same physical spaces captured at all three tiers (Photos, Video, LiDAR). Currently, only LiDAR data exists in the repository.
3. **Formal Repeatability Gate:**
   - Requires two distinct captures of the *exact same room* at the *same tier* showing < 1 cm or 0.5% difference per wall. While ceiling height is consistent between `single_room` and `single_scan_with_ceiling`, the scanned footprints differ.
4. **Head-to-Head Incumbent Comparison (Part 3):**
   - Requires comparing pipeline output on 2 benchmark rooms against a consumer scanning app (e.g. Polycam or Magicplan export) showing tie/beat on ≥ 70% of dimensions. No third-party exports are currently present.
5. **Multi-Room Whole-Property Photo Stitch:**
   - Photo-tier per-room folder stitching into a unified property plan has not been executed or evaluated.
6. **Live Walk-In Defense Readiness:**
   - Cold run on an unseen capture requires a lightweight, verified turnkey script tested on fresh captures with minimal dependencies.

---

## 5. Fix-Loop Evidence (Part 4 Compliance)

The repository contains concrete, regenerable before/after evidence from the Checkpoint 2 and Checkpoint 3 geometry repairs:

1. **Problem:**
   - Initial pipeline used RGB camera intrinsics directly on depth frames without resolution scaling ($1599.7\,\text{px}$ instead of $213.3\,\text{px}$) and used unaligned coordinates.
   - Result: Implausible room geometry ($1.098\,\text{m}$ ceiling height, $7.19\,\text{m}^2$ floor area, $12$ walls with arbitrary orientations).
2. **Evidence:**
   - Captured in `benchmark/before/report.json` and `benchmark/diff/comparison.json`.
3. **Shipped Change:**
   - Depth intrinsic scaling by resolution ratio ($256/1920$ and $192/1440$).
   - IMU gravity vector alignment via Rodrigues rotation to establish a physical horizontal floor plane.
   - Dominant azimuth clustering with orthogonal pair selection.
   - Room boundary polygon extraction via wall center-line intersections rather than scanner path convex hull.
   - Liang-Barsky parametric finite wall segment clipping.
4. **Observed Before / After Result:**
   - Ceiling height: $1.098\,\text{m}$ (broken) $\rightarrow$ $2.127 \pm 0.009\,\text{m}$ (physically plausible room height).
   - Ceiling inlier confidence: $6{,}698$ $\rightarrow$ $25{,}793$ inliers.
   - Wall RMS residual: Reduced to $0.028$–$0.029\,\text{m}$ ($2.8\,\text{cm}$ true plane residual).
   - Wall representation: Infinite intersecting lines replaced with finite segments bounded by room corners.
5. **What Remains Unverified:**
   - True absolute metric scale cannot be proven without external laser measurement confirming whether the depth sensor unit is exactly $1.000\,\text{mm/unit}$ or subject to scale drift.

---

## 6. Current Submission Risk

A technical reviewer auditing the repository against the take-home specification will identify the following primary risks:
1. **Empty Ground-Truth Matrix:** `benchmark/results/benchmark_report.md` accurately indicates `⚠️ NOT SCORED (GT null)`. Without physical measurements, quantitative metric compliance cannot be validated.
2. **LiDAR-Only Demonstration:** The pipeline currently demonstrates robust single-room LiDAR processing, but the Video and Photo tiers remain unverified on benchmark datasets.
3. **Unresolved Multi-Room Partitioning:** Large scans (`single_scan_floor_only`) are correctly prevented from hallucinating single boxes, but automated partitioning into individual numbered rooms with doorway adjacency remains incomplete.

---

## 7. Recommended Next Work

The following three tasks represent the sequential path to complete the submission:

1. **Task 1 (Physical / Measurement Dependent):**  
   Populate `benchmark/ground_truth.json` with actual physical measurements (or an authentic captured space with laser-measured dimensions) for at least one scan to enable legitimate execution of the accuracy gates.

2. **Task 2 (Software / Pipeline Independent of New Data):**  
   Execute the Head-to-Head comparison artifact (Part 3) by benchmarking the pipeline against an existing consumer app export (e.g. Polycam or Magicplan free-tier export on a standard sample).

3. **Task 3 (Documentation & Submission Packaging):**  
   Finalize the Technical Report (`docs/technical_report.md`) to incorporate the completed compliance matrix, the fix-loop declaration, the error budget, and the device matrix required for submission.
