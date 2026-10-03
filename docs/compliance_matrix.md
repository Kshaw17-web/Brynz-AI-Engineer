# Assignment Compliance Matrix

**Date:** October 3, 2026  
**Repository:** https://github.com/Kshaw17-web/Brynz-AI-Engineer  
**Basis:** Factual audit of existing repository artifacts only. No synthetic measurements. No fabricated evidence.

---

## Status Key

| Symbol | Meaning |
| :--- | :--- |
| **EVIDENCED** | Repository contains concrete, verifiable supporting artifact for this requirement |
| **PARTIALLY EVIDENCED** | Repository contains partial/structural/skeleton support but requirement is not fully satisfied |
| **NOT EVIDENCED** | No artifact or evidence found in the repository for this requirement |

---

## Compliance Matrix

| # | Requirement | Required Evidence | Repository File / Artifact | Status | Notes |
| :- | :---------- | :---------------- | :------------------------- | :----- | :---- |
| 1 | **Capture Route documented** | Written protocol for how to capture scan data with a non-engineer-friendly app | `docs/capture_protocol.md` | **EVIDENCED** | Full walk-through for Record3D app, per-room duration, export steps, device list |
| 2 | **Device matrix documented** | List of compatible devices and tier support per device | `README.md`, `docs/technical_report.md` S2 | **EVIDENCED** | iPhone 12 Pro through 15 Pro LiDAR tier; all iPhone 15+ video/photo tier; table in README and technical report |
| 3 | **Photo tier -- pipeline exists** | Code that accepts photo inputs and produces output schema | `src/tiers/photo_processor.py`, `src/pipeline.py` | **PARTIALLY EVIDENCED** | Pipeline skeleton runs and falls back to heuristic reconstruction. DUSt3R/COLMAP not installed in test env. Output schema produced with tier-specific disclosure. No benchmark photos captured on same space as LiDAR. |
| 4 | **Video tier -- pipeline exists** | Code that accepts video input and produces output schema | `src/tiers/video_processor.py`, `src/pipeline.py` | **PARTIALLY EVIDENCED** | Monocular depth heuristic pipeline runs. single_room_output/report.json produced via video tier (schema v2.0). No verified metric depth -- heuristic scale assumption disclosed. |
| 5 | **LiDAR tier -- pipeline exists and runs** | Functional LiDAR processor producing dimensioned floor plan and report | `src/lidar_processor.py`, `src/geometry.py`, `src/floor_plan.py`; `single_room_output/report.json` | **EVIDENCED** | All three supplied LiDAR datasets run to exit code 0. Walls, floor area (wall_intersections), ceiling height, CI, SVG/PNG floor plan all produced. |
| 6 | **Common output contract** | All tiers produce the same JSON schema with required fields | `src/output_schema.py`; `single_room_output/report.json` (schema_version 2.0) | **EVIDENCED** | Schema v2.0 with rooms/{id}/geometry, walls, openings, damage, scope_items, concealed_flags, assumptions block, drift_audit. |
| 7 | **Per-room dimensions** | floor_area_m2, ceiling_height_m, wall lengths and orientations with CI | `benchmark/checkpoint3_after/report.json`; `single_room_output/report.json` | **EVIDENCED** | floor_area_m2 +/- CI (wall_intersections), ceiling_height_m +/- CI, 7-8 finite wall segments with length_ci_m, azimuth, rms_residual_m. Physical accuracy unscored (no ground truth). |
| 8 | **Multi-room stitched plan** | Multiple room outputs joined into a single property plan with adjacency | `src/stitcher.py`, `src/connected_spaces.py`; `stitch.py` CLI | **PARTIALLY EVIDENCED** | ConnectedSpaceGraph data structure and adjacency list exist in output schema. Automated doorway partitioning incomplete on complex multi-room layouts. No end-to-end stitched property output artifact exists. |
| 9 | **Damage regions detected** | Damage detections with class, area_m2, bbox, and confidence | `benchmark/checkpoint3_after/report.json` (34 damage regions) | **PARTIALLY EVIDENCED** | Damage regions present in CP3 LiDAR run (water_stain, mold; confidence 0.5). Current single_room_output (video tier) has 0 damage detections. Damage uses color heuristic, not verified YOLOv8. |
| 10 | **Concealed-damage flags** | hidden_mold or similar concealed damage flags linked to damage regions | `benchmark/checkpoint3_after/report.json` (16 concealed_flags) | **PARTIALLY EVIDENCED** | concealed_flags present in CP3 LiDAR run with hidden_mold rule. Current single_room_output video run has 0 concealed_flags. Heuristic-only detection. |
| 11 | **Scope line items** | Repair scope CSV/JSON with surface, damage class, quantity, unit | `benchmark/checkpoint3_after/report.json` (34 scope_items); `src/output_schema.py` | **PARTIALLY EVIDENCED** | scope_items present in LiDAR CP3 run (water_damage, mold remediation, unit m2). Current single_room_output has 0 scope_items (video tier). |
| 12 | **Confidence intervals** | 95% CI on floor_area, ceiling_height, wall lengths, opening widths | `single_room_output/report.json`; `benchmark/checkpoint3_after/report.json` | **EVIDENCED** | floor_area_ci_m2, ceiling_height_ci_m, length_ci_m per wall, width_ci_m per opening all present. CI methodology documented in technical_report.md S4. |
| 13 | **Three-tier benchmark** | Same physical spaces captured and processed at all three tiers | `benchmark/run_benchmark.py` | **NOT EVIDENCED** | Benchmark harness exists and handles null ground truth safely. Repository contains only LiDAR captures. Photo and Video captures of the same physical spaces do not exist in sample data. |
| 14 | **Laser / tape ground truth** | Physical measurements for at least one room to score accuracy gates | `benchmark/ground_truth.json` | **NOT EVIDENCED** | ground_truth.json is a null template. All fields (ceiling_height_m, floor_area_m2, walls, openings) are explicitly null for all three datasets. No physical measurements exist anywhere in the repository. |
| 15 | **Opening-width gate** | Opening widths within +/-2cm of measured ground truth (85th percentile) | `benchmark/run_benchmark.py`; `benchmark/ground_truth.json` | **NOT EVIDENCED** | Gate logic implemented and reports not_scored when ground truth is null. Cannot score -- ground truth absent. 7 conservative opening detections exist in CP3 LiDAR run but are unverified. |
| 16 | **Ceiling-height gate** | Ceiling height within +/-1.5cm of measured ground truth | `benchmark/run_benchmark.py`; `benchmark/ground_truth.json` | **NOT EVIDENCED** | Gate logic implemented and reports not_scored. Reconstructed ceiling height 2.127 +/- 0.009 m is self-consistent across two captures (delta 0.001 m) but cannot be scored without physical measurement. |
| 17 | **Repeatability gate** | Two captures of same room at same tier within +/-1cm / +/-0.5% per wall | `benchmark/checkpoint3_after/report.json`; `single_room_output/report.json` | **PARTIALLY EVIDENCED** | Ceiling height consistent to 0.001 m across single_room and single_scan_with_ceiling (same building). Full formal repeatability of identical room boundary across two distinct captures at identical tier not demonstrated. |
| 18 | **Drift accountability + on/off ablation** | Quantified drift report; result with and without drift correction | `single_room_output/report.json` (drift_audit block); `docs/technical_report.md` S3; `benchmark/diff/comparison.json` | **PARTIALLY EVIDENCED** | drift_audit block in output schema documents open-loop status, wall residuals (~2.87 cm avg), loop_closure_status: not_implemented. No executable ablation toggle (--no-drift-correction flag) exists. |
| 19 | **Photo whole-property stitch** | Photo tier stitching multiple rooms into unified property plan | `src/tiers/photo_processor.py`, `stitch.py` | **NOT EVIDENCED** | Photo processor code exists. No photo captures of multiple rooms exist in sample data. No whole-property photo stitch output exists in the repository. |
| 20 | **Photo accuracy gate** | Photo-tier wall error <= +/-8% vs physical ground truth | `benchmark/run_benchmark.py` | **NOT EVIDENCED** | Gate reports not_scored. No photo-tier captures of benchmark spaces. No physical ground truth. |
| 21 | **Video accuracy gate** | Video-tier wall error <= +/-3-5% vs physical ground truth | `benchmark/run_benchmark.py` | **NOT EVIDENCED** | Gate reports not_scored. No physical ground truth. Video tier runs and produces output schema but metric accuracy is unverified. |
| 22 | **Calibration documented at every tier** | Intrinsic calibration methodology documented for LiDAR, Video, Photo | `docs/technical_report.md` S5; `single_room_output/report.json` (assumptions block) | **EVIDENCED** | LiDAR: per-frame fx/fy/cx/cy from odometry.csv with scaling assumption documented. Video: assumed intrinsics with scale from vertical extent. Photo: DUSt3R self-calibrating, COLMAP SIMPLE_RADIAL. All assumptions tagged in output schema. |
| 23 | **Consumer-app head-to-head comparison** | Pipeline vs Polycam / Magicplan on >= 2 rooms, pipeline ties or beats >= 70% dims | -- | **NOT EVIDENCED** | No third-party app export exists in the repository. No comparison artifacts. This is the most costly missing deliverable. |
| 24 | **Fix loop documented** | Identify worst gate, root cause, ship fix, before/after evidence | `docs/fix_loop.md`; `benchmark/checkpoint3_before/`; `benchmark/checkpoint3_after/`; `benchmark/diff/comparison.json` | **EVIDENCED** | Full 9-section fix loop: trajectory hull contamination to wall intersection polygon. Before: 20.77 m2 area, 18 walls, 24 openings, infinite lines. After: 5.0 m2 area, 7 walls, clean orthogonal azimuths, Liang-Barsky clipped segments. Committed in 4b9b171 and 776e9b1. |
| 25 | **Process / commit evidence** | Git history demonstrating iterative development and checkpoints | `git log` (commits ab12b54 to fa2d8d9, 7 commits total) | **EVIDENCED** | 7 commits from initial skeleton to final documentation. Checkpoint artifacts (before/after) committed with descriptive messages. |
| 26 | **Reproduction bundle** | Single command to reproduce results from sample data | `README.md` (Quickstart); `run.py`; `requirements.txt` | **EVIDENCED** | `python run.py --input single_room --tier lidar` documented. Requirements tracked. pip install -r requirements.txt is sufficient. |
| 27 | **Benchmark report** | Machine-readable JSON + human-readable markdown benchmark results | `benchmark/results/benchmark_report.json`; `benchmark/results/benchmark_report.md` | **PARTIALLY EVIDENCED** | Benchmark harness generates both files. All accuracy gates report not_scored due to null ground truth. Schema validity, test suite, and fix loop gates are EVIDENCED. |
| 28 | **Technical report** | Architecture, tier design, drift handling, error budget, calibration | `docs/technical_report.md` | **EVIDENCED** | 6-section report: architecture diagram, tier design with device matrix, drift accountability, error budget table, calibration analysis per tier, known failure modes, fix loop evidence summary. |
| 29 | **Raw benchmark data** | Raw per-dataset output artifacts (report.json, floor_plan.png, etc.) | `benchmark/before/`; `benchmark/after/`; `benchmark/checkpoint3_before/`; `benchmark/checkpoint3_after/` | **EVIDENCED** | Before/after checkpoint artifacts with report.json, floor_plan.png, debug panels committed. diff/comparison.json documents CP2 and CP3 transitions. |
| 30 | **Walk-in readiness** | Cold run on an unseen capture in <= 15 min with documented steps | `README.md` Quickstart; `run.py` CLI | **PARTIALLY EVIDENCED** | README documents a cold-start in <15 min. run.py is the single-entry-point CLI. Model weights (YOLOv8, DUSt3R) require a separate download step not verified on a clean machine. |

---

## Current Blocking Evidence Gaps

The following represent genuinely missing or unresolvable requirements given the current state of the repository:

1. **Physical Ground Truth Absent (Req. 14)** -- `benchmark/ground_truth.json` is entirely null. This blocks all absolute accuracy gate scoring: ceiling-height gate (Req. 16), opening-width gate (Req. 15), photo accuracy gate (Req. 20), and video accuracy gate (Req. 21). Requires physical tape/laser measurements of at least one benchmark room.

2. **Consumer-App Head-to-Head Comparison Absent (Req. 23)** -- No Polycam, Magicplan, or any other consumer-app export exists in the repository. This is an assignment-required comparative deliverable (Part 3) with no substitute. Requires capturing the same space with a competitor app and comparing dimensions.

3. **Three-Tier Same-Space Capture Missing (Req. 13)** -- The sample data contains only LiDAR captures. The assignment requires Photo and Video captures of the same physical spaces as the LiDAR captures. Current Photo/Video tier evidence is limited to heuristic skeleton outputs on different data.

4. **Photo Whole-Property Stitch Not Executed (Req. 19)** -- No multi-room photo captures exist. The stitching code exists but has not been exercised on any data.

5. **Drift Ablation Toggle Not Implemented (Req. 18)** -- The drift_audit output block documents the open-loop state, but there is no --no-drift-correction execution flag to demonstrate the "off" state required for a formal ablation comparison.
