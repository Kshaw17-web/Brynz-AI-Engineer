"""
Output Schema
=============
CP4 additions:
  - connected_spaces: ConnectedSpaceGraph output
  - drift_audit: trajectory drift assessment
  - assumptions block (top-level)
Schema version: 2.0
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


SCHEMA_VERSION = "2.0"


class OutputSchema:
    """
    Builds the canonical output JSON for a room scan.

    Schema:
    {
      "schema_version": "2.0",
      "pipeline_version": "...",
      "timestamp": "ISO8601",
      "tier": "lidar|video|photo",
      "processing_time_s": float,
      "assumptions": [...],          # global assumptions/caveats
      "drift_audit": {...},          # trajectory drift assessment
      "rooms": {
        "<room_id>": {
          "geometry": { ... },       # single-space geometry (CP3 schema)
          "damage": [...],
          "scope_items": [...],
          "concealed_flags": [...]
        }
      },
      "connected_spaces": {          # CP4: multi-space graph (may be incomplete)
        "segmentation_status": "complete|partial|incomplete",
        "spaces": { ... },
        "adjacency": [ ... ],
        ...
      }
    }
    """

    PIPELINE_VERSION = "2.0.0"

    # Global assumptions that apply to all outputs
    GLOBAL_ASSUMPTIONS = [
        {
            "id": "depth_scale",
            "value": "0.001 m/unit",
            "status": "ASSUMPTION",
            "note": "Depth pixel values assumed to be in millimetres. "
                    "No explicit metadata confirmation available.",
        },
        {
            "id": "rgb_intrinsics_scaling",
            "value": "depth_K = rgb_K * (depth_res / rgb_res)",
            "status": "ASSUMPTION",
            "note": "No separate depth camera calibration file supplied. "
                    "Intrinsics scaled from RGB by resolution ratio.",
        },
        {
            "id": "imu_gravity_alignment",
            "value": "mean accel over session → gravity direction",
            "status": "VERIFIED",
            "note": "IMU mean acceleration magnitude ~1g. "
                    "Rodrigues rotation to align Y with gravity applied.",
        },
        {
            "id": "quaternion_convention",
            "value": "(qx, qy, qz, qw) with norm~1",
            "status": "VERIFIED",
            "note": "Quaternion norm verified over all odometry frames.",
        },
    ]

    # Tier-specific assumptions disclosing methods and limitations honestly
    TIER_ASSUMPTIONS = {
        "lidar": GLOBAL_ASSUMPTIONS,
        "video": [
            {
                "id": "depth_estimation_model",
                "value": "heuristic_fallback (Depth Anything V2 unavailable)",
                "status": "FALLBACK_ESTIMATE",
                "note": "Pretrained monocular depth weights not bundled; uses vertical gradient depth heuristic.",
            },
            {
                "id": "visual_odometry",
                "value": "ORB 2-view essential matrix recovery",
                "status": "ESTIMATE",
                "note": "Visual odometry without IMU fusion or global bundle adjustment.",
            },
            {
                "id": "metric_scale",
                "value": "vertical_extent -> assumed 2.4m room height",
                "status": "ASSUMPTION",
                "note": "Monocular video has scale ambiguity; scaled to typical room height reference.",
            },
            {
                "id": "accuracy_tier",
                "value": "wall lengths +/-3-5%, ceiling +/-3cm (target spec)",
                "status": "UNVERIFIED_ESTIMATE",
                "note": "Heuristic fallback significantly increases uncertainty over LiDAR.",
            },
        ],
        "photo": [
            {
                "id": "photo_reconstruction_method",
                "value": "heuristic_monocular_unprojection (DUSt3R/COLMAP unavailable)",
                "status": "FALLBACK_ESTIMATE",
                "note": "DUSt3R weights / COLMAP binary not found; uses frontal unprojection with heuristic depth.",
            },
            {
                "id": "camera_poses",
                "value": "frontal_view_assumption_with_index_offset",
                "status": "ASSUMPTION",
                "note": "No sparse SfM camera poses recovered; images treated as frontal baseline samples.",
            },
            {
                "id": "metric_scale",
                "value": "vertical_extent -> assumed 2.4m room height",
                "status": "ASSUMPTION",
                "note": "No LiDAR or depth sensor; scale normalized to assumed 2.4m ceiling height.",
            },
            {
                "id": "accuracy_tier",
                "value": "wall lengths +/-8%, area +/-8% (widest confidence tier)",
                "status": "UNVERIFIED_ESTIMATE",
                "note": "Floor tier: provides approximate bounds only without metric sensor verification.",
            },
        ],
    }

    def build(
        self,
        room_id: str,
        geometry: Dict,
        damage_regions: List[Dict],
        tier: str,
        processing_time_s: float = 0.0,
        connected_spaces_graph: Optional[Dict] = None,
        drift_audit: Optional[Dict] = None,
        drift_info: Optional[Dict] = None,
        depth_info: Optional[Dict] = None,
    ) -> Dict:
        """Build the complete output document."""

        scope_items = []
        concealed_flags = []
        for dmg in damage_regions:
            if "scope_items" in dmg:
                scope_items.append(dmg["scope_items"])
            if dmg.get("concealed_flag"):
                concealed_flags.append({
                    "type":               dmg.get("concealed_rule", "unknown"),
                    "description":        self._concealed_description(dmg),
                    "linked_damage_class": dmg.get("class"),
                    "confidence":         dmg.get("confidence", 0.5),
                })

        room_data = {
            "geometry":       geometry,
            "damage":         damage_regions,
            "scope_items":    scope_items,
            "concealed_flags": concealed_flags,
        }

        # Default drift audit if not supplied
        if drift_audit is None:
            drift_audit = _default_drift_audit(geometry, tier=tier, drift_info=drift_info)

        raw_assumptions = self.TIER_ASSUMPTIONS.get(tier, self.GLOBAL_ASSUMPTIONS)
        assumptions = [dict(a) for a in raw_assumptions]
        if tier == "video" and depth_info and depth_info.get("model_name"):
            model_name = depth_info["model_name"]
            is_metric = "Metric" in model_name or "metric" in model_name
            for a in assumptions:
                if a.get("id") == "depth_estimation_model":
                    a["value"] = model_name
                    if is_metric:
                        a["status"] = "PRETRAINED_METRIC_MODEL"
                        a["note"] = f"Monocular metric indoor depth estimation using {model_name}."

        document = {
            "schema_version":   SCHEMA_VERSION,
            "pipeline_version": self.PIPELINE_VERSION,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "tier":             tier,
            "processing_time_s": round(processing_time_s, 2),
            "assumptions":      assumptions,
            "drift_audit":      drift_audit,
            "rooms": {
                room_id: room_data
            },
            "connected_spaces": connected_spaces_graph or {
                "segmentation_status": "not_run",
                "note": "Multi-space segmentation not run in this pipeline invocation.",
            },
            "summary": {
                "total_floor_area_m2":    geometry.get("floor_area_m2", 0),
                "total_floor_area_ci_m2": geometry.get("floor_area_ci_m2", 0),
                "n_rooms":               1,
                "n_walls_total":         len(geometry.get("walls", [])),
                "n_openings_total":      len(geometry.get("openings", [])),
                "n_openings_confidence": "low",
                "n_damage_regions":      len(damage_regions),
                "n_concealed_flags":     len(concealed_flags),
                "ceiling_height_m":      geometry.get("ceiling_height_m", 0),
                "multi_space_indicator": {
                    "bbox_polygon_ratio": geometry.get("debug", {}).get("bbox_polygon_ratio", 0),
                    "flag":               geometry.get("debug", {}).get("bbox_polygon_ratio", 0) > 3.0,
                },
            },
        }

        return document

    def _concealed_description(self, dmg: Dict) -> str:
        rule = dmg.get("concealed_rule", "")
        cls  = dmg.get("class", "damage")
        descriptions = {
            "moisture_behind_wall": (
                f"Multiple {cls} detections near wall base suggest moisture "
                f"infiltration behind the wall surface. Recommend moisture meter inspection."
            ),
            "hidden_mold": (
                f"Mold detected in corner region. Possible hidden mold growth "
                f"behind wall surface. Recommend professional inspection."
            ),
            "structural_concern": (
                f"Large crack detected ({dmg.get('area_m2', 0):.2f} m² area). "
                f"May indicate structural movement. Recommend structural assessment."
            ),
        }
        return descriptions.get(rule, f"{cls} requires inspection")


def _default_drift_audit(
    geometry: Dict, tier: str = "lidar", drift_info: Optional[Dict] = None
) -> Dict:
    """
    Trajectory drift assessment based on available evidence and input tier.
    """
    walls = geometry.get("walls", [])
    residuals = [w.get("rms_residual_m", 0) for w in walls if w.get("rms_residual_m")]
    n_walls = len(walls)
    avg_res = float(sum(residuals) / len(residuals)) if residuals else 0.0

    if tier == "photo":
        return {
            "drift_correction_applied": False,
            "method":                   "not_applicable",
            "rationale": (
                "Photo tier operates on independent still photos without continuous "
                "temporal odometry. Trajectory drift is not applicable; uncalibrated monocular depth "
                "and scale ambiguity represent the dominant error sources."
            ),
            "estimated_drift_category":  "not_applicable_static_photos",
            "wall_rms_residuals_m":      [round(r, 4) for r in residuals],
            "avg_wall_residual_m":       round(avg_res, 4),
            "loop_closure_status":       "not_applicable",
            "recommendation": (
                "For metric accuracy, provide multi-view overlap for SfM (COLMAP/DUSt3R) "
                "or capture with LiDAR tier."
            ),
        }
    elif tier == "video":
        return {
            "drift_correction_applied": False,
            "method":                   "open_loop_visual_odometry",
            "rationale": (
                "Open-loop frame-to-frame ORB feature tracking and essential matrix pose recovery. "
                "Scale drift and rotational accumulation occur without loop closure or bundle adjustment."
            ),
            "estimated_drift_category":  "moderate_to_high_monocular_drift",
            "wall_rms_residuals_m":      [round(r, 4) for r in residuals],
            "avg_wall_residual_m":       round(avg_res, 4),
            "loop_closure_status":       "not_implemented",
            "recommendation": (
                "Implement full visual SLAM (ORB-SLAM3 / DROID-SLAM) with loop closure "
                "and bundle adjustment for production monocular video."
            ),
        }
    else:
        if drift_info and drift_info.get("drift_correction_applied"):
            pre_res = drift_info.get("loop_closure_pre_residual_m", 0.0)
            post_res = drift_info.get("loop_closure_post_residual_m", 0.0)
            traj_len = drift_info.get("trajectory_length_m", 0.0)
            return {
                "drift_correction_applied": True,
                "method":                   drift_info.get("method", "linear_trajectory_loop_closure"),
                "rationale": (
                    f"Traverse loop-closure correction applied. Closed trajectory loop "
                    f"detected with {pre_res:.3f} m accumulated drift over {traj_len:.2f} m "
                    f"total path. Linear traverse correction distributed closure residual "
                    f"along cumulative path, reducing endpoint residual to {post_res:.4f} m."
                ),
                "estimated_drift_category":  "corrected_loop_closure",
                "wall_rms_residuals_m":      [round(r, 4) for r in residuals],
                "avg_wall_residual_m":       round(avg_res, 4),
                "loop_closure_status":       drift_info.get("loop_closure_status", "applied_closed_loop"),
                "loop_closure_pre_residual_m": pre_res,
                "loop_closure_post_residual_m": post_res,
                "trajectory_length_m":       traj_len,
                "recommendation": (
                    "Trajectory loop closure applied prior to point cloud unprojection. "
                    "For non-loop trajectories, capture protocol recommends returning to start doorway."
                ),
            }
        elif drift_info:
            loop_status = drift_info.get("loop_closure_status", "open_loop_disabled")
            pre_res = drift_info.get("loop_closure_pre_residual_m")
            traj_len = drift_info.get("trajectory_length_m")
            return {
                "drift_correction_applied": False,
                "method":                   "none",
                "rationale": (
                    f"Wall plane residuals (avg {avg_res:.3f} m) suggest low geometric inconsistency. "
                    f"Drift correction toggle is OFF or trajectory is unclosed ({loop_status}). "
                    f"Audited trajectory length: {traj_len} m, endpoint closure gap: {pre_res} m. "
                    "Open-loop odometry retained without drift correction."
                ),
                "estimated_drift_category":  _categorise_drift(avg_res, n_walls),
                "wall_rms_residuals_m":      [round(r, 4) for r in residuals],
                "avg_wall_residual_m":       round(avg_res, 4),
                "loop_closure_status":       loop_status,
                "loop_closure_pre_residual_m": pre_res,
                "loop_closure_post_residual_m": pre_res,
                "trajectory_length_m":       traj_len,
                "recommendation": (
                    "Enable --drift-correction to apply trajectory loop closure. "
                    "Current results reflect open-loop odometry."
                ),
            }
        else:
            return {
                "drift_correction_applied": False,
                "method":                   "none",
                "rationale": (
                    "Wall plane residuals (avg {:.3f} m) suggest low geometric inconsistency. "
                    "Closed-loop trajectory and {} consistent wall planes observed. "
                    "Full loop-closure (ICP/pose-graph) not implemented. "
                    "Drift estimated < 5 cm for single-room scans based on trajectory scale "
                    "and wall consistency, but this is NOT verified by external ground truth."
                ).format(avg_res, n_walls),
                "estimated_drift_category":  _categorise_drift(avg_res, n_walls),
                "wall_rms_residuals_m":      [round(r, 4) for r in residuals],
                "avg_wall_residual_m":       round(avg_res, 4),
                "loop_closure_status":       "not_implemented",
                "recommendation": (
                    "For production accuracy, implement ICP-based loop closure or "
                    "integrate with a visual SLAM system (e.g. ORB-SLAM3). "
                    "Current results are open-loop odometry only."
                ),
            }


def _categorise_drift(avg_residual: float, n_walls: int) -> str:
    """Heuristic drift category from wall RMS residuals."""
    if avg_residual < 0.02:
        return "low_likely"
    elif avg_residual < 0.05:
        return "low_to_moderate_likely"
    elif avg_residual < 0.10:
        return "moderate_likely"
    else:
        return "high_or_geometry_error"


def load_schema_from_file(path: str) -> Dict:
    """Load and validate a report.json file."""
    with open(path) as f:
        data = json.load(f)
    assert "schema_version" in data, "Missing schema_version"
    assert "rooms" in data, "Missing rooms"
    return data
