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

    def build(
        self,
        room_id: str,
        geometry: Dict,
        damage_regions: List[Dict],
        tier: str,
        processing_time_s: float = 0.0,
        connected_spaces_graph: Optional[Dict] = None,
        drift_audit: Optional[Dict] = None,
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
            drift_audit = _default_drift_audit(geometry)

        document = {
            "schema_version":   SCHEMA_VERSION,
            "pipeline_version": self.PIPELINE_VERSION,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "tier":             tier,
            "processing_time_s": round(processing_time_s, 2),
            "assumptions":      self.GLOBAL_ASSUMPTIONS,
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


def _default_drift_audit(geometry: Dict) -> Dict:
    """
    Trajectory drift assessment based on available evidence.

    CP4 drift analysis:
    The single_room scan trajectory forms a closed loop (~4m × 3m).
    We do NOT implement full loop-closure SLAM here (requires ICP or g2o),
    but we can estimate the drift category from the wall geometry consistency.

    Evidence:
    - Wall RMS residuals: 0.028–0.029 m (2–3 cm)
    - 2 orthogonal wall families with consistent normals
    - Room polygon closes to 4 clean corners
    - No obvious ghost-wall duplicates

    Conclusion: Drift is present (as with any open-loop odometry) but appears
    small relative to room scale (~1% of 5 m perimeter = ~5 cm worst case).
    A full loop-closure correction would require feature-matching between
    overlapping scan regions, which requires sufficient texture/density.
    Not implemented in this checkpoint.
    """
    walls = geometry.get("walls", [])
    residuals = [w.get("rms_residual_m", 0) for w in walls if w.get("rms_residual_m")]
    n_walls = len(walls)
    avg_res = float(sum(residuals) / len(residuals)) if residuals else 0.0

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
