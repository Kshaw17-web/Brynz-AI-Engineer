"""
Output Schema
=============
Defines and builds the JSON output schema for the Brynz pipeline.

Schema version: 1.0
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional


SCHEMA_VERSION = "1.0"


class OutputSchema:
    """
    Builds the canonical output JSON for a room scan.
    
    Schema:
    {
      "schema_version": "1.0",
      "pipeline_version": "1.0.0",
      "timestamp": "ISO8601",
      "tier": "lidar|video|photo",
      "processing_time_s": float,
      "rooms": {
        "<room_id>": {
          "geometry": {
            "floor_area_m2": float,
            "floor_area_ci_m2": float,
            "ceiling_height_m": float,
            "ceiling_height_ci_m": float,
            "footprint_polygon": [[x,z], ...],
            "walls": [WallObject, ...],
            "openings": [OpeningObject, ...]
          },
          "damage": [DamageObject, ...],
          "scope_items": [ScopeItem, ...],
          "concealed_flags": [ConcealedFlag, ...]
        }
      }
    }
    """
    
    PIPELINE_VERSION = "1.0.0"

    def build(
        self,
        room_id: str,
        geometry: Dict,
        damage_regions: List[Dict],
        tier: str,
        processing_time_s: float = 0.0,
    ) -> Dict:
        """Build the complete output document."""
        
        # Extract scope items and concealed flags from damage
        scope_items = []
        concealed_flags = []
        for dmg in damage_regions:
            if "scope_items" in dmg:
                scope_items.append(dmg["scope_items"])
            if dmg.get("concealed_flag"):
                concealed_flags.append({
                    "type": dmg.get("concealed_rule", "unknown"),
                    "description": self._concealed_description(dmg),
                    "linked_damage_class": dmg.get("class"),
                    "confidence": dmg.get("confidence", 0.5),
                })
        
        room_data = {
            "geometry": geometry,
            "damage": damage_regions,
            "scope_items": scope_items,
            "concealed_flags": concealed_flags,
        }
        
        document = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": self.PIPELINE_VERSION,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tier": tier,
            "processing_time_s": round(processing_time_s, 2),
            "rooms": {
                room_id: room_data
            },
            "summary": {
                "total_floor_area_m2": geometry.get("floor_area_m2", 0),
                "total_floor_area_ci_m2": geometry.get("floor_area_ci_m2", 0),
                "n_rooms": 1,
                "n_walls_total": len(geometry.get("walls", [])),
                "n_openings_total": len(geometry.get("openings", [])),
                "n_damage_regions": len(damage_regions),
                "n_concealed_flags": len(concealed_flags),
                "ceiling_height_m": geometry.get("ceiling_height_m", 0),
            }
        }
        
        return document

    def _concealed_description(self, dmg: Dict) -> str:
        rule = dmg.get("concealed_rule", "")
        cls = dmg.get("class", "damage")
        
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


def load_schema_from_file(path: str) -> Dict:
    """Load and validate a report.json file."""
    with open(path) as f:
        data = json.load(f)
    
    # Basic validation
    assert "schema_version" in data, "Missing schema_version"
    assert "rooms" in data, "Missing rooms"
    
    return data
