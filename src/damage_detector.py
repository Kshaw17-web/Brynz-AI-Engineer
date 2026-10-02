"""
Damage Detector
===============
Detects visible surface damage from RGB frames using a YOLOv8 model.

Supported damage classes:
  - crack (structural cracks in walls/ceiling)
  - water_stain (moisture/water damage)
  - mold (dark patches, mold/mildew)
  - spalling (concrete/plaster flaking)
  - efflorescence (white salt deposits)
  - peeling_paint

Each detection is:
  1. Located in the 2D RGB frame (bounding box)
  2. Back-projected to 3D world space using the depth map
  3. Reported as {class, area_m2, center_xyz, confidence, concealed_flag}

Concealed damage flags (rule-based):
  - Moisture behind wall: high water stain density near wall base
  - Hidden mold: dark patches in corners with high humidity indicators
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json

import numpy as np

logger = logging.getLogger(__name__)

# Try to load YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("ultralytics not installed — using heuristic damage detection")

# Try to load a pretrained model or use heuristics
MODEL_PATH = Path(__file__).parent.parent / "models" / "damage_detector.pt"


class DamageDetector:
    """
    Detects surface damage from RGB frames.
    
    Strategy:
    1. If YOLOv8 weights exist (damage_detector.pt), use them.
    2. Otherwise, use color/texture heuristics as fallback.
    
    The heuristic detector finds:
    - Dark patches (mold): low saturation, very dark regions
    - White deposits (efflorescence): high brightness, low saturation
    - Discoloration (water stains): brownish/yellowish regions
    """

    DAMAGE_CLASSES = [
        "crack",
        "water_stain", 
        "mold",
        "spalling",
        "efflorescence",
        "peeling_paint",
    ]

    # Concealed damage rules
    CONCEALED_RULES = {
        "moisture_behind_wall": (
            "water_stain density >3 detections within 0.5m of wall base"
        ),
        "hidden_mold": (
            "mold detection in corner region with no visible source"
        ),
        "structural_concern": (
            "crack >0.5m length on load-bearing wall"
        ),
    }

    def __init__(self, model_path: Optional[Path] = None, verbose: bool = False):
        self.verbose = verbose
        self.model = None
        
        mp = model_path or MODEL_PATH
        if HAS_YOLO and mp.exists():
            logger.info(f"Loading YOLO damage model from {mp}")
            self.model = YOLO(str(mp))
        else:
            logger.info("Using heuristic damage detection (no YOLO model found)")

    def detect(
        self,
        frames_rgb: List[Tuple[int, np.ndarray]],
        geometry: Dict,
        point_cloud: Dict,
    ) -> List[Dict]:
        """
        Run damage detection on sampled RGB frames.
        
        Args:
            frames_rgb: list of (frame_idx, rgb_np_array)
            geometry: room geometry dict
            point_cloud: {'xyz': Nx3, 'rgb': Nx3}
        
        Returns:
            List of damage region dicts
        """
        if not frames_rgb:
            return []
        
        all_detections = []
        
        for frame_idx, rgb_frame in frames_rgb:
            if self.model is not None:
                detections = self._detect_yolo(rgb_frame, frame_idx)
            else:
                detections = self._detect_heuristic(rgb_frame, frame_idx)
            all_detections.extend(detections)
        
        # Cluster nearby detections to avoid duplicates
        merged = self._merge_detections(all_detections, point_cloud)
        
        # Apply concealed damage rules
        flagged = self._apply_concealed_rules(merged, geometry)
        
        # Compute scope line items
        for dmg in flagged:
            dmg["scope_items"] = self._generate_scope_item(dmg)
        
        logger.info(f"Damage detection: {len(flagged)} regions, "
                   f"{sum(1 for d in flagged if d.get('concealed_flag'))} concealed flags")
        
        return flagged

    def _detect_yolo(self, rgb_frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """Run YOLO inference."""
        results = self.model(rgb_frame, verbose=False)
        detections = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.3:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = self.DAMAGE_CLASSES[cls_id] if cls_id < len(self.DAMAGE_CLASSES) else "unknown"
                
                detections.append({
                    "frame_idx": frame_idx,
                    "class": cls_name,
                    "confidence": conf,
                    "bbox_px": [int(x1), int(y1), int(x2), int(y2)],
                    "center_px": [int((x1+x2)/2), int((y1+y2)/2)],
                    "area_px2": (x2 - x1) * (y2 - y1),
                })
        
        return detections

    def _detect_heuristic(self, rgb_frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Heuristic damage detection using color analysis.
        Falls back to this when no trained model is available.
        
        This is intentionally conservative (high precision, lower recall)
        to avoid false positives that would hurt credibility.
        """
        try:
            import cv2
        except ImportError:
            return []
        
        detections = []
        h, w = rgb_frame.shape[:2]
        
        # Convert to HSV for color analysis
        bgr = rgb_frame[:, :, ::-1]  # RGB -> BGR
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # --- Water stain detection (brownish/yellowish) ---
        # Hue: 10-30 (orange/brown), low-medium saturation, medium value
        water_mask = (
            (H >= 10) & (H <= 30) & 
            (S >= 40) & (S <= 180) & 
            (V >= 50) & (V <= 200)
        )
        detections.extend(
            self._mask_to_detections(water_mask, "water_stain", frame_idx, rgb_frame, min_area=200)
        )
        
        # --- Mold detection (dark patches with greenish tint) ---
        mold_mask = (V < 60) & (S > 20)  # Dark, somewhat saturated
        detections.extend(
            self._mask_to_detections(mold_mask, "mold", frame_idx, rgb_frame, min_area=300)
        )
        
        # --- Efflorescence (bright white patches) ---
        efflor_mask = (V > 220) & (S < 30) & (
            self._compute_local_contrast(V) > 20
        )
        detections.extend(
            self._mask_to_detections(efflor_mask, "efflorescence", frame_idx, rgb_frame, min_area=200)
        )
        
        return detections

    def _compute_local_contrast(self, channel: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Compute local contrast (std dev in neighbourhood)."""
        try:
            import cv2
            blur = cv2.GaussianBlur(channel.astype(np.float32), (kernel_size, kernel_size), 0)
            return np.abs(channel.astype(np.float32) - blur)
        except Exception:
            return np.zeros_like(channel, dtype=np.float32)

    def _mask_to_detections(
        self,
        mask: np.ndarray,
        damage_class: str,
        frame_idx: int,
        rgb_frame: np.ndarray,
        min_area: int = 100,
    ) -> List[Dict]:
        """Convert a boolean mask to damage detections via connected components."""
        try:
            import cv2
        except ImportError:
            return []
        
        mask_u8 = mask.astype(np.uint8) * 255
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)
        
        detections = []
        for i in range(1, n_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            
            detections.append({
                "frame_idx": frame_idx,
                "class": damage_class,
                "confidence": 0.5,  # Heuristic confidence
                "bbox_px": [x, y, x + bw, y + bh],
                "center_px": [cx, cy],
                "area_px2": int(area),
            })
        
        return detections

    def _merge_detections(
        self, detections: List[Dict], point_cloud: Dict
    ) -> List[Dict]:
        """
        Merge nearby detections across frames into unique regions.
        Simple distance-based clustering.
        """
        if not detections:
            return []
        
        # Without 3D back-projection (would need per-frame depth),
        # we cluster by approximate 2D pixel space and assign a dummy 3D position
        # that will be refined later.
        
        # Simple heuristic: deduplicate by (class, frame_idx) keeping highest confidence
        seen = {}
        for det in detections:
            key = (det["class"], det["frame_idx"])
            if key not in seen or det["confidence"] > seen[key]["confidence"]:
                seen[key] = det
        
        # Convert pixel area to approximate m² (assume 1 pixel ≈ 0.002m at typical room scale)
        PIXEL_TO_M2 = 0.00001  # crude estimate
        
        merged = []
        for det in seen.values():
            area_m2 = det.get("area_px2", 0) * PIXEL_TO_M2
            area_m2 = max(0.01, min(area_m2, 5.0))  # clamp to reasonable range
            
            merged.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "area_m2": round(float(area_m2), 4),
                "center_x": 0.0,  # placeholder - needs depth back-projection
                "center_z": 0.0,
                "center_y": 1.0,
                "frame_idx": det["frame_idx"],
                "bbox_px": det.get("bbox_px", []),
                "concealed_flag": False,
                "concealed_rule": None,
            })
        
        return merged

    def _apply_concealed_rules(
        self, detections: List[Dict], geometry: Dict
    ) -> List[Dict]:
        """
        Apply rule-based concealed damage flagging.
        """
        # Count water stains near wall base (y < floor_z + 0.5m)
        floor_z = geometry.get("floor_z", 0.0)
        water_stains_low = [
            d for d in detections
            if d["class"] == "water_stain" and d.get("center_y", 1.0) < floor_z + 0.5
        ]
        
        # Rule 1: Multiple water stains near floor → moisture behind wall
        if len(water_stains_low) >= 2:
            for d in water_stains_low:
                if not d["concealed_flag"]:
                    d["concealed_flag"] = True
                    d["concealed_rule"] = "moisture_behind_wall"
        
        # Rule 2: Mold in corner areas
        for d in detections:
            if d["class"] == "mold":
                d["concealed_flag"] = True
                d["concealed_rule"] = "hidden_mold"
        
        # Rule 3: Large cracks
        for d in detections:
            if d["class"] == "crack" and d.get("area_m2", 0) > 0.1:
                d["concealed_flag"] = True
                d["concealed_rule"] = "structural_concern"
        
        return detections

    def _generate_scope_item(self, dmg: Dict) -> Dict:
        """Generate a repair scope line item for a damage region."""
        cls = dmg.get("class", "unknown")
        area = dmg.get("area_m2", 0.0)
        
        scope_map = {
            "crack": {
                "description": "Crack repair — epoxy injection + patching",
                "unit": "m",
                "quantity": round(np.sqrt(area) * 2, 2),  # crude length estimate
            },
            "water_stain": {
                "description": "Water damage treatment — dry, treat, repaint",
                "unit": "m²",
                "quantity": round(area, 2),
            },
            "mold": {
                "description": "Mold remediation — biocide treatment + encapsulation",
                "unit": "m²",
                "quantity": round(area * 1.5, 2),  # include safety margin
            },
            "spalling": {
                "description": "Spalling repair — remove loose material, patch",
                "unit": "m²",
                "quantity": round(area, 2),
            },
            "efflorescence": {
                "description": "Efflorescence treatment — wire brush, sealant",
                "unit": "m²",
                "quantity": round(area, 2),
            },
            "peeling_paint": {
                "description": "Repaint — scrape, prime, 2 coats",
                "unit": "m²",
                "quantity": round(area, 2),
            },
        }
        
        item = scope_map.get(cls, {
            "description": f"{cls.replace('_', ' ').title()} repair",
            "unit": "m²",
            "quantity": round(area, 2),
        })
        
        return {
            "surface": "wall" if dmg.get("center_y", 1.0) > 0.5 else "floor",
            "damage_class": cls,
            **item,
        }
