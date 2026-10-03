# Brynz Room Scanner Pipeline

> **Applied AI Engineering Assessment — Room Scanning Pipeline**
> 
> Converts iPhone LiDAR/video/photo scans into dimensioned floor plans, damage reports, and repair scopes. One command per capture.

---

## Quickstart (< 15 minutes on a clean machine)

```bash
# 1. Clone repo
git clone https://github.com/<your-repo>/brynz-scanner.git
cd brynz-scanner

# 2. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 3. Download model weights
python scripts/download_models.py

# 4. Run on a scan (LiDAR tier — your sample data)
python run.py --input single_room --tier lidar

# Outputs in: single_room_output/
#   floor_plan.png       — rendered dimensioned floor plan
#   report.json          — full measurements + damage + scope
#   floor_plan.svg       — vector floor plan
#   scope_items.csv      — repair scope line items
```

---

## Capture Route (Route 2 — Stock App)

**App**: [Record3D](https://record3d.app) (App Store, free)

**Supported Hardware**: iPhone 12 Pro, 13 Pro, 14 Pro, 15 Pro (Pro/Pro Max models with LiDAR)

### One-Page Capture Protocol

See [`docs/capture_protocol.md`](docs/capture_protocol.md)

---

## Input Tiers

| Tier | Input | Command | Wall Accuracy | Ceiling Accuracy |
|------|-------|---------|---------------|-----------------|
| **LiDAR** | Record3D scan (depth + poses) | `--tier lidar` | ±1% | ±1.5cm |
| **Video** | iPhone walkthrough video | `--tier video` | ±3% | ±3cm |
| **Photo** | 2–8 room photos | `--tier photo` | ±8% | ±5cm |

---

## Repository Structure

```
├── run.py                    # Main entry point
├── src/
│   ├── pipeline.py           # Orchestrator
│   ├── lidar_processor.py    # LiDAR depth + pose → point cloud
│   ├── geometry.py           # RANSAC plane detection → room geometry
│   ├── floor_plan.py         # Dimensioned floor plan renderer
│   ├── damage_detector.py    # YOLOv8 damage detection
│   ├── stitcher.py           # Multi-room stitching
│   ├── output_schema.py      # JSON output schema
│   └── tiers/
│       ├── video_processor.py  # Video tier (monocular depth + VO)
│       └── photo_processor.py  # Photo tier (DUSt3R / COLMAP SfM)
├── benchmark/
│   ├── run_benchmark.py      # Benchmark harness
│   └── ground_truth.json     # Ground truth measurements (fill in!)
├── docs/
│   ├── capture_protocol.md   # 1-page capture protocol
│   └── technical_report.md   # 6-page technical report
├── scripts/
│   └── download_models.py    # Download model weights
└── requirements.txt
```

---

## Output Schema

```json
{
  "schema_version": "1.0",
  "tier": "lidar",
  "rooms": {
    "room_001": {
      "geometry": {
        "floor_area_m2": 14.23,
        "floor_area_ci_m2": 0.28,
        "ceiling_height_m": 2.485,
        "ceiling_height_ci_m": 0.008,
        "walls": [
          {"id": "wall_00", "length_m": 4.12, "length_ci_m": 0.015, ...}
        ],
        "openings": [
          {"id": "opening_00", "type": "door", "width_m": 0.91, "width_ci_m": 0.025, ...}
        ]
      },
      "damage": [...],
      "scope_items": [...],
      "concealed_flags": [...]
    }
  }
}
```

---

## Multi-Room Stitching

```bash
# Process each room
python run.py --input room_001 --room-id room_001 --output stitched_output
python run.py --input room_002 --room-id room_002 --output stitched_output
python run.py --input room_003 --room-id room_003 --output stitched_output

# Stitch all rooms
python stitch.py --input stitched_output --output stitched_output/final
```

---

## Benchmark

```bash
# Fill in ground truth first (laser measurements)
# Edit: benchmark/ground_truth.json

python benchmark/run_benchmark.py \
    --ground-truth benchmark/ground_truth.json \
    --datasets . \
    --tier lidar video photo
```

> **Benchmark Status**: See [`docs/benchmark_audit.md`](docs/benchmark_audit.md) for the benchmark compliance audit, and [`docs/fix_loop.md`](docs/fix_loop.md) for the reproducible geometry fix loop. Physical accuracy gates remain unscored because `benchmark/ground_truth.json` contains no physical laser/tape measurements.

---

## Device Matrix

| Device | LiDAR Tier | Video Tier | Photo Tier |
|--------|-----------|-----------|-----------|
| iPhone 12 Pro / Pro Max | ✅ | ✅ | ✅ |
| iPhone 13 Pro / Pro Max | ✅ | ✅ | ✅ |
| iPhone 14 Pro / Pro Max | ✅ | ✅ | ✅ |
| iPhone 15 / 15 Plus | ❌ (no LiDAR) | ✅ | ✅ |
| iPhone 15 Pro / Pro Max | ✅ | ✅ | ✅ |
| Any iPhone 15+ (non-Pro) | ❌ | ✅ | ✅ |

**Target accuracy specifications (design goals; physical ground truth unscored):**

| Tier | Wall lengths | Ceiling height | Floor area | Opening widths |
|------|-------------|----------------|------------|----------------|
| LiDAR | ±1–2cm | ±1.5cm | ±2% | ±2cm (85%@2cm) |
| Video | ±3–5% | ±3cm | ±5% | ±5cm |
| Photo | ±5–8% | ±5cm | ±8% | ±8cm |

---

## Technical Report

See [`docs/technical_report.md`](docs/technical_report.md)

---

## Known Limitations

- **Mirrors / glass**: LiDAR returns are unreliable. Pipeline clips depth to <0.1 confidence.
- **Low light**: Depth accuracy degrades. Recommend good lighting during capture.
- **Large rooms (>10m wall)**: ARKit odometry is open-loop. Drift is audited via wall residuals (~2.8 cm avg); full pose-graph loop closure is planned for production.
- **Photo tier scale**: Monocular scale resolved using vertical extent assumption (2.4m default). Provide `--reference-height` for calibrated scaling.
