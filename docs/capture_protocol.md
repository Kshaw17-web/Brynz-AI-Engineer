# Capture Protocol — Brynz Room Scanner

**Version 1.0 | For non-engineers**

---

## What to Install

1. Open the App Store on your iPhone (requires iPhone 12 Pro or newer with LiDAR)
2. Search for **"Record3D"** — by Marek Šmolík (free download)
3. Install and open the app. Grant camera access when prompted.

---

## Equipment

- ✅ iPhone 12 Pro, 13 Pro, 14 Pro, or 15 Pro (any Pro/Pro Max model)
- ✅ Record3D app (free on App Store)
- ✅ Good lighting in the room (turn on all lights)
- ❌ Do NOT use in very dark rooms — it affects accuracy

---

## How to Walk

### Before you start:
- Clear the center of the room if possible (move chairs, bags)
- Turn on all lights
- Open doors and windows you want measured

### Recording:
1. Hold the phone **vertically** at chest height (~1.2m from floor)
2. Tap the **red circle** button to start recording
3. Walk slowly around the **perimeter** of the room, staying 0.5–1m from the walls
4. Move the phone to also point **at the floor** (tilt down 45°) during the walk
5. Move the phone to also point **at the ceiling** (tilt up 45°) — scan at least once around
6. Walk at a pace of roughly **1 step per second** (no rushing)
7. After one complete loop, do a **second loop** at a slightly different path
8. Tap the **square button** to stop recording

### For doorways/openings:
- When you reach each door or window, stop and slowly scan the opening from left to right
- Hold steady for 2-3 seconds facing the opening

### How long: 
- Small room (< 15m²): 30–60 seconds
- Medium room (15–30m²): 60–90 seconds
- Large room (> 30m²): 90–120 seconds

### What to avoid:
- ❌ Moving too fast (causes motion blur and missed depth)
- ❌ Pointing directly at mirrors (LiDAR bounces off mirrors incorrectly)
- ❌ Pointing at windows with direct sunlight (saturates the sensor)
- ❌ Recording from one spot only (must walk around the room)

---

## Exporting the Files

1. After recording, tap the recording in the app's file list
2. Tap **"Share"** → **"Export to Files"**
3. Choose a folder name like "living_room" and save
4. The exported folder contains:
   - `rgb.mp4` — video recording
   - `depth/` — folder of depth images
   - `confidence/` — folder of confidence images  
   - `odometry.csv` — camera positions
   - `imu.csv` — motion sensor data
   - `camera_matrix.csv` — camera settings
5. Transfer this folder to your computer via AirDrop, iCloud Drive, or USB cable

---

## Handing Files to the Pipeline

Once you have the exported folder on your computer:

```bash
python run.py --input path/to/living_room --tier lidar
```

That's it. Results appear in `living_room_output/` in about 60–120 seconds.

---

## For Multi-Room Scans

Scan each room separately, naming the export folders by room:
- `living_room/`
- `bedroom_1/`
- `kitchen/`
- etc.

Then stitch them together:
```bash
python stitch.py --rooms living_room bedroom_1 kitchen --output whole_property
```
