# Patient Positioning Gate

A real-time preprocessing gate for a pose-based pain-evaluation pipeline. It
runs on a live webcam **before** pain evaluation and only lets a frame through
when the patient is well positioned:

1. **Full body** is inside the frame
2. Patient is **facing** the camera (not turned side-on)
3. Patient is standing **upright** (not leaning or tilted)
4. Patient is **looking at** the camera (real head pose, both eyes to the lens)

When these hold for enough consecutive frames, the gate reports `READY` and you
trigger your existing pain pipeline.

## Setup (conda + VS Code)

```bash
conda env create -f environment.yml
conda activate pose-gate
python download_model.py     # one-time: fetches pose + face models
python run.py                # ESC to quit
```

In VS Code, select the `pose-gate` interpreter (Command Palette →
"Python: Select Interpreter") before running.

## Project structure

```
pose_gate_project/
├── environment.yml          # conda environment
├── download_model.py        # one-time model fetch (pose + face bundles)
├── run.py                   # live webcam entry point
├── models/                  # created by download_model.py
└── pose_gate/
    ├── config.py            # landmark indices + all thresholds
    ├── geometry.py          # pure body-angle math, no MediaPipe -> testable
    ├── face.py              # head pose: Face Landmarker + solvePnP -> degrees
    ├── gate.py              # checks + temporal smoothing + feedback
    └── detector.py          # Pose Landmarker wrapper -> plain numpy arrays
```

## The four checks

| Check        | Source                    | Metric                         | Threshold        |
|--------------|---------------------------|--------------------------------|------------------|
| Full body    | pose image landmarks      | visibility + in-frame + fill   | `visibility`, `max_body_fill` |
| Facing       | pose world landmarks      | shoulder-line yaw (deg)        | `max_yaw`        |
| Upright      | pose world landmarks      | shoulder roll + torso tilt     | `max_roll`, `max_torso_tilt` |
| Looking      | face landmarks + solvePnP | head yaw + pitch (deg)         | `max_head_yaw`, `max_head_pitch` |

All body angles come from **world** landmarks (metric, hip-centered) to avoid
the aspect-ratio distortion of taking angles in normalized image space.

## Head pose (face.py)

`face.py` runs MediaPipe Face Landmarker, matches six facial points (nose, chin,
eye corners, mouth corners) to a generic 3D face model, and uses `cv2.solvePnP`
+ Rodrigues + Euler decomposition to recover real head **yaw / pitch / roll in
degrees**. A frontal head reads ~0 on every axis, so `look_at_camera` passes
only when both eyes are actually toward the lens.

Set `use_face_landmarker = False` in `config.py` to fall back to the lighter
pose-only nose/ear proxy (one fewer model, less accurate).

## Why the design is split this way

- `geometry.py` has no camera or MediaPipe dependency, so the math is
  unit-testable in isolation.
- `detector.py` and `face.py` are the only files that touch MediaPipe. Swapping
  a model later touches only that one file.
- `gate.py` owns all state (temporal smoothing) and turns raw geometry into a
  single patient-facing instruction.

## Tuning

Every threshold lives in `pose_gate/config.py`. For elderly or injured patients
who cannot stand perfectly straight, loosen `max_roll` / `max_torso_tilt`. Watch
the on-screen `yaw / roll / tilt / head_yaw / head_pitch` readout to pick values
from real data.

## Notes

- Running Pose Landmarker + Face Landmarker together is heavier. If you need more
  speed, switch the pose bundle to `pose_landmarker_full` or `_lite`.
- OpenCV's `putText` cannot render Bangla script. For Bangla on-screen text,
  draw with PIL and a Bengali font (e.g. Noto Sans Bengali); the message keys in
  `gate.py` stay the same.
