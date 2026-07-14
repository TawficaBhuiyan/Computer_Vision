# Real-Time Side Profile Full Body Pose Validation System

A real-time computer-vision gate that validates whether a person is standing in a
**clean full-body side profile** — like the reference below — and gives
**directional corrective feedback** until they reach it. Rejects front, back,
diagonal, partial-side and incomplete-body poses.

Built on **MediaPipe BlazePose GHUM** with pure-geometry validation (no training
data), interpretable metrics, and temporal smoothing.

Every rendered frame carries the full readout: **pose skeleton, bounding box,
confidence, side-profile status, body orientation, head pose, validation result,
failure reasons, corrective guidance, and an overall confidence score.**

---

## Quick start

```bash
conda env create -f environment.yml
conda activate side-profile-gate
# The model is bundled in models/. Only run this if that folder is empty:
python download_model.py
```

```bash
# Live webcam (ESC to quit)
python run.py --source webcam

# Single image -> annotated image
python run.py --source image --input path/to/photo.png \
              --output out/annotated.png

# Video file -> annotated output video
python run.py --source video --input input_video.MOV \
              --output output_video.mp4

# Accept only one facing direction, and run the tests
python run.py --source webcam --accept-facing left
pytest -q
```

> Without conda: `pip install -r requirements.txt` (Python 3.10–3.12).

---

## What it validates (9 modules, in priority order)

1. **Single person** — rejects zero or multiple people (`num_poses=2`).
2. **Full-body visibility** — head → shoulders → hips → knees → ankles → feet,
   *profile-aware* (the occluded far side is not penalised).
3. **Side-profile detection (core)** — world-frame shoulder yaw (`≥ 60°` = side-on)
   fused with four independent facing cues (weighted vote) → `PROFILE_LEFT` /
   `PROFILE_RIGHT` / `OBLIQUE` / `FRONT` / `BACK`.
4. **Head pose** — geometric head-level band + head/body facing agreement
   (solvePnP is shown but not gated at 90°, where it is unreliable).
5. **Body alignment** — torso tilt, knee angle, hip angle (well-conditioned
   side-on); shoulder roll shown but not gated.
6. **Camera distance** — `body_fill ∈ [0.45, 0.95]` (rejects too far / too close).
7. **Frame boundary** — rejects cropped head, cropped feet, body out of frame.
8. **Temporal validation** — One Euro Filter (angles) + sliding-window vote
   (label) + streak latch (READY needs 12 consecutive valid frames).
9. **Feedback engine** — one prioritised, *directional* instruction per frame,
   e.g. *"Turn 90 degrees to show your side profile"*, *"Keep turning"*, *"Show
   your feet — step back"*, *"Stand up straight"*, *"Perfect side profile — hold
   still"*.

---

## Project structure

```
SideProfilePoseValidation/
├── run.py                     # CLI: webcam / image / video
├── download_model.py          # one-time model fetch (bundled by default)
├── requirements.txt
├── environment.yml
├── README.md                  # this file — quick start
├── models/
│   └── pose_landmarker_heavy.task
├── docs/
│   ├── Documentation.md       # full deep-dive: architecture, every module, every config value
│   ├── ARCHITECTURE.md        # short software-architecture reference
│   └── MIGRATION.md           # front-view → side-profile migration guide
├── tests/
│   └── test_core.py           # unit tests (no camera needed)
├── input_video*.MOV           # sample input clips you can test with
├── output*_video.mp4          # example annotated outputs
└── profile_gate/
    ├── config.py              # landmark indices + all thresholds
    ├── landmarks.py           # array access, per-level confidence, bbox
    ├── geometry.py            # pure body-angle math (world landmarks)
    ├── orientation.py         # Module 3: side-profile classifier (core)
    ├── head.py                # Module 4: head level + facing
    ├── filters.py             # Module 8: One Euro / EMA / vote / streak
    ├── validators.py          # Modules 1,2,5,6,7 as independent checks
    ├── feedback.py            # Module 9: priority + directional guidance
    ├── detector.py            # MediaPipe wrapper (IMAGE/VIDEO, num_poses=2)
    ├── gate.py                # orchestrator → GateResult
    └── render.py              # skeleton + bbox + status panel
```

See [`docs/Documentation.md`](docs/Documentation.md) for a full walkthrough of
every file, module, and config value — written for readers new to this
codebase.

---

## Integration hook

When the gate reports `READY`, trigger your downstream pipeline. In
`run.py::run_webcam`:

```python
if result.ready:
    # >>> Trigger your downstream pipeline here <<<
    # e.g. capture(frame); analysis.run(result.person_image)
    pass
```

`GateResult` exposes everything you need: `valid`, `ready`, `label`, `facing`,
`confidence`, `bbox`, `metrics`, `reasons`, `message`, `person_image`.

---

## Tuning

Every threshold lives in `profile_gate/config.py`.

| Want to… | Change |
|---|---|
| Accept only left/right profiles | `accept_facing="left"` / `"right"` or `--accept-facing` |
| Allow less-perfect side turn | lower `profile_yaw_min` (e.g. 50°) |
| Be lenient on posture (elderly/injured) | raise `max_torso_tilt`, lower `min_knee_angle`/`min_hip_angle` |
| Allow closer / farther standing | widen `min_body_fill` / `max_body_fill` |
| Smoother (less jitter) vs snappier | lower/raise `oe_min_cutoff`, `oe_beta` |
| Faster READY latch | lower `stable_frames` |
| Faster model | swap to `pose_landmarker_full` / `_lite` and update `--model` |

---

## Notes

- All **angular** decisions use **world** landmarks (metric, hip-centered) to
  avoid aspect-ratio/perspective distortion; **framing** decisions use image
  landmarks, where apparent size is the point.
- BlazePose reports full visibility even for occluded far-side face landmarks, so
  the system **never** relies on face-visibility asymmetry — see
  `docs/Documentation.md`.
- OpenCV `putText` cannot render Bangla/Unicode script; for localized on-screen
  text, draw the message with PIL + a suitable font. The message *keys* in
  `feedback.py` stay the same.
