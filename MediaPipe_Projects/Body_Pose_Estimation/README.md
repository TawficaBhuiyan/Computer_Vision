# Body Positioning Gate (side-profile)

Processes a recorded video and writes an annotated output video that labels,
frame by frame, whether the person is in the correct pose for evaluation.

**Correct pose = standing SIDE-ON to the camera (side profile), upright, with the
whole body in frame.** (Not facing the camera.)

Three checks:
1. **Whole body** in frame (head + at least one side of shoulders/hips/knees/ankles)
2. **Side profile** -- body roughly 90 degrees to the camera
3. **Upright** -- torso close to vertical

When these hold for enough consecutive frames, the frame is marked
`READY - valid pose`; that is where you feed the frame to your  pipeline.

## Setup (conda + VS Code)

```bash
conda env create -f environment.yml
conda activate pose-gate
python download_model.py     # one-time: fetches the pose model
```

Put your clip next to run.py as `input_video.mp4`, then:

```bash
python run.py                # writes output.mp4, ESC to stop early
```

To use a live webcam instead of a file, set `SOURCE = 0` in `run.py`.

## Project structure

```
pose_gate_project/
├── environment.yml
├── download_model.py        # one-time pose-model fetch
├── run.py                   # video in -> annotated video out
├── models/                  # created by download_model.py
└── pose_gate/
    ├── config.py            # landmark indices + all thresholds
    ├── geometry.py          # body angles (side profile + upright)
    ├── gate.py              # checks + temporal smoothing + labels
    └── detector.py          # Pose Landmarker wrapper -> numpy arrays
```

## How the side-profile check works

From the shoulder line in world coordinates we compute one angle:
`0 deg = facing the camera, 90 deg = perfect side profile`. When side-on, one
shoulder is near the camera and the other is far, so the depth difference is
large and the angle is large. The gate passes when the angle is above
`min_side_angle` (default 55).

The on-screen `side_angle` and `tilt` readout lets you tune thresholds from your
own footage. If valid poses are being rejected, lower `min_side_angle`; if bad
poses slip through, raise it.

## Notes

- Output keeps the input video's real frame rate, so it plays at normal speed.
- If `output.mp4` ends up empty/unplayable, the code automatically falls back to
  `output.avi` (XVID) and prints a message.
- The head-facing-camera check was removed: in a side profile both eyes are never
  toward the camera, so that check does not apply here.