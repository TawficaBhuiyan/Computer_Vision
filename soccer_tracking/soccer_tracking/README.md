# Soccer Match Tracking — BoT-SORT + C-BIoU

A modular multi-object tracker for football footage. Combines **BoT-SORT**
(Kalman motion + Camera Motion Compensation + two-stage association) with
**C-BIoU** (Cascaded Buffered IoU), plus fast-motion enhancements that
drastically reduce ID switches on quick-moving players and the ball.

## Project layout

```
soccer_tracking/
├── main.py                     # CLI entry point
├── requirements.txt
├── config/
│   └── tracker.yaml            # all tunable parameters
├── soccer_tracker/             # the package
│   ├── __init__.py             # public API
│   ├── config.py               # UTF-8-safe YAML loader
│   ├── geometry.py             # box maths: IoU, buffer, center distance
│   ├── kalman.py               # constant-velocity Kalman filter
│   ├── track.py                # one object's lifecycle
│   ├── cmc.py                  # camera motion compensation
│   ├── association.py          # Hungarian matching (buffered IoU + distance)
│   ├── tracker.py              # orchestrator (per-frame loop)
│   ├── detector.py             # YOLO wrapper
│   ├── annotator.py            # drawing
│   └── pipeline.py             # video read -> track -> write
└── tests/
    └── test_tracker.py         # synthetic ID-stability tests (no video needed)
```

## Install

```bash
conda create -n soccer python=3.11 -y
conda activate soccer
pip install -r requirements.txt
```

## Run

```bash
python main.py --source video.mp4 --output output_video.mp4
```

Options: `--model yolov8n/s/m/x.pt`, `--imgsz 1280`, `--conf 0.10`,
`--config config/tracker.yaml`.

## Test (no video or model required)

```bash
python -m tests.test_tracker
```

## Notes

- On Windows, the config is read as UTF-8 explicitly, so non-ASCII bytes never
  cause a `UnicodeDecodeError`.
- Detector quality is the biggest factor in fast-motion stability — prefer
  `yolov8m`/`yolov8x` if your hardware allows.

See the full PDF documentation for concept explanations, the tracker comparison
table, real-time selection guidance, and a method-by-method walkthrough.
