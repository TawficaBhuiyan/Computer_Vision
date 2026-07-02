# Office Scuffle Tracker — Person + Ball detection with stable IDs

Detects **every person** and the **ball** in a CCTV clip, assigns each person a
**steady, non-shuffling ID**, and draws **stable, non-jittery bounding boxes**.

## Why this design
| Requirement | How it is solved |
|---|---|
| IDs must not shuffle | **BoT-SORT + ReID** re-matches the same person after they cross/overlap |
| IDs survive occlusion (people jostling) | `track_buffer: 60` keeps a lost track alive ~2s |
| No spurious new IDs | high `new_track_thresh: 0.60` |
| Bounding boxes must be steady | **EMA smoothing** on person boxes (`ema_alpha`) |
| Ball must not flicker on/off | **BallStabilizer**: smooths + predicts through detection gaps |
| Static camera | `gmc_method: none` (no camera-motion compensation) |

## Folder structure
```
Multiple_Object_Tracker_with_BoT-SORT_with_ReID/
├── run.py                     # entry point  ->  python run.py
├── requirements.txt
├── config/
│   └── config.yaml            # ALL tunable settings live here
├── trackers/
│   ├── botsort_stable.yaml    # primary: BoT-SORT + ReID
│   └── bytetrack_stable.yaml  # lighter fallback
├── src/
│   ├── config_loader.py       # yaml -> dotted-access object
│   ├── video_io.py            # VideoReader / VideoWriter
│   ├── detector.py            # YOLO + integrated tracker
│   ├── tracker.py             # parse results -> Track objects
│   ├── smoothing.py           # EMASmoother + BallStabilizer  (anti-jitter)
│   ├── visualizer.py          # boxes, IDs, colors, trajectory, HUD
│   └── pipeline.py            # orchestrates everything
├── input/                     # put your video here
└── outputs/                   # annotated result lands here
```

## Setup
```bash
cd Multiple_Object_Tracker_with_BoT-SORT_with_ReID
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
1. Put your clip in `input/input_video.mp4` (or edit the path in `config/config.yaml`).
2. Then:
```bash
python run.py
# no GPU:
python run.py --device cpu
```
Result is written to `outputs/tracked_output.mp4`.

## Tuning cheatsheet (edit config/config.yaml)
- **Boxes still jitter?** lower `smoothing.ema_alpha` (e.g. 0.25).
- **Ball disappears too often?** raise `smoothing.ball_max_lost` (e.g. 40) and lower `model.ball_conf`.
- **IDs still switch on heavy overlap?** raise `track_buffer` to 90 in `botsort_stable.yaml`.
- **Random new IDs pop up?** raise `new_track_thresh` to 0.7.
- **Too slow?** use `yolov8m.pt`, set `imgsz: 640`, or switch to `bytetrack_stable.yaml`.

## Notes
- COCO class ids: person = 0, sports ball = 32. A football/soccer ball is
  detected as "sports ball".
- BoT-SORT ReID needs `ultralytics >= 8.3.114`. If you hit a config error,
  set `with_reid: false` in `botsort_stable.yaml` or use the ByteTrack config.
