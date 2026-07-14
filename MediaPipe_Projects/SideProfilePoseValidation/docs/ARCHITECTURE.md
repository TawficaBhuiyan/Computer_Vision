# Software Architecture
### Real-Time Side Profile Full Body Pose Validation System

## Design goals

1. **One boundary to MediaPipe.** Only `detector.py` and `head.py` touch the
   framework; everything else operates on plain numpy arrays and is unit-testable
   without a camera.
2. **Single Responsibility per module.** Each requirement (Modules 1–9) lives in
   one place, so a rule can be changed or tested in isolation.
3. **All tuning in one file.** Every threshold and landmark index is in
   `config.py`; logic files contain no magic numbers.
4. **Pure geometry.** The math in `geometry.py` depends only on numpy and is
   deterministic — the core of the test suite.

## Layered view

```
                        ┌───────────────────────────┐
   Presentation         │           run.py          │  CLI: webcam/image/video
                        │           render.py       │  skeleton + bbox + panel
                        └────────────┬──────────────┘
                                     │ GateResult
                        ┌────────────┴──────────────┐
   Orchestration        │           gate.py         │  compose modules + state
                        └───┬───────────┬───────────┘
                            │           │
        ┌───────────────────┴──┐   ┌────┴───────────────────────┐
   Decision                    │   │                            │
   logic   validators.py  orientation.py  feedback.py  filters.py
        └───────────────┬──────┘   └────────────┬───────────────┘
                        │  numpy arrays          │
        ┌───────────────┴────────────────────────┴──────────────┐
   Math / IO   geometry.py   landmarks.py   head.py   detector.py
   boundary    (pure math)  (array access) (cv2 PnP) (MediaPipe)
        └────────────────────────────────────────────────────────┘
                                     │
                              config.py  (indices + thresholds)
```

## Per-frame data flow

```
frame (BGR)
  └─ detector.detect(rgb, ts) ────────────► List[(image[33,4], world[33,3])]
        │
        ▼
  gate.evaluate(persons, w, h)
        1. validate_single_person(n)                     ─ Module 1
        2. orientation.classify(image, world)            ─ Module 3 (core)
        3. validators:
             validate_full_body(image)                   ─ Module 2
             validate_frame_boundary(image)              ─ Module 7
             validate_distance(image)                    ─ Module 6
             validate_orientation(orient)                ─ Module 3 gate
             head.estimate_head_pose(image, w, h)        ─ Module 4
             validate_head(head, orient)                 ─ Module 4 gate
             validate_posture(world, orient)             ─ Module 5
        4. filters: OneEuro(angles) + vote(label) + streak(valid)  ─ Module 8
        5. feedback.build_message(reasons, orient)       ─ Module 9
        └─────────────────────────► GateResult
        │
        ▼
  render.draw(frame, result, cfg) ──────────► annotated frame
```

## Module responsibilities

| File | Responsibility | Touches MediaPipe/cv2 | Stateful |
|------|----------------|:---:|:---:|
| `config.py` | Landmark indices, body levels, all thresholds, `Orientation` labels | – | – |
| `landmarks.py` | Safe array access, per-level confidence, bbox, body-fill, image facing cue | – | – |
| `geometry.py` | Pure body-angle math (yaw, tilt, joint angles, neck) | – | – |
| `orientation.py` | **Module 3** side-profile classifier (yaw + 4-cue weighted-vote fusion) | – | – |
| `head.py` | **Module 4** solvePnP (display) + geometric head level/facing | cv2 | – |
| `filters.py` | **Module 8** One Euro, EMA, window vote, streak latch | – | ✓ |
| `validators.py` | **Modules 1,2,5,6,7** + orientation/head gates → `Check` | – | – |
| `feedback.py` | **Module 9** priority + directional guidance | – | – |
| `detector.py` | MediaPipe wrapper (IMAGE/VIDEO, `num_poses=2`) → numpy | MediaPipe | (model) |
| `gate.py` | Orchestrate all modules + own temporal state → `GateResult` | – | ✓ |
| `render.py` | Skeleton, bbox, status panel overlay | cv2 | – |
| `run.py` | CLI entry point (webcam/image/video), writes `output_video.mp4` | cv2 | – |

## Core data structures

```python
# detector.py
Person = Tuple[np.ndarray, np.ndarray]   # (image[33,4], world[33,3])

# validators.py
Check(ok: bool, reason: Optional[str], detail: dict)

# orientation.py
OrientationResult(label, facing, yaw, signed_yaw, hip_yaw, confidence, cues)

# head.py
HeadPose(yaw, pitch, roll, facing, level, spread, ok)

# gate.py — the single object the UI consumes
GateResult(valid, ready, streak, label, facing, message,
           reasons, metrics, confidence, bbox, person_image, head, n_persons)
```

## SOLID mapping

* **S**ingle Responsibility — one module per requirement; `Check` isolates a
  single rule's outcome.
* **O**pen/Closed — add a rule by writing a new `validate_*` returning `Check`
  and inserting its key into `feedback.PRIORITY`; existing modules untouched.
* **L**iskov — every validator shares the `(_, cfg) -> Check` shape and is
  interchangeable in the orchestration list.
* **I**nterface Segregation — `geometry` needs only `world`; `landmarks`/framing
  need only `image`; `head` needs only face keypoints. No module receives more
  than it uses.
* **D**ependency Inversion — `gate` depends on the `Check` abstraction and an
  injected `ProfileGateConfig`, not on concrete threshold constants.

## Configuration & extension points

* **Accept one facing only** — `ProfileGateConfig(accept_facing="left")` (or the
  `--accept-facing` CLI flag).
* **Different population** (elderly, injured) — loosen `max_torso_tilt`,
  `min_knee_angle`, `min_hip_angle`.
* **Faster model** — point `download_model.py` and `--model` at
  `pose_landmarker_full`/`_lite`.
* **Smoothing feel** — `oe_min_cutoff` (jitter), `oe_beta` (lag), `vote_window`,
  `stable_frames`.

## Performance notes

The dominant cost is the pose network. Everything downstream is a few dozen
`atan2`/dot-product/`sign` operations plus one small `solvePnP` — negligible.
For higher FPS, switch the pose bundle to `_full` or `_lite`; the entire
decision layer is unchanged because it consumes only landmark arrays.
