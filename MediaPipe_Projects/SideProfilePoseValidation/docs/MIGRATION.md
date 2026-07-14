# Migration Guide
### From "Patient Positioning Gate" (front view) → "Side Profile Full Body Pose Validation"

## 1. Analysis of the original application

**Original purpose.** A real-time pre-gate for a pain-evaluation pipeline that
passed a frame only when the patient was **facing** the camera, standing upright,
fully in frame, and **looking at** the lens.

**Original structure.**

```
Body_Pose_Estimation/
├── run.py                  webcam-only entry point
├── download_model.py       one-time model fetch
├── environment.yml
├── models/pose_landmarker_heavy.task
└── pose_gate/
    ├── config.py           indices + GateConfig thresholds
    ├── geometry.py         facing_yaw, shoulder_roll, torso_tilt (world)
    ├── head.py             solvePnP head pose from pose face keypoints
    ├── detector.py         PoseLandmarker wrapper (VIDEO, num_poses=1)
    └── gate.py             checks + EMA + feedback → GateResult
```

**Strengths (worth preserving).**
- Clean separation: only `detector.py`/`head.py` touched MediaPipe.
- `geometry.py` was pure and testable.
- All thresholds already centralised in `config.py`.
- `head.py` already recovered head pose via solvePnP with no second model.
- A single-instruction feedback idea (priority list) already existed.

**Weaknesses / mismatches for the new goal.**
- The **orientation semantics were inverted**: it accepted `yaw < 20°` (frontal)
  and rejected side-on — the opposite of what we now need.
- `num_poses = 1` — could not detect/reject a second person.
- Full-body check required **both** sides of each pair visible — impossible for a
  profile because of self-occlusion.
- solvePnP pitch/roll were trusted; they are **ill-conditioned at 90°**.
- Fixed-α **EMA** only; no adaptive filter, label debounce is implicit.
- **Webcam-only**; no image/video input, no annotated `output_video.mp4`.
- No bounding box, distance band, foot/heel landmarks, or joint-angle posture.

**Code smells / bottlenecks.** Minimal — the original was small and tidy. The
only genuine bottleneck (shared by both apps) is the heavy pose model itself.

## 2. Migration decisions

| Original file | Action | Why |
|---|---|---|
| `pose_gate/geometry.py` | **KEEP + EXTEND** | Pure, reusable. Added signed yaw, depth sign, joint angles, neck flex. `shoulder_roll`/`torso_tilt` reused verbatim. |
| `pose_gate/head.py` | **MODIFY** | solvePnP core reused; added geometric head-level + facing; solvePnP pitch/roll demoted to display. |
| `pose_gate/detector.py` | **MODIFY** | `num_poses 1→2`; added IMAGE mode; returns a **list** of persons. |
| `pose_gate/config.py` | **REWRITE** | New indices (elbow/wrist/heel/foot), `BODY_LEVELS`, `Orientation`, side-profile thresholds. |
| `pose_gate/gate.py` | **REWRITE** | New pipeline order, One Euro + vote + streak, richer `GateResult`. |
| `run.py` | **REWRITE** | Unified CLI (webcam/image/video), annotated output, `output_video.mp4`. |
| `download_model.py` | **KEEP** | Same model URL. |
| `environment.yml` | **KEEP (renamed env)** | Same deps; env renamed `side-profile-gate`. |
| `README.md` | **REWRITE** | New product. |
| package `pose_gate/` | **RENAME → `profile_gate/`** | Reflects the new purpose. |

**New files created**

| New file | Module | Purpose |
|---|---|---|
| `profile_gate/landmarks.py` | — | Array-access helpers, per-level confidence, bbox, body-fill, image facing cue. |
| `profile_gate/orientation.py` | **3** | Side-profile classifier (the core): world yaw + 3-cue fusion. |
| `profile_gate/filters.py` | **8** | One Euro filter, EMA, sliding-window vote, streak latch. |
| `profile_gate/validators.py` | **1,2,5,6,7** | Independent `Check`-returning validators. |
| `profile_gate/feedback.py` | **9** | Priority + directional corrective guidance. |
| `profile_gate/render.py` | — | Skeleton, bbox, full status panel overlay. |
| `tests/test_core.py` | — | Unit tests for geometry/filters/orientation. |
| `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`, `docs/MIGRATION.md` | — | Deliverable documentation. |

**Removed:** nothing was deleted outright — the original front-view *thresholds*
and the `facing_yaw < max_yaw` *acceptance rule* were replaced by side-profile
semantics, but every reusable line of geometry/head/detector logic was carried
forward.

## 3. Behavioural mapping (old rule → new rule)

| Concept | Old (front view) | New (side profile) |
|---|---|---|
| Orientation accept | `facing_yaw < 20°` | `yaw_magnitude ≥ 60°` **and** ≥2/3 facing cues agree |
| Person count | implicit single | explicit reject of 0 or >1 |
| Full body | both sides visible | per-level near-side `max` visibility |
| Head | solvePnP yaw & pitch small | geometric head-level band + head/body facing agreement |
| Posture | roll + torso tilt | torso tilt + knee + hip (+ generous neck); roll shown not gated |
| Distance | `body_fill` upper bound only | `body_fill ∈ [0.45, 0.95]` (too far *and* too close) |
| Smoothing | fixed EMA | One Euro (angles) + window vote (label) + streak (READY) |
| Input | webcam only | webcam / image / video → `output_video.mp4` |
| Output | text lines | skeleton + bbox + full status panel |

## 4. How to run the migrated system

```bash
conda env create -f environment.yml && conda activate side-profile-gate
# model is bundled; only needed if models/ is empty:
python download_model.py

python run.py --source webcam
python run.py --source image --input samples/side_profile_left.png --output out/annotated.png
python run.py --source video --input input_video.MOV --output output_video.mp4
pytest -q
```

## 5. Validation performed during migration

- Both supplied reference photos classify as **READY** perfect profiles
  (`PROFILE_RIGHT` / `PROFILE_LEFT`, confidence ≈ 0.98).
- Synthetic front / oblique / left / right sweep classifies correctly with the
  right directional feedback.
- A synthesized demo clip (blank → right → left) latches **READY** only after the
  streak builds, producing an annotated `output_video.mp4`.
- 9/9 unit tests pass.
