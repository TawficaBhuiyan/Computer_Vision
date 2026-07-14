# Real-Time Side Profile Full Body Pose Validation System


---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [The AI Model Behind Everything](#2-the-ai-model-behind-everything)
3. [Big Picture: How It Works](#3-big-picture-how-it-works)
4. [Folder & File Map](#4-folder--file-map)
5. [Full Architecture Diagram](#5-full-architecture-diagram)
6. [Step-by-Step: The Journey of ONE Video Frame](#6-step-by-step-the-journey-of-one-video-frame)
7. [File-by-File Explanation](#7-file-by-file-explanation)
8. [The 9 Validation Modules, Explained Simply](#8-the-9-validation-modules-explained-simply)
9. [Every Config Value Explained (the tuning bible)](#9-every-config-value-explained-the-tuning-bible)
10. [How to Run the Project](#10-how-to-run-the-project)
11. [Tuning Cookbook — Common Situations](#11-tuning-cookbook--common-situations)
12. [Testing](#12-testing)
13. [Troubleshooting / FAQ](#13-troubleshooting--faq)
14. [Glossary](#14-glossary)

---

## 1. What This Project Does

Imagine a camera is watching a person, and you want the computer to
automatically tell:

> "Is this person standing in a **clean, full-body SIDE view** (like a
> passport photo taken from the side), or not?"

If **not**, the system tells the person exactly what to fix, one instruction
at a time, for example:

- "Turn 90 degrees to show your side profile"
- "Show your feet — step back"
- "Stand up straight"
- "Perfect side profile — hold still" ✅

This is useful as a **gate** (a checkpoint) before some other process runs —
for example, before taking a measurement photo, before a body-analysis AI
runs, or before a medical/fitness screenshot is captured. The gate only says
"READY" when the person is standing correctly, so the next step in your
pipeline always gets a clean, well-framed side-profile image.

**Key facts:**
- It works on **webcam (live)**, a **single image**, or a **video file**.
- It uses **no training data of its own** — it does not "learn" from
  thousands of side-profile photos. Instead, it uses **pure geometry and
  math** (angles, distances) on top of a ready-made body-tracking AI model.
  This makes it fast, predictable, and easy to tune by hand.
- It rejects: nobody in frame, more than one person, front view, back view,
  diagonal view, half-turned view, cropped body, too far / too close, bad
  posture (leaning, bent knees), and head turned the wrong way.

---

## 2. The AI Model Behind Everything

The one and only AI model used is:

```
models/pose_landmarker_heavy.task
```

This is Google's **MediaPipe BlazePose GHUM (Heavy)** model. You do **not**
train or modify this model — it is a ready-made "body skeleton detector".

### What the model gives you

For every person in a video frame, the model outputs **33 body points**
("landmarks") — nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
ankles, heels, and toes. Each point comes in **two coordinate systems**:

| Type | Shape | Meaning |
|---|---|---|
| `image` landmarks | `(33, 4)` → `[x, y, z, visibility]` | `x, y` are positions **on the picture** (0 to 1, left-to-right / top-to-bottom). `visibility` (0 to 1) is "how sure the model is this point is really visible". Use this for **framing** questions (is the head near the top edge? is the person too close to the camera?). |
| `world` landmarks | `(33, 3)` → `[x, y, z]` | Real-world **meters**, centered on the mid-hip point, camera-distance independent. Use this for **angle** questions (is the body turned 90°? is the spine straight?). |

**Why two systems matter:** if you measured "is the person turned
sideways" using only the flat picture (`image` x, y), the answer would
change depending on how far the camera is or the image's aspect ratio. The
`world` coordinates fix this — they are like a 3D skeleton floating in real
space, so one angle threshold (e.g. "60 degrees") works whether the person
is 1 meter or 4 meters from the camera.

### Model variants (speed vs accuracy)

Google provides 3 sizes of this same model:

| Model file | Speed | Accuracy | When to use |
|---|---|---|---|
| `pose_landmarker_lite` | Fastest | Lowest | Old/slow computers, need max FPS |
| `pose_landmarker_full` | Medium | Medium | Good balance |
| `pose_landmarker_heavy` (**used by default**) | Slowest | Best | Default — most accurate landmarks |

You can switch by downloading a different `.task` file (see
[`download_model.py`](download_model.py)) and passing it with `--model`.
See [Section 11](#11-tuning-cookbook--common-situations) for exact steps.

### Where the model is loaded

Only **one file** in the whole project talks to MediaPipe directly for body
landmarks: [`profile_gate/detector.py`](profile_gate/detector.py). Every
other file only works with plain numbers (numpy arrays), never with
MediaPipe itself. This is a deliberate design choice — it means 90% of the
codebase can be tested without a camera or the AI model at all (see
[Section 12](#12-testing)).

There is a **second**, much smaller use of AI/math: OpenCV's `solvePnP`
(a classic geometry solver, not a trained model) is used inside
[`profile_gate/head.py`](profile_gate/head.py) to estimate head rotation
from the 7 face points the body model already gives us. This does **not**
need a separate model file — it reuses the same 33 landmarks.

---

## 3. Big Picture: How It Works

Every video frame goes through a **pipeline of checks**, like a factory
assembly line. Each station checks one thing. If everything passes, the
frame is "valid". If it stays valid for many frames in a row, it becomes
"READY" (a stable, trustworthy detection — not just a lucky single frame).

```
Camera / Video / Image
        │
        ▼
 [AI Model: BlazePose] ───► 33 body points (skeleton)
        │
        ▼
 [9 Validation Modules] ───► pass / fail + WHY it failed
        │
        ▼
 [Smoothing over time]  ───► removes jitter / flicker
        │
        ▼
 [One instruction]      ───► "Turn 90 degrees" / "Perfect — hold still"
        │
        ▼
 Annotated video frame (skeleton + box + status text)
```

The system checks 9 things, always in the same priority order (most basic
first):

1. **Is there exactly one person?**
2. **Is the whole body visible** (head to feet)?
3. **Is the body turned sideways enough**, and which way (left/right)? *(the core check)*
4. **Is the head also turned sideways** (not looking at the camera)?
5. **Is the posture good** (standing straight, legs not bent)?
6. **Is the camera distance okay** (not too close, not too far)?
7. **Is the whole body inside the frame** (nothing cut off)?
8. **Has this been true for enough consecutive frames** (not just a flicker)?
9. **What ONE sentence should we tell the person right now?**

These map exactly to the 9 files inside `profile_gate/` — one file per job
(explained fully in [Section 7](#7-file-by-file-explanation)).

---

## 4. Folder & File Map

```
SideProfilePoseValidation/
│
├── DOCUMENTATION.md            ← YOU ARE HERE (this file)
├── README.md                   ← short quick-start version
├── run.py                      ← the program you actually run (CLI)
├── download_model.py           ← one-time download of the AI model file
├── requirements.txt            ← Python package list (pip)
├── environment.yml             ← Python package list (conda)
│
├── models/
│   └── pose_landmarker_heavy.task   ← the AI model file (BlazePose GHUM)
│
├── docs/
│   ├── Documentation.md        ← deep research notes: WHY each method was chosen
│   ├── ARCHITECTURE.md         ← short software-architecture reference
│   
│
├── tests/
│   └── test_core.py            ← automatic tests (no camera needed)
│
├── input_video*.MOV            ← sample input videos you can test with
├── output*_video.mp4           ← example annotated output videos
│
└── profile_gate/                ← THE CORE LOGIC (all 9 modules live here)
    ├── config.py                ← ⭐ every number/threshold you can tune
    ├── landmarks.py              ← reads raw points from the AI model's output
    ├── geometry.py                ← pure math: angles, tilt, joint bends
    ├── orientation.py              ← Module 3: decides LEFT/RIGHT/FRONT/BACK/OBLIQUE
    ├── head.py                      ← Module 4: is the head turned sideways too?
    ├── filters.py                    ← Module 8: smooths out jitter over time
    ├── validators.py                  ← Modules 1,2,5,6,7: the individual pass/fail checks
    ├── feedback.py                     ← Module 9: turns failures into ONE instruction
    ├── detector.py                      ← the ONLY file that talks to MediaPipe
    ├── gate.py                           ← the "manager" that runs everything, in order
    └── render.py                          ← draws the skeleton + status box on screen
```

**Golden rule of this codebase:** every file has exactly **one job**. If you
want to change how posture is judged, you only touch `validators.py`
(logic) and `config.py` (numbers) — nothing else needs to change. This
pattern is called **Single Responsibility** and it's why the project is
easy to tune safely.

---

## 5. Full Architecture Diagram

```
                         ┌───────────────────────────┐
   PRESENTATION          │   run.py                  │  CLI: webcam / image / video
   (what you see)        │   render.py               │  draws skeleton + status panel
                         └────────────┬───────────────┘
                                      │  GateResult (the final answer object)
                         ┌────────────┴───────────────┐
   ORCHESTRATION          │   gate.py                 │  runs all modules, in order,
   (the manager)          │                           │  and remembers state between frames
                         └───┬───────────┬─────────────┘
                             │           │
            ┌────────────────┴──┐    ┌────┴──────────────────────────┐
   DECISION                     │    │                                │
   LOGIC     validators.py   orientation.py    feedback.py    filters.py
            (5 simple checks)  (core: side view?) (1 message)  (smoothing)
            └───────────────┬───┘    └─────────────┬──────────────────┘
                            │   plain numbers only  │
            ┌───────────────┴────────────────────────┴───────────────┐
   MATH/IO   geometry.py     landmarks.py        head.py    detector.py
   BOUNDARY  (pure math)    (reads AI points)   (solvePnP)  (talks to MediaPipe)
            └────────────────────────────────────────────────────────┘
                                      │
                              config.py  (every number lives here)
```

**How to read this diagram:** information flows top-to-bottom (a decision
needs data from below it). Only the very bottom row (`detector.py`, and
partly `head.py`) ever imports MediaPipe or OpenCV's camera-related code.
Everything above that is pure Python math that a computer-vision beginner
can read like a spreadsheet formula.

---

## 6. Step-by-Step: The Journey of ONE Video Frame

This is exactly what happens, in order, every single frame (roughly 30
times per second on a webcam). Corresponding file/function shown in `code`.

**Step 1 — Capture.** `run.py` grabs one frame from the webcam/video/image
and converts it from OpenCV's color format (BGR) to the format MediaPipe
needs (RGB).

**Step 2 — Detect the skeleton.** `detector.py`'s `PoseDetector.detect()`
sends the frame to the AI model and gets back a list of people, each with
33 body points in both `image` and `world` coordinates. The model is
configured to look for **up to 2 people** (`num_poses=2`) — not 1 — so that
if a second person walks into frame, the system can *notice* and reject
them, instead of silently locking onto whoever is most prominent.

**Step 3 — Hand off to the Gate.** `gate.py`'s `ProfileGate.evaluate()` is
the manager. If zero people were found, it immediately fails with "Step
into the frame". If more than one, it fails with "Only one person in
frame, please". Otherwise it takes person #1 and continues.

**Step 4 — Classify orientation (the core decision).**
`orientation.py`'s `classify()` looks at the shoulder and hip lines in 3D
world space and computes **how sideways** the body is turned (an angle
from 0° = facing camera, to 90° = perfect side view). It also independently
checks **which way** (left or right) the person is facing, using 4 separate
clues that vote together (explained fully in
[Section 8, Module 3](#module-3-side-profile-orientation-the-core-module)).

**Step 5 — Smooth the numbers.** Before any pass/fail decision is made,
`gate.py` runs the raw angle values through `filters.py`'s **One Euro
Filter** (removes camera jitter), and the raw orientation *label* (e.g.
"OBLIQUE") through a **sliding-window vote** (removes flip-flopping between
two labels at a boundary). This happens **before** gating, not just for
display — this is what stops the on-screen box from flickering
red/green/red every frame even when the person is holding perfectly still.

**Step 6 — Run all the individual checks.** `validators.py` runs 5 more
independent checks using the *smoothed* values from Step 5:
- Module 2: full-body visibility (`validate_full_body`)
- Module 7: frame boundary — nothing cropped (`validate_frame_boundary`)
- Module 6: camera distance (`validate_distance`)
- Module 3 (gate step): does the orientation label count as accepted
  (`validate_orientation`)
- Module 4: head pose (`validate_head`), which needs `head.py`'s
  `estimate_head_pose()` to run first
- Module 5: posture — torso tilt, knee angle, hip angle
  (`validate_posture`)

Every single check returns a small `Check` object: `ok` (True/False),
`reason` (a short code like `"bent_knee"`), and `detail` (numbers, for the
on-screen debug panel). **All checks always run** — even after one fails —
so the status panel always has full numbers to show, not just the first
failure.

**Step 7 — Combine into one verdict.** `gate.py` collects every failing
`reason` into a dictionary. If the dictionary is empty, this frame is
`valid = True`. It feeds `valid` into `filters.py`'s **StreakLatch**: only
after 12 valid frames *in a row* does `ready` become `True`. This stops a
single lucky frame from triggering your downstream pipeline.

**Step 8 — Build one message.** `feedback.py`'s `build_message()` looks at
every failing reason, but only returns the **single highest-priority**
instruction (see the `PRIORITY` list in that file) — e.g. it will always
tell you to step into frame before it complains about your knee being bent,
because there's no point telling someone about posture if they're not even
in the picture yet.

**Step 9 — Draw and show.** `render.py`'s `draw()` paints the skeleton
lines, a bounding box (red = invalid, orange = valid-but-not-yet-stable,
green = READY), and a status panel with all the live numbers, onto the
frame. `run.py` then either shows it in a window (webcam) or writes it to
an output video/image file.

**Step 10 — Your code reacts.** In `run.py::run_webcam`, there's a marked
hook:
```python
if result.ready:
    # >>> Trigger your downstream pipeline here <<<
    pass
```
This is where you plug in whatever should happen once a clean side profile
is confirmed (e.g., save the photo, run your analysis model, etc).

---

## 7. File-by-File Explanation

### `profile_gate/config.py` — the settings file
Holds **every number** used anywhere in the project: which landmark index
is "left shoulder", which body parts count as one "level", and all pass/fail
thresholds. No other file contains a hard-coded number. If you want to
change how strict or lenient the system is, **this is the only file you
should normally edit**. Full breakdown in [Section 9](#9-every-config-value-explained-the-tuning-bible).

### `profile_gate/landmarks.py` — reading the AI model's output
Small helper functions that answer simple questions about the raw 33
points, e.g.:
- `visibility(image, idx)` — how confident is the model about this one point?
- `level_confidence(image, left_idx, right_idx)` — is at least ONE side of a
  pair (e.g. left OR right ankle) visible? (Explained in Module 2 below —
  this "at least one side" idea is the key trick that makes side-profile
  detection possible at all.)
- `body_fill(image)` — what fraction of the picture's height does the body
  take up? (Used to judge camera distance.)
- `facing_from_image(image)` — is the nose to the left or right of the ear
  midpoint? (One vote for "which way is the person facing".)
- `bounding_box(image, w, h)` — the rectangle box drawn around the person.

### `profile_gate/geometry.py` — pure angle math
Contains only mathematics — no AI, no camera, no config values. Given 3D
points, it calculates:
- `shoulder_yaw_magnitude` — how sideways the shoulders are turned (0°–90°).
- `hip_yaw_magnitude` — same, for the hips (used to detect a twisted torso).
- `torso_twist` — difference between shoulder-turn and hip-turn.
- `torso_tilt` — is the spine leaning forward/back/sideways?
- `knee_angle`, `hip_angle` — are the legs straight, is the person bent
  at the waist?
- `neck_flex` — is the head craned forward?
- `shoulder_roll` — shoulder levelness (calculated but **not used to
  reject** — explained in Module 5).

Because this file only uses plain numpy math, it can be tested with
made-up numbers, with no camera and no AI model — see `tests/test_core.py`.

### `profile_gate/orientation.py` — Module 3, the core decision
The single most important file. Decides the orientation **label**:
`PROFILE_LEFT`, `PROFILE_RIGHT`, `OBLIQUE` (partway turned), `FRONT`,
`BACK`, or `UNKNOWN`. Full explanation in
[Section 8, Module 3](#module-3-side-profile-orientation-the-core-module).

### `profile_gate/head.py` — Module 4, head direction
Estimates whether the **head** (not just the body) is also turned to the
side, and whether it's level (not nodding up/down). Uses a classic
technique called `solvePnP` (matches 2D face points to a rough 3D head
shape) for pitch/roll display, but for the actual left/right and
level/nod decision it uses simpler, more reliable 2D geometry — because
`solvePnP` becomes mathematically unstable exactly at 90° side-turn, which
is the pose this whole system is trying to detect. See
[Module 4](#module-4-head-pose) below for the full reasoning.

### `profile_gate/filters.py` — Module 8, smoothing over time
Three tools that each solve a different "jitter" problem:
- **`OneEuroFilter`** — smooths continuous numbers (angles). Adapts: very
  smooth when the person is holding still, quick to react when they move.
- **`SlidingWindowVote`** — smooths the text label (e.g. `"OBLIQUE"`) by
  taking the majority answer over the last few frames.
- **`StreakLatch`** — only says `READY` after N frames pass **in a row**.

### `profile_gate/validators.py` — Modules 1, 2, 5, 6, 7
Five independent pass/fail checks, each a small pure function. Every check
returns the same shape (`Check(ok, reason, detail)`), which is what lets
`gate.py` run them all in a simple loop.

### `profile_gate/feedback.py` — Module 9, the message
Turns a pile of failing reason-codes into exactly one sentence, using a
priority list (`PRIORITY`) so the most fundamental problem is always
mentioned first (e.g. "no person in frame" beats "bad posture").

### `profile_gate/detector.py` — talking to the AI model
The **only** file that imports `mediapipe`. Wraps the model so the rest of
the codebase never has to know MediaPipe exists — it just gets back plain
numpy arrays. Uses `num_poses=2` (see Step 2 above) and runs in `VIDEO`
mode for webcam/video (one guaranteed result per frame, no dropped frames)
or `IMAGE` mode for single stills.

### `profile_gate/gate.py` — the manager / orchestrator
Runs all the modules above in the correct order, holds the temporal
filters (so it "remembers" the last several frames), and returns one
`GateResult` object per frame — the single source of truth that
`render.py` and your own downstream code should read.

### `profile_gate/render.py` — drawing the overlay
Purely cosmetic: draws the skeleton lines, the bounding box (colored by
status), and the on-screen text panel showing live numbers. Does not
affect any pass/fail decision — you can freely edit this file to change
what's displayed without breaking any logic.

### `run.py` — the command you actually run
The command-line entry point. Reads `--source`, `--input`, `--output`,
`--model`, `--camera`, `--accept-facing` from the terminal, wires
`detector.py` → `gate.py` → `render.py` together, and either shows a live
window (webcam) or writes an output file (image/video).

### `download_model.py` — one-time setup
Downloads `pose_landmarker_heavy.task` from Google's servers if it's not
already in `models/`. You normally don't need to run this — the model file
is already included in the repo.

---

## 8. The 9 Validation Modules, Explained Simply

### Module 1: Single Person
**Question:** Is there exactly one person in frame?
**How:** The AI model is told to look for up to 2 people. 0 people →
reject ("step into frame"). 2+ people → reject ("only one person,
please"). Looking for 2 (not 1) matters: if you only ask for 1, the model
would silently lock onto whichever person is clearest and never tell you a
second person walked in.

### Module 2: Full-Body Visibility
**Question:** Can we see the whole body, head to feet?
**The tricky part:** In a true side view, one entire side of the body
(the far shoulder, far hip, far knee, etc.) is naturally hidden behind the
near side. So a rule like "both left AND right ankle must be visible"
would make a **correct** side profile **impossible** to pass!
**The fix:** the body is split into 6 vertical "levels" (head, shoulder,
hip, knee, ankle, foot). For each level, we only require that **at least
one side** (whichever one the camera can see) is confidently visible. Feet
get an extra-relaxed threshold because they're naturally the least
confidently detected part of the skeleton.

### Module 3: Side-Profile Orientation (the core module)
**Question:** Is the body turned to a clean side view, and which way (left
or right)?

This has **two separate sub-questions**, solved independently:

**(a) How side-on is the body?** Measured as the angle of the line between
the two shoulders, in real-world 3D space:
- 0° → shoulders point across the picture → facing the camera or back to it
- 90° → shoulders point into the depth of the scene → a true side view

Because this uses the metric `world` coordinates (not flat pixels), the
same threshold works no matter how close or far the camera is.

**(b) Which way is the person facing (left or right)?** The shoulder angle
alone can't tell — 0° could be front OR back, and even at 90° we don't
automatically know left vs right. So the system asks **4 independent
questions** and combines their answers by a **weighted vote**, not by
requiring all 4 to agree (requiring perfect agreement turned out to be too
fragile — one noisy landmark could silently ruin a genuinely correct pose):
1. Is the nose to the left or right of the ear midpoint, in the flat image?
   *(most trusted — gets the highest vote weight)*
2. Which shoulder is physically nearer the camera (world depth)?
3. Which hip is physically nearer the camera (world depth)?
4. Same idea as #2, but measured from the model's rougher image-space depth
   estimate *(least trusted — gets the lowest weight, used only as a tie-breaker)*

There are also two extra "sanity" checks layered on top:
- **Twist check:** the shoulder line and hip line should have turned by
  roughly the same amount. If they haven't, the torso is **twisted**, not
  cleanly rotated — rejected with the reason `twisted_torso`.
- **Overlap check:** in a true side view, the left and right shoulders
  (and hips) should nearly line up on top of each other in the flat image
  (because one is directly behind the other from the camera's point of
  view). If they're still spread apart sideways, it's a "half-turned" pose
  — rejected with reason `not_full_side`.

If the body isn't turned far enough at all, the result is `FRONT` or
`BACK` (decided by whether the nose is nearer the camera than the ears).
If it's partway turned, the result is `OBLIQUE` (with a helpful sub-reason
like `slight_rotation` or `three_quarter`). Only a fully-turned,
untwisted, non-overlapping, confidently-voted pose becomes `PROFILE_LEFT`
or `PROFILE_RIGHT`.

Finally, all of these individual signals are also combined into one
**0–1 confidence score**. A pose can technically pass every individual
hard rule and *still* be rejected if the overall evidence, taken together,
is weak (`low_confidence`) — this catches "borderline everywhere" cases
that no single rule would catch alone.

### Module 4: Head Pose
**Question:** Is the head also turned sideways (not just the body), and
is it level (not nodding up/down or turned back toward the camera)?

The classical technique for head pose is `solvePnP`: match a handful of
face points (nose, eyes, ears, mouth) to a rough 3D head shape and solve
for rotation. This works great when facing the camera — but at a true 90°
side turn, the far half of the face is hidden, and the near-side points
become almost a straight line in the image, which makes the math
**unstable**. (Measured on real photos: it returned nonsense pitch/roll
values like 75°/−83° on a perfectly good, level side profile.)

**The fix:** split the job.
- **Head level (nodding up/down)** — measured geometrically, from the
  angle between the nose and the near ear, in pixels. This stays reliable
  at any head rotation.
- **Head turned toward the camera** — measured by how spread apart the two
  eyes/ears appear. In a true head-profile they nearly overlap; if the
  head rotates toward the camera they visibly separate.
- **Head facing matches body facing** — makes sure the person isn't
  looking back over their shoulder while their body faces the other way.
- `solvePnP`'s pitch/roll numbers are still calculated and shown on
  screen for reference, but they are **never used to reject** a frame.

### Module 5: Body Alignment / Posture
**Question:** Is the person standing up straight?

Not every posture measurement survives being viewed from the side. The
system only **gates** (enforces) the measurements that stay mathematically
reliable in a side view:
- **Torso tilt** — is the spine roughly vertical? (catches leaning
  forward/back or sideways)
- **Knee angle** — are the legs roughly straight? (catches sitting/crouching)
- **Hip angle** — is the person bent at the waist?
- **Neck flex** — only rejects an *extreme* forward head craning (a side
  view naturally shows some neck angle even when standing normally, so
  this threshold is deliberately generous).

**Shoulder roll** (are the shoulders level side-to-side) is calculated and
shown on screen, but **not enforced** — from directly the side, tiny
vertical camera noise makes this number swing wildly even for a person
standing perfectly straight, so gating on it would cause false rejections.

### Module 6: Camera Distance
**Question:** Is the person a good distance from the camera?
**How:** measures what fraction of the picture's height the body takes up
(`body_fill`, from the nose down to the lowest visible foot point). Too
small a fraction → person is too far away (small, hard to analyze). Too
large a fraction → person is too close (risk of head/feet getting cropped
by the camera edge as they move slightly).

### Module 7: Frame Boundary
**Question:** Is any part of the body cut off by the edge of the picture?
**How:** checks whether any confidently-visible landmark has crossed a
small safety margin near the top/bottom/left/right edge. Only
*confidently*-visible points are checked, so a low-confidence "guessed"
point (e.g. a hallucinated hidden landmark) can't wrongly trigger this.

### Module 8: Temporal Validation (Smoothing)
**Question:** Has this really been true for a while, or is it just one
lucky/unlucky frame?
**How:** three tools working together (see `filters.py` above) — a
**One Euro Filter** smooths angle numbers, a **sliding window vote**
smooths the text label, and a **streak latch** requires 12 valid frames in
a row before declaring `READY`. This is what makes the on-screen box hold
steady green instead of flickering.

### Module 9: Feedback Engine
**Question:** Out of everything that might be wrong right now, what is the
**one** most useful thing to tell the person?
**How:** every failing check contributes a "reason code". A fixed priority
list (existence → framing → distance → orientation → head → posture)
picks the single most fundamental one and shows a friendly sentence for it
— e.g. it won't tell you to fix your posture if you're not even fully in
frame yet.

---

## 9. Every Config Value Explained (the tuning bible)

All of these live in **[`profile_gate/config.py`](profile_gate/config.py)**,
inside the `ProfileGateConfig` class. You change behavior by editing the
default value there, or by constructing
`ProfileGateConfig(setting_name=new_value)` in your own code.

> **General rule:** raising a threshold usually makes the system **stricter**
> (harder to pass), lowering it makes the system **more lenient** (easier to
> pass) — except where noted, because a few settings work in the opposite
> direction (e.g. "minimum" values where lowering = more lenient, and
> "maximum" values where raising = more lenient).

### Module 1 — Single person

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `max_persons` | `1` | Maximum number of people allowed in frame at once. | Allows more than 1 person before rejecting (rarely useful for this app). | N/A (1 is already the strictest sensible value). |

### Module 2 — Full-body visibility

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `level_visibility` | `0.5` | Minimum confidence (0–1) needed for the near side of head/shoulder/hip/knee/ankle to count as "visible". | System becomes **stricter** — rejects more often when the model is even slightly unsure about a body part (e.g. in poor lighting). | System becomes **more lenient** — accepts blurrier/less confident detections, but risks accepting a genuinely bad detection. |
| `foot_visibility` | `0.3` | Same idea, but just for the "foot" level, which is naturally the least confidently detected. | Stricter about feet — good if you must clearly see full feet. | More lenient about feet — good if feet are often at the edge of frame or partially in shadow. |

### Module 3 — Side-profile orientation (core)

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `profile_yaw_min` | `60` (deg) | Minimum "how sideways is the body" angle to count as a genuine side profile (90 = perfect). | Stricter — the person must turn closer to a perfect 90°. | More lenient — accepts a less complete turn as "side profile". |
| `oblique_yaw_min` | `30.0` (deg) | Below this angle, the pose is called FRONT/BACK instead of OBLIQUE (partial turn). | Shrinks the "OBLIQUE" range — more poses get called FRONT/BACK directly. | Grows the "OBLIQUE" range — more slightly-turned poses get the more helpful "keep turning" feedback instead of a blunt "turn 90 degrees". |
| `yaw_ideal` | `90.0` (deg) | The angle considered a "perfect" profile, used only for scoring confidence (not a hard gate). | N/A — this is the physical ideal (90° is the mathematical maximum), don't change. | Same. |
| `accept_facing` | `"both"` | Which direction(s) count as valid: `"both"`, `"left"`, or `"right"`. Also settable from the terminal with `--accept-facing`. | N/A (it's a choice, not a number). | — |
| `max_shoulder_overlap` | `0.65` | How much the left/right shoulders may still be spread apart sideways (normalized) and still count as "collapsed" (side-on). | Stricter — shoulders must overlap almost perfectly (very hard for a person with natural arm swing while walking). | More lenient — accepts a less-collapsed shoulder line (useful if your subjects are walking, not standing still — natural gait needs more room here). |
| `max_hip_overlap` | `0.45` | Same idea as above, for the hips. | Stricter hip alignment required. | More lenient — allows more visible hip spread. |
| `max_torso_twist` | `25.0` (deg) | Maximum allowed difference between how much the shoulders turned vs how much the hips turned, before calling it a "twisted torso". | Stricter — rejects even a small shoulder/hip mismatch (very rigid, statue-like pose required). | More lenient — allows natural body twist (e.g. mid-stride while walking) without rejecting. |
| `min_visibility_gap` | `0.03` | A minor corroborating signal — how much visibility difference between left/right pairs counts as "one side is clearly hidden". Weighted very lightly (see `w_visibility`) because MediaPipe often "guesses" hidden points at high confidence, making this signal weak in practice. | Rarely worth tuning — has almost no practical effect since it's barely weighted. | Same. |
| `side_weight_image` | `0.40` | Vote weight for the "nose left/right of ears in the flat image" facing clue — the most trusted clue. | Trusts this clue even more relative to the others. | Trusts it less; other clues matter more. |
| `side_weight_shoulder_z` | `0.25` | Vote weight for "which shoulder is physically nearer the camera" (3D world depth). | Trusts this clue more. | Trusts it less. |
| `side_weight_hip_z` | `0.25` | Vote weight for "which hip is physically nearer the camera". | Trusts this clue more. | Trusts it less. |
| `side_weight_image_z` | `0.10` | Vote weight for the model's rougher flat-image depth estimate — the least trusted clue. | Trusts this noisier clue more (not usually recommended). | Trusts it less (recommended if you see occasional wrong-direction flips). |
| `min_side_agreement` | `0.34` | Minimum combined vote strength (0–1) needed to confidently call "LEFT" or "RIGHT". Below this, facing stays "unknown" (→ OBLIQUE / `ambiguous_facing`). | Stricter — needs stronger agreement between the 4 clues before committing to a direction; more frames end up "ambiguous". | More lenient — commits to a direction more readily, even with weaker agreement; slightly higher risk of guessing the wrong side under noisy conditions. |
| `w_yaw` | `0.35` | How much the "closeness to a perfect 90°" score contributes to overall confidence. | Confidence score cares more about a perfect turn angle. | Confidence score cares less about the exact angle. |
| `w_overlap` | `0.30` | How much the shoulder/hip "collapse" (overlap) score contributes to confidence. | Confidence score cares more about visual overlap. | Confidence score cares less about it. |
| `w_visibility` | `0.05` | How much the near/far visibility-gap score contributes to confidence (kept small deliberately — this signal is weak with this AI model). | Rarely useful to raise, since the underlying signal is noisy. | Fine to keep low or lower further. |
| `w_twist` | `0.15` | How much the "shoulders and hips turned together, not twisted" score contributes to confidence. | Confidence cares more about a non-twisted torso. | Confidence cares less about twist. |
| `w_side_agreement` | `0.15` | How much the strength of the left/right vote contributes to confidence. | Confidence cares more about a strongly-agreed facing direction. | Confidence cares less about it. |
| `min_profile_confidence` | `0.60` | Overall confidence floor (0–1). Even if every hard rule passes, a frame is still rejected here if the combined evidence is weak (reason: `low_confidence`). | Stricter overall — rejects "technically passing but weak-looking" poses more often. | More lenient overall — accepts weaker overall evidence as long as hard rules pass. |

*(Note: the `w_*` weights don't need to add up to exactly 1.0 — they get
normalized automatically — but keeping them summing to roughly 1.0 keeps
the resulting confidence score intuitively in the 0–1 range.)*

### Module 4 — Head pose

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `head_level_min` | `-30.0` (deg) | Lower limit of the accepted head-level band — how far "looking up" is still okay. | Allows looking up further before rejecting. | Rejects a smaller upward head tilt (stricter). |
| `head_level_max` | `55.0` (deg) | Upper limit — how far "looking down / nodding" is still okay. | Allows looking down further before rejecting. | Rejects sooner when the head tilts down (stricter). |
| `head_min_visibility` | `0.5` | Minimum confidence needed on face landmarks before trusting head-pose numbers at all. | Stricter — needs a clearer face detection before judging head pose (may reject more in low light). | More lenient — trusts head numbers even from a shakier face detection. |
| `max_head_spread` | `0.45` | Maximum allowed eye/ear horizontal spread (normalized by head height) — how "turned toward camera" the head may be while the body stays sideways. | Stricter — requires the head to be more perfectly in profile (rejects even a small head turn back toward camera). | More lenient — allows the head to turn more toward the camera before rejecting. |

### Module 5 — Body alignment / posture

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `max_torso_tilt` | `18.0` (deg) | Maximum allowed lean of the spine away from vertical. | More lenient — allows more leaning before rejecting. | Stricter — requires a more upright stance. |
| `min_knee_angle` | `150.0` (deg, 180=straight) | Minimum knee straightness required. | Stricter — legs must be straighter. | More lenient — allows more knee bend (e.g. elderly/injured subjects, or mid-walking-stride). |
| `min_hip_angle` | `145.0` (deg, 180=straight) | Minimum "not bent at the waist" angle. | Stricter — requires a more upright torso-to-thigh angle. | More lenient — allows some forward bend at the hips. |
| `max_neck_flex` | `80.0` (deg) | Maximum allowed forward head-craning before rejecting (deliberately generous — a side view naturally shows the head somewhat ahead of the spine). | Stricter — rejects a smaller amount of forward head craning. | More lenient — allows more forward head craning. |

### Module 6 — Camera distance

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `min_body_fill` | `0.45` | Minimum fraction of the frame height the body must fill. Below this = "too far". | Requires the person to stand closer (stricter about far distance). | Allows the person to stand farther away. |
| `max_body_fill` | `0.95` | Maximum fraction of the frame height. Above this = "too close". | Allows the person to stand closer to the camera. | Requires the person to stand farther back (stricter about closeness). |

### Module 7 — Frame boundary

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `frame_margin` | `0.02` (2% of frame) | Safety margin from every edge — landmarks inside this margin count as "about to be cropped". | Stricter — requires more empty space around the body before accepting. | More lenient — allows the body to stand closer to the picture's edge. |

### Module 8 — Temporal smoothing

| Config | Default | Meaning | If you increase it | If you decrease it |
|---|---|---|---|---|
| `oe_min_cutoff` | `1.0` (Hz) | Baseline smoothing strength for angle signals when the person is holding still. | **Smoother / less jittery** display and gating, but slightly slower to react when the person actually moves. | **Snappier**, reacts faster to real movement, but shows more jitter when the person is standing still. |
| `oe_beta` | `0.02` | How much the smoothing "opens up" (reduces lag) when the signal is changing quickly (e.g. the person is actively turning). | Reacts faster during real movement (less lag while turning), but slightly noisier during fast movement. | Stays smoother even during movement, at the cost of a bit more lag while the person is turning. |
| `oe_dcutoff` | `1.0` (Hz) | Smoothing applied to the *speed* estimate used internally by the filter above. Rarely needs tuning. | Smoother speed-estimate (indirectly smoother overall response). | Noisier speed-estimate (indirectly snappier overall response). |
| `fps_assumed` | `30.0` | Frames-per-second assumed when real frame timing isn't available (e.g. very first frame). | Assumes a faster camera (matters only in edge cases). | Assumes a slower camera. |
| `vote_window` | `7` (frames) | How many recent frames are used for the majority-vote on the orientation label. | More stable label (less flicker at OBLIQUE/PROFILE boundaries), but slower to update after a real orientation change. | Faster to update after a real change, but more prone to label flicker. |
| `stable_frames` | `12` (frames) | How many consecutive fully-valid frames are required before `READY` becomes true. | Takes longer to reach READY, but the READY signal is more trustworthy (less likely to be a fluke). | Reaches READY faster, but more likely to trigger on a brief, not-fully-stable moment. |

### Rendering

| Config | Default | Meaning | If you change it |
|---|---|---|---|
| `draw_landmark_indices` | `False` | Whether to draw the numeric index (0–32) next to each skeleton point — useful for debugging which landmark is which. | Set to `True` while developing/debugging; leave `False` for a clean end-user display. |

---

## 10. How to Run the Project

**Step 1 — Install dependencies.**

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate side-profile-gate
```
Or with plain pip (Python 3.10–3.12):
```bash
pip install -r requirements.txt
```

**Step 2 — Make sure the AI model file exists.** It's already included at
`models/pose_landmarker_heavy.task`. Only run this if that file is missing:
```bash
python download_model.py
```

**Step 3 — Run it.**

Live webcam (press ESC to quit):
```bash
python run.py --source webcam
```

Single image → annotated output image:
```bash
python run.py --source image --input path/to/photo.png --output out/annotated.png
```

Video file → annotated output video:
```bash
python run.py --source video --input input_video.MOV --output output_video.mp4
```

Only accept one facing direction (left or right):
```bash
python run.py --source webcam --accept-facing left
```

Run the automatic tests (no camera needed):
```bash
pytest -q
```

**All available `run.py` flags:**

| Flag | Values | Meaning |
|---|---|---|
| `--source` | `webcam` (default) / `image` / `video` | Where the frames come from. |
| `--input` | file path | Required for `image` / `video`. Path to the input file. |
| `--output` | file path | Where to save the annotated result. Defaults to `out/annotated.png` (image) or `output_video.mp4` (video). |
| `--model` | file path | Which `.task` model file to use. Default: `models/pose_landmarker_heavy.task`. |
| `--camera` | integer | Webcam index (0 = default camera). |
| `--accept-facing` | `both` / `left` / `right` | Which side-profile direction(s) count as valid. |

---

## 11. Tuning Cookbook — Common Situations

| I want to... | Edit this in `config.py` | Direction |
|---|---|---|
| Accept only left OR right profiles | `accept_facing` | Set to `"left"` or `"right"` (or use `--accept-facing` on the command line) |
| Accept a less-perfect turn (not fully 90°) | `profile_yaw_min` | Lower it (e.g. `50`) |
| Be lenient with elderly / injured subjects' posture | `max_torso_tilt` ↑, `min_knee_angle` ↓, `min_hip_angle` ↓ | Loosen |
| Allow more natural arm swing / walking gait | `max_shoulder_overlap` ↑, `max_hip_overlap` ↑, `max_torso_twist` ↑ | Loosen |
| Allow the subject to stand closer or farther from camera | `min_body_fill` ↓ (farther) / `max_body_fill` ↑ (closer) | Loosen |
| Make the picture smoother (less jittery numbers) | `oe_min_cutoff` ↓ | Lower |
| Make the system react faster to real movement | `oe_beta` ↑ | Raise |
| Reach the READY (green) state faster | `stable_frames` ↓ | Lower |
| Make READY more trustworthy / stable (slower but safer) | `stable_frames` ↑ | Raise |
| Reduce false "wrong direction" flips | `side_weight_image_z` ↓ (trust the noisiest clue less) | Lower |
| Run faster (lower-power computer) | Swap model file: use `pose_landmarker_full` or `pose_landmarker_lite`, update `--model` | Change the model, not config.py |
| Show landmark numbers for debugging | `draw_landmark_indices` | Set to `True` |

**Important habit:** change **one** value at a time, then re-run on your
sample video/image, and check the on-screen numbers panel (yaw, twist,
confidence, etc.) before changing the next value. Because everything is
plain geometry (no re-training needed), you see the effect of a change
immediately.

---

## 12. Testing

Run all tests with:
```bash
pytest -q
```

The tests in `tests/test_core.py` build **fake, made-up** skeleton data
(no camera, no AI model needed) and check that the math behaves correctly
— for example:
- a 90° shoulder turn is correctly read as ~90° (`shoulder_yaw_magnitude`)
- a mirror-image LEFT pose and RIGHT pose score identical confidence (no
  hidden bias toward one side)
- flipping one single noisy "facing" clue doesn't change the final answer
  (proves the voting system survives one bad signal)
- flipping a *world-depth* clue (a real physical mismatch, not noise) is
  correctly reported as `"twisted_torso"`, not just a vague "ambiguous"
- a bent knee / bent hip is correctly rejected with the right reason code
- head turned back over the shoulder is correctly rejected

This is why the project is organized the way it is: because `geometry.py`,
`filters.py`, `orientation.py` and `validators.py` never touch MediaPipe
directly, they can be fully tested with hand-built numbers, fast, and
without needing a webcam or the (large) AI model file.

---

## 13. Troubleshooting / FAQ

**Q: The system never reaches "READY", it stays orange forever.**
A: Orange means "valid this frame, but the `stable_frames` streak (default
12 frames) hasn't been completed yet". Hold the pose still for under a
second. If it never turns green even when holding still, check the status
panel for which `reasons` are still failing — they're listed on screen.

**Q: It keeps saying "turn 90 degrees" even though I'm clearly sideways.**
A: Check the on-screen `yaw` number. If it's well below 60°, the AI model
may be misjudging your shoulder positions (poor lighting, baggy clothing).
Try improving lighting, or lower `profile_yaw_min` slightly for testing.

**Q: It picks the wrong direction (says LEFT when I'm facing RIGHT).**
A: This is controlled by the 4-clue vote in Module 3. Check `min_side_agreement`
— if it's too low, a couple of noisy clues might be outvoting a correct
one. Also check lighting/visibility of your ears and nose.

**Q: Everything about posture keeps failing even though I'm standing normally.**
A: Check `torso_tilt`, `knee_angle`, `hip_angle` numbers on the panel.
Camera angle matters — a camera placed too low or too high changes what
"straight" looks like. You can loosen `max_torso_tilt`, `min_knee_angle`,
`min_hip_angle` in `config.py`.

**Q: It's too slow / low FPS.**
A: The pose detection model itself (BlazePose Heavy) is the slow part —
everything downstream is a handful of simple angle calculations and is
essentially free by comparison. Switch to `pose_landmarker_full` or
`pose_landmarker_lite` (see `download_model.py` and the `--model` flag).

**Q: Can I show the feedback text in Bangla or another language?**
A: OpenCV's on-screen text drawing (`cv2.putText`, used in `render.py`)
cannot render Bangla/Unicode script — only basic Latin characters. To show
localized text, you'd render the message with a library like PIL and a
suitable font instead of `cv2.putText`. The internal message **keys** in
`feedback.py` (e.g. `"bent_knee"`) would stay exactly the same — you'd
just translate the **values** (the human-readable strings) for display.

---

## 14. Glossary

| Term | Plain-English meaning |
|---|---|
| **Landmark** | One tracked body point (e.g. "left shoulder"). BlazePose gives 33 of these. |
| **Yaw** | Sideways rotation angle. 0° = facing the camera, 90° = a perfect side view. |
| **Pitch** | Up/down head tilt (nodding). |
| **Roll** | Head tilting toward one shoulder (like tilting your head to listen). |
| **World coordinates** | 3D, real-world-scale (meters) positions, independent of camera distance. Used for angle decisions. |
| **Image coordinates** | Flat, 2D positions on the picture itself (0 to 1 across, 0 to 1 down). Used for framing decisions (in-frame, distance). |
| **Visibility** | The AI model's own confidence (0–1) that a given landmark is genuinely, visibly there. |
| **Gate / Gating** | The act of accepting or rejecting a frame based on a rule (like a checkpoint gate). |
| **Threshold** | A cutoff number used to decide pass/fail (e.g. "angle must be ≥ 60°"). |
| **Confidence score** | A single 0–1 number summarizing how strongly the evidence supports a decision, blended from several individual signals. |
| **Jitter** | Small, meaningless frame-to-frame wobble in a number, caused by tiny AI model noise, not real movement. |
| **One Euro Filter** | A smoothing technique that removes jitter when a value is still, but reacts quickly when it's genuinely changing. |
| **Streak Latch** | A rule that requires something to be true for many frames in a row before trusting it. |
| **solvePnP** | A classic (non-AI, pure-math) OpenCV function that estimates 3D rotation from a small set of known 2D↔3D point pairs. |
| **BlazePose GHUM** | The specific body-tracking AI model used in this project (made by Google/MediaPipe). |
| **Orchestrator** | The "manager" piece of code (`gate.py`) that calls every other module in the right order. |
