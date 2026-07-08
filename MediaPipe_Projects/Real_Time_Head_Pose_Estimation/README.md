# Real-Time Head Pose Estimation

Estimate the 3D orientation of a human head (pitch, yaw, roll) in real time from a standard webcam using **MediaPipe Face Mesh** and **OpenCV**. The application detects facial landmarks, solves the Perspective-n-Point (PnP) problem to recover head rotation, and overlays live direction feedback ("Looking Left / Right / Up / Down / Forward") along with a projected gaze line. Sessions can be recorded to an MP4 file.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-green">
  <img alt="MediaPipe" src="https://img.shields.io/badge/MediaPipe-0.10%2B-orange">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [Output](#output)
- [Project Structure](#project-structure)
- [Configuration & Customization](#configuration--customization)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Real-time face landmark detection** with MediaPipe Face Mesh (468 landmarks).
- **6-DoF head pose estimation** — recovers pitch (x), yaw (y), and roll (z) angles.
- **Live directional feedback** — classifies gaze as *Looking Left / Right / Up / Down / Forward*.
- **Gaze/nose projection line** drawn from the nose to visualize head direction.
- **On-screen telemetry** — displays x/y/z rotation angles and live FPS.
- **Video recording** — saves the annotated feed to `output.mp4`.
- **Graceful frame handling** — skips dropped camera frames instead of crashing.

---

## How It Works

The pipeline runs once per frame:

1. **Capture & preprocess** — a frame is read from the webcam, flipped horizontally for a natural selfie view, and converted from BGR to RGB (the format MediaPipe expects).
2. **Landmark detection** — MediaPipe Face Mesh returns normalized coordinates for every facial landmark.
3. **Point selection** — six stable landmarks are chosen as reference points for pose solving:

   | Landmark index | Facial feature      |
   |---------------:|---------------------|
   | 1              | Nose tip            |
   | 33             | Left eye corner     |
   | 263            | Right eye corner    |
   | 61             | Left mouth corner   |
   | 291            | Right mouth corner  |
   | 199            | Chin                |

4. **Camera model** — a pinhole camera matrix is built from the image dimensions, assuming focal length ≈ image width and zero lens distortion.
5. **Solve PnP** — `cv2.solvePnP` estimates the rotation and translation that map the 3D face model to the observed 2D points.
6. **Extract angles** — the rotation vector is converted to a rotation matrix via Rodrigues, then decomposed with `cv2.RQDecomp3x3` into Euler angles (pitch, yaw, roll).
7. **Interpret & render** — angles are thresholded to a direction label, a projected nose line is drawn, and telemetry is overlaid on the frame.

> **Note:** This is a monocular estimation using an assumed (uncalibrated) camera matrix. Angles are approximate and best used for relative/directional cues rather than precise measurement. For metric accuracy, calibrate your camera (see [Configuration](#configuration--customization)).

---

## Requirements

- Python 3.8 or newer
- A working webcam
- The following Python packages:

```
opencv-python
mediapipe>=0.10
numpy
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Real_Time_Head_Pose_Estimation.git
cd Real_Time_Head_Pose_Estimation

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install opencv-python mediapipe numpy
```

Or, if a `requirements.txt` is included:

```bash
pip install -r requirements.txt
```

**Suggested `requirements.txt`:**

```
opencv-python
mediapipe>=0.10
numpy
```

---

## Usage

Run the script from the project directory:

```bash
python HeadPoseEstimation.py
```

A window titled **Head Pose Estimation** opens showing your webcam feed with the pose overlay. The window stays open as long as you want.

---

## Controls

| Key   | Action              |
|-------|---------------------|
| `ESC` | Quit the application |

The recording starts automatically when the app launches and stops (and finalizes the file) when you press `ESC`.

---

## Output

- **`output.mp4`** — the recorded, annotated session, saved to the project directory.
- The recording is written at a fixed 20 FPS. If playback speed looks off, adjust the FPS value passed to `cv2.VideoWriter` to better match your camera's actual rate.

---

## Project Structure

```
Real_Time_Head_Pose_Estimation/
├── HeadPoseEstimation.py   # Main application
├── output.mp4              # Generated recording (created at runtime)
├── requirements.txt        # Dependencies (optional)
└── README.md               # This file
```

---

## Configuration & Customization

**Detection sensitivity** — tune the confidence thresholds when creating the Face Mesh:

```python
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

**Direction thresholds** — the ±10° cutoffs control how far you must turn before a label changes. Lower them for a more sensitive response, raise them to reduce jitter:

```python
if y < -10:   text = "Looking Left"
elif y > 10:  text = "Looking Right"
elif x < -10: text = "Looking Down"
elif x > 10:  text = "Looking Up"
else:         text = "Forward"
```

**Recording FPS / filename** — change in the VideoWriter setup:

```python
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_w, frame_h))
```

**Camera calibration (advanced)** — for accurate metric angles, replace the assumed camera matrix and zero distortion coefficients with values obtained from OpenCV's chessboard calibration routine.

---

## Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Newer MediaPipe releases (0.10+) don't expose `solutions` at the top level. Import it explicitly:

```python
from mediapipe import solutions as mp_solutions
mp_face_mesh = mp_solutions.face_mesh
mp_drawing = mp_solutions.drawing_utils
```

Also make sure no file named `mediapipe.py` exists in your working directory — it will shadow the installed package.

**The window opens and closes instantly**
Usually the camera isn't being read. The app guards against this by skipping empty frames:

```python
success, image = cap.read()
if not success:
    print("Camera frame not received. Skipping...")
    continue
```

If it persists, try a different camera index — `cv2.VideoCapture(1)` instead of `0`.

**`TypeError` on `FaceMesh(...)` arguments**
Argument names are case-sensitive. Use `min_detection_confidence`, not `Min_detection_confidence`.

**Camera won't open / black screen**
- Close any other app using the webcam (Zoom, Teams, browser tabs).
- On Windows, grant camera permission under *Settings → Privacy → Camera*.
- Verify the correct camera index.

**Low FPS**
Face Mesh is CPU-intensive. Close background apps, reduce the capture resolution, or run on a machine with more cores.

---

## Roadmap

- [ ] Real camera calibration for metric-accurate angles
- [ ] Multi-face pose estimation
- [ ] Smoothing filter (e.g., moving average / Kalman) to reduce angle jitter
- [ ] Toggle recording with a hotkey
- [ ] Configurable settings via command-line arguments
- [ ] Export pose angles to CSV for analysis

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgements

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — face landmark detection.
- [OpenCV](https://opencv.org/) — computer vision and PnP solving.
- The classic solvePnP-based head pose technique widely used in the CV community.