"""Landmark index constants and all tunable thresholds.

Keeping every magic number here means tuning the gate never requires touching
the logic in geometry.py, head.py, or gate.py.
"""
from dataclasses import dataclass

# --- BlazePose 33-landmark indices. "LEFT" means the SUBJECT's left, which
# appears on the right side of a mirrored webcam feed. The pose model already
# returns face keypoints (0-10), so no separate face model is needed. ---
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Landmarks that must all be visible and in-frame for a valid full-body capture.
FULL_BODY_LANDMARKS = (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
)


@dataclass(frozen=True)
class GateConfig:
    """All tunable thresholds. Adjust these to your patient population."""

    # framing / visibility
    visibility: float = 0.6       # minimum per-landmark confidence
    frame_margin: float = 0.02    # normalized edge margin (clipping guard)
    max_body_fill: float = 0.92   # body-height / frame-height ceiling (too close)

    # body orientation, in degrees
    max_yaw: float = 20.0         # facing camera (shoulder-line depth angle)
    max_roll: float = 10.0        # shoulders level
    max_torso_tilt: float = 12.0  # torso vertical

    # head orientation, in degrees (solvePnP on the pose model's face keypoints)
    max_head_yaw: float = 15.0        # head turned left/right
    max_head_pitch: float = 15.0      # head nodding up/down
    head_min_visibility: float = 0.5  # min confidence on the face keypoints

    # temporal smoothing
    stable_frames: int = 15       # consecutive good frames required for READY
    ema_alpha: float = 0.4        # angle smoothing (0 = heavy smoothing, 1 = none)
