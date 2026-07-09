"""Pure geometric computations for body orientation.

Depends only on numpy, so every function here is deterministic and unit-testable
without a camera or MediaPipe. All angles are derived from WORLD landmarks
(metric, hip-centered) to avoid the aspect-ratio distortion you get from taking
angles in normalized image space.

Array shapes expected:
    world : (33, 3) -> columns [x, y, z] in meters
"""
import math

import numpy as np

from .config import LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP


def _deg(opposite: float, adjacent: float) -> float:
    """Magnitude of arctan(opposite / adjacent) in degrees. Sign-agnostic, so
    it never depends on the sign convention of the coordinate axes."""
    return math.degrees(math.atan2(abs(opposite), abs(adjacent)))


def facing_yaw(world: np.ndarray) -> float:
    """Body yaw from the shoulder line. 0 deg = frontal, ~90 deg = side-on."""
    dx, _, dz = world[RIGHT_SHOULDER] - world[LEFT_SHOULDER]
    return _deg(dz, dx)


def shoulder_roll(world: np.ndarray) -> float:
    """Shoulder-line tilt away from horizontal. 0 deg = shoulders level."""
    dx, dy, dz = world[RIGHT_SHOULDER] - world[LEFT_SHOULDER]
    return _deg(dy, math.hypot(dx, dz))


def torso_tilt(world: np.ndarray) -> float:
    """Torso deviation from vertical. 0 deg = upright.
    Catches both forward/back lean and sideways lean in one number."""
    mid_shoulder = (world[LEFT_SHOULDER] + world[RIGHT_SHOULDER]) / 2
    mid_hip = (world[LEFT_HIP] + world[RIGHT_HIP]) / 2
    tx, ty, tz = mid_shoulder - mid_hip
    return _deg(math.hypot(tx, tz), ty)
