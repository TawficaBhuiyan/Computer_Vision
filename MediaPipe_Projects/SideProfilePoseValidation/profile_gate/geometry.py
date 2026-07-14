"""Pure geometric computations for body orientation and posture.

Depends only on numpy, so every function here is deterministic and unit-testable
without a camera or MediaPipe. Angles are derived from WORLD landmarks (metric,
hip-centered) which removes the aspect-ratio and perspective distortion you get
from measuring angles in normalized image space.

Conventions
-----------
world : (33, 3) array, columns [x, y, z] in meters.
        x -> subject right(+) in image, y -> down(+), z -> depth.
All returned angles are in degrees.
"""
import math
from typing import Tuple

import numpy as np

from .config import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
)


# --------------------------------------------------------------------------- #
# low-level helpers                                                           #
# --------------------------------------------------------------------------- #
def _deg(opposite: float, adjacent: float) -> float:
    """|arctan(opposite / adjacent)| in degrees. Sign-agnostic."""
    return math.degrees(math.atan2(abs(opposite), abs(adjacent)))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex ``b`` formed by segments b->a and b->c, degrees.

    180 deg means a, b, c are collinear (e.g. a fully straight knee).
    """
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return 180.0
    cos = float(np.dot(ba, bc) / (nba * nbc))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


# --------------------------------------------------------------------------- #
# body orientation                                                            #
# --------------------------------------------------------------------------- #
def signed_shoulder_yaw(world: np.ndarray) -> float:
    """Body yaw from the shoulder line in the horizontal (x-z) plane.

    0 deg  -> shoulders span the image (frontal / back view)
    ~90 deg-> shoulders span depth (true side profile)
    The SIGN encodes rotation direction (which shoulder is toward the camera).
    """
    d = world[RIGHT_SHOULDER] - world[LEFT_SHOULDER]
    dx, dz = d[0], d[2]
    return math.degrees(math.atan2(dz, dx))


def shoulder_yaw_magnitude(world: np.ndarray) -> float:
    """How side-on the shoulders are, folded into [0, 90]. 90 = perfect side."""
    d = world[RIGHT_SHOULDER] - world[LEFT_SHOULDER]
    return _deg(d[2], d[0])


def hip_yaw_magnitude(world: np.ndarray) -> float:
    """Same as shoulder yaw but for the hip line. Corroborates the torso turn."""
    d = world[RIGHT_HIP] - world[LEFT_HIP]
    return _deg(d[2], d[0])


def signed_hip_yaw(world: np.ndarray) -> float:
    """Signed hip yaw, mirroring signed_shoulder_yaw. Used to detect twist:
    a clean turn rotates the shoulder line and the hip line together, so the
    two signed yaws should closely agree; a twisted torso decouples them.
    """
    d = world[RIGHT_HIP] - world[LEFT_HIP]
    return math.degrees(math.atan2(d[2], d[0]))


def torso_twist(signed_shoulder_yaw_deg: float, signed_hip_yaw_deg: float) -> float:
    """Angular disagreement between shoulder-line and hip-line yaw, in [0, 180].

    Folds the difference into (-180, 180] before taking the magnitude so a
    wraparound near +-180 deg doesn't read as a huge, spurious twist.
    """
    diff = (signed_shoulder_yaw_deg - signed_hip_yaw_deg + 180.0) % 360.0 - 180.0
    return abs(diff)


def _depth_sign(near: np.ndarray, far: np.ndarray, axis: int = 2) -> int:
    """sign(near[axis] - far[axis]), 0 if the two are indistinguishable."""
    d = near[axis] - far[axis]
    if abs(d) < 1e-6:
        return 0
    return 1 if d > 0 else -1


def shoulder_depth_sign(world: np.ndarray) -> int:
    """sign(z_left_shoulder - z_right_shoulder): which shoulder is nearer camera.

    A world-space facing cue that is independent of the image-space nose cue.
    """
    return _depth_sign(world[LEFT_SHOULDER], world[RIGHT_SHOULDER])


def hip_depth_sign(world: np.ndarray) -> int:
    """sign(z_left_hip - z_right_hip): which hip is nearer the camera.

    Corroborates shoulder_depth_sign from an independent landmark pair.
    """
    return _depth_sign(world[LEFT_HIP], world[RIGHT_HIP])


def shoulder_roll(world: np.ndarray) -> float:
    """Shoulder-line tilt away from horizontal. 0 deg = shoulders level."""
    d = world[RIGHT_SHOULDER] - world[LEFT_SHOULDER]
    return _deg(d[1], math.hypot(d[0], d[2]))


def torso_tilt(world: np.ndarray) -> float:
    """Torso deviation from vertical. 0 deg = upright.

    Catches forward/back lean and sideways lean in a single number by comparing
    the shoulder-midpoint -> hip-midpoint vector against the vertical axis.
    """
    mid_shoulder = (world[LEFT_SHOULDER] + world[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world[LEFT_HIP] + world[RIGHT_HIP]) / 2.0
    t = mid_shoulder - mid_hip
    return _deg(math.hypot(t[0], t[2]), t[1])


# --------------------------------------------------------------------------- #
# posture joint angles (near-side indices supplied by the caller)             #
# --------------------------------------------------------------------------- #
def knee_angle(world: np.ndarray, hip: int, knee: int, ankle: int) -> float:
    """Interior knee angle (hip-knee-ankle). ~180 = straight leg."""
    return joint_angle(world[hip], world[knee], world[ankle])


def hip_angle(world: np.ndarray, shoulder: int, hip: int, knee: int) -> float:
    """Interior hip angle (shoulder-hip-knee). ~180 = not bent at the waist."""
    return joint_angle(world[shoulder], world[hip], world[knee])


def neck_flex(world: np.ndarray, ear: int) -> float:
    """Head-forward/back flexion as DEVIATION from a straight spine-to-head line.

    We take the interior angle (hip-mid) - (shoulder-mid) - (ear) and return
    ``180 - angle`` so that 0 = ear stacked over the spine and larger values mean
    the head is craned forward or thrown back. In a side profile the ear sits
    naturally ahead of the spine, so this reads non-zero even when upright; only
    an extreme value indicates a genuinely poor posture.
    """
    mid_shoulder = (world[LEFT_SHOULDER] + world[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world[LEFT_HIP] + world[RIGHT_HIP]) / 2.0
    return 180.0 - joint_angle(mid_hip, mid_shoulder, world[ear])
