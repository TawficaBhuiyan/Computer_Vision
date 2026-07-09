"""Head-pose estimation from the pose model's own face keypoints.

No second model. The Pose Landmarker already returns coarse face keypoints
(nose, eyes, ears, mouth). We match them to a small 3D head model and solve with
cv2.solvePnP to recover head yaw / pitch / roll in degrees. This is much lighter
than running a separate Face Landmarker, at the cost of some precision (pose face
keypoints are coarser than a 478-point face mesh, which the EMA smoothing damps).

Only the MAGNITUDE of the angles is used downstream, so a global sign flip from
a mirrored feed does not affect the "is the head frontal" decision.
"""
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import (
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT,
)

# Face keypoints taken from the pose landmarks, in a fixed semantic order.
_FACE_IDS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT)

# Approximate 3D head model (millimeters), OpenCV camera convention:
# +x right, +y down, +z into the scene. Origin at the nose tip. Subject-left
# features sit at +x (image right); ears sit far back (+z) for good yaw leverage.
# Absolute scale is irrelevant to the recovered angles.
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # nose
    (35.0,  -32.0,  28.0),    # left eye
    (-35.0, -32.0,  28.0),    # right eye
    (85.0,  -18.0, 100.0),    # left ear
    (-85.0, -18.0, 100.0),    # right ear
    (28.0,   38.0,  30.0),    # left mouth corner
    (-28.0,  38.0,  30.0),    # right mouth corner
], dtype=np.float64)


def _wrap(angle: float) -> float:
    """Fold an angle into [-90, 90] so a frontal head reads near 0 regardless of
    a +/-180 offset from the Euler decomposition."""
    a = (angle + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return a


def _euler_from_rmat(rmat: np.ndarray) -> Tuple[float, float, float]:
    """Rotation matrix -> (pitch, yaw, roll) in degrees."""
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:  # gimbal lock
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return (_wrap(math.degrees(pitch)),
            _wrap(math.degrees(yaw)),
            _wrap(math.degrees(roll)))


def estimate_head_pose(
    image: np.ndarray, width: int, height: int, min_visibility: float = 0.5
) -> Optional[Tuple[float, float, float]]:
    """Return (yaw, pitch, roll) in degrees, or None if the face keypoints are
    not reliable this frame.

    image : (33, 4) pose landmarks [x, y, z, visibility]; x, y normalized to [0, 1].
    """
    if any(image[i, 3] < min_visibility for i in _FACE_IDS):
        return None

    image_points = np.array(
        [(image[i, 0] * width, image[i, 1] * height) for i in _FACE_IDS],
        dtype=np.float64,
    )

    # Pinhole approximation: focal length ~ image width, principal point at
    # the image center, no lens distortion.
    focal = float(width)
    camera_matrix = np.array([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _euler_from_rmat(rmat)
    return yaw, pitch, roll
