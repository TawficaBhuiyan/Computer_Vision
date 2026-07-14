"""Module 4 - Head-pose estimation from the pose model's own face keypoints.

No second model. BlazePose already returns coarse face keypoints (nose, eyes,
ears, mouth); we match them to a small 3D head model and solve with cv2.solvePnP
to recover head yaw / pitch / roll in degrees.

Design note for the SIDE-PROFILE use case
------------------------------------------
At ~90 deg of true head yaw the far-side face keypoints (far eye, far ear, far
mouth corner) are self-occluded, and the 7-point PnP becomes ill-conditioned in
yaw. We therefore split responsibilities:

* solvePnP  -> PITCH and ROLL only (well conditioned even side-on): "is the head
               level and not nodding".
* geometry  -> the SIDEWAYS-FACING requirement and HEAD/BODY ALIGNMENT, taken
               from the image-space nose-vs-ear vector, which stays reliable at
               large yaw and needs no depth prior.

Only angle MAGNITUDES are used downstream, so a global sign flip from a mirrored
webcam feed does not change the decision.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import (
    FACE_IDS, NOSE, LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE,
    MOUTH_LEFT, MOUTH_RIGHT,
)

# face landmarks whose vertical extent defines the head height (yaw-invariant)
_HEAD_H_IDS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
               MOUTH_LEFT, MOUTH_RIGHT)

# Approximate 3D head model (mm), OpenCV camera convention (+x right, +y down,
# +z into scene). Origin at the nose tip. Absolute scale is irrelevant to angles.
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # nose
    (35.0,  -32.0,  28.0),    # left eye
    (-35.0, -32.0,  28.0),    # right eye
    (85.0,  -18.0, 100.0),    # left ear
    (-85.0, -18.0, 100.0),    # right ear
    (28.0,   38.0,  30.0),    # left mouth corner
    (-28.0,  38.0,  30.0),    # right mouth corner
], dtype=np.float64)


@dataclass
class HeadPose:
    yaw: Optional[float]        # solvePnP yaw (unreliable near +-90; display only)
    pitch: Optional[float]      # solvePnP pitch (display only in strong profile)
    roll: Optional[float]       # solvePnP roll  (display only in strong profile)
    facing: int                 # geometric: -1 left, +1 right, 0 unknown
    level: Optional[float]      # geometric head-level angle (deg); nose vs near ear
    spread: Optional[float]     # head-yaw cue: eye/ear horizontal spread / head h
    ok: bool                    # face keypoints were reliable this frame


def _wrap(angle: float) -> float:
    a = (angle + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return a


def _euler_from_rmat(rmat: np.ndarray) -> Tuple[float, float, float]:
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return (_wrap(math.degrees(pitch)),
            _wrap(math.degrees(yaw)),
            _wrap(math.degrees(roll)))


def estimate_head_pose(image: np.ndarray, width: int, height: int,
                       min_visibility: float = 0.5) -> HeadPose:
    """image : (33, 4) pose landmarks [x, y, z, visibility], x,y normalized."""
    # geometric facing from the nose-vs-ear vector (reliable at any yaw)
    ear_mid_x = (image[LEFT_EAR, 0] + image[RIGHT_EAR, 0]) / 2.0
    dxf = image[NOSE, 0] - ear_mid_x
    facing = 0 if abs(dxf) < 1e-3 else (1 if dxf > 0 else -1)

    # geometric head-level angle: nose relative to the NEAR (visible) ear, in
    # pixel space. +deg = nose below ear (normal), -deg = nose above (looking up),
    # large +deg = looking down. This stays reliable at a full side profile where
    # solvePnP does not, so it is what the head check actually gates on.
    near_ear = LEFT_EAR if image[LEFT_EAR, 3] >= image[RIGHT_EAR, 3] else RIGHT_EAR
    ndx = (image[NOSE, 0] - image[near_ear, 0]) * width
    ndy = (image[NOSE, 1] - image[near_ear, 1]) * height
    level = math.degrees(math.atan2(ndy, abs(ndx))) if abs(ndx) > 1e-6 else 0.0

    # head-yaw cue: when the head is truly side-on, the two eyes (and the two
    # ears) project to almost the same x, so their horizontal spread collapses.
    # As the head rotates toward/away from the camera the far eye/ear swings out
    # and the spread grows. Normalize by head height, which is ~invariant to
    # head yaw, so the ratio is scale- and distance-independent.
    ys = [image[i, 1] for i in _HEAD_H_IDS]
    head_h = (max(ys) - min(ys)) * height
    if head_h > 1e-6:
        eye_sep = abs(image[LEFT_EYE, 0] - image[RIGHT_EYE, 0]) * width
        ear_sep = abs(image[LEFT_EAR, 0] - image[RIGHT_EAR, 0]) * width
        spread = spread = max(eye_sep, ear_sep) / head_h
    else:
        spread = None

    if any(image[i, 3] < min_visibility for i in FACE_IDS):
        return HeadPose(None, None, None, facing, level, spread, ok=False)

    image_points = np.array(
        [(image[i, 0] * width, image[i, 1] * height) for i in FACE_IDS],
        dtype=np.float64,
    )
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
        return HeadPose(None, None, None, facing, level, spread, ok=False)

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _euler_from_rmat(rmat)
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll, facing=facing,
                    level=level, spread=spread, ok=True)