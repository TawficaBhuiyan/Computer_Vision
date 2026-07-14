"""Small, dependency-light helpers for reading BlazePose landmark arrays.

Isolating array access here keeps geometry / orientation / validators free of
index bookkeeping and makes the "which landmark, which column" decisions
testable in one place.

Array shapes used throughout the codebase:
    image : (33, 4) -> columns [x, y, z, visibility]; x, y normalized to [0, 1]
    world : (33, 3) -> columns [x, y, z] in meters, hip-centered
"""
from typing import Tuple

import numpy as np

from .config import (
    NOSE, LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HEEL, RIGHT_HEEL, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
)


def xy(image: np.ndarray, idx: int) -> np.ndarray:
    """Normalized (x, y) of one landmark."""
    return image[idx, :2]


def visibility(image: np.ndarray, idx: int) -> float:
    return float(image[idx, 3])


def level_confidence(image: np.ndarray, left_idx: int, right_idx: int) -> float:
    """Confidence that a vertical body level is present at all.

    We take the MAX of the two sides: in a profile the near side is confident
    even when the far side is occluded, so the level is genuinely 'covered'.
    """
    return max(visibility(image, left_idx), visibility(image, right_idx))


def in_frame(image: np.ndarray, idx: int, margin: float) -> bool:
    x, y = image[idx, 0], image[idx, 1]
    return (margin <= x <= 1.0 - margin) and (margin <= y <= 1.0 - margin)


def head_top_y(image: np.ndarray, min_visibility: float) -> float:
    """Smallest (topmost) y among confidently-visible head landmarks.

    Returns 1.0 (frame bottom, i.e. "no crop risk") if none are confident, so
    a momentarily-unreliable frame never falsely reads as cropped.
    """
    ids = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
    ys = [image[i, 1] for i in ids if image[i, 3] >= min_visibility]
    return min(ys) if ys else 1.0


def foot_bottom_y(image: np.ndarray, min_visibility: float) -> float:
    """Largest (bottommost) y among confidently-visible ankle/foot landmarks.

    Returns 0.0 (frame top, i.e. "no crop risk") if none are confident.
    """
    ids = (LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL,
          LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX)
    ys = [image[i, 1] for i in ids if image[i, 3] >= min_visibility]
    return max(ys) if ys else 0.0


def body_fill(image: np.ndarray) -> float:
    """Normalized vertical extent of the body: head-top proxy to lowest foot.

    Uses the nose as the head-top proxy and the lowest of the foot landmarks as
    the bottom. Robust to which foot is nearer the camera.
    """
    top = image[NOSE, 1]
    feet = [image[LEFT_ANKLE, 1], image[RIGHT_ANKLE, 1],
            image[LEFT_HEEL, 1], image[RIGHT_HEEL, 1],
            image[LEFT_FOOT_INDEX, 1], image[RIGHT_FOOT_INDEX, 1]]
    bottom = max(feet)
    return float(bottom - top)


def facing_from_image(image: np.ndarray) -> int:
    """Image-space facing cue from nose position relative to the ear midpoint.

    Returns -1 if the subject faces the LEFT edge of the frame (nose left of
    ears), +1 if facing RIGHT, 0 if ambiguous. This cue does not depend on any
    world-depth sign convention, so it is the primary facing signal.
    """
    ear_mid_x = (image[LEFT_EAR, 0] + image[RIGHT_EAR, 0]) / 2.0
    dx = image[NOSE, 0] - ear_mid_x
    if abs(dx) < 1e-3:
        return 0
    return 1 if dx > 0 else -1


def torso_scale(image: np.ndarray) -> float:
    """Image-space distance from shoulder-midpoint to hip-midpoint.

    Reference length used to normalize the overlap / width-ratio cues below so
    they are independent of subject distance and image resolution.
    """
    shoulder_mid = (xy(image, LEFT_SHOULDER) + xy(image, RIGHT_SHOULDER)) / 2.0
    hip_mid = (xy(image, LEFT_HIP) + xy(image, RIGHT_HIP)) / 2.0
    scale = float(np.linalg.norm(shoulder_mid - hip_mid))
    return scale if scale > 1e-6 else 1e-6


def x_overlap_ratio(image: np.ndarray, left_idx: int, right_idx: int,
                    scale: float) -> float:
    """Horizontal separation of a left/right pair, normalized by torso scale.

    In a true side profile the far landmark sits almost directly behind the
    near one, so their x-coordinates nearly coincide (ratio ~0). In a frontal
    or back view the pair spans the full body width (ratio ~ shoulder width /
    torso height, well above 0). This same normalized value doubles as the
    "width ratio" cue: it IS how collapsed that pair's width is relative to
    the body's own vertical scale.
    """
    dx = abs(image[left_idx, 0] - image[right_idx, 0])
    return float(dx / scale)


def visibility_gap(image: np.ndarray, left_idx: int, right_idx: int) -> float:
    """Absolute visibility difference between a left/right landmark pair.

    A true profile self-occludes one side, so a genuinely side-on subject
    shows a LARGE gap here (only one side "clearly visible"). A small gap
    means both sides are equally visible, i.e. a frontal/back view.
    """
    return abs(visibility(image, left_idx) - visibility(image, right_idx))


def bounding_box(image: np.ndarray, width: int, height: int,
                 pad: float = 0.04) -> Tuple[int, int, int, int]:
    """Axis-aligned pixel bounding box (x1, y1, x2, y2) over all landmarks."""
    xs = np.clip(image[:, 0], 0.0, 1.0)
    ys = np.clip(image[:, 1], 0.0, 1.0)
    x1 = max(0.0, xs.min() - pad) * width
    y1 = max(0.0, ys.min() - pad) * height
    x2 = min(1.0, xs.max() + pad) * width
    y2 = min(1.0, ys.max() + pad) * height
    return int(x1), int(y1), int(x2), int(y2)
