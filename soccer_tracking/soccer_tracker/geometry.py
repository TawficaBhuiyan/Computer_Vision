"""
geometry.py
===========
Pure, stateless box maths used everywhere in the tracker. No tracking logic
lives here - just NumPy functions that convert, inflate, and compare boxes.

Box formats:
    xyxy   = [x1, y1, x2, y2]   (top-left + bottom-right corners)
    cxcywh = [cx, cy, w, h]     (center + size)
"""

import numpy as np


def xyxy_to_cxcywh(boxes):
    """Corner format -> center+size format."""
    boxes = np.asarray(boxes, dtype=np.float64)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([cx, cy, w, h], axis=1)


def cxcywh_to_xyxy(boxes):
    """Center+size format -> corner format."""
    boxes = np.asarray(boxes, dtype=np.float64)
    x1 = boxes[:, 0] - boxes[:, 2] / 2.0
    y1 = boxes[:, 1] - boxes[:, 3] / 2.0
    x2 = boxes[:, 0] + boxes[:, 2] / 2.0
    y2 = boxes[:, 1] + boxes[:, 3] / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def buffer_boxes(boxes_xyxy, ratio):
    """
    C-BIoU core operation. Inflate every box by `ratio` of its own size on all
    four sides. ratio=0.3 grows a box ~1.6x in each dimension. This is what
    creates artificial overlap for objects that moved between frames.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).copy()
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    boxes[:, 0] -= ratio * w
    boxes[:, 1] -= ratio * h
    boxes[:, 2] += ratio * w
    boxes[:, 3] += ratio * h
    return boxes


def expand_abs(boxes_xyxy, margins):
    """
    Grow each box by an ABSOLUTE per-box pixel margin on all sides.
    Used for the velocity-aware buffer (fast tracks get a larger search box).
    `margins` is a 1-D array, one value per box.
    """
    out = np.asarray(boxes_xyxy, dtype=np.float64).copy()
    m = np.asarray(margins, dtype=np.float64).reshape(-1)
    out[:, 0] -= m
    out[:, 1] -= m
    out[:, 2] += m
    out[:, 3] += m
    return out


def iou_batch(a, b):
    """
    Vectorized IoU between two sets of boxes.
    a: (T,4), b: (D,4)  ->  (T,D) matrix of IoU values in [0, 1].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def center_dist_batch(a, b):
    """
    Pairwise Euclidean distance between the centers of two box sets.
    a: (T,4), b: (D,4)  ->  (T,D) matrix of pixel distances.
    Used as a fallback similarity when IoU drops to zero (fast motion).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ca = (a[:, :2] + a[:, 2:]) / 2.0
    cb = (b[:, :2] + b[:, 2:]) / 2.0
    diff = ca[:, None, :] - cb[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))
