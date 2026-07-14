"""Frame overlay rendering: skeleton, bounding box and the status panel.

Every rendered frame carries the full spec readout: skeleton, bounding box,
confidence, side-profile status, body orientation, head pose, validation result,
failure reasons, corrective guidance and overall confidence score.
"""
from typing import List

import cv2
import numpy as np

from .config import POSE_CONNECTIONS, Orientation
from .gate import GateResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX

_GREEN = (0, 200, 0)
_ORANGE = (0, 165, 255)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_PANEL = (28, 28, 28)


def _status_color(r: GateResult):
    if r.ready:
        return _GREEN
    if r.valid:
        return _ORANGE
    return _RED


def _draw_skeleton(frame, image: np.ndarray, w: int, h: int, color) -> None:
    for a, b in POSE_CONNECTIONS:
        if image[a, 3] < 0.3 or image[b, 3] < 0.3:
            continue
        pa = (int(image[a, 0] * w), int(image[a, 1] * h))
        pb = (int(image[b, 0] * w), int(image[b, 1] * h))
        cv2.line(frame, pa, pb, color, 2)
    for i in range(image.shape[0]):
        if image[i, 3] < 0.3:
            continue
        p = (int(image[i, 0] * w), int(image[i, 1] * h))
        cv2.circle(frame, p, 3, _WHITE, -1)


def _panel(frame, lines: List[tuple], x: int = 15, y: int = 30) -> None:
    """lines: list of (text, color). Draws a translucent backing box."""
    pad = 10
    line_h = 26
    box_w = 430
    box_h = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - line_h),
                  (x - pad + box_w, y - line_h + box_h), _PANEL, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x, y + i * line_h - 6),
                    _FONT, 0.55, color, 1, cv2.LINE_AA)


def draw(frame: np.ndarray, result: GateResult, cfg) -> np.ndarray:
    h, w = frame.shape[:2]
    color = _status_color(result)

    if result.person_image is not None:
        _draw_skeleton(frame, result.person_image, w, h, color)

    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    m = result.metrics
    status = "READY" if result.ready else ("VALID" if result.valid else "INVALID")
    hl = m.get("head_level")
    hf = m.get("head_facing")
    hs = m.get("head_spread")
    head_txt = (f"level:{hl} spread:{hs} facing:{hf:+d}"
                if hl is not None else "n/a")

    lines = [
        (f"Side Profile Gate            {status}", color),
        (f"Orientation : {result.label}   (facing {result.facing:+d})", _WHITE),
        (f"Body yaw    : {m.get('yaw')} deg   hip:{m.get('hip_yaw')}", _WHITE),
        (f"Head pose   : {head_txt}", _WHITE),
        (f"Confidence  : {result.confidence:.2f}   stable {result.streak}/{cfg.stable_frames}", _WHITE),
        (f">> {result.message}", color),
    ]
    if result.reasons:
        shown = ", ".join(result.reasons[:3])
        lines.insert(5, (f"Issues      : {shown}", _ORANGE))

    _panel(frame, lines)
    return frame