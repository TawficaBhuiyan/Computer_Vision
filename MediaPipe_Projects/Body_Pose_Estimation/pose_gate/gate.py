"""Combines the geometric checks with framing checks and temporal smoothing
into a single stateful gate that emits one clear instruction per frame.

The head check uses real yaw/pitch degrees produced by head.estimate_head_pose
from the pose model's own face keypoints (no separate face model).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import geometry as geo
from .config import GateConfig, FULL_BODY_LANDMARKS, NOSE, LEFT_ANKLE, RIGHT_ANKLE

# Feedback priority. Only the first matching reason is shown so the patient
# gets one instruction at a time instead of a wall of errors.
_PRIORITY = (
    ("full_body",      "Step back - get your full body in frame"),
    ("too_close",      "Move back a little"),
    ("turn_to_face",   "Turn to face the camera"),
    ("stand_straight", "Stand up straight"),
    ("look_at_camera", "Look straight at the camera"),
)


@dataclass
class GateResult:
    ready: bool          # stable for enough frames -> safe to trigger pipeline
    frame_ok: bool       # this single frame passed every check
    streak: int          # current run of consecutive good frames
    message: str         # single instruction to show the patient
    metrics: dict        # smoothed values, for logging / debugging
    reasons: list        # all failing checks this frame


class _EMA:
    """Exponential moving average to damp per-frame jitter."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


def _check_full_body(image: np.ndarray, cfg: GateConfig):
    """Return (ok, reason). Verifies every critical landmark is confident and
    inside the frame, and that the patient is not standing too close."""
    m = cfg.frame_margin
    for idx in FULL_BODY_LANDMARKS:
        x, y, _, visibility = image[idx]
        if visibility < cfg.visibility or not (m <= x <= 1 - m and m <= y <= 1 - m):
            return False, "full_body"

    body_fill = max(image[LEFT_ANKLE, 1], image[RIGHT_ANKLE, 1]) - image[NOSE, 1]
    if body_fill > cfg.max_body_fill:
        return False, "too_close"
    return True, None


def _message(reasons: list) -> str:
    for key, text in _PRIORITY:
        if key in reasons:
            return text
    return "READY - hold still"


class PoseQualityGate:
    """Stateful gate. Feed it one frame's data; it returns a GateResult."""

    def __init__(self, cfg: GateConfig = GateConfig()):
        self.cfg = cfg
        self._streak = 0
        self._yaw = _EMA(cfg.ema_alpha)
        self._roll = _EMA(cfg.ema_alpha)
        self._tilt = _EMA(cfg.ema_alpha)
        self._head_yaw = _EMA(cfg.ema_alpha)
        self._head_pitch = _EMA(cfg.ema_alpha)

    def evaluate(
        self,
        image: np.ndarray,
        world: np.ndarray,
        head_pose: Optional[Tuple[float, float, float]] = None,
    ) -> GateResult:
        """head_pose is (yaw, pitch, roll) degrees, or None when the face
        keypoints were not reliable this frame."""
        cfg = self.cfg
        reasons: list = []

        body_ok, body_reason = _check_full_body(image, cfg)
        if body_reason:
            reasons.append(body_reason)

        yaw = self._yaw.update(geo.facing_yaw(world))
        roll = self._roll.update(geo.shoulder_roll(world))
        tilt = self._tilt.update(geo.torso_tilt(world))

        facing_ok = yaw < cfg.max_yaw
        upright_ok = roll < cfg.max_roll and tilt < cfg.max_torso_tilt
        head_ok, head_metrics = self._check_head(head_pose)

        if not facing_ok:
            reasons.append("turn_to_face")
        if not upright_ok:
            reasons.append("stand_straight")
        if not head_ok:
            reasons.append("look_at_camera")

        frame_ok = body_ok and facing_ok and upright_ok and head_ok
        self._streak = self._streak + 1 if frame_ok else 0
        ready = self._streak >= cfg.stable_frames

        metrics = {"yaw": round(yaw, 1), "roll": round(roll, 1),
                   "tilt": round(tilt, 1), **head_metrics}
        return GateResult(
            ready=ready, frame_ok=frame_ok, streak=self._streak,
            message=_message(reasons), metrics=metrics, reasons=reasons,
        )

    def _check_head(self, head_pose):
        cfg = self.cfg
        if head_pose is None:  # face keypoints unreliable -> not looking at camera
            return False, {"head_yaw": None, "head_pitch": None}
        h_yaw = self._head_yaw.update(head_pose[0])
        h_pitch = self._head_pitch.update(head_pose[1])
        ok = abs(h_yaw) < cfg.max_head_yaw and abs(h_pitch) < cfg.max_head_pitch
        return ok, {"head_yaw": round(h_yaw, 1), "head_pitch": round(h_pitch, 1)}
