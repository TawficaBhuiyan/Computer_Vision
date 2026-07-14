"""Temporal smoothing primitives (Module 8).

Three complementary tools, each matched to the signal it smooths:

* OneEuroFilter  -> continuous angle signals (yaw, tilt, joint angles).
    Adaptive low-pass: low cutoff when the value is still (kills jitter), high
    cutoff when the value moves fast (kills lag). Strictly better than a fixed
    EMA for interactive positioning, at negligible cost.
* SlidingWindowVote -> the discrete orientation LABEL.
    A majority vote over the last N frames debounces classification flips at
    boundaries (e.g. OBLIQUE <-> PROFILE) without adding lag to angles.
* StreakLatch    -> the final boolean READY decision.
    Requires K consecutive valid frames, so a single spurious good/bad frame
    cannot toggle the capture trigger.

A Kalman filter was deliberately NOT used: it needs a motion model and process/
measurement noise tuning to beat One Euro here, and the signals are quasi-static
(a person holding a pose), so the extra machinery buys no accuracy.
"""
import math
from collections import Counter, deque
from typing import Deque, Optional


class OneEuroFilter:
    """Scalar 1-Euro filter (Casiez, Roussel & Vogel, CHI 2012).

    cutoff(t) = min_cutoff + beta * |x_dot(t)|
    Small min_cutoff -> smooth at rest; larger beta -> more responsive when the
    signal changes quickly.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 dcutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0.0:
            dt = 1e-3
        if self._x_prev is None:
            self._x_prev = x
            return x
        # derivative, low-pass filtered at dcutoff
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        # adaptive cutoff for the signal itself
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class EMA:
    """Fixed-alpha exponential moving average. Kept as a lightweight fallback."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.value: Optional[float] = None

    def reset(self) -> None:
        self.value = None

    def __call__(self, x: float) -> float:
        self.value = x if self.value is None \
            else self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class SlidingWindowVote:
    """Majority vote over a fixed-length window of discrete labels."""

    def __init__(self, window: int = 7):
        self.window = max(1, window)
        self._buf: Deque[str] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._buf.clear()

    def __call__(self, label: str) -> str:
        self._buf.append(label)
        return Counter(self._buf).most_common(1)[0][0]


class StreakLatch:
    """Counts consecutive True inputs; latches READY at >= stable_frames."""

    def __init__(self, stable_frames: int = 12):
        self.stable_frames = max(1, stable_frames)
        self.streak = 0

    def reset(self) -> None:
        self.streak = 0

    def __call__(self, ok: bool) -> bool:
        self.streak = self.streak + 1 if ok else 0
        return self.streak >= self.stable_frames
