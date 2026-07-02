"""
Anti-fluctuation module.

EMASmoother   : exponential moving average on person boxes -> no jitter.
BallStabilizer: keeps ONE steady ball box, smooths it, and bridges short
                detection gaps with constant-velocity prediction so the
                ball box does not flicker on/off.
"""
import numpy as np


class EMASmoother:
    """Per-ID exponential moving average of box corners."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._state = {}          # track_id -> xyxy

    def smooth(self, track_id: int, xyxy: np.ndarray) -> np.ndarray:
        prev = self._state.get(track_id)
        if prev is None:
            self._state[track_id] = xyxy.copy()
            return xyxy
        sm = self.alpha * xyxy + (1.0 - self.alpha) * prev
        self._state[track_id] = sm
        return sm

    def cleanup(self, active_ids):
        for k in list(self._state.keys()):
            if k not in active_ids:
                del self._state[k]


class BallStabilizer:
    """
    Single-object constant-velocity smoother for the ball.
    - Smooths center + size so the box stops shaking.
    - If the ball is not detected for <= max_lost frames, it keeps
      moving the box along its last velocity (bridges the gap).
    - Returns None only after the ball is truly gone.
    """

    def __init__(self, max_lost=25, pos_alpha=0.5, vel_alpha=0.3):
        self.max_lost = max_lost
        self.pos_alpha = pos_alpha
        self.vel_alpha = vel_alpha
        self.reset()

    def reset(self):
        self.cx = self.cy = self.w = self.h = None
        self.vx = self.vy = 0.0
        self.lost = 0
        self.initialized = False

    def update(self, xyxy):
        """xyxy = measurement array, or None if ball not detected."""
        if xyxy is not None:
            mcx = (xyxy[0] + xyxy[2]) / 2.0
            mcy = (xyxy[1] + xyxy[3]) / 2.0
            mw = xyxy[2] - xyxy[0]
            mh = xyxy[3] - xyxy[1]

            if not self.initialized:
                self.cx, self.cy, self.w, self.h = mcx, mcy, mw, mh
                self.vx = self.vy = 0.0
                self.initialized = True
            else:
                self.vx = self.vel_alpha * (mcx - self.cx) + (1 - self.vel_alpha) * self.vx
                self.vy = self.vel_alpha * (mcy - self.cy) + (1 - self.vel_alpha) * self.vy
                self.cx = self.pos_alpha * mcx + (1 - self.pos_alpha) * self.cx
                self.cy = self.pos_alpha * mcy + (1 - self.pos_alpha) * self.cy
                self.w = self.pos_alpha * mw + (1 - self.pos_alpha) * self.w
                self.h = self.pos_alpha * mh + (1 - self.pos_alpha) * self.h
            self.lost = 0
            return self._box()

        # ---- no detection this frame ----
        if not self.initialized:
            return None
        self.lost += 1
        if self.lost > self.max_lost:
            return None
        self.cx += self.vx        # predict along velocity
        self.cy += self.vy
        return self._box()

    def _box(self):
        return np.array([
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        ], dtype=float)
