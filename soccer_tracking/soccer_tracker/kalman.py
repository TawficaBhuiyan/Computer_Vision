"""
kalman.py
=========
A constant-velocity Kalman filter for a single object's bounding box. This is
the "predict then correct" motion model from BoT-SORT.

State vector (8 numbers):
    [cx, cy, w, h, vcx, vcy, vw, vh]
    - first 4  : where the box is and how big it is
    - last 4   : how fast each of those is changing (velocity)

Each frame the filter PREDICTS the next box from velocity, and when a detection
arrives it CORRECTS that prediction. The prediction is what lets a track keep
moving while it is briefly occluded, so it can be re-acquired with the same ID.
"""

import numpy as np

from .geometry import xyxy_to_cxcywh, cxcywh_to_xyxy


class KalmanBox:
    def __init__(self, box_xyxy):
        z = xyxy_to_cxcywh(box_xyxy[None, :])[0]
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = z
        self.last_z = z.copy()
        self.updated_once = False

        # F: state transition. Adds velocity to position each step (dt = 1).
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # H: measurement matrix. We can only measure [cx, cy, w, h].
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # P: state covariance (uncertainty). Velocity starts highly uncertain.
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 1000.0

        # Q: process noise. Larger velocity term -> filter adapts faster to
        #    speed changes (important for fast, irregular motion).
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.5

        # R: measurement noise (how noisy detections are).
        self.R = np.eye(4) * 1.0

    def predict(self):
        """Advance the state one frame using the motion model."""
        self.x = self.F @ self.x
        self.x[2] = max(self.x[2], 1.0)   # never let width/height collapse to 0
        self.x[3] = max(self.x[3], 1.0)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, box_xyxy):
        """Correct the prediction with a new measured box."""
        z = xyxy_to_cxcywh(box_xyxy[None, :])[0]
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        # Velocity seeding: on the very first correction, set velocity directly
        # from the first observed displacement so prediction is correct from
        # frame 2 instead of taking several frames to "learn" the speed.
        if not self.updated_once:
            self.x[4:8] = z - self.last_z
            self.updated_once = True
        self.last_z = z.copy()

    def apply_cmc(self, affine):
        """
        Warp position and velocity by a camera-motion affine (2x3), so the track
        lives in the current frame's coordinate system before association.
        """
        cx, cy = self.x[0], self.x[1]
        p = affine @ np.array([cx, cy, 1.0])
        self.x[0], self.x[1] = p[0], p[1]
        scale = np.sqrt(max(affine[0, 0] ** 2 + affine[0, 1] ** 2, 1e-9))
        self.x[2] *= scale
        self.x[3] *= scale
        vx, vy = self.x[4], self.x[5]
        self.x[4] = affine[0, 0] * vx + affine[0, 1] * vy
        self.x[5] = affine[1, 0] * vx + affine[1, 1] * vy

    def state_xyxy(self):
        return cxcywh_to_xyxy(self.x[:4][None, :])[0]

    @property
    def speed(self):
        """Scalar speed (pixels/frame) of the box center."""
        return float(np.hypot(self.x[4], self.x[5]))
