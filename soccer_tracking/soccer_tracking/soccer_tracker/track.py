"""
track.py
========
A `Track` is one object's life story: a Kalman filter plus bookkeeping that the
tracker uses to decide when an identity is born, confirmed, and retired.

Lifecycle:
    tentative  -> just created, not trusted yet
    confirmed  -> seen `minimum_consecutive_frames` times; allowed to survive
                  long occlusions
"""

import numpy as np

from .kalman import KalmanBox


class Track:
    def __init__(self, box_xyxy, track_id, conf):
        self.kf = KalmanBox(box_xyxy)
        self.id = track_id
        self.conf = conf
        self.time_since_update = 0   # frames since last successful match
        self.hits = 1                # total successful matches
        self.state = "tentative"

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, box_xyxy, conf):
        self.kf.update(box_xyxy)
        self.conf = conf
        self.time_since_update = 0
        self.hits += 1

    @property
    def box(self):
        """Current predicted box in xyxy."""
        return self.kf.state_xyxy()

    @property
    def speed(self):
        return self.kf.speed

    @property
    def diag(self):
        """Diagonal length of the predicted box (a size proxy)."""
        b = self.box
        return float(np.hypot(b[2] - b[0], b[3] - b[1]))
