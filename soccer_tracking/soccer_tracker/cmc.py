"""
cmc.py
======
Camera Motion Compensation (the "BoT" in BoT-SORT).

Broadcast/handheld cameras pan and zoom, which shifts the whole scene. Without
correcting for it, the tracker mistakes camera movement for object movement.
This module estimates the global 2-D motion between two consecutive frames as a
2x3 affine transform, which the tracker then applies to every track's predicted
position.

Method: detect strong corner features in the previous frame, follow them into
the current frame with sparse optical flow, then fit a similarity transform
(rotation + scale + translation) robustly with RANSAC. If anything is
unreliable, fall back to identity ("assume the camera did not move").
"""

import cv2
import numpy as np


class CameraMotionCompensator:
    def __init__(self, max_corners=1000, quality=0.01, min_distance=7):
        self.prev_gray = None
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance

    def estimate(self, frame_bgr):
        """
        Return the 2x3 affine mapping the PREVIOUS frame onto the CURRENT one,
        and remember the current frame for next time.
        """
        cur_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        affine = self._affine(self.prev_gray, cur_gray)
        self.prev_gray = cur_gray
        return affine

    def _affine(self, prev_gray, cur_gray):
        identity = np.eye(2, 3, dtype=np.float64)
        if prev_gray is None:
            return identity
        feats = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=self.max_corners,
            qualityLevel=self.quality, minDistance=self.min_distance, blockSize=3)
        if feats is None or len(feats) < 10:
            return identity
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, feats, None)
        if nxt is None:
            return identity
        status = status.reshape(-1)
        good_old = feats[status == 1]
        good_new = nxt[status == 1]
        if len(good_old) < 10:
            return identity
        affine, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
        if affine is None:
            return identity
        return affine.astype(np.float64)
