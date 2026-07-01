"""
tracker.py
==========
The orchestrator. `BoTSORTCBIoUTracker.update()` runs the per-frame tracking
loop, wiring together the Kalman tracks, camera-motion compensation, and the
association stages.

Per-frame flow (7 steps):
    0. Estimate camera motion (CMC).
    1. Predict every track, then warp it by the camera motion.
    2. Split detections into high / low confidence (ByteTrack idea).
    3. Stage 1: high-conf detections vs all tracks, small buffer b1.
    4. Stage 2: low-conf detections vs leftover tracks, large buffer b2.
    5. Stage 3: long-range recovery for leftover high-conf detections.
    6. Spawn new tracks from still-unmatched high-conf detections.
    7. Lifecycle: confirm mature tracks, delete dead ones.

`update()` returns an array of track IDs aligned to the input detections;
detections that matched nothing get -1.
"""

import numpy as np

from .track import Track
from .cmc import CameraMotionCompensator
from .association import Associator


class BoTSORTCBIoUTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.next_id = 1
        self.cmc = CameraMotionCompensator()
        self.assoc = Associator(cfg)

    def update(self, boxes_xyxy, confidences, frame_bgr):
        """
        boxes_xyxy   : (N, 4) detection boxes
        confidences  : (N,)   detection confidences
        frame_bgr    : the raw frame (needed for camera-motion estimation)
        returns      : (N,)   track id per detection (-1 if unassigned)
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        confs = np.asarray(confidences, dtype=np.float64).reshape(-1)
        n = len(boxes)
        result_ids = np.full(n, -1, dtype=int)

        # 0) camera motion
        affine = self.cmc.estimate(frame_bgr)

        # 1) predict + compensate
        for t in self.tracks:
            t.predict()
            t.kf.apply_cmc(affine)

        # 2) confidence split
        high_thr = self.cfg["high_conf_det_threshold"]
        low_thr = self.cfg["track_activation_threshold"]
        high_idx = np.where(confs >= high_thr)[0]
        low_idx = np.where((confs < high_thr) & (confs >= low_thr))[0]
        track_indices = list(range(len(self.tracks)))

        # 3) Stage 1
        m1, un_tracks1, un_dets1 = self.assoc.associate(
            self.tracks, track_indices, high_idx, boxes,
            self.cfg["buffer_ratio_first"],
            self.cfg["minimum_iou_threshold_first_assoc"])
        self._apply(m1, boxes, confs, result_ids)

        # 4) Stage 2
        m2, un_tracks2, _ = self.assoc.associate(
            self.tracks, un_tracks1, low_idx, boxes,
            self.cfg["buffer_ratio_second"],
            self.cfg["minimum_iou_threshold_second_assoc"])
        self._apply(m2, boxes, confs, result_ids)

        # 5) Stage 3 (long-range recovery, high-conf leftovers)
        m3, _, un_dets3 = self.assoc.recover(
            self.tracks, un_tracks2, un_dets1, boxes)
        self._apply(m3, boxes, confs, result_ids)

        # 6) spawn new tracks
        for d_i in un_dets3:
            t = Track(boxes[d_i], self.next_id, confs[d_i])
            self.tracks.append(t)
            result_ids[d_i] = self.next_id
            self.next_id += 1

        # 7) lifecycle
        for t in self.tracks:
            if t.hits >= self.cfg["minimum_consecutive_frames"]:
                t.state = "confirmed"
        kept = []
        for t in self.tracks:
            buf = (self.cfg["lost_track_buffer"] if t.state == "confirmed"
                   else self.cfg["minimum_consecutive_frames"])
            if t.time_since_update <= buf:
                kept.append(t)
        self.tracks = kept

        return result_ids

    def _apply(self, matches, boxes, confs, result_ids):
        """Apply matched (detection, track) pairs to the track set."""
        for d_i, t_i in matches:
            self.tracks[t_i].update(boxes[d_i], confs[d_i])
            result_ids[d_i] = self.tracks[t_i].id
