"""
association.py
==============
Data association: deciding which detection belongs to which existing track.
This is where C-BIoU (buffered IoU) and the fast-motion enhancements live.

Two routines, both returning (matches, unmatched_tracks, unmatched_detections)
where each match is a (detection_index, track_index) pair:

1. associate()  - the main matcher. Uses a velocity-aware buffered IoU FUSED
                  with center distance, solved optimally by the Hungarian
                  algorithm, then gated. A pair is accepted if EITHER the
                  buffered IoU clears its threshold OR the predicted centers are
                  close enough (the OR is what saves fast objects whose boxes no
                  longer overlap).

2. recover()    - a last-resort long-range matcher using center distance only,
                  for big jumps and reappearance after occlusion.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import buffer_boxes, expand_abs, iou_batch, center_dist_batch


class Associator:
    def __init__(self, cfg):
        self.cfg = cfg

    def associate(self, tracks, track_indices, det_indices, boxes,
                  buffer_ratio, iou_thr):
        track_indices = list(track_indices)
        det_indices = list(det_indices)
        if not track_indices or not det_indices:
            return [], track_indices, det_indices

        det_boxes = boxes[det_indices]
        trk_boxes = np.array([tracks[t].box for t in track_indices])
        speeds = np.array([tracks[t].speed for t in track_indices])
        diags = np.array([max(tracks[t].diag, 1.0) for t in track_indices])

        # Velocity-aware buffer: faster tracks get a larger search box (capped).
        gain = self.cfg["motion_buffer_gain"]
        motion_margin = np.minimum(gain * speeds, 4.0 * diags)
        det_buf = buffer_boxes(det_boxes, buffer_ratio)
        trk_buf = expand_abs(buffer_boxes(trk_boxes, buffer_ratio), motion_margin)

        iou = iou_batch(trk_buf, det_buf)                 # (T, D)
        cdist = center_dist_batch(trk_boxes, det_boxes)   # (T, D)

        # Fused cost: low overlap is forgiven if the centers are close.
        w_c = self.cfg["center_dist_weight"]
        cost = (1.0 - iou) + w_c * (cdist / diags[:, None])
        row, col = linear_sum_assignment(cost)

        # Per-track acceptance gate, scaled by box size and speed.
        gate = (self.cfg["center_gate_diag_mult"] * diags +
                self.cfg["center_gate_speed_mult"] * speeds)

        matches, matched_t, matched_d = [], set(), set()
        for r, c in zip(row, col):
            if iou[r, c] >= iou_thr or cdist[r, c] <= gate[r]:
                matches.append((det_indices[c], track_indices[r]))
                matched_t.add(r)
                matched_d.add(c)

        un_tracks = [track_indices[r] for r in range(len(track_indices)) if r not in matched_t]
        un_dets = [det_indices[c] for c in range(len(det_indices)) if c not in matched_d]
        return matches, un_tracks, un_dets

    def recover(self, tracks, track_indices, det_indices, boxes):
        track_indices = list(track_indices)
        det_indices = list(det_indices)
        if not track_indices or not det_indices:
            return [], track_indices, det_indices

        det_boxes = boxes[det_indices]
        trk_boxes = np.array([tracks[t].box for t in track_indices])
        speeds = np.array([tracks[t].speed for t in track_indices])
        diags = np.array([max(tracks[t].diag, 1.0) for t in track_indices])

        cdist = center_dist_batch(trk_boxes, det_boxes)
        row, col = linear_sum_assignment(cdist)

        radius = (self.cfg["recovery_diag_mult"] * diags +
                  self.cfg["recovery_speed_mult"] * speeds)

        matches, matched_t, matched_d = [], set(), set()
        for r, c in zip(row, col):
            if cdist[r, c] <= radius[r]:
                matches.append((det_indices[c], track_indices[r]))
                matched_t.add(r)
                matched_d.add(c)

        un_tracks = [track_indices[r] for r in range(len(track_indices)) if r not in matched_t]
        un_dets = [det_indices[c] for c in range(len(det_indices)) if c not in matched_d]
        return matches, un_tracks, un_dets
