"""
Turn the raw Ultralytics Results into a clean list of Track objects,
applying per-class confidence gates (person vs ball differ).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Track:
    track_id: int
    cls: int
    conf: float
    xyxy: np.ndarray          # [x1, y1, x2, y2] float


def parse_results(result, person_cls, ball_cls, person_conf, ball_conf):
    tracks = []
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return tracks

    ids = boxes.id.cpu().numpy().astype(int)
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxys = boxes.xyxy.cpu().numpy()

    for tid, c, cf, box in zip(ids, clss, confs, xyxys):
        if c == person_cls and cf < person_conf:
            continue
        if c == ball_cls and cf < ball_conf:
            continue
        tracks.append(Track(int(tid), int(c), float(cf), box.astype(float)))
    return tracks
