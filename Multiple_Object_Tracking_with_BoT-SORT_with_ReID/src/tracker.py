"""Parse Ultralytics Results -> Track objects. PERSONS ONLY."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Track:
    track_id: int
    cls: int
    conf: float
    xyxy: np.ndarray


def parse_results(result, person_cls, ball_cls, person_conf):
    """Extract persons only. Ball OFF."""
    tracks = []
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return tracks

    ids = boxes.id.cpu().numpy().astype(int)
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxys = boxes.xyxy.cpu().numpy()

    for tid, c, cf, box in zip(ids, clss, confs, xyxys):
        # ONLY persons
        if c != person_cls:
            continue
        if cf < person_conf:
            continue
        # Size filter
        w, h = box[2] - box[0], box[3] - box[1]
        if min(w, h) < 4:
            continue
        tracks.append(Track(int(tid), int(c), float(cf), box.astype(float)))
    return tracks