"""
test_tracker.py
===============
Fast, dependency-light sanity tests that exercise the tracker logic WITHOUT a
YOLO model or a real video. They feed synthetic detections (fake boxes) and
assert that identities stay stable through the three classic failure modes:

    1. fast motion (box displacement >> box size)
    2. short occlusion (a detection disappears, then returns)
    3. two fast objects crossing each other

Run:
    python -m tests.test_tracker        (from the project root)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soccer_tracker import load_config, BoTSORTCBIoUTracker

CFG = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "tracker.yaml"))
FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def _ids(tracker, boxes, confs):
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    confs = np.array(confs, dtype=float)
    return list(tracker.update(boxes, confs, FRAME))


def test_fast_motion_and_occlusion():
    tr = BoTSORTCBIoUTracker(CFG)
    fast = np.array([100, 300, 140, 360], float)   # 40px-wide box
    slow = np.array([900, 300, 940, 360], float)
    fast_id = None
    for f in range(14):
        fast += np.array([55, 0, 55, 0])           # 55px jump => buffered IoU ~ 0
        slow += np.array([4, 0, 4, 0])
        boxes, confs = [fast.copy(), slow.copy()], [0.85, 0.85]
        if f in (6, 7):                            # occlude the fast object
            boxes, confs = boxes[1:], confs[1:]
            _ids(tr, boxes, confs)
            continue
        ids = _ids(tr, boxes, confs)
        if fast_id is None:
            fast_id = ids[0]
        assert ids[0] == fast_id, f"fast object ID switched at frame {f}"
    print("PASS: fast motion + occlusion -> no ID switch")


def test_crossing():
    tr = BoTSORTCBIoUTracker(CFG)
    a = np.array([300, 300, 340, 360], float)      # moving right
    b = np.array([900, 300, 940, 360], float)      # moving left (will cross)
    a_id = b_id = None
    for _ in range(20):
        a += np.array([34, 0, 34, 0])
        b += np.array([-34, 0, -34, 0])
        ids = _ids(tr, [a.copy(), b.copy()], [0.85, 0.85])
        if a_id is None:
            a_id, b_id = ids[0], ids[1]
        assert ids[0] == a_id and ids[1] == b_id, "IDs swapped during crossing"
    print("PASS: crossing -> no ID swap")


if __name__ == "__main__":
    test_fast_motion_and_occlusion()
    test_crossing()
    print("\nAll tests passed.")
