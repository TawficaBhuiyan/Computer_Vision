"""
Detector = YOLO model + Ultralytics' integrated tracker.

We run detection AND tracking in one call (model.track). Doing them
together lets BoT-SORT's Kalman filter + ReID assign IDs frame-to-frame,
which is what gives us steady, non-shuffling IDs.
"""
import torch
from ultralytics import YOLO


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.model.device)
        self.model = YOLO(cfg.model.weights)

    @staticmethod
    def _resolve_device(device):
        if "cuda" in str(device) and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available -> using CPU.")
            return "cpu"
        return device

    def track(self, frame):
        """Return the Ultralytics Results object for a single frame."""
        m = self.cfg.model
        results = self.model.track(
            source=frame,
            persist=True,                       # keep IDs across frames
            tracker=self.cfg.tracker.config,
            classes=[m.person_class_id, m.ball_class_id],
            conf=min(m.person_conf, m.ball_conf),  # low gate; filter later
            iou=m.iou,
            imgsz=m.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0]
