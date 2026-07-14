"""Detector = YOLO + Ultralytics tracker. PERSONS ONLY. No ball."""
import torch
from ultralytics import YOLO


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.model.device)
        self.half = bool(getattr(cfg.model, "half", False)) and "cuda" in str(self.device)
        self.model = YOLO(cfg.model.weights)

    @staticmethod
    def _resolve_device(device):
        if "cuda" in str(device) and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available -> using CPU.")
            return "cpu"
        return device

    def track(self, frame):
        """Track only PERSONS. Ball OFF."""
        m = self.cfg.model
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.cfg.tracker.config,
            classes=[m.person_class_id],      # ONLY persons, no ball
            conf=m.person_conf,
            iou=m.iou,
            imgsz=m.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        return results[0]