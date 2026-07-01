"""
detector.py
===========
Thin wrapper around an Ultralytics YOLO model. It runs detection on a frame and
returns only the classes we care about, as a `supervision.Detections` object.

Why a wrapper: it isolates the detector behind a single `.detect(frame)` call,
so the rest of the pipeline never touches Ultralytics/YOLO directly. Swapping in
a different detector later means changing only this file.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO


class YoloDetector:
    # class 0 = person, class 32 = sports ball (COCO ids)
    def __init__(self, model_name="yolov8m.pt", imgsz=1280, conf=0.10,
                 keep_classes=(0, 32)):
        # Detector quality is the single biggest factor in fast-motion ID
        # stability. yolov8m/x miss far fewer blurry distant players than n/s.
        self.model = YOLO(model_name)
        self.imgsz = imgsz
        self.conf = conf
        self.keep_classes = keep_classes

    def detect(self, frame_bgr):
        results = self.model(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                             verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        if detections.class_id is None or len(detections) == 0:
            return detections
        mask = np.isin(detections.class_id, np.array(self.keep_classes))
        return detections[mask]
