"""
pipeline.py
===========
Top-level orchestration for a whole video: read frames, detect, track, draw,
write. This is the only place that knows about video files and frame loops;
detection, tracking, and drawing are delegated to their own modules.

Flow per frame:
    frame ->  YoloDetector.detect()        (boxes + confidences + classes)
          ->  BoTSORTCBIoUTracker.update()  (assign persistent IDs)
          ->  Annotator.draw()              (boxes + labels)
          ->  VideoWriter                   (save frame)
"""

import cv2

from .detector import YoloDetector
from .tracker import BoTSORTCBIoUTracker
from .annotator import Annotator


class VideoTrackingPipeline:
    def __init__(self, cfg, model_name="yolov8m.pt", imgsz=1280, conf=0.10):
        self.detector = YoloDetector(model_name=model_name, imgsz=imgsz, conf=conf)
        self.tracker = BoTSORTCBIoUTracker(cfg)
        self.annotator = Annotator()

    def run(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        print(f"Processing '{input_path}'  ({total} frames, {width}x{height} @ {fps}fps)")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = self.detector.detect(frame)
            ids = self.tracker.update(
                detections.xyxy, detections.confidence, frame)
            detections.tracker_id = ids
            detections = detections[detections.tracker_id != -1]

            frame = self.annotator.draw(frame, detections)
            writer.write(frame)

            idx += 1
            if idx % 50 == 0:
                print(f"  {idx}/{total} frames")

        cap.release()
        writer.release()
        print(f"Done. Saved -> {output_path}")
