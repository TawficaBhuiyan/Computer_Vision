"""Thin, safe wrappers around OpenCV video read/write."""
import cv2


class VideoReader:
    """Iterate over frames of a video file."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            self.release()
            raise StopIteration
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """Write annotated frames to an mp4 file."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open writer for: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
