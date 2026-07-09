"""Thin wrapper around MediaPipe Pose Landmarker.

It is the only module that knows about MediaPipe. It converts the raw result
into plain numpy arrays so the rest of the codebase stays framework-agnostic
and testable.
"""
from typing import Optional, Tuple

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class PoseDetector:
    """Runs the landmarker in VIDEO mode for a live webcam.

    VIDEO mode (rather than LIVE_STREAM) is deliberate: it returns exactly one
    result per frame synchronously. That keeps the gate's consecutive-good-frame
    counter deterministic. LIVE_STREAM drops frames under load, which would
    silently corrupt the streak logic.
    """

    def __init__(self, model_path: str):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(
        self, frame_rgb: np.ndarray, timestamp_ms: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (image[33,4], world[33,3]) or (None, None) if no pose found."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None, None

        image = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]]
        )
        world = np.array(
            [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]]
        )
        return image, world

    def close(self) -> None:
        self._landmarker.close()
