"""Thin wrapper around MediaPipe Pose Landmarker.

The only module that knows about MediaPipe. It converts raw results into plain
numpy arrays so the rest of the codebase stays framework-agnostic and testable.

* num_poses = 2 so Module 1 can DETECT (and then reject) a second person rather
  than silently locking onto one.
* Supports IMAGE mode (single stills) and VIDEO mode (files / webcam). VIDEO
  mode returns exactly one result per frame synchronously, which keeps the
  temporal streak counter deterministic; LIVE_STREAM would drop frames under
  load and corrupt that logic.
"""
from typing import List, Tuple

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# (image[33,4], world[33,3]) for one detected person.
Person = Tuple[np.ndarray, np.ndarray]


class PoseDetector:
    def __init__(self, model_path: str, mode: str = "video",
                 num_poses: int = 2):
        running_mode = (vision.RunningMode.IMAGE if mode == "image"
                        else vision.RunningMode.VIDEO)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mode = mode
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> List[Person]:
        """Return a list of (image, world) arrays, one per detected person."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self._mode == "image":
            result = self._landmarker.detect(mp_image)
        else:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks or not result.pose_world_landmarks:
            return []

        persons: List[Person] = []
        for lms, wlms in zip(result.pose_landmarks, result.pose_world_landmarks):
            image = np.array([[p.x, p.y, p.z, p.visibility] for p in lms])
            world = np.array([[p.x, p.y, p.z] for p in wlms])
            persons.append((image, world))
        return persons

    def close(self) -> None:
        self._landmarker.close()
