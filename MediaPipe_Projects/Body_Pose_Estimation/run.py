"""Live webcam entry point for the patient positioning gate.

    python run.py

Press ESC to quit. When the gate reports READY, plug your existing
pain-evaluation pipeline into the marked hook below.

Only ONE model runs (Pose Landmarker). Head pose is recovered from the pose
model's own face keypoints via solvePnP -- no separate face model.
"""
import time

import cv2

from pose_gate.config import GateConfig
from pose_gate.detector import PoseDetector
from pose_gate.head import estimate_head_pose
from pose_gate.gate import GateResult, PoseQualityGate

POSE_MODEL_PATH = "models/pose_landmarker_heavy.task"
CAMERA_INDEX = 0

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _status_color(result: GateResult):
    if result.ready:
        return (0, 200, 0)       # green
    if result.frame_ok:
        return (0, 165, 255)     # orange (good frame, not yet stable)
    return (0, 0, 255)           # red


def _draw(frame, result: GateResult, cfg: GateConfig):
    color = _status_color(result)
    cv2.putText(frame, result.message, (20, 40), _FONT, 0.9, color, 2)

    m = result.metrics
    cv2.putText(
        frame,
        f"yaw:{m['yaw']} roll:{m['roll']} tilt:{m['tilt']}  "
        f"{result.streak}/{cfg.stable_frames}",
        (20, 75), _FONT, 0.55, color, 1,
    )
    cv2.putText(
        frame,
        f"head_yaw:{m['head_yaw']} head_pitch:{m['head_pitch']}",
        (20, 100), _FONT, 0.55, color, 1,
    )
    if result.ready:
        cv2.putText(frame, "READY -> capturing", (20, 130), _FONT, 0.7, (0, 200, 0), 2)


def main() -> None:
    cfg = GateConfig()
    detector = PoseDetector(POSE_MODEL_PATH)
    gate = PoseQualityGate(cfg)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    start = time.time()

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.time() - start) * 1000)
            image, world = detector.detect(rgb, timestamp_ms)

            if image is None:
                cv2.putText(frame, "No person detected", (20, 40),
                            _FONT, 0.9, (0, 0, 255), 2)
            else:
                head_pose = estimate_head_pose(image, width, height,
                                               cfg.head_min_visibility)
                result = gate.evaluate(image, world, head_pose=head_pose)
                _draw(frame, result, cfg)

                if result.ready:
                    # >>> Trigger your existing pain-evaluation pipeline here <<<
                    # e.g. pain_pipeline.run(frame, image, world)
                    pass

            cv2.imshow("Patient Positioning Gate", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
