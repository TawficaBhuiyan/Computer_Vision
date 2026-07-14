"""Real-Time Side Profile Full Body Pose Validation System - entry point.

Examples
--------
    # live webcam (ESC to quit)
    python run.py --source webcam

    # a single image -> annotated image
    python run.py --source image --input samples/side_profile_left.png \
                  --output out/annotated.png

    # a video file -> annotated output_video.mp4
    python run.py --source video --input input_video.MOV \
                  --output output_video.mp4

When the gate reports READY, plug your downstream pipeline into the marked hook.
"""
import argparse
import os
import time

import cv2

from profile_gate.config import ProfileGateConfig
from profile_gate.detector import PoseDetector
from profile_gate.gate import ProfileGate
from profile_gate.render import draw

DEFAULT_MODEL = "models/pose_landmarker_heavy.task"


def _process_frame(frame_bgr, detector, gate, cfg, timestamp_ms):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    persons = detector.detect(rgb, timestamp_ms)
    result = gate.evaluate(persons, w, h)
    draw(frame_bgr, result, cfg)
    return frame_bgr, result


def run_image(args, cfg):
    detector = PoseDetector(args.model, mode="image")
    gate = ProfileGate(cfg)
    frame = cv2.imread(args.input)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.input}")
    # evaluate repeatedly so the streak latch can reach READY on a still
    for _ in range(cfg.stable_frames + 2):
        out, result = _process_frame(frame.copy(), detector, gate, cfg, 0)
    output = args.output or "out/annotated.png"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    cv2.imwrite(output, out)
    detector.close()
    print(f"[image] {result.label}  valid={result.valid} ready={result.ready} "
          f"conf={result.confidence}  ->  {output}")


def run_video(args, cfg):
    detector = PoseDetector(args.model, mode="video")
    gate = ProfileGate(cfg)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_assumed
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = args.output or "output_video.mp4"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx, ready_frames = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(idx * 1000.0 / fps)
        out, result = _process_frame(frame, detector, gate, cfg, ts)
        if result.ready:
            ready_frames += 1
        writer.write(out)
        idx += 1

    cap.release()
    writer.release()
    detector.close()
    print(f"[video] {idx} frames, {ready_frames} READY  ->  {output}")


def run_webcam(args, cfg):
    detector = PoseDetector(args.model, mode="video")
    gate = ProfileGate(cfg)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_assumed
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    start = time.time()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            ts = int((time.time() - start) * 1000)
            out, result = _process_frame(frame, detector, gate, cfg, ts)
            if result.ready:
                # >>> Trigger your downstream pipeline here <<<
                pass
            if writer is not None:
                writer.write(out)
            cv2.imshow("Side Profile Pose Validation", out)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        detector.close()


def build_config(args) -> ProfileGateConfig:
    overrides = {}
    if args.accept_facing:
        overrides["accept_facing"] = args.accept_facing
    return ProfileGateConfig(**overrides) if overrides else ProfileGateConfig()


def main():
    p = argparse.ArgumentParser(description="Side Profile Full Body Pose Validation")
    p.add_argument("--source", choices=["webcam", "image", "video"],
                   default="webcam")
    p.add_argument("--input", help="path to image/video (for image/video sources)")
    p.add_argument("--output", help="output path (annotated image or mp4)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--accept-facing", choices=["both", "left", "right"],
                   help="which profile direction to accept (default: both)")
    args = p.parse_args()

    cfg = build_config(args)
    if args.source == "image":
        if not args.input:
            raise SystemExit("--input is required for --source image")
        run_image(args, cfg)
    elif args.source == "video":
        if not args.input:
            raise SystemExit("--input is required for --source video")
        run_video(args, cfg)
    else:
        run_webcam(args, cfg)


if __name__ == "__main__":
    main()
