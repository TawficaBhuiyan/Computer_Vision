"""
main.py
=======
Command-line entry point.

Examples:
    python main.py                                   # uses defaults
    python main.py --source match.mp4 --output out.mp4
    python main.py --model yolov8x.pt --imgsz 1536   # max accuracy (slower)
    python main.py --model yolov8s.pt                # faster on CPU
"""

import argparse

from soccer_tracker import load_config, VideoTrackingPipeline


def parse_args():
    p = argparse.ArgumentParser(description="BoT-SORT + C-BIoU football tracker")
    p.add_argument("--source", default="video.mp4", help="input video path")
    p.add_argument("--output", default="output_video.mp4", help="output video path")
    p.add_argument("--config", default="config/tracker.yaml", help="tracker config yaml")
    p.add_argument("--model", default="yolov8m.pt",
                   help="YOLO weights: yolov8n/s/m/x.pt (bigger = more stable, slower)")
    p.add_argument("--imgsz", type=int, default=1280, help="inference image size")
    p.add_argument("--conf", type=float, default=0.10, help="detector confidence floor")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = VideoTrackingPipeline(
        cfg, model_name=args.model, imgsz=args.imgsz, conf=args.conf)
    pipeline.run(args.source, args.output)


if __name__ == "__main__":
    main()
