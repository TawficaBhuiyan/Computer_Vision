"""
Entry point. Run from the project root:

    python run.py
    python run.py --input input/input_video.mp4 --device cpu
"""
import argparse
from pathlib import Path

from src.config_loader import load_config
from src.pipeline import Pipeline


def main():
    p = argparse.ArgumentParser(description="Office scuffle: person + ball tracker")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--input", default=None, help="override input video path")
    p.add_argument("--output", default=None, help="override output video path")
    p.add_argument("--device", default=None, help='"cpu" or "cuda:0"')
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.input:
        cfg.video.input_path = args.input
    if args.output:
        cfg.video.output_path = args.output
    if args.device:
        cfg.model.device = args.device

    Path(cfg.video.output_path).parent.mkdir(parents=True, exist_ok=True)
    Pipeline(cfg).run()


if __name__ == "__main__":
    main()
