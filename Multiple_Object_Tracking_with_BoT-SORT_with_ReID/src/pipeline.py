"""
The orchestrator. Ties every module together:

  read frame
    -> detector.track (YOLO + BoT-SORT)
    -> parse into Track objects
    -> split persons / ball
    -> stabilize ids, EMA-smooth persons, stabilize the ball
    -> draw
    -> write
"""
from src.video_io import VideoReader, VideoWriter
from src.detector import Detector
from src.tracker import parse_results
from src.smoothing import EMASmoother, BallStabilizer
from src.id_stabilizer import IDStabilizer
from src.visualizer import Visualizer


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.person_smoother = EMASmoother(cfg.smoothing.ema_alpha)
        self.ball = BallStabilizer(max_lost=cfg.smoothing.ball_max_lost)
        self.id_fix = IDStabilizer(max_lost=150, device=self.detector.device)
        self.viz = Visualizer(cfg)

    def run(self):
        cfg = self.cfg
        m = cfg.model
        reader = VideoReader(cfg.video.input_path)
        writer = VideoWriter(cfg.video.output_path, reader.fps, reader.width, reader.height)
        print(f"[INFO] {reader.total} frames @ {reader.fps:.1f} fps "
              f"({reader.width}x{reader.height})")

        frame_idx = 0
        for frame in reader:
            result = self.detector.track(frame)
            tracks = parse_results(
                result, m.person_class_id, m.ball_class_id,
                m.person_conf, m.ball_conf,
            )
            persons = [t for t in tracks if t.cls == m.person_class_id]
            balls = [t for t in tracks if t.cls == m.ball_class_id]

            # ---- persons: stabilize ids, smooth, draw ----
            id_map = self.id_fix.update(frame, persons)
            active_ids = set()
            for t in persons:
                stable_id = id_map.get(t.track_id, t.track_id)
                box = (self.person_smoother.smooth(stable_id, t.xyxy)
                       if cfg.smoothing.enable else t.xyxy)
                self.viz.draw_person(frame, stable_id, box, t.conf)
                active_ids.add(stable_id)
            self.person_smoother.cleanup(active_ids)

            # ---- ball: pick best detection, stabilize, draw ----
            ball_meas = max(balls, key=lambda b: b.conf).xyxy if balls else None
            ball_box = self.ball.update(ball_meas)
            if ball_box is not None:
                self.viz.draw_ball(frame, ball_box)

            self.viz.draw_hud(frame, len(persons), ball_box is not None, frame_idx)
            writer.write(frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[INFO] processed {frame_idx}/{reader.total}")

        writer.release()
        print(f"[DONE] saved -> {cfg.video.output_path}")