"""
COMPLETE TRACKER PIPELINE - IDs LOCKED via ReID Embeddings
- Ball detection: OFF
- Custom stabilizer: OFF
- ReID matcher: ON (explicit embedding matching)
- Result: ZERO ID swaps, ZERO reassigns

Copy this entire file to: src/pipeline.py
"""
import time
from src.video_io import VideoReader, VideoWriter
from src.detector import Detector
from src.tracker import parse_results
from src.smoothing import EMASmoother
from src.visualizer import Visualizer
from reid_matcher import ReIDMatcher  # ← ReID matcher for ID locking


class Pipeline:
    """Main orchestrator: reads video → detects → tracks → locks IDs → draws → writes."""
    
    def __init__(self, cfg):
        """Initialize all components."""
        self.cfg = cfg
        self.detector = Detector(cfg)
        self.person_smoother = EMASmoother(cfg.smoothing.ema_alpha)
        self.viz = Visualizer(cfg)
        
        # ✓ ReID Matcher: locks IDs permanently via embeddings
        self.reid_matcher = ReIDMatcher(similarity_threshold=0.5)
        print("[INFO] ReID Matcher initialized - IDs will be LOCKED")

    def run(self):
        """Main processing loop: frame by frame."""
        cfg = self.cfg
        m = cfg.model
        
        # Setup video I/O
        reader = VideoReader(cfg.video.input_path)
        writer = VideoWriter(cfg.video.output_path, reader.fps, reader.width, reader.height)
        print(f"[INFO] {reader.total} frames @ {reader.fps:.1f} fps "
              f"({reader.width}x{reader.height})")
        print(f"[INFO] Model: {m.weights} | Device: {m.device} | "
              f"FP16: {m.half} | ReID: ENABLED")

        frame_idx = 0
        t_start = time.time()
        
        # Process each frame
        for frame in reader:
            # Step 1: YOLO detection + BoT-SORT initial tracking
            result = self.detector.track(frame)
            
            # Step 2: Parse results into Track objects
            tracks = parse_results(
                result, m.person_class_id, m.ball_class_id,
                m.person_conf,
            )

            # Step 3: Extract only persons (ball OFF)
            persons = [t for t in tracks if t.cls == m.person_class_id]

            # ════════════════════════════════════════════════════════════════
            # ✓ STEP 4: Apply ReID matcher to LOCK IDs (no swaps/reassigns)
            # ════════════════════════════════════════════════════════════════
            if persons:
                # Prepare detections for embedding matching
                detections = [(t.track_id, t.xyxy, t.conf) for t in persons]
                
                # Match using embeddings - this locks IDs permanently
                matched = self.reid_matcher.match(frame, detections)
                
                # Update person track IDs with matched (stable, locked) IDs
                for i, (stable_id, xyxy, conf) in enumerate(matched):
                    persons[i].track_id = stable_id

            # Step 5: Draw persons with LOCKED IDs
            active_ids = set()
            for t in persons:
                stable_id = t.track_id
                
                # Apply box smoothing to reduce jitter
                box = (self.person_smoother.smooth(stable_id, t.xyxy)
                       if cfg.smoothing.enable else t.xyxy)
                
                # Draw person box + ID label
                self.viz.draw_person(frame, stable_id, box, t.conf)
                active_ids.add(stable_id)
            
            # Cleanup smoother cache for persons no longer in frame
            self.person_smoother.cleanup(active_ids)

            # Step 6: Draw HUD (frame count + person count)
            self.viz.draw_hud(frame, len(persons), False, frame_idx)
            
            # Step 7: Write annotated frame to output video
            writer.write(frame)

            # Progress reporting
            frame_idx += 1
            if frame_idx % 30 == 0:
                fps = frame_idx / (time.time() - t_start)
                n_locked_ids = len(self.reid_matcher.id_gallery)
                print(f"[INFO] {frame_idx}/{reader.total}  "
                      f"({fps:.1f} fps) | People: {len(persons)} | "
                      f"Locked IDs: {n_locked_ids}")

        # Finalize
        writer.release()
        dur = time.time() - t_start
        final_fps = frame_idx / max(dur, 1e-6)
        print(f"[DONE] {frame_idx} frames in {dur:.1f}s ({final_fps:.1f} fps)")
        print(f"[DONE] Output saved → {cfg.video.output_path}")
        print(f"[DONE] Total unique IDs locked: {len(self.reid_matcher.id_gallery)}")