"""
All drawing lives here. Colors are deterministic per-ID (golden-ratio
hue hashing) so the SAME id always gets the SAME color -> visually stable.
"""
import cv2
import numpy as np
from collections import deque

BALL_COLOR = (0, 215, 255)     # BGR amber for the ball


class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ball_trail = deque(maxlen=cfg.visual.trajectory_length)

    # ---- deterministic color per person id ----
    @staticmethod
    def _color(track_id: int):
        hue = (track_id * 0.61803398875) % 1.0          # golden ratio spacing
        hsv = np.uint8([[[int(hue * 179), 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _label(self, frame, x1, y1, text, color):
        fs = self.cfg.visual.font_scale
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        y1 = max(y1, th + bl + 6)
        cv2.rectangle(frame, (x1, y1 - th - bl - 6), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_person(self, frame, track_id, xyxy, conf):
        x1, y1, x2, y2 = map(int, xyxy)
        color = self._color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.cfg.visual.person_thickness)
        self._label(frame, x1, y1, f"ID {track_id}", color)

    def draw_ball(self, frame, xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, self.cfg.visual.ball_thickness)
        cv2.circle(frame, (cx, cy), 3, BALL_COLOR, -1)
        self._label(frame, x1, y1, "BALL", BALL_COLOR)

        if self.cfg.visual.show_trajectory:
            self.ball_trail.append((cx, cy))
            pts = list(self.ball_trail)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], BALL_COLOR, 2)

    def draw_hud(self, frame, n_people, ball_present, frame_idx):
        txt = f"Frame {frame_idx} | People: {n_people} | Ball: {'yes' if ball_present else 'no'}"
        cv2.rectangle(frame, (0, 0), (360, 30), (0, 0, 0), -1)
        cv2.putText(frame, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
