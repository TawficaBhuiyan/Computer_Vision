"""Draw persons. Ball OFF. Deterministic color per ID."""
import cv2
import numpy as np


class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg

    # Golden-ratio hue for stable per-ID colors
    @staticmethod
    def _color(track_id: int):
        hue = (track_id * 0.61803398875) % 1.0
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

    def draw_hud(self, frame, n_people, ball_present, frame_idx):
        txt = f"Frame {frame_idx} | People: {n_people}"
        cv2.rectangle(frame, (0, 0), (300, 30), (0, 0, 0), -1)
        cv2.putText(frame, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)