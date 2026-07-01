"""
annotator.py
============
Draws tracked boxes and ID labels onto frames using `supervision`. Colors are
chosen by track ID, so each player keeps one consistent color for the whole
clip (which makes ID switches easy to spot by eye).
"""

import supervision as sv

BALL_CLASS_ID = 32

PALETTE = [
    "#00FF00", "#FF0055", "#00FFFF", "#FFFF00", "#FF00FF",
    "#FF9900", "#FFFFFF", "#FF3333", "#33FFFF", "#FFFF33",
]


class Annotator:
    def __init__(self):
        palette = sv.ColorPalette.from_hex(PALETTE)
        self.box = sv.RoundBoxAnnotator(
            color=palette, thickness=3, color_lookup=sv.ColorLookup.TRACK)
        self.label = sv.LabelAnnotator(
            color=palette, text_scale=1.2, text_thickness=3,
            text_padding=10, color_lookup=sv.ColorLookup.TRACK)

    def draw(self, frame, detections):
        if len(detections) == 0:
            return frame
        labels = [
            f"BALL #{tid}" if cid == BALL_CLASS_ID else f"#{tid}"
            for tid, cid in zip(detections.tracker_id, detections.class_id)
        ]
        frame = self.box.annotate(scene=frame, detections=detections)
        frame = self.label.annotate(scene=frame, detections=detections, labels=labels)
        return frame
