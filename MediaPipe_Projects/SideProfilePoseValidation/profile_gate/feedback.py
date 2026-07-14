"""Module 9 - Feedback engine.

Turns the set of failing reason keys into ONE actionable instruction. Only the
highest-priority correction is surfaced so the user fixes one thing at a time
instead of reading a wall of errors. Orientation guidance is directional: it
inspects the current yaw / facing / label to say *which way* to turn and *how
much*, rather than a flat "invalid".
"""
from typing import Dict, List

from .config import ProfileGateConfig, Orientation
from .orientation import OrientationResult

# Priority order (top = most important). The first reason present wins.
PRIORITY: List[str] = [
    "no_person",
    "multiple_persons",
    "show_head",
    "show_feet",
    "full_body",
    "too_far",
    "too_close",
    "head_cropped",
    "feet_cropped",
    "body_out_of_frame",
    "orientation",        # directional message built separately
    "wrong_facing",
    "head_not_profile",
    "head_misaligned",
    "leaning",
    "bent_hip",
    "bent_knee",
    "shoulders_uneven",
    "neck_flex",
    "head_pitch",
    "head_roll",
    "head_unreliable",
    "low_confidence",
]

# Machine reason-slugs from OrientationResult.reason -> human-readable text.
# Covers every non-profile orientation outcome from Module 3's hard gates.
_ORIENTATION_REASON: Dict[str, str] = {
    "slight_rotation":  "Body not perpendicular - turn further to your side",
    "three_quarter":    "Three-quarter pose - turn a little more",
    "twisted_torso":    "Twisted torso - square your shoulders and hips",
    "not_full_side":    "Half side - turn until the far side is hidden",
    "ambiguous_facing": "Can't tell which way you're facing - turn more clearly to the side",
}

_STATIC: Dict[str, str] = {
    "no_person":         "Step into the frame",
    "multiple_persons":  "Only one person in frame, please",
    "show_head":         "Show your head - move down or step back",
    "show_feet":         "Show your feet - step back",
    "full_body":         "Get your full body in the frame",
    "too_far":           "Move closer to the camera",
    "too_close":         "Step back from the camera",
    "head_cropped":      "Your head is cut off - step back",
    "feet_cropped":      "Your feet are cut off - step back",
    "body_out_of_frame": "Keep your whole body inside the frame",
    "wrong_facing":      "Turn to face the other direction",
    "head_not_profile":  "Turn your head to the side too - keep it in profile",
    "head_misaligned":   "Face forward along your body, don't look back",
    "leaning":           "Stand up straight, don't lean",
    "bent_hip":          "Straighten up - don't bend at the waist",
    "bent_knee":         "Straighten your legs",
    "shoulders_uneven":  "Level your shoulders",
    "neck_flex":         "Keep your head up and aligned",
    "head_pitch":        "Look straight ahead, don't nod up or down",
    "head_roll":         "Head turned toward camera - keep it in profile",
    "head_unreliable":   "Face the camera area so your head is visible",
    "low_confidence":    "Hold a cleaner side profile - straighten up and turn fully",
}

READY_MESSAGE = "Perfect side profile - hold still"


def _orientation_message(orient: OrientationResult,
                         cfg: ProfileGateConfig) -> str:
    """Directional guidance toward the nearest valid side profile.

    Prefers the specific machine reason slug set by orientation.classify()
    (twisted torso, half side, ambiguous facing, ...) and falls back to a
    generic FRONT/BACK/OBLIQUE message when no slug applies (e.g. a clean
    profile rejected only for facing preference).
    """
    if orient.label == Orientation.FRONT:
        return "Front Facing - turn 90 degrees to show your side profile"
    if orient.label == Orientation.BACK:
        return "Back Facing - turn 90 degrees to show your side profile"

    if orient.reason and orient.reason in _ORIENTATION_REASON:
        return _ORIENTATION_REASON[orient.reason]

    if orient.label == Orientation.OBLIQUE:
        return "Keep turning to your side"

    # already a clean profile but rejected for facing preference
    if cfg.accept_facing == "left":
        return "Turn to face the left"
    if cfg.accept_facing == "right":
        return "Turn to face the right"
    return "Turn to show a clean side profile"


def build_message(reasons: Dict[str, dict], orient: OrientationResult,
                  cfg: ProfileGateConfig) -> str:
    """reasons: mapping reason_key -> detail dict for every failing check."""
    if not reasons:
        return READY_MESSAGE
    for key in PRIORITY:
        if key in reasons:
            if key == "orientation":
                return _orientation_message(orient, cfg)
            return _STATIC.get(key, "Adjust your position")
    return "Adjust your position"