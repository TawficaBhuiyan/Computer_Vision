"""Independent validators, one per requirement module.

Each validator is a pure function of (data, cfg) that returns a Check:
    Check(ok: bool, reason: Optional[str], detail: dict)

Keeping them independent (Single Responsibility) means a module can be reworked
or unit-tested in isolation, and gate.py just composes them in priority order.
The ``reason`` string is a stable key consumed by feedback.py.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import geometry as geo
from . import landmarks as lm
from .config import (
    ProfileGateConfig, Orientation, BODY_LEVELS, LEFT_SIDE, RIGHT_SIDE,
)
from .head import HeadPose
from .orientation import OrientationResult


@dataclass
class Check:
    ok: bool
    reason: Optional[str] = None
    detail: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Module 1 - single person                                                    #
# --------------------------------------------------------------------------- #
def validate_single_person(n_persons: int, cfg: ProfileGateConfig) -> Check:
    if n_persons == 0:
        return Check(False, "no_person", {"n": 0})
    if n_persons > cfg.max_persons:
        return Check(False, "multiple_persons", {"n": n_persons})
    return Check(True, detail={"n": n_persons})


# --------------------------------------------------------------------------- #
# Module 2 - full-body visibility (profile-aware)                             #
# --------------------------------------------------------------------------- #
def validate_full_body(image: np.ndarray, cfg: ProfileGateConfig) -> Check:
    """Every vertical level (head..foot) must be covered by its NEAR side.

    We take the max visibility over each left/right pair, because the far side
    of a legitimate profile is occluded. A missing level names itself so the
    feedback engine can say 'show your feet' / 'show your head'.
    """
    missing = []
    for level, (l_idx, r_idx) in BODY_LEVELS.items():
        thr = cfg.foot_visibility if level == "foot" else cfg.level_visibility
        if lm.level_confidence(image, l_idx, r_idx) < thr:
            missing.append(level)
    if missing:
        # prioritise the most informative correction
        if "foot" in missing or "ankle" in missing:
            return Check(False, "show_feet", {"missing": missing})
        if "head" in missing:
            return Check(False, "show_head", {"missing": missing})
        return Check(False, "full_body", {"missing": missing})
    return Check(True)


# --------------------------------------------------------------------------- #
# Module 7 - frame boundary                                                   #
# --------------------------------------------------------------------------- #
def validate_frame_boundary(image: np.ndarray, cfg: ProfileGateConfig,
                            head_top: Optional[float] = None,
                            foot_bottom: Optional[float] = None) -> Check:
    """`head_top` / `foot_bottom` let the caller substitute TEMPORALLY SMOOTHED
    values for the raw per-frame extremes (see validate_orientation's
    docstring for why). A foot resting right at the frame edge naturally
    straddles the margin line every other frame from ordinary jitter; gating
    on the raw extreme flickers exactly there.
    """
    m = cfg.frame_margin
    head_top = lm.head_top_y(image, cfg.level_visibility) if head_top is None else head_top
    if head_top < m:
        return Check(False, "head_cropped", {"y": round(head_top, 3)})
    foot_bottom = lm.foot_bottom_y(image, cfg.foot_visibility) if foot_bottom is None else foot_bottom
    if foot_bottom > 1.0 - m:
        return Check(False, "feet_cropped", {"y": round(foot_bottom, 3)})
    # any confident landmark past the left/right edges
    for idx in range(image.shape[0]):
        if image[idx, 3] >= cfg.level_visibility:
            x = image[idx, 0]
            if x < m or x > 1.0 - m:
                return Check(False, "body_out_of_frame", {"idx": idx})
    return Check(True)


# --------------------------------------------------------------------------- #
# Module 6 - camera distance                                                  #
# --------------------------------------------------------------------------- #
def validate_distance(image: np.ndarray, cfg: ProfileGateConfig) -> Check:
    fill = lm.body_fill(image)
    if fill > cfg.max_body_fill:
        return Check(False, "too_close", {"fill": round(fill, 3)})
    if fill < cfg.min_body_fill:
        return Check(False, "too_far", {"fill": round(fill, 3)})
    return Check(True, detail={"fill": round(fill, 3)})


# --------------------------------------------------------------------------- #
# Module 3 - orientation gate (consumes the classifier output)                #
# --------------------------------------------------------------------------- #
def validate_orientation(orient: OrientationResult, cfg: ProfileGateConfig,
                         label: Optional[str] = None,
                         confidence: Optional[float] = None) -> Check:
    """Gate on the orientation label / confidence.

    `label` and `confidence` let the caller substitute TEMPORALLY SMOOTHED
    values (majority-voted label, One-Euro-filtered confidence) for the raw
    per-frame ones on ``orient``. MediaPipe landmarks jitter frame to frame;
    gating on raw values means a single noisy frame can flip the label at a
    boundary (e.g. PROFILE_RIGHT <-> OBLIQUE) or dip confidence a hair below
    threshold, which resets the caller's READY streak even though the pose
    never actually changed. Defaulting to ``orient``'s own raw fields keeps
    this function usable standalone (e.g. in unit tests) without a gate.
    """
    label = orient.label if label is None else label
    confidence = orient.confidence if confidence is None else confidence
    detail = {"label": label, "yaw": orient.yaw, "facing": orient.facing,
             "side": orient.side, "confidence": confidence,
             "reason": orient.reason}
    if label in Orientation.VALID_PROFILES:
        if cfg.accept_facing == "left" and label != Orientation.PROFILE_LEFT:
            return Check(False, "wrong_facing", detail)
        if cfg.accept_facing == "right" and label != Orientation.PROFILE_RIGHT:
            return Check(False, "wrong_facing", detail)
        # hard gates passed, but the fused evidence is still too weak -
        # score fusion catches cases no single threshold would (e.g. every
        # cue individually borderline).
        if confidence < cfg.min_profile_confidence:
            return Check(False, "low_confidence", detail)
        return Check(True, detail=detail)
    # not a clean profile -> hand the label + reason slug to the feedback engine
    return Check(False, "orientation", detail)


# --------------------------------------------------------------------------- #
# Module 4 - head pose                                                        #
# --------------------------------------------------------------------------- #
def validate_head(head: HeadPose, orient: OrientationResult,
                  cfg: ProfileGateConfig, level: Optional[float] = None,
                  spread: Optional[float] = None) -> Check:
    """`level` / `spread` let the caller substitute TEMPORALLY SMOOTHED values
    for head.level / head.spread (see validate_orientation's docstring for
    why). head.spread in particular is a small-numerator/small-denominator
    ratio - eye/ear pixel separation over head height - so raw per-frame
    landmark jitter swings it by a large relative amount even when the head
    genuinely never left profile; gating on the raw value flickers.
    """
    if not head.ok:
        return Check(False, "head_unreliable", {})
    level = head.level if level is None else level
    spread = head.spread if spread is None else spread
    if level is not None and not (
            cfg.head_level_min <= level <= cfg.head_level_max):
        return Check(False, "head_pitch", {"level": round(level, 1)})
    if spread is None:
        return Check(False, "head_unreliable", {})
    if spread > cfg.max_head_spread:
        return Check(False, "head_roll", {"spread": round(spread, 3)})

    # head facing must agree with the body facing (not looking back/over shoulder)
    if (orient.facing != 0 and head.facing != 0
            and head.facing != orient.facing):
        return Check(False, "head_misaligned",
                     {"head": head.facing, "body": orient.facing})
    return Check(True)


# --------------------------------------------------------------------------- #
# Module 5 - body alignment / posture                                         #
# --------------------------------------------------------------------------- #
def validate_posture(world: np.ndarray, orient: OrientationResult,
                     cfg: ProfileGateConfig) -> Check:
    # near-side leg chain, chosen from the facing direction
    side = LEFT_SIDE if orient.facing < 0 else RIGHT_SIDE

    tilt = geo.torso_tilt(world)
    roll = geo.shoulder_roll(world)                 # display only (degenerate)
    knee = geo.knee_angle(world, side["hip"], side["knee"], side["ankle"])
    hipang = geo.hip_angle(world, side["shoulder"], side["hip"], side["knee"])
    neck = geo.neck_flex(world, side["ear"])

    # full metric set is always returned so the panel can show live posture,
    # regardless of which (if any) condition fails.
    detail = {
        "torso_tilt": round(tilt, 1), "shoulder_roll": round(roll, 1),
        "knee_angle": round(knee, 1), "hip_angle": round(hipang, 1),
        "neck_flex": round(neck, 1),
    }

    if tilt > cfg.max_torso_tilt:
        return Check(False, "leaning", detail)
    if knee < cfg.min_knee_angle:
        return Check(False, "bent_knee", detail)
    if hipang < cfg.min_hip_angle:
        return Check(False, "bent_hip", detail)
    if neck > cfg.max_neck_flex:
        return Check(False, "neck_flex", detail)
    return Check(True, detail=detail)
