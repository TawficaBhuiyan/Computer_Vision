"""Module 3 - Side-profile orientation classification (the core module).

Approach
--------
A true side profile (LEFT or RIGHT, symmetric by construction) must satisfy
THREE independent geometric conditions at once - no single number is trusted
alone:

1. SIDE-ON:   the shoulder/hip line spans depth, not width.
                  yaw = atan2(dz, dx),  d = shoulder_R - shoulder_L
              ~0 deg  -> frontal / back (shoulders span x)
              ~90 deg -> true profile  (shoulders span depth)
              Measured in the metric WORLD frame so it is independent of
              camera distance, focal length and image aspect ratio.

2. NOT TWISTED: the shoulder line and the hip line must have turned by
              (about) the same amount. A large gap between signed shoulder
              yaw and signed hip yaw means the torso is twisted, not simply
              rotated - a real side profile turns as one rigid unit.

3. ONE SIDE VISIBLE: the near/far landmark of every left/right pair should
              overlap in image x (they're stacked along the camera's line of
              sight) and differ sharply in visibility (the far side is
              genuinely occluded). This is the 2D corroboration of (1) that
              catches a "half side" pose where the yaw angle alone might
              already read as side-on but the body hasn't actually turned
              far enough to hide the far side.

Which way (LEFT vs RIGHT) the subject faces is a SEPARATE question from "is
this a side profile at all", and is answered by fusing four independent
signed cues by WEIGHTED VOTE (not requiring unanimous agreement, which is
fragile to single-cue noise and was the root cause of asymmetric flakiness in
the previous implementation):

    * image nose-vs-ear x      (facing_from_image)   - no depth-sign
      assumption at all, so it gets the largest weight.
    * world shoulder depth     (shoulder_depth_sign)  - which shoulder is
      nearer the camera.
    * world hip depth          (hip_depth_sign)       - independent
      corroboration of the shoulder cue.
    * image-space shoulder depth (image z column)     - a fourth, noisier
      corroborating signal computed the same way as the world cue but from
      the pose model's raw image-space z.

All four use different underlying quantities, so they rarely fail together;
weighting and summing (instead of AND-ing) means one noisy cue can be
outvoted rather than veto the whole frame - this is what makes LEFT and RIGHT
symmetric in practice, not just in the formulas.

FRONT vs BACK (both low-yaw) is disambiguated by whether the face points
toward the camera, using nose depth relative to the ears.

Every sub-score below is normalized to [0, 1] (1 = ideal) and blended into one
fused `confidence`, per cfg.w_*. A pose can pass every individual hard gate
and still be rejected if the fused evidence is weak (cfg.min_profile_confidence).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import geometry as geo
from . import landmarks as lm
from .config import (
    ProfileGateConfig, Orientation, NOSE, LEFT_EAR, RIGHT_EAR,
    LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
)


@dataclass
class OrientationResult:
    label: str                    # Orientation.*
    side: Optional[str] = None    # "LEFT" | "RIGHT" | None
    facing: int = 0                # -1 faces frame-left, +1 faces frame-right, 0
    yaw: float = 0.0                # shoulder yaw magnitude in [0, 90]
    signed_yaw: float = 0.0         # signed shoulder yaw (rotation direction)
    hip_yaw: float = 0.0            # hip yaw magnitude
    twist: float = 0.0              # |shoulder yaw - hip yaw|, deg
    overlap_score: float = 0.0      # 0-1, 1 = shoulders/hips fully overlap in x
    visibility_score: float = 0.0   # 0-1, 1 = strong near/far visibility gap
    side_agreement: float = 0.0     # 0-1, strength of the L/R vote fusion
    confidence: float = 0.0         # [0, 1] fused overall score
    reason: Optional[str] = None    # machine slug explaining a non-profile label
    cues: Dict[str, float] = field(default_factory=dict)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _front_or_back(image: np.ndarray) -> str:
    """Low-yaw disambiguation. Nose nearer than ears (in image z) -> FRONT."""
    nose_z = image[NOSE, 2]
    ear_z = (image[LEFT_EAR, 2] + image[RIGHT_EAR, 2]) / 2.0
    # BlazePose image-z: smaller (more negative) is closer to the camera.
    return Orientation.FRONT if nose_z <= ear_z else Orientation.BACK


def _side_vote(image: np.ndarray, world: np.ndarray,
              cfg: ProfileGateConfig) -> tuple:
    """Fuse four independent signed facing cues into one weighted vote.

    Returns (side_score in [-1, 1], agreement in [0, 1], cues dict). Positive
    side_score -> RIGHT, negative -> LEFT. `agreement` is |side_score|: how
    strongly the cues agree, used both to gate ambiguous frames and as a
    confidence contributor.
    """
    cue_img = lm.facing_from_image(image)
    cue_sh = geo.shoulder_depth_sign(world)
    cue_hip = geo.hip_depth_sign(world)
    cue_imgz = 1 if image[LEFT_SHOULDER, 2] > image[RIGHT_SHOULDER, 2] else (
        -1 if image[LEFT_SHOULDER, 2] < image[RIGHT_SHOULDER, 2] else 0)

    weights = (cfg.side_weight_image, cfg.side_weight_shoulder_z,
              cfg.side_weight_hip_z, cfg.side_weight_image_z)
    votes = (cue_img, cue_sh, cue_hip, cue_imgz)
    total_w = sum(weights) or 1.0
    side_score = sum(w * v for w, v in zip(weights, votes)) / total_w

    cues = {"image_x": cue_img, "shoulder_z": cue_sh,
           "hip_z": cue_hip, "image_z": cue_imgz}
    return side_score, abs(side_score), cues


def classify(image: np.ndarray, world: np.ndarray,
             cfg: ProfileGateConfig) -> OrientationResult:
    # -- Module 3a: how side-on is the body (world-frame, distance-invariant) #
    yaw = geo.shoulder_yaw_magnitude(world)
    signed_yaw = geo.signed_shoulder_yaw(world)
    hip_yaw = geo.hip_yaw_magnitude(world)
    signed_hip_yaw = geo.signed_hip_yaw(world)
    twist = geo.torso_twist(signed_yaw, signed_hip_yaw)

    # -- Module 3b: 2D corroboration (image-frame) -------------------------- #
    scale = lm.torso_scale(image)
    shoulder_overlap = lm.x_overlap_ratio(image, LEFT_SHOULDER, RIGHT_SHOULDER, scale)
    hip_overlap = lm.x_overlap_ratio(image, LEFT_HIP, RIGHT_HIP, scale)
    shoulder_vis_gap = lm.visibility_gap(image, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_vis_gap = lm.visibility_gap(image, LEFT_HIP, RIGHT_HIP)
    ear_vis_gap = lm.visibility_gap(image, LEFT_EAR, RIGHT_EAR)
    eye_vis_gap = lm.visibility_gap(image, LEFT_EYE, RIGHT_EYE)

    # -- Module 3c: which side (LEFT/RIGHT), fused by weighted vote --------- #
    side_score, side_agreement, cues = _side_vote(image, world, cfg)
    if side_agreement >= cfg.min_side_agreement:
        facing = 1 if side_score > 0 else -1
        side = "RIGHT" if facing > 0 else "LEFT"
    else:
        facing, side = 0, None

    # -- sub-scores, each normalized to [0, 1] (1 = ideal profile) ---------- #
    yaw_score = _clamp01(1.0 - abs(cfg.yaw_ideal - yaw) / cfg.yaw_ideal)
    overlap_score = _clamp01(1.0 - max(
        shoulder_overlap / cfg.max_shoulder_overlap,
        hip_overlap / cfg.max_hip_overlap,
    ))
    visibility_score = _clamp01(min(
        max(shoulder_vis_gap, hip_vis_gap),
        max(ear_vis_gap, eye_vis_gap),
    ) / cfg.min_visibility_gap)
    twist_score = _clamp01(1.0 - twist / cfg.max_torso_twist)

    confidence = (
        cfg.w_yaw * yaw_score
        + cfg.w_overlap * overlap_score
        + cfg.w_visibility * visibility_score
        + cfg.w_twist * twist_score
        + cfg.w_side_agreement * side_agreement
    )
    confidence = round(_clamp01(confidence), 3)

    # -- classification: hard gates first, most specific reason wins -------- #
    label, reason = _decide_label(
        yaw, twist, shoulder_overlap, hip_overlap, side, facing, image, cfg,
    )

    return OrientationResult(
        label=label, side=side, facing=facing,
        yaw=round(yaw, 1), signed_yaw=round(signed_yaw, 1),
        hip_yaw=round(hip_yaw, 1), twist=round(twist, 1),
        overlap_score=round(overlap_score, 3),
        visibility_score=round(visibility_score, 3),
        side_agreement=round(side_agreement, 3),
        confidence=confidence, reason=reason, cues=cues,
    )


def _decide_label(yaw: float, twist: float, shoulder_overlap: float,
                  hip_overlap: float, side: Optional[str], facing: int,
                  image: np.ndarray, cfg: ProfileGateConfig) -> tuple:
    """Priority-ordered hard gates. Returns (Orientation label, reason slug)."""
    if yaw < cfg.oblique_yaw_min:
        return _front_or_back(image), None  # reason carried by the label itself

    if yaw < cfg.profile_yaw_min:
        midpoint = (cfg.oblique_yaw_min + cfg.profile_yaw_min) / 2.0
        reason = "slight_rotation" if yaw < midpoint else "three_quarter"
        return Orientation.OBLIQUE, reason

    if twist > cfg.max_torso_twist:
        return Orientation.OBLIQUE, "twisted_torso"

    if shoulder_overlap > cfg.max_shoulder_overlap or hip_overlap > cfg.max_hip_overlap:
        return Orientation.OBLIQUE, "not_full_side"

    if side is None:
        return Orientation.OBLIQUE, "ambiguous_facing"

    label = Orientation.PROFILE_RIGHT if facing > 0 else Orientation.PROFILE_LEFT
    return label, None
