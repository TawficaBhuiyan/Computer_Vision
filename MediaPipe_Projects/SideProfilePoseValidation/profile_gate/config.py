"""Landmark index constants and every tunable threshold.

Keeping all magic numbers in one place means tuning the validator never
requires editing logic in geometry.py, orientation.py, head.py, validators.py
or gate.py. Adjust thresholds here for a different camera, distance or subject
population.

BlazePose GHUM returns 33 landmarks. "LEFT" / "RIGHT" always refer to the
SUBJECT's own left / right, which appear mirrored on a selfie webcam feed.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# BlazePose 33-landmark indices                                               #
# --------------------------------------------------------------------------- #
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Face keypoints used by solvePnP head-pose (fixed semantic order).
FACE_IDS: Tuple[int, ...] = (
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, MOUTH_LEFT, MOUTH_RIGHT,
)

# --------------------------------------------------------------------------- #
# Body "levels" for profile-aware full-body coverage (Module 2).              #
#                                                                             #
# In a TRUE side profile the far-side landmark of each pair is self-occluded  #
# and gets a low visibility score, so we must NOT require both sides. Instead #
# we require that at least the NEAR side of each vertical level is visible and #
# in-frame. That verifies head-to-toe coverage without making a valid profile #
# impossible to pass.                                                         #
# --------------------------------------------------------------------------- #
BODY_LEVELS: Dict[str, Tuple[int, int]] = {
    "head":     (LEFT_EAR, RIGHT_EAR),
    "shoulder": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "hip":      (LEFT_HIP, RIGHT_HIP),
    "knee":     (LEFT_KNEE, RIGHT_KNEE),
    "ankle":    (LEFT_ANKLE, RIGHT_ANKLE),
    "foot":     (LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX),
}

# Left/right index lookup, used to pick the near-side chain once facing is known.
LEFT_SIDE = {
    "shoulder": LEFT_SHOULDER, "elbow": LEFT_ELBOW, "wrist": LEFT_WRIST,
    "hip": LEFT_HIP, "knee": LEFT_KNEE, "ankle": LEFT_ANKLE,
    "heel": LEFT_HEEL, "foot": LEFT_FOOT_INDEX, "ear": LEFT_EAR, "eye": LEFT_EYE,
}
RIGHT_SIDE = {
    "shoulder": RIGHT_SHOULDER, "elbow": RIGHT_ELBOW, "wrist": RIGHT_WRIST,
    "hip": RIGHT_HIP, "knee": RIGHT_KNEE, "ankle": RIGHT_ANKLE,
    "heel": RIGHT_HEEL, "foot": RIGHT_FOOT_INDEX, "ear": RIGHT_EAR, "eye": RIGHT_EYE,
}

# Skeleton connections for rendering (subject-agnostic subset of BlazePose).
POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT_INDEX),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
    (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
)


# --------------------------------------------------------------------------- #
# Orientation labels                                                          #
# --------------------------------------------------------------------------- #
class Orientation:
    FRONT = "FRONT"
    BACK = "BACK"
    OBLIQUE = "OBLIQUE"
    PROFILE_LEFT = "PROFILE_LEFT"     # subject faces the LEFT edge of the frame
    PROFILE_RIGHT = "PROFILE_RIGHT"   # subject faces the RIGHT edge of the frame
    UNKNOWN = "UNKNOWN"

    VALID_PROFILES = (PROFILE_LEFT, PROFILE_RIGHT)


@dataclass(frozen=True)
class ProfileGateConfig:
    """Every tunable threshold for the Side Profile validator."""

    # -- Module 1: single person ------------------------------------------- #
    max_persons: int = 1

    # -- Module 2: full-body visibility ------------------------------------ #
    # Tolerant because the far side of a profile is legitimately low-visibility.
    level_visibility: float = 0.5     # min visibility for a level's near landmark
    foot_visibility: float = 0.3      # feet are often the least confident level

    # -- Module 3: side-profile orientation -------------------------------- #
    profile_yaw_min: float = 60     # >= this body yaw (deg) counts as side-on
    oblique_yaw_min: float = 30.0     # [oblique_min, profile_min) = partial turn
    yaw_ideal: float = 90.0           # a perfect profile

    # accepted facing direction(s): "both", "left", or "right"
    accept_facing: str = "both"

    # -- Module 3b: 2D corroboration for "true" side-on (image-space) ------ #
    # Shoulder/hip pair x-separation normalized by torso scale (shoulder-mid to
    # hip-mid distance). ~0 -> the pair overlaps in x (one occludes the other,
    # as it must in a true profile). ~1 -> spread as wide as the torso is tall
    # (frontal/back view). Also used as the "shoulder/hip width ratio" cue: the
    # same normalized separation IS the width-collapse ratio for that pair.
    # Calibrated against real footage (not a synthetic guess): a genuine side
    # profile with natural arm swing (e.g. walking) reads up to ~0.55-0.60 for
    # shoulders / ~0.30-0.35 for hips, while a frontal/back view reads ~1.0+.
    # The original 0.22 guess left no room for real gait motion and rejected
    # every walking-sideways frame as "half side".
    max_shoulder_overlap: float = 0.65
    max_hip_overlap: float = 0.45
    # |shoulder yaw - hip yaw|: shoulders and hips rotating by different amounts
    # means the torso is twisted, not a clean side-on turn.
    max_torso_twist: float = 25.0
    # near-vs-far visibility gap that proves "only one side of the body is
    # clearly visible" (shoulders, hips, ears, eyes all self-occlude in a true
    # profile; a small gap means both sides are equally visible, i.e. frontal).
    # NOTE: on real footage this signal is far weaker than the geometric ones -
    # MediaPipe estimates ("hallucinates") occluded landmarks at high
    # confidence rather than dropping their visibility score, so the gap stays
    # near 0 even at a perfect 90 deg profile. Kept as a minor, lightly-weighted
    # corroborating cue (see w_visibility) rather than a primary discriminator.
    min_visibility_gap: float = 0.03

    # -- Module 3c: left/right side-vote fusion ----------------------------- #
    # Four INDEPENDENT signed cues for which way the subject faces, combined by
    # weighted vote instead of requiring unanimous agreement. Weighted so the
    # sign-convention-free image cue (nose vs ear-mid x) dominates, while the
    # noisier world/image depth cues corroborate. Weights need not sum to 1;
    # they are normalized internally.
    side_weight_image: float = 0.40      # image nose-vs-ear x (most robust)
    side_weight_shoulder_z: float = 0.25  # world shoulder depth sign
    side_weight_hip_z: float = 0.25       # world hip depth sign
    side_weight_image_z: float = 0.10     # image-space shoulder depth (noisier)
    # net weighted vote, folded to [0, 1]; below this the facing direction is
    # too ambiguous to call LEFT or RIGHT even if the body is side-on.
    min_side_agreement: float = 0.34

    # -- Module 3d: fused confidence (score fusion, not one threshold) ------ #
    # Weights for the overall profile-confidence score; must sum to ~1.0.
    # w_visibility is deliberately small: real MediaPipe output rarely shows a
    # usable near/far visibility gap (see min_visibility_gap above), so
    # weighting it like the geometric cues would dock every frame's confidence
    # for a signal that carries almost no real information on this landmarker.
    w_yaw: float = 0.35           # how close to a perfect 90 deg turn
    w_overlap: float = 0.30       # shoulder/hip x-overlap (2D corroboration)
    w_visibility: float = 0.05    # near/far visibility asymmetry (weak signal)
    w_twist: float = 0.15         # shoulder/hip yaw agreement (not twisted)
    w_side_agreement: float = 0.15  # strength of the L/R vote fusion
    # fused score floor (0-1); a frame can pass every hard gate and still be
    # rejected here if the combined evidence is weak.
    min_profile_confidence: float = 0.60

    # -- Module 4: head pose ----------------------------------------------- #
    # Geometric head-level angle (nose vs near ear, deg). +down / -up. The near-
    # neutral band accepts a natural forward gaze and rejects a strong nod. This
    # replaces solvePnP pitch/roll, which is ill-conditioned at a full profile.
    head_level_min: float = -30.0     # looking up limit
    # Calibrated against real footage: this angle's neutral resting value
    # depends on camera height relative to the subject's head (a camera below
    # eye level reads a naturally higher "level" even with a level gaze), so
    # 45 deg left almost no margin for real subjects/camera setups - real data
    # showed a genuine standing profile at median ~23 deg but p90 ~46 deg.
    head_level_max: float = 55.0      # looking down limit
    head_min_visibility: float = 0.5
    # Head yaw / "is the head itself side-on". Horizontal spread of the eyes and
    # ears, normalized by head height: ~0.03-0.09 for a true head profile, but
    # ~0.4-1.1 once the head rotates toward (or away from) the camera. Above this
    # the head is no longer in profile even if the body is, so the frame is
    # invalid - head AND body must both be side-on.
    # NOTE: this is a ratio of two small pixel distances (eye/ear separation
    # over head height), so it is disproportionately sensitive to ordinary
    # per-frame landmark jitter. Calibrated against real footage: on real
    # video (vs. the idealized 0.03-0.09 comment above) a genuinely side-on
    # head reads median ~0.24 with a fat noisy tail (p75 ~0.37), while a truly
    # frontal head reads >=0.87. 0.45 keeps a solid ~2x margin below the
    # frontal floor while accepting the bulk of genuine profile frames -
    # pushing this further starts eating into the frontal range and defeats
    # the check (see docs/threshold_tuning.md for the full calibration data).
    max_head_spread: float = 0.45

    # -- Module 5: body alignment / posture -------------------------------- #
    # Only metrics that are WELL-CONDITIONED in a profile geometry gate here.
    # Shoulder roll (frontal-plane levelness) is degenerate side-on and is shown
    # for reference but not gated.
    max_torso_tilt: float = 18.0      # spine deviation from vertical (deg)
    min_knee_angle: float = 150.0     # legs straight (180 = fully straight)
    min_hip_angle: float = 145.0      # not bent forward at the waist
    max_neck_flex: float = 80.0       # only an extreme forward head is rejected

    # -- Module 6: camera distance ----------------------------------------- #
    # body_fill = normalized vertical body extent (head-top to foot) in [0, 1].
    min_body_fill: float = 0.45       # below -> too far
    max_body_fill: float = 0.95       # above -> too close

    # -- Module 7: frame boundary ------------------------------------------ #
    frame_margin: float = 0.02        # normalized safe margin at every edge

    # -- Module 8: temporal smoothing -------------------------------------- #
    # One Euro Filter parameters for continuous angle signals.
    oe_min_cutoff: float = 1.0        # baseline cutoff (Hz) -> jitter at rest
    oe_beta: float = 0.02             # speed coefficient -> lag when moving
    oe_dcutoff: float = 1.0           # derivative cutoff (Hz)
    fps_assumed: float = 30.0         # used when frame dt is unavailable
    vote_window: int = 7              # sliding-window majority for the label
    stable_frames: int = 12           # consecutive valid frames -> READY latch

    # -- Rendering --------------------------------------------------------- #
    draw_landmark_indices: bool = False