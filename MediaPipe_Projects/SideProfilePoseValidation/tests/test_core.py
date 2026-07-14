"""Unit tests for the framework-independent core (no camera / MediaPipe needed).

Run with:  pytest -q
These pass on synthetic arrays, demonstrating that geometry.py, filters.py and
orientation.py are testable in isolation from MediaPipe.
"""
import math

import numpy as np

from profile_gate import geometry as geo
from profile_gate.filters import OneEuroFilter, SlidingWindowVote, StreakLatch
from profile_gate.orientation import classify
from profile_gate import validators as val
from profile_gate.head import HeadPose
from profile_gate.config import (
    ProfileGateConfig, Orientation,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE, NOSE,
)


def _world_with_shoulder_yaw(yaw_deg, facing=1):
    w = np.zeros((33, 3))
    r = math.radians(yaw_deg)
    half = 0.2
    dx = math.cos(r) * half
    dz = math.sin(r) * half * facing
    w[LEFT_SHOULDER] = [-dx, 0.0, -dz]
    w[RIGHT_SHOULDER] = [dx, 0.0, dz]
    w[LEFT_HIP] = [-dx, 0.5, -dz]
    w[RIGHT_HIP] = [dx, 0.5, dz]
    return w


# --------------------------- geometry -------------------------------------- #
def test_frontal_yaw_is_zero():
    w = _world_with_shoulder_yaw(0)
    assert geo.shoulder_yaw_magnitude(w) < 2.0


def test_side_yaw_is_ninety():
    w = _world_with_shoulder_yaw(90)
    assert geo.shoulder_yaw_magnitude(w) > 85.0


def test_straight_knee_is_180():
    hip = np.array([0.0, 0.0, 0.0])
    knee = np.array([0.0, 1.0, 0.0])
    ankle = np.array([0.0, 2.0, 0.0])
    assert abs(geo.joint_angle(hip, knee, ankle) - 180.0) < 1e-6


def test_right_angle_joint():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(geo.joint_angle(a, b, c) - 90.0) < 1e-6


def test_torso_twist_zero_when_aligned():
    assert geo.torso_twist(88.0, 88.0) < 1e-6


def test_torso_twist_folds_wraparound():
    # 179 vs -179 deg are 2 deg apart, not ~358 deg apart
    assert geo.torso_twist(179.0, -179.0) < 3.0


def test_hip_depth_sign_matches_shoulder_for_pure_yaw():
    w = _world_with_shoulder_yaw(80, facing=1)
    assert geo.shoulder_depth_sign(w) == geo.hip_depth_sign(w)


# --------------------------- filters --------------------------------------- #
def test_one_euro_tracks_constant():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    for _ in range(50):
        y = f(10.0, 1 / 30)
    assert abs(y - 10.0) < 1e-3


def test_sliding_vote_majority():
    v = SlidingWindowVote(window=5)
    for lab in ["A", "A", "B", "A", "A"]:
        out = v(lab)
    assert out == "A"


def test_streak_latch():
    latch = StreakLatch(stable_frames=3)
    assert latch(True) is False
    assert latch(True) is False
    assert latch(True) is True
    assert latch(False) is False


# --------------------------- orientation: synthetic frame builder ---------- #
# A single, physically-consistent frame builder used across every orientation
# test below. "Physically consistent" means the image-space cues (nose x vs
# ear-mid x) and the world-space cues (which shoulder/hip is nearer camera)
# agree with each other for a given `facing`, exactly as they would for a real
# captured frame of a person actually turning that way. facing=+1 -> the
# subject's RIGHT shoulder is nearer the camera (-> PROFILE_RIGHT). facing=-1
# -> LEFT shoulder nearer (-> PROFILE_LEFT). This mirrors config.py's note that
# LEFT/RIGHT always refer to the SUBJECT's own left/right.
def _profile_frame(shoulder_yaw_deg=88.0, hip_yaw_deg=None, facing=1,
                   shoulder_overlap=0.02, hip_overlap=0.02, vis_gap=0.85,
                   flip_cue=None):
    """Build (image, world) landmark arrays for a synthetic side-profile pose.

    flip_cue: optionally invert one side-vote cue ("image", "shoulder_z",
    "hip_z", "image_z") to simulate a single noisy/occluded landmark, without
    touching the others - used to prove the fused vote survives one bad cue.
    """
    if hip_yaw_deg is None:
        hip_yaw_deg = shoulder_yaw_deg

    world = np.zeros((33, 3))
    half = 0.2

    def place(l_idx, r_idx, yaw_deg, y):
        r = math.radians(yaw_deg)
        dx = math.cos(r) * half
        dz = math.sin(r) * half * facing
        world[l_idx] = [-dx, y, dz]
        world[r_idx] = [dx, y, -dz]

    place(LEFT_SHOULDER, RIGHT_SHOULDER, shoulder_yaw_deg, 0.0)
    place(LEFT_HIP, RIGHT_HIP, hip_yaw_deg, 0.5)

    image = np.zeros((33, 4))
    image[:, 3] = 0.9  # baseline visibility for every landmark

    cx, cy = 0.5, 0.5
    nose_swing = math.sin(math.radians(shoulder_yaw_deg)) * 0.08 * facing
    image[NOSE] = [cx + nose_swing, cy - 0.30, -0.05, 0.9]
    image[LEFT_EAR] = [cx, cy - 0.30, 0.0, 0.9]
    image[RIGHT_EAR] = [cx, cy - 0.30, 0.0, 0.9]

    image[LEFT_SHOULDER] = [cx - shoulder_overlap / 2, cy - 0.10, 0.0, 0.9]
    image[RIGHT_SHOULDER] = [cx + shoulder_overlap / 2, cy - 0.10, 0.0, 0.9]
    image[LEFT_HIP] = [cx - hip_overlap / 2, cy + 0.20, 0.0, 0.9]
    image[RIGHT_HIP] = [cx + hip_overlap / 2, cy + 0.20, 0.0, 0.9]

    # image-space shoulder "z" cue, built the same way as the world cue
    image[LEFT_SHOULDER, 2] = 0.05 if facing > 0 else -0.05
    image[RIGHT_SHOULDER, 2] = -0.05 if facing > 0 else 0.05

    # near side (facing direction) clearly visible, far side self-occluded
    near_vis, far_vis = 0.9, max(0.9 - vis_gap, 0.02)
    for left_idx, right_idx in ((LEFT_SHOULDER, RIGHT_SHOULDER),
                               (LEFT_HIP, RIGHT_HIP),
                               (LEFT_EAR, RIGHT_EAR), (LEFT_EYE, RIGHT_EYE)):
        if facing > 0:
            image[left_idx, 3], image[right_idx, 3] = far_vis, near_vis
        else:
            image[left_idx, 3], image[right_idx, 3] = near_vis, far_vis

    if flip_cue == "image":
        image[NOSE, 0] = cx - nose_swing
    elif flip_cue == "shoulder_z":
        world[LEFT_SHOULDER, 2], world[RIGHT_SHOULDER, 2] = (
            world[RIGHT_SHOULDER, 2], world[LEFT_SHOULDER, 2])
    elif flip_cue == "hip_z":
        world[LEFT_HIP, 2], world[RIGHT_HIP, 2] = (
            world[RIGHT_HIP, 2], world[LEFT_HIP, 2])
    elif flip_cue == "image_z":
        image[LEFT_SHOULDER, 2], image[RIGHT_SHOULDER, 2] = (
            image[RIGHT_SHOULDER, 2], image[LEFT_SHOULDER, 2])

    return image, world


# --------------------------- orientation: LEFT/RIGHT symmetry -------------- #
def test_classify_right_profile_valid():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(facing=1)
    o = classify(image, world, cfg)
    assert o.label == Orientation.PROFILE_RIGHT
    assert o.side == "RIGHT"
    assert o.facing == 1
    assert o.confidence >= cfg.min_profile_confidence


def test_classify_left_profile_valid():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(facing=-1)
    o = classify(image, world, cfg)
    assert o.label == Orientation.PROFILE_LEFT
    assert o.side == "LEFT"
    assert o.facing == -1
    assert o.confidence >= cfg.min_profile_confidence


def test_left_and_right_confidence_are_symmetric():
    """A mirror-image LEFT and RIGHT pose must score (almost) identically -
    no hidden LEFT-only bias in the fused confidence."""
    cfg = ProfileGateConfig()
    right = classify(*_profile_frame(facing=1), cfg)
    left = classify(*_profile_frame(facing=-1), cfg)
    assert abs(right.confidence - left.confidence) < 1e-6


def test_fusion_survives_one_noisy_cue_right():
    """Regression test for the original bug: the old classifier required
    UNANIMOUS agreement between only 2 world-depth votes, so any single noisy
    cue silently downgraded a true profile to OBLIQUE. Here the low-weight,
    independent image-space depth cue is flipped (simulating the kind of
    noise MediaPipe's coarse image-z estimate is prone to); the three
    remaining cues should still out-vote it and resolve correctly. Note this
    must be a cue independent of world yaw (not shoulder_z/hip_z) - flipping
    either of those changes the actual shoulder/hip yaw sign and legitimately
    becomes a twisted-torso case, covered separately below."""
    cfg = ProfileGateConfig()
    image, world = _profile_frame(facing=1, flip_cue="image_z")
    o = classify(image, world, cfg)
    assert o.label == Orientation.PROFILE_RIGHT
    assert o.side == "RIGHT"


def test_fusion_survives_one_noisy_cue_left():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(facing=-1, flip_cue="image_z")
    o = classify(image, world, cfg)
    assert o.label == Orientation.PROFILE_LEFT
    assert o.side == "LEFT"


def test_flipped_world_depth_cue_reads_as_twisted_not_silently_ambiguous():
    """A flipped WORLD depth cue (shoulder_z or hip_z) is not just vote noise -
    it means that landmark pair's yaw sign genuinely disagrees with the other,
    which is a real torso twist. The classifier should name it as such
    (informative rejection) rather than silently fall back to 'ambiguous'."""
    cfg = ProfileGateConfig()
    image, world = _profile_frame(facing=1, flip_cue="hip_z")
    o = classify(image, world, cfg)
    assert o.label == Orientation.OBLIQUE
    assert o.reason == "twisted_torso"


def test_classify_ambiguous_when_cues_genuinely_conflict():
    """If the image cue disagrees with BOTH world cues (not just one noisy
    landmark), the vote should stay ambiguous rather than guess."""
    cfg = ProfileGateConfig()
    image, world = _profile_frame(facing=1, flip_cue="image")
    o = classify(image, world, cfg)
    assert o.side is None
    assert o.label == Orientation.OBLIQUE
    assert o.reason == "ambiguous_facing"


# --------------------------- orientation: rejections ------------------------ #
def test_classify_frontal_rejected():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(shoulder_yaw_deg=2, hip_yaw_deg=2)
    image[NOSE, 2] = -0.05  # nose nearer than ears -> FRONT
    o = classify(image, world, cfg)
    assert o.label == Orientation.FRONT
    assert o.label not in Orientation.VALID_PROFILES


def test_classify_back_rejected():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(shoulder_yaw_deg=2, hip_yaw_deg=2)
    image[NOSE, 2] = 0.05  # nose farther than ears -> BACK
    o = classify(image, world, cfg)
    assert o.label == Orientation.BACK


def test_classify_45_degrees_is_three_quarter_oblique():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(shoulder_yaw_deg=45, hip_yaw_deg=45)
    o = classify(image, world, cfg)
    assert o.label == Orientation.OBLIQUE
    assert o.reason == "three_quarter"


def test_classify_slight_rotation_near_frontal():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(shoulder_yaw_deg=32, hip_yaw_deg=32)
    o = classify(image, world, cfg)
    assert o.label == Orientation.OBLIQUE
    assert o.reason == "slight_rotation"


def test_classify_twisted_torso_rejected():
    """Shoulders turned to a full profile but hips left mostly frontal -
    a twisted torso, not a clean side-on turn."""
    cfg = ProfileGateConfig()
    image, world = _profile_frame(shoulder_yaw_deg=88, hip_yaw_deg=40, facing=1)
    o = classify(image, world, cfg)
    assert o.label == Orientation.OBLIQUE
    assert o.reason == "twisted_torso"


def test_classify_half_side_rejected_on_overlap():
    """Yaw and twist both look fine, but the shoulders/hips have not actually
    collapsed together in the image - i.e. both sides still visible."""
    cfg = ProfileGateConfig()
    image, world = _profile_frame(shoulder_overlap=0.5, hip_overlap=0.5)
    o = classify(image, world, cfg)
    assert o.label == Orientation.OBLIQUE
    assert o.reason == "not_full_side"


def test_classify_profile_accepted():
    cfg = ProfileGateConfig()
    image, world = _profile_frame(shoulder_yaw_deg=88, facing=1)
    o = classify(image, world, cfg)
    assert o.label in Orientation.VALID_PROFILES
    assert o.yaw > 80.0


# --------------------------- posture: bent body / sitting ------------------ #
def _standing_world(knee_angle_deg=180.0, hip_angle_deg=180.0):
    """Side-profile leg landmarks with the torso kept exactly vertical (shoulder
    directly above hip, tilt=0), so knee_angle_deg / hip_angle_deg isolate
    exactly the joint under test instead of also perturbing torso_tilt.

    hip_bend rotates the thigh (hip->knee) away from the straight-down
    continuation of the torso; knee_bend then rotates the shin (knee->ankle)
    away from the straight continuation of the thigh. Both angles match
    geo.hip_angle / geo.knee_angle's own vertex convention exactly.
    """
    w = np.zeros((33, 3))
    shoulder = np.array([0.0, 0.0, 0.0])
    hip = np.array([0.0, 0.5, 0.0])
    w[LEFT_SHOULDER] = w[RIGHT_SHOULDER] = shoulder
    w[LEFT_HIP] = w[RIGHT_HIP] = hip

    hip_bend = math.radians(180.0 - hip_angle_deg)
    thigh_dir = np.array([math.sin(hip_bend), math.cos(hip_bend), 0.0])
    knee = hip + 0.5 * thigh_dir
    w[LEFT_KNEE] = w[RIGHT_KNEE] = knee

    knee_bend = math.radians(180.0 - knee_angle_deg)
    cos_b, sin_b = math.cos(knee_bend), math.sin(knee_bend)
    shin_dir = np.array([
        thigh_dir[0] * cos_b + thigh_dir[1] * sin_b,
        -thigh_dir[0] * sin_b + thigh_dir[1] * cos_b,
        0.0,
    ])
    w[LEFT_ANKLE] = w[RIGHT_ANKLE] = knee + 0.5 * shin_dir
    return w


def test_straight_standing_posture_valid():
    cfg = ProfileGateConfig()
    from profile_gate.orientation import OrientationResult
    orient = OrientationResult(label=Orientation.PROFILE_RIGHT, facing=1)
    world = _standing_world(knee_angle_deg=180.0, hip_angle_deg=180.0)
    c = val.validate_posture(world, orient, cfg)
    assert c.ok


def test_walking_stride_knee_bend_within_tolerance():
    """A natural walking stride bends the knee somewhat; it should stay valid
    as long as it doesn't cross the 'bent knee' threshold."""
    cfg = ProfileGateConfig()
    from profile_gate.orientation import OrientationResult
    orient = OrientationResult(label=Orientation.PROFILE_RIGHT, facing=1)
    world = _standing_world(knee_angle_deg=cfg.min_knee_angle + 5)
    c = val.validate_posture(world, orient, cfg)
    assert c.ok


def test_sitting_sideways_bent_knee_invalid():
    cfg = ProfileGateConfig()
    from profile_gate.orientation import OrientationResult
    orient = OrientationResult(label=Orientation.PROFILE_RIGHT, facing=1)
    world = _standing_world(knee_angle_deg=90.0)
    c = val.validate_posture(world, orient, cfg)
    assert not c.ok
    assert c.reason == "bent_knee"


def test_sitting_sideways_bent_hip_invalid():
    cfg = ProfileGateConfig()
    from profile_gate.orientation import OrientationResult
    orient = OrientationResult(label=Orientation.PROFILE_RIGHT, facing=1)
    world = _standing_world(hip_angle_deg=100.0)
    c = val.validate_posture(world, orient, cfg)
    assert not c.ok
    assert c.reason == "bent_hip"


# --------------------------- head pose --------------------------------------#
def test_head_aligned_with_body_valid():
    cfg = ProfileGateConfig()
    from profile_gate.orientation import OrientationResult
    orient = OrientationResult(label=Orientation.PROFILE_RIGHT, facing=1)
    head = HeadPose(yaw=None, pitch=None, roll=None, facing=1,
                    level=10.0, spread=0.05, ok=True)
    c = val.validate_head(head, orient, cfg)
    assert c.ok


def test_head_turned_toward_camera_invalid():
    """Head yaw spread blows up once the head rotates toward the camera even
    if the body stays side-on - this is exactly 'body sideways, head turned
    toward camera', which must be rejected."""
    cfg = ProfileGateConfig()
    from profile_gate.orientation import OrientationResult
    orient = OrientationResult(label=Orientation.PROFILE_RIGHT, facing=1)
    head = HeadPose(yaw=None, pitch=None, roll=None, facing=1,
                    level=10.0, spread=0.5, ok=True)
    c = val.validate_head(head, orient, cfg)
    assert not c.ok
    assert c.reason == "head_roll"


def test_head_looking_over_shoulder_invalid():
    """Body faces right but head faces left (looking back) - must be rejected
    regardless of which direction the body itself is valid for."""
    cfg = ProfileGateConfig()
    from profile_gate.orientation import OrientationResult
    orient = OrientationResult(label=Orientation.PROFILE_RIGHT, facing=1)
    head = HeadPose(yaw=None, pitch=None, roll=None, facing=-1,
                    level=10.0, spread=0.05, ok=True)
    c = val.validate_head(head, orient, cfg)
    assert not c.ok
    assert c.reason == "head_misaligned"
