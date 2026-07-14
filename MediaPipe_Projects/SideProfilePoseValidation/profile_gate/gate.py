"""Stateful orchestrator: composes every module into one decision per frame.

Pipeline order (short-circuit friendly, but all metrics are still computed for
the on-screen panel):

    1  single person           (Module 1)
    2  full-body visibility    (Module 2)
    7  frame boundary          (Module 7)
    6  camera distance         (Module 6)
    3  orientation classify    (Module 3)  <- core
    4  head pose               (Module 4)
    5  posture / alignment     (Module 5)
    8  temporal smoothing      (Module 8)  <- One Euro + vote + streak
    9  feedback                (Module 9)

The gate owns all temporal state, so callers just feed frames and read results.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import feedback as fb
from . import validators as val
from .config import ProfileGateConfig, Orientation
from .detector import Person
from .filters import OneEuroFilter, SlidingWindowVote, StreakLatch
from .head import HeadPose, estimate_head_pose
from .landmarks import bounding_box, head_top_y, foot_bottom_y
from .orientation import OrientationResult, classify


@dataclass
class GateResult:
    valid: bool                       # this frame passed every module
    ready: bool                       # valid for >= stable_frames consecutively
    streak: int
    label: str                        # smoothed orientation label
    facing: int
    message: str                      # single corrective instruction
    reasons: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    confidence: float = 0.0           # overall [0, 1]
    bbox: Optional[tuple] = None
    person_image: Optional[np.ndarray] = None
    head: Optional[HeadPose] = None
    n_persons: int = 0

    def report(self) -> str:
        """Render the standard is_valid / side / confidence / reason summary.

            VALID
            LEFT
            Confidence: 96%

            INVALID
            Reason:
            Front Facing
        """
        pct = round(self.confidence * 100)
        if self.valid:
            side = self.metrics.get("side") or self.label.replace("PROFILE_", "")
            return f"VALID\n{side}\nConfidence: {pct}%"
        return f"INVALID\nReason:\n{self.message}"


class ProfileGate:
    def __init__(self, cfg: ProfileGateConfig = ProfileGateConfig()):
        self.cfg = cfg
        # One Euro filters for continuous signals
        oe = lambda: OneEuroFilter(cfg.oe_min_cutoff, cfg.oe_beta, cfg.oe_dcutoff)
        self._f_yaw = oe()
        self._f_hipyaw = oe()
        self._f_conf = oe()
        # one filter per bbox edge, so the drawn box tracks landmark motion
        # without jittering on MediaPipe's frame-to-frame landmark noise
        self._f_bbox = [oe() for _ in range(4)]
        self._f_head_level = oe()
        self._f_head_spread = oe()
        # feet/head resting right at the frame edge straddle the crop margin
        # every other frame from ordinary jitter - smooth before gating.
        self._f_head_top = oe()
        self._f_foot_bottom = oe()
        self._vote = SlidingWindowVote(cfg.vote_window)
        self._latch = StreakLatch(cfg.stable_frames)
        self._last_t: Optional[float] = None

    def reset(self) -> None:
        for f in (self._f_yaw, self._f_hipyaw, self._f_conf,
                 self._f_head_level, self._f_head_spread,
                 self._f_head_top, self._f_foot_bottom, *self._f_bbox):
            f.reset()
        self._vote.reset()
        self._latch.reset()
        self._last_t = None

    def _dt(self, now: float) -> float:
        if self._last_t is None:
            self._last_t = now
            return 1.0 / self.cfg.fps_assumed
        dt = now - self._last_t
        self._last_t = now
        return dt if dt > 0 else 1.0 / self.cfg.fps_assumed

    def evaluate(self, persons: List[Person], width: int, height: int,
                 now: Optional[float] = None) -> GateResult:
        cfg = self.cfg
        now = now if now is not None else time.time()
        dt = self._dt(now)
        reasons: Dict[str, dict] = {}

        # ---- Module 1: single person ---------------------------------- #
        c1 = val.validate_single_person(len(persons), cfg)
        if not c1.ok:
            self._latch(False)
            orient = OrientationResult(label=Orientation.UNKNOWN)
            msg = fb.build_message({c1.reason: c1.detail}, orient, cfg)
            return GateResult(False, False, self._latch.streak,
                              Orientation.UNKNOWN, 0, msg, [c1.reason],
                              n_persons=len(persons))

        image, world = persons[0]

        # ---- Module 3: orientation (needed by posture/head/feedback) -- #
        orient = classify(image, world, cfg)

        # ---- Module 8: temporal smoothing (BEFORE gating, not just for
        # display) ------------------------------------------------------ #
        # MediaPipe landmarks jitter every frame. Gating on raw per-frame
        # values means a single noisy frame can flip the discrete label at a
        # boundary (e.g. PROFILE_RIGHT <-> OBLIQUE from twist/overlap noise)
        # or dip confidence a hair under threshold - both reset the READY
        # streak and read as a flickering/never-green box even while the
        # subject holds a genuinely valid pose. Smoothing these signals FIRST
        # and feeding the smoothed values into the checks below (not just
        # into the on-screen metrics) is what actually stabilizes the gate.
        sm_yaw = self._f_yaw(orient.yaw, dt)
        sm_hipyaw = self._f_hipyaw(orient.hip_yaw, dt)
        sm_conf = self._f_conf(orient.confidence, dt)
        smoothed_label = self._vote(orient.label)
        sm_head_top = self._f_head_top(head_top_y(image, cfg.level_visibility), dt)
        sm_foot_bottom = self._f_foot_bottom(foot_bottom_y(image, cfg.foot_visibility), dt)

        # ---- run remaining validators (collect ALL failures) ---------- #
        checks = [
            val.validate_full_body(image, cfg),        # Module 2
            val.validate_frame_boundary(image, cfg,    # Module 7
                                       head_top=sm_head_top, foot_bottom=sm_foot_bottom),
            val.validate_distance(image, cfg),         # Module 6
            val.validate_orientation(orient, cfg,      # Module 3 gate
                                     label=smoothed_label, confidence=sm_conf),
        ]
        head = estimate_head_pose(image, width, height, cfg.head_min_visibility)
        sm_head_level = (self._f_head_level(head.level, dt)
                         if head.level is not None else None)
        sm_head_spread = (self._f_head_spread(head.spread, dt)
                          if head.spread is not None else None)
        checks.append(val.validate_head(head, orient, cfg,        # Module 4
                                        level=sm_head_level, spread=sm_head_spread))
        checks.append(val.validate_posture(world, orient, cfg))   # Module 5

        for c in checks:
            if not c.ok:
                reasons[c.reason] = c.detail

        frame_valid = len(reasons) == 0
        ready = self._latch(frame_valid)

        # ---- Module 9: feedback --------------------------------------- #
        message = fb.build_message(reasons, orient, cfg)

        # ---- overall confidence --------------------------------------- #
        # geometric-cleanliness * pass-fraction, so a wrong pose can't score high
        n_checks = len(checks) + 1  # + single-person
        n_pass = n_checks - len(reasons)
        overall = round(sm_conf * (n_pass / n_checks), 3)

        # smoothed bounding box: tracks the body without per-frame jitter
        raw_bbox = bounding_box(image, width, height)
        smoothed_bbox = tuple(
            int(f(v, dt)) for f, v in zip(self._f_bbox, raw_bbox)
        )

        metrics = {
            "yaw": round(sm_yaw, 1),
            "hip_yaw": round(sm_hipyaw, 1),
            "signed_yaw": orient.signed_yaw,
            "orient_conf": round(sm_conf, 2),
            "side": smoothed_label.replace("PROFILE_", "") if smoothed_label in Orientation.VALID_PROFILES else None,
            "twist": orient.twist,
            "overlap_score": orient.overlap_score,
            "visibility_score": orient.visibility_score,
            "side_agreement": orient.side_agreement,
            "cues": orient.cues,
            "head_level": None if sm_head_level is None else round(sm_head_level, 1),
            "head_facing": head.facing,
            "head_spread": None if sm_head_spread is None else round(sm_head_spread, 2),
            **{k: v for c in checks for k, v in c.detail.items()},
        }

        return GateResult(
            valid=frame_valid, ready=ready, streak=self._latch.streak,
            label=smoothed_label, facing=orient.facing, message=message,
            reasons=list(reasons.keys()), metrics=metrics, confidence=overall,
            bbox=smoothed_bbox, person_image=image,
            head=head, n_persons=len(persons),
        )