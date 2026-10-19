"""Shared NumPy/Torch hand scoring state for training, evaluation and live UI.

Inputs retain a target/contact axis: force and speed must pass on the SAME
physical contact. Call at every physics substep, including empty substeps.
Debounce uses all physical contacts with legal targets, regardless of thresholds.
"""

import math
import os

SCORING_VERSION = (
    "hands_only_normal_impulse100ms_10Ns_cooldown100ms_fall030_ko1000_v5"
)
FORCE_LEVELS = (1.0, 4.0, 16.0, 64.0, 128.0)
SPEED_LEVELS = (0.1, 0.4, 1.6)
REARM_SECONDS = 0.1


def scoring_config():
    """Official impulse rules; legacy force/speed gate settings are separate."""
    return dict(
        round_duration_s=60,
        ko_bonus=1000,
        normal_impulse_threshold_ns=10,
        window_ms=100,
        per_hand_cooldown_ms=100,
        consume_scored_impulse=True,
        fall_height_m=0.30,
        fall_bodies=["pelvis", "torso_link"],
        ko_confirmation_s=3,
        force_threshold_enabled=False,
        speed_threshold_enabled=False,
    )


def configured_thresholds():
    return validate_thresholds(
        float(os.getenv("BOXING_MIN_CONTACT_FORCE_N", "1")),
        float(os.getenv("BOXING_MIN_CLOSING_SPEED_MPS", "0.1")),
    )


def validate_thresholds(force, speed):
    force, speed = float(force), float(speed)
    if not math.isfinite(force) or force <= 0:
        raise ValueError("contact force threshold must be finite and positive")
    if not math.isfinite(speed) or speed < 0:
        raise ValueError(
            "closing speed threshold must be finite and nonnegative"
        )
    return force, speed


def hand_index(name):
    name = name.lower()
    if not any(part in name for part in ("hand", "wrist", "glove")):
        return None
    if "left" in name:
        return 0
    if "right" in name:
        return 1
    return None


def target_part(name):
    name = name.lower()
    if "head" in name:
        return "head"
    if any(
        part in name
        for part in ("torso", "chest", "abdomen", "stomach", "pelvis")
    ):
        return "torso"
    return None


class HandHitGate:
    """Backend-independent, batched gate, shape (..., fighter, hand, contact).

    A hand starts armed. Scoring disarms it until ALL legal target contacts have
    been absent continuously for 100ms. Threshold changes never clear this state.
    Physical samples must be chronological; never feed threshold-filtered input.
    """

    def __init__(self, force=None, speed=None):
        defaults = configured_thresholds()
        self.set_thresholds(
            defaults[0] if force is None else force,
            defaults[1] if speed is None else speed,
        )
        self.last_contact = self.armed = None
        self.no_contact_since = self.was_touching = None
        self.last_time = None

    def set_thresholds(self, force, speed):
        self.force, self.speed = validate_thresholds(force, speed)

    def reset(self, indices=None):
        if indices is None:
            self.last_contact = self.armed = None
            self.no_contact_since = self.was_touching = None
            self.last_time = None
        elif self.armed is not None:
            self.armed[indices] = True
            self.last_contact[indices] = -float("inf")
            self.no_contact_since[indices] = -float("inf")
            self.was_touching[indices] = False

    def step(self, now, physical, forces, speeds, *, eligibility=None):
        now = float(now)
        if not math.isfinite(now) or (
            self.last_time is not None and now < self.last_time - 1e-9
        ):
            raise ValueError("physics time must be finite and monotonic")
        self.last_time = now
        if (
            physical.shape != forces.shape
            or physical.shape != speeds.shape
            or physical.shape[-2] != 2
        ):
            raise ValueError(
                "expected matching (..., fighter, two hands, contacts) arrays"
            )
        is_torch = type(forces).__module__.startswith("torch")
        if is_torch:
            import torch as xp

            any_axis = lambda a: a.any(dim=-1)
            if self.armed is None:
                self.armed = xp.ones(
                    physical.shape[:-1], dtype=xp.bool, device=forces.device
                )
                self.last_contact = xp.full(
                    physical.shape[:-1],
                    -float("inf"),
                    dtype=xp.float64,
                    device=forces.device,
                )
                self.no_contact_since = self.last_contact.clone()
                self.was_touching = xp.zeros_like(self.armed)
        else:
            import numpy as xp

            any_axis = lambda a: a.any(axis=-1)
            if self.armed is None:
                self.armed = xp.ones(physical.shape[:-1], dtype=bool)
                self.last_contact = xp.full(
                    physical.shape[:-1], -float("inf"), dtype=xp.float64
                )
                self.no_contact_since = self.last_contact.copy()
                self.was_touching = xp.zeros_like(self.armed)
        touching = any_axis(physical)
        # Start at the first observed absent substep. Never infer separation from
        # missing/threshold-filtered samples or from a gap between timestamps.
        self.no_contact_since[~touching & self.was_touching] = now
        self.armed |= ~self.was_touching & (
            now - self.no_contact_since >= REARM_SECONDS - 1e-9
        )
        self.last_contact[touching] = now
        self.no_contact_since[touching] = float("inf")
        self.was_touching = touching
        eligible = (
            physical & (forces >= self.force) & (speeds >= self.speed)
            if eligibility is None
            else physical & eligibility
        )
        hits = self.armed & any_axis(eligible)
        self.armed &= ~hits
        return hits, eligible
