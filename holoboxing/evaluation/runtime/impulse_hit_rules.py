"""Causal 20ms integral of measured legal-hand normal forces (N·s)."""

from collections import deque
import math
import numpy as np
from .hand_hit_rules import HandHitGate, hand_index, target_part

WINDOW_SECONDS = 0.02
THRESHOLD_NS = 5.0
COOLDOWN_SECONDS = 0.2
PHYSICS_DT = 0.005


class RollingNormalImpulse:
    def __init__(self):
        self.samples = [[deque() for _ in range(2)] for _ in range(2)]
        self.last_time = None

    def consume(self, hits):
        """A scored hand starts a fresh impulse window after this sample."""
        for actor, hand in np.argwhere(hits):
            self.samples[actor][hand].clear()

    def update(self, now, contacts):
        now = float(now)
        if (
            not math.isfinite(now)
            or self.last_time is not None
            and now <= self.last_time + 1e-12
        ):
            raise ValueError(
                "Normal impulse samples must advance once per physics step"
            )
        self.last_time = now
        increments = np.zeros((2, 2), dtype=np.float64)
        seen = set()
        for index, item in enumerate(contacts):
            actor, target = int(item["attacker"]), int(item["defender"])
            hand = hand_index(str(item["attacking_body"]))
            if (
                actor not in (0, 1)
                or target != 1 - actor
                or hand is None
                or target_part(str(item["target_body"])) is None
                or not item.get("physical_contact", True)
            ):
                continue
            key = (actor, hand, item.get("pair_id", index))
            if key in seen:
                continue
            seen.add(key)
            # GPU records sum(max(contact_force[normal],0)) for ALL points in
            # this semantic slot. Never integrate peak magnitude or friction.
            if "normal_impulse_step_ns" in item:
                value = float(item["normal_impulse_step_ns"])
            elif "normal_force_n" in item:
                value = float(item["normal_force_n"]) * float(
                    item.get("sample_dt_s", PHYSICS_DT)
                )
            else:
                raise ValueError(
                    "Impulse scoring requires measured normal force or step impulse"
                )
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid normal impulse")
            increments[actor, hand] += value
        totals = np.zeros((2, 2), dtype=np.float64)
        for actor in range(2):
            for hand in range(2):
                queue = self.samples[actor][hand]
                while queue and queue[0][0] <= now - WINDOW_SECONDS + 1e-9:
                    queue.popleft()
                if increments[actor, hand]:
                    queue.append((now, float(increments[actor, hand])))
                totals[actor, hand] = math.fsum(value for _, value in queue)
        return totals, increments


class ImpulseHandHitGate(HandHitGate):
    def __init__(self):
        super().__init__(1.0, 0.0)
        self.window = np.zeros((2, 2))
        self.last_hit = np.full((2, 2), -np.inf)

    def reset(self, indices=None):
        super().reset(indices)
        if indices is None:
            self.last_hit.fill(-np.inf)
            self.window.fill(0)
        else:
            self.last_hit[indices] = -np.inf

    def step(self, now, physical, forces, speeds):
        now = float(now)
        if not math.isfinite(now) or (
            self.last_time is not None and now < self.last_time
        ):
            raise ValueError("Physics time must be finite and monotonic")
        if (
            physical.shape != forces.shape
            or physical.shape != speeds.shape
            or physical.shape[:2] != (2, 2)
        ):
            raise ValueError(
                "Expected two fighters and two hands with contact slots"
            )
        self.last_time = now
        eligible = self.window[..., None] >= THRESHOLD_NS - 1e-9
        eligible = physical & eligible
        hits = eligible.any(axis=-1) & (
            now - self.last_hit >= COOLDOWN_SECONDS - 1e-9
        )
        self.last_hit[hits] = now
        return hits, eligible
