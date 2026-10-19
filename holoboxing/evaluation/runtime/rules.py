"""HoloMotion 2.0 two-fighter match rules and JSON logging."""

from __future__ import annotations
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from .hand_hit_rules import (
    HandHitGate,
    configured_thresholds,
    hand_index,
    target_part,
)
from .impulse_hit_rules import (
    ImpulseHandHitGate,
    RollingNormalImpulse,
    THRESHOLD_NS,
)

NUM_FIGHTERS = 2
DOF_COUNT = 29


def _vector(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class MatchRules:
    num_rounds: int = 1
    round_duration_s: float = 60.0
    ko_confirmation_s: float = 3.0
    minimum_closing_speed_mps: float = configured_thresholds()[1]
    minimum_contact_force_n: float = configured_thresholds()[0]

    def __post_init__(self) -> None:
        from .hand_hit_rules import validate_thresholds

        validate_thresholds(
            self.minimum_contact_force_n, self.minimum_closing_speed_mps
        )
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        for name in (
            "round_duration_s",
            "ko_confirmation_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.minimum_closing_speed_mps) or (
            self.minimum_closing_speed_mps < 0.0
        ):
            raise ValueError("minimum_closing_speed_mps must be non-negative")


@dataclass(frozen=True)
class MatchFrameResult:
    score: tuple[int, int]
    events: tuple[Mapping[str, Any], ...]
    round_number: int
    round_ended: bool
    match_over: bool
    winner: int | None


class MatchStateMachine:
    """Simulator-independent single-round boxing referee with KO +1000."""

    _TORSO_TARGETS = frozenset(
        {
            "torso",
            "chest",
            "abdomen",
            "stomach",
            "pelvis",
        }
    )

    def __init__(
        self, rules: MatchRules | None = None, *, scoring_mode="impulse100ms"
    ) -> None:
        if scoring_mode not in ("impulse100ms", "force_speed"):
            raise ValueError("Unknown scoring mode")
        self.scoring_mode = scoring_mode
        self.rules = rules or MatchRules()
        self.reset(0.0)

    def reset(self, sim_time_s: float = 0.0) -> None:
        self.score = np.zeros(NUM_FIGHTERS, dtype=np.int64)
        self.current_round = 1
        self.round_start_s = float(sim_time_s)
        self.match_over = False
        self.winner: int | None = None
        self.reset_motion_state()
        self._last_sim_time_s = float(sim_time_s)

    def reset_motion_state(self) -> None:
        """Clear teleport-sensitive state, preserving scores and match clocks."""
        self._previous_contacts: set[tuple[int, int, str, str]] = set()
        self._fallen = np.zeros(NUM_FIGHTERS, dtype=np.bool_)
        self._fall_started_s: list[float | None] = [None, None]
        self._ko_latched = np.zeros(NUM_FIGHTERS, dtype=np.bool_)
        self.hand_gate = (
            ImpulseHandHitGate()
            if self.scoring_mode == "impulse100ms"
            else HandHitGate(
                self.rules.minimum_contact_force_n,
                self.rules.minimum_closing_speed_mps,
            )
        )
        self.impulse_clock = RollingNormalImpulse()
        self.contact_diagnostics: list[dict[str, Any]] = []

    def set_thresholds(self, force_n: float, speed_mps: float) -> None:
        if self.scoring_mode == "force_speed":
            self.hand_gate.set_thresholds(force_n, speed_mps)

    def _score_physics_sample(
        self, now: float, contacts: Iterable[Any]
    ) -> list[dict[str, Any]]:
        contacts = list(contacts)
        impulses = None
        if self.scoring_mode == "impulse100ms":
            impulses, _ = self.impulse_clock.update(now, contacts)
            self.hand_gate.window = impulses
        count = max(1, len(contacts))
        physical = np.zeros((2, 2, count), dtype=bool)
        forces = np.zeros((2, 2, count))
        speeds = np.zeros_like(forces)
        records = {}
        for index, contact in enumerate(contacts):
            item = (
                dict(contact)
                if isinstance(contact, Mapping)
                else vars(contact)
            )
            actor, target = int(item["attacker"]), int(item["defender"])
            hand = hand_index(str(item["attacking_body"]))
            part = target_part(str(item["target_body"]))
            if (
                actor in (0, 1)
                and target == 1 - actor
                and part is not None
                and hand is None
                and any(
                    name in str(item["attacking_body"]).lower()
                    for name in ("foot", "ankle")
                )
                and item.get("physical_contact", True)
            ):
                self.contact_diagnostics.append(
                    dict(
                        actor=actor,
                        target=target,
                        hand="foot",
                        body_part=part,
                        target_body=item["target_body"],
                        attacking_body=item["attacking_body"],
                        peak_force=float(item.get("peak_force", 0)),
                        closing_speed=float(item["closing_speed"]),
                        position=np.asarray(
                            item["position"], dtype=float
                        ).tolist(),
                        physics_time_s=now,
                        force_threshold_n=self.hand_gate.force,
                        speed_threshold_mps=self.hand_gate.speed,
                        valid=False,
                        reason="foot_not_scored",
                    )
                )
            if (
                actor not in (0, 1)
                or target != 1 - actor
                or hand is None
                or part is None
            ):
                continue
            if "peak_force" not in item:
                raise ValueError(
                    "hand contacts require raw peak_force; threshold-filtered samples are insufficient"
                )
            physical[actor, hand, index] = bool(
                item.get("physical_contact", True)
            )
            forces[actor, hand, index] = float(item["peak_force"])
            speeds[actor, hand, index] = float(item["closing_speed"])
            records[actor, hand, index] = (item, part)
        hits, eligible = self.hand_gate.step(now, physical, forces, speeds)
        if impulses is not None:
            self.impulse_clock.consume(hits)
        events = []
        for (actor, hand, index), (item, part) in records.items():
            if not physical[actor, hand, index]:
                continue
            chosen = int(
                np.argmax(
                    np.where(eligible[actor, hand], forces[actor, hand], -1)
                )
            )
            valid = bool(hits[actor, hand] and index == chosen)
            reason = (
                "effective_hit"
                if valid
                else "impulse_below_threshold"
                if impulses is not None
                and impulses[actor, hand] < THRESHOLD_NS - 1e-9
                else "hand_cooldown"
                if impulses is not None
                else "force_below_threshold"
                if forces[actor, hand, index] < self.hand_gate.force
                else "speed_below_threshold"
                if speeds[actor, hand, index] < self.hand_gate.speed
                else "hand_not_rearmed"
            )
            detail = dict(
                actor=actor,
                target=1 - actor,
                hand=["left", "right"][hand],
                body_part=part,
                target_body=item["target_body"],
                attacking_body=item["attacking_body"],
                peak_force=float(forces[actor, hand, index]),
                closing_speed=float(speeds[actor, hand, index]),
                position=np.asarray(item["position"], dtype=float).tolist(),
                physics_time_s=now,
                force_threshold_n=self.hand_gate.force,
                speed_threshold_mps=self.hand_gate.speed,
                valid=valid,
                reason=reason,
            )
            self.contact_diagnostics.append(detail)
            if impulses is not None:
                detail.update(
                    normal_impulse_ns=float(impulses[actor, hand]),
                    normal_impulse_step_ns=float(
                        item.get(
                            "normal_impulse_step_ns",
                            float(item.get("normal_force_n", 0)) * 0.005,
                        )
                    ),
                    normal_force_sum_n=item.get(
                        "normal_force_sum_n", item.get("normal_force_n")
                    ),
                    impulse_window_ms=100,
                    impulse_threshold_ns=THRESHOLD_NS,
                    force_threshold_n=None,
                    speed_threshold_mps=None,
                    sample_dt_s=item.get("sample_dt_s", 0.005),
                    pair_id=item.get("pair_id", index),
                )
            if valid:
                delta = [0, 0]
                delta[actor] = 1
                self.score[actor] += 1
                events.append(
                    self._event(
                        "HIT",
                        now,
                        score_delta=delta,
                        strike_part="hand",
                        **detail,
                    )
                )
        return events

    @staticmethod
    def _contact_value(contact: Any, name: str) -> Any:
        if isinstance(contact, Mapping):
            return contact[name]
        return getattr(contact, name)

    @classmethod
    def _strike_part(cls, body: str) -> str | None:
        value = body.lower()
        if any(token in value for token in ("hand", "wrist", "glove")):
            return "hand"
        return None

    @classmethod
    def _body_part(cls, body: str) -> str | None:
        value = body.lower()
        if "head" in value:
            return "head"
        if value in cls._TORSO_TARGETS or any(
            token in value for token in cls._TORSO_TARGETS
        ):
            return "torso"
        return None

    def _event(
        self,
        event_type: str,
        sim_time_s: float,
        *,
        actor: int | None = None,
        target: int | None = None,
        score_delta: Sequence[int] = (0, 0),
        **payload: Any,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": event_type,
            "round": self.current_round,
            "sim_time_s": float(sim_time_s),
            "round_elapsed_s": float(sim_time_s - self.round_start_s),
            "score_delta": [int(score_delta[0]), int(score_delta[1])],
        }
        if actor is not None:
            event["actor"] = int(actor)
        if target is not None:
            event["target"] = int(target)
        event.update(_jsonable(payload))
        return event

    def _finish_round(
        self, sim_time_s: float, events: list[dict[str, Any]], reason: str
    ) -> MatchFrameResult:
        ended_round = self.current_round
        events.append(
            self._event(
                "ROUND_END",
                sim_time_s,
                reason=reason,
                final_round_score=self.score.copy(),
            )
        )
        if ended_round >= self.rules.num_rounds:
            self.match_over = True
            ko_actors = {event["actor"] for event in events if event["type"] == "KO"}
            if len(ko_actors) == 2:
                self.winner = None
            elif ko_actors:
                self.winner = next(iter(ko_actors))
            elif self.score[0] != self.score[1]:
                self.winner = int(np.argmax(self.score))
            events.append(
                self._event(
                    "MATCH_END",
                    sim_time_s,
                    winner=self.winner,
                    final_score=self.score.copy(),
                )
            )
        else:
            self.current_round += 1
            self.round_start_s = float(sim_time_s)
            self._previous_contacts.clear()
            self._fallen.fill(False)
            self._fall_started_s = [None, None]
            self._ko_latched.fill(False)
            self.hand_gate.reset()
        return MatchFrameResult(
            score=(int(self.score[0]), int(self.score[1])),
            events=tuple(events),
            round_number=ended_round,
            round_ended=True,
            match_over=self.match_over,
            winner=self.winner,
        )

    def observe_frame(
        self,
        *,
        sim_time_s: float,
        contacts: Iterable[Any],
        root_pos: Any,
        fallen: Any,
        physics_samples: Iterable[tuple[float, Iterable[Any]]] | None = None,
    ) -> MatchFrameResult:
        if self.match_over:
            raise RuntimeError("match is already complete")
        now = float(sim_time_s)
        if not math.isfinite(now) or now < self._last_sim_time_s:
            raise ValueError("sim_time_s must be finite and monotonic")
        _vector(root_pos, name="root_pos", shape=(2, 3))
        fallen_now = np.asarray(fallen, dtype=np.bool_)
        if fallen_now.shape != (2,):
            raise ValueError("fallen must have shape (2,)")
        self._last_sim_time_s = now
        events: list[dict[str, Any]] = []
        self.contact_diagnostics = []

        for fighter in range(NUM_FIGHTERS):
            if fallen_now[fighter] and not self._fallen[fighter]:
                self._fall_started_s[fighter] = now
                events.append(self._event("FALL", now, target=fighter))
            elif not fallen_now[fighter]:
                self._fall_started_s[fighter] = None
                self._ko_latched[fighter] = False
            self._fallen[fighter] = fallen_now[fighter]

        round_end_reason: str | None = None
        ko_candidates = [
            fighter
            for fighter in range(NUM_FIGHTERS)
            if self._fall_started_s[fighter] is not None
            and not self._ko_latched[fighter]
            and now - self._fall_started_s[fighter] >= self.rules.ko_confirmation_s
        ]
        # A delayed observation can cross both confirmation deadlines. Only
        # identical observed fall times are simultaneous; array order is irrelevant.
        first_fall = min((self._fall_started_s[fighter] for fighter in ko_candidates), default=None)
        for fighter in ko_candidates:
            started = self._fall_started_s[fighter]
            if (
                started == first_fall
            ):
                actor = 1 - fighter
                self._ko_latched[fighter] = True
                events.append(
                    self._event("KNOCKDOWN", now, actor=actor, target=fighter)
                )
                delta = [0, 0]
                delta[actor] = 1000
                self.score[actor] += 1000
                events.append(
                    self._event(
                        "KO",
                        now,
                        actor=actor,
                        target=fighter,
                        score_delta=delta,
                    )
                )
                round_end_reason = "KO"

        if round_end_reason is None:
            samples = (
                physics_samples
                if physics_samples is not None
                else [(now, contacts)]
            )
            for sample_time, sample_contacts in samples:
                events.extend(
                    self._score_physics_sample(
                        float(sample_time), sample_contacts
                    )
                )

        if round_end_reason is not None:
            return self._finish_round(now, events, round_end_reason)
        half_frame_tolerance_s = 0.5 / 50.0
        if (
            now - self.round_start_s
            >= self.rules.round_duration_s - half_frame_tolerance_s
        ):
            return self._finish_round(now, events, "TIME")
        return MatchFrameResult(
            score=(int(self.score[0]), int(self.score[1])),
            events=tuple(events),
            round_number=self.current_round,
            round_ended=False,
            match_over=False,
            winner=None,
        )


class MatchLogger:
    """Collect frame records and derive a summary from those records."""

    def __init__(
        self,
        *,
        model_id: Sequence[str],
        match_id: str,
        rules: MatchRules | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if len(model_id) != 2:
            raise ValueError("model_id must contain two values")
        self.model_id = [str(value) for value in model_id]
        self.match_id = str(match_id)
        self.rules = rules or MatchRules()
        self.frames: list[dict[str, Any]] = []
        self.extra_metadata = dict(metadata or {})

    def record_frame(
        self,
        *,
        root_pos: Any,
        root_quat: Any,
        dof_pos: Any,
        action_raw: Any,
        policy_start_ns: Sequence[int],
        policy_end_ns: Sequence[int],
        score: Sequence[int],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        frame = {
            "root_pos": _vector(
                root_pos, name="root_pos", shape=(2, 3)
            ).tolist(),
            "root_quat": _vector(
                root_quat, name="root_quat", shape=(2, 4)
            ).tolist(),
            "dof_pos": _vector(
                dof_pos, name="dof_pos", shape=(2, 29)
            ).tolist(),
            "action_raw": _vector(
                action_raw, name="action_raw", shape=(2, 29)
            ).tolist(),
            "policy_start_ns": [int(value) for value in policy_start_ns],
            "policy_end_ns": [int(value) for value in policy_end_ns],
            "score": [int(score[0]), int(score[1])],
            "events": _jsonable(events),
        }
        if (
            len(frame["policy_start_ns"]) != 2
            or len(frame["policy_end_ns"]) != 2
        ):
            raise ValueError("policy timestamps must each contain two values")
        if any(
            end < start
            for start, end in zip(
                frame["policy_start_ns"], frame["policy_end_ns"], strict=True
            )
        ):
            raise ValueError("policy_end_ns must not precede policy_start_ns")
        self.frames.append(frame)

    def summary(self, final_score: Sequence[int]) -> dict[str, Any]:
        counts = {
            "hit_count": [0, 0],
            "fall_count": [0, 0],
            "knockdown_count": [0, 0],
            "ko_count": [0, 0],
        }
        for frame in self.frames:
            for event in frame["events"]:
                event_type = str(event.get("type", "")).upper()
                if event_type == "HIT":
                    counts["hit_count"][int(event["actor"])] += 1
                elif event_type == "FALL":
                    counts["fall_count"][int(event["target"])] += 1
                elif event_type == "KNOCKDOWN":
                    counts["knockdown_count"][int(event["actor"])] += 1
                elif event_type == "KO":
                    counts["ko_count"][int(event["actor"])] += 1

        durations = [[], []]
        starts = [[], []]
        for frame in self.frames:
            for fighter in range(NUM_FIGHTERS):
                starts[fighter].append(frame["policy_start_ns"][fighter])
                duration = (
                    frame["policy_end_ns"][fighter]
                    - frame["policy_start_ns"][fighter]
                )
                if duration > 0:
                    durations[fighter].append(duration)
        average_latency = [
            (sum(values) / len(values) / 1.0e6) if values else 0.0
            for values in durations
        ]
        average_fps = []
        for values in starts:
            elapsed_ns = values[-1] - values[0] if len(values) > 1 else 0
            average_fps.append(
                (1.0e9 * (len(values) - 1) / elapsed_ns)
                if elapsed_ns > 0
                else 0.0
            )
        scores = [int(final_score[0]), int(final_score[1])]
        winner = None if scores[0] == scores[1] else int(np.argmax(scores))
        for frame in reversed(self.frames):
            match_end = next((event for event in reversed(frame["events"]) if event["type"] == "MATCH_END"), None)
            if match_end is not None:
                winner = match_end["winner"]
                break
        return {
            "model_id": list(self.model_id),
            "final_score": scores,
            "end_reason": next(
                (
                    event["reason"]
                    for frame in reversed(self.frames)
                    for event in reversed(frame["events"])
                    if event["type"] == "ROUND_END"
                ),
                "INCOMPLETE",
            ),
            "winner": winner,
            **counts,
            "avg_fps": average_fps,
            "avg_latency_ms": average_latency,
        }

    def payload(self, final_score: Sequence[int]) -> dict[str, Any]:
        return {
            "summary": self.summary(final_score),
            "metadata": {
                "holomotion_version": "2.0_link_10500",
                "match_id": self.match_id,
                "num_rounds": self.rules.num_rounds,
                "round_duration_s": self.rules.round_duration_s,
                **_jsonable(self.extra_metadata),
            },
            "frames": self.frames,
        }

    def write(self, path: str | Path, final_score: Sequence[int]) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payload(final_score)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            Path(temporary_name).replace(target)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return target


__all__ = [
    "MatchFrameResult",
    "MatchLogger",
    "MatchRules",
    "MatchStateMachine",
]
