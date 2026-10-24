"""Lightweight, timestamped policy records for learning-process videos."""

import math
from pathlib import Path

import torch

from holomotion.src.utils.artifacts import atomic_json, fingerprint


class PolicyProgressSnapshots:
    def __init__(
        self,
        output,
        policy,
        specification,
        schedule,
        clock,
        interval_minutes,
        budget_seconds,
    ):
        if not math.isfinite(interval_minutes) or interval_minutes <= 0:
            raise ValueError("Snapshot interval must be finite and positive")
        self.output = Path(output)
        self.policy, self.specification, self.schedule = (
            policy,
            specification,
            schedule,
        )
        self.clock = clock
        self.interval = interval_minutes * 60
        self.budget = budget_seconds
        self.next_index = 1
        self.records = []
        self.finished = False

    def _capture(self, target, label):
        started = self.clock()
        # Copy only actor parameters and buffers (including normalizers).
        # No replay, optimizer, simulator state, RNG draw or evaluation.
        state = {
            k: v.detach().to("cpu", copy=True)
            for k, v in self.policy.state_dict().items()
        }
        schedule = self.schedule.state_dict()
        relative = (
            Path("policy_progress") / f"policy_{len(self.records):03d}.pt"
        )
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = dict(
            format="supermimic_policy_progress_v1",
            purpose="policy_progress_only",
            resume_supported=False,
            pi=state,
            specification=self.specification,
            schedule=schedule,
            label=label,
            target_seconds=target,
            training_wall_s=started,
        )
        temporary = path.with_suffix(".pt.tmp")
        torch.save(bundle, temporary)
        temporary.replace(path)
        record = dict(
            path=str(relative),
            sha256=fingerprint(path),
            purpose="policy_progress_only",
            label=label,
            target_seconds=target,
            training_wall_s=started,
            late_by_seconds=max(0.0, started - target),
            vector_steps=schedule["vector_steps"],
            world_updates=schedule["world_updates"],
            actor_updates=schedule["actor_updates"],
        )
        record["capture_finished_training_wall_s"] = self.clock()
        record["save_seconds"] = (
            record["capture_finished_training_wall_s"] - started
        )
        self.records.append(record)
        atomic_json(
            self.output / "policy_progress/index.json",
            dict(
                format="supermimic_policy_progress_v1",
                purpose="policy_progress_only",
                interval_seconds=self.interval,
                budget_seconds=self.budget,
                snapshots=self.records,
                finished=self.finished,
            ),
        )
        return record

    def initial(self):
        if self.records:
            raise RuntimeError("Initial policy progress was already captured")
        return self._capture(0.0, "initial")

    def due(self):
        # Targets use integer multiplication, avoiding cumulative interval drift.
        elapsed = self.clock()
        while self.next_index * self.interval < self.budget and (
            self.next_index * self.interval <= elapsed
        ):
            target = self.next_index * self.interval
            self._capture(target, "periodic")
            self.next_index += 1

    def final(self):
        if self.finished:
            raise RuntimeError("Final policy progress was already captured")
        self.finished = True
        return self._capture(self.budget, "final")
