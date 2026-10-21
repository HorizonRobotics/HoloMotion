"""Cooperative training deadlines checked at resumable update boundaries."""

from __future__ import annotations
import math
import signal
import time
from contextlib import contextmanager


class TrainingBudget:
    """Reserve saving time; zero CLI limits disable deadlines."""

    def __init__(
        self,
        *,
        max_runtime_seconds=None,
        max_training_seconds=None,
        save_reserve_seconds=600.0,
        checkpoint_interval_seconds=300.0,
        started_at=None,
        clock=None,
    ):
        for name, value in (
            ("max_runtime_seconds", max_runtime_seconds),
            ("max_training_seconds", max_training_seconds),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or None")
        if not math.isfinite(save_reserve_seconds) or save_reserve_seconds < 0:
            raise ValueError("Save reserve must be finite and nonnegative")
        if (
            max_runtime_seconds is not None
            and save_reserve_seconds >= max_runtime_seconds
        ):
            raise ValueError(
                "Save reserve must be shorter than the runtime limit"
            )
        if (
            not math.isfinite(checkpoint_interval_seconds)
            or checkpoint_interval_seconds <= 0
        ):
            raise ValueError("Checkpoint interval must be positive and finite")
        self.clock = clock or time.monotonic
        self.started_at = self.clock() if started_at is None else started_at
        if not math.isfinite(self.started_at):
            raise ValueError("Start time must be finite")
        self.max_runtime_seconds = max_runtime_seconds
        self.max_training_seconds = max_training_seconds
        self.save_reserve_seconds = save_reserve_seconds
        self.checkpoint_interval_seconds = checkpoint_interval_seconds
        limits = [
            value
            for value in (
                max_training_seconds,
                None
                if max_runtime_seconds is None
                else max_runtime_seconds - save_reserve_seconds,
            )
            if value is not None
        ]
        self.work_deadline = (
            self.started_at + min(limits) if limits else math.inf
        )
        self.last_checkpoint_at = self.started_at
        self.longest_update_seconds = 0.0
        self.stop_reason = None

    def request_stop(self, signum=None, _frame=None, *, reason=None):
        """Signal handlers only request a stop; the main loop performs I/O."""
        if self.stop_reason is None:
            self.stop_reason = (
                reason or "stop_requested"
                if signum is None
                else f"signal_{signal.Signals(signum).name}"
            )

    def should_stop(self):
        if self.stop_reason is not None:
            return True
        # Avoid starting an update that is likely to consume the save reserve.
        if (
            self.clock() + 1.25 * self.longest_update_seconds
            >= self.work_deadline
        ):
            self.stop_reason = "time_budget_reached"
            return True
        return False

    def update_finished(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Update duration must be finite and nonnegative")
        self.longest_update_seconds = max(self.longest_update_seconds, seconds)

    def checkpoint_due(self):
        return (
            self.clock() - self.last_checkpoint_at
            >= self.checkpoint_interval_seconds
        )

    def checkpoint_saved(self):
        self.last_checkpoint_at = self.clock()

    def report(self):
        return {
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_training_seconds": self.max_training_seconds,
            "save_reserve_seconds": self.save_reserve_seconds,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "runtime_seconds": max(0.0, self.clock() - self.started_at),
            "longest_update_seconds": self.longest_update_seconds,
            "stop_reason": self.stop_reason or "requested_progress_complete",
        }

    @contextmanager
    def handle_signals(self):
        previous = {}
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, self.request_stop)
            yield self
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def add_budget_arguments(parser):
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0,
        help="optional runtime limit including setup/save; 0 disables it",
    )
    parser.add_argument(
        "--max-training-seconds",
        type=float,
        default=0,
        help="stop at this elapsed time; 0 disables the work limit",
    )
    parser.add_argument("--save-reserve-seconds", type=float, default=600)
    parser.add_argument(
        "--checkpoint-interval-seconds", type=float, default=300
    )


def budget_from_arguments(args, *, started_at):
    return TrainingBudget(
        max_runtime_seconds=None
        if args.max_runtime_seconds == 0
        else args.max_runtime_seconds,
        max_training_seconds=None
        if args.max_training_seconds == 0
        else args.max_training_seconds,
        save_reserve_seconds=args.save_reserve_seconds,
        checkpoint_interval_seconds=args.checkpoint_interval_seconds,
        started_at=started_at,
    )
