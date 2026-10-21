"""Training deadlines must leave an intact, resumable update checkpoint."""

import hashlib
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holoboxing.training.checkpoint_io import save_checkpoint, training_result
from holoboxing.training.rsl_rl.runners.on_policy_runner import OnPolicyRunner
from holoboxing.training.time_budget import TrainingBudget


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def budget(clock, **changes):
    return TrainingBudget(
        **{
            "max_runtime_seconds": 100,
            "max_training_seconds": 80,
            "save_reserve_seconds": 20,
            "checkpoint_interval_seconds": 10,
            "clock": clock,
            **changes,
        }
    )


def test_setup_time_counts_and_next_update_cannot_consume_save_reserve():
    clock = Clock()
    timer = budget(clock)
    clock.advance(70)
    timer.update_finished(9)
    assert timer.should_stop()
    assert timer.report()["stop_reason"] == "time_budget_reached"
    assert clock() < 100 - 20


def test_unbounded_training_keeps_periodic_saves_and_signal_stopping():
    clock = Clock()
    timer = budget(clock, max_runtime_seconds=None, max_training_seconds=None)
    clock.advance(100000)
    assert not timer.should_stop()
    assert timer.checkpoint_due()
    timer.checkpoint_saved()
    assert not timer.checkpoint_due()
    timer.request_stop(signal.SIGTERM)
    assert timer.should_stop()
    assert timer.report()["stop_reason"] == "signal_SIGTERM"


def test_default_cli_has_no_implicit_wall_time_cutoff():
    import argparse

    from holoboxing.training.time_budget import (
        add_budget_arguments,
        budget_from_arguments,
    )

    parser = argparse.ArgumentParser()
    add_budget_arguments(parser)
    timer = budget_from_arguments(parser.parse_args([]), started_at=0)
    timer.clock = Clock()
    timer.clock.advance(100000)
    assert timer.max_runtime_seconds is timer.max_training_seconds is None
    assert not timer.should_stop()
    assert timer.checkpoint_due()


def test_deadline_includes_work_done_before_training_loop():
    clock = Clock()
    clock.advance(90)
    timer = budget(clock, started_at=0)
    assert timer.should_stop()


def test_signal_handlers_request_stop_and_restore_previous_handlers():
    clock = Clock()
    timer = budget(clock)
    previous = {
        s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)
    }
    with timer.handle_signals():
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert timer.should_stop()
    assert {s: signal.getsignal(s) for s in previous} == previous


@pytest.mark.parametrize(
    "changes",
    [
        {"max_runtime_seconds": 0},
        {"max_training_seconds": float("nan")},
        {"save_reserve_seconds": 100},
        {"save_reserve_seconds": -1},
        {"checkpoint_interval_seconds": 0},
    ],
)
def test_invalid_budget_cannot_silently_disable_stopping(changes):
    with pytest.raises(ValueError):
        budget(Clock(), **changes)


def test_failed_save_preserves_last_valid_checkpoint_and_has_no_completion(
    tmp_path, monkeypatch
):
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"step": 2})
    valid = path.read_bytes()

    def fail(state, stream):
        stream.write(b"incomplete")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="storage failure"):
        save_checkpoint(path, {"step": 3})
    assert path.read_bytes() == valid
    assert (tmp_path / ".last.pt.partial").read_bytes() == b"incomplete"
    assert not (tmp_path / "training-result.json").exists()


def runner_at_update_boundary(output, clock):
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.output = output
    runner.iteration = runner.transitions_per_role = (
        runner.invalid_transitions
    ) = 0
    runner.elapsed_seconds = 0
    runner.config = SimpleNamespace(num_envs=2, num_steps_per_env=3)
    storage = SimpleNamespace(step=0)

    def collect():
        storage.step = 3
        clock.advance(8)
        return {"local_recovery_count": 0}

    def update():
        assert storage.step == 3
        storage.step = 0
        clock.advance(2)
        return {}

    runner.collect = collect
    runner.algorithm = SimpleNamespace(
        update=update,
        storage=storage,
        learning_rates={},
        optimizer_steps={},
        last_parity={},
    )
    runner.lower_guard = SimpleNamespace(check=lambda: None)
    runner.amp = SimpleNamespace(optimizer_steps=0)
    runner.policy = SimpleNamespace(actor_cuda_graph_status=lambda: {})
    saved = []

    def save(path):
        assert storage.step == 0
        save_checkpoint(
            path,
            {
                "iteration": runner.iteration,
                "optimizer_steps": runner.iteration,
            },
        )
        saved.append(Path(path))

    runner.save = save
    return runner, saved


def test_runner_stops_before_next_update_and_reports_saved_checkpoint(
    tmp_path, monkeypatch
):
    clock = Clock()
    monkeypatch.setattr(
        "holoboxing.training.rsl_rl.runners.on_policy_runner.time.monotonic",
        clock,
    )
    runner, saved = runner_at_update_boundary(tmp_path, clock)
    timer = budget(clock, max_training_seconds=25)
    result = runner.learn(20, save_interval=100, time_budget=timer)
    assert runner.iteration == 2
    assert result["progress"]["completed_updates"] == 2
    assert result["budget"]["stop_reason"] == "time_budget_reached"
    checkpoint = Path(result["checkpoint"])
    assert checkpoint == saved[-1]
    assert torch.load(checkpoint, weights_only=True)["optimizer_steps"] == 2
    assert (
        result["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert (
        json.loads((tmp_path / "training-result.json").read_text()) == result
    )


def test_exhausted_startup_budget_still_saves_initial_learning_state(tmp_path):
    clock = Clock()
    runner, saved = runner_at_update_boundary(tmp_path, clock)
    timer = budget(clock)
    clock.advance(90)
    result = runner.learn(10, time_budget=timer)
    assert runner.iteration == 0
    assert result["progress"]["completed_updates"] == 0
    assert len(saved) == 1


def test_missing_checkpoint_cannot_publish_completion(tmp_path):
    with pytest.raises(FileNotFoundError):
        training_result(tmp_path, tmp_path / "missing.pt", progress={})
    assert not (tmp_path / "training-result.json").exists()
