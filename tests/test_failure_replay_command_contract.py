"""CPU contract checks for the command's failure-replay reset wiring.

The tested methods are parsed from the current command source so this module
does not import Isaac Lab or duplicate the production reset implementation.
"""

from __future__ import annotations
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holomotion.src.training.failure_replay import (
    FailureReplayConfig,
    RollingSourceExposure,
    retry_start_frames,
    select_failure_replays,
)

_COMMAND_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion/src/env/isaaclab_components/"
    "isaaclab_motion_tracking_command.py"
)
_METHOD_NAMES = (
    "_record_source_exposure",
    "reset",
    "_validate_failure_replay_terms",
    "_select_failure_replay_candidates",
    "_replay_failed_command",
    "_resample_command",
    "_filter_env_ids_for_motion_task",
)


class _FakeCommandTerm:
    def reset(self, env_ids: object = None) -> dict[str, float]:
        return {"base_reset": 1.0}


class _FakeTerminationManager:
    active_terms = ("synthetic_failure",)

    def __init__(self, failures: list[bool]) -> None:
        self.failures = torch.tensor(failures, dtype=torch.bool)

    def get_term_cfg(self, name: str) -> SimpleNamespace:
        assert name == "synthetic_failure"
        return SimpleNamespace(time_out=False)

    def get_term(self, name: str) -> torch.Tensor:
        assert name == "synthetic_failure"
        return self.failures


class _FakeMotionCache:
    source_names = ("source_zero", "source_one")

    def __init__(
        self,
        assignments: list[tuple[list[int], list[int]]],
    ) -> None:
        self.assignments = list(assignments)
        self.sample_calls: list[tuple[int, int, bool]] = []
        self.source_by_clip = torch.tensor([0, 1])
        self.length_by_clip = torch.tensor([10, 10])

    def sample_env_assignments(
        self,
        count: int,
        n_future_frames: int,
        device: torch.device,
        deterministic_start: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.sample_calls.append((count, n_future_frames, deterministic_start))
        assert self.assignments, "Unexpected ordinary sampling"
        clips, frames = self.assignments.pop(0)
        assert len(clips) == count
        assert len(frames) == count
        return (
            torch.tensor(clips, dtype=torch.long, device=device),
            torch.tensor(frames, dtype=torch.long, device=device),
        )

    def source_indices_for_indices(self, clips: torch.Tensor) -> torch.Tensor:
        return self.source_by_clip[clips]

    def lengths_for_indices(self, clips: torch.Tensor) -> torch.Tensor:
        return self.length_by_clip[clips]


@pytest.fixture(scope="module")
def command_class() -> type:
    source = _COMMAND_PATH.read_text()
    tree = ast.parse(source, filename=str(_COMMAND_PATH))
    production = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RefMotionCommand"
    )
    methods = {
        node.name: node
        for node in production.body
        if isinstance(node, ast.FunctionDef)
    }
    assert set(_METHOD_NAMES) <= methods.keys()
    isolated_class = ast.ClassDef(
        name="RefMotionCommandForTest",
        bases=[ast.Name(id="_FakeCommandTerm", ctx=ast.Load())],
        keywords=[],
        body=[methods[name] for name in _METHOD_NAMES],
        decorator_list=[],
    )
    isolated_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            isolated_class,
        ],
        type_ignores=[],
    )
    namespace = {
        "_FakeCommandTerm": _FakeCommandTerm,
        "torch": torch,
        "select_failure_replays": select_failure_replays,
        "retry_start_frames": retry_start_frames,
    }
    exec(
        compile(
            ast.fix_missing_locations(isolated_module),
            str(_COMMAND_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["RefMotionCommandForTest"]


def _make_command(
    command_class: type,
    *,
    clips: list[int],
    frames: list[int],
    failures: list[bool],
    assignments: list[tuple[list[int], list[int]]] | None = None,
    enabled: bool = True,
    evaluating: bool = False,
    budget: bool = False,
    exposure_history: list[int] | None = None,
    upper_probabilities: tuple[float, float] = (0.5, 1.0),
    retries: list[int] | None = None,
) -> object:
    assert len(clips) == len(frames) == len(failures)
    count = len(clips)
    command = command_class()
    command.num_envs = count
    command.device = torch.device("cpu")
    command.cfg = SimpleNamespace(n_fut_frames=2, target_fps=20)
    command._env = SimpleNamespace(
        termination_manager=_FakeTerminationManager(failures)
    )
    command._motion_cache = _FakeMotionCache(assignments or [])
    command._is_evaluating = evaluating
    command._failure_replay_cfg = (
        FailureReplayConfig.parse(
            {
                "enabled": True,
                "trigger_termination_terms": ["synthetic_failure"],
                "replay_probability": 1.0,
                "max_retries_per_motion": 2,
                "count_against_source_budget": budget,
            }
        )
        if enabled
        else None
    )
    command._failure_replay_terms_validated = False
    command._failure_replay_stats = defaultdict(float)
    command._failure_replay_count = torch.tensor(
        retries if retries is not None else [0] * count,
        dtype=torch.long,
    )
    command._clip_indices = torch.tensor(clips, dtype=torch.long)
    command._frame_indices = torch.tensor(frames, dtype=torch.long)
    command._start_frame_indices = torch.tensor(frames, dtype=torch.long)
    command._reward_sum_since_assign = torch.ones(count)
    command._step_count_since_assign = torch.ones(count)
    command._motion_end_mask = torch.ones(count, dtype=torch.bool)
    command.motion_end_counter = torch.ones(count, dtype=torch.long)

    command._failure_replay_exposure = None
    command._failure_replay_upper_probabilities = None
    command._failure_replay_global_source_counts = None
    command._failure_replay_source_counts = None
    if budget:
        exposure = RollingSourceExposure(2, 4, device="cpu")
        if exposure_history is not None:
            exposure.record(torch.tensor(exposure_history))
        command._failure_replay_exposure = exposure
        command._failure_replay_upper_probabilities = torch.tensor(
            upper_probabilities
        )
        command._failure_replay_global_source_counts = torch.zeros(
            2, dtype=torch.long
        )
        command._failure_replay_source_counts = torch.zeros(
            2, dtype=torch.long
        )

    command.completion_calls = []
    command.state_calls = []
    command._record_completion_rate_for_envs = lambda ids: (
        command.completion_calls.append(ids.tolist())
    )
    command._update_ref_motion_state_from_cache = lambda env_ids=None: (
        command.state_calls.append(
            ("update", None if env_ids is None else env_ids.tolist())
        )
    )
    command._align_root_to_ref = lambda ids: command.state_calls.append(
        ("root", ids.tolist())
    )
    command._align_dof_to_ref = lambda ids: command.state_calls.append(
        ("dof", ids.tolist())
    )
    return command


@pytest.mark.parametrize(
    ("failed_frame", "expected_start"),
    [(3, 4), (7, 0)],
)
def test_same_clip_replays_from_next_frame_or_wraps(
    command_class: type,
    failed_frame: int,
    expected_start: int,
) -> None:
    command = _make_command(
        command_class,
        clips=[0],
        frames=[failed_frame],
        failures=[True],
    )

    assert command.reset([0]) == {"base_reset": 1.0}
    assert command._clip_indices.tolist() == [0]
    assert command._frame_indices.tolist() == [expected_start]
    assert command._start_frame_indices.tolist() == [expected_start]
    assert command._failure_replay_count.tolist() == [1]
    assert command._failure_replay_stats["accepted_replays"] == 1
    assert command._motion_cache.sample_calls == []
    assert command.completion_calls == [[0]]


def test_full_budget_window_rejects_replay_and_uses_ordinary_fallback(
    command_class: type,
) -> None:
    command = _make_command(
        command_class,
        clips=[0],
        frames=[3],
        failures=[True],
        assignments=[([1], [5])],
        budget=True,
        exposure_history=[0, 1, 1, 1],
        upper_probabilities=(0.2, 1.0),
        retries=[1],
    )

    command.reset([0])

    assert command._clip_indices.tolist() == [1]
    assert command._frame_indices.tolist() == [5]
    assert command._failure_replay_count.tolist() == [0]
    assert command._failure_replay_stats["blocked_source_budget"] == 1
    assert command._failure_replay_stats["accepted_replays"] == 0
    assert command._motion_cache.sample_calls == [(1, 2, False)]
    assert command._failure_replay_exposure.counts.tolist() == [0, 4]
    assert command._failure_replay_global_source_counts.tolist() == [0, 1]
    assert command._failure_replay_source_counts.tolist() == [0, 0]


def test_multi_candidate_preview_matches_actual_write_order(
    command_class: type,
) -> None:
    command = _make_command(
        command_class,
        clips=[0, 0, 0],
        frames=[3, 3, 3],
        failures=[True, True, True],
        assignments=[([1, 1, 1], [5, 6, 7])],
        budget=True,
        exposure_history=[0, 0, 1, 1],
    )

    command.reset(torch.tensor([2, 0, 1]))

    assert command._clip_indices.tolist() == [0, 1, 0]
    assert command._frame_indices.tolist() == [4, 7, 4]
    assert command._failure_replay_count.tolist() == [1, 0, 1]
    assert command._failure_replay_stats["accepted_replays"] == 2
    assert command._failure_replay_stats["blocked_source_budget"] == 1
    assert command._failure_replay_exposure.position == 3
    assert command._failure_replay_exposure.counts.tolist() == [2, 2]
    assert command._failure_replay_global_source_counts.tolist() == [0, 1]
    assert command._failure_replay_source_counts.tolist() == [2, 0]
    assert command._motion_cache.sample_calls == [(3, 2, False)]
    assert command.completion_calls == [[1], [2, 0]]


def test_ordinary_first_write_can_enable_later_replay(
    command_class: type,
) -> None:
    command = _make_command(
        command_class,
        clips=[1, 0],
        frames=[2, 3],
        failures=[False, True],
        assignments=[([1, 1], [5, 6])],
        budget=True,
        exposure_history=[0, 1, 1, 1],
        upper_probabilities=(0.25, 1.0),
    )

    command.reset([0, 1])

    assert command._clip_indices.tolist() == [1, 0]
    assert command._frame_indices.tolist() == [5, 4]
    assert command._failure_replay_count.tolist() == [0, 1]
    assert command._failure_replay_stats["accepted_replays"] == 1
    assert command._failure_replay_exposure.position == 2
    assert command._failure_replay_exposure.buffer.tolist() == [1, 0, 1, 1]
    assert command._failure_replay_exposure.counts.tolist() == [1, 3]
    assert command._failure_replay_global_source_counts.tolist() == [0, 1]
    assert command._failure_replay_source_counts.tolist() == [1, 0]


@pytest.mark.parametrize(
    ("enabled", "evaluating"),
    [(False, False), (True, True)],
)
def test_disabled_or_evaluation_never_replays(
    command_class: type,
    enabled: bool,
    evaluating: bool,
) -> None:
    command = _make_command(
        command_class,
        clips=[0],
        frames=[3],
        failures=[True],
        assignments=[([1], [5])],
        enabled=enabled,
        evaluating=evaluating,
        budget=evaluating,
        exposure_history=[0, 1, 1, 1] if evaluating else None,
    )
    before_exposure = (
        command._failure_replay_exposure.buffer.clone() if evaluating else None
    )

    command.reset([0])

    assert command._clip_indices.tolist() == [1]
    assert command._frame_indices.tolist() == [5]
    assert command._failure_replay_count.tolist() == [0]
    assert command._failure_replay_stats["accepted_replays"] == 0
    assert command._motion_cache.sample_calls == [(1, 2, evaluating)]
    if evaluating:
        assert torch.equal(
            command._failure_replay_exposure.buffer, before_exposure
        )
        assert command._failure_replay_source_counts.tolist() == [0, 0]


def test_retry_limit_falls_back_and_clears_counter(
    command_class: type,
) -> None:
    command = _make_command(
        command_class,
        clips=[0],
        frames=[3],
        failures=[True],
        assignments=[([1], [5])],
        retries=[2],
    )

    command.reset([0])

    assert command._clip_indices.tolist() == [1]
    assert command._failure_replay_count.tolist() == [0]
    assert command._failure_replay_stats["blocked_max_retries"] == 1
    assert command._failure_replay_stats["accepted_replays"] == 0
