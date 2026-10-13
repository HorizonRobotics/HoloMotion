"""Pure-memory regression tests for failure replay decisions."""

import pytest
import torch

from holomotion.src.training.failure_replay import (
    FailureReplayConfig,
    RollingSourceExposure,
    retry_start_frames,
    select_failure_replays,
)


def _config(**overrides: object) -> FailureReplayConfig:
    values: dict[str, object] = {
        "enabled": True,
        "trigger_termination_terms": ("synthetic_failure",),
    }
    values.update(overrides)
    return FailureReplayConfig(**values)


def test_config_is_disabled_by_default_and_validates_enabled_values() -> None:
    assert FailureReplayConfig().enabled is False
    assert FailureReplayConfig.parse(None) is None
    assert FailureReplayConfig.parse({}) is None
    assert FailureReplayConfig.parse({"enabled": False}) is None

    parsed = FailureReplayConfig.parse(
        {"enabled": True, "trigger_termination_terms": ["synthetic_failure"]}
    )
    assert parsed is not None
    assert parsed.trigger_termination_terms == ("synthetic_failure",)

    with pytest.raises(ValueError, match="trigger_termination_terms"):
        FailureReplayConfig.parse({"enabled": True})
    with pytest.raises(ValueError, match="replay_probability"):
        FailureReplayConfig.parse(
            {
                "enabled": True,
                "trigger_termination_terms": ["synthetic_failure"],
                "replay_probability": 1.1,
            }
        )
    with pytest.raises(ValueError, match="max_retries_per_motion"):
        FailureReplayConfig.parse(
            {
                "enabled": True,
                "trigger_termination_terms": ["synthetic_failure"],
                "max_retries_per_motion": -1,
            }
        )


def test_replay_selection_obeys_retry_and_probability_and_is_eval_noop() -> (
    None
):
    failures = torch.tensor([True, True, True, False])
    retries = torch.tensor([0, 1, 2, 0])
    draws = torch.tensor([0.1, 0.4, 0.2, 0.1])
    config = _config(replay_probability=0.5, max_retries_per_motion=2)

    selected = select_failure_replays(failures, retries, draws, config=config)
    assert selected.tolist() == [True, True, False, False]
    assert retries.tolist() == [0, 1, 2, 0]
    assert not select_failure_replays(
        failures, retries, draws, config=config, evaluating=True
    ).any()
    assert not select_failure_replays(
        failures, retries, draws, config=None
    ).any()
    assert not select_failure_replays(
        failures, retries, draws, config=FailureReplayConfig()
    ).any()


def test_retry_start_next_frame_wrap_and_backtrack() -> None:
    failed = torch.tensor([40, 3, 44, 2])
    lengths = torch.tensor([100, 20, 50, 3])
    assert retry_start_frames(
        failed,
        lengths,
        n_future_frames=5,
        backtrack_frames=0,
        replay_start_mode="next_frame",
    ).tolist() == [41, 4, 0, 0]
    assert retry_start_frames(
        torch.tensor([40, 3, 99]),
        torch.tensor([100, 20, 50]),
        n_future_frames=5,
        replay_start_mode="backtrack",
        backtrack_frames=10,
    ).tolist() == [30, 0, 44]
    with pytest.raises(ValueError, match="start mode"):
        retry_start_frames(
            failed,
            lengths,
            n_future_frames=5,
            backtrack_frames=0,
            replay_start_mode="unknown",
        )


def test_full_window_gate_accounts_for_the_old_slot_being_overwritten() -> (
    None
):
    exposure = RollingSourceExposure(num_sources=2, capacity=4, device="cpu")
    exposure.record(torch.tensor([0, 0, 1, 1]))
    before_counts = exposure.counts.clone()
    before_position = exposure.position
    before_buffer = exposure.buffer.clone()

    plan = exposure.plan_replays(
        ordinary_source_indices=torch.tensor([1]),
        candidate_replay_sources=torch.tensor([0]),
        upper_probabilities=torch.tensor([0.5, 1.0]),
    )
    assert plan.accepted_mask.tolist() == [True]
    assert plan.final_source_indices.tolist() == [0]
    assert torch.equal(exposure.counts, before_counts)
    assert exposure.position == before_position
    assert torch.equal(exposure.buffer, before_buffer)

    exposure.record(plan.final_source_indices)
    assert exposure.counts.tolist() == [2, 2]
    assert exposure.position == 1


def test_multiple_candidates_follow_true_ring_write_order() -> None:
    exposure = RollingSourceExposure(num_sources=2, capacity=4, device="cpu")
    exposure.record(torch.tensor([0, 0, 1, 1]))

    plan = exposure.plan_replays(
        ordinary_source_indices=torch.tensor([1, 1, 1]),
        candidate_replay_sources=torch.tensor([0, 0, 0]),
        upper_probabilities=torch.tensor([0.5, 1.0]),
    )
    assert plan.accepted_mask.tolist() == [True, True, False]
    assert plan.final_source_indices.tolist() == [0, 0, 1]
    assert exposure.counts.tolist() == [2, 2]
    assert exposure.position == 0

    exposure.record(plan.final_source_indices)
    assert exposure.counts.tolist() == [2, 2]
    assert exposure.position == 3


def test_ordinary_assignment_order_changes_later_replay_decision() -> None:
    exposure = RollingSourceExposure(num_sources=2, capacity=4, device="cpu")
    exposure.record(torch.tensor([0, 1, 1, 1]))
    upper = torch.tensor([0.25, 1.0])

    first_normal_zero = exposure.plan_replays(
        ordinary_source_indices=torch.tensor([0, 1]),
        candidate_replay_sources=torch.tensor([-1, 0]),
        upper_probabilities=upper,
    )
    assert first_normal_zero.accepted_mask.tolist() == [False, False]
    assert first_normal_zero.final_source_indices.tolist() == [0, 1]

    first_normal_one = exposure.plan_replays(
        ordinary_source_indices=torch.tensor([1, 0]),
        candidate_replay_sources=torch.tensor([-1, 0]),
        upper_probabilities=upper,
    )
    assert first_normal_one.accepted_mask.tolist() == [False, True]
    assert first_normal_one.final_source_indices.tolist() == [1, 0]


def test_ordinary_assignment_is_not_a_global_source_cap() -> None:
    exposure = RollingSourceExposure(num_sources=2, capacity=4, device="cpu")
    exposure.record(torch.tensor([0, 1, 1, 1]))

    plan = exposure.plan_replays(
        ordinary_source_indices=torch.tensor([0]),
        candidate_replay_sources=torch.tensor([0]),
        upper_probabilities=torch.tensor([0.1, 1.0]),
    )
    assert plan.accepted_mask.tolist() == [False]
    assert plan.final_source_indices.tolist() == [0]
    exposure.record(plan.final_source_indices)
    assert exposure.counts.tolist() == [1, 3]
    assert exposure.counts[0].item() / exposure.size > 0.1


def test_record_large_batch_keeps_ring_and_counts_consistent() -> None:
    exposure = RollingSourceExposure(num_sources=2, capacity=4, device="cpu")
    exposure.record(torch.tensor([1, 0, 0, 1, 0]))
    assert exposure.size == 4
    assert exposure.position == 0
    assert exposure.buffer.tolist() == [0, 0, 1, 0]
    assert exposure.counts.tolist() == [3, 1]
    exposure.record(torch.tensor([1]))
    assert exposure.buffer.tolist() == [1, 0, 1, 0]
    assert exposure.counts.tolist() == [2, 2]
    assert exposure.position == 1
