"""Monitor episode weighting independently from the optimizer and evaluator."""

import pytest
import torch

from holomotion.src.algo.supermimic.monitoring import (
    EpisodeMonitor,
    TrainingRewardArea,
)


def test_wall_time_area_counts_startup_as_zero_and_clips_overrun():
    area = TrainingRewardArea(horizon_seconds=30, tail_seconds=5)
    area.observe(5, 2)
    area.observe(20, 4)
    area.observe(32, 100)
    result = area.summary(35)
    # [0,5):0, [5,20):2, [20,30):4. The overrun earns nothing.
    assert result["reward_seconds"] == 70
    assert result["time_mean_reward"] == 70 / 30
    assert result["final_window_mean_reward"] == 4
    assert result["complete"]


def test_incomplete_training_and_negative_reward_remain_visible():
    area = TrainingRewardArea(horizon_seconds=30, tail_seconds=5)
    area.observe(0, -2)
    area.observe(10, -1)
    result = area.summary(15)
    assert result["reward_seconds"] == -25
    assert not result["complete"]
    assert result["time_mean_reward"] is None
    assert result["final_window_mean_reward"] is None
    assert area.summary(30)["reward_seconds"] == -40
    # Summary calls must not change the curve or integrate twice.
    assert area.summary(15) == result


def test_empty_curve_cannot_be_a_completed_training_result():
    result = TrainingRewardArea().summary(1800)
    assert not result["complete"]
    assert result["time_mean_reward"] is None


@pytest.mark.parametrize(
    "time,reward", [(float("nan"), 0), (0, float("inf")), (-1, 0)]
)
def test_invalid_reward_observation_is_rejected(time, reward):
    with pytest.raises(ValueError):
        TrainingRewardArea().observe(time, reward)


def test_reward_clock_cannot_go_backwards():
    area = TrainingRewardArea()
    area.observe(5, 1)
    with pytest.raises(ValueError):
        area.observe(5, 2)
    with pytest.raises(ValueError):
        area.summary(4)


def test_episode_means_weight_completed_episodes_and_keep_training_clock():
    monitor = EpisodeMonitor()
    monitor.add(
        dict(
            done=torch.tensor([True, True]),
            term=torch.tensor([True, False]),
            ep_len=torch.tensor([10.0, 20.0]),
            ep_ret_norm=torch.tensor([0.2, 0.4]),
        )
    )
    monitor.add(
        dict(
            done=torch.tensor([True, False]),
            term=torch.tensor([False, False]),
            ep_len=torch.tensor([60.0]),
            ep_ret_norm=torch.tensor([0.9]),
        )
    )
    result = monitor.summary(vector_steps=2, training_seconds=10)
    assert result["real_transitions"] == 4
    assert result["sampled_episode_mean_length"] == 30.0
    assert abs(result["sampled_episode_mean_normalized_return"] - 0.5) < 1e-7
    assert result["termination_fraction_among_endings"] == 1 / 3
    assert result["real_transitions_per_training_second"] == 0.4
    assert result["training_gpu_hours"] == 10 / 3600


def test_no_completed_episode_has_unknown_mean_rather_than_zero_return():
    monitor = EpisodeMonitor()
    monitor.add(
        dict(
            done=torch.zeros(2, dtype=torch.bool),
            term=torch.zeros(2, dtype=torch.bool),
        )
    )
    result = monitor.summary(vector_steps=1, training_seconds=2)
    assert result["sampled_episode_mean_normalized_return"] is None
    assert result["termination_fraction_among_endings"] is None
