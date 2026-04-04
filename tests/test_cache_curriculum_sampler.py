import json
from types import SimpleNamespace

import torch

import holomotion.src.training.h5_dataloader as h5_dataloader
from holomotion.src.algo.ppo import PPO
from holomotion.src.training.h5_dataloader import (
    MotionClipBatchCache,
    ClipBatch,
    PrioritizedInfiniteSampler,
)


def _update_sampler(
    sampler: PrioritizedInfiniteSampler,
    completion_rates: list[float],
    *,
    swap_index: int,
) -> bool:
    num_windows = len(completion_rates)
    window_indices = torch.arange(num_windows, dtype=torch.long)
    completion_rate_means = torch.tensor(completion_rates, dtype=torch.float32)
    mpkpe_signal_means = torch.zeros(num_windows, dtype=torch.float32)
    counts = torch.ones(num_windows, dtype=torch.float32)
    return sampler.maybe_update_from_observations(
        window_indices=window_indices,
        mpkpe_signal_means=mpkpe_signal_means,
        completion_rate_means=completion_rate_means,
        counts=counts,
        swap_index=swap_index,
    )


def _update_sampler_subset(
    sampler: PrioritizedInfiniteSampler,
    *,
    window_indices: list[int],
    completion_rates: list[float],
    swap_index: int,
    counts: list[float] | None = None,
) -> bool:
    if counts is None:
        counts = [1.0] * len(window_indices)
    window_index_tensor = torch.tensor(window_indices, dtype=torch.long)
    completion_rate_tensor = torch.tensor(
        completion_rates, dtype=torch.float32
    )
    count_tensor = torch.tensor(counts, dtype=torch.float32)
    mpkpe_signal_means = torch.zeros(len(window_indices), dtype=torch.float32)
    return sampler.maybe_update_from_observations(
        window_indices=window_index_tensor,
        mpkpe_signal_means=mpkpe_signal_means,
        completion_rate_means=completion_rate_tensor,
        counts=count_tensor,
        swap_index=swap_index,
    )


class _ChunkLimitedSampler:
    def __init__(self, *, max_query_size: int) -> None:
        self.max_query_size = int(max_query_size)
        self.state_version = 7
        self.query_sizes: list[int] = []

    def _checked_indices(self, window_indices: torch.Tensor) -> torch.Tensor:
        indices = window_indices.to(dtype=torch.long).reshape(-1)
        size = int(indices.numel())
        self.query_sizes.append(size)
        if size > self.max_query_size:
            raise AssertionError(
                f"expected chunked access <= {self.max_query_size}, got {size}"
            )
        return indices

    def get_scores_for_indices(
        self, window_indices: torch.Tensor
    ) -> torch.Tensor:
        indices = self._checked_indices(window_indices)
        return indices.to(dtype=torch.float32)

    def get_window_state_for_indices(
        self, window_indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        indices = self._checked_indices(window_indices)
        count = int(indices.numel())
        return {
            "ema_completion_rate": torch.zeros(count, dtype=torch.float32),
            "completion_rate_rel_improve": torch.zeros(
                count, dtype=torch.float32
            ),
            "selection_count": indices + 10,
            "seen": torch.zeros(count, dtype=torch.bool),
            "in_prioritized_pool": torch.zeros(count, dtype=torch.bool),
        }

    def get_pool_statistics(self) -> dict[str, float]:
        return {
            "prioritized_pool_size": 0.0,
            "prioritized_pool_mean_score": 0.0,
            "uniform_pool_mean_score": 0.0,
            "entered_prioritized_pool_count": 0.0,
            "exited_prioritized_pool_count": 0.0,
        }


class _FakeTrainDataset:
    def __init__(self, windows: list[SimpleNamespace]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def close(self) -> None:
        return


def test_sampler_uses_configured_uniform_ratio_immediately():
    sampler = PrioritizedInfiniteSampler(
        dataset_len=8,
        batch_size=10,
        seed=0,
        p_a_ratio=0.3,
    )

    assert sampler._pool_batch_sizes() == (3, 7)


def test_completion_rate_relative_improvement_alone_drives_scores():
    sampler = PrioritizedInfiniteSampler(
        dataset_len=3,
        batch_size=2,
        seed=0,
        p_a_ratio=0.5,
        ema_alpha_signal=0.5,
        ema_alpha_rel_improve=1.0,
    )

    assert _update_sampler(sampler, [0.2, 0.2, 0.2], swap_index=1)
    assert _update_sampler(sampler, [0.8, 0.2, 0.2], swap_index=2)

    scores = sampler.get_scores_for_indices(torch.arange(3, dtype=torch.long))

    assert scores[0].item() > 0.0
    assert scores[1].item() == 0.0
    assert scores[2].item() == 0.0


def test_sampler_weights_progress_by_remaining_difficulty():
    sampler = PrioritizedInfiniteSampler(
        dataset_len=2,
        batch_size=2,
        seed=0,
        p_a_ratio=0.5,
        ema_alpha_signal=1.0,
        ema_alpha_rel_improve=1.0,
    )

    assert _update_sampler(sampler, [0.1, 0.8], swap_index=1)
    assert _update_sampler(sampler, [0.2, 0.9], swap_index=2)

    scores = sampler.get_scores_for_indices(torch.arange(2, dtype=torch.long))

    assert scores[0].item() > scores[1].item()


def test_sampler_tracks_cumulative_selection_counts():
    sampler = PrioritizedInfiniteSampler(
        dataset_len=4,
        batch_size=2,
        seed=0,
        p_a_ratio=0.5,
    )

    iterator = iter(sampler)
    selected_indices = [next(iterator) for _ in range(4)]
    state = sampler.get_window_state_for_indices(torch.arange(4))
    selection_count = state["selection_count"]
    expected_count = torch.bincount(
        torch.tensor(selected_indices, dtype=torch.long), minlength=4
    )

    assert int(selection_count.sum().item()) == 4
    assert torch.equal(selection_count, expected_count)


def test_low_completion_plateau_drops_from_prioritized_replay():
    sampler = PrioritizedInfiniteSampler(
        dataset_len=3,
        batch_size=3,
        seed=0,
        p_a_ratio=1.0 / 3.0,
        ema_alpha_signal=1.0,
        ema_alpha_rel_improve=1.0,
    )

    assert _update_sampler(sampler, [0.1, 0.2, 0.2], swap_index=1)
    assert _update_sampler(sampler, [0.4, 0.2, 0.2], swap_index=2)
    assert _update_sampler(sampler, [0.4, 0.8, 0.8], swap_index=3)

    state = sampler.get_window_state_for_indices(
        torch.arange(3, dtype=torch.long)
    )
    assert not bool(state["in_prioritized_pool"][0].item())

    generator = torch.Generator().manual_seed(0)
    uniform_pick = sampler._sample_uniform_indices(generator, 3)
    assert 0 in uniform_pick.tolist()


def test_prioritized_windows_persist_beyond_immediate_batch():
    sampler = PrioritizedInfiniteSampler(
        dataset_len=6,
        batch_size=4,
        seed=0,
        p_a_ratio=0.5,
        ema_alpha_signal=1.0,
        ema_alpha_rel_improve=1.0,
    )

    assert _update_sampler_subset(
        sampler,
        window_indices=[0, 1],
        completion_rates=[0.2, 0.2],
        swap_index=1,
    )
    assert _update_sampler_subset(
        sampler,
        window_indices=[0, 1],
        completion_rates=[0.8, 0.7],
        swap_index=2,
    )
    assert _update_sampler_subset(
        sampler,
        window_indices=[2, 3],
        completion_rates=[0.3, 0.3],
        swap_index=3,
    )

    state = sampler.get_window_state_for_indices(torch.tensor([0, 1]))
    assert torch.equal(
        state["in_prioritized_pool"],
        torch.tensor([True, True], dtype=torch.bool),
    )


def test_sampler_reports_pool_means_and_membership_churn():
    sampler = PrioritizedInfiniteSampler(
        dataset_len=4,
        batch_size=4,
        seed=0,
        p_a_ratio=0.5,
        ema_alpha_signal=0.5,
        ema_alpha_rel_improve=1.0,
    )

    assert _update_sampler(sampler, [0.2, 0.2, 0.2, 0.2], swap_index=1)
    assert _update_sampler(sampler, [0.9, 0.8, 0.2, 0.2], swap_index=2)
    assert _update_sampler(sampler, [0.1, 0.2, 0.9, 0.8], swap_index=3)

    next(iter(sampler))
    stats = sampler.get_pool_statistics()

    assert stats is not None
    assert set(stats) == {
        "prioritized_pool_size",
        "prioritized_pool_mean_score",
        "uniform_pool_mean_score",
        "entered_prioritized_pool_count",
        "exited_prioritized_pool_count",
    }
    assert stats["prioritized_pool_size"] == 2.0
    assert stats["entered_prioritized_pool_count"] == 2.0
    assert stats["exited_prioritized_pool_count"] == 2.0
    assert (
        stats["prioritized_pool_mean_score"] > stats["uniform_pool_mean_score"]
    )


def test_sampler_hot_path_avoids_full_dataset_temporaries(monkeypatch):
    sampler = PrioritizedInfiniteSampler(
        dataset_len=1_000_000,
        batch_size=8,
        seed=0,
        p_a_ratio=0.5,
        ema_alpha_signal=1.0,
        ema_alpha_rel_improve=1.0,
    )

    orig_zeros = h5_dataloader.torch.zeros
    orig_arange = h5_dataloader.torch.arange
    orig_randperm = h5_dataloader.torch.randperm

    def _guard_size(arg) -> int | None:
        if isinstance(arg, int):
            return arg
        if (
            isinstance(arg, tuple)
            and len(arg) == 1
            and isinstance(arg[0], int)
        ):
            return arg[0]
        return None

    def guarded_zeros(*args, **kwargs):
        size = _guard_size(args[0]) if args else None
        if size == sampler._ds_len:
            raise AssertionError("full-dataset zeros in hot path")
        return orig_zeros(*args, **kwargs)

    def guarded_arange(*args, **kwargs):
        if args and args[0] == sampler._ds_len:
            raise AssertionError("full-dataset arange in hot path")
        return orig_arange(*args, **kwargs)

    def guarded_randperm(*args, **kwargs):
        if args and args[0] == sampler._ds_len:
            raise AssertionError("full-dataset randperm in hot path")
        return orig_randperm(*args, **kwargs)

    monkeypatch.setattr(h5_dataloader.torch, "zeros", guarded_zeros)
    monkeypatch.setattr(h5_dataloader.torch, "arange", guarded_arange)
    monkeypatch.setattr(h5_dataloader.torch, "randperm", guarded_randperm)

    assert _update_sampler_subset(
        sampler,
        window_indices=[5, 25, 125, 625],
        completion_rates=[0.1, 0.2, 0.3, 0.4],
        swap_index=1,
    )
    batch_indices = sampler._sample_batch_indices(
        torch.Generator().manual_seed(0)
    )
    assert int(batch_indices.numel()) == 8


def test_ppo_logs_only_core_curriculum_metrics():
    algo = PPO.__new__(PPO)
    algo.actor_learning_rate = 1.0e-4
    algo.critic_learning_rate = 2.0e-4
    algo._last_update_metrics = {}
    algo.command_name = "ref_motion"
    algo._get_mean_policy_std = lambda: torch.tensor(0.0)

    cache = SimpleNamespace(
        swap_index=12,
        cache_curriculum_pool_statistics=lambda: {
            "prioritized_pool_size": 2.0,
            "prioritized_pool_mean_score": 0.8,
            "uniform_pool_mean_score": 0.1,
            "entered_prioritized_pool_count": 1.0,
            "exited_prioritized_pool_count": 1.0,
        },
    )
    motion_cmd = SimpleNamespace(_motion_cache=cache)
    algo.env = SimpleNamespace(
        _env=SimpleNamespace(
            command_manager=SimpleNamespace(
                get_term=lambda name: motion_cmd,
            )
        )
    )

    metrics = algo._get_additional_log_metrics()

    assert metrics["1-Perf/Cache/swap_index"] == 12.0
    assert metrics["1-Perf/Cache/prioritized_pool_size"] == 2.0
    assert metrics["1-Perf/Cache/prioritized_pool_mean_score"] == 0.8
    assert metrics["1-Perf/Cache/uniform_pool_mean_score"] == 0.1
    assert metrics["1-Perf/Cache/entered_prioritized_pool_count"] == 1.0
    assert metrics["1-Perf/Cache/exited_prioritized_pool_count"] == 1.0
    assert (
        "1-Perf/Cache/curriculum_probability_coefficient_of_variation"
        not in metrics
    )
    assert (
        "1-Perf/Cache/curriculum_max_probability_over_uniform" not in metrics
    )
    assert "1-Perf/Cache/uniform_floor_ratio" not in metrics


def test_cache_curriculum_dumps_on_scheduled_swap_even_without_state_update():
    cache = MotionClipBatchCache.__new__(MotionClipBatchCache)
    cache._datasets = {}
    cache._cache_curriculum_sampler = SimpleNamespace(
        maybe_update_from_observations=lambda **kwargs: False,
    )
    dumped_swaps = []
    cache._maybe_dump_cache_curriculum_scores_json = (
        lambda *, swap_index: dumped_swaps.append(swap_index)
    )

    updated = cache.update_cache_curriculum(
        window_indices=torch.tensor([0], dtype=torch.long),
        mpkpe_signal_means=torch.tensor([0.0], dtype=torch.float32),
        completion_rate_means=torch.tensor([0.0], dtype=torch.float32),
        counts=torch.tensor([1.0], dtype=torch.float32),
        swap_index=5,
    )

    assert updated is False
    assert dumped_swaps == [5]


def test_update_cache_curriculum_refreshes_prefetched_batch_when_state_changes():
    cache = MotionClipBatchCache.__new__(MotionClipBatchCache)
    cache._datasets = {}
    cache._cache_curriculum_sampler = SimpleNamespace(
        maybe_update_from_observations=lambda **kwargs: True,
    )
    cache._cache_curriculum_dump_enabled = False
    cache._next_batch = ClipBatch(
        tensors={},
        lengths=torch.tensor([1], dtype=torch.long),
        motion_keys=["stale"],
        raw_motion_keys=["stale"],
        window_indices=torch.tensor([0], dtype=torch.long),
        max_frame_length=1,
    )
    refreshed_batch = ClipBatch(
        tensors={},
        lengths=torch.tensor([1], dtype=torch.long),
        motion_keys=["fresh"],
        raw_motion_keys=["fresh"],
        window_indices=torch.tensor([1], dtype=torch.long),
        max_frame_length=1,
    )
    cache._fetch_next_batch = lambda: refreshed_batch

    updated = cache.update_cache_curriculum(
        window_indices=torch.tensor([0], dtype=torch.long),
        mpkpe_signal_means=torch.tensor([0.0], dtype=torch.float32),
        completion_rate_means=torch.tensor([0.0], dtype=torch.float32),
        counts=torch.tensor([1.0], dtype=torch.float32),
        swap_index=5,
    )

    assert updated is True
    assert cache._next_batch.motion_keys == ["fresh"]


def test_cache_curriculum_whole_window_dump_streams_rows_in_chunks(
    tmp_path,
):
    cache = MotionClipBatchCache.__new__(MotionClipBatchCache)
    cache._datasets = {
        "train": _FakeTrainDataset(
            [
                SimpleNamespace(
                    raw_motion_key=f"raw_{idx}",
                    motion_key=f"motion_{idx}",
                    start=idx,
                    length=idx + 1,
                )
                for idx in range(5)
            ]
        )
    }
    sampler = _ChunkLimitedSampler(max_query_size=2)
    cache._cache_curriculum_sampler = sampler
    cache._cache_curriculum_dump_enabled = True
    cache._cache_curriculum_dump_every_swaps = 1
    cache._cache_curriculum_dump_chunk_size = 2
    cache._cache_curriculum_last_dump_swap = -1
    cache._cache_curriculum_dump_dir = tmp_path
    cache._sampler_rank = 3

    cache._maybe_dump_cache_curriculum_scores_json(swap_index=1)

    output_path = tmp_path / "whole_window_scores_rank_0003_swap_000001.json"
    payload = json.loads(output_path.read_text())

    assert output_path.exists()
    assert payload["swap_index"] == 1
    assert payload["rank"] == 3
    assert payload["sampler_state_version"] == 7
    assert payload["num_windows"] == 5
    assert len(payload["rows"]) == 5
    assert payload["rows"][0]["window_index"] == 0
    assert payload["rows"][0]["selection_count"] == 10
    assert payload["rows"][-1]["window_index"] == 4
    assert payload["rows"][-1]["selection_count"] == 14
    assert "probability" not in payload["rows"][0]
    assert max(sampler.query_sizes) == 2
