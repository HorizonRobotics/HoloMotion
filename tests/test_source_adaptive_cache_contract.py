"""CPU-only contract checks for the H5 source-adaptive cache boundary."""

from types import SimpleNamespace

import pytest
import torch

from holomotion.src.algo.algo_base import BaseOnpolicyRL
from holomotion.src.training.h5_dataloader import MotionClipBatchCache
from holomotion.src.training.source_adaptive_sampling import (
    SourceAdaptiveInfiniteSampler,
)

SOURCE_PROBABILITIES = {"alpha": 0.2, "beta": 0.3, "gamma": 0.5}
ADAPTIVE_CONFIG = {"enabled": True, "min_observations": 1}


class SyntheticDataset:
    max_frame_length = 300

    def __init__(
        self,
        *,
        changed_start=False,
        shard_path="/tmp/source_adaptive_data/train/shard_000.h5",
    ):
        self._manifest_paths = [
            "/tmp/source_adaptive_data/train/manifest.json"
        ]
        self._shard_paths = [shard_path]
        source_by_key = {
            "alpha_one": "alpha",
            "alpha_two": "alpha",
            "beta_one": "beta",
            "gamma_one": "gamma",
        }
        self.clips = {
            key: {"metadata": {"source_id": source}}
            for key, source in source_by_key.items()
        }
        keys = (
            "alpha_one",
            "beta_one",
            "alpha_two",
            "gamma_one",
            "gamma_one",
        )
        self.windows = [
            SimpleNamespace(
                raw_motion_key=key,
                shard_index=0,
                start=index * 10 + (1 if changed_start and index == 2 else 0),
                length=10,
            )
            for index, key in enumerate(keys)
        ]

    def __len__(self):
        return len(self.windows)

    def set_progress_counter(self, counter):
        pass

    def close(self):
        pass


@pytest.fixture
def cache_factory(monkeypatch):
    # Keep DataLoader/sampler construction real; only skip fetching HDF5 data.
    monkeypatch.setattr(
        MotionClipBatchCache, "_prime_buffers", lambda self: None
    )

    def create(*, dataset=None, probabilities=None):
        return MotionClipBatchCache(
            dataset if dataset is not None else SyntheticDataset(),
            batch_size=4,
            stage_device=None,
            num_workers=0,
            pin_memory=False,
            seed=7,
        )

    return create


def _enable(cache, probabilities=None):
    cache.enable_source_adaptive_sampling(
        source_probabilities=probabilities or SOURCE_PROBABILITIES,
        source_adaptive_cfg=ADAPTIVE_CONFIG,
    )


def _algo_with_cache(cache):
    algorithm = BaseOnpolicyRL.__new__(BaseOnpolicyRL)
    algorithm.command_name = "ref_motion"
    algorithm.is_offline_eval = False
    algorithm.config = {}
    command = SimpleNamespace(_motion_cache=cache)
    manager = SimpleNamespace(get_term=lambda name: command)
    algorithm.env = SimpleNamespace(
        _env=SimpleNamespace(command_manager=manager)
    )
    return algorithm


def test_adaptive_cache_selects_controller_and_maps_current_clip_sources(
    cache_factory,
):
    cache = cache_factory()
    _enable(cache)

    assert cache.source_sampling_enabled
    assert cache.adaptive_sampling_enabled
    assert cache.adaptive_sampling_kind == "source_adaptive"
    assert isinstance(cache._sampler, SourceAdaptiveInfiniteSampler)
    assert cache._sampler._controller is cache.adaptive_controller
    assert cache.source_names == ("alpha", "beta", "gamma")
    assert cache.source_base_probabilities().tolist() == pytest.approx(
        [0.2, 0.3, 0.5]
    )
    assert cache.source_upper_probabilities(2.0).tolist() == pytest.approx(
        [0.4, 0.6, 1.0]
    )
    cache._current_batch = SimpleNamespace(
        window_indices=torch.tensor([4, 2, 1, 3], dtype=torch.long)
    )
    assert cache.source_indices_for_indices(
        torch.tensor([0, 1, 2], dtype=torch.long)
    ).tolist() == [2, 0, 1]
    metrics = cache.source_sampling_metrics()
    assert metrics["source_adaptive_enabled"] == 1.0
    assert metrics["adaptation_enabled"] == 1.0


def test_adaptive_checkpoint_identity_is_strict(cache_factory):
    original = cache_factory()
    _enable(original)
    state = original.adaptive_sampling_state_dict()

    same = cache_factory()
    _enable(same)
    same.load_adaptive_sampling_state_dict(state)

    changed_probability = cache_factory()
    _enable(changed_probability, {"alpha": 0.3, "beta": 0.2, "gamma": 0.5})
    with pytest.raises(ValueError, match="identity mismatch"):
        changed_probability.load_adaptive_sampling_state_dict(state)

    changed_window = cache_factory(
        dataset=SyntheticDataset(changed_start=True)
    )
    _enable(changed_window)
    with pytest.raises(ValueError, match="identity mismatch"):
        changed_window.load_adaptive_sampling_state_dict(state)


def test_identical_windows_with_different_shard_path_reject_resume(
    cache_factory,
):
    original_dataset = SyntheticDataset()
    moved_dataset = SyntheticDataset(
        shard_path="/tmp/other_source_adaptive_data/train/shard_000.h5"
    )
    assert original_dataset.windows == moved_dataset.windows

    original = cache_factory(dataset=original_dataset)
    moved = cache_factory(dataset=moved_dataset)
    _enable(original)
    _enable(moved)
    state = original.adaptive_sampling_state_dict()
    assert (
        state["identity"] != moved.adaptive_sampling_state_dict()["identity"]
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        moved.load_adaptive_sampling_state_dict(state)


def test_fixed_cache_remains_nonadaptive_and_rejects_s1_state(cache_factory):
    adaptive = cache_factory()
    _enable(adaptive)
    state = adaptive.adaptive_sampling_state_dict()

    fixed = cache_factory()
    fixed.enable_source_fixed_sampling(
        source_probabilities=SOURCE_PROBABILITIES
    )
    assert fixed.source_sampling_enabled
    assert not fixed.adaptive_sampling_enabled
    assert fixed.adaptive_sampling_state_dict() is None
    assert fixed.source_sampling_metrics()["source_adaptive_enabled"] == 0.0
    with pytest.raises(ValueError, match="disabled"):
        fixed.load_adaptive_sampling_state_dict(state)


def test_training_resume_requires_matching_s1_state(cache_factory):
    cache = cache_factory()
    _enable(cache)
    algorithm = _algo_with_cache(cache)
    saved = algorithm._extra_checkpoint_state()
    assert "source_adaptive_sampling" in saved
    algorithm._load_extra_checkpoint_state(saved)

    with pytest.raises(ValueError, match="requires matching checkpoint state"):
        algorithm._load_extra_checkpoint_state({})
    with pytest.raises(ValueError, match="Legacy weighted-bin"):
        algorithm._load_extra_checkpoint_state(
            {"adaptive_weighted_bin": {}, **saved}
        )

    algorithm.config = {"checkpoint_load_mode": "finetune"}
    algorithm._load_extra_checkpoint_state({})
    algorithm.config = {}
    algorithm.is_offline_eval = True
    algorithm._load_extra_checkpoint_state({})


def test_fixed_training_resume_rejects_adaptive_checkpoint(cache_factory):
    fixed = cache_factory()
    fixed.enable_source_fixed_sampling(
        source_probabilities=SOURCE_PROBABILITIES
    )
    algorithm = _algo_with_cache(fixed)
    with pytest.raises(ValueError, match="disables it"):
        algorithm._load_extra_checkpoint_state(
            {"source_adaptive_sampling": {}}
        )
