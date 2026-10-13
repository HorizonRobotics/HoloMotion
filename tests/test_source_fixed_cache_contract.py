"""CPU contract checks for fixed source sampling at the cache boundary."""

from types import SimpleNamespace

import pytest

from holomotion.src.training.h5_dataloader import (
    MotionClipBatchCache,
    _configure_source_baseline,
)

SOURCE_PROBABILITIES = {
    "alpha": 0.2,
    "beta": 0.3,
    "gamma": 0.5,
}


class SyntheticDataset:
    """Metadata and windows only; no manifest, HDF5 payload, or GPU."""

    max_frame_length = 300

    def __init__(self):
        sources = {
            "alpha_one": "alpha",
            "alpha_two": "alpha",
            "beta_one": "beta",
            "beta_two": "beta",
            "gamma_one": "gamma",
        }
        self.clips = {
            key: {"metadata": {"source_id": source}}
            for key, source in sources.items()
        }
        self.windows = [
            SimpleNamespace(raw_motion_key=key)
            for key in (
                "alpha_one",
                "beta_one",
                "alpha_two",
                "gamma_one",
                "beta_two",
                "gamma_one",
            )
        ]

    def __len__(self):
        return len(self.windows)

    def close(self):
        pass


def _indices(values):
    return [int(value) for value in values]


def test_source_groups_cover_every_window_and_preserve_clip_metadata():
    dataset = SyntheticDataset()

    names, groups, probabilities = _configure_source_baseline(
        dataset, SOURCE_PROBABILITIES
    )

    assert set(names) == set(SOURCE_PROBABILITIES)
    assert len(groups) == len(probabilities) == len(names)
    assert {
        name: _indices(group)
        for name, group in zip(names, groups, strict=True)
    } == {"alpha": [0, 2], "beta": [1, 4], "gamma": [3, 5]}
    observed = dict(zip(names, probabilities, strict=True))
    assert observed == pytest.approx(SOURCE_PROBABILITIES)
    grouped_indices = [index for group in groups for index in _indices(group)]
    assert sorted(grouped_indices) == list(range(len(dataset)))
    assert {
        dataset.windows[index].raw_motion_key
        for group in groups
        for index in _indices(group)
    } == set(dataset.clips)


@pytest.mark.parametrize(
    "configured",
    [
        {"alpha": 0.4, "beta": 0.6},
        {"alpha": 0.2, "beta": 0.3, "gamma": 0.4, "delta": 0.1},
    ],
)
def test_source_probabilities_require_exact_source_cover(configured):
    with pytest.raises(ValueError):
        _configure_source_baseline(SyntheticDataset(), configured)


@pytest.mark.parametrize(
    "bad_probability", [0.0, -0.1, float("nan"), float("inf"), True]
)
def test_source_probabilities_must_be_positive_and_finite(bad_probability):
    configured = {**SOURCE_PROBABILITIES, "alpha": bad_probability}

    with pytest.raises(ValueError):
        _configure_source_baseline(SyntheticDataset(), configured)


@pytest.mark.parametrize("metadata", [None, {}, {"source_id": ""}])
def test_every_window_requires_nonempty_source_metadata(metadata):
    dataset = SyntheticDataset()
    dataset.clips["beta_two"]["metadata"] = metadata

    with pytest.raises(ValueError):
        _configure_source_baseline(dataset, SOURCE_PROBABILITIES)


@pytest.fixture
def cache_factory(monkeypatch):
    # Constructor and sampler reconfiguration can be tested without loading a
    # ClipBatch, opening a shard, or initializing a DataLoader worker.
    monkeypatch.setattr(
        MotionClipBatchCache, "_build_dataloader", lambda self: None
    )
    monkeypatch.setattr(
        MotionClipBatchCache, "_prime_buffers", lambda self: None
    )

    def create():
        return MotionClipBatchCache(
            SyntheticDataset(),
            batch_size=4,
            stage_device=None,
            num_workers=0,
            pin_memory=False,
            seed=7,
        )

    return create


def test_fixed_source_can_be_enabled_without_adaptation(cache_factory):
    cache = cache_factory()
    assert not cache.source_sampling_enabled
    assert cache.source_sampling_metrics() is None

    cache.enable_source_fixed_sampling(
        source_probabilities=SOURCE_PROBABILITIES
    )

    assert cache.source_sampling_enabled
    metrics = cache.source_sampling_metrics()
    assert isinstance(metrics, dict)
    assert metrics


@pytest.mark.parametrize(
    "existing_flag", ["_weighted_bin_enabled", "_cache_curriculum_enabled"]
)
def test_fixed_source_rejects_existing_legacy_sampler(
    cache_factory, existing_flag
):
    cache = cache_factory()
    setattr(cache, existing_flag, True)

    with pytest.raises(ValueError):
        cache.enable_source_fixed_sampling(
            source_probabilities=SOURCE_PROBABILITIES
        )

    assert not cache.source_sampling_enabled


@pytest.mark.parametrize("legacy", ["weighted_bin", "curriculum"])
def test_legacy_sampler_rejects_enabled_fixed_source(cache_factory, legacy):
    cache = cache_factory()
    cache.enable_source_fixed_sampling(
        source_probabilities=SOURCE_PROBABILITIES
    )

    with pytest.raises(ValueError):
        if legacy == "weighted_bin":
            cache.enable_weighted_bin_sampling(
                {"bin_regex_patterns": [{"regex": ".*", "ratio": 1.0}]}
            )
        else:
            cache.enable_cache_curriculum_sampling({})

    assert cache.source_sampling_enabled
    assert not cache._weighted_bin_enabled
    assert not cache._cache_curriculum_enabled
