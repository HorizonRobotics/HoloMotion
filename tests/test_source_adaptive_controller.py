"""CPU contracts for opt-in adaptive source and window sampling."""

from collections import Counter
from copy import deepcopy
from itertools import islice

import pytest
import torch

from holomotion.src.training.source_adaptive_sampling import (
    SourceAdaptiveConfig,
    SourceAdaptiveInfiniteSampler,
    SourceAdaptiveWeights,
    project_bounded_probabilities,
)


def _config(**overrides):
    settings = {
        "lower_multiplier": 0.5,
        "upper_multiplier": 1.5,
        "difficulty_ema_alpha": 1.0,
        "difficulty_scale_m": 1.0,
        "min_observations": 1,
        "update_interval_swaps": 1,
        "warmup_swaps": 0,
    }
    settings.update(overrides)
    return SourceAdaptiveConfig(**settings)


def _controller(
    *, config=None, groups=((0, 1), (2, 3)), probabilities=(0.5, 0.5)
):
    return SourceAdaptiveWeights(
        size=4,
        source_names=("a/b", "a_b"),
        source_window_indices=groups,
        base_probabilities=probabilities,
        config=config or _config(),
        identity="0" * 64,
    )


def _observe_easy_and_hard(controller):
    return controller.observe(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([1.0, 1.0, 0.1, 0.1]),
        torch.ones(4),
    )


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, right[key]), key
        else:
            assert value == right[key], key


def test_adaptation_requires_explicit_enable_and_valid_settings():
    assert SourceAdaptiveConfig.parse(None) is None
    assert SourceAdaptiveConfig.parse({}) is None
    assert SourceAdaptiveConfig.parse({"enabled": False}) is None
    assert SourceAdaptiveConfig.parse({"enabled": True}).enabled

    for invalid in (
        {"enabled": 1},
        {"enabled": True, "unknown": 1},
        {"enabled": True, "group_by": "clip"},
        {"enabled": True, "lower_multiplier": -0.1},
        {"enabled": True, "upper_multiplier": 0.9},
        {"enabled": True, "min_window_weight": 0.0},
        {"enabled": True, "difficulty_scale_m": float("nan")},
        {"enabled": True, "update_interval_swaps": True},
    ):
        with pytest.raises(ValueError):
            SourceAdaptiveConfig.parse(invalid)


def test_bounded_projection_and_invalid_inputs():
    projected = project_bounded_probabilities(
        torch.tensor([10.0, 1.0]),
        torch.tensor([0.25, 0.25]),
        torch.tensor([0.75, 0.75]),
    )
    assert projected.dtype == torch.float64
    assert projected.tolist() == pytest.approx([0.75, 0.25])

    for raw, lower, upper in (
        ([float("nan"), 1.0], [0.0, 0.0], [1.0, 1.0]),
        ([0.0, 1.0], [0.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0], [0.6, 0.6], [1.0, 1.0]),
        ([1.0, 1.0], [0.0, 0.0], [0.4, 0.4]),
    ):
        with pytest.raises(ValueError):
            project_bounded_probabilities(
                torch.tensor(raw), torch.tensor(lower), torch.tensor(upper)
            )


@pytest.mark.parametrize(
    "groups",
    [((0, 1), (2,)), ((0, 1), (1, 2, 3)), ((0, 0), (1, 2, 3))],
)
def test_adaptation_preserves_strict_exact_cover(groups):
    with pytest.raises(ValueError):
        _controller(groups=groups)


@pytest.mark.parametrize(
    "probabilities", [(True, True), (-1.0, 2.0), (0.0, 1.0)]
)
def test_adaptation_rejects_invalid_baseline_weights(probabilities):
    with pytest.raises(ValueError, match="probabilit"):
        _controller(probabilities=probabilities)


def test_observation_updates_bounded_sources_windows_and_indexed_metrics():
    controller = _controller()
    assert _observe_easy_and_hard(controller)
    assert controller.version == 1
    assert controller.source_probabilities.tolist() == pytest.approx(
        [0.75, 0.25]
    )
    assert controller.window_weights.tolist() == pytest.approx(
        [1.0, 1.0, 0.145, 0.145]
    )

    controller.record_selected_sources(torch.tensor([0, 1, 1]))
    metrics = controller.metrics()
    assert metrics["adaptation_enabled"] == 1.0
    assert metrics["source/00_a_b/rank_local_dispatched_count"] == 1.0
    assert metrics["source/01_a_b/rank_local_dispatched_count"] == 2.0
    assert metrics["source/00_a_b/target_probability"] == pytest.approx(0.75)
    assert metrics["source/01_a_b/target_probability"] == pytest.approx(0.25)


def test_warmup_and_minimum_observation_accumulation():
    controller = _controller(
        config=_config(
            warmup_swaps=2, update_interval_swaps=1, min_observations=2
        )
    )
    assert not _observe_easy_and_hard(controller)
    assert controller.version == 0
    assert _observe_easy_and_hard(controller)
    assert controller.version == 1
    assert controller.metrics()["pending_windows"] == 0.0
    assert controller.metrics()["last_updated_sources"] == 2.0


def test_adaptive_sampler_draws_with_bounded_source_and_window_bias():
    controller = _controller()
    _observe_easy_and_hard(controller)
    sampler = SourceAdaptiveInfiniteSampler(
        source_window_indices=controller.source_window_indices,
        controller=controller,
        batch_size=64,
        seed=17,
    )
    draws = list(islice(iter(sampler), 64 * 200))
    counts = Counter(draws)
    first_source = counts[0] + counts[1]
    assert set(counts) == {0, 1, 2, 3}
    assert abs(first_source / len(draws) - 0.75) < 0.025
    assert abs(counts[2] / (len(draws) - first_source) - 0.5) < 0.04
    assert controller.metrics()[
        "source/00_a_b/rank_local_dispatched_count"
    ] == float(first_source)


def test_invalid_observation_does_not_mutate_state():
    controller = _controller()
    before = controller.state_dict()
    with pytest.raises(ValueError):
        controller.observe(
            torch.tensor([4]), torch.tensor([1.0]), torch.tensor([1.0])
        )
    _assert_state_equal(controller.state_dict(), before)
    with pytest.raises(ValueError):
        controller.observe(
            torch.tensor([0]), torch.tensor([1.0]), torch.tensor([0.0])
        )
    _assert_state_equal(controller.state_dict(), before)


def test_checkpoint_round_trip_and_identity_guard():
    original = _controller()
    _observe_easy_and_hard(original)
    original.record_selected_sources(torch.tensor([0, 1, 1]))
    state = original.state_dict()
    restored = _controller()
    restored.load_state_dict(state)
    _assert_state_equal(restored.state_dict(), state)

    wrong_identity = deepcopy(state)
    wrong_identity["identity"] = "1" * 64
    with pytest.raises(ValueError, match="identity"):
        restored.load_state_dict(wrong_identity)
    _assert_state_equal(restored.state_dict(), state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_weights", -0.1),
        ("window_weights", 0.0),
        ("window_weights", 1.1),
        ("window_weights", float("nan")),
        ("window_ema_error", 1.1),
        ("source_probabilities", 0.0),
        ("source_pending_counts", -1.0),
    ],
)
def test_bad_checkpoint_tensor_is_rejected_atomically(field, value):
    controller = _controller()
    _observe_easy_and_hard(controller)
    before = controller.state_dict()
    corrupted = deepcopy(before)
    corrupted[field][0] = value
    with pytest.raises(ValueError):
        controller.load_state_dict(corrupted)
    _assert_state_equal(controller.state_dict(), before)


def test_negative_pending_window_is_rejected_atomically():
    controller = _controller(config=_config(min_observations=2))
    _observe_easy_and_hard(controller)
    before = controller.state_dict()
    corrupted = deepcopy(before)
    corrupted["pending"][0] = [-1.0, 1.0]
    with pytest.raises(ValueError):
        controller.load_state_dict(corrupted)
    _assert_state_equal(controller.state_dict(), before)


@pytest.mark.parametrize(
    "missing", ["window_weights", "pending", "round", "observed_samples"]
)
def test_incomplete_checkpoint_is_rejected_atomically(missing):
    controller = _controller()
    before = controller.state_dict()
    corrupted = deepcopy(before)
    corrupted.pop(missing)
    with pytest.raises(ValueError):
        controller.load_state_dict(corrupted)
    _assert_state_equal(controller.state_dict(), before)


def test_malformed_pending_window_is_rejected_atomically():
    controller = _controller()
    before = controller.state_dict()
    corrupted = deepcopy(before)
    corrupted["pending"] = {0: "bad"}
    with pytest.raises(ValueError):
        controller.load_state_dict(corrupted)
    _assert_state_equal(controller.state_dict(), before)


def test_tiny_window_weight_rejects_nearly_zero_checkpoint_value():
    controller = _controller(config=_config(min_window_weight=1e-8))
    before = controller.state_dict()
    corrupted = deepcopy(before)
    corrupted["window_weights"][0] = 1e-10
    with pytest.raises(ValueError, match="window weights"):
        controller.load_state_dict(corrupted)
    _assert_state_equal(controller.state_dict(), before)


def test_duplicate_cache_rows_are_aggregated_by_dataset_window_index():
    controller = _controller(config=_config(min_observations=2))
    assert controller.observe(
        torch.tensor([0, 0, 2, 2]),
        torch.tensor([0.5, 0.5, 0.1, 0.1]),
        torch.ones(4),
    )
    assert controller.round == 1
    assert controller.window_weights[0].item() == pytest.approx(0.525)
    assert controller.window_weights[2].item() == pytest.approx(0.145)
    assert controller.metrics()["last_updated_windows"] == 2.0


def test_within_source_bias_preserves_uniform_reachability():
    controller = _controller()
    assert controller.observe(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([1.0, 0.0, 1.0, 1.0]),
        torch.ones(4),
    )
    draws = controller.choose(
        torch.tensor([0, 1]), 6000, torch.Generator().manual_seed(17)
    )
    counts = Counter(draws)
    assert set(counts) == {0, 1}
    assert 0.82 < counts[0] / len(draws) < 0.9
