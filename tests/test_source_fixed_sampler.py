"""CPU checks for fixed source budgets and window reachability."""

from collections import Counter
from itertools import islice

import pytest
import torch

from holomotion.src.training.source_adaptive_sampling import (
    FixedSourceWeights,
    SourceAdaptiveInfiniteSampler,
)


def _controller(
    *,
    size=8,
    groups=((0, 1), (2, 3, 4, 5, 6, 7)),
    probabilities=(0.7, 0.3),
):
    return FixedSourceWeights(
        size=size,
        source_names=("source_a", "source_b"),
        source_window_indices=groups,
        base_probabilities=probabilities,
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        (1.0,),
        (0.0, 1.0),
        (-0.1, 1.1),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (True, True),
    ],
)
def test_rejects_invalid_source_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilit"):
        _controller(probabilities=probabilities)


def test_positive_source_weights_are_normalized():
    controller = _controller(probabilities=(7.0, 3.0))

    assert torch.allclose(
        controller.source_probabilities,
        torch.tensor((0.7, 0.3), dtype=torch.float64),
    )
    assert controller.metrics()["adaptation_enabled"] == 0.0


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (((0, 1), (2, 3, 4, 5, 6)), "cover"),
        (((0, 1, 2), (2, 3, 4, 5, 6, 7)), "disjoint"),
        (((0, 0, 1), (2, 3, 4, 5, 6, 7)), "duplicate"),
        (((0, 1), (2, 3, 4, 5, 6, 8)), "invalid"),
        (((), tuple(range(8))), "nonempty"),
    ],
)
def test_source_groups_must_partition_all_windows(groups, message):
    with pytest.raises(ValueError, match=message):
        _controller(groups=groups)


def test_fixed_budget_draws_keep_every_window_reachable():
    controller = _controller()
    sampler = SourceAdaptiveInfiniteSampler(
        source_window_indices=controller.source_window_indices,
        controller=controller,
        batch_size=64,
        seed=17,
    )
    draws = list(islice(iter(sampler), 64 * 300))
    counts = Counter(draws)
    source_a_count = counts[0] + counts[1]

    assert set(counts) == set(range(8))
    assert len(set(draws[:64])) < 64  # Draws are with replacement.
    assert abs(source_a_count / len(draws) - 0.7) < 0.02
    assert abs(counts[0] / source_a_count - 0.5) < 0.02
    for window in range(2, 8):
        assert (
            abs(counts[window] / (len(draws) - source_a_count) - 1 / 6) < 0.03
        )
    assert controller.metrics()[
        "source/00_source_a/rank_local_dispatched_count"
    ] == float(source_a_count)


def test_rank_seed_streams_are_independent_and_repeatable():
    def draw_for_rank(rank):
        controller = _controller()
        sampler = SourceAdaptiveInfiniteSampler(
            source_window_indices=controller.source_window_indices,
            controller=controller,
            batch_size=32,
            seed=41 + rank * 100003,
        )
        return list(islice(iter(sampler), 32 * 8))

    rank_zero = draw_for_rank(0)
    rank_one = draw_for_rank(1)

    assert rank_zero == draw_for_rank(0)
    assert rank_one == draw_for_rank(1)
    assert rank_zero != rank_one


def test_metric_tags_remain_distinct_when_source_names_sanitize_equally():
    controller = FixedSourceWeights(
        size=2,
        source_names=("a/b", "a_b"),
        source_window_indices=((0,), (1,)),
        base_probabilities=(0.4, 0.6),
    )
    metrics = controller.metrics()
    assert metrics["source/00_a_b/baseline_probability"] == pytest.approx(0.4)
    assert metrics["source/01_a_b/baseline_probability"] == pytest.approx(0.6)
