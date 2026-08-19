import pytest
import torch

from holomotion.src.utils.g1_variants import (
    sample_correlated_hand_payloads,
)


def test_correlated_payload_never_exceeds_total_object_limit():
    torch.manual_seed(20260815)
    added, object_by_hand = sample_correlated_hand_payloads(
        20000,
        object_mass_range=(0.0, 2.5),
        hardware_mass_range=(0.0, 0.4),
        mode_probabilities=(0.1, 0.3, 0.3, 0.3),
        bimanual_split_range=(0.3, 0.7),
    )

    assert added.shape == (20000, 2)
    assert object_by_hand.shape == (20000, 2)
    assert float(object_by_hand.min()) >= 0.0
    assert float(object_by_hand.sum(dim=1).max()) <= 2.5 + 1.0e-6
    assert float((added - object_by_hand).min()) >= 0.0
    assert float((added - object_by_hand).max()) <= 0.4 + 1.0e-6


def test_bimanual_payload_splits_one_total_mass():
    torch.manual_seed(7)
    _, object_by_hand = sample_correlated_hand_payloads(
        1000,
        object_mass_range=(2.5, 2.5),
        mode_probabilities=(0.0, 0.0, 0.0, 1.0),
        bimanual_split_range=(0.3, 0.7),
    )

    torch.testing.assert_close(
        object_by_hand.sum(dim=1), torch.full((1000,), 2.5)
    )
    assert float(object_by_hand.min()) >= 0.75 - 1.0e-6
    assert float(object_by_hand.max()) <= 1.75 + 1.0e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"object_mass_range": (-0.1, 2.5)},
        {"object_mass_range": (2.5, 1.0)},
        {
            "object_mass_range": (0.0, 2.5),
            "mode_probabilities": (0.0, 0.0, 0.0, 0.0),
        },
        {
            "object_mass_range": (0.0, 2.5),
            "bimanual_split_range": (-0.1, 1.1),
        },
    ],
)
def test_correlated_payload_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        sample_correlated_hand_payloads(4, **kwargs)
