"""CPU contracts for opt-in actor-only link command feature corruption."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from holomotion.src.motion_tracking.link_feature_noise import (
    FEATURES,
    FeatureNoiseLinkAdapter,
    add_link_feature_noise,
    validate_feature_noise,
)


@dataclass(frozen=True)
class _Observation:
    effective_mask: torch.Tensor
    target_pos: torch.Tensor
    position_error: torch.Tensor
    target_rot6d: torch.Tensor
    rotation_error6d: torch.Tensor


def _observation() -> _Observation:
    mask = torch.tensor([[True, False], [True, True]])
    return _Observation(
        effective_mask=mask,
        target_pos=torch.ones(2, 2, 3),
        position_error=torch.ones(2, 2, 3),
        target_rot6d=torch.ones(2, 2, 6),
        rotation_error6d=torch.ones(2, 2, 6),
    )


def _noise_config(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "distribution": "uniform",
        "features": {name: [-0.05, 0.05] for name in FEATURES},
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_noise_returns_same_object_without_rng_draw(enabled):
    observation = _observation()
    torch.manual_seed(29)
    before = torch.random.get_rng_state().clone()
    actual = add_link_feature_noise(
        observation, _noise_config(enabled), enabled=False
    )
    assert actual is observation
    assert torch.equal(torch.random.get_rng_state(), before)


def test_noise_changes_valid_actor_features_without_in_place_write():
    observation = _observation()
    originals = {name: getattr(observation, name).clone() for name in FEATURES}
    torch.manual_seed(31)
    noisy = add_link_feature_noise(observation, _noise_config())

    assert noisy is not observation
    for name in FEATURES:
        original = originals[name]
        actual = getattr(noisy, name)
        assert torch.equal(getattr(observation, name), original)
        assert torch.count_nonzero(actual[~observation.effective_mask]) == 0
        delta = (
            actual[observation.effective_mask]
            - original[observation.effective_mask]
        )
        assert torch.all(delta >= -0.050001)
        assert torch.all(delta <= 0.050001)
        assert torch.count_nonzero(delta) > 0


def test_snapshot_reuses_one_actor_draw_and_keeps_critic_clean():
    actor = _observation()
    critic = _observation()
    clean = {"actor": actor, "critic": critic}
    command = SimpleNamespace(
        _reference_observation_cache={},
        _is_evaluating=False,
        _env=SimpleNamespace(
            cfg=SimpleNamespace(
                observations=SimpleNamespace(
                    unified=SimpleNamespace(enable_corruption=True)
                )
            )
        ),
    )
    base = SimpleNamespace(command=command, snapshot=lambda: clean)
    adapter = FeatureNoiseLinkAdapter(base, _noise_config())

    torch.manual_seed(37)
    first = adapter.snapshot()
    after_first = torch.random.get_rng_state().clone()
    second = adapter.snapshot()
    assert second is first
    assert torch.equal(torch.random.get_rng_state(), after_first)
    assert first["critic"] is critic
    assert first["actor"] is not actor

    command._is_evaluating = True
    evaluation = adapter.snapshot()
    assert evaluation["actor"] is actor
    assert evaluation["critic"] is critic
    assert torch.equal(torch.random.get_rng_state(), after_first)


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": 1},
        {"enabled": True, "distribution": "gaussian"},
        {"enabled": True, "features": {"critic_feature": [-0.1, 0.1]}},
        {"enabled": True, "features": {"target_pos": [0.1, -0.1]}},
        {"enabled": True, "features": {"target_pos": [0.0, float("inf")]}},
    ],
)
def test_reject_invalid_noise_configuration(config):
    with pytest.raises((TypeError, ValueError)):
        validate_feature_noise(config)
