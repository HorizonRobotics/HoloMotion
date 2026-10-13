from types import SimpleNamespace

import pytest
import torch

from holomotion.src.motion_tracking.link_mode_mask import (
    ModeMaskLinkAdapter,
    ModeMaskSampler,
)

LINKS = ("pelvis", "left_wrist", "right_wrist")
MODES = {
    "Pelvis-1": ["pelvis"],
    "LeftWrist-1": ["left_wrist"],
    "WholeBody-3": list(LINKS),
}


def config(weights=None):
    return {
        "strategy": "modes",
        "mode_candidates": MODES,
        "mode_weights": weights,
    }


class FakeLinkAdapter:
    def __init__(self, num_envs=4):
        self.link_names = LINKS
        self.command = SimpleNamespace(
            num_envs=num_envs, device=torch.device("cpu")
        )
        self.link_mask = torch.ones(num_envs, len(LINKS))
        self.invalidations = 0
        self.delegated_property = "base adapter"

    def invalidate(self):
        self.invalidations += 1

    def set_link_mask(self, mask, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.command.num_envs)
        value = torch.as_tensor(mask, dtype=torch.float32).expand(
            len(env_ids), len(LINKS)
        )
        self.link_mask[env_ids] = value
        self.invalidate()


def test_named_modes_compile_to_ordered_binary_masks_and_sample_rows():
    sampler = ModeMaskSampler(LINKS, config())
    assert sampler.mode_names == tuple(MODES)
    torch.testing.assert_close(
        sampler.table,
        torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.float32),
    )
    torch.testing.assert_close(sampler.probabilities, torch.full((3,), 1 / 3))
    assert sampler.description()["sampling"] == "uniform_mode"

    indices, masks = sampler.sample(32)
    assert indices.shape == (32,)
    torch.testing.assert_close(masks, sampler.table[indices])
    masks[0, 0] = -1
    assert (
        sampler.table[indices[0], 0] >= 0
    )  # Samples cannot mutate the table.
    empty_indices, empty_masks = sampler.sample(0)
    assert empty_indices.shape == (0,)
    assert empty_masks.shape == (0, len(LINKS))


def test_weighted_modes_respect_zero_weight_and_validate_mode_ids():
    sampler = ModeMaskSampler(
        LINKS,
        config({"Pelvis-1": 0, "LeftWrist-1": 4, "WholeBody-3": 0}),
    )
    indices, masks = sampler.sample(64)
    assert indices.tolist() == [1] * 64
    torch.testing.assert_close(masks, sampler.table[1].expand(64, -1))
    assert sampler.description()["sampling"] == "weighted_mode"
    assert sampler.mode_index("WholeBody-3") == 2
    assert sampler.mode_index(0) == 0
    with pytest.raises(ValueError, match="Unknown command mode"):
        sampler.mode_index("missing")
    with pytest.raises(ValueError, match="outside the configured table"):
        sampler.mode_index(3)
    with pytest.raises(TypeError, match="configured name or integer"):
        sampler.mode_index(True)


@pytest.mark.parametrize(
    ("links", "settings", "error"),
    [
        (("pelvis", "pelvis"), config(), "nonempty and unique"),
        (LINKS, {**config(), "strategy": "random"}, "mask.strategy=modes"),
        (LINKS, config() | {"mode_candidates": {}}, "nonempty name-to-links"),
        (
            LINKS,
            config() | {"mode_candidates": {"bad": ["pelvis", "pelvis"]}},
            "nonempty unique list",
        ),
        (
            LINKS,
            config() | {"mode_candidates": {"bad": ["unknown"]}},
            "unknown command links",
        ),
        (
            LINKS,
            config({"Pelvis-1": 1}),
            "name every configured mode exactly once",
        ),
        (
            LINKS,
            config(
                {"Pelvis-1": 1, "LeftWrist-1": float("nan"), "WholeBody-3": 1}
            ),
            "finite, nonnegative",
        ),
    ],
)
def test_invalid_mode_tables_are_rejected(links, settings, error):
    with pytest.raises(ValueError, match=error):
        ModeMaskSampler(links, settings)


def test_adapter_resamples_selected_envs_and_preserves_controls():
    base = FakeLinkAdapter()
    adapter = ModeMaskLinkAdapter(
        base,
        config({"Pelvis-1": 0, "LeftWrist-1": 1, "WholeBody-3": 0}),
    )
    assert adapter.delegated_property == "base adapter"
    assert base.invalidations == 1
    torch.testing.assert_close(
        base.link_mask, adapter.sampler.table[1].expand(4, -1)
    )
    assert adapter.sampled_mode_ids.tolist() == [1] * 4

    adapter.set_mode("Pelvis-1", env_ids=[1, 3])
    torch.testing.assert_close(
        base.link_mask[[1, 3]], adapter.sampler.table[0].expand(2, -1)
    )
    torch.testing.assert_close(
        base.link_mask[[0, 2]], adapter.sampler.table[1].expand(2, -1)
    )
    assert adapter.sampled_mode_ids.tolist() == [1, 0, 1, 0]

    adapter.set_link_mask([1, 0, 1], env_ids=[0])
    adapter.set_link_mask(adapter.sampler.table[2], env_ids=[2])
    assert adapter.sampled_mode_ids.tolist() == [-1, 0, 2, 0]
    assert adapter.statistics() == {
        "mode_names": list(MODES),
        "mode_counts": [2, 0, 1],
        "custom_count": 1,
        "num_envs": 4,
        "resample_calls": 1,
    }

    before = base.link_mask.clone()
    adapter.resample_mask([])
    assert base.invalidations == 4
    adapter.resample_mask([1, 3])
    torch.testing.assert_close(base.link_mask[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(
        base.link_mask[[1, 3]], adapter.sampler.table[1].expand(2, -1)
    )
    assert adapter.sampled_mode_ids.tolist() == [-1, 1, 2, 1]
    assert adapter.statistics()["resample_calls"] == 2
    assert base.invalidations == 5

    with pytest.raises(ValueError, match="Unknown command mode"):
        adapter.set_mode("missing")
    with pytest.raises(ValueError, match="one-dimensional"):
        adapter.resample_mask([[0, 1]])


@pytest.mark.parametrize(
    "active", [["pelvis"], ["right_wrist", "pelvis"], list(LINKS)]
)
def test_training_modes_share_fixed_control_mask_resolution(active):
    from holomotion.src.motion_tracking.link_mode_mask import (
        resolve_control_links,
    )

    names, mask = resolve_control_links(
        LINKS, active_link_names=active, mask_mode="custom"
    )
    sampler = ModeMaskSampler(
        LINKS, {"strategy": "modes", "mode_candidates": {"fixed": active}}
    )
    assert names == [name for name in LINKS if name in active]
    assert sampler.table[0].bool().tolist() == mask
    assert all(resolve_control_links(LINKS)[1])
