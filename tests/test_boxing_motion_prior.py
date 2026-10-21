"""Reference-only contracts, frozen gradients and exact learning restart."""

import json

import numpy as np
import pytest
import torch
from test_boxing_supervised_resume import TwoUpdates

from holoboxing.data.common import sha256
from holoboxing.training.motion_prior import (
    MotionPrior,
    PriorConfig,
    load_prior,
    unit_vector,
)
from holoboxing.training.prepare_prior import FORMAT
from holoboxing.training.prior_data import PriorRecords
from holoboxing.training.train_prior import train


def reference_population(root):
    root.mkdir()
    rows = []
    for index, (dataset, split) in enumerate(
        (
            ("robostriker", "train"),
            ("interhuman", "train"),
            ("robostriker", "validation"),
            ("interhuman", "validation"),
        )
    ):
        points = np.tile(
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 1], np.float32), (8, 25, 1)
        )
        points[..., :3] = np.random.default_rng(index).normal(
            index, 0.1, (8, 25, 3)
        )
        path = root / f"{index}.npz"
        # No observations exist in these files.
        np.savez_compressed(path, target_pose=points.reshape(8, 225))
        rows.append(
            dict(
                path=path.name,
                bytes=path.stat().st_size,
                sha256=sha256(path),
                source_sha256=str(index) * 64,
                samples=8,
                frames=13,
                roles=1,
                dataset=dataset,
                split=split,
                group=str(index),
            )
        )
    manifest = dict(
        format=FORMAT,
        control_hz=50,
        future_offsets=[1, 2, 3, 4, 5],
        coordinate_frame="source_current_pelvis_full_3d",
        input="reference_only_225",
        synthetic_opponent_samples=0,
        asset_identity={},
        records=rows,
    )
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_reference_only32_decoder_has_unit_axes_and_encoder_contract():
    torch.manual_seed(0)
    prior = MotionPrior(
        torch.zeros(225), torch.ones(225), PriorConfig(hidden_dims=(16,))
    )
    reference = torch.randn(4, 225)
    latent = prior.encode(reference)
    assert latent.shape == (4, 32)
    torch.testing.assert_close(latent.norm(dim=-1), torch.ones(4))
    points = prior.decoder(latent).reshape(4, 25, 9)
    x, z = points[..., 3:6], points[..., 6:]
    torch.testing.assert_close(x.norm(dim=-1), torch.ones(4, 25))
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(4, 25))
    torch.testing.assert_close(
        (x * z).sum(-1), torch.zeros(4, 25), atol=1e-6, rtol=0
    )
    with pytest.raises(TypeError):
        prior.encode(reference, torch.zeros(4, 108))
    with pytest.raises(ValueError, match="32-dimensional"):
        PriorConfig(latent_dim=64)


def test_zero_and_large_latents_remain_unit_with_finite_tangent_gradients():
    values = torch.zeros(3, 32, requires_grad=True)
    normalized = unit_vector(values)
    normalized[:, 1:].sum().backward()
    assert (
        torch.isfinite(values.grad).all()
        and values.grad[:, 1:].abs().min() > 0
    )
    torch.testing.assert_close(normalized.norm(dim=-1), torch.ones(3))
    large = unit_vector(torch.full((3, 32), 1e30))
    torch.testing.assert_close(large.norm(dim=-1), torch.ones(3))


def test_reference_training_resume_and_frozen_decoder_gradient(tmp_path):
    torch.set_num_threads(1)
    manifest = reference_population(tmp_path / "data")
    records = PriorRecords(manifest)
    heldout = PriorRecords(manifest, split="validation")
    assert records.selection["samples"] == 16
    assert records.targets.mean() < heldout.targets.mean()
    recipe = dict(
        epochs=2, batch_size=4, hidden_dims=(16,), learning_rate=0.001
    )
    whole = train(records, heldout, tmp_path / "whole", **recipe)
    stopped = train(
        records,
        heldout,
        tmp_path / "stopped",
        time_budget=TwoUpdates(),
        **recipe,
    )
    partial = torch.load(stopped, weights_only=False)
    assert partial["steps"] == partial["cursor"]["batch"] == 2
    resumed = train(
        records, heldout, tmp_path / "resumed", resume=stopped, **recipe
    )
    expected, actual = (
        torch.load(p, weights_only=False) for p in (whole, resumed)
    )
    assert expected["steps"] == actual["steps"] == 8
    assert all(
        torch.equal(v, actual["model"][k])
        for k, v in expected["model"].items()
    )
    for key, values in expected["optimizer"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, actual["optimizer"]["state"][key][name])
    frozen, _ = load_prior(resumed)
    latent = torch.randn(4, 32, requires_grad=True)
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    frozen.decoder(latent).square().mean().backward()
    assert latent.grad.abs().sum() > 0
    assert all(
        p.grad is None and not p.requires_grad for p in frozen.parameters()
    )
    assert all(
        torch.equal(v, frozen.state_dict()[k]) for k, v in before.items()
    )


def test_original_and_mirror_reference_groups_cannot_cross_splits(tmp_path):
    manifest = reference_population(tmp_path / "data")
    data = json.loads(manifest.read_text())
    data["records"][2]["group"] = data["records"][0]["group"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="crosses train/validation"):
        PriorRecords(manifest)


def test_old_retargeted_pairs_cannot_silently_acquire_current_asset_identity(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from holoboxing.training import prepare_prior

    source = tmp_path / "old-pairs"
    source.mkdir()
    (source / "motions.json").write_text(
        json.dumps({"asset_identity": {"scene_sha256": "old"}})
    )
    monkeypatch.setattr(
        prepare_prior.CompetitionAssets,
        "load",
        lambda _: SimpleNamespace(
            identity=lambda: {"scene_sha256": "current"}
        ),
    )
    monkeypatch.setattr(
        prepare_prior,
        "motion_kinematics",
        lambda *a: pytest.fail("stale data reached FK"),
    )
    with pytest.raises(ValueError, match="different robot assets"):
        prepare_prior.prepare([source], None, tmp_path / "references")
    assert not (tmp_path / "references").exists()
