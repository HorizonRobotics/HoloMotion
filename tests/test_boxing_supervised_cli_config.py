"""Compact CLI settings survive training and export loading."""

import json
from types import SimpleNamespace

import torch
from test_boxing_training_splits import population, receipt

from holoboxing.training import resources, train_sl
from holoboxing.training.export import load_policy


def test_compact_cli_checkpoint_keeps_actor_width_and_train_only_statistics(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    manifest = population(tmp_path)
    metadata = json.loads(manifest.read_text())
    metadata["asset_identity"]["resource_manifest_sha256"] = "f" * 64
    manifest.write_text(json.dumps(metadata))
    receipt(tmp_path)
    neutral = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 1.0]).repeat(25)
    monkeypatch.setattr(
        resources,
        "resolve_resources",
        lambda *_: (
            SimpleNamespace(manifest_sha256="f" * 64),
            {"lower_onnx": None, "arena_scene": None},
            None,
        ),
    )
    monkeypatch.setattr(
        resources, "load_neutral_reference", lambda *_: neutral
    )
    config = tmp_path / "compact.json"
    config.write_text(json.dumps({"hidden_dims": [16, 8]}))
    destination = tmp_path / "trained"
    train_sl.main(
        [
            "--assets",
            str(tmp_path),
            "--lower",
            str(tmp_path),
            "--data",
            str(manifest.parent),
            "--output",
            str(destination),
            "--config",
            str(config),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )
    result = json.loads((destination / "training-result.json").read_text())
    saved = torch.load(result["checkpoint"], weights_only=False)
    assert tuple(saved["config"]["hidden_dims"]) == (16, 8)
    assert saved["normalizer"]["count"] == 12
    policy = load_policy(result["checkpoint"])
    with torch.inference_mode():
        output = policy(torch.ones(2, 1097))
    assert output.shape == (2, 225) and torch.isfinite(output).all()


def test_sample_config_initializes_the_frozen_latent_rl_actor():
    from pathlib import Path

    from holoboxing.training.motion_prior import (
        MotionPrior,
        PriorConfig,
        state_sha256,
    )
    from holoboxing.training.spherical_actor import KIND, SphericalActor
    from holoboxing.training.spherical_rl import SphericalActorCritic
    from holoboxing.training.tasks.combat.config.sandbag import TrainConfig

    path = Path(train_sl.__file__).parent / "configs/sample.json"
    config = TrainConfig(**json.loads(path.read_text()))
    prior = MotionPrior(
        torch.zeros(225), torch.ones(225), PriorConfig(hidden_dims=(16,))
    )
    actor = SphericalActor(
        prior.decoder,
        config,
        {
            "kind": KIND,
            "decoder_sha256": state_sha256(prior.decoder),
            "opponent_condition": True,
        },
    )
    policy = SphericalActorCritic(actor, config)
    with torch.inference_mode():
        latent = policy.taskpolicy_ego(torch.randn(2, 1097))
        reference = policy.execute("ego", latent)
    assert latent.shape == (2, 32) and reference.shape == (2, 225)
    assert torch.isfinite(reference).all()
    actor.check_frozen()
