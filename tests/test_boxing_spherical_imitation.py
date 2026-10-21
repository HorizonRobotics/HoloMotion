"""The frozen representation stays inside the exported causal actor."""

import json
from dataclasses import replace
from types import SimpleNamespace

import onnx
import pytest
import torch
from test_boxing_motion_prior import reference_population
from test_boxing_supervised_resume import TwoUpdates
from test_boxing_training_splits import population

from holoboxing.training.export import export_policy
from holoboxing.training.motion_prior import (
    MotionPrior,
    PriorConfig,
    state_sha256,
)
from holoboxing.training.prior_data import PriorRecords
from holoboxing.training.spherical_actor import (
    KIND,
    SphericalActor,
    actor_from_prior,
)
from holoboxing.training.supervised.data import Records
from holoboxing.training.supervised.statistics import fit_statistics
from holoboxing.training.supervised.training import load_supervised, train
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig
from holoboxing.training.train_prior import train as train_prior


def test_current_state_conditioning_and_frozen_decoder_gradients():
    torch.manual_seed(3)
    config = replace(TrainConfig(), hidden_dims=(16,))
    prior = MotionPrior(
        torch.zeros(225), torch.ones(225), PriorConfig(hidden_dims=(16,))
    )
    spec = dict(
        kind=KIND,
        opponent_condition=True,
        self_observation_mode="current_state",
        preserve_input_mask=True,
        decoder_sha256=state_sha256(prior.decoder),
    )
    actor = SphericalActor(prior.decoder, config, spec)
    obs = torch.randn(4, 1097)
    altered = obs.clone()
    altered[:, 1080:1095] += 20
    assert not torch.equal(actor(obs), actor(altered))
    before = {k: v.clone() for k, v in actor.decoder.state_dict().items()}
    actor(obs).square().mean().backward()
    assert actor.latent_policy[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in actor.decoder.parameters())
    optimizer = torch.optim.Adam(
        (p for p in actor.parameters() if p.requires_grad)
    )
    optimizer.step()
    actor.check_frozen()
    assert all(
        torch.equal(v, actor.decoder.state_dict()[k])
        for k, v in before.items()
    )
    with torch.no_grad():
        actor.decoder.network[0].weight.add_(0.1)
    with pytest.raises(ValueError, match="decoder changed"):
        actor.check_frozen()


def test_train_save_export_and_prior_validation_leak_guard(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / "measured"
    root.mkdir()
    measured = population(root)
    records, validation = (
        Records(measured),
        Records(measured, split="validation"),
    )
    prior_manifest = reference_population(tmp_path / "references")
    meta = json.loads(prior_manifest.read_text())
    meta["asset_identity"] = records.manifest["asset_identity"]
    prior_manifest.write_text(json.dumps(meta))
    prior_path = train_prior(
        PriorRecords(prior_manifest),
        PriorRecords(prior_manifest, split="validation"),
        tmp_path / "prior",
        epochs=1,
        hidden_dims=(16,),
        batch_size=4,
    )
    config = replace(TrainConfig(), hidden_dims=(16,))
    statistics, normalizer = fit_statistics(records)
    neutral = torch.zeros(225)
    checkpoint = train(
        records,
        config,
        neutral,
        statistics,
        normalizer,
        tmp_path / "imitation",
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        prior_checkpoint=prior_path,
        validation_records=validation,
    )
    actor, _, payload = load_supervised(checkpoint, config, neutral)
    actor.check_frozen()
    assert payload["coverage"]["whole_epoch_complete"]
    assert payload["policy_spec"]["kind"] == KIND
    assert payload["policy_spec"]["self_observation_mode"] == "current_state"
    assert payload["policy_spec"]["preserve_input_mask"]
    assert not any(p.requires_grad for p in actor.decoder.parameters())
    output = tmp_path / "policy.onnx"
    export_policy(checkpoint, output)
    graph = onnx.load(output)
    assert [i.name for i in graph.graph.input] == ["obs"]
    assert [o.name for o in graph.graph.output] == ["target_pose"]
    assert (
        graph.graph.input[0].type.tensor_type.shape.dim[-1].dim_value == 1097
    )
    assert (
        graph.graph.output[0].type.tensor_type.shape.dim[-1].dim_value == 225
    )
    import numpy as np
    import onnxruntime as ort

    keep = torch.zeros(1097, dtype=torch.bool)
    keep[972:1035] = True
    keep[1080:1095] = True
    observation = torch.randn(1, 1097).numpy()
    altered = observation.copy()
    altered[:, ~keep.numpy()] += 100
    session = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    assert np.array_equal(
        session.run(None, {"obs": observation})[0],
        session.run(None, {"obs": altered})[0],
    )
    with pytest.raises(ValueError, match="leaked imitation validation"):
        actor_from_prior(
            prior_path,
            config,
            records,
            SimpleNamespace(
                source_metadata=lambda: [
                    dict(dataset="interhuman", group="1", split="validation")
                ]
            ),
        )


def test_current_state_contract_survives_rl_and_ignores_cached_history():
    from holoboxing.training.spherical_rl import SphericalActorCritic

    torch.manual_seed(12)
    config = replace(TrainConfig(), hidden_dims=(16,), horizon_rho=0)
    prior = MotionPrior(
        torch.zeros(225), torch.ones(225), PriorConfig(hidden_dims=(16,))
    )
    actor = SphericalActor(
        prior.decoder,
        config,
        dict(
            kind=KIND,
            decoder_sha256=state_sha256(prior.decoder),
            opponent_condition=True,
            self_observation_mode="current_state",
            preserve_input_mask=True,
        ),
    )
    keep = torch.zeros(1097, dtype=torch.bool)
    keep[972:1035] = True
    keep[1080:1095] = True
    obs = torch.randn(4, 1097, requires_grad=True)
    changed = obs.detach().clone()
    changed[:, ~keep] += torch.randn_like(changed[:, ~keep]) * 100
    assert torch.equal(actor(obs), actor(changed))
    actor.latent(obs).square()[..., 0].sum().backward()
    assert obs.grad[:, ~keep].count_nonzero() == 0
    assert obs.grad[:, keep].abs().max() > 0
    rl = SphericalActorCritic(actor, config)
    for role in ("ego", "opponent"):
        mean = getattr(rl, f"taskpolicy_{role}")
        assert mean.spec == actor.spec
        assert torch.equal(mean(obs), mean(changed))


def test_spherical_imitation_resumes_without_original_prior(tmp_path):
    torch.set_num_threads(1)
    measured_root = tmp_path / "measured"
    measured_root.mkdir()
    records = Records(population(measured_root))
    manifest = reference_population(tmp_path / "references")
    packet = json.loads(manifest.read_text())
    packet["asset_identity"] = records.manifest["asset_identity"]
    manifest.write_text(json.dumps(packet))
    prior_path = train_prior(
        PriorRecords(manifest),
        PriorRecords(manifest, split="validation"),
        tmp_path / "prior",
        epochs=1,
        hidden_dims=(16,),
        batch_size=4,
    )
    config = replace(TrainConfig(), hidden_dims=(16,))
    statistics, normalizer = fit_statistics(records)
    args = (records, config, torch.zeros(225), statistics, normalizer)
    recipe = dict(epochs=2, batch_size=2, learning_rate=0.001)
    expected = train(
        *args,
        tmp_path / "whole",
        prior_checkpoint=prior_path,
        **recipe,
    )
    stopped = train(
        *args,
        tmp_path / "stopped",
        prior_checkpoint=prior_path,
        time_budget=TwoUpdates(),
        **recipe,
    )
    prior_path.unlink()
    actual = train(*args, tmp_path / "resume", resume=stopped, **recipe)
    a, b = (
        torch.load(path, weights_only=False) for path in (expected, actual)
    )
    assert a["step"] == b["step"] and a["policy_spec"] == b["policy_spec"]
    assert b["policy_spec"]["opponent_condition"]
    assert b["policy_spec"]["self_observation_mode"] == "current_state"
    assert all(
        torch.equal(value, b["actor"][key])
        for key, value in a["actor"].items()
    )
    assert a["resume"]["cursor"] == b["resume"]["cursor"]
    for index, state in a["resume"]["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value,
                b["resume"]["optimizer"]["state"][index][name],
                rtol=0,
                atol=0,
            )
