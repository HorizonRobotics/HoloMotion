"""PPO likelihoods belong to sampled vectors before sphere projection."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from holoboxing.training.motion_prior import (
    MotionPrior,
    PriorConfig,
    state_sha256,
    unit_vector,
)
from holoboxing.training.rsl_rl.algorithms.ppo import PPO
from holoboxing.training.rsl_rl.modules.amp import AMP
from holoboxing.training.spherical_actor import KIND, SphericalActor
from holoboxing.training.spherical_rl import SphericalActorCritic
from holoboxing.training.tasks.combat.config.sandbag import (
    AGENTS,
    ROLES,
    TrainConfig,
)


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_raw_latent_storage_likelihood_and_executed_reference(rho):
    torch.manual_seed(42)
    config = replace(
        TrainConfig(),
        num_envs=2,
        num_steps_per_env=2,
        num_mini_batches=1,
        hidden_dims=(16,),
        horizon_rho=0,
        exploration_rho=rho,
    )
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
        ),
    )
    policy = SphericalActorCritic(actor, config)
    assert policy.action_dim == 32 and config.action_dim == 225
    algorithm = PPO(policy, SimpleNamespace(), config, "cpu")
    snapshots = {
        agent: (torch.randn(2, 1097), torch.randn(2, 225)) for agent in AGENTS
    }
    for _ in range(2):
        actions = algorithm.act(snapshots)
        for role, agent in zip(ROLES, AGENTS, strict=True):
            raw = getattr(algorithm.transition, f"actions_{role}")
            saved = getattr(algorithm.transition, f"actions_log_prob_{role}")
            assert raw.shape == (2, 32) and actions[agent].shape == (2, 225)
            law = policy.distributions[role]
            torch.testing.assert_close(saved, law.log_prob(raw).sum(-1))
            assert not torch.allclose(
                saved, law.log_prob(unit_vector(raw)).sum(-1)
            )
            torch.testing.assert_close(
                actions[agent], policy.execute(role, raw)
            )
    assert policy.taskpolicy_ego.spec["opponent_condition"]
    optimizer = algorithm.optimizers["ego"]
    trainable = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all(
        id(p) not in trainable
        for p in policy.taskpolicy_ego.decoder.parameters()
    )
    raw = torch.randn(2, 32)
    loss = (
        -policy.distribution("ego", algorithm.transition.observations_ego)
        .log_prob(raw)
        .mean()
    )
    loss += policy.value("ego", snapshots[AGENTS[0]][1]).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    policy.check_frozen()


@pytest.mark.parametrize("rho", [0.0, 0.9])
def test_full_latent_rollout_update_preserves_frozen_decoder(rho):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = replace(
        TrainConfig(),
        num_envs=2,
        num_steps_per_env=2,
        num_learning_epochs=1,
        num_mini_batches=1,
        hidden_dims=(16,),
        disc_hidden_dims=(8,),
        horizon_rho=0,
        exploration_rho=rho,
        amp_batch_size=4,
        amp_replay_buffer_size=8,
    )
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
        ),
    )
    policy = SphericalActorCritic(actor, config)
    amp = AMP(torch.randn(16, 122), config, "cpu")
    algorithm = PPO(policy, amp, config, "cpu")
    observations = {
        agent: (torch.randn(2, 1097), torch.randn(2, 225)) for agent in AGENTS
    }
    algorithm.begin_rollout()
    for _ in range(2):
        executed = algorithm.act(observations)
        assert all(value.shape == (2, 225) for value in executed.values())
        algorithm.process_env_step(
            {agent: torch.randn(2) for agent in AGENTS},
            torch.zeros(2, dtype=torch.bool),
            {
                "amp_obs_ego": torch.randn(2, 122),
                "amp_obs_opponent": torch.randn(2, 122),
            },
        )
    algorithm.compute_returns(observations)
    result = algorithm.update()
    assert algorithm.storage.step == 0
    assert algorithm.optimizer_steps == dict.fromkeys(ROLES, 1)
    assert amp.optimizer_steps == 1
    assert all(
        value >= 0
        for key, value in result.items()
        if key.endswith("zero_ratio_count")
    )
    assert all(
        row["max_abs_log_ratio"] < 1e-5
        for row in algorithm.last_parity.values()
    )
    policy.check_frozen()
