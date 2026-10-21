"""Stage isolation and resumable self-play state, without simulated physics."""

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from holoboxing.training.motion_prior import (
    MotionPrior,
    PriorConfig,
    state_sha256,
)
from holoboxing.training.rsl_rl.algorithms.ppo import PPO
from holoboxing.training.rsl_rl.modules.amp import AMP
from holoboxing.training.rsl_rl.runners import (
    on_policy_runner as runner_module,
)
from holoboxing.training.spherical_actor import KIND, SphericalActor
from holoboxing.training.spherical_rl import SphericalActorCritic
from holoboxing.training.standing_reference import WorldStandingReference
from holoboxing.training.tasks.combat.config.sandbag import AGENTS, TrainConfig


def small_policy():
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
        amp_batch_size=4,
        amp_replay_buffer_size=16,
    )
    prior = MotionPrior(
        torch.zeros(225), torch.ones(225), PriorConfig(hidden_dims=(16,))
    )
    actor = SphericalActor(
        prior.decoder,
        config,
        {
            "kind": KIND,
            "prior_config": asdict(prior.config),
            "decoder_sha256": state_sha256(prior.decoder),
            "opponent_condition": True,
        },
    )
    return config, SphericalActorCritic(actor, config)


def test_standing_world_targets_ignore_motion_until_individual_reset():
    neutral = torch.zeros(5, 5, 9)
    neutral[..., 3] = neutral[..., 8] = 1
    neutral[..., 2] = torch.arange(5) / 10
    hold = WorldStandingReference(neutral.flatten(), 2, "cpu")
    measured = {
        "position": torch.zeros(4, 5, 3),
        "quaternion": torch.zeros(4, 5, 4),
    }
    measured["quaternion"][..., 0] = 1
    measured["position"][:, 0, 2] = 0.8

    def inputs():
        anchor = measured["position"][:, 0].clone()
        return (
            torch.randn(4, 5, 5, 3),
            torch.randn(4, 5, 5, 4),
            anchor,
            anchor[:, 2:3].clone(),
        )

    first_inputs = inputs()
    active = [value[::2].clone() for value in first_inputs]
    first = [value.clone() for value in hold.apply(measured, *first_inputs)]
    for actual, expected in zip(first, active, strict=True):
        torch.testing.assert_close(actual[::2], expected)
    measured["position"] += 2
    measured["quaternion"][..., 0] = 0
    measured["quaternion"][..., 3] = 1
    second = hold.apply(measured, *inputs())
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a[1::2], b[1::2], rtol=0, atol=0)
    hold.reset(torch.tensor([1]))
    third = hold.apply(measured, *inputs())
    for a, b in zip(first, third, strict=True):
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    assert not torch.equal(first[0][3], third[0][3])
    torch.testing.assert_close(
        measured["position"][:, 0, 2], torch.full((4,), 2.8)
    )


def test_single_learner_ppo_excludes_opponent_updates_and_amp_samples():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    config, policy = small_policy()
    amp = AMP(torch.randn(16, 122), config, "cpu")
    algorithm = PPO(policy, amp, config, "cpu", active_roles=("ego",))
    observations = {
        agent: (torch.randn(2, 1097), torch.randn(2, 225)) for agent in AGENTS
    }
    frozen = copy.deepcopy(policy.taskpolicy_opponent.state_dict())
    learner = copy.deepcopy(policy.taskpolicy_ego.state_dict())
    with pytest.raises(ValueError, match="external action"):
        algorithm.act(observations)
    algorithm.reset_exploration()
    for _ in range(2):
        algorithm.act(
            observations, opponent_actions={AGENTS[1]: torch.zeros(2, 225)}
        )
        algorithm.process_env_step(
            {AGENTS[0]: torch.randn(2)},
            torch.zeros(2, dtype=torch.bool),
            {
                "amp_obs_ego": torch.full((2, 122), 0.25),
                "amp_obs_opponent": torch.full((2, 122), 99.0),
            },
        )
    algorithm.compute_returns(observations)
    result = algorithm.update()
    assert algorithm.optimizer_steps == {"ego": 1, "opponent": 0}
    assert set(algorithm.last_parity) == {"ego"}
    assert not any(name.startswith("opponent/") for name in result)
    assert len(amp.replay) == 4
    torch.testing.assert_close(
        amp.replay.buffer[:4], torch.full((4, 122), 0.25)
    )
    for name, value in policy.taskpolicy_opponent.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, learner[name])
        for name, value in policy.taskpolicy_ego.state_dict().items()
    )
    policy.check_frozen()


class LearningFixture:
    """CPU observations exercise orchestration; this is not a physics model."""

    device, num_envs, actor_dim, controllers = "cpu", 2, 1097, {}

    def __init__(self, mode):
        self.generator = torch.Generator().manual_seed(42)
        self.actor_noise = SimpleNamespace(
            generator=torch.Generator().manual_seed(43),
        )
        self.actor_noise.rng_state = self.actor_noise.generator.get_state
        self.standing_reference = object() if mode == "standing" else None
        self.reward_terms, self.metrics = {}, {}

    def reset(self):
        return {
            agent: {
                "task_policy": torch.randn(2, 1097, generator=self.generator),
                "critic": torch.randn(2, 225, generator=self.generator),
            }
            for agent in AGENTS
        }

    def step(self, actions):
        assert all(value.shape == (2, 225) for value in actions.values())
        return (
            self.reset(),
            {agent: torch.ones(2) for agent in AGENTS},
            torch.zeros(2, dtype=torch.bool),
            {
                **{
                    agent: {"time_outs": torch.zeros(2, dtype=torch.bool)}
                    for agent in AGENTS
                },
                **{
                    "amp_obs_" + role: torch.ones(2, 122)
                    for role in ("ego", "opponent")
                },
                **{
                    "terminal_critic_" + role: torch.zeros(2, 225)
                    for role in ("ego", "opponent")
                },
            },
        )


def make_runner(tmp_path, mode, monkeypatch):
    config, policy = small_policy()
    monkeypatch.setattr(
        runner_module,
        "FrozenLowerGuard",
        lambda _: SimpleNamespace(check=lambda: None),
    )
    runner = runner_module.OnPolicyRunner(
        LearningFixture(mode),
        torch.randn(16, 122),
        torch.zeros(225),
        config,
        tmp_path,
        {"resources": {"test": "fixture"}},
        policy=policy,
        training_mode=mode,
        opponent_pool_capacity=3,
        opponent_refresh_updates=1,
    )
    for role in ("ego", "opponent"):
        runner.normalizers["actor_" + role].update(
            torch.randn(2, 108), torch.randn(2, 123)
        )
    runner.supervised_initialization = {
        "checkpoint": "fixture.pt",
        "sha256": "a" * 64,
    }
    return runner


@pytest.mark.parametrize("mode", ["standing", "selfplay"])
def test_stage_checkpoint_restores_learner_counts_pool_and_rejects_mode_change(
    tmp_path, monkeypatch, mode
):
    torch.set_num_threads(1)
    runner = make_runner(tmp_path / "first", mode, monkeypatch)
    runner.learn(2)
    path = tmp_path / "first/model_2.pt"
    state = torch.load(path, weights_only=False)
    assert state["optimizer_steps"] == {"ego": 2, "opponent": 0}
    assert state["amp_replay"]["size"] == 8
    fresh = make_runner(tmp_path / "resume", mode, monkeypatch)
    fresh.supervised_initialization = None
    fresh.resume(path)
    assert fresh.iteration == 2
    if mode == "selfplay":
        assert fresh.opponent_pool.selected == runner.opponent_pool.selected
        assert len(fresh.opponent_pool.snapshots) == 2
        torch.testing.assert_close(
            fresh.opponent_pool.generator.get_state(),
            runner.opponent_pool.generator.get_state(),
        )
    fresh.learn(1)
    assert fresh.algorithm.optimizer_steps == {"ego": 3, "opponent": 0}
    fresh.policy.check_frozen()
    other = "selfplay" if mode == "standing" else "standing"
    incompatible = make_runner(tmp_path / "wrong", other, monkeypatch)
    incompatible.supervised_initialization = None
    with pytest.raises(ValueError, match="Incomplete resume|training mode"):
        incompatible.resume(path)


def test_historical_pool_keeps_seed_and_resumes_selection_randomness(
    tmp_path, monkeypatch
):
    from holoboxing.training.opponent_pool import OpponentPool

    torch.set_num_threads(1)
    runner = make_runner(tmp_path / "run", "selfplay", monkeypatch)
    runner.learn(5)
    saved = runner.opponent_pool.state_dict()
    assert [row["iteration"] for row in saved["snapshots"]] == [0, 3, 4]
    clone = OpponentPool(
        runner.policy, runner.normalizers, seed=0, capacity=3, interval=1
    )
    clone.load_state_dict(saved)
    assert clone.prepare(5) and runner.opponent_pool.prepare(5)
    assert clone.selected == runner.opponent_pool.selected
    assert [row["sha256"] for row in clone.snapshots] == [
        row["sha256"] for row in runner.opponent_pool.snapshots
    ]
    corrupted = copy.deepcopy(saved)
    first_tensor = next(iter(corrupted["snapshots"][0]["actor"].values()))
    first_tensor.add_(1)
    with pytest.raises(ValueError, match="hash differs"):
        clone.load_state_dict(corrupted)


def test_new_stage_uses_actor_weights_with_fresh_learner_and_exports(
    tmp_path, monkeypatch
):
    from holoboxing.training.export import export_policy

    torch.set_num_threads(1)
    runner = make_runner(tmp_path / "standing", "standing", monkeypatch)
    runner.learn(1)
    checkpoint = tmp_path / "standing/model_1.pt"
    staged = make_runner(tmp_path / "selfplay", "selfplay", monkeypatch)
    staged.supervised_initialization = None
    staged.initialize_from_policy(checkpoint)
    assert staged.iteration == staged.amp.optimizer_steps == 0
    assert staged.algorithm.optimizer_steps == {"ego": 0, "opponent": 0}
    for role in ("ego", "opponent"):
        for key, value in (
            getattr(staged.policy, "taskpolicy_" + role).state_dict().items()
        ):
            torch.testing.assert_close(
                value,
                runner.policy.taskpolicy_ego.state_dict()[key],
                rtol=0,
                atol=0,
            )
    staged.learn(1)
    result = export_policy(
        tmp_path / "selfplay/model_1.pt", tmp_path / "policy.onnx"
    )
    assert result["bytes"] < 50_000_000 and result["max_abs_error"] < 1e-4
