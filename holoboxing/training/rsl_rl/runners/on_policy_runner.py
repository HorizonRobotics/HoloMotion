"""C225 paired sandbag training and checkpoint saving."""

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from holoboxing.training.rsl_rl.algorithms.ppo import PPO, require_finite
from holoboxing.training.rsl_rl.modules.actor_critic import ActorCritic
from holoboxing.training.rsl_rl.modules.amp import AMP
from holoboxing.training.rsl_rl.modules.normalization import (
    ActorNormalizer,
    unpack_actor_observations,
)
from holoboxing.training.rsl_rl.modules.normalizer import (
    EmpiricalNormalization,
)
from holoboxing.training.tasks.combat.config.sandbag import AGENTS, ROLES

CHECKPOINT_FORMAT = "holoboxing_c_joint_sandbag_public_v3"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalizers(device):
    return nn.ModuleDict(
        {
            **{f"actor_{role}": ActorNormalizer() for role in ROLES},
            **{
                f"critic_{role}": EmpiricalNormalization(
                    [225], until=100_000_000
                ).eval()
                for role in ROLES
            },
        }
    ).to(device)


@torch.no_grad()
def normalize_observations(modules, observations, update=False):
    result = {}
    for role, agent in zip(ROLES, AGENTS, strict=True):
        packed = observations[agent]["task_policy"]
        actor, critic = modules[f"actor_{role}"], modules[f"critic_{role}"]
        if packed.ndim != 2 or packed.shape[-1] != actor.actor_dim:
            raise ValueError(
                "Environment observation layout differs from actor"
            )
        current, task, history = unpack_actor_observations(packed)
        raw_critic = observations[agent]["critic"]
        if update:
            actor.update(current, task)
            critic.update(raw_critic)
        result[agent] = (
            actor({"self": current, "task": task, "history": history}),
            critic(raw_critic),
        )
    require_finite(result, "normalized observation")
    return result


class FrozenLowerGuard:
    def __init__(self, controllers):
        self.controllers = controllers
        self.identities = self.snapshot()

    def snapshot(self):
        from holoboxing.training.envs.onnx_lower import OnnxLower

        result = {}
        for agent, controller in self.controllers.items():
            if not isinstance(controller.module, OnnxLower):
                raise ValueError(
                    "Lower controller must use the frozen public ONNX"
                )
            result[agent] = id(controller.module.session)
        return result

    def check(self):
        if self.identities != self.snapshot():
            raise RuntimeError("Frozen lower parameters changed")


class OnPolicyRunner:
    def __init__(self, env, demo, neutral_reference, config, output, identity):
        self.env, self.config = env, config
        self.device, self.output = env.device, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if env.num_envs != config.num_envs:
            raise ValueError(
                "Environment count differs from the training config"
            )
        if env.actor_dim != config.actor_dim:
            raise ValueError(
                "Environment actor width differs from training config"
            )
        seed_everything(config.seed)
        self.policy = ActorCritic(neutral_reference, config).to(self.device)
        self.normalizers = normalizers(self.device)
        self.amp = AMP(demo, config, self.device)
        self.algorithm = PPO(self.policy, self.amp, config, self.device)
        self.lower_guard = FrozenLowerGuard(env.controllers)
        self.identity = copy.deepcopy(identity)
        self.identity["amp_tensor_sha256"] = hashlib.sha256(
            demo.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        self.iteration, self.transitions_per_role, self.elapsed_seconds = (
            0,
            0,
            0.0,
        )
        self.observations = None
        self.supervised_initialization = None

    def initialize_from_supervised(self, path):
        """Import only the attacker; keep both critics, AMP and B untouched."""
        if (
            self.iteration
            or self.algorithm.storage.step
            or self.observations is not None
        ):
            raise ValueError(
                "Supervised initialization must precede RL collection"
            )
        if self.supervised_initialization is not None:
            raise ValueError("Supervised initialization can only happen once")
        from holoboxing.training.supervised.training import (
            load_supervised,
        )
        from holoboxing.training.tasks.combat.config.sandbag import (
            sha256,
        )

        device = torch.device(self.device)
        devices = (
            [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices):
            actor, normalizer, payload = load_supervised(
                path,
                self.config,
                self.policy.taskpolicy_ego.neutral_reference.detach().cpu(),
            )
        self.policy.taskpolicy_ego.load_state_dict(
            actor.state_dict(), strict=True
        )
        self.normalizers["actor_ego"].load_state_dict(
            normalizer.state_dict(), strict=True
        )
        self.supervised_initialization = {
            "checkpoint": str(Path(path).resolve()),
            "sha256": sha256(path),
            "step": payload.get("step"),
            "epoch": payload.get("epoch"),
        }

    def _reset(self):
        self.observations = self.env.reset()
        self.algorithm.reset_exploration()
        raw = getattr(self.env, "env", None)
        if raw is not None and hasattr(raw, "episode_length_buf"):
            raw.episode_length_buf.random_(0, raw.max_episode_length)

    @torch.no_grad()
    def collect(self):
        if self.observations is None:
            self._reset()
        algorithm, cfg = self.algorithm, self.config
        algorithm.begin_rollout()
        sums = {}
        for _ in range(cfg.num_steps_per_env):
            snapshots = normalize_observations(
                self.normalizers, self.observations, update=True
            )
            actions = algorithm.act(snapshots)
            self.observations, task_rewards, done, infos = self.env.step(
                actions
            )
            rewards = {}
            for role, agent in zip(ROLES, AGENTS, strict=True):
                style = self.amp.reward(infos[f"amp_obs_{role}"])
                reward = (
                    cfg.task_reward_weight * task_rewards[agent]
                    + cfg.style_reward_weight * style
                )
                timeout = infos[agent]["time_outs"]
                # Bootstrap actual executed terminal state, never reset observation.
                terminal = self.normalizers[f"critic_{role}"](
                    infos[f"terminal_critic_{role}"]
                )
                reward = (
                    reward
                    + cfg.gamma
                    * algorithm.policy.value(role, terminal).squeeze(-1)
                    * timeout
                )
                rewards[agent] = reward
                values = {
                    "task_reward": task_rewards[agent].mean(),
                    "style_reward": style.mean(),
                    "done_fraction": done.float().mean(),
                    "fallen_fraction": self.env.metrics.get(agent, {})
                    .get("fallen", torch.zeros_like(done))
                    .float()
                    .mean(),
                }
                values.update(
                    {
                        f"reward/{name}": value.mean()
                        for name, value in self.env.reward_terms.get(
                            agent, {}
                        ).items()
                    }
                )
                for name, value in values.items():
                    key = f"{role}/{name}"
                    sums[key] = sums.get(key, 0.0) + value.item()
            algorithm.process_env_step(rewards, done, infos)
        algorithm.compute_returns(
            normalize_observations(self.normalizers, self.observations)
        )
        return {
            key: value / cfg.num_steps_per_env for key, value in sums.items()
        }

    def learn(self, updates, save_interval=100, extra_save_interval=None):
        if type(updates) is not int or updates <= 0 or save_interval <= 0:
            raise ValueError("Positive update and save budgets required")
        if extra_save_interval is not None and (
            type(extra_save_interval) is not int or extra_save_interval <= 0
        ):
            raise ValueError("Extra save interval must be a positive integer")
        for _ in range(updates):
            started = time.monotonic()
            try:
                metrics = self.collect()
                metrics.update(self.algorithm.update())
                self.lower_guard.check()
            except Exception as error:
                # A diagnostic snapshot is not a resumable successful checkpoint.
                torch.save(
                    {
                        "error": repr(error),
                        "iteration": self.iteration,
                        "parity": self.algorithm.last_parity,
                        "policy": self.policy.state_dict(),
                        "storage": vars(self.algorithm.storage),
                        "optimizer_steps": self.algorithm.optimizer_steps,
                    },
                    self.output / "failure.pt",
                )
                raise
            self.iteration += 1
            self.transitions_per_role += (
                self.config.num_envs * self.config.num_steps_per_env
            )
            self.elapsed_seconds += time.monotonic() - started
            record = {
                "iteration": self.iteration,
                "transitions_per_role": self.transitions_per_role,
                "total_role_transitions": 2 * self.transitions_per_role,
                "elapsed_seconds": self.elapsed_seconds,
                "learning_rates": self.algorithm.learning_rates,
                "optimizer_steps": self.algorithm.optimizer_steps,
                "disc_optimizer_steps": self.amp.optimizer_steps,
                "probability_check": self.algorithm.last_parity,
                "actor_cuda_graphs": self.policy.actor_cuda_graph_status(),
                **metrics,
            }
            with (self.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
            if self.iteration % save_interval == 0 or (
                extra_save_interval is not None
                and self.iteration % extra_save_interval == 0
            ):
                self.save(self.output / f"model_{self.iteration}.pt")
        self.save(self.output / f"model_{self.iteration}.pt")

    def save(self, path):
        if self.algorithm.storage.step:
            raise ValueError("Save only at completed update boundaries")
        self.lower_guard.check()
        state = {
            "format": CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "identity": self.identity,
            "supervised_initialization": self.supervised_initialization,
            "policy": self.policy.state_dict(),
            "normalizers": self.normalizers.state_dict(),
            "learning_rates": self.algorithm.learning_rates,
            "optimizer_steps": self.algorithm.optimizer_steps,
            "discriminator": self.amp.discriminator.state_dict(),
            "disc_optimizer_steps": self.amp.optimizer_steps,
            "amp_normalizer": self.amp.normalizer.state_dict(),
            "iteration": self.iteration,
            "transitions_per_role": self.transitions_per_role,
            "elapsed_seconds": self.elapsed_seconds,
        }
        require_finite(state, "checkpoint")
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
