from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tensordict import TensorDict

from holomotion.src.algo.ppo import PPO


class _DummyAccelerator:
    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)

    def reduce(self, tensor, reduction="mean"):
        return tensor


class _DummyActor(nn.Module):
    def __init__(self, num_actions: int, mirror_offset: float):
        super().__init__()
        self.mu_param = nn.Parameter(torch.full((num_actions,), 0.25))
        self.log_std = nn.Parameter(torch.zeros(num_actions))
        self.mirror_offset = float(mirror_offset)

    def forward(
        self,
        obs_td: TensorDict,
        actions: torch.Tensor | None = None,
        mode: str = "sampling",
        *,
        update_obs_norm: bool = True,
    ) -> TensorDict:
        del obs_td, update_obs_norm
        batch_size = int(actions.shape[0]) if actions is not None else 2
        mu = self.mu_param.unsqueeze(0).expand(batch_size, -1)
        sigma = torch.exp(self.log_std).unsqueeze(0).expand(batch_size, -1)
        out = TensorDict({}, batch_size=[batch_size])
        out.set("mu", mu)
        out.set("sigma", sigma)
        if mode == "inference":
            out.set("actions", mu + self.mirror_offset)
            return out
        if actions is None:
            actions = mu
        out.set("actions", actions)
        zero_with_grad = mu.sum(dim=-1) * 0.0
        out.set("actions_log_prob", zero_with_grad)
        out.set("entropy", zero_with_grad)
        return out


class _DummyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, obs_td: TensorDict, *, update_obs_norm: bool = True):
        del obs_td, update_obs_norm
        batch_size = 2
        out = TensorDict({}, batch_size=[batch_size])
        out.set("values", self.value.view(1, 1).expand(batch_size, 1))
        return out


class _SingleBatchStorage:
    def __init__(self, batch):
        self._batch = batch
        self.data = {
            "returns": torch.zeros(2, 1, 1, dtype=torch.float32),
            "values": torch.zeros(2, 1, 1, dtype=torch.float32),
        }
        self.num_envs = 1
        self.num_transitions_per_env = 2
        self.cleared = False

    def iter_minibatches(self, num_mini_batches: int, num_epochs: int):
        del num_mini_batches, num_epochs
        yield self._batch

    def clear(self):
        self.cleared = True


def _install_mirror_stub():
    module = ModuleType(
        "holomotion.src.env.isaaclab_components.isaaclab_observation"
    )

    class MirrorFunctions:
        @staticmethod
        def mirror_dof(
            x: torch.Tensor, *, perm: torch.Tensor, sign: torch.Tensor
        ):
            perm = perm.to(device=x.device, dtype=torch.long)
            sign = sign.to(device=x.device, dtype=x.dtype)
            mirrored = torch.index_select(x, dim=x.ndim - 1, index=perm)
            view_shape = [1] * (mirrored.ndim - 1) + [int(sign.numel())]
            return mirrored * sign.view(*view_shape)

        @staticmethod
        def mirror_action(
            actions: torch.Tensor, *, perm: torch.Tensor, sign: torch.Tensor
        ):
            return MirrorFunctions.mirror_dof(actions, perm=perm, sign=sign)

        @staticmethod
        def mirror_vec3(x: torch.Tensor):
            sign = torch.tensor(
                [1.0, -1.0, 1.0], device=x.device, dtype=x.dtype
            )
            view_shape = [1] * (x.ndim - 1) + [3]
            return x * sign.view(*view_shape)

        @staticmethod
        def mirror_axial_vec3(x: torch.Tensor):
            sign = torch.tensor(
                [-1.0, 1.0, -1.0], device=x.device, dtype=x.dtype
            )
            view_shape = [1] * (x.ndim - 1) + [3]
            return x * sign.view(*view_shape)

        @staticmethod
        def mirror_velocity_command(x: torch.Tensor):
            if x.shape[-1] == 3:
                sign = torch.tensor(
                    [1.0, -1.0, -1.0], device=x.device, dtype=x.dtype
                )
            else:
                sign = torch.tensor(
                    [1.0, 1.0, -1.0, -1.0], device=x.device, dtype=x.dtype
                )
            view_shape = [1] * (x.ndim - 1) + [int(sign.numel())]
            return x * sign.view(*view_shape)

    module.MirrorFunctions = MirrorFunctions
    sys.modules[module.__name__] = module


def test_setup_symmetry_builds_expected_dof_permutation_and_signs():
    _install_mirror_stub()
    algo = PPO.__new__(PPO)
    algo.device = torch.device("cpu")
    algo.num_actions = 5
    algo.command_name = "base_velocity"
    algo.symmetry_loss_enabled = True
    algo.is_main_process = False
    algo.config = OmegaConf.create(
        {
            "module_dict": {
                "actor": {
                    "obs_schema": {
                        "flattened_obs": {
                            "seq_len": 2,
                            "terms": ["unified/actor_dof_pos"],
                        }
                    }
                }
            },
            "symmetry_loss": {
                "enabled": True,
                "coef": 0.1,
                "dof_sign_by_name": {
                    "left_hip_pitch_joint": 1.0,
                    "right_hip_pitch_joint": 1.0,
                    "waist_yaw_joint": -1.0,
                    "left_knee_joint": 1.0,
                    "right_knee_joint": 1.0,
                },
            },
        }
    )
    algo.env = SimpleNamespace(
        _env=SimpleNamespace(
            scene={
                "robot": SimpleNamespace(
                    joint_names=[
                        "left_hip_pitch_joint",
                        "right_hip_pitch_joint",
                        "waist_yaw_joint",
                        "left_knee_joint",
                        "right_knee_joint",
                    ]
                )
            }
        )
    )
    algo.env_config = OmegaConf.create(
        {
            "config": {
                "robot": {
                    "dof_sign_by_name": {
                        "left_hip_pitch_joint": 1.0,
                        "right_hip_pitch_joint": 1.0,
                        "waist_yaw_joint": -1.0,
                        "left_knee_joint": 1.0,
                        "right_knee_joint": 1.0,
                    }
                },
                "obs": {
                    "obs_groups": {
                        "unified": {
                            "atomic_obs_list": [
                                {
                                    "actor_dof_pos": {
                                        "mirror_func": "mirror_dof",
                                    }
                                }
                            ]
                        }
                    }
                },
            }
        }
    )

    algo._setup_symmetry()

    assert algo._sym_dof_perm.tolist() == [1, 0, 2, 4, 3]
    assert algo._sym_dof_sign.tolist() == [1.0, 1.0, -1.0, 1.0, 1.0]


def test_mirror_actor_obs_uses_slash_qualified_actor_terms_only():
    _install_mirror_stub()
    algo = PPO.__new__(PPO)
    algo.command_name = "base_velocity"
    algo.symmetry_loss_enabled = True
    algo.symmetry_loss_coef = 0.1
    algo._obs_mirror_map = {
        "unified/actor_velocity_command": lambda x: x * 2.0,
        "unified/actor_dof_pos": lambda x: x + 1.0,
    }
    obs_td = TensorDict.from_dict(
        {
            "unified": {
                "actor_velocity_command": torch.tensor(
                    [[[1.0, 2.0, 3.0]]], dtype=torch.float32
                ),
                "actor_dof_pos": torch.tensor(
                    [[[0.1, 0.2]]], dtype=torch.float32
                ),
                "critic_dof_pos": torch.tensor(
                    [[9.0, 8.0]], dtype=torch.float32
                ),
            }
        },
        batch_size=[1],
        device="cpu",
    )

    mirrored = algo._mirror_actor_obs(obs_td)

    torch.testing.assert_close(
        mirrored["unified", "actor_velocity_command"],
        torch.tensor([[[2.0, 4.0, 6.0]]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        mirrored["unified", "actor_dof_pos"],
        torch.tensor([[[1.1, 1.2]]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        mirrored["unified", "critic_dof_pos"],
        obs_td["unified", "critic_dof_pos"],
    )


def test_update_reports_symmetry_loss_only_for_velocity_tracking():
    algo = PPO.__new__(PPO)
    algo.device = torch.device("cpu")
    algo.accelerator = _DummyAccelerator()
    algo.actor = _DummyActor(num_actions=2, mirror_offset=1.0)
    algo.critic = _DummyCritic()
    algo.actor_optimizer = torch.optim.SGD(algo.actor.parameters(), lr=0.01)
    algo.critic_optimizer = torch.optim.SGD(algo.critic.parameters(), lr=0.01)
    algo.storage = _SingleBatchStorage(
        SimpleNamespace(
            obs=TensorDict.from_dict(
                {
                    "unified": {
                        "actor_dof_pos": torch.zeros(2, 1, 2),
                        "critic_dof_pos": torch.zeros(2, 2),
                    }
                },
                batch_size=[2],
                device="cpu",
            ),
            actions=torch.zeros(2, 2),
            values=torch.zeros(2, 1),
            advantages=torch.zeros(2, 1),
            returns=torch.zeros(2, 1),
            actions_log_prob=torch.zeros(2, 1),
            mu=torch.zeros(2, 2),
            sigma=torch.ones(2, 2),
        )
    )
    algo.value_loss_coef = 1.0
    algo.clip_param = 0.2
    algo.max_grad_norm = 1.0
    algo.schedule = "fixed"
    algo.desired_kl = None
    algo.distributed_update_mode = "legacy"
    algo.num_mini_batches = 1
    algo.num_learning_epochs = 1
    algo.configured_num_mini_batches = 1
    algo.requested_num_mini_batches = 1
    algo.distributed_lr_scale_factor = 1.0
    algo.entropy_coef = 0.0
    algo.initial_entropy_coef = 0.0
    algo.anneal_entropy = False
    algo.use_clipped_value_loss = False
    algo.actor_learning_rate = 1.0e-3
    algo.critic_learning_rate = 1.0e-3
    algo.global_advantage_norm = True
    algo.is_distributed = False
    algo.symmetry_loss_enabled = True
    algo.symmetry_loss_coef = 0.5
    algo._mirror_actor_obs = lambda obs_td: obs_td
    algo._mirror_env_action = lambda actions: actions
    algo._post_update_hook = lambda loss_dict: None

    algo.command_name = "base_velocity"
    velocity_loss = algo.update()

    assert velocity_loss["symmetry_loss"] == pytest.approx(1.0)

    algo.storage = _SingleBatchStorage(algo.storage._batch)
    algo.command_name = "ref_motion"
    non_velocity_loss = algo.update()

    assert "symmetry_loss" not in non_velocity_loss
