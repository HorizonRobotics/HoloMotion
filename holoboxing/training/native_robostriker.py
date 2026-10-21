# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Native Stage-3 G1 opponent; its full-state inputs are evaluation-only.

The deterministic policy equations follow RoboStriker-Preview's BSD-3-Clause
ActorCriticStage3 and EmpiricalNormalization implementations. No reference
tracker is substituted for the checkpoint's original prior and decoder.
The accompanying license is retained in rsl_rl/LICENSE.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)
from holoboxing.training.tasks.combat.mdp.geometry import matrix, rotate

SOURCE = "https://github.com/yinkangning0124/RoboStriker-Preview"


def _network(state, prefix):
    keys = sorted(
        (
            key
            for key in state
            if key.startswith(prefix + ".") and key.endswith(".weight")
        ),
        key=lambda key: int(key[len(prefix) + 1 :].split(".")[0]),
    )
    if not keys:
        raise ValueError(f"Missing native opponent network: {prefix}")
    layers, width = [], None
    for index, key in enumerate(keys):
        weight, bias = state[key], state[key.removesuffix("weight") + "bias"]
        if (
            weight.ndim != 2
            or bias.shape != (weight.shape[0],)
            or (width is not None and weight.shape[1] != width)
        ):
            raise ValueError(f"Invalid native opponent layer: {key}")
        if index:
            layers.append(nn.ELU())
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        layers.append(layer)
        width = weight.shape[0]
    return nn.Sequential(*layers)


class NativeNormalizer(nn.Module):
    def __init__(self, state, width):
        super().__init__()
        mean, std = (state[key].float().clone() for key in ("_mean", "_std"))
        if (
            mean.shape != (1, width)
            or std.shape != mean.shape
            or (std < 0).any()
        ):
            raise ValueError("Invalid native opponent statistics")
        require_finite((mean, std), "native opponent statistics")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, value):
        return (value - self.mean) / (self.std + 0.01)


class NativeRoboStriker(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint["model_state_dict"]
        require_finite(state, "native opponent weights")
        self.actor = _network(state, "taskpolicy_ego")
        self.prior = _network(state, "prior")
        self.decoder = _network(state, "decoder")
        dimensions = tuple(
            (net[0].in_features, net[-1].out_features)
            for net in (self.actor, self.prior, self.decoder)
        )
        if dimensions != ((192, 32), (900, 64), (122, 29)):
            raise ValueError(
                "Expected native RoboStriker Stage-3 G1 dimensions"
            )
        self.task_norm = NativeNormalizer(
            checkpoint["taskpolicy_obs_normalizer_ego_state_dict"], 192
        )
        self.prior_norm = NativeNormalizer(
            checkpoint["prior_obs_normalizer_state_dict"], 900
        )
        self.decoder_norm = NativeNormalizer(
            checkpoint["decoder_obs_normalizer_state_dict"], 90
        )
        self.eval().requires_grad_(False)

    def forward(self, task, history, proprioception):
        latent = self.actor(self.task_norm(task))
        latent = F.normalize(
            latent + self.prior(self.prior_norm(history))[:, :32], dim=-1
        )
        return self.decoder(
            torch.cat((self.decoder_norm(proprioception), latent), -1)
        )


def controller_parameters():
    """Original G1 joint offsets, gains and action scales."""
    defaults, armature, effort = [], [], []
    for name in JOINT_NAMES:
        default = 0.0
        for suffix, value in (
            ("hip_pitch_joint", -0.312),
            ("knee_joint", 0.669),
            ("ankle_pitch_joint", -0.363),
            ("elbow_joint", 0.6),
            ("shoulder_pitch_joint", 0.2),
        ):
            if name.endswith(suffix):
                default = value
        if name in ("left_shoulder_roll_joint", "right_shoulder_roll_joint"):
            default = 0.2 if name.startswith("left") else -0.2
        if "hip_roll" in name or "knee" in name:
            inertia, limit = 0.025101925, 139.0
        elif "hip" in name or name == "waist_yaw_joint":
            inertia, limit = 0.010177520, 88.0
        elif "ankle" in name or name in (
            "waist_roll_joint",
            "waist_pitch_joint",
        ):
            inertia, limit = 2 * 0.003609725, 50.0
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            inertia, limit = 0.00425, 5.0
        else:
            inertia, limit = 0.003609725, 25.0
        defaults.append(default)
        armature.append(inertia)
        effort.append(limit)
    kp = torch.tensor(armature) * (20 * math.pi) ** 2
    kd = 4 * torch.tensor(armature) * 20 * math.pi
    return torch.tensor(defaults), kp, kd, 0.25 * torch.tensor(effort) / kp


def native_proprioception(
    omega_b, joint_position, joint_velocity, last_action, default
):
    return torch.cat(
        (omega_b, joint_position - default, joint_velocity, last_action), -1
    )


def native_task_observation(position, quaternion, proprioception):
    """Express both native roles in their own pelvis frames."""
    if (
        position.ndim != 4
        or position.shape[1:] != (2, 30, 3)
        or quaternion.shape != (*position.shape[:-1], 4)
        or proprioception.shape != (len(position), 2, 90)
    ):
        raise ValueError("Expected paired native G1 states in named BFS order")
    inverse = matrix(quaternion[:, :, 0]).transpose(-1, -2)
    local = rotate(inverse[:, :, None], position - position[:, :, :1])
    fists = [
        BODY_NAMES.index(name)
        for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")
    ]
    torso = BODY_NAMES.index("torso_link")
    attack = rotate(
        inverse[:, :, None],
        position[:, :, fists] - position.flip(1)[:, :, torso, None],
    )
    defense = rotate(
        inverse[:, :, None],
        position.flip(1)[:, :, fists] - position[:, :, torso, None],
    )
    return torch.cat(
        (
            proprioception,
            local.flatten(-2),
            attack.flatten(-2),
            defense.flatten(-2),
        ),
        -1,
    )


class NativeRoboController:
    """Frozen inference with one fighter's own history and last action."""

    def __init__(self, policy, batch_size=1):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.batch_size = batch_size
        self.default, self.kp, self.kd, self.scale = (
            value.to(self.device) for value in controller_parameters()
        )
        self.last_action = torch.zeros(batch_size, 29, device=self.device)
        self.history = None

    def reset(self):
        self.last_action.zero_()
        self.history = None

    @torch.no_grad()
    def act(
        self,
        position,
        quaternion,
        omega_b,
        joint_position,
        joint_velocity,
        fighter,
    ):
        if fighter not in (0, 1) or len(position) != self.batch_size:
            raise ValueError("Invalid native opponent role or population")
        require_finite(
            (position, quaternion, omega_b, joint_position, joint_velocity),
            "native opponent state",
        )
        # Last actions for the other fighter never enter this policy's input.
        last = torch.zeros_like(joint_position)
        last[:, fighter] = self.last_action
        proprioception = native_proprioception(
            omega_b, joint_position, joint_velocity, last, self.default
        )
        current = proprioception[:, fighter]
        self.history = (
            current[:, None].expand(-1, 10, -1).clone()
            if self.history is None
            else torch.cat((self.history[:, 1:], current[:, None]), 1)
        )
        task = native_task_observation(position, quaternion, proprioception)[
            :, fighter
        ]
        self.last_inputs = (task, self.history.flatten(1), current)
        action = self.policy(*self.last_inputs)
        require_finite(action, "native opponent action")
        self.last_action.copy_(action)
        return action, self.default + self.scale * action
