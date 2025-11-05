# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from __future__ import annotations
from copy import deepcopy
from typing import List, Union

import torch
import torch.nn as nn
from torch.distributions import Normal

from holomotion.src.modules.network_modules import MLP

from loguru import logger


class SimpleObsSerializer:
    """Simple observation serializer for flat observations."""

    def __init__(self, obs_dim):
        self.obs_dim_dict = {"obs": obs_dim}
        self.obs_seq_len_dict = {"obs": 1}
        self.obs_flat_dim = obs_dim

    def serialize(self, obs_list):
        return obs_list[0]

    def deserialize(self, obs_tensor):
        return {"obs": obs_tensor[:, None, :]}  # Add sequence dimension


class ObsSeqSerializer:
    def __init__(self, schema_list: List[dict]):
        self.schema_list = schema_list
        self.obs_dim_dict = self._build_obs_dim_dict()
        self.obs_seq_len_dict = self._build_obs_seq_len_dict()
        self.obs_flat_dim = self._build_obs_flat_dim()

    def _build_obs_dim_dict(self):
        obs_dim_dict = {}
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            feat_dim = schema["feat_dim"]
            obs_dim_dict[obs_name] = feat_dim
        return obs_dim_dict

    def _build_obs_seq_len_dict(self):
        obs_seq_len_dict = {}
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            seq_len = schema["seq_len"]
            obs_seq_len_dict[obs_name] = seq_len
        return obs_seq_len_dict

    def _build_obs_flat_dim(self):
        obs_flat_dim = 0
        for schema in self.schema_list:
            seq_len = schema["seq_len"]
            feat_dim = schema["feat_dim"]
            obs_flat_dim += seq_len * feat_dim
        return obs_flat_dim

    def serialize(self, obs_seq_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(obs_seq_list) == len(self.schema_list)
        B = obs_seq_list[0].shape[0]
        output_tensor = []
        for schema, obs_seq in zip(self.schema_list, obs_seq_list):
            assert obs_seq.ndim == 3
            assert obs_seq.shape[0] == B
            assert obs_seq.shape[1] == schema["seq_len"]
            assert obs_seq.shape[2] == schema["feat_dim"]
            output_tensor.append(obs_seq.reshape(B, -1))
        return torch.cat(output_tensor, dim=-1)

    def deserialize(self, obs_seq_tensor: torch.Tensor) -> List[torch.Tensor]:
        assert obs_seq_tensor.ndim == 2
        output_dict = {}
        array_start_idx = 0
        B = obs_seq_tensor.shape[0]
        for schema in self.schema_list:
            obs_name = schema["obs_name"]
            seq_len = schema["seq_len"]
            feat_dim = schema["feat_dim"]
            obs_size = seq_len * feat_dim
            array_end_idx = array_start_idx + obs_size
            output_dict[obs_name] = obs_seq_tensor[
                :, array_start_idx:array_end_idx
            ].reshape(B, seq_len, feat_dim)
            array_start_idx = array_end_idx

        return output_dict


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict: Union[dict, ObsSeqSerializer],
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
    ):
        super(PPOActor, self).__init__()

        self.use_logvar = module_config_dict.get("use_logvar", False)

        module_config_dict = self._process_module_config(
            module_config_dict, num_actions
        )

        self.actor_net_type = module_config_dict.get("type", "MLP")
        print(self.actor_net_type)

        if self.actor_net_type == "MLP":
            self.actor_module = MLP(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        else:
            raise NotImplementedError

        self.fix_sigma = module_config_dict.get("fix_sigma", False)
        self.max_sigma = module_config_dict.get("max_sigma", 1.0)
        self.min_sigma = module_config_dict.get("min_sigma", 0.1)

        # Noise std type aligned with rsl_rl ("scalar" or "log").
        # Fallback to legacy use_logvar if provided.
        self.noise_std_type = module_config_dict.get(
            "noise_std_type", "scalar"
        ).lower()
        if self.use_logvar:
            self.noise_std_type = "log"

        # Action noise parameters (kept outside nets so optimizer updates them)
        if self.noise_std_type == "log":
            logger.info("Using log-std parameterization for action noise")
            self.log_std = nn.Parameter(
                torch.log(torch.ones(num_actions) * init_noise_std)
            )
            if self.fix_sigma:
                self.log_std.requires_grad = False
        else:  # scalar (default)
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            if self.fix_sigma:
                self.std.requires_grad = False
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict["output_dim"]):
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"][idx] = num_actions
        return module_config_dict

    @property
    def actor(self):
        return self.actor_module

    def reset(self, dones=None):
        pass

    def forward(
        self,
        actor_obs: torch.Tensor,
        actions: torch.Tensor = None,
        mode: str = "sampling",
    ):
        """Forward pass for PPOActor supporting rollout, training, and inference.

        Args:
            actor_obs: [B, obs_dim] normalized policy observations.
            actions: [B, action_dim] actions for log-prob evaluation (used when mode="logp").
            mode: one of {"sampling", "logp", "inference"}.

        Returns:
            actions_out: [B, action_dim]
            actions_log_prob: [B] or None
            mu: [B, action_dim]
            sigma: [B, action_dim]
            entropy: [B] or None
        """
        if mode not in ("sampling", "logp", "inference"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.update_distribution(actor_obs)

        if mode == "inference":
            actions_out = self.distribution.mean
            actions_log_prob = None
            entropy = None
        elif mode == "sampling":
            actions_out = self.distribution.sample()
            actions_log_prob = self.distribution.log_prob(actions_out).sum(
                dim=-1
            )
            entropy = self.distribution.entropy().sum(dim=-1)
        else:  # mode == "logp"
            if actions is None:
                raise ValueError("actions must be provided when mode='logp'")
            actions_out = actions
            actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1)
            entropy = self.distribution.entropy().sum(dim=-1)

        mu = self.distribution.mean
        sigma = self.distribution.stddev
        return actions_out, actions_log_prob, mu, sigma, entropy

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        # Resolve std according to parameterization
        if hasattr(self, "log_std"):
            std_val = torch.exp(self.log_std)
        else:
            std_val = self.std

        std_val = torch.clamp(std_val, min=1e-6)
        self.distribution = Normal(mean, std_val)

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        if not self.use_logvar:
            self.std.to("cpu")


class PPOCritic(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(PPOCritic, self).__init__()
        self.critic_net_type = module_config_dict.get("type", "MLP")
        if self.critic_net_type == "MLP":
            self.critic_module = MLP(
                obs_serializer=obs_dim_dict,
                module_config_dict=module_config_dict,
            )
        else:
            raise NotImplementedError

    def reset(self, dones=None):
        pass

    def forward(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for PPOCritic.

        Args:
            critic_obs: [B, critic_obs_dim] normalized critic observations.

        Returns:
            values: [B, 1] state values
        """
        return self.critic_module(critic_obs)
