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


import inspect
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from loguru import logger


class DeepseekV3MLP(nn.Module):
    def __init__(self, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )
        return down_proj


class DeepseekV3TopkRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        top_k: int,
        n_routed_experts: int,
        routed_scaling_factor: float,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.scores = None
        self.router_logits = None

        self.linear_layer = nn.Linear(hidden_size, n_routed_experts)
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.n_routed_experts)
        )

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(
            -1, self.n_routed_experts
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = self.linear_layer(hidden_states.type(torch.float32))
        scores = router_logits.sigmoid()
        self.router_logits = router_logits
        self.scores = scores
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class DeepseekV3MoE(nn.Module):
    """A mixed expert module containing shared experts."""

    def __init__(
        self,
        hidden_size: int,
        top_k: int,
        n_routed_experts: int,
        routed_scaling_factor: float,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        moe_intermediate_size: int,
        n_shared_experts: int,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=hidden_size,
                    intermediate_size=moe_intermediate_size,
                )
                for _ in range(n_routed_experts)
            ]
        )
        self.gate = DeepseekV3TopkRouter(
            hidden_size=hidden_size,
            top_k=top_k,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size * n_shared_experts,
        )

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(
            hidden_states, dtype=topk_weights.dtype
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output
                )

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(
            hidden_states, topk_indices, topk_weights
        ).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class MoEMLP(nn.Module):
    def __init__(
        self,
        obs_dim_dict: dict,
        module_config_dict: dict,
        num_actions: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict
        self.num_actions = num_actions

        self.projection_dim = module_config_dict["projection_dim"]
        self.hidden_dim = module_config_dict["hidden_dim"]
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.moe_routed_scaling_factor = module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        self.moe_n_group = module_config_dict.get("moe_n_group", 1)
        self.moe_topk_group = module_config_dict.get("moe_topk_group", 1)
        self.moe_norm_topk_prob = module_config_dict.get(
            "moe_norm_topk_prob", True
        )

        self._calculate_input_dim()
        self._calculate_output_dim()

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        self.gate = DeepseekV3TopkRouter(
            hidden_size=self.projection_dim,
            top_k=self.top_k,
            n_routed_experts=self.num_fine_experts,
            routed_scaling_factor=self.moe_routed_scaling_factor,
            n_group=self.moe_n_group,
            topk_group=self.moe_topk_group,
            norm_topk_prob=self.moe_norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=self.projection_dim,
                    intermediate_size=self.hidden_dim,
                )
                for _ in range(self.num_fine_experts)
            ]
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=self.projection_dim,
            intermediate_size=self.hidden_dim * self.num_shared_experts,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("tokens_per_expert", None, persistent=False)

    def compute_load_balancing_loss(self) -> torch.Tensor:
        if self.router_probs is None or self.tokens_per_expert is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        num_tokens = self.router_probs.shape[0]
        if num_tokens == 0:
            # Avoid division by zero if batch size is zero
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        # Fraction of tokens dispatched to each expert f_i
        # self.tokens_per_expert has shape [num_fine_experts]
        fraction_tokens_per_expert = (
            self.tokens_per_expert.float() / num_tokens
        )

        # Average router probability for each expert P_i
        # self.router_probs has shape [num_tokens, num_fine_experts]
        avg_router_prob_per_expert = torch.mean(self.router_probs, dim=0)

        # Compute the loss: N * sum(f_i * P_i)
        load_balancing_loss = self.num_fine_experts * torch.sum(
            fraction_tokens_per_expert * avg_router_prob_per_expert
        )

        # Multiply by the scaling factor alpha
        return load_balancing_loss

    def compute_bound_loss(self) -> torch.Tensor:
        # penalize out of bound actions

        lb_loss = (
            torch.relu(self.action_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.action_output_ub)
            .square()
            .mean()
        )

        return (lb_loss + ub_loss) * 0.5

    def _calculate_input_dim(self):
        input_dim = 0
        for each_input in self.module_config_dict["input_dim"]:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                input_dim += int(each_input)  # Ensure int conversion
            elif isinstance(each_input, str) and each_input.isdigit():
                input_dim += int(each_input)
            else:
                raise ValueError(f"Unknown input type: {each_input}")
        self.input_dim = input_dim

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale

        self.register_buffer("action_output_lb", lb[None, :])
        self.register_buffer("action_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            assert isinstance(each_output, int), (
                f"Output dim placeholder not replaced: {each_output}"
            )
            output_dim += each_output
        self.output_dim = output_dim

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(hidden_states)
        tokens_per_expert = torch.zeros(
            self.num_fine_experts,
            device=hidden_states.device,
            dtype=torch.long,
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_fine_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(self.num_fine_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, topk_pos_indices = torch.where(mask)
            if token_indices.numel() > 0:
                tokens_per_expert[expert_idx] = token_indices.numel()
                expert_weights = topk_weights[token_indices, topk_pos_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output
                )

        self.tokens_per_expert = tokens_per_expert
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, (
            f"Expected input shape [B, D_in], but got {x.shape}"
        )
        projected_x = self.input_projection(x)
        residuals = projected_x
        topk_indices, topk_weights = self.gate(projected_x)
        self.router_probs = F.softmax(
            self.gate.router_logits,
            dim=-1,
            dtype=torch.float32,
        )
        routed_output = self.moe(projected_x, topk_indices, topk_weights)
        shared_output = self.shared_experts(residuals)
        combined_hidden = routed_output + shared_output
        self.output_actions = self.output_mlp(combined_hidden)

        if self.clamp_actor_output:
            # self.output_actions = self.output_actions.clamp(
            #     min=self.action_output_lb,
            #     max=self.action_output_ub,
            # )
            self.bound_loss = self.compute_bound_loss()

        return self.output_actions


class MoEMLPV2(nn.Module):
    def __init__(
        self,
        obs_serializer,
        module_config_dict: dict,
    ):
        super().__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict

        self.projection_dim = module_config_dict["projection_dim"]
        self.hidden_dim = module_config_dict["hidden_dim"]
        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.moe_routed_scaling_factor = module_config_dict.get(
            "moe_routed_scaling_factor", 1.0
        )
        self.moe_n_group = module_config_dict.get("moe_n_group", 1)
        self.moe_topk_group = module_config_dict.get("moe_topk_group", 1)
        self.moe_norm_topk_prob = module_config_dict.get(
            "moe_norm_topk_prob", True
        )

        self._calculate_output_dim()

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.input_dim = sum(
            [
                self.obs_dim_dict[each_input]
                * self.obs_seq_len_dict[each_input]
                for each_input in self.obs_dim_dict
            ]
        )

        self.input_projection = nn.Linear(
            self.input_dim,
            self.projection_dim,
            bias=True,
        )

        self.gate = DeepseekV3TopkRouter(
            hidden_size=self.projection_dim,
            top_k=self.top_k,
            n_routed_experts=self.num_fine_experts,
            routed_scaling_factor=self.moe_routed_scaling_factor,
            n_group=self.moe_n_group,
            topk_group=self.moe_topk_group,
            norm_topk_prob=self.moe_norm_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(
                    hidden_size=self.projection_dim,
                    intermediate_size=self.hidden_dim,
                )
                for _ in range(self.num_fine_experts)
            ]
        )
        self.shared_experts = DeepseekV3MLP(
            hidden_size=self.projection_dim,
            intermediate_size=self.hidden_dim * self.num_shared_experts,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

        self.register_buffer("router_probs", None, persistent=False)
        self.register_buffer("tokens_per_expert", None, persistent=False)

    def compute_load_balancing_loss(self) -> torch.Tensor:
        # Implements the load balancing loss from Switch Transformers (https://arxiv.org/abs/2101.03961)
        # loss = alpha * N * sum(f_i * P_i)
        # N = num_experts, f_i = fraction of tokens dispatched to expert i, P_i = avg router prob for expert i

        if self.router_probs is None or self.tokens_per_expert is None:
            # Return zero loss if forward pass hasn't happened or buffers are not populated
            # Try to get the device from a parameter that is guaranteed to exist
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        num_tokens = self.router_probs.shape[0]
        if num_tokens == 0:
            # Avoid division by zero if batch size is zero
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)

        # Fraction of tokens dispatched to each expert f_i
        # self.tokens_per_expert has shape [num_fine_experts]
        fraction_tokens_per_expert = (
            self.tokens_per_expert.float() / num_tokens
        )

        # Average router probability for each expert P_i
        # self.router_probs has shape [num_tokens, num_fine_experts]
        avg_router_prob_per_expert = torch.mean(self.router_probs, dim=0)

        # Compute the loss: N * sum(f_i * P_i)
        load_balancing_loss = self.num_fine_experts * torch.sum(
            fraction_tokens_per_expert * avg_router_prob_per_expert
        )

        # Multiply by the scaling factor alpha
        return load_balancing_loss

    def compute_bound_loss(self) -> torch.Tensor:
        lb_loss = (
            torch.relu(self.action_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.action_output_ub)
            .square()
            .mean()
        )

        return (lb_loss + ub_loss) * 0.5

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale

        self.register_buffer("action_output_lb", lb[None, :])
        self.register_buffer("action_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            assert isinstance(each_output, int), (
                f"Output dim placeholder not replaced: {each_output}"
            )
            output_dim += each_output
        self.output_dim = output_dim

    def moe(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        final_hidden_states = torch.zeros_like(hidden_states)
        tokens_per_expert = torch.zeros(
            self.num_fine_experts,
            device=hidden_states.device,
            dtype=torch.long,
        )
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=self.num_fine_experts
        )
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(self.num_fine_experts):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, topk_pos_indices = torch.where(mask)
            if token_indices.numel() > 0:
                tokens_per_expert[expert_idx] = token_indices.numel()
                expert_weights = topk_weights[token_indices, topk_pos_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0, token_indices, weighted_output
                )

        self.tokens_per_expert = tokens_per_expert
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, (
            f"Expected input shape [B, D_in], but got {x.shape}"
        )
        projected_x = self.input_projection(x)
        residuals = projected_x
        topk_indices, topk_weights = self.gate(projected_x)
        self.router_probs = F.softmax(
            self.gate.router_logits,
            dim=-1,
            dtype=torch.float32,
        )
        routed_output = self.moe(projected_x, topk_indices, topk_weights)
        shared_output = self.shared_experts(residuals)
        combined_hidden = routed_output + shared_output
        self.output_actions = self.output_head(
            self.output_mlp(combined_hidden)
        )

        return self.output_actions


class MLP(nn.Module):
    def __init__(self, obs_serializer, module_config_dict):
        super(MLP, self).__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict
        self.input_dim = obs_serializer.obs_flat_dim

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get("enabled", False)
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.predict_local_body_pos = module_config_dict.get(
            "predict_local_body_pos",
            False,
        )

        self.use_layernorm = module_config_dict.get("use_layernorm", False)

        self._calculate_output_dim()
        self._build_mlp(self.module_config_dict.layer_config)

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale
        self.register_buffer("actor_output_lb", lb[None, :])
        self.register_buffer("actor_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(
                    f"{current_function_name} - Unknown output type: {each_output}"
                )
        self.output_dim = output_dim

    def _build_mlp(self, layer_config):
        layers = []
        hidden_dims = layer_config["hidden_dims"]
        output_dim = self.output_dim
        activation = getattr(nn, layer_config["activation"])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
            else:
                if self.use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dims[i]))
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        output = self.module(input)
        if self.clamp_actor_output:
            if hasattr(self, "actor_output_lb") and hasattr(
                self, "actor_output_ub"
            ):
                if self.actor_output_lb.device != output.device:
                    self.actor_output_lb = self.actor_output_lb.to(
                        output.device
                    )
                    self.actor_output_ub = self.actor_output_ub.to(
                        output.device
                    )
                self.output_actions = output
        return output

    def compute_bound_loss(self) -> torch.Tensor:
        lb_loss = (
            torch.relu(self.actor_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.actor_output_ub)
            .square()
            .mean()
        )
        return (lb_loss + ub_loss) * 0.5


class TFStudent(nn.Module):
    def __init__(self, obs_serializer, module_config_dict):
        super(TFStudent, self).__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict
        self.input_dim = obs_serializer.obs_flat_dim

        self.patch_size = module_config_dict.get("patch_size", 2)
        self.num_obs_tokens = (
            self.obs_seq_len_dict["student_actor_realworld_obs_seq"]
            // self.patch_size
        )

        self.fut_patch_size = module_config_dict.get("fut_patch_size", 1)
        self.num_fut_ref_tokens = (
            self.obs_seq_len_dict.get("fut_ref_motion_seq", 0)
            // self.fut_patch_size
        )

        self.clamp_actor_output = module_config_dict.get(
            "clamp_output", {}
        ).get(
            "enabled",
            False,
        )
        if self.clamp_actor_output:
            self._build_output_bounds()

        self.predict_local_body_pos = module_config_dict.get(
            "predict_local_body_pos",
            False,
        )

        self.predict_local_body_vel = module_config_dict.get(
            "predict_local_body_vel",
            False,
        )

        self.predict_root_lin_vel = module_config_dict.get(
            "predict_root_lin_vel",
            False,
        )

        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _build_output_bounds(self):
        default_dof_pos = torch.tensor(
            [
                self.module_config_dict.clamp_output.default_dof_pos_dict[dof]
                for dof in self.module_config_dict.clamp_output.dof_order
            ]
        )
        action_scale = self.module_config_dict.clamp_output.action_scale
        lb = (
            torch.tensor(self.module_config_dict.clamp_output.raw_lower_bound)
            - default_dof_pos
        ) / action_scale
        ub = (
            torch.tensor(self.module_config_dict.clamp_output.raw_upper_bound)
            - default_dof_pos
        ) / action_scale
        self.register_buffer("actor_output_lb", lb[None, :])
        self.register_buffer("actor_output_ub", ub[None, :])

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        hidden_dim = layer_config["hidden_dim"]

        # Query tokens for action and auxiliary outputs
        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.aux_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        obs_sin_pe = get_sinusoid_encoding_table(
            self.num_obs_tokens, hidden_dim
        )[0]
        self.register_buffer("obs_pe", obs_sin_pe, persistent=False)

        # Projection layer for observation sequence patches
        self.obs_proj_layer = nn.Linear(
            self.obs_dim_dict["student_actor_realworld_obs_seq"]
            * self.patch_size,
            hidden_dim,
        )

        self.fut_ref_proj_layer = nn.Linear(
            self.obs_dim_dict["fut_ref_motion_seq"] * self.fut_patch_size,
            hidden_dim,
        )

        # Sinusoidal positional encoding for future reference sequence
        fut_ref_sin_pe = get_sinusoid_encoding_table(
            self.num_fut_ref_tokens, hidden_dim
        )[0]
        self.register_buffer("fut_ref_pe", fut_ref_sin_pe, persistent=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.0,
            activation=nn.SiLU(),
        )
        self.tf_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=layer_config["num_layers"]
        )

        # Output layers
        self.action_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.output_dim),
        )

        # Prediction heads
        if self.predict_local_body_pos:
            self.local_body_pos_pred_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(
                    128, self.module_config_dict.pred_local_body_pos_dim
                ),
            )

        if self.predict_local_body_vel:
            self.local_body_vel_pred_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(
                    128, self.module_config_dict.pred_local_body_vel_dim
                ),
            )

        if self.predict_root_lin_vel:
            self.root_lin_vel_pred_head = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.SiLU(),
                nn.Linear(128, 3),
            )

    def forward(self, input_tensor):
        seq_obs_dict = self.obs_serializer.deserialize(input_tensor)
        obs_seq = seq_obs_dict["student_actor_realworld_obs_seq"]

        # Process observation sequence patches
        obs_seq_patches = rearrange(
            obs_seq, "B (T P) C -> B T (P C)", P=self.patch_size
        )
        assert obs_seq_patches.shape[1] == self.num_obs_tokens

        obs_seq_patches = self.obs_proj_layer(obs_seq_patches)
        obs_seq_patches = obs_seq_patches + self.obs_pe

        # Create query sequence (action + aux queries + observation patches)
        query_seq = torch.cat(
            [
                self.action_query.repeat(obs_seq_patches.shape[0], 1, 1),
                self.aux_query.repeat(obs_seq_patches.shape[0], 1, 1),
                obs_seq_patches,
            ],
            dim=1,
        )  # [B, 2 + num_obs_tokens, hidden_dim]

        fut_ref_seq = seq_obs_dict["fut_ref_motion_seq"]
        fut_ref_valid_mask = seq_obs_dict.get("fut_ref_valid_mask", None)

        # Process future reference sequence
        fut_ref_patches = rearrange(
            fut_ref_seq, "B (T P) C -> B T (P C)", P=self.fut_patch_size
        )
        fut_ref_patches = self.fut_ref_proj_layer(fut_ref_patches)
        fut_ref_patches = fut_ref_patches + self.fut_ref_pe

        # Handle validity mask for future reference
        memory_key_padding_mask = None
        if fut_ref_valid_mask is not None:
            fut_ref_valid_mask = fut_ref_valid_mask.squeeze(-1).bool()
            if self.fut_patch_size > 1:
                fut_ref_valid_mask = rearrange(
                    fut_ref_valid_mask,
                    "B (T P) -> B T P",
                    P=self.fut_patch_size,
                ).any(dim=-1)
            memory_key_padding_mask = ~fut_ref_valid_mask

        # Apply transformer decoder (cross-attention)
        out_seq = self.tf_decoder(
            tgt=query_seq,
            memory=fut_ref_patches,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Extract outputs
        action_out = self.action_output_layer(out_seq[:, 0, :])  # Action token
        aux_out = out_seq[:, 1, :]  # Auxiliary token

        # Handle action bounds
        if self.clamp_actor_output:
            if hasattr(self, "actor_output_lb") and hasattr(
                self, "actor_output_ub"
            ):
                if self.actor_output_lb.device != action_out.device:
                    self.actor_output_lb = self.actor_output_lb.to(
                        action_out.device
                    )
                    self.actor_output_ub = self.actor_output_ub.to(
                        action_out.device
                    )
                self.output_actions = action_out

        # Prediction heads
        if self.predict_local_body_pos:
            self.predicted_local_body_pos = self.local_body_pos_pred_head(
                aux_out
            )

        if self.predict_local_body_vel:
            self.predicted_local_body_vel = self.local_body_vel_pred_head(
                aux_out
            )

        if self.predict_root_lin_vel:
            self.predicted_root_lin_vel = self.root_lin_vel_pred_head(aux_out)

        return action_out

    def compute_bound_loss(self) -> torch.Tensor:
        lb_loss = (
            torch.relu(self.actor_output_lb - self.output_actions)
            .square()
            .mean()
        )
        ub_loss = (
            torch.relu(self.output_actions - self.actor_output_ub)
            .square()
            .mean()
        )

        return (lb_loss + ub_loss) * 0.5

    def compute_local_body_pos_reg_loss(
        self, gt_local_body_pos_extend_flat
    ) -> torch.Tensor:
        if not self.predict_local_body_pos or not hasattr(
            self, "predicted_local_body_pos"
        ):
            return torch.tensor(
                0.0, device=gt_local_body_pos_extend_flat.device
            )
        return nn.functional.mse_loss(
            self.predicted_local_body_pos, gt_local_body_pos_extend_flat
        )

    def compute_local_body_vel_reg_loss(
        self, gt_local_body_vel_extend_flat
    ) -> torch.Tensor:
        if not self.predict_local_body_vel or not hasattr(
            self, "predicted_local_body_vel"
        ):
            return torch.tensor(
                0.0, device=gt_local_body_vel_extend_flat.device
            )
        return nn.functional.mse_loss(
            self.predicted_local_body_vel, gt_local_body_vel_extend_flat
        )

    def compute_root_lin_vel_reg_loss(self, gt_root_lin_vel) -> torch.Tensor:
        if not self.predict_root_lin_vel or not hasattr(
            self, "predicted_root_lin_vel"
        ):
            return torch.tensor(0.0, device=gt_root_lin_vel.device)
        return nn.functional.mse_loss(
            self.predicted_root_lin_vel, gt_root_lin_vel
        )


class RunningMeanStdNormalizer(nn.Module):
    def __init__(self, feature_dim: int, epsilon: float = 1e-8):
        """Initializes a running mean and standard deviation normalizer.

        Normalization is performed independently for each feature dimension.
        Buffers will be on the default device (CPU) unless the module instance
        is moved using .to(device).

        Args:
            feature_dim (int): The number of features (D) to normalize
                independently.
            epsilon (float): A small value to prevent division by zero and to
                initialize count. This value is stored in the 'epsilon_val'
                buffer.

        """
        super(RunningMeanStdNormalizer, self).__init__()
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")
        self.feature_dim = feature_dim

        self.register_buffer(
            "running_mean", torch.zeros(feature_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "running_var", torch.ones(feature_dim, dtype=torch.float32)
        )
        # running_count tracks the number of samples (from batches) processed.
        self.register_buffer(
            "running_count", torch.tensor(epsilon, dtype=torch.float32)
        )
        self.register_buffer(
            "epsilon_val", torch.tensor(epsilon, dtype=torch.float32)
        )

    def update(self, x: torch.Tensor):
        """Updates the running mean and variance.

        The internal state (buffers) remains on its current device.
        Input x is moved to the buffer's device for computation.

        Args:
            x (torch.Tensor): A tensor of shape [B, D]
                              or [D] (a single D-dimensional feature vector).

        """
        if not (x.ndim == 2 and x.shape[1] == self.feature_dim) and not (
            x.ndim == 1 and x.shape[0] == self.feature_dim
        ):
            raise ValueError(
                f"Input x for update must be shape "
                f"[B, {self.feature_dim}] or [{self.feature_dim}], "
                f"got {x.shape}"
            )

        if x.ndim == 1:  # Single D-dimensional sample
            x = x.unsqueeze(0)  # Treat as a batch of 1

        buffer_device = self.running_mean.device
        x_on_buffer_device = x.to(buffer_device)

        batch_size_float = torch.tensor(
            x_on_buffer_device.shape[0],
            dtype=torch.float32,
            device=buffer_device,
        )

        if batch_size_float == 0:
            return

        batch_mean = torch.mean(x_on_buffer_device, dim=0)  # Shape [D]
        batch_var = torch.var(
            x_on_buffer_device, dim=0, unbiased=False
        )  # Shape [D]

        mean1, var1, n1 = (
            self.running_mean,
            self.running_var,
            self.running_count,
        )
        mean2, var2, n2 = batch_mean, batch_var, batch_size_float

        n_total = n1 + n2
        delta_mean = mean2 - mean1  # Shape [D]
        new_mean = mean1 + delta_mean * (n2 / n_total)  # Shape [D]

        m2_1 = var1 * n1
        m2_2 = var2 * n2
        new_m2 = (
            m2_1 + m2_2 + torch.square(delta_mean) * (n1 * n2 / n_total)
        )  # Shape [D]
        new_var = new_m2 / n_total  # Shape [D]

        self.running_mean = new_mean
        self.running_var = torch.max(
            new_var, self.epsilon_val.to(buffer_device)
        )  # Epsilon broadcast if needed
        self.running_count = n_total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes multi-dimensional data.

        Using the running mean and variance for each feature.
        Output tensor is on the same device as the input x.

        Args:
            x (torch.Tensor): A tensor of shape [B, D] or [D] to normalize.

        Returns:
            torch.Tensor: The normalized data, on the same device as input x.

        """
        original_x_ndim = x.ndim
        if not (
            original_x_ndim == 2 and x.shape[1] == self.feature_dim
        ) and not (original_x_ndim == 1 and x.shape[0] == self.feature_dim):
            raise ValueError(
                f"Input x for normalize must be shape [B, {self.feature_dim}] "
                f"or [{self.feature_dim}], got {x.shape}"
            )

        if original_x_ndim == 1:
            x = x.unsqueeze(0)  # Treat as batch of 1

        buffer_device = self.running_mean.device
        original_x_device = x.device

        x_on_buffer_device = x.to(buffer_device)

        mean = self.running_mean
        var = self.running_var
        eps = self.epsilon_val.to(buffer_device)

        normalized_x_on_buffer_device = (
            x_on_buffer_device - mean
        ) / torch.sqrt(var + eps)

        result = normalized_x_on_buffer_device.to(original_x_device)
        return result.squeeze(0) if original_x_ndim == 1 else result

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalizes multi-dimensional data.

        Using the running mean and variance for each feature.
        Output tensor is on the same device as the input x_norm.

        Args:
            x_norm (torch.Tensor): A tensor of shape [B, D] or [D] of
                normalized values.

        Returns:
            torch.Tensor: The denormalized (original scale) data, on the same
                device as input x_norm.

        """
        original_x_norm_ndim = x_norm.ndim
        if not (
            original_x_norm_ndim == 2 and x_norm.shape[1] == self.feature_dim
        ) and not (
            original_x_norm_ndim == 1 and x_norm.shape[0] == self.feature_dim
        ):
            raise ValueError(
                f"Input x_norm for denormalize must be shape "
                f"[B, {self.feature_dim}] or [{self.feature_dim}], "
                f"got {x_norm.shape}"
            )

        if original_x_norm_ndim == 1:
            x_norm = x_norm.unsqueeze(0)  # Treat as batch of 1

        buffer_device = self.running_mean.device
        original_x_norm_device = x_norm.device

        x_norm_on_buffer_device = x_norm.to(buffer_device)

        mean = self.running_mean
        var = self.running_var
        eps = self.epsilon_val.to(buffer_device)

        denormalized_x_on_buffer_device = (
            x_norm_on_buffer_device * torch.sqrt(var + eps) + mean
        )

        result = denormalized_x_on_buffer_device.to(original_x_norm_device)
        return result.squeeze(0) if original_x_norm_ndim == 1 else result

    def get_state(self) -> dict:
        """Returns the current state of the normalizer.

        Tensors are on their current buffer device.
        """
        return {
            "feature_dim": self.feature_dim,
            "running_mean": self.running_mean.clone(),
            "running_var": self.running_var.clone(),
            "running_count": self.running_count.clone(),
            "epsilon_val": self.epsilon_val.clone(),
        }

    def set_state(self, state: dict, new_buffer_device: str = None):
        """Sets the state of the normalizer from a state dictionary.

        Args:
            state (dict): A dictionary containing:
                'feature_dim', 'running_mean', 'running_var',
                          'running_count', and 'epsilon_val'.
            new_buffer_device (str, optional): The device to move the loaded
                state and internal buffers to (e.g., 'cpu', 'cuda:0'). If None,
                state is loaded to the current device of the module's buffers.

        """
        target_device = (
            torch.device(new_buffer_device)
            if new_buffer_device
            else self.running_mean.device
        )

        required_keys = [
            "feature_dim",
            "running_mean",
            "running_var",
            "running_count",
            "epsilon_val",
        ]
        for key in required_keys:
            if key not in state:
                raise KeyError(
                    f"Key '{key}' not found in the "
                    f"provided state dictionary for RunningMeanStdNormalizer."
                )

        loaded_feature_dim = state["feature_dim"]
        if loaded_feature_dim != self.feature_dim:
            raise ValueError(
                f"Mismatched feature_dim in state. "
                f"Expected {self.feature_dim}, got {loaded_feature_dim}"
            )

        self.running_mean = state["running_mean"].clone().to(target_device)
        self.running_var = state["running_var"].clone().to(target_device)
        self.running_count = state["running_count"].clone().to(target_device)
        self.epsilon_val = state["epsilon_val"].clone().to(target_device)

        if self.running_mean.shape[0] != self.feature_dim:
            self.running_mean = torch.zeros(
                self.feature_dim, dtype=torch.float32, device=target_device
            )
            logger.warning(
                "Reinitialized running_mean due to feature_dim mismatch "
                "during set_state. This should not happen if state is valid."
            )
        if self.running_var.shape[0] != self.feature_dim:
            self.running_var = torch.ones(
                self.feature_dim, dtype=torch.float32, device=target_device
            )
            logger.warning(
                "Reinitialized running_var due to feature_dim mismatch during "
                "set_state. This should not happen if state is valid."
            )


def get_sinusoid_encoding_table(seq_len: int, hidden_dim: int) -> torch.Tensor:
    """Creates a sinusoidal positional encoding table.

    Args:
        seq_len: The number of positions to encode.
        hidden_dim: The hidden dimension of the encoding.

    Returns:
        A tensor of shape (1, seq_len, hidden_dim) containing the sinusoidal
        positional encoding table.

    """

    def get_position_angle_vec(position):
        return [
            position
            / torch.pow(10000, torch.tensor(2 * (hid_j // 2) / hidden_dim))
            for hid_j in range(hidden_dim)
        ]

    sinusoid_table = torch.tensor(
        [get_position_angle_vec(pos_i) for pos_i in range(seq_len)],
        dtype=torch.float32,
    )
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table[None, ...]
