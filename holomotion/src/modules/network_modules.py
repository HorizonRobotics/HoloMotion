# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

import math
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(
        self,
        shape,
        eps: float = 1e-2,
        until: int | None = None,
        *,
        update_method: str = "cumulative",
        ema_momentum: float | None = None,
    ):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except
                batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns
                input values until the sum of batch sizes
            exceeds it.
            update_method:
                One of {"cumulative", "ema"}.
                - "cumulative": count-based updates (legacy behavior).
                - "ema": EMA updates of mean and second moment.
            ema_momentum:
                EMA momentum in (0, 1]. Required when update_method == "ema".
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.update_method = str(update_method).lower()
        self.ema_momentum = (
            float(ema_momentum) if ema_momentum is not None else None
        )
        if self.update_method in ("count", "cumulative"):
            self.update_method = "cumulative"
        elif self.update_method in ("ema", "exp", "exponential"):
            self.update_method = "ema"
        else:
            raise ValueError(
                f"update_method must be one of {{'cumulative','ema'}}, got {update_method}"
            )
        if self.update_method == "ema":
            if self.ema_momentum is None:
                raise ValueError(
                    "ema_momentum must be provided when update_method == 'ema'"
                )
            if not (0.0 < self.ema_momentum <= 1.0):
                raise ValueError(
                    f"ema_momentum must be in (0, 1], got {self.ema_momentum}"
                )
        self.register_buffer("_mean", torch.zeros(shape)[None, ...])
        self.register_buffer("_var", torch.ones(shape)[None, ...])
        self.register_buffer("_std", torch.ones(shape)[None, ...])
        self.register_buffer("_ex2", torch.ones(shape)[None, ...])
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_last_sync_mean", torch.zeros(shape)[None, ...])
        self.register_buffer("_last_sync_var", torch.ones(shape)[None, ...])
        self.register_buffer(
            "_last_sync_count", torch.tensor(0, dtype=torch.long)
        )

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    def normalize_only(self, x):
        return (x - self._mean) / (self._std + self.eps)

    @torch.compiler.disable
    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them."""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        if self.update_method == "ema":
            m = float(self.ema_momentum)
            mean_x = torch.mean(x, dim=0, keepdim=True)
            ex2_x = torch.mean(x * x, dim=0, keepdim=True)
            self._mean.mul_(1.0 - m).add_(mean_x, alpha=m)
            self._ex2.mul_(1.0 - m).add_(ex2_x, alpha=m)
            var = torch.clamp(self._ex2 - self._mean * self._mean, min=0.0)
            self._var.copy_(var)
            self._std.copy_(torch.sqrt(self._var))
            return

        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
            var_x - self._var + delta_mean * (mean_x - self._mean)
        )
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

    def sync_stats_across_processes(self, accelerator):
        """Synchronize normalization statistics across distributed processes."""
        if accelerator.num_processes <= 1:
            return

        if self.update_method == "ema":
            # EMA stats are already running estimates.
            # Sync by averaging across ranks.
            mean_g = accelerator.reduce(
                self._mean.to(dtype=torch.float32), reduction="mean"
            )
            ex2_g = accelerator.reduce(
                self._ex2.to(dtype=torch.float32), reduction="mean"
            )
            var_g = torch.clamp(ex2_g - mean_g * mean_g, min=0.0)
            self._mean.copy_(mean_g.to(self._mean.dtype))
            self._ex2.copy_(ex2_g.to(self._ex2.dtype))
            self._var.copy_(var_g.to(self._var.dtype))
            self._std.copy_(torch.sqrt(self._var))
            return

        # Weighted synchronization with correction to avoid double counting
        device = self._mean.device
        count_local = self.count.to(device=device, dtype=torch.float32)
        mean_local = self._mean.to(device=device, dtype=torch.float32)
        var_local = self._var.to(device=device, dtype=torch.float32)

        # Local weighted sums
        sum_count = accelerator.reduce(count_local, reduction="sum")
        sum_mean_count = accelerator.reduce(
            mean_local * count_local, reduction="sum"
        )
        sum_ex2_count = accelerator.reduce(
            (var_local + mean_local * mean_local) * count_local,
            reduction="sum",
        )

        # Correct for replication of previously-synced global stats
        # across ranks.
        last_c = self._last_sync_count.to(device=device, dtype=torch.float32)
        if last_c.item() > 0:
            w_minus_1 = float(accelerator.num_processes - 1)
            last_mean = self._last_sync_mean.to(
                device=device, dtype=torch.float32
            )
            last_var = self._last_sync_var.to(
                device=device, dtype=torch.float32
            )
            sum_count = sum_count - w_minus_1 * last_c
            sum_mean_count = sum_mean_count - w_minus_1 * (last_mean * last_c)
            sum_ex2_count = sum_ex2_count - w_minus_1 * (
                (last_var + last_mean * last_mean) * last_c
            )

        if sum_count.item() <= 0:
            return

        global_mean = sum_mean_count / sum_count
        global_ex2 = sum_ex2_count / sum_count
        global_var = torch.clamp(
            global_ex2 - global_mean * global_mean, min=0.0
        )
        global_std = torch.sqrt(global_var)

        # Copy back (keep original buffer shapes)
        self._mean.copy_(global_mean.to(self._mean.dtype))
        self._var.copy_(global_var.to(self._var.dtype))
        self._std.copy_(global_std.to(self._std.dtype))
        # Set global sample count and remember snapshot for next correction
        self.count.copy_(sum_count.to(self.count.dtype))
        self._last_sync_mean.copy_(global_mean.to(self._last_sync_mean.dtype))
        self._last_sync_var.copy_(global_var.to(self._last_sync_var.dtype))
        self._last_sync_count.copy_(self.count)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_config_dict: dict,
    ):
        super().__init__()
        self.module_config_dict = module_config_dict
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim <= 0:
            raise ValueError(
                f"MLP input_dim must be positive, got {self.input_dim}"
            )
        if self.output_dim <= 0:
            raise ValueError(
                f"MLP output_dim must be positive, got {self.output_dim}"
            )

        def _make_norm(
            norm_type: str,
            dim: int,
            *,
            eps: float,
        ) -> nn.Module:
            t = str(norm_type).lower()
            if t in ("none", "identity", "null"):
                return nn.Identity()
            if t in ("layernorm", "ln"):
                return nn.LayerNorm(dim, eps=eps)
            if t in ("rmsnorm", "rms"):
                return RMSNorm(dim, eps=eps)
            raise ValueError(
                f"Unknown norm '{t}'. Expected one of {'none', 'layernorm', 'rmsnorm'}."
            )

        self.hidden_norm_type = module_config_dict.get("hidden_norm", "none")
        self.hidden_norm_eps = float(
            module_config_dict.get("hidden_norm_eps", 1.0e-6)
        )

        layer_config = self.module_config_dict["layer_config"]
        hidden_dims: list[int] = list(layer_config.get("hidden_dims", []))
        activation = getattr(nn, str(layer_config["activation"]))()

        layers: list[nn.Module] = []
        prev = self.input_dim
        for h in hidden_dims:
            h_i = int(h)
            layers.append(nn.Linear(prev, h_i))
            layers.append(
                _make_norm(
                    self.hidden_norm_type,
                    h_i,
                    eps=self.hidden_norm_eps,
                )
            )
            layers.append(activation)
            prev = h_i
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_head = nn.Linear(prev, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: [..., input_dim] assembled tensor observations.

        Returns:
            y: [..., output_dim]
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"MLP expects torch.Tensor input, got {type(x)}")
        h = self.trunk(x)
        return self.output_head(h)


class ConvMLP(nn.Module):
    """Conv1d + pooling history encoder with an MLP head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_config_dict: dict,
    ):
        super().__init__()
        self.module_config_dict = module_config_dict
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        layer_cfg = dict(module_config_dict.get("layer_config", {}))
        activation = str(layer_cfg.get("activation", "SiLU"))

        self.conv_channels = int(module_config_dict.get("conv_channels", 128))
        self.conv_layers = int(module_config_dict.get("conv_layers", 2))
        self.conv_kernel_size = int(
            module_config_dict.get("conv_kernel_size", 3)
        )
        self.pool_type = str(
            module_config_dict.get("pool_type", "avg")
        ).lower()

        conv_modules: list[nn.Module] = []
        padding = self.conv_kernel_size // 2
        in_ch = int(self.input_dim)
        for _ in range(self.conv_layers):
            conv_modules.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=self.conv_channels,
                    kernel_size=self.conv_kernel_size,
                    padding=padding,
                    bias=True,
                )
            )
            conv_modules.append(getattr(nn, activation)())
            in_ch = self.conv_channels

        conv_modules.append(nn.AdaptiveAvgPool1d(1))

        self.hist_encoder = nn.Sequential(*conv_modules)

        fused_dim = int(self.conv_channels + self.input_dim)
        self.mlp_head = MLP(
            input_dim=fused_dim,
            output_dim=int(self.output_dim),
            module_config_dict=module_config_dict,
        )

    def forward(self, hist_seq: torch.Tensor) -> torch.Tensor:
        ctx = self.hist_encoder(hist_seq.transpose(1, 2)).squeeze(-1)
        latest = hist_seq[:, -1, :]
        fused = torch.cat([ctx, latest], dim=-1)
        return self.mlp_head(fused)


class ReferenceMotionConvRouterEncoder(nn.Module):
    """Conv1d encoder for reference-motion router sequences."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        conv_channels: int = 128,
        conv_layers: int = 2,
        conv_kernel_size: int = 3,
        pool_type: str = "avg",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.conv_channels = int(conv_channels)
        self.conv_layers = int(conv_layers)
        self.conv_kernel_size = int(conv_kernel_size)
        self.pool_type = str(pool_type).lower()
        if self.input_dim <= 0:
            raise ValueError(
                f"input_dim must be positive, got {self.input_dim}"
            )
        if self.output_dim <= 0:
            raise ValueError(
                f"output_dim must be positive, got {self.output_dim}"
            )
        if self.conv_channels <= 0:
            raise ValueError(
                f"conv_channels must be positive, got {self.conv_channels}"
            )
        if self.conv_layers <= 0:
            raise ValueError(
                f"conv_layers must be positive, got {self.conv_layers}"
            )
        if self.conv_kernel_size <= 0:
            raise ValueError(
                f"conv_kernel_size must be positive, got {self.conv_kernel_size}"
            )
        if self.pool_type not in {"avg", "max"}:
            raise ValueError(
                f"pool_type must be one of {{'avg','max'}}, got {self.pool_type}"
            )

        padding = self.conv_kernel_size // 2
        conv_modules: list[nn.Module] = []
        in_ch = self.input_dim
        for _ in range(self.conv_layers):
            conv_modules.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=self.conv_channels,
                    kernel_size=self.conv_kernel_size,
                    padding=padding,
                    bias=True,
                )
            )
            conv_modules.append(nn.SiLU())
            in_ch = self.conv_channels
        self.temporal_trunk = nn.Sequential(*conv_modules)
        if self.pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        self.out_proj = nn.Sequential(
            nn.Linear(self.conv_channels, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 3:
            raise ValueError(
                f"Expected router seq with shape [B, T, D], got {tuple(seq.shape)}."
            )
        if int(seq.shape[-1]) != self.input_dim:
            raise ValueError(
                "Router seq dim mismatch: expected "
                f"{self.input_dim}, got {int(seq.shape[-1])}."
            )
        x = seq.transpose(1, 2)
        x = self.temporal_trunk(x)
        x = self.pool(x).squeeze(-1)
        return self.out_proj(x)


class SingleQueryAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        self.scale = float(self.d_model) ** -0.5
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim == 2:
            if tokens.ndim != 3:
                raise ValueError(
                    "SingleQueryAttentionPool expected [B, N, D] tokens for "
                    f"2D query, got {tuple(tokens.shape)}."
                )
            q = self.q_proj(query).unsqueeze(-2)
            k = self.k_proj(tokens)
            v = self.v_proj(tokens)
            attn = torch.softmax(
                (q * k).sum(dim=-1, keepdim=True) * self.scale,
                dim=-2,
            )
            return self.out_proj((attn * v).sum(dim=-2))
        if query.ndim == 3:
            if tokens.ndim != 4:
                raise ValueError(
                    "SingleQueryAttentionPool expected [B, T, N, D] tokens for "
                    f"3D query, got {tuple(tokens.shape)}."
                )
            q = self.q_proj(query).unsqueeze(-2)
            k = self.k_proj(tokens)
            v = self.v_proj(tokens)
            attn = torch.softmax(
                (q * k).sum(dim=-1, keepdim=True) * self.scale,
                dim=-2,
            )
            return self.out_proj((attn * v).sum(dim=-2))
        raise ValueError(
            f"SingleQueryAttentionPool query must be 2D or 3D, got {query.ndim}."
        )


class GroupedMoETransformerPolicy(nn.Module):
    """Hybrid Modern Transformer decoder policy with SOTA improvements.
    Structure:
        - Layer 0: Dense MLP (ModernTransformerBlock)
        - Optional final layer: Dense MLP when dense_layer_at_last=True
        - Intermediate layers: MoE MLP (GroupedMoEBlock)
    Features:
        - RealRoPE.
        - RMSNorm: Root Mean Square Normalization.
        - GQA: Grouped Query Attention (configurable n_kv_heads).
        - QK-Norm: RMSNorm on Queries and Keys.
        - Gated Attention: Qwen-style element-wise sigmoid gating.
        - SwiGLU MLP: DeepseekV3MLP for feed-forward.
        - Flash Attention: via F.scaled_dot_product_attention.
        - Gradient Checkpointing: optional for memory efficiency.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_config_dict: dict,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.module_config_dict = module_config_dict

        self.num_fine_experts = module_config_dict["num_fine_experts"]
        self.num_shared_experts = module_config_dict["num_shared_experts"]
        self.top_k = module_config_dict["top_k"]
        self.use_dynamic_bias = module_config_dict.get(
            "use_dynamic_bias", False
        )
        self.bias_update_rate = module_config_dict.get(
            "bias_update_rate", 0.001
        )
        self.routing_score_fn = str(
            module_config_dict.get("routing_score_fn", "softmax")
        ).lower()
        self.freeze_router = bool(
            module_config_dict.get("freeze_router", False)
        )
        self.routing_scale = float(
            module_config_dict.get("routing_scale", 1.0)
        )
        self.expert_bias_clip = float(
            module_config_dict.get("expert_bias_clip", 0.0)
        )
        load_balance_cfg = module_config_dict.get("moe_load_balance", {})
        self.moe_load_balance_enabled = bool(
            load_balance_cfg.get("enabled", False)
        )
        self.routed_expert_usage_ema_decay = float(
            module_config_dict.get("routed_expert_usage_ema_decay", 0.99)
        )
        self.routed_expert_usage_ema_dead_threshold = float(
            module_config_dict.get(
                "routed_expert_usage_ema_dead_threshold", 1.0e-6
            )
        )
        inactive_margin_cfg = module_config_dict.get(
            "inactive_expert_margin_to_topk", {}
        )
        selected_margin_cfg = module_config_dict.get(
            "selected_expert_margin_to_unselected", {}
        )
        self.inactive_expert_margin_to_topk_enabled = bool(
            inactive_margin_cfg.get("enabled", False)
        )
        self.inactive_expert_margin_to_topk_ratio_floor = float(
            inactive_margin_cfg.get("ratio_floor", 0.0)
        )
        self.selected_expert_margin_to_unselected_enabled = bool(
            selected_margin_cfg.get("enabled", False)
        )
        self.selected_expert_margin_to_unselected_target = float(
            selected_margin_cfg.get("target", 0.0)
        )
        if self.routing_score_fn not in ("softmax", "sigmoid"):
            raise ValueError(
                f"routing_score_fn must be one of {{'softmax','sigmoid'}}, got {self.routing_score_fn}"
            )
        if self.routing_scale <= 0.0:
            raise ValueError(
                f"routing_scale must be > 0, got {self.routing_scale}"
            )
        if self.expert_bias_clip < 0.0:
            raise ValueError(
                f"expert_bias_clip must be >= 0, got {self.expert_bias_clip}"
            )
        if not (0.0 <= self.routed_expert_usage_ema_decay < 1.0):
            raise ValueError(
                "routed_expert_usage_ema_decay must be in [0, 1), got "
                f"{self.routed_expert_usage_ema_decay}"
            )
        if self.routed_expert_usage_ema_dead_threshold < 0.0:
            raise ValueError(
                "routed_expert_usage_ema_dead_threshold must be >= 0, got "
                f"{self.routed_expert_usage_ema_dead_threshold}"
            )
        if self.selected_expert_margin_to_unselected_target < 0.0:
            raise ValueError(
                "selected_expert_margin_to_unselected.target must be >= 0, "
                f"got {self.selected_expert_margin_to_unselected_target}"
            )
        if not (0.0 <= self.inactive_expert_margin_to_topk_ratio_floor <= 1.0):
            raise ValueError(
                "inactive_expert_margin_to_topk.ratio_floor must be in "
                f"[0, 1], got {self.inactive_expert_margin_to_topk_ratio_floor}"
            )

        _ov = module_config_dict.get("input_dim_override", None)
        self.obs_input_dim = (
            int(_ov) if isinstance(_ov, (int, float)) else None
        )

        self.obs_embed_mlp_hidden = int(
            module_config_dict.get("obs_embed_mlp_hidden", 1024)
        )

        self.d_model = int(module_config_dict.get("d_model", 256))
        self.n_layers = int(module_config_dict.get("n_layers", 4))
        self.dense_layer_at_first = bool(
            module_config_dict.get("dense_layer_at_first", True)
        )
        self.dense_layer_at_last = bool(
            module_config_dict.get("dense_layer_at_last", False)
        )
        self.n_heads = int(module_config_dict.get("n_heads", 4))
        self.n_kv_heads = int(
            module_config_dict.get("n_kv_heads", self.n_heads // 2)
        )
        self.ff_mult = float(module_config_dict.get("ff_mult", 4))
        self.ff_mult_dense = int(
            module_config_dict.get("ff_mult_dense", self.ff_mult * 3)
        )
        self.attn_dropout = float(module_config_dict.get("attn_dropout", 0.0))
        self.mlp_dropout = float(module_config_dict.get("mlp_dropout", 0.0))
        self.max_ctx_len = int(module_config_dict.get("max_ctx_len", 64))
        self.use_qk_norm = module_config_dict.get("use_qk_norm", True)
        self.use_gated_attn = module_config_dict.get("use_gated_attn", True)
        self.gated_attn_type = module_config_dict.get(
            "gated_attn_type", "headwise"
        )
        self.use_checkpointing = module_config_dict.get(
            "use_checkpointing", False
        )
        self.use_future_cross_attn = bool(
            module_config_dict.get("use_future_cross_attn", False)
        )
        self.state_obs_dim = int(
            module_config_dict.get(
                "state_obs_dim", self.obs_input_dim or self.input_dim
            )
        )
        self.future_seq_len = int(module_config_dict.get("future_seq_len", 0))
        self.future_token_dim = int(
            module_config_dict.get("future_token_dim", 0)
        )

        self.head_dim = self.d_model // self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, got head_dim={self.head_dim}"
            )

        # RoPE configuration (used in both sequence and KV-cached single-step inference)
        self.rope_theta = float(module_config_dict.get("rope_theta", 10000.0))
        self.rope_max_seq_len = int(
            module_config_dict.get("rope_max_seq_len", 8192)
        )
        if self.rope_max_seq_len <= 0:
            raise ValueError(
                f"rope_max_seq_len must be positive, got {self.rope_max_seq_len}"
            )
        self.inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                / self.head_dim
            )
        )  # [head_dim//2]
        self.register_buffer("_rope_inv_freq", self.inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=self.rope_max_seq_len)

        obs_in = self.obs_input_dim or self.input_dim
        if self.use_future_cross_attn:
            if self.future_seq_len <= 0:
                raise ValueError(
                    "future_seq_len must be positive when use_future_cross_attn=True"
                )
            if self.future_token_dim <= 0:
                raise ValueError(
                    "future_token_dim must be positive when use_future_cross_attn=True"
                )
            self.state_obs_embed = nn.Sequential(
                nn.Linear(self.state_obs_dim, self.obs_embed_mlp_hidden),
                nn.SiLU(),
                nn.Linear(self.obs_embed_mlp_hidden, self.d_model),
            )
            # Keep a single state embedding module so DDP doesn't see unused
            # parameters from an extra unused `obs_embed` in conditional mode.
            self.obs_embed = self.state_obs_embed
            self.future_obs_embed = nn.Sequential(
                nn.Linear(self.future_token_dim, self.obs_embed_mlp_hidden),
                nn.SiLU(),
                nn.Linear(self.obs_embed_mlp_hidden, self.d_model),
            )
            self.future_pos_embed = nn.Embedding(
                self.future_seq_len, self.d_model
            )
        else:
            self.obs_embed = nn.Sequential(
                nn.Linear(obs_in, self.obs_embed_mlp_hidden),
                nn.SiLU(),
                nn.Linear(self.obs_embed_mlp_hidden, self.d_model),
            )
            self.state_obs_embed = None
            self.future_obs_embed = None
            self.future_pos_embed = None
        # Stack of TransformerBlocks: the first/last layer can be configured
        # as dense, otherwise the layer uses the standard sparse MoE block.
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            use_dense_layer = (self.dense_layer_at_first and i == 0) or (
                self.dense_layer_at_last and i == self.n_layers - 1
            )
            if use_dense_layer:
                layer = ModernTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    ff_mult=self.ff_mult_dense,
                    use_qk_norm=self.use_qk_norm,
                    use_gated_attn=self.use_gated_attn,
                    gated_attn_type=self.gated_attn_type,
                    attn_dropout=self.attn_dropout,
                    mlp_dropout=self.mlp_dropout,
                    use_cross_attn=self.use_future_cross_attn,
                )
            else:
                layer = GroupedMoEBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    ff_mult=self.ff_mult,
                    use_qk_norm=self.use_qk_norm,
                    use_gated_attn=self.use_gated_attn,
                    gated_attn_type=self.gated_attn_type,
                    attn_dropout=self.attn_dropout,
                    mlp_dropout=self.mlp_dropout,
                    num_fine_experts=self.num_fine_experts,
                    num_shared_experts=self.num_shared_experts,
                    top_k=self.top_k,
                    use_dynamic_bias=self.use_dynamic_bias,
                    bias_update_rate=self.bias_update_rate,
                    routing_score_fn=self.routing_score_fn,
                    freeze_router=self.freeze_router,
                    routing_scale=self.routing_scale,
                    expert_bias_clip=self.expert_bias_clip,
                    moe_load_balance_enabled=self.moe_load_balance_enabled,
                    routed_expert_usage_ema_decay=(
                        self.routed_expert_usage_ema_decay
                    ),
                    routed_expert_usage_ema_dead_threshold=(
                        self.routed_expert_usage_ema_dead_threshold
                    ),
                    inactive_expert_margin_to_topk_enabled=(
                        self.inactive_expert_margin_to_topk_enabled
                    ),
                    inactive_expert_margin_to_topk_ratio_floor=(
                        self.inactive_expert_margin_to_topk_ratio_floor
                    ),
                    selected_expert_margin_to_unselected_enabled=(
                        self.selected_expert_margin_to_unselected_enabled
                    ),
                    selected_expert_margin_to_unselected_target=(
                        self.selected_expert_margin_to_unselected_target
                    ),
                    use_cross_attn=self.use_future_cross_attn,
                )
            self.layers.append(layer)
        self._last_moe_layer_idx = None
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, GroupedMoEBlock):
                self._last_moe_layer_idx = layer_idx

        self.norm_f = RMSNorm(self.d_model)
        self.action_mu_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.output_dim),
        )
        aux_cfg = module_config_dict.get("aux_state_pred", {})
        self.aux_state_pred_enabled = bool(aux_cfg.get("enabled", False))
        self.aux_contact_dim = int(
            len(aux_cfg.get("keybody_contact_names", []))
        )
        self.aux_keybody_pos_dim = int(
            len(aux_cfg.get("keybody_rel_pos_names", []))
        )
        if self.aux_state_pred_enabled:
            self.aux_vel_head = nn.Linear(self.d_model, 6)
            self.aux_height_head = nn.Linear(self.d_model, 2)
            self.aux_contact_head = (
                nn.Linear(self.d_model, self.aux_contact_dim)
                if self.aux_contact_dim > 0
                else None
            )
            self.aux_ref_keybody_pos_head = (
                nn.Linear(self.d_model, self.aux_keybody_pos_dim * 3)
                if self.aux_keybody_pos_dim > 0
                else None
            )
            self.aux_robot_keybody_pos_head = (
                nn.Linear(self.d_model, self.aux_keybody_pos_dim * 3)
                if self.aux_keybody_pos_dim > 0
                else None
            )
        else:
            self.aux_vel_head = None
            self.aux_height_head = None
            self.aux_contact_head = None
            self.aux_ref_keybody_pos_head = None
            self.aux_robot_keybody_pos_head = None

        # True per-layer KV cache for single-step inference.
        # K/V shapes: [B, n_layers, max_ctx_len, n_kv_heads, head_dim]
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None
        # Cache state per environment
        self._kv_cache_len: torch.Tensor | None = None  # [B]
        self._kv_cache_write_idx: torch.Tensor | None = None  # [B]
        self._kv_cache_abs_pos: torch.Tensor | None = None  # [B]
        self._prev_last_moe_router_p: torch.Tensor | None = None
        self._prev_last_moe_router_valid: torch.Tensor | None = None
        self._last_moe_router_js_sum: torch.Tensor | None = None
        self._last_moe_router_js_count: torch.Tensor | None = None
        self._last_moe_router_top1_switch_sum: torch.Tensor | None = None
        self._last_moe_router_top1_switch_count: torch.Tensor | None = None
        aux_cmd_cfg = module_config_dict.get("aux_router_command_recon", {})
        self.aux_router_command_recon_enabled = bool(
            aux_cmd_cfg.get("enabled", False)
        )
        self.aux_router_command_recon_output_dim = int(
            aux_cmd_cfg.get("output_dim", 0)
        )
        self.aux_router_command_recon_hidden_dim = int(
            aux_cmd_cfg.get("hidden_dim", self.d_model)
        )
        self._num_moe_layers = sum(
            1 for layer in self.layers if isinstance(layer, GroupedMoEBlock)
        )
        if self.aux_router_command_recon_enabled:
            if self._num_moe_layers <= 0:
                raise ValueError(
                    "aux_router_command_recon requires at least one GroupedMoEBlock."
                )
            if self.aux_router_command_recon_output_dim <= 0:
                raise ValueError(
                    "aux_router_command_recon.output_dim must be positive when enabled."
                )
            router_feature_dim = self._num_moe_layers * self.num_fine_experts
            self.aux_router_command_recon_head = nn.Sequential(
                nn.Linear(
                    router_feature_dim,
                    self.aux_router_command_recon_hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.aux_router_command_recon_hidden_dim,
                    self.aux_router_command_recon_output_dim,
                ),
            )
        else:
            self.aux_router_command_recon_head = None
        aux_router_future_cfg = module_config_dict.get(
            "aux_router_future_recon", {}
        )
        self.aux_router_future_recon_enabled = bool(
            aux_router_future_cfg.get("enabled", False)
        )
        self.aux_router_future_recon_output_dim = int(
            aux_router_future_cfg.get("output_dim", 0)
        )
        self.aux_router_future_recon_hidden_dim = int(
            aux_router_future_cfg.get("hidden_dim", self.d_model)
        )
        aux_router_future_norm_cfg = aux_router_future_cfg.get(
            "target_norm", {}
        )
        self.aux_router_future_recon_norm_eps = float(
            aux_router_future_norm_cfg.get("epsilon", 1.0e-2)
        )
        self.aux_router_future_recon_norm_update_method = str(
            aux_router_future_norm_cfg.get("update_method", "cumulative")
        ).lower()
        aux_router_future_norm_ema = aux_router_future_norm_cfg.get(
            "ema_momentum", None
        )
        self.aux_router_future_recon_norm_ema_momentum = (
            float(aux_router_future_norm_ema)
            if aux_router_future_norm_ema is not None
            else None
        )
        if self.aux_router_future_recon_enabled:
            if self.aux_router_future_recon_output_dim <= 0:
                raise ValueError(
                    "aux_router_future_recon.output_dim must be positive when enabled."
                )
            self.aux_router_future_recon_head = nn.Sequential(
                nn.Linear(
                    self.d_model,
                    self.aux_router_future_recon_hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.aux_router_future_recon_hidden_dim,
                    self.aux_router_future_recon_output_dim,
                ),
            )
            self.aux_router_future_recon_normalizer = EmpiricalNormalization(
                shape=self.aux_router_future_recon_output_dim,
                eps=self.aux_router_future_recon_norm_eps,
                update_method=self.aux_router_future_recon_norm_update_method,
                ema_momentum=self.aux_router_future_recon_norm_ema_momentum,
            )
        else:
            self.aux_router_future_recon_head = None
            self.aux_router_future_recon_normalizer = None
        self._apply_base_freeze_router_state()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.use_future_cross_attn:
            # In conditional mode, `obs_embed` is tied to `state_obs_embed`.
            # Older checkpoints may contain separate weights for both; ensure we
            # always load the trained state embedding weights.
            obs_prefix = prefix + "obs_embed."
            state_prefix = prefix + "state_obs_embed."
            for suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
                s_key = state_prefix + suffix
                o_key = obs_prefix + suffix
                if s_key in state_dict:
                    state_dict[o_key] = state_dict[s_key]

        legacy_aux_prefix = prefix + "aux_command_recon_head."
        current_aux_prefix = prefix + "aux_router_command_recon_head."
        legacy_aux_keys = [
            key
            for key in list(state_dict.keys())
            if key.startswith(legacy_aux_prefix)
        ]
        if legacy_aux_keys:
            if self.aux_router_command_recon_head is not None:
                for legacy_key in legacy_aux_keys:
                    suffix = legacy_key.removeprefix(legacy_aux_prefix)
                    current_key = current_aux_prefix + suffix
                    state_dict.setdefault(current_key, state_dict[legacy_key])
            for legacy_key in legacy_aux_keys:
                state_dict.pop(legacy_key, None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._apply_freeze_router_state()

    def _router_no_grad_context(self):
        if self.freeze_router:
            return torch.no_grad()
        return nullcontext()

    def _apply_base_freeze_router_state(self) -> None:
        for layer in self.layers:
            if isinstance(layer, GroupedMoEBlock):
                layer._apply_freeze_router_state()

    def _apply_freeze_router_state(self) -> None:
        self._apply_base_freeze_router_state()
        if self.aux_router_future_recon_head is not None:
            self.aux_router_future_recon_head.requires_grad_(
                not self.freeze_router
            )

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )

        # outer product: [seq_len, head_dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # Concatenate to match rotate_half: [seq_len, head_dim]
        # Different from complex, here we just concat freqs to match the real-valued rotation logic
        emb = torch.cat((freqs, freqs), dim=-1)

        # [seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_cos_sin(self, x, position_ids):
        """根据 position_ids 获取 cos/sin
        x: [B, T, D]
        position_ids: [B, T]
        Returns: cos, sin -> [B, T, D] (broadcastable)
        """
        # cos_cached: [MaxLen, D]
        # F.embedding(pos, cache) -> [B, T, D]
        cos = F.embedding(position_ids, self.cos_cached)
        sin = F.embedding(position_ids, self.sin_cached)
        return cos.to(x.dtype), sin.to(x.dtype)

    def _init_last_moe_router_shift_state(self, num_envs: int, device) -> None:
        if self._last_moe_layer_idx is None:
            self._prev_last_moe_router_p = None
            self._prev_last_moe_router_valid = None
            self._last_moe_router_js_sum = None
            self._last_moe_router_js_count = None
            self._last_moe_router_top1_switch_sum = None
            self._last_moe_router_top1_switch_count = None
            return
        self._prev_last_moe_router_p = torch.zeros(
            num_envs,
            self.num_fine_experts,
            device=device,
            dtype=torch.float32,
        )
        self._prev_last_moe_router_valid = torch.zeros(
            num_envs, device=device, dtype=torch.bool
        )
        self._last_moe_router_js_sum = torch.zeros(
            (), device=device, dtype=torch.float32
        )
        self._last_moe_router_js_count = torch.zeros(
            (), device=device, dtype=torch.float32
        )
        self._last_moe_router_top1_switch_sum = torch.zeros(
            (), device=device, dtype=torch.float32
        )
        self._last_moe_router_top1_switch_count = torch.zeros(
            (), device=device, dtype=torch.float32
        )

    def _accumulate_last_moe_router_shift(
        self, router_distribution: torch.Tensor
    ) -> None:
        if (
            self._prev_last_moe_router_p is None
            or self._prev_last_moe_router_valid is None
            or self._last_moe_router_js_sum is None
            or self._last_moe_router_js_count is None
            or self._last_moe_router_top1_switch_sum is None
            or self._last_moe_router_top1_switch_count is None
        ):
            return
        if (
            router_distribution.ndim != 3
            or int(router_distribution.shape[1]) != 1
        ):
            return
        curr_p = router_distribution[:, 0, :].to(torch.float32)
        if int(curr_p.shape[0]) != int(self._prev_last_moe_router_p.shape[0]):
            return
        prev_valid = self._prev_last_moe_router_valid
        if torch.any(prev_valid):
            prev_p = self._prev_last_moe_router_p[prev_valid]
            curr_p_valid = curr_p[prev_valid]
            mix_p = 0.5 * (curr_p_valid + prev_p)
            eps = 1.0e-20
            curr_safe = curr_p_valid.clamp_min(eps)
            prev_safe = prev_p.clamp_min(eps)
            mix_safe = mix_p.clamp_min(eps)
            kl_curr = (
                curr_p_valid * (torch.log(curr_safe) - torch.log(mix_safe))
            ).sum(dim=-1)
            kl_prev = (
                prev_p * (torch.log(prev_safe) - torch.log(mix_safe))
            ).sum(dim=-1)
            js = 0.5 * (kl_curr + kl_prev)
            self._last_moe_router_js_sum.add_(js.sum())
            self._last_moe_router_js_count.add_(float(js.numel()))
            curr_top1 = curr_p_valid.argmax(dim=-1)
            prev_top1 = prev_p.argmax(dim=-1)
            switch = (curr_top1 != prev_top1).to(torch.float32)
            self._last_moe_router_top1_switch_sum.add_(switch.sum())
            self._last_moe_router_top1_switch_count.add_(float(switch.numel()))
        self._prev_last_moe_router_p.copy_(curr_p)
        self._prev_last_moe_router_valid.fill_(True)

    def get_last_moe_router_shift_stats(
        self,
    ) -> dict[str, torch.Tensor | None]:
        return {
            "js_sum": self._last_moe_router_js_sum,
            "js_count": self._last_moe_router_js_count,
            "top1_switch_sum": self._last_moe_router_top1_switch_sum,
            "top1_switch_count": self._last_moe_router_top1_switch_count,
        }

    def reset_kv_cache(self, num_envs: int, device):
        """Initialize per-environment KV cache for single-step inference."""
        cache_dtype = (
            torch.float16
            if torch.device(device).type == "cuda"
            else torch.float32
        )
        self._k_cache = torch.zeros(
            num_envs,
            self.n_layers,
            self.max_ctx_len,
            self.n_kv_heads,
            self.head_dim,
            device=device,
            dtype=cache_dtype,
        )
        self._v_cache = torch.zeros_like(self._k_cache)
        self._kv_cache_len = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._kv_cache_write_idx = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._kv_cache_abs_pos = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._init_last_moe_router_shift_state(num_envs, device)

    def clear_env_cache(self, env_ids: torch.Tensor | None):
        """Reset KV cache state for specific environments."""
        if self._k_cache is None:
            return
        if env_ids is None:
            self._k_cache.zero_()
            self._v_cache.zero_()
            self._kv_cache_len.zero_()
            self._kv_cache_write_idx.zero_()
            self._kv_cache_abs_pos.zero_()
            if self._prev_last_moe_router_p is not None:
                self._prev_last_moe_router_p.zero_()
            if self._prev_last_moe_router_valid is not None:
                self._prev_last_moe_router_valid.zero_()
            if self._last_moe_router_js_sum is not None:
                self._last_moe_router_js_sum.zero_()
            if self._last_moe_router_js_count is not None:
                self._last_moe_router_js_count.zero_()
            if self._last_moe_router_top1_switch_sum is not None:
                self._last_moe_router_top1_switch_sum.zero_()
            if self._last_moe_router_top1_switch_count is not None:
                self._last_moe_router_top1_switch_count.zero_()
        else:
            self._k_cache[env_ids] = 0.0
            self._v_cache[env_ids] = 0.0
            self._kv_cache_len[env_ids] = 0
            self._kv_cache_write_idx[env_ids] = 0
            self._kv_cache_abs_pos[env_ids] = 0
            if self._prev_last_moe_router_valid is not None:
                self._prev_last_moe_router_valid[env_ids] = False
            if self._prev_last_moe_router_p is not None:
                self._prev_last_moe_router_p[env_ids] = 0.0

    def set_collect_routing_stats(self, collect: bool) -> None:
        collect_flag = bool(collect)
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, GroupedMoEBlock):
                layer.collect_routing_stats = collect_flag
                layer.collect_router_distribution = (
                    collect_flag and layer_idx == self._last_moe_layer_idx
                )

    def reset_routing_stats(self) -> None:
        for layer in self.layers:
            if isinstance(layer, GroupedMoEBlock):
                layer.reset_routing_stats()

    def clear_router_distribution_cache(self) -> None:
        for layer in self.layers:
            if isinstance(layer, GroupedMoEBlock):
                layer.last_router_distribution = None
                layer.last_router_logits = None
                layer.capture_router_distribution = False
                layer.capture_router_logits = False

    def _set_capture_router_distributions(self, capture: bool) -> None:
        self._set_capture_router_features(
            capture_distributions=capture,
            capture_logits=False,
        )

    def _set_capture_router_features(
        self,
        *,
        capture_distributions: bool,
        capture_logits: bool,
    ) -> None:
        capture_distribution_flag = bool(capture_distributions)
        capture_logits_flag = bool(capture_logits)
        for layer in self.layers:
            if isinstance(layer, GroupedMoEBlock):
                layer.capture_router_distribution = capture_distribution_flag
                layer.capture_router_logits = capture_logits_flag

    def apply_dynamic_bias_update_from_stats(self) -> None:
        for layer in self.layers:
            if isinstance(layer, GroupedMoEBlock):
                layer.apply_bias_update_from_counts()

    def _make_causal_mask(self, T: int, device) -> torch.Tensor:
        """Generate causal attention mask: shape [T, T], True where attend allowed."""
        return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))

    def _forward_layers_range(
        self,
        h: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mask: torch.Tensor | None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        router_h: torch.Tensor | None = None,
        router_h_per_layer: list[torch.Tensor | None] | None = None,
        *,
        start_layer: int,
        end_layer: int,
        return_pre_moe_hidden: bool = False,
        return_router_features: bool = False,
        return_router_temporal_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward through a contiguous layer range with optional checkpointing."""
        if (
            start_layer < 0
            or end_layer < start_layer
            or end_layer > len(self.layers)
        ):
            raise ValueError(
                "Invalid layer range for _forward_layers_range: "
                f"start_layer={start_layer}, end_layer={end_layer}, "
                f"num_layers={len(self.layers)}."
            )
        pre_moe_hidden = None
        router_features = []
        router_temporal_features = []
        self._set_capture_router_features(
            capture_distributions=return_router_features,
            capture_logits=return_router_temporal_features,
        )
        try:
            for layer_idx in range(start_layer, end_layer):
                layer = self.layers[layer_idx]
                layer_router_h = router_h
                if router_h_per_layer is not None:
                    layer_router_h = router_h_per_layer[layer_idx]
                if self.use_checkpointing and self.training:
                    if isinstance(layer, GroupedMoEBlock):
                        h = checkpoint.checkpoint(
                            layer,
                            h,
                            cos,
                            sin,
                            mask,
                            memory,
                            memory_mask,
                            layer_router_h,
                            use_reentrant=False,
                        )
                    else:
                        h = checkpoint.checkpoint(
                            layer,
                            h,
                            cos,
                            sin,
                            mask,
                            memory,
                            memory_mask,
                            use_reentrant=False,
                        )
                else:
                    if isinstance(layer, GroupedMoEBlock):
                        h = layer(
                            h,
                            cos,
                            sin,
                            mask,
                            memory,
                            memory_mask,
                            router_x=layer_router_h,
                        )
                    else:
                        h = layer(h, cos, sin, mask, memory, memory_mask)
                if return_pre_moe_hidden and layer_idx == 0:
                    pre_moe_hidden = h
                if return_router_features and isinstance(
                    layer, GroupedMoEBlock
                ):
                    if layer.last_router_distribution is None:
                        raise ValueError(
                            f"Missing router distribution for MoE layer {layer_idx}."
                        )
                    router_features.append(layer.last_router_distribution)
                if return_router_temporal_features and isinstance(
                    layer, GroupedMoEBlock
                ):
                    if layer.last_router_logits is None:
                        raise ValueError(
                            f"Missing router logits for MoE layer {layer_idx}."
                        )
                    router_temporal_features.append(layer.last_router_logits)
        finally:
            self._set_capture_router_features(
                capture_distributions=False,
                capture_logits=False,
            )

        outputs: list[torch.Tensor] = [h]
        if return_pre_moe_hidden:
            if pre_moe_hidden is None:
                raise ValueError(
                    "Missing pre-MoE hidden state from the leading dense layer."
                )
            outputs.append(pre_moe_hidden)
        if return_router_features:
            if len(router_features) == 0:
                raise ValueError(
                    "Missing router features while return_router_features=True."
                )
            outputs.append(torch.cat(router_features, dim=-1))
        if return_router_temporal_features:
            if len(router_temporal_features) == 0:
                raise ValueError(
                    "Missing router temporal features while "
                    "return_router_temporal_features=True."
                )
            outputs.append(torch.cat(router_temporal_features, dim=-1))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _forward_layers(
        self,
        h: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mask: torch.Tensor | None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        router_h: torch.Tensor | None = None,
        router_h_per_layer: list[torch.Tensor | None] | None = None,
        return_pre_moe_hidden: bool = False,
        return_router_features: bool = False,
        return_router_temporal_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self._forward_layers_range(
            h,
            cos,
            sin,
            mask,
            memory,
            memory_mask,
            router_h,
            router_h_per_layer,
            start_layer=0,
            end_layer=len(self.layers),
            return_pre_moe_hidden=return_pre_moe_hidden,
            return_router_features=return_router_features,
            return_router_temporal_features=return_router_temporal_features,
        )

    def _compute_router_hidden(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def sequence_mu(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_pre_moe_hidden: bool = False,
        return_router_features: bool = False,
        return_router_temporal_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Compute per-token action mean for sequences.

        Args:
            x: [B, T, D] flat obs per token.
            attn_mask: [B, T, T] boolean mask (True if attend allowed), or None for causal.
            return_hidden: If True, also return the hidden states.

        Returns:
            mu: [B, T, A]
            h: [B, T, d_model] (only if return_hidden=True)
        """
        B, T, _ = x.shape
        h = self.obs_embed(x)  # [B, T, d_model]
        router_h = self._compute_router_hidden(x)

        # SDPA bool attention mask uses True = allowed (can attend).
        if attn_mask is not None:
            tgt_mask = attn_mask.unsqueeze(1)  # [B, 1, T, T]
            # Episode-aware positions: first attendable token is episode start.
            start_idx = attn_mask.to(torch.int64).argmax(dim=-1)  # [B, T]
            t_idx = torch.arange(T, device=x.device, dtype=torch.long)[
                None, :
            ].expand(B, T)
            pos = t_idx - start_idx  # [B, T]
        else:
            tgt_mask = None
            pos = torch.arange(T, device=x.device, dtype=torch.long)[
                None, :
            ].expand(B, T)

        cos, sin = self.get_cos_sin(h, pos)  # [B, T, head_dim//2]
        if return_hidden and return_pre_moe_hidden:
            raise ValueError(
                "return_hidden and return_pre_moe_hidden cannot both be True."
            )
        forward_out = self._forward_layers(
            h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h=router_h,
            return_pre_moe_hidden=return_pre_moe_hidden,
            return_router_features=return_router_features,
            return_router_temporal_features=return_router_temporal_features,
        )
        extras: list[torch.Tensor] = []
        if isinstance(forward_out, tuple):
            h = forward_out[0]
            extras = list(forward_out[1:])
        else:
            h = forward_out
        h = self.norm_f(h)
        mu = self.action_mu_head(h)
        outputs: list[torch.Tensor] = [mu]
        if return_pre_moe_hidden:
            outputs.append(extras.pop(0))
        if return_router_features:
            outputs.append(extras.pop(0))
        if return_router_temporal_features:
            outputs.append(extras.pop(0))
        if len(outputs) > 1:
            return tuple(outputs)
        if return_hidden:
            return mu, h
        return mu

    def sequence_hidden(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-token latent features for sequences.

        Args:
            x: [B, T, D] flat obs per token.
            attn_mask: [B, T, T] boolean mask (True if attend allowed).

        Returns:
            h_f: [B, T, d_model]
        """
        B, T, _ = x.shape
        h = self.obs_embed(x)  # [B, T, d_model]
        router_h = self._compute_router_hidden(x)

        if attn_mask is not None:
            tgt_mask = attn_mask.unsqueeze(1)  # [B, 1, T, T]
            start_idx = attn_mask.to(torch.int64).argmax(dim=-1)  # [B, T]
            t_idx = torch.arange(T, device=x.device, dtype=torch.long)[
                None, :
            ].expand(B, T)
            pos = t_idx - start_idx  # [B, T]
        else:
            tgt_mask = None
            pos = torch.arange(T, device=x.device, dtype=torch.long)[
                None, :
            ].expand(B, T)

        cos, sin = self.get_cos_sin(h, pos)
        h = self._forward_layers(
            h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h=router_h,
        )
        h = self.norm_f(h)
        return h

    def _embed_future_tokens(
        self, future_tokens: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_future_cross_attn:
            raise ValueError(
                "_embed_future_tokens requires use_future_cross_attn=True"
            )
        if future_tokens.ndim == 3:
            b, n, d = future_tokens.shape
            if n != self.future_seq_len:
                raise ValueError(
                    f"future token length mismatch: expected {self.future_seq_len}, got {n}"
                )
            if d != self.future_token_dim:
                raise ValueError(
                    f"future token dim mismatch: expected {self.future_token_dim}, got {d}"
                )
            pos = torch.arange(
                n, device=future_tokens.device, dtype=torch.long
            )
            pos_emb = self.future_pos_embed(pos)[None, :, :]
            return self.future_obs_embed(future_tokens) + pos_emb
        if future_tokens.ndim == 4:
            b, t, n, d = future_tokens.shape
            if n != self.future_seq_len:
                raise ValueError(
                    f"future token length mismatch: expected {self.future_seq_len}, got {n}"
                )
            if d != self.future_token_dim:
                raise ValueError(
                    f"future token dim mismatch: expected {self.future_token_dim}, got {d}"
                )
            pos = torch.arange(
                n, device=future_tokens.device, dtype=torch.long
            )
            pos_emb = self.future_pos_embed(pos)[None, None, :, :]
            return self.future_obs_embed(future_tokens) + pos_emb
        raise ValueError(
            f"future_tokens must be 3D or 4D, got shape {tuple(future_tokens.shape)}"
        )

    def sequence_mu_cond(
        self,
        state_seq: torch.Tensor,
        future_seq: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        future_mask: torch.Tensor | None = None,
        return_pre_moe_hidden: bool = False,
        return_router_features: bool = False,
        return_router_temporal_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not self.use_future_cross_attn:
            raise ValueError(
                "sequence_mu_cond requires use_future_cross_attn=True"
            )
        if state_seq.ndim != 3:
            raise ValueError(
                f"state_seq must have shape [B, T, D], got {tuple(state_seq.shape)}"
            )
        if future_seq.ndim != 4:
            raise ValueError(
                "future_seq must have shape [B, T, N_fut, D_fut], "
                f"got {tuple(future_seq.shape)}"
            )
        b, t, d_state = state_seq.shape
        bf, tf, n_fut, d_fut = future_seq.shape
        if bf != b or tf != t:
            raise ValueError(
                "state_seq and future_seq batch/time mismatch: "
                f"state={tuple(state_seq.shape)}, future={tuple(future_seq.shape)}"
            )
        if d_state != self.state_obs_dim:
            raise ValueError(
                f"state_seq dim mismatch: expected {self.state_obs_dim}, got {d_state}"
            )
        if n_fut != self.future_seq_len:
            raise ValueError(
                f"future_seq len mismatch: expected {self.future_seq_len}, got {n_fut}"
            )
        if d_fut != self.future_token_dim:
            raise ValueError(
                f"future_seq dim mismatch: expected {self.future_token_dim}, got {d_fut}"
            )

        h = self.state_obs_embed(state_seq)
        memory = self._embed_future_tokens(future_seq)
        if future_mask is None:
            future_mask = torch.ones(
                b,
                t,
                n_fut,
                dtype=torch.bool,
                device=state_seq.device,
            )
        if future_mask.shape != (b, t, n_fut):
            raise ValueError(
                "future_mask shape mismatch: expected "
                f"{(b, t, n_fut)}, got {tuple(future_mask.shape)}"
            )

        if attn_mask is not None:
            tgt_mask = attn_mask.unsqueeze(1)
            start_idx = attn_mask.to(torch.int64).argmax(dim=-1)
            t_idx = torch.arange(t, device=state_seq.device, dtype=torch.long)[
                None, :
            ].expand(b, t)
            pos = t_idx - start_idx
        else:
            tgt_mask = None
            pos = torch.arange(t, device=state_seq.device, dtype=torch.long)[
                None, :
            ].expand(b, t)

        cos, sin = self.get_cos_sin(h, pos)
        forward_out = self._forward_layers(
            h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            memory=memory,
            memory_mask=future_mask,
            return_pre_moe_hidden=return_pre_moe_hidden,
            return_router_features=return_router_features,
            return_router_temporal_features=return_router_temporal_features,
        )
        extras: list[torch.Tensor] = []
        if isinstance(forward_out, tuple):
            h = forward_out[0]
            extras = list(forward_out[1:])
        else:
            h = forward_out
        h = self.norm_f(h)
        mu = self.action_mu_head(h)
        outputs: list[torch.Tensor] = [mu]
        if return_pre_moe_hidden:
            outputs.append(extras.pop(0))
        if return_router_features:
            outputs.append(extras.pop(0))
        if return_router_temporal_features:
            outputs.append(extras.pop(0))
        if len(outputs) > 1:
            return tuple(outputs)
        return mu

    def predict_aux_from_pre_moe(
        self,
        pre_moe_hidden: torch.Tensor,
        *,
        ref_aux_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.aux_state_pred_enabled:
            raise ValueError(
                "predict_aux_from_pre_moe requires aux_state_pred.enabled=True."
            )
        if pre_moe_hidden.ndim != 3:
            raise ValueError(
                f"Expected pre_moe_hidden with shape [B, T, D], got {tuple(pre_moe_hidden.shape)}"
            )
        vel_params = self.aux_vel_head(pre_moe_hidden)
        height_params = self.aux_height_head(pre_moe_hidden)
        vel_loc, vel_log_std = vel_params.chunk(2, dim=-1)
        height_loc, height_log_std = height_params.chunk(2, dim=-1)
        aux_outputs = {
            "base_lin_vel_loc": vel_loc,
            "base_lin_vel_log_std": vel_log_std,
            "root_height_loc": height_loc,
            "root_height_log_std": height_log_std,
        }
        if self.aux_contact_head is not None:
            aux_outputs["keybody_contact_logits"] = self.aux_contact_head(
                pre_moe_hidden
            )
        else:
            aux_outputs["keybody_contact_logits"] = pre_moe_hidden.new_zeros(
                pre_moe_hidden.shape[0],
                pre_moe_hidden.shape[1],
                0,
            )
        if self.aux_ref_keybody_pos_head is not None:
            aux_outputs["ref_keybody_rel_pos"] = self.aux_ref_keybody_pos_head(
                pre_moe_hidden
            ).reshape(
                pre_moe_hidden.shape[0],
                pre_moe_hidden.shape[1],
                self.aux_keybody_pos_dim,
                3,
            )
            aux_outputs["robot_keybody_rel_pos"] = (
                self.aux_robot_keybody_pos_head(pre_moe_hidden).reshape(
                    pre_moe_hidden.shape[0],
                    pre_moe_hidden.shape[1],
                    self.aux_keybody_pos_dim,
                    3,
                )
            )
        else:
            aux_outputs["ref_keybody_rel_pos"] = pre_moe_hidden.new_zeros(
                pre_moe_hidden.shape[0],
                pre_moe_hidden.shape[1],
                0,
                3,
            )
            aux_outputs["robot_keybody_rel_pos"] = pre_moe_hidden.new_zeros(
                pre_moe_hidden.shape[0],
                pre_moe_hidden.shape[1],
                0,
                3,
            )
        return aux_outputs

    def predict_aux_router_command_from_router_features(
        self, router_features: torch.Tensor
    ) -> torch.Tensor:
        if not self.aux_router_command_recon_enabled:
            raise ValueError(
                "predict_aux_router_command_from_router_features requires "
                "aux_router_command_recon.enabled=True."
            )
        if router_features.ndim != 3:
            raise ValueError(
                "Expected router_features with shape [B, T, D], got "
                f"{tuple(router_features.shape)}."
            )
        if self.aux_router_command_recon_head is None:
            raise ValueError(
                "aux_router_command_recon_head is not initialized."
            )
        return self.aux_router_command_recon_head(router_features)

    def update_aux_router_future_recon_normalizer(
        self, future_target: torch.Tensor
    ) -> None:
        if not self.aux_router_future_recon_enabled:
            raise ValueError(
                "update_aux_router_future_recon_normalizer requires "
                "aux_router_future_recon.enabled=True."
            )
        if self.aux_router_future_recon_normalizer is None:
            raise ValueError(
                "aux_router_future_recon_normalizer is not initialized."
            )
        if future_target.ndim < 2:
            raise ValueError(
                "Expected future_target with shape [B, D] or [B, T, D], got "
                f"{tuple(future_target.shape)}."
            )
        flat_target = future_target.reshape(
            -1, future_target.shape[-1]
        ).detach()
        self.aux_router_future_recon_normalizer.update(flat_target)

    def normalize_aux_router_future_recon_target(
        self, future_target: torch.Tensor
    ) -> torch.Tensor:
        if not self.aux_router_future_recon_enabled:
            raise ValueError(
                "normalize_aux_router_future_recon_target requires "
                "aux_router_future_recon.enabled=True."
            )
        if self.aux_router_future_recon_normalizer is None:
            raise ValueError(
                "aux_router_future_recon_normalizer is not initialized."
            )
        if future_target.ndim < 2:
            raise ValueError(
                "Expected future_target with shape [B, D] or [B, T, D], got "
                f"{tuple(future_target.shape)}."
            )
        flat_target = future_target.reshape(-1, future_target.shape[-1])
        norm_target = self.aux_router_future_recon_normalizer.normalize_only(
            flat_target
        )
        return norm_target.reshape_as(future_target)

    def predict_aux_router_future_recon_from_router_hidden(
        self, router_hidden: torch.Tensor
    ) -> torch.Tensor:
        if not self.aux_router_future_recon_enabled:
            raise ValueError(
                "predict_aux_router_future_recon_from_router_hidden requires "
                "aux_router_future_recon.enabled=True."
            )
        if router_hidden.ndim != 3:
            raise ValueError(
                "Expected router_hidden with shape [B, T, D], got "
                f"{tuple(router_hidden.shape)}."
            )
        if self.aux_router_future_recon_head is None:
            raise ValueError(
                "aux_router_future_recon_head is not initialized."
            )
        return self.aux_router_future_recon_head(router_hidden)

    def single_step_mu_cond(
        self,
        state_x: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        future_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_future_cross_attn:
            raise ValueError(
                "single_step_mu_cond requires use_future_cross_attn=True"
            )
        if state_x.ndim != 2:
            raise ValueError(f"Expected state_x [B, D], got {state_x.shape}")
        if future_tokens.ndim != 3:
            raise ValueError(
                "Expected future_tokens [B, N_fut, D_fut], "
                f"got {future_tokens.shape}"
            )
        b, d_state = state_x.shape
        bf, n_fut, d_fut = future_tokens.shape
        if bf != b:
            raise ValueError(
                f"Batch mismatch between state_x and future_tokens: {b} vs {bf}"
            )
        if d_state != self.state_obs_dim:
            raise ValueError(
                f"state_x dim mismatch: expected {self.state_obs_dim}, got {d_state}"
            )
        if n_fut != self.future_seq_len:
            raise ValueError(
                f"future len mismatch: expected {self.future_seq_len}, got {n_fut}"
            )
        if d_fut != self.future_token_dim:
            raise ValueError(
                f"future dim mismatch: expected {self.future_token_dim}, got {d_fut}"
            )

        if self._k_cache is None:
            state_seq = state_x[:, None, :]
            future_seq = future_tokens[:, None, :, :]
            if future_mask is not None:
                future_mask = future_mask[:, None, :]
            mu_seq = self.sequence_mu_cond(
                state_seq,
                future_seq,
                attn_mask=None,
                future_mask=future_mask,
            )
            return mu_seq[:, 0, :]

        if self._k_cache.device != state_x.device:
            self._k_cache = self._k_cache.to(state_x.device)
            self._v_cache = self._v_cache.to(state_x.device)
            self._kv_cache_len = self._kv_cache_len.to(state_x.device)
            self._kv_cache_write_idx = self._kv_cache_write_idx.to(
                state_x.device
            )
            self._kv_cache_abs_pos = self._kv_cache_abs_pos.to(state_x.device)

        h = self.state_obs_embed(state_x)[:, None, :]
        memory = self._embed_future_tokens(future_tokens)

        if self._k_cache.dtype != h.dtype:
            self._k_cache = self._k_cache.to(h.dtype)
            self._v_cache = self._v_cache.to(h.dtype)

        cache_len = self._kv_cache_len
        insert_pos = self._kv_cache_write_idx
        max_len = int(self.max_ctx_len)
        new_len = torch.clamp(cache_len + 1, max=max_len)

        self._kv_cache_len = new_len
        self._kv_cache_write_idx = (insert_pos + 1) % max_len

        pos = self._kv_cache_abs_pos
        self._kv_cache_abs_pos = pos + 1
        pos_ids = pos.unsqueeze(1)
        cos, sin = self.get_cos_sin(h, pos_ids)

        memory_mask = None
        if future_mask is not None:
            if future_mask.shape != (b, n_fut):
                raise ValueError(
                    "future_mask shape mismatch for single-step path: expected "
                    f"{(b, n_fut)}, got {tuple(future_mask.shape)}"
                )
            memory_mask = future_mask[:, None, None, :]

        for layer_idx, layer in enumerate(self.layers):
            x_norm = layer.norm1(h)
            k_cache_l = self._k_cache[:, layer_idx]
            v_cache_l = self._v_cache[:, layer_idx]
            attn_out, _, _ = layer.attn.forward_single_token(
                x_norm,
                cos,
                sin,
                k_cache_l,
                v_cache_l,
                new_len,
                insert_pos,
            )
            h = h + attn_out
            if layer.use_cross_attn:
                h = h + layer.cross_attn(
                    layer.norm_cross(h), memory, memory_mask
                )
            h2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                ffn = layer.compute_moe_ffn(h2)
                if (
                    layer_idx == self._last_moe_layer_idx
                    and layer.collect_routing_stats
                    and layer.last_router_distribution is not None
                ):
                    self._accumulate_last_moe_router_shift(
                        layer.last_router_distribution
                    )
            else:
                ffn = layer.mlp_dropout(layer.mlp(h2))
            h = h + ffn

        h = self.norm_f(h)
        return self.action_mu_head(h[:, 0, :])

    def forward(
        self,
        input: torch.Tensor,
        past_key_values: torch.Tensor | None = None,
        current_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single-step inference (no history)."""
        if past_key_values is not None:
            return self._forward_inference_onnx(
                input, past_key_values, current_pos
            )
        if input.ndim != 2:
            raise ValueError(f"Expected [B, D], got {input.shape}")
        mu_seq = self.sequence_mu(input[:, None, :], attn_mask=None)
        return mu_seq[:, 0, :]

    def single_step_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action mean for a single step using per-layer KV cache.

        Uses a ring-buffer KV cache with per-env absolute positions for RoPE.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected [B, D], got {x.shape}")
        B, _ = x.shape

        if self._k_cache is None:
            mu_seq = self.sequence_mu(x[:, None, :], attn_mask=None)
            return mu_seq[:, 0, :]

        # Ensure cache device matches
        if self._k_cache.device != x.device:
            self._k_cache = self._k_cache.to(x.device)
            self._v_cache = self._v_cache.to(x.device)
            self._kv_cache_len = self._kv_cache_len.to(x.device)
            self._kv_cache_write_idx = self._kv_cache_write_idx.to(x.device)
            self._kv_cache_abs_pos = self._kv_cache_abs_pos.to(x.device)

        h = self.obs_embed(x)[:, None, :]  # [B, 1, d_model]
        router_h = self._compute_router_hidden(x)
        if router_h is not None:
            router_h = router_h[:, None, :]

        # Ensure cache dtype matches compute dtype (convert once if needed)
        if self._k_cache.dtype != h.dtype:
            self._k_cache = self._k_cache.to(h.dtype)
            self._v_cache = self._v_cache.to(h.dtype)

        cache_len = self._kv_cache_len  # [B]
        insert_pos = self._kv_cache_write_idx  # [B]
        max_len = int(self.max_ctx_len)
        new_len = torch.clamp(cache_len + 1, max=max_len)  # [B]

        self._kv_cache_len = new_len
        self._kv_cache_write_idx = (insert_pos + 1) % max_len

        # RoPE frequencies for current absolute position
        pos = self._kv_cache_abs_pos  # [B]
        self._kv_cache_abs_pos = pos + 1
        pos_ids = pos.unsqueeze(1)  # [B, 1]
        cos, sin = self.get_cos_sin(h, pos_ids)

        for layer_idx, layer in enumerate(self.layers):
            x_norm = layer.norm1(h)
            k_cache_l = self._k_cache[
                :, layer_idx
            ]  # [B, L, n_kv_heads, head_dim]
            v_cache_l = self._v_cache[:, layer_idx]
            attn_out, _, _ = layer.attn.forward_single_token(
                x_norm,
                cos,
                sin,
                k_cache_l,
                v_cache_l,
                new_len,
                insert_pos,
            )
            h = h + attn_out
            # FFN/MoE path for single token (h: [B,1,D])
            h2 = layer.norm2(h)  # [B,1,D]
            if isinstance(layer, GroupedMoEBlock):
                ffn = layer.compute_moe_ffn(h2, router_x=router_h)
                if (
                    layer_idx == self._last_moe_layer_idx
                    and layer.collect_routing_stats
                    and layer.last_router_distribution is not None
                ):
                    self._accumulate_last_moe_router_shift(
                        layer.last_router_distribution
                    )

            else:
                ffn = layer.mlp_dropout(layer.mlp(h2))  # 保持 [B, 1, D]

            h = h + ffn

        h = self.norm_f(h)
        return self.action_mu_head(h[:, 0, :])

    def _forward_inference_onnx(
        self,
        x: torch.Tensor,
        past_key_values: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Single-step inference compatible with ONNX export.
        Aligns strictly with `single_step_mu` logic using Real-valued RoPE.

        Args:
            x: [B, D] (Batch=1 for ONNX usually)
            past_key_values: [n_layers, 2, B, max_len, n_kv_heads, head_dim]
            current_pos: [B] or scalar, the absolute step index (0, 1, 2...)

        Returns:
            action: [B, A]
            present_key_values: Updated KV cache tensor
        """
        # Embedding [B, D] -> [B, 1, D]
        h = self.obs_embed(x)[:, None, :]  # [1, 1, 512]
        router_h = self._compute_router_hidden(x)
        if router_h is not None:
            router_h = router_h[:, None, :]
        B = h.shape[0]  # 1

        # Calculate Cache Indices (Ring Buffer Logic)
        # past_key_values shape: [L, 2, B, T, H, D] -> T is index 3
        max_len = past_key_values.shape[3]  # 32

        if current_pos.ndim == 0:
            current_pos = current_pos.view(1).expand(B)

        # insert_pos: [B]
        insert_pos = current_pos % max_len

        # new_len: [B]
        new_len = torch.clamp(current_pos + 1, max=max_len)

        # position_ids: [B, 1]
        position_ids = current_pos.unsqueeze(1)

        # cos, sin shape: [B, 1, head_dim]
        cos, sin = self.get_cos_sin(h, position_ids)

        present_key_values_list = []
        routing_debug_outputs: list[torch.Tensor] = []
        export_routing_debug = torch.onnx.is_in_onnx_export()

        for i, layer in enumerate(self.layers):
            # Unpack Cache: [2, B, T, H, D]
            layer_past = past_key_values[i]
            k_cache = layer_past[0]
            v_cache = layer_past[1]

            h_norm = layer.norm1(h)

            # Attention
            attn_out, new_k_cache, new_v_cache = (
                layer.attn.forward_single_token(
                    x=h_norm,
                    cos=cos,
                    sin=sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    new_len=new_len,
                    insert_pos=insert_pos,
                )
            )

            h = h + attn_out

            # FFN / MoE
            h_norm2 = layer.norm2(h)

            if isinstance(layer, GroupedMoEBlock):
                if export_routing_debug:
                    ffn_out, topk_idx, router_logits = layer.compute_moe_ffn(
                        h_norm2,
                        router_x=router_h,
                        return_routing_debug=True,
                    )
                    routing_debug_outputs.extend([topk_idx, router_logits])
                else:
                    ffn_out = layer.compute_moe_ffn(h_norm2, router_x=router_h)
            else:
                # Dense MLP
                ffn_out = layer.mlp_dropout(layer.mlp(h_norm2))
            h = h + ffn_out
            current_layer_kv = torch.stack([new_k_cache, new_v_cache], dim=0)
            present_key_values_list.append(current_layer_kv)

        h = self.norm_f(h)
        action = self.action_mu_head(h[:, 0, :])
        present_key_values = torch.stack(present_key_values_list, dim=0)

        if export_routing_debug and routing_debug_outputs:
            return (action, present_key_values, *routing_debug_outputs)
        return action, present_key_values


class ReferenceRoutedGroupedMoETransformerPolicy(GroupedMoETransformerPolicy):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_config_dict: dict,
    ):
        module_config = dict(module_config_dict)
        if bool(module_config.get("use_future_cross_attn", False)):
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicy does not support "
                "use_future_cross_attn=True."
            )

        router_input_dim = module_config.get("router_input_dim", None)
        router_feature_indices = module_config.get(
            "router_feature_indices", None
        )
        if router_input_dim is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicy requires router_input_dim."
            )
        if router_feature_indices is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicy requires "
                "router_feature_indices."
            )

        self.router_input_dim = int(router_input_dim)
        self.router_feature_indices = tuple(
            int(idx) for idx in router_feature_indices
        )
        if self.router_input_dim <= 0:
            raise ValueError(
                f"router_input_dim must be positive, got {self.router_input_dim}."
            )
        if len(self.router_feature_indices) != self.router_input_dim:
            raise ValueError(
                "router_input_dim must match len(router_feature_indices): "
                f"{self.router_input_dim} vs {len(self.router_feature_indices)}."
            )
        if any(idx < 0 for idx in self.router_feature_indices):
            raise ValueError(
                f"router_feature_indices must be non-negative, got {self.router_feature_indices}."
            )
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            module_config_dict=module_config,
        )
        obs_in = int(self.obs_input_dim or self.input_dim)
        if any(idx >= obs_in for idx in self.router_feature_indices):
            raise ValueError(
                "router_feature_indices exceed the flat actor obs dim "
                f"{obs_in}: {self.router_feature_indices}"
            )
        self.router_embed_mlp_hidden = int(
            module_config.get(
                "router_embed_mlp_hidden", self.obs_embed_mlp_hidden
            )
        )
        self.register_buffer(
            "_router_feature_indices",
            torch.tensor(self.router_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.router_obs_embed = nn.Sequential(
            nn.Linear(self.router_input_dim, self.router_embed_mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.router_embed_mlp_hidden, self.d_model),
        )
        self._apply_freeze_router_state()

    def _apply_freeze_router_state(self) -> None:
        super()._apply_freeze_router_state()
        self.router_obs_embed.requires_grad_(not self.freeze_router)

    def _compute_router_hidden(self, x: torch.Tensor) -> torch.Tensor | None:
        if x.shape[-1] != int(self.obs_input_dim or self.input_dim):
            raise ValueError(
                "Reference-routed policy expected flat obs dim "
                f"{int(self.obs_input_dim or self.input_dim)}, got {x.shape[-1]}."
            )
        router_idx = self._router_feature_indices
        if router_idx.device != x.device:
            router_idx = router_idx.to(x.device)
        router_obs = torch.index_select(x, dim=x.ndim - 1, index=router_idx)
        return self.router_obs_embed(router_obs)

    def _forward_inference_onnx_cond(
        self,
        state_x: torch.Tensor,
        future_tokens: torch.Tensor,
        past_key_values: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if not self.use_future_cross_attn:
            raise ValueError(
                "_forward_inference_onnx_cond requires use_future_cross_attn=True"
            )
        if state_x.ndim != 2:
            raise ValueError(
                f"state_x must have shape [B, D_state], got {tuple(state_x.shape)}"
            )
        if future_tokens.ndim != 3:
            raise ValueError(
                "future_tokens must have shape [B, N_fut, D_fut], "
                f"got {tuple(future_tokens.shape)}"
            )
        h = self.state_obs_embed(state_x)[:, None, :]
        memory = self._embed_future_tokens(future_tokens)
        b = h.shape[0]
        max_len = past_key_values.shape[3]
        if current_pos.ndim == 0:
            current_pos = current_pos.view(1).expand(b)
        insert_pos = current_pos % max_len
        new_len = torch.clamp(current_pos + 1, max=max_len)
        position_ids = current_pos.unsqueeze(1)
        cos, sin = self.get_cos_sin(h, position_ids)

        present_key_values_list = []
        routing_debug_outputs: list[torch.Tensor] = []
        export_routing_debug = torch.onnx.is_in_onnx_export()
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i]
            k_cache = layer_past[0]
            v_cache = layer_past[1]

            h_norm = layer.norm1(h)
            attn_out, new_k_cache, new_v_cache = (
                layer.attn.forward_single_token(
                    x=h_norm,
                    cos=cos,
                    sin=sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    new_len=new_len,
                    insert_pos=insert_pos,
                )
            )
            h = h + attn_out

            if layer.use_cross_attn:
                h = h + layer.cross_attn(layer.norm_cross(h), memory, None)

            h_norm2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                if export_routing_debug:
                    ffn_out, topk_idx, router_logits = layer.compute_moe_ffn(
                        h_norm2,
                        return_routing_debug=True,
                    )
                    routing_debug_outputs.extend([topk_idx, router_logits])
                else:
                    ffn_out = layer.compute_moe_ffn(h_norm2)
            else:
                ffn_out = layer.mlp_dropout(layer.mlp(h_norm2))
            h = h + ffn_out
            current_layer_kv = torch.stack([new_k_cache, new_v_cache], dim=0)
            present_key_values_list.append(current_layer_kv)

        h = self.norm_f(h)
        action = self.action_mu_head(h[:, 0, :])
        present_key_values = torch.stack(present_key_values_list, dim=0)
        if export_routing_debug and routing_debug_outputs:
            return (action, present_key_values, *routing_debug_outputs)
        return action, present_key_values


class ReferenceRoutedGroupedMoETransformerPolicyV2(
    GroupedMoETransformerPolicy
):
    supports_explicit_ref_aux_hidden = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_config_dict: dict,
    ):
        module_config = dict(module_config_dict)
        if bool(module_config.get("use_future_cross_attn", False)):
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 does not "
                "support use_future_cross_attn=True."
            )
        state_obs_input_dim = module_config.get("state_obs_input_dim", None)
        ref_cur_token_dim = module_config.get("ref_cur_token_dim", None)
        ref_fut_token_dim = module_config.get("ref_fut_token_dim", None)
        ref_fut_seq_len = module_config.get("ref_fut_seq_len", None)
        state_feature_indices = module_config.get(
            "state_feature_indices", None
        )
        ref_cur_feature_indices = module_config.get(
            "ref_cur_feature_indices", None
        )
        ref_fut_slices = module_config.get("ref_fut_slices", None)
        if state_obs_input_dim is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 requires "
                "state_obs_input_dim."
            )
        if ref_cur_token_dim is None or ref_fut_token_dim is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 requires "
                "ref_cur_token_dim and ref_fut_token_dim."
            )
        if ref_fut_seq_len is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 requires "
                "ref_fut_seq_len."
            )
        if state_feature_indices is None or ref_cur_feature_indices is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 requires "
                "state_feature_indices and ref_cur_feature_indices."
            )
        if ref_fut_slices is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 requires "
                "ref_fut_slices."
            )

        self.full_obs_input_dim = int(input_dim)
        self.state_obs_input_dim = int(state_obs_input_dim)
        self.ref_cur_token_dim = int(ref_cur_token_dim)
        self.ref_fut_token_dim = int(ref_fut_token_dim)
        self.ref_fut_seq_len = int(ref_fut_seq_len)
        self.state_feature_indices = tuple(
            int(idx) for idx in state_feature_indices
        )
        self.ref_cur_feature_indices = tuple(
            int(idx) for idx in ref_cur_feature_indices
        )
        self.ref_fut_slices = tuple(
            (int(start), int(end), int(dim))
            for start, end, dim in ref_fut_slices
        )
        if self.state_obs_input_dim <= 0:
            raise ValueError(
                "state_obs_input_dim must be positive, got "
                f"{self.state_obs_input_dim}."
            )
        if self.ref_cur_token_dim <= 0 or self.ref_fut_token_dim <= 0:
            raise ValueError(
                "ref token dims must be positive, got "
                f"{self.ref_cur_token_dim} and {self.ref_fut_token_dim}."
            )
        if self.ref_cur_token_dim != self.ref_fut_token_dim:
            raise ValueError(
                "current/future ref token dims must match, got "
                f"{self.ref_cur_token_dim} and {self.ref_fut_token_dim}."
            )
        if self.ref_fut_seq_len <= 0:
            raise ValueError(
                f"ref_fut_seq_len must be positive, got {self.ref_fut_seq_len}."
            )
        if len(self.state_feature_indices) != self.state_obs_input_dim:
            raise ValueError(
                "state_obs_input_dim must match len(state_feature_indices): "
                f"{self.state_obs_input_dim} vs {len(self.state_feature_indices)}."
            )
        if len(self.ref_cur_feature_indices) != self.ref_cur_token_dim:
            raise ValueError(
                "ref_cur_token_dim must match len(ref_cur_feature_indices): "
                f"{self.ref_cur_token_dim} vs {len(self.ref_cur_feature_indices)}."
            )
        fut_flat_dim = 0
        for start, end, dim in self.ref_fut_slices:
            if end <= start or dim <= 0:
                raise ValueError(
                    f"Invalid ref_fut_slices entry {(start, end, dim)}."
                )
            if (end - start) != self.ref_fut_seq_len * dim:
                raise ValueError(
                    "Future ref slice span must equal ref_fut_seq_len * dim, got "
                    f"{(start, end, dim)} with ref_fut_seq_len={self.ref_fut_seq_len}."
                )
            fut_flat_dim += end - start
        expected_full_input_dim = (
            self.state_obs_input_dim + self.ref_cur_token_dim + fut_flat_dim
        )
        if self.full_obs_input_dim != expected_full_input_dim:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 expected full "
                f"input dim {expected_full_input_dim}, got {self.full_obs_input_dim}."
            )

        self.ref_hist_n_layers = int(module_config.get("ref_hist_n_layers", 1))
        if self.ref_hist_n_layers != 1:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 currently supports "
                "exactly one ref history attention layer."
            )
        self.ref_future_conv_channels = int(
            module_config.get(
                "ref_future_conv_channels", self.ref_cur_token_dim
            )
        )
        self.ref_future_conv_layers = int(
            module_config.get("ref_future_conv_layers", 2)
        )
        self.ref_future_conv_kernel_size = int(
            module_config.get("ref_future_conv_kernel_size", 3)
        )
        self.ref_future_conv_stride = int(
            module_config.get("ref_future_conv_stride", 2)
        )
        if self.ref_future_conv_layers <= 0:
            raise ValueError(
                "ref_future_conv_layers must be positive, got "
                f"{self.ref_future_conv_layers}."
            )
        if self.ref_future_conv_kernel_size <= 0:
            raise ValueError(
                "ref_future_conv_kernel_size must be positive, got "
                f"{self.ref_future_conv_kernel_size}."
            )
        if self.ref_future_conv_stride <= 0:
            raise ValueError(
                "ref_future_conv_stride must be positive, got "
                f"{self.ref_future_conv_stride}."
            )

        module_config["input_dim_override"] = self.state_obs_input_dim
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            module_config_dict=module_config,
        )
        self.onnx_kv_layers = int(self.ref_hist_n_layers + self.n_layers)
        self.register_buffer(
            "_state_feature_indices",
            torch.tensor(self.state_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_ref_cur_feature_indices",
            torch.tensor(self.ref_cur_feature_indices, dtype=torch.long),
            persistent=False,
        )

        self.ref_frame_embed = nn.Sequential(
            nn.Linear(self.ref_cur_token_dim, self.obs_embed_mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.obs_embed_mlp_hidden, self.d_model),
        )
        self.ref_hist_norm = RMSNorm(self.d_model)
        self.ref_hist_attn = ModernAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            use_qk_norm=self.use_qk_norm,
            use_gated_attn=self.use_gated_attn,
            gated_attn_type=self.gated_attn_type,
            attn_dropout=self.attn_dropout,
        )
        self.ref_hist_out_norm = RMSNorm(self.d_model)

        padding = self.ref_future_conv_kernel_size // 2
        conv_modules: list[nn.Module] = []
        in_ch = self.d_model
        for layer_idx in range(self.ref_future_conv_layers):
            out_ch = (
                self.d_model
                if layer_idx == self.ref_future_conv_layers - 1
                else self.ref_future_conv_channels
            )
            conv_modules.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.ref_future_conv_kernel_size,
                    stride=self.ref_future_conv_stride,
                    padding=padding,
                    bias=True,
                )
            )
            conv_modules.append(nn.SiLU())
            in_ch = out_ch
        self.ref_future_conv = nn.Sequential(*conv_modules)

        self.actor_ref_pool = SingleQueryAttentionPool(self.d_model)
        self.router_ref_pool = SingleQueryAttentionPool(self.d_model)
        self.router_query = nn.Parameter(torch.zeros(self.d_model))
        self.actor_ref_ctx_norm = RMSNorm(self.d_model)
        self.actor_film_hidden_norm = RMSNorm(self.d_model)
        self.actor_ref_film = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 2 * self.d_model),
        )
        nn.init.zeros_(self.actor_ref_film[-1].weight)
        nn.init.zeros_(self.actor_ref_film[-1].bias)
        self.actor_film_gain_max = float(
            module_config.get("actor_film_gain_max", 1.0)
        )
        self.actor_film_gain_init = float(
            module_config.get("actor_film_gain_init", 0.05)
        )
        if self.actor_film_gain_max <= 0.0:
            raise ValueError(
                "actor_film_gain_max must be positive, got "
                f"{self.actor_film_gain_max}."
            )
        if not (0.0 < self.actor_film_gain_init < self.actor_film_gain_max):
            raise ValueError(
                "actor_film_gain_init must be in (0, actor_film_gain_max), "
                f"got {self.actor_film_gain_init} with max "
                f"{self.actor_film_gain_max}."
            )
        gain_init_ratio = self.actor_film_gain_init / self.actor_film_gain_max
        self.actor_film_gain_raw = nn.Parameter(
            torch.full(
                (self.d_model,),
                math.log(gain_init_ratio / (1.0 - gain_init_ratio)),
            )
        )
        self.actor_film_scale_max = 0.5
        self.actor_film_shift_max = 0.5
        self.actor_film_delta_rms_eps = float(
            module_config.get("actor_film_delta_rms_eps", 1.0e-6)
        )
        if self.actor_film_delta_rms_eps <= 0.0:
            raise ValueError(
                "actor_film_delta_rms_eps must be positive, got "
                f"{self.actor_film_delta_rms_eps}."
            )

        self._ref_hist_k_cache: torch.Tensor | None = None
        self._ref_hist_v_cache: torch.Tensor | None = None
        self._apply_freeze_router_state()

    def _apply_freeze_router_state(self) -> None:
        super()._apply_freeze_router_state()
        requires_grad = not self.freeze_router
        self.ref_frame_embed.requires_grad_(requires_grad)
        self.ref_hist_norm.requires_grad_(requires_grad)
        self.ref_hist_attn.requires_grad_(requires_grad)
        self.ref_hist_out_norm.requires_grad_(requires_grad)
        self.ref_future_conv.requires_grad_(requires_grad)
        self.router_ref_pool.requires_grad_(requires_grad)
        self.router_query.requires_grad_(requires_grad)

    def _build_shared_ref_tokens(
        self,
        ref_cur_x: torch.Tensor,
        ref_fut_x: torch.Tensor,
        pos: torch.Tensor,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        with self._router_no_grad_context():
            ref_cur_h = self.ref_frame_embed(ref_cur_x)
            ref_hist_attn = self.ref_hist_attn(
                self.ref_hist_norm(ref_cur_h),
                *self.get_cos_sin(ref_cur_h, pos),
                mask=tgt_mask,
            )
            ref_hist_h = self.ref_hist_out_norm(ref_cur_h + ref_hist_attn)
            ref_fut_tokens = self._encode_future_tokens(ref_fut_x)
            return torch.cat([ref_hist_h.unsqueeze(2), ref_fut_tokens], dim=2)

    def _build_shared_ref_tokens_single_step(
        self,
        ref_cur_x: torch.Tensor,
        ref_fut_x: torch.Tensor,
        pos_ids: torch.Tensor,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        new_len: torch.Tensor,
        insert_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._router_no_grad_context():
            ref_cur_h = self.ref_frame_embed(ref_cur_x)[:, None, :]
            ref_cos, ref_sin = self.get_cos_sin(ref_cur_h, pos_ids)
            ref_hist_attn, ref_k_cache, ref_v_cache = (
                self.ref_hist_attn.forward_single_token(
                    self.ref_hist_norm(ref_cur_h),
                    ref_cos,
                    ref_sin,
                    k_cache,
                    v_cache,
                    new_len,
                    insert_pos,
                )
            )
            ref_hist_h = self.ref_hist_out_norm(ref_cur_h + ref_hist_attn)
            ref_fut_tokens = self._encode_future_tokens(ref_fut_x)
            shared_ref_tokens = torch.cat([ref_hist_h, ref_fut_tokens], dim=1)
        return shared_ref_tokens, ref_k_cache, ref_v_cache

    def _split_actor_ref_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim not in (2, 3):
            raise ValueError(
                f"Expected full obs tensor with ndim 2 or 3, got {x.ndim}."
            )
        if int(x.shape[-1]) != self.full_obs_input_dim:
            raise ValueError(
                "Full obs dim mismatch for reference router V2: expected "
                f"{self.full_obs_input_dim}, got {int(x.shape[-1])}."
            )
        state_idx = self._state_feature_indices.to(x.device)
        ref_cur_idx = self._ref_cur_feature_indices.to(x.device)
        state_x = torch.index_select(x, dim=x.ndim - 1, index=state_idx)
        ref_cur_x = torch.index_select(x, dim=x.ndim - 1, index=ref_cur_idx)
        fut_parts: list[torch.Tensor] = []
        for start, end, dim in self.ref_fut_slices:
            chunk = x[..., start:end]
            if x.ndim == 2:
                fut_parts.append(
                    chunk.reshape(int(x.shape[0]), self.ref_fut_seq_len, dim)
                )
            else:
                fut_parts.append(
                    chunk.reshape(
                        int(x.shape[0]),
                        int(x.shape[1]),
                        self.ref_fut_seq_len,
                        dim,
                    )
                )
        ref_fut_x = torch.cat(fut_parts, dim=-1)
        return state_x, ref_cur_x, ref_fut_x

    def _encode_future_tokens(self, ref_fut_x: torch.Tensor) -> torch.Tensor:
        if ref_fut_x.ndim == 3:
            fut = self.ref_frame_embed(ref_fut_x)
            return self.ref_future_conv(fut.transpose(1, 2)).transpose(1, 2)
        if ref_fut_x.ndim == 4:
            batch, time, seq_len, dim = ref_fut_x.shape
            fut = self.ref_frame_embed(
                ref_fut_x.reshape(batch * time, seq_len, dim)
            )
            fut = self.ref_future_conv(fut.transpose(1, 2)).transpose(1, 2)
            return fut.reshape(batch, time, fut.shape[1], self.d_model)
        raise ValueError(
            f"Expected ref_fut_x with ndim 3 or 4, got {ref_fut_x.ndim}."
        )

    def _pool_router_context(
        self, shared_ref_tokens: torch.Tensor
    ) -> torch.Tensor:
        with self._router_no_grad_context():
            if shared_ref_tokens.ndim == 3:
                query = self.router_query.to(
                    device=shared_ref_tokens.device,
                    dtype=shared_ref_tokens.dtype,
                )[None, :].expand(int(shared_ref_tokens.shape[0]), -1)
            elif shared_ref_tokens.ndim == 4:
                query = self.router_query.to(
                    device=shared_ref_tokens.device,
                    dtype=shared_ref_tokens.dtype,
                )[None, None, :].expand(
                    int(shared_ref_tokens.shape[0]),
                    int(shared_ref_tokens.shape[1]),
                    -1,
                )
            else:
                raise ValueError(
                    "shared_ref_tokens must have ndim 3 or 4, got "
                    f"{shared_ref_tokens.ndim}."
                )
            return self.router_ref_pool(query, shared_ref_tokens)

    def _apply_actor_ref_film(
        self, state_hidden: torch.Tensor, actor_ref_ctx: torch.Tensor
    ) -> torch.Tensor:
        ctx = self.actor_ref_ctx_norm(actor_ref_ctx)
        scale_raw, shift_raw = self.actor_ref_film(ctx).chunk(2, dim=-1)
        scale = self.actor_film_scale_max * torch.tanh(scale_raw)
        shift = self.actor_film_shift_max * torch.tanh(shift_raw)
        hidden_norm = self.actor_film_hidden_norm(state_hidden)
        delta = scale * hidden_norm + shift
        delta = self._normalize_actor_film_delta(delta)
        gain = self._actor_film_gain().to(
            device=state_hidden.device, dtype=state_hidden.dtype
        )
        expand_shape = [1] * (delta.ndim - 1) + [self.d_model]
        return state_hidden + delta * gain.view(*expand_shape)

    def _actor_film_gain(self) -> torch.Tensor:
        return self.actor_film_gain_max * torch.sigmoid(
            self.actor_film_gain_raw
        )

    def _normalize_actor_film_delta(self, delta: torch.Tensor) -> torch.Tensor:
        rms = delta.pow(2).mean(dim=-1, keepdim=True)
        return delta * torch.rsqrt(rms + self.actor_film_delta_rms_eps)

    def _ensure_internal_cache_device(
        self,
        device,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        if self._k_cache is not None and self._k_cache.device != device:
            self._k_cache = self._k_cache.to(device)
            self._v_cache = self._v_cache.to(device)
            self._ref_hist_k_cache = self._ref_hist_k_cache.to(device)
            self._ref_hist_v_cache = self._ref_hist_v_cache.to(device)
            self._kv_cache_len = self._kv_cache_len.to(device)
            self._kv_cache_write_idx = self._kv_cache_write_idx.to(device)
            self._kv_cache_abs_pos = self._kv_cache_abs_pos.to(device)
        if (
            dtype is not None
            and self._k_cache is not None
            and self._k_cache.dtype != dtype
        ):
            self._k_cache = self._k_cache.to(dtype)
            self._v_cache = self._v_cache.to(dtype)
            self._ref_hist_k_cache = self._ref_hist_k_cache.to(dtype)
            self._ref_hist_v_cache = self._ref_hist_v_cache.to(dtype)

    def reset_kv_cache(self, num_envs: int, device):
        cache_dtype = (
            torch.float16
            if torch.device(device).type == "cuda"
            else torch.float32
        )
        self._k_cache = torch.zeros(
            num_envs,
            self.n_layers,
            self.max_ctx_len,
            self.n_kv_heads,
            self.head_dim,
            device=device,
            dtype=cache_dtype,
        )
        self._v_cache = torch.zeros_like(self._k_cache)
        self._ref_hist_k_cache = torch.zeros(
            num_envs,
            self.ref_hist_n_layers,
            self.max_ctx_len,
            self.n_kv_heads,
            self.head_dim,
            device=device,
            dtype=cache_dtype,
        )
        self._ref_hist_v_cache = torch.zeros_like(self._ref_hist_k_cache)
        self._kv_cache_len = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._kv_cache_write_idx = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._kv_cache_abs_pos = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )
        self._init_last_moe_router_shift_state(num_envs, device)

    def clear_env_cache(self, env_ids: torch.Tensor | None):
        if self._k_cache is None:
            return
        if env_ids is None:
            self._k_cache.zero_()
            self._v_cache.zero_()
            self._ref_hist_k_cache.zero_()
            self._ref_hist_v_cache.zero_()
            self._kv_cache_len.zero_()
            self._kv_cache_write_idx.zero_()
            self._kv_cache_abs_pos.zero_()
            if self._prev_last_moe_router_p is not None:
                self._prev_last_moe_router_p.zero_()
            if self._prev_last_moe_router_valid is not None:
                self._prev_last_moe_router_valid.zero_()
            if self._last_moe_router_js_sum is not None:
                self._last_moe_router_js_sum.zero_()
            if self._last_moe_router_js_count is not None:
                self._last_moe_router_js_count.zero_()
            if self._last_moe_router_top1_switch_sum is not None:
                self._last_moe_router_top1_switch_sum.zero_()
            if self._last_moe_router_top1_switch_count is not None:
                self._last_moe_router_top1_switch_count.zero_()
            return
        self._k_cache[env_ids] = 0.0
        self._v_cache[env_ids] = 0.0
        self._ref_hist_k_cache[env_ids] = 0.0
        self._ref_hist_v_cache[env_ids] = 0.0
        self._kv_cache_len[env_ids] = 0
        self._kv_cache_write_idx[env_ids] = 0
        self._kv_cache_abs_pos[env_ids] = 0
        if self._prev_last_moe_router_valid is not None:
            self._prev_last_moe_router_valid[env_ids] = False
        if self._prev_last_moe_router_p is not None:
            self._prev_last_moe_router_p[env_ids] = 0.0

    def predict_aux_from_pre_moe(
        self,
        pre_moe_hidden: torch.Tensor,
        *,
        ref_aux_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        aux_outputs = super().predict_aux_from_pre_moe(
            pre_moe_hidden, ref_aux_hidden=ref_aux_hidden
        )
        if self.aux_ref_keybody_pos_head is not None:
            if ref_aux_hidden is None:
                raise ValueError(
                    "Missing shared-ref auxiliary hidden state for "
                    "ref_keybody_rel_pos prediction."
                )
            aux_outputs["ref_keybody_rel_pos"] = self.aux_ref_keybody_pos_head(
                ref_aux_hidden
            ).reshape(
                ref_aux_hidden.shape[0],
                ref_aux_hidden.shape[1],
                self.aux_keybody_pos_dim,
                3,
            )
        return aux_outputs

    def sequence_mu(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_pre_moe_hidden: bool = False,
        return_ref_aux_hidden: bool = False,
        return_router_features: bool = False,
        return_router_temporal_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        _, time, _ = x.shape
        state_x, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        state_h = self.obs_embed(state_x)

        if attn_mask is not None:
            tgt_mask = attn_mask.unsqueeze(1)
            start_idx = attn_mask.to(torch.int64).argmax(dim=-1)
            t_idx = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(int(x.shape[0]), time)
            pos = t_idx - start_idx
        else:
            tgt_mask = None
            pos = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(int(x.shape[0]), time)

        shared_ref_tokens = self._build_shared_ref_tokens(
            ref_cur_x=ref_cur_x,
            ref_fut_x=ref_fut_x,
            pos=pos,
            tgt_mask=tgt_mask,
        )
        actor_ref_ctx = self.actor_ref_pool(state_h, shared_ref_tokens)
        router_h = self._pool_router_context(shared_ref_tokens)
        cos, sin = self.get_cos_sin(state_h, pos)
        if return_hidden and return_pre_moe_hidden:
            raise ValueError(
                "return_hidden and return_pre_moe_hidden cannot both be True."
            )
        block0_h = self._forward_layers_range(
            state_h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h=router_h,
            start_layer=0,
            end_layer=1,
        )
        h = self._apply_actor_ref_film(block0_h, actor_ref_ctx)
        forward_out = self._forward_layers_range(
            h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h=router_h,
            start_layer=1,
            end_layer=len(self.layers),
            return_router_features=return_router_features,
            return_router_temporal_features=return_router_temporal_features,
        )
        extras: list[torch.Tensor] = []
        if isinstance(forward_out, tuple):
            h = forward_out[0]
            extras = list(forward_out[1:])
        else:
            h = forward_out
        h = self.norm_f(h)
        mu = self.action_mu_head(h)
        outputs: list[torch.Tensor] = [mu]
        if return_pre_moe_hidden:
            outputs.append(block0_h)
        if return_ref_aux_hidden:
            outputs.append(router_h)
        if return_router_features:
            outputs.append(extras.pop(0))
        if return_router_temporal_features:
            outputs.append(extras.pop(0))
        if len(outputs) > 1:
            return tuple(outputs)
        if return_hidden:
            return mu, h
        return mu

    def sequence_hidden(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, time, _ = x.shape
        state_x, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        state_h = self.obs_embed(state_x)

        if attn_mask is not None:
            tgt_mask = attn_mask.unsqueeze(1)
            start_idx = attn_mask.to(torch.int64).argmax(dim=-1)
            t_idx = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(int(x.shape[0]), time)
            pos = t_idx - start_idx
        else:
            tgt_mask = None
            pos = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(int(x.shape[0]), time)

        shared_ref_tokens = self._build_shared_ref_tokens(
            ref_cur_x=ref_cur_x,
            ref_fut_x=ref_fut_x,
            pos=pos,
            tgt_mask=tgt_mask,
        )
        actor_ref_ctx = self.actor_ref_pool(state_h, shared_ref_tokens)
        router_h = self._pool_router_context(shared_ref_tokens)
        cos, sin = self.get_cos_sin(state_h, pos)
        h = self._forward_layers_range(
            state_h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h=router_h,
            start_layer=0,
            end_layer=1,
        )
        h = self._apply_actor_ref_film(h, actor_ref_ctx)
        h = self._forward_layers_range(
            h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h=router_h,
            start_layer=1,
            end_layer=len(self.layers),
        )
        h = self.norm_f(h)
        return h

    def forward(
        self,
        input: torch.Tensor,
        past_key_values: torch.Tensor | None = None,
        current_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is not None:
            return self._forward_inference_onnx(
                input, past_key_values, current_pos
            )
        if input.ndim != 2:
            raise ValueError(f"Expected [B, D], got {input.shape}")
        mu_seq = self.sequence_mu(input[:, None, :], attn_mask=None)
        return mu_seq[:, 0, :]

    def single_step_mu(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [B, D], got {x.shape}")
        state_x, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        batch = int(state_x.shape[0])
        if self._k_cache is None:
            mu_seq = self.sequence_mu(x[:, None, :], attn_mask=None)
            return mu_seq[:, 0, :]

        state_h = self.obs_embed(state_x)
        self._ensure_internal_cache_device(x.device, dtype=state_h.dtype)

        cache_len = self._kv_cache_len
        insert_pos = self._kv_cache_write_idx
        max_len = int(self.max_ctx_len)
        new_len = torch.clamp(cache_len + 1, max=max_len)

        self._kv_cache_len = new_len
        self._kv_cache_write_idx = (insert_pos + 1) % max_len

        pos = self._kv_cache_abs_pos
        self._kv_cache_abs_pos = pos + 1
        pos_ids = pos.unsqueeze(1)
        shared_ref_tokens, _, _ = self._build_shared_ref_tokens_single_step(
            ref_cur_x=ref_cur_x,
            ref_fut_x=ref_fut_x,
            pos_ids=pos_ids,
            k_cache=self._ref_hist_k_cache[:, 0],
            v_cache=self._ref_hist_v_cache[:, 0],
            new_len=new_len,
            insert_pos=insert_pos,
        )
        actor_ref_ctx = self.actor_ref_pool(state_h, shared_ref_tokens)[
            :, None, :
        ]
        router_h = self._pool_router_context(shared_ref_tokens)[:, None, :]
        cos, sin = self.get_cos_sin(state_h[:, None, :], pos_ids)

        h = state_h[:, None, :]
        for layer_idx, layer in enumerate(self.layers[:1]):
            x_norm = layer.norm1(h)
            k_cache_l = self._k_cache[:, layer_idx]
            v_cache_l = self._v_cache[:, layer_idx]
            attn_out, _, _ = layer.attn.forward_single_token(
                x_norm,
                cos,
                sin,
                k_cache_l,
                v_cache_l,
                new_len,
                insert_pos,
            )
            h = h + attn_out
            h2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                ffn = layer.compute_moe_ffn(h2, router_x=router_h)
                if (
                    layer_idx == self._last_moe_layer_idx
                    and layer.collect_routing_stats
                    and layer.last_router_distribution is not None
                ):
                    self._accumulate_last_moe_router_shift(
                        layer.last_router_distribution
                    )
            else:
                ffn = layer.mlp_dropout(layer.mlp(h2))
            h = h + ffn

        h = self._apply_actor_ref_film(h, actor_ref_ctx)

        for layer_idx, layer in enumerate(self.layers[1:], start=1):
            x_norm = layer.norm1(h)
            k_cache_l = self._k_cache[:, layer_idx]
            v_cache_l = self._v_cache[:, layer_idx]
            attn_out, _, _ = layer.attn.forward_single_token(
                x_norm,
                cos,
                sin,
                k_cache_l,
                v_cache_l,
                new_len,
                insert_pos,
            )
            h = h + attn_out
            h2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                ffn = layer.compute_moe_ffn(h2, router_x=router_h)
                if (
                    layer_idx == self._last_moe_layer_idx
                    and layer.collect_routing_stats
                    and layer.last_router_distribution is not None
                ):
                    self._accumulate_last_moe_router_shift(
                        layer.last_router_distribution
                    )
            else:
                ffn = layer.mlp_dropout(layer.mlp(h2))
            h = h + ffn

        h = self.norm_f(h)
        return self.action_mu_head(h[:, 0, :]).reshape(batch, -1)

    def _forward_inference_onnx(
        self,
        x: torch.Tensor,
        past_key_values: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        state_x, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        state_h = self.obs_embed(state_x)
        batch = state_h.shape[0]
        max_len = past_key_values.shape[3]

        if current_pos.ndim == 0:
            current_pos = current_pos.view(1).expand(batch)

        insert_pos = current_pos % max_len
        new_len = torch.clamp(current_pos + 1, max=max_len)
        position_ids = current_pos.unsqueeze(1)
        ref_layer_past = past_key_values[0]
        shared_ref_tokens, ref_k_cache, ref_v_cache = (
            self._build_shared_ref_tokens_single_step(
                ref_cur_x=ref_cur_x,
                ref_fut_x=ref_fut_x,
                pos_ids=position_ids,
                k_cache=ref_layer_past[0],
                v_cache=ref_layer_past[1],
                new_len=new_len,
                insert_pos=insert_pos,
            )
        )
        actor_ref_ctx = self.actor_ref_pool(state_h, shared_ref_tokens)[
            :, None, :
        ]
        router_h = self._pool_router_context(shared_ref_tokens)[:, None, :]
        cos, sin = self.get_cos_sin(state_h[:, None, :], position_ids)

        present_key_values_list = [
            torch.stack([ref_k_cache, ref_v_cache], dim=0)
        ]
        routing_debug_outputs: list[torch.Tensor] = []
        export_routing_debug = torch.onnx.is_in_onnx_export()

        h = state_h[:, None, :]
        for i, layer in enumerate(self.layers[:1]):
            layer_past = past_key_values[self.ref_hist_n_layers + i]
            k_cache = layer_past[0]
            v_cache = layer_past[1]

            h_norm = layer.norm1(h)
            attn_out, new_k_cache, new_v_cache = (
                layer.attn.forward_single_token(
                    x=h_norm,
                    cos=cos,
                    sin=sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    new_len=new_len,
                    insert_pos=insert_pos,
                )
            )
            h = h + attn_out

            h_norm2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                if export_routing_debug:
                    ffn_out, topk_idx, router_logits = layer.compute_moe_ffn(
                        h_norm2,
                        router_x=router_h,
                        return_routing_debug=True,
                    )
                    routing_debug_outputs.extend([topk_idx, router_logits])
                else:
                    ffn_out = layer.compute_moe_ffn(h_norm2, router_x=router_h)
            else:
                ffn_out = layer.mlp_dropout(layer.mlp(h_norm2))
            h = h + ffn_out
            current_layer_kv = torch.stack([new_k_cache, new_v_cache], dim=0)
            present_key_values_list.append(current_layer_kv)

        h = self._apply_actor_ref_film(h, actor_ref_ctx)

        for i, layer in enumerate(self.layers[1:], start=1):
            layer_past = past_key_values[self.ref_hist_n_layers + i]
            k_cache = layer_past[0]
            v_cache = layer_past[1]

            h_norm = layer.norm1(h)
            attn_out, new_k_cache, new_v_cache = (
                layer.attn.forward_single_token(
                    x=h_norm,
                    cos=cos,
                    sin=sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    new_len=new_len,
                    insert_pos=insert_pos,
                )
            )
            h = h + attn_out

            h_norm2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                if export_routing_debug:
                    ffn_out, topk_idx, router_logits = layer.compute_moe_ffn(
                        h_norm2,
                        router_x=router_h,
                        return_routing_debug=True,
                    )
                    routing_debug_outputs.extend([topk_idx, router_logits])
                else:
                    ffn_out = layer.compute_moe_ffn(h_norm2, router_x=router_h)
            else:
                ffn_out = layer.mlp_dropout(layer.mlp(h_norm2))
            h = h + ffn_out
            current_layer_kv = torch.stack([new_k_cache, new_v_cache], dim=0)
            present_key_values_list.append(current_layer_kv)

        h = self.norm_f(h)
        action = self.action_mu_head(h[:, 0, :])
        present_key_values = torch.stack(present_key_values_list, dim=0)

        if export_routing_debug and routing_debug_outputs:
            return (action, present_key_values, *routing_debug_outputs)
        return action, present_key_values


class ReferenceRoutedGroupedMoETransformerPolicyV3(
    ReferenceRoutedGroupedMoETransformerPolicyV2
):
    supports_explicit_ref_aux_hidden = True

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_config_dict: dict,
    ):
        module_config = dict(module_config_dict)
        if bool(module_config.get("use_future_cross_attn", False)):
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 does not "
                "support use_future_cross_attn=True."
            )
        state_obs_input_dim = module_config.get("state_obs_input_dim", None)
        ref_cur_token_dim = module_config.get("ref_cur_token_dim", None)
        ref_fut_token_dim = module_config.get("ref_fut_token_dim", None)
        ref_fut_seq_len = module_config.get("ref_fut_seq_len", None)
        state_feature_indices = module_config.get(
            "state_feature_indices", None
        )
        ref_cur_feature_indices = module_config.get(
            "ref_cur_feature_indices", None
        )
        ref_fut_slices = module_config.get("ref_fut_slices", None)
        if state_obs_input_dim is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 requires "
                "state_obs_input_dim."
            )
        if ref_cur_token_dim is None or ref_fut_token_dim is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 requires "
                "ref_cur_token_dim and ref_fut_token_dim."
            )
        if ref_fut_seq_len is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 requires "
                "ref_fut_seq_len."
            )
        if state_feature_indices is None or ref_cur_feature_indices is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 requires "
                "state_feature_indices and ref_cur_feature_indices."
            )
        if ref_fut_slices is None:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 requires "
                "ref_fut_slices."
            )

        self.full_obs_input_dim = int(input_dim)
        self.state_obs_input_dim = int(state_obs_input_dim)
        self.ref_cur_token_dim = int(ref_cur_token_dim)
        self.ref_fut_token_dim = int(ref_fut_token_dim)
        self.ref_fut_seq_len = int(ref_fut_seq_len)
        self.state_feature_indices = tuple(
            int(idx) for idx in state_feature_indices
        )
        self.ref_cur_feature_indices = tuple(
            int(idx) for idx in ref_cur_feature_indices
        )
        self.ref_fut_slices = tuple(
            (int(start), int(end), int(dim))
            for start, end, dim in ref_fut_slices
        )
        if self.state_obs_input_dim <= 0:
            raise ValueError(
                "state_obs_input_dim must be positive, got "
                f"{self.state_obs_input_dim}."
            )
        if self.ref_cur_token_dim <= 0 or self.ref_fut_token_dim <= 0:
            raise ValueError(
                "ref token dims must be positive, got "
                f"{self.ref_cur_token_dim} and {self.ref_fut_token_dim}."
            )
        if self.ref_cur_token_dim != self.ref_fut_token_dim:
            raise ValueError(
                "current/future ref token dims must match, got "
                f"{self.ref_cur_token_dim} and {self.ref_fut_token_dim}."
            )
        if self.ref_fut_seq_len <= 0:
            raise ValueError(
                f"ref_fut_seq_len must be positive, got {self.ref_fut_seq_len}."
            )
        if len(self.state_feature_indices) != self.state_obs_input_dim:
            raise ValueError(
                "state_obs_input_dim must match len(state_feature_indices): "
                f"{self.state_obs_input_dim} vs {len(self.state_feature_indices)}."
            )
        if len(self.ref_cur_feature_indices) != self.ref_cur_token_dim:
            raise ValueError(
                "ref_cur_token_dim must match len(ref_cur_feature_indices): "
                f"{self.ref_cur_token_dim} vs {len(self.ref_cur_feature_indices)}."
            )
        fut_flat_dim = 0
        for start, end, dim in self.ref_fut_slices:
            if end <= start or dim <= 0:
                raise ValueError(
                    f"Invalid ref_fut_slices entry {(start, end, dim)}."
                )
            if (end - start) != self.ref_fut_seq_len * dim:
                raise ValueError(
                    "Future ref slice span must equal ref_fut_seq_len * dim, got "
                    f"{(start, end, dim)} with ref_fut_seq_len={self.ref_fut_seq_len}."
                )
            fut_flat_dim += end - start
        expected_full_input_dim = (
            self.state_obs_input_dim + self.ref_cur_token_dim + fut_flat_dim
        )
        if self.full_obs_input_dim != expected_full_input_dim:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 expected full "
                f"input dim {expected_full_input_dim}, got {self.full_obs_input_dim}."
            )

        self.ref_hist_n_layers = int(module_config.get("ref_hist_n_layers", 1))
        if self.ref_hist_n_layers != 1:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV3 currently supports "
                "exactly one ref history attention layer."
            )

        layer_proj_hidden_default = int(
            module_config.get(
                "router_layer_proj_hidden_dim",
                module_config.get("d_model", 256),
            )
        )
        self.ref_motion_input_dim = int(
            self.ref_cur_token_dim
            + self.ref_fut_seq_len * self.ref_fut_token_dim
        )
        self.router_layer_proj_hidden_dim = int(layer_proj_hidden_default)
        if self.router_layer_proj_hidden_dim <= 0:
            raise ValueError(
                "router_layer_proj_hidden_dim must be positive, got "
                f"{self.router_layer_proj_hidden_dim}."
            )

        GroupedMoETransformerPolicy.__init__(
            self,
            input_dim=input_dim,
            output_dim=output_dim,
            module_config_dict=module_config,
        )
        self.onnx_kv_layers = int(self.ref_hist_n_layers + self.n_layers)
        self.register_buffer(
            "_state_feature_indices",
            torch.tensor(self.state_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_ref_cur_feature_indices",
            torch.tensor(self.ref_cur_feature_indices, dtype=torch.long),
            persistent=False,
        )

        self.ref_frame_embed = nn.Sequential(
            nn.Linear(self.ref_motion_input_dim, self.obs_embed_mlp_hidden),
            nn.SiLU(),
            nn.Linear(self.obs_embed_mlp_hidden, self.d_model),
        )
        self.ref_hist_norm = RMSNorm(self.d_model)
        self.ref_hist_attn = ModernAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            use_qk_norm=self.use_qk_norm,
            use_gated_attn=self.use_gated_attn,
            gated_attn_type=self.gated_attn_type,
            attn_dropout=self.attn_dropout,
        )
        self.ref_hist_out_norm = RMSNorm(self.d_model)

        self._moe_layer_indices = tuple(
            i
            for i, layer in enumerate(self.layers)
            if isinstance(layer, GroupedMoEBlock)
        )
        self.router_layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    RMSNorm(self.d_model),
                    nn.Linear(self.d_model, self.router_layer_proj_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.router_layer_proj_hidden_dim, self.d_model),
                )
                for _ in self._moe_layer_indices
            ]
        )

        self._ref_hist_k_cache: torch.Tensor | None = None
        self._ref_hist_v_cache: torch.Tensor | None = None
        self._apply_freeze_router_state()

    def _apply_freeze_router_state(self) -> None:
        GroupedMoETransformerPolicy._apply_freeze_router_state(self)
        requires_grad = not self.freeze_router
        self.ref_frame_embed.requires_grad_(requires_grad)
        self.ref_hist_norm.requires_grad_(requires_grad)
        self.ref_hist_attn.requires_grad_(requires_grad)
        self.ref_hist_out_norm.requires_grad_(requires_grad)
        self.router_layer_projections.requires_grad_(requires_grad)

    def _build_router_ref_motion(
        self,
        ref_cur_x: torch.Tensor,
        ref_fut_x: torch.Tensor,
    ) -> torch.Tensor:
        if ref_cur_x.ndim not in (2, 3):
            raise ValueError(
                f"Expected ref_cur_x with ndim 2 or 3, got {ref_cur_x.ndim}."
            )
        if ref_fut_x.ndim != ref_cur_x.ndim + 1:
            raise ValueError(
                "Expected ref_fut_x to add one future-seq axis relative to "
                f"ref_cur_x, got cur={tuple(ref_cur_x.shape)}, "
                f"fut={tuple(ref_fut_x.shape)}."
            )
        ref_fut_flat = torch.flatten(ref_fut_x, start_dim=-2)
        return torch.cat([ref_cur_x, ref_fut_flat], dim=-1)

    def _build_shared_router_summary(
        self,
        ref_hist_h: torch.Tensor,
    ) -> torch.Tensor:
        with self._router_no_grad_context():
            return ref_hist_h

    def _build_router_h_per_layer(
        self,
        shared_router_summary: torch.Tensor,
    ) -> list[torch.Tensor | None]:
        with self._router_no_grad_context():
            router_h_per_layer: list[torch.Tensor | None] = [
                None for _ in self.layers
            ]
            for proj, layer_idx in zip(
                self.router_layer_projections, self._moe_layer_indices
            ):
                router_h_per_layer[layer_idx] = proj(shared_router_summary)
            return router_h_per_layer

    def _build_ref_hist_hidden(
        self,
        ref_motion_x: torch.Tensor,
        pos: torch.Tensor,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        with self._router_no_grad_context():
            ref_motion_h = self.ref_frame_embed(ref_motion_x)
            ref_hist_attn = self.ref_hist_attn(
                self.ref_hist_norm(ref_motion_h),
                *self.get_cos_sin(ref_motion_h, pos),
                mask=tgt_mask,
            )
            return self.ref_hist_out_norm(ref_motion_h + ref_hist_attn)

    def _build_ref_hist_hidden_single_step(
        self,
        ref_motion_x: torch.Tensor,
        pos_ids: torch.Tensor,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        new_len: torch.Tensor,
        insert_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._router_no_grad_context():
            ref_motion_h = self.ref_frame_embed(ref_motion_x)[:, None, :]
            ref_cos, ref_sin = self.get_cos_sin(ref_motion_h, pos_ids)
            ref_hist_attn, ref_k_cache, ref_v_cache = (
                self.ref_hist_attn.forward_single_token(
                    self.ref_hist_norm(ref_motion_h),
                    ref_cos,
                    ref_sin,
                    k_cache,
                    v_cache,
                    new_len,
                    insert_pos,
                )
            )
            ref_hist_h = self.ref_hist_out_norm(ref_motion_h + ref_hist_attn)
        return ref_hist_h, ref_k_cache, ref_v_cache

    def predict_aux_from_pre_moe(
        self,
        pre_moe_hidden: torch.Tensor,
        *,
        ref_aux_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return GroupedMoETransformerPolicy.predict_aux_from_pre_moe(
            self,
            pre_moe_hidden,
            ref_aux_hidden=ref_aux_hidden,
        )

    def sequence_mu(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        return_pre_moe_hidden: bool = False,
        return_ref_aux_hidden: bool = False,
        return_router_features: bool = False,
        return_router_temporal_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        batch, time, _ = x.shape
        h = self.obs_embed(x)
        _, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        ref_motion_x = self._build_router_ref_motion(ref_cur_x, ref_fut_x)

        if attn_mask is not None:
            tgt_mask = attn_mask.unsqueeze(1)
            start_idx = attn_mask.to(torch.int64).argmax(dim=-1)
            t_idx = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(batch, time)
            pos = t_idx - start_idx
        else:
            tgt_mask = None
            pos = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(batch, time)

        ref_hist_h = self._build_ref_hist_hidden(
            ref_motion_x=ref_motion_x,
            pos=pos,
            tgt_mask=tgt_mask,
        )
        shared_router_summary = self._build_shared_router_summary(ref_hist_h)
        router_h_per_layer = self._build_router_h_per_layer(
            shared_router_summary
        )
        cos, sin = self.get_cos_sin(h, pos)
        if return_hidden and return_pre_moe_hidden:
            raise ValueError(
                "return_hidden and return_pre_moe_hidden cannot both be True."
            )
        forward_out = self._forward_layers(
            h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h_per_layer=router_h_per_layer,
            return_pre_moe_hidden=return_pre_moe_hidden,
            return_router_features=return_router_features,
            return_router_temporal_features=return_router_temporal_features,
        )
        extras: list[torch.Tensor] = []
        if isinstance(forward_out, tuple):
            h = forward_out[0]
            extras = list(forward_out[1:])
        else:
            h = forward_out
        h = self.norm_f(h)
        mu = self.action_mu_head(h)
        outputs: list[torch.Tensor] = [mu]
        if return_pre_moe_hidden:
            outputs.append(extras.pop(0))
        if return_ref_aux_hidden:
            outputs.append(shared_router_summary)
        if return_router_features:
            outputs.append(extras.pop(0))
        if return_router_temporal_features:
            outputs.append(extras.pop(0))
        if len(outputs) > 1:
            return tuple(outputs)
        if return_hidden:
            return mu, h
        return mu

    def sequence_hidden(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, time, _ = x.shape
        h = self.obs_embed(x)
        _, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        ref_motion_x = self._build_router_ref_motion(ref_cur_x, ref_fut_x)

        if attn_mask is not None:
            tgt_mask = attn_mask.unsqueeze(1)
            start_idx = attn_mask.to(torch.int64).argmax(dim=-1)
            t_idx = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(batch, time)
            pos = t_idx - start_idx
        else:
            tgt_mask = None
            pos = torch.arange(time, device=x.device, dtype=torch.long)[
                None, :
            ].expand(batch, time)

        ref_hist_h = self._build_ref_hist_hidden(
            ref_motion_x=ref_motion_x,
            pos=pos,
            tgt_mask=tgt_mask,
        )
        shared_router_summary = self._build_shared_router_summary(ref_hist_h)
        router_h_per_layer = self._build_router_h_per_layer(
            shared_router_summary
        )
        cos, sin = self.get_cos_sin(h, pos)
        h = self._forward_layers(
            h,
            cos=cos,
            sin=sin,
            mask=tgt_mask,
            router_h_per_layer=router_h_per_layer,
        )
        h = self.norm_f(h)
        return h

    def forward(
        self,
        input: torch.Tensor,
        past_key_values: torch.Tensor | None = None,
        current_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is not None:
            return self._forward_inference_onnx(
                input, past_key_values, current_pos
            )
        if input.ndim != 2:
            raise ValueError(f"Expected [B, D], got {input.shape}")
        mu_seq = self.sequence_mu(input[:, None, :], attn_mask=None)
        return mu_seq[:, 0, :]

    def single_step_mu(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [B, D], got {x.shape}")
        _, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        ref_motion_x = self._build_router_ref_motion(ref_cur_x, ref_fut_x)
        batch = int(x.shape[0])
        if self._k_cache is None:
            mu_seq = self.sequence_mu(x[:, None, :], attn_mask=None)
            return mu_seq[:, 0, :]

        h = self.obs_embed(x)[:, None, :]
        self._ensure_internal_cache_device(x.device, dtype=h.dtype)

        cache_len = self._kv_cache_len
        insert_pos = self._kv_cache_write_idx
        max_len = int(self.max_ctx_len)
        new_len = torch.clamp(cache_len + 1, max=max_len)

        self._kv_cache_len = new_len
        self._kv_cache_write_idx = (insert_pos + 1) % max_len

        pos = self._kv_cache_abs_pos
        self._kv_cache_abs_pos = pos + 1
        pos_ids = pos.unsqueeze(1)

        ref_hist_h, _, _ = self._build_ref_hist_hidden_single_step(
            ref_motion_x=ref_motion_x,
            pos_ids=pos_ids,
            k_cache=self._ref_hist_k_cache[:, 0],
            v_cache=self._ref_hist_v_cache[:, 0],
            new_len=new_len,
            insert_pos=insert_pos,
        )
        shared_router_summary = self._build_shared_router_summary(ref_hist_h)
        router_h_per_layer = self._build_router_h_per_layer(
            shared_router_summary
        )
        cos, sin = self.get_cos_sin(h, pos_ids)

        for layer_idx, layer in enumerate(self.layers):
            x_norm = layer.norm1(h)
            k_cache_l = self._k_cache[:, layer_idx]
            v_cache_l = self._v_cache[:, layer_idx]
            attn_out, _, _ = layer.attn.forward_single_token(
                x_norm,
                cos,
                sin,
                k_cache_l,
                v_cache_l,
                new_len,
                insert_pos,
            )
            h = h + attn_out
            h2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                ffn = layer.compute_moe_ffn(
                    h2, router_x=router_h_per_layer[layer_idx]
                )
                if (
                    layer_idx == self._last_moe_layer_idx
                    and layer.collect_routing_stats
                    and layer.last_router_distribution is not None
                ):
                    self._accumulate_last_moe_router_shift(
                        layer.last_router_distribution
                    )
            else:
                ffn = layer.mlp_dropout(layer.mlp(h2))
            h = h + ffn

        h = self.norm_f(h)
        return self.action_mu_head(h[:, 0, :]).reshape(batch, -1)

    def _forward_inference_onnx(
        self,
        x: torch.Tensor,
        past_key_values: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        _, ref_cur_x, ref_fut_x = self._split_actor_ref_inputs(x)
        ref_motion_x = self._build_router_ref_motion(ref_cur_x, ref_fut_x)
        h = self.obs_embed(x)[:, None, :]
        batch = h.shape[0]
        max_len = past_key_values.shape[3]

        if current_pos.ndim == 0:
            current_pos = current_pos.view(1).expand(batch)

        insert_pos = current_pos % max_len
        new_len = torch.clamp(current_pos + 1, max=max_len)
        position_ids = current_pos.unsqueeze(1)

        ref_layer_past = past_key_values[0]
        ref_hist_h, ref_k_cache, ref_v_cache = (
            self._build_ref_hist_hidden_single_step(
                ref_motion_x=ref_motion_x,
                pos_ids=position_ids,
                k_cache=ref_layer_past[0],
                v_cache=ref_layer_past[1],
                new_len=new_len,
                insert_pos=insert_pos,
            )
        )
        shared_router_summary = self._build_shared_router_summary(ref_hist_h)
        router_h_per_layer = self._build_router_h_per_layer(
            shared_router_summary
        )
        cos, sin = self.get_cos_sin(h, position_ids)

        present_key_values_list = [
            torch.stack([ref_k_cache, ref_v_cache], dim=0)
        ]
        routing_debug_outputs: list[torch.Tensor] = []
        export_routing_debug = torch.onnx.is_in_onnx_export()

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[self.ref_hist_n_layers + i]
            k_cache = layer_past[0]
            v_cache = layer_past[1]

            h_norm = layer.norm1(h)
            attn_out, new_k_cache, new_v_cache = (
                layer.attn.forward_single_token(
                    x=h_norm,
                    cos=cos,
                    sin=sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    new_len=new_len,
                    insert_pos=insert_pos,
                )
            )
            h = h + attn_out

            h_norm2 = layer.norm2(h)
            if isinstance(layer, GroupedMoEBlock):
                if export_routing_debug:
                    ffn_out, topk_idx, router_logits = layer.compute_moe_ffn(
                        h_norm2,
                        router_x=router_h_per_layer[i],
                        return_routing_debug=True,
                    )
                    routing_debug_outputs.extend([topk_idx, router_logits])
                else:
                    ffn_out = layer.compute_moe_ffn(
                        h_norm2, router_x=router_h_per_layer[i]
                    )
            else:
                ffn_out = layer.mlp_dropout(layer.mlp(h_norm2))
            h = h + ffn_out
            current_layer_kv = torch.stack([new_k_cache, new_v_cache], dim=0)
            present_key_values_list.append(current_layer_kv)

        h = self.norm_f(h)
        action = self.action_mu_head(h[:, 0, :])
        present_key_values = torch.stack(present_key_values_list, dim=0)

        if export_routing_debug and routing_debug_outputs:
            return (action, present_key_values, *routing_debug_outputs)
        return action, present_key_values


class GroupedMoEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_fine_experts: int,
        num_shared_experts: int,
        top_k: int,
        n_kv_heads: int | None = None,
        ff_mult: float = 2,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        gated_attn_type: str = "headwise",
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        use_dynamic_bias: bool = False,
        bias_update_rate: float = 0.001,
        routing_score_fn: str = "softmax",
        freeze_router: bool = False,
        routing_scale: float = 1.0,
        expert_bias_clip: float = 0.0,
        moe_load_balance_enabled: bool = False,
        inactive_expert_margin_to_topk_enabled: bool = False,
        inactive_expert_margin_to_topk_ratio_floor: float = 0.0,
        selected_expert_margin_to_unselected_enabled: bool = False,
        selected_expert_margin_to_unselected_target: float = 0.0,
        routed_expert_usage_ema_decay: float = 0.99,
        routed_expert_usage_ema_dead_threshold: float = 1.0e-6,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_fine_experts = num_fine_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.use_dynamic_bias = use_dynamic_bias
        self.bias_update_rate = bias_update_rate
        self.routing_score_fn = str(routing_score_fn).lower()
        self.freeze_router = bool(freeze_router)
        self.routing_scale = float(routing_scale)
        self.expert_bias_clip = float(expert_bias_clip)
        self.moe_load_balance_enabled = bool(moe_load_balance_enabled)
        self.inactive_expert_margin_to_topk_enabled = bool(
            inactive_expert_margin_to_topk_enabled
        )
        self.inactive_expert_margin_to_topk_ratio_floor = float(
            inactive_expert_margin_to_topk_ratio_floor
        )
        self.selected_expert_margin_to_unselected_enabled = bool(
            selected_expert_margin_to_unselected_enabled
        )
        self.selected_expert_margin_to_unselected_target = float(
            selected_expert_margin_to_unselected_target
        )
        self.routed_expert_usage_ema_decay = float(
            routed_expert_usage_ema_decay
        )
        self.routed_expert_usage_ema_dead_threshold = float(
            routed_expert_usage_ema_dead_threshold
        )
        if self.routing_score_fn not in ("softmax", "sigmoid"):
            raise ValueError(
                f"routing_score_fn must be one of {{'softmax','sigmoid'}}, got {self.routing_score_fn}"
            )
        if self.routing_scale <= 0.0:
            raise ValueError(
                f"routing_scale must be > 0, got {self.routing_scale}"
            )
        if self.expert_bias_clip < 0.0:
            raise ValueError(
                f"expert_bias_clip must be >= 0, got {self.expert_bias_clip}"
            )
        if self.selected_expert_margin_to_unselected_target < 0.0:
            raise ValueError(
                "selected_expert_margin_to_unselected_target must be >= 0, "
                f"got {self.selected_expert_margin_to_unselected_target}"
            )
        if not (0.0 <= self.inactive_expert_margin_to_topk_ratio_floor <= 1.0):
            raise ValueError(
                "inactive_expert_margin_to_topk_ratio_floor must be in "
                f"[0, 1], got {self.inactive_expert_margin_to_topk_ratio_floor}"
            )
        if not (0.0 <= self.routed_expert_usage_ema_decay < 1.0):
            raise ValueError(
                "routed_expert_usage_ema_decay must be in [0, 1), got "
                f"{self.routed_expert_usage_ema_decay}"
            )
        if self.routed_expert_usage_ema_dead_threshold < 0.0:
            raise ValueError(
                "routed_expert_usage_ema_dead_threshold must be >= 0, got "
                f"{self.routed_expert_usage_ema_dead_threshold}"
            )
        self.register_buffer("expert_bias", torch.zeros(num_fine_experts))
        self.register_buffer(
            "routing_counts_accum",
            torch.zeros(num_fine_experts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "last_routed_expert_usage",
            torch.zeros(num_fine_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "last_routed_active_expert_count",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_routed_max_expert_frac",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_active_expert_ratio", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_max_expert_frac", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_expert_count_cv", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_min_expert_frac", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_dead_expert_ratio", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "ema_routed_expert_usage",
            torch.zeros(num_fine_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ema_routed_expert_usage_initialized",
            torch.tensor(False),
            persistent=False,
        )
        self.register_buffer(
            "last_ema_dead_expert_ratio",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_ema_max_expert_frac",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_dense_expert_usage",
            torch.zeros(num_fine_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "last_inactive_expert_margin_to_topk_loss_value",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_inactive_expert_margin_to_topk_target",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_selected_expert_margin_to_unselected",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_selected_expert_margin_to_unselected_loss_value",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "last_moe_load_balance_loss_value",
            torch.tensor(0.0),
            persistent=False,
        )
        self.collect_routing_stats = False
        self.collect_router_distribution = False
        self.capture_router_distribution = False
        self.capture_router_logits = False
        self.last_router_distribution: torch.Tensor | None = None
        self.last_router_logits: torch.Tensor | None = None
        self.last_inactive_expert_margin_to_topk_loss: torch.Tensor | None = (
            None
        )
        self.last_selected_expert_margin_to_unselected_loss: (
            torch.Tensor | None
        ) = None
        self.last_moe_load_balance_loss: torch.Tensor | None = None
        self.use_cross_attn = bool(use_cross_attn)

        self.norm1 = RMSNorm(d_model)
        self.attn = ModernAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_qk_norm=use_qk_norm,
            use_gated_attn=use_gated_attn,
            gated_attn_type=gated_attn_type,
            attn_dropout=attn_dropout,
        )
        if self.use_cross_attn:
            self.norm_cross = RMSNorm(d_model)
            self.cross_attn = ModernCrossAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                use_qk_norm=use_qk_norm,
                use_gated_attn=use_gated_attn,
                gated_attn_type=gated_attn_type,
                attn_dropout=attn_dropout,
            )
        else:
            self.norm_cross = None
            self.cross_attn = None

        self.norm2 = RMSNorm(d_model)
        self.intermediate_dim = int(d_model * ff_mult)

        self.router = nn.Linear(d_model, num_fine_experts, bias=False)
        self._apply_freeze_router_state()

        # Gate + Up (Combined)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_fine_experts, self.d_model, 2 * self.intermediate_dim
            )
        )
        # Down
        self.down_proj = nn.Parameter(
            torch.empty(num_fine_experts, self.intermediate_dim, self.d_model)
        )

        self.shared_experts = DeepseekV3MLP(
            hidden_size=d_model,
            intermediate_size=int(d_model * ff_mult * num_shared_experts),
        )
        self.mlp_dropout = (
            nn.Dropout(mlp_dropout) if mlp_dropout > 0.0 else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate_up_proj)
        nn.init.xavier_uniform_(self.down_proj)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        gate_up_key = prefix + "gate_up_proj"
        current_gate_up_shape = tuple(self.gate_up_proj.shape)
        legacy_gate_up_shape = (
            self.num_fine_experts,
            2 * self.intermediate_dim,
            self.d_model,
        )

        down_key = prefix + "down_proj"
        current_down_shape = tuple(self.down_proj.shape)
        legacy_down_shape = (
            self.num_fine_experts,
            self.d_model,
            self.intermediate_dim,
        )

        is_legacy_layout = None
        if gate_up_key in state_dict:
            gate_up_shape = tuple(state_dict[gate_up_key].shape)
            gate_up_is_current = gate_up_shape == current_gate_up_shape
            gate_up_is_legacy = gate_up_shape == legacy_gate_up_shape
            if gate_up_is_current and not gate_up_is_legacy:
                is_legacy_layout = False
            elif gate_up_is_legacy and not gate_up_is_current:
                is_legacy_layout = True
        if is_legacy_layout is None and down_key in state_dict:
            down_shape = tuple(state_dict[down_key].shape)
            down_is_current = down_shape == current_down_shape
            down_is_legacy = down_shape == legacy_down_shape
            if down_is_current and not down_is_legacy:
                is_legacy_layout = False
            elif down_is_legacy and not down_is_current:
                is_legacy_layout = True

        if gate_up_key in state_dict:
            gate_up_w = state_dict[gate_up_key]
            gate_up_shape = tuple(gate_up_w.shape)
            gate_up_is_legacy_only = (
                gate_up_shape == legacy_gate_up_shape
                and gate_up_shape != current_gate_up_shape
            )
            gate_up_is_ambiguous = (
                gate_up_shape == legacy_gate_up_shape
                and gate_up_shape == current_gate_up_shape
            )
            if gate_up_is_legacy_only or (
                gate_up_is_ambiguous and is_legacy_layout
            ):
                state_dict[gate_up_key] = gate_up_w.transpose(
                    -2, -1
                ).contiguous()

        if down_key in state_dict:
            down_w = state_dict[down_key]
            down_shape = tuple(down_w.shape)
            down_is_legacy_only = (
                down_shape == legacy_down_shape
                and down_shape != current_down_shape
            )
            down_is_ambiguous = (
                down_shape == legacy_down_shape
                and down_shape == current_down_shape
            )
            if down_is_legacy_only or (down_is_ambiguous and is_legacy_layout):
                state_dict[down_key] = down_w.transpose(-2, -1).contiguous()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._apply_freeze_router_state()

    def _apply_freeze_router_state(self) -> None:
        self.router.requires_grad_(not self.freeze_router)

    def reset_routing_stats(self) -> None:
        self.routing_counts_accum.zero_()
        self.last_router_distribution = None
        self.last_router_logits = None
        self.last_routed_expert_usage.zero_()
        self.last_routed_active_expert_count.zero_()
        self.last_routed_max_expert_frac.zero_()
        self.last_dense_expert_usage.zero_()
        self.last_inactive_expert_margin_to_topk_loss_value.zero_()
        self.last_inactive_expert_margin_to_topk_target.zero_()
        self.last_inactive_expert_margin_to_topk_loss = None
        self.last_selected_expert_margin_to_unselected.zero_()
        self.last_selected_expert_margin_to_unselected_loss_value.zero_()
        self.last_selected_expert_margin_to_unselected_loss = None
        self.last_moe_load_balance_loss_value.zero_()
        self.last_moe_load_balance_loss = None

    def accumulate_routing_stats(self, topk_idx: torch.Tensor) -> None:
        with torch.no_grad():
            counts = torch.bincount(
                topk_idx.reshape(-1), minlength=self.num_fine_experts
            )
            self.routing_counts_accum.add_(counts)

    def _apply_bias_update_from_counts(self, counts: torch.Tensor) -> None:
        with torch.no_grad():
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            total = counts.sum()
            if int(total.item()) == 0:
                self.last_active_expert_ratio.zero_()
                self.last_max_expert_frac.zero_()
                self.last_expert_count_cv.zero_()
                self.last_min_expert_frac.zero_()
                self.last_dead_expert_ratio.zero_()
                return
            if self.use_dynamic_bias:
                avg = counts.float().mean()
                error = avg - counts.float()
                self.expert_bias.add_(
                    self.bias_update_rate * torch.sign(error)
                )
            total = total.clamp_min(1)
            active_ratio = (counts > 0).to(torch.float32).mean()
            max_expert_frac = counts.max().to(torch.float32) / total.to(
                torch.float32
            )
            min_expert_frac = counts.min().to(torch.float32) / total.to(
                torch.float32
            )
            dead_expert_ratio = (counts == 0).to(torch.float32).mean()
            counts_f = counts.to(torch.float32)
            counts_mean = counts_f.mean().clamp_min(1.0e-6)
            counts_std = counts_f.std(unbiased=False)
            expert_count_cv = counts_std / counts_mean
            self.last_active_expert_ratio.copy_(active_ratio)
            self.last_max_expert_frac.copy_(max_expert_frac)
            self.last_expert_count_cv.copy_(expert_count_cv)
            self.last_min_expert_frac.copy_(min_expert_frac)
            self.last_dead_expert_ratio.copy_(dead_expert_ratio)
            usage = counts.to(torch.float32) / total.to(torch.float32)
            if bool(self.ema_routed_expert_usage_initialized.item()):
                self.ema_routed_expert_usage.mul_(
                    self.routed_expert_usage_ema_decay
                ).add_(
                    usage,
                    alpha=1.0 - self.routed_expert_usage_ema_decay,
                )
            else:
                self.ema_routed_expert_usage.copy_(usage)
                self.ema_routed_expert_usage_initialized.fill_(True)
            ema_usage = self.ema_routed_expert_usage
            ema_dead_ratio = (
                ema_usage <= self.routed_expert_usage_ema_dead_threshold
            ).to(torch.float32).mean()
            self.last_ema_dead_expert_ratio.copy_(ema_dead_ratio)
            self.last_ema_max_expert_frac.copy_(ema_usage.max())
            if self.use_dynamic_bias and self.expert_bias_clip > 0.0:
                self.expert_bias.clamp_(
                    min=-self.expert_bias_clip, max=self.expert_bias_clip
                )

    def apply_bias_update_from_counts(self) -> None:
        with torch.no_grad():
            counts = self.routing_counts_accum.clone()
            self.routing_counts_accum.zero_()
        self._apply_bias_update_from_counts(counts)

    def _update_routed_expert_stats_and_floor_loss(
        self,
        topk_idx: torch.Tensor,
        dense_distribution: torch.Tensor,
        choice_scores: torch.Tensor,
    ) -> torch.Tensor:
        counts = torch.bincount(
            topk_idx.reshape(-1), minlength=self.num_fine_experts
        ).to(torch.float32)
        total_assignments = max(int(topk_idx.numel()), 1)
        hard_usage = counts / float(total_assignments)
        active_count = (counts > 0).to(torch.float32).sum()
        max_frac = hard_usage.max() if hard_usage.numel() > 0 else counts.sum()
        with torch.no_grad():
            self.last_routed_expert_usage.copy_(
                hard_usage.to(self.last_routed_expert_usage.dtype)
            )
            self.last_routed_active_expert_count.copy_(
                active_count.to(self.last_routed_active_expert_count.dtype)
            )
            self.last_routed_max_expert_frac.copy_(
                max_frac.to(self.last_routed_max_expert_frac.dtype)
            )
        dense_usage = dense_distribution.to(torch.float32).mean(dim=(0, 1))
        with torch.no_grad():
            self.last_dense_expert_usage.copy_(
                dense_usage.detach().to(self.last_dense_expert_usage.dtype)
            )
        if self.moe_load_balance_enabled:
            load_balance_loss = (
                self.num_fine_experts
                * (hard_usage.detach() * dense_usage).sum()
            )
        else:
            load_balance_loss = dense_distribution.new_zeros(())
        with torch.no_grad():
            self.last_moe_load_balance_loss_value.copy_(
                load_balance_loss.detach().to(
                    self.last_moe_load_balance_loss_value.dtype
                )
            )
        self.last_moe_load_balance_loss = load_balance_loss
        kth_choice_score = choice_scores.gather(-1, topk_idx)[..., -1:]
        if self.top_k < self.num_fine_experts:
            selected_mask = F.one_hot(
                topk_idx, num_classes=self.num_fine_experts
            ).any(dim=-2)
            best_unselected_score = (
                choice_scores.masked_fill(
                    selected_mask, torch.finfo(choice_scores.dtype).min
                )
                .max(dim=-1, keepdim=True)
                .values
            )
            selected_margin_gap = kth_choice_score - best_unselected_score
            selected_margin = selected_margin_gap.mean()
        else:
            selected_margin_gap = choice_scores.new_zeros(
                choice_scores.shape[:2] + (1,)
            )
            selected_margin = choice_scores.new_zeros(())
        if self.selected_expert_margin_to_unselected_enabled:
            selected_margin_loss = torch.relu(
                self.selected_expert_margin_to_unselected_target
                - selected_margin_gap
            ).mean()
        else:
            selected_margin_loss = choice_scores.new_zeros(())
        with torch.no_grad():
            self.last_selected_expert_margin_to_unselected.copy_(
                selected_margin.detach().to(
                    self.last_selected_expert_margin_to_unselected.dtype
                )
            )
            self.last_selected_expert_margin_to_unselected_loss_value.copy_(
                selected_margin_loss.detach().to(
                    self.last_selected_expert_margin_to_unselected_loss_value.dtype
                )
            )
        self.last_selected_expert_margin_to_unselected_loss = (
            selected_margin_loss
        )

        if not self.inactive_expert_margin_to_topk_enabled:
            margin_loss = dense_distribution.new_zeros(())
            with torch.no_grad():
                self.last_inactive_expert_margin_to_topk_loss_value.zero_()
                self.last_inactive_expert_margin_to_topk_target.zero_()
            self.last_inactive_expert_margin_to_topk_loss = margin_loss
        else:
            inactive_usage_target = (
                hard_usage.max().clamp_min(1.0e-12)
                * self.inactive_expert_margin_to_topk_ratio_floor
            )
            inactive_mask = (hard_usage < inactive_usage_target).to(
                choice_scores.dtype
            )
            margin_gap = torch.relu(kth_choice_score - choice_scores)
            inactive_margin_sum = (
                margin_gap * inactive_mask.view(1, 1, self.num_fine_experts)
            ).sum()
            inactive_count = inactive_mask.sum()
            num_tokens = choice_scores.new_ones(
                choice_scores.shape[:2], dtype=choice_scores.dtype
            ).sum()
            normalizer = inactive_count.clamp_min(1.0) * num_tokens
            margin_loss = inactive_margin_sum / normalizer
            with torch.no_grad():
                self.last_inactive_expert_margin_to_topk_loss_value.copy_(
                    margin_loss.detach().to(
                        self.last_inactive_expert_margin_to_topk_loss_value.dtype
                    )
                )
                self.last_inactive_expert_margin_to_topk_target.copy_(
                    kth_choice_score.mean()
                    .detach()
                    .to(self.last_inactive_expert_margin_to_topk_target.dtype)
                )
            self.last_inactive_expert_margin_to_topk_loss = margin_loss

        return margin_loss

    @torch.compiler.disable
    def _compute_sparse_experts(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_scores: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x.size()
        num_top_k = self.top_k

        is_exporting = torch.onnx.is_in_onnx_export()
        if is_exporting:
            # ONNX/runtime path: compute only selected experts (top_k),
            # avoiding O(num_experts) per-step overhead at bs=1.
            return self._compute_with_topk_selection(x, topk_idx, topk_scores)

        x_flat = x.view(-1, D)
        expert_ids = topk_idx.view(-1)
        scores = topk_scores.view(-1)

        raw_token_indices = (
            torch.arange(B * T, device=x.device)
            .unsqueeze(1)
            .expand(-1, num_top_k)
            .reshape(-1)
        )
        sorted_expert_ids, perm = torch.sort(expert_ids)
        sorted_token_indices = raw_token_indices[perm]
        x_sorted = x_flat[sorted_token_indices]
        scores_sorted = scores[perm]

        # Path B: High-Performance Grouped GEMM
        output_sorted = self._compute_with_grouped_mm(
            x_sorted, sorted_expert_ids
        )

        output_sorted = output_sorted * scores_sorted.unsqueeze(-1)
        inv_perm = torch.argsort(perm)
        output_flat = output_sorted[inv_perm]
        output_final = output_flat.view(B * T, num_top_k, D).sum(dim=1)

        return output_final.view(B, T, D)

    def _compute_with_grouped_mm(
        self, x_sorted: torch.Tensor, sorted_expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Based on official implementation logic:
        - offsets must be Cumsum (End-Indices).
        - offsets length must be exactly Num_Experts (NOT N+1).
        - dtype must be int32.
        """
        tokens_per_expert = torch.bincount(
            sorted_expert_ids.long(), minlength=self.num_fine_experts
        )
        counts = tokens_per_expert[: self.num_fine_experts]

        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

        gate_up_out = _grouped_linear(
            x_sorted, self.gate_up_proj, offs=offsets
        )

        x1, x2 = gate_up_out.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2

        out = _grouped_linear(hidden, self.down_proj, offs=offsets)

        return out

    def _compute_with_topk_selection(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_scores: torch.Tensor,
    ) -> torch.Tensor:
        """ONNX-friendly sparse expert compute that scales with top_k, not
        num_fine_experts.
        """
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        orig_dtype = x.dtype

        x_tokens = x.reshape(N, D)
        idx = topk_idx.reshape(N, K)
        scores = topk_scores.reshape(N, K)

        x_rep = x_tokens[:, None, :].expand(N, K, D).reshape(N * K, D)
        idx_flat = idx.reshape(N * K)
        compute_dtype = self.gate_up_proj.dtype
        if x_rep.dtype != compute_dtype:
            x_rep = x_rep.to(compute_dtype)

        gate_up_w = self.gate_up_proj.index_select(0, idx_flat)
        gate_up_out = torch.bmm(x_rep.unsqueeze(1), gate_up_w).squeeze(1)

        x1, x2 = gate_up_out.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2

        down_w = self.down_proj.index_select(0, idx_flat)
        sparse_flat = torch.bmm(hidden.unsqueeze(1), down_w).squeeze(1)
        if sparse_flat.dtype != orig_dtype:
            sparse_flat = sparse_flat.to(orig_dtype)

        sparse = sparse_flat.view(N, K, D)
        weighted = sparse * scores.to(sparse.dtype).unsqueeze(-1)
        out = weighted.sum(dim=1)
        return out.view(B, T, D)

    def _compute_with_loop_fallback(
        self, x_sorted: torch.Tensor, sorted_expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Path A: Loop Fallback (Compatible with F.linear and 3D Weights)"""
        results = []
        for i in range(self.num_fine_experts):
            mask = sorted_expert_ids == i
            inp_i = x_sorted[mask]

            # Gate + Up
            w_gate_up = self.gate_up_proj[i].transpose(0, 1)
            gate_up_out = F.linear(inp_i, w_gate_up)

            x1, x2 = gate_up_out.chunk(2, dim=-1)
            hidden = F.silu(x1) * x2

            # Down
            w_down = self.down_proj[i].transpose(0, 1)
            out_i = F.linear(hidden, w_down)
            results.append(out_i)

        return torch.cat(results, dim=0)

    def compute_moe_ffn(
        self,
        x: torch.Tensor,
        router_x: torch.Tensor | None = None,
        *,
        return_routing_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        should_cache_router_distribution = (
            self.collect_routing_stats and self.collect_router_distribution
        ) or self.capture_router_distribution
        should_cache_router_logits = self.capture_router_logits
        # 1. Shared Experts (Dense Path)
        shared_out = self.shared_experts(x)

        # 2. Router (Gating)
        router_input = x if router_x is None else router_x
        if router_input.shape != x.shape:
            raise ValueError(
                "router_x shape must match x shape in compute_moe_ffn: "
                f"x={tuple(x.shape)}, router_x={tuple(router_input.shape)}"
            )
        if self.freeze_router:
            with torch.no_grad():
                logits = self.router(router_input)
        else:
            logits = self.router(router_input)
        logits_fp32 = logits.to(torch.float32)
        bias_fp32 = None
        if self.use_dynamic_bias:
            bias_fp32 = self.expert_bias.to(
                device=logits.device, dtype=torch.float32
            )

        if self.routing_score_fn == "softmax":
            choice_logits = logits_fp32
            if bias_fp32 is not None:
                # Keep dynamic bias as a selection correction, not a mixture-weight shaper.
                choice_logits = choice_logits + bias_fp32
            choice_scores = choice_logits
            _, topk_idx = torch.topk(choice_scores, self.top_k, dim=-1)
            dense_distribution = torch.softmax(logits_fp32, dim=-1)
            if torch.onnx.is_in_onnx_export():
                selected_probs = dense_distribution.gather(-1, topk_idx)
            else:
                selected_logits = logits_fp32.gather(-1, topk_idx)
                log_z = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
                selected_probs = torch.exp(selected_logits - log_z)
            topk_scores = selected_probs / selected_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-20)
            router_distribution = None
            if should_cache_router_distribution:
                router_distribution = dense_distribution
        else:  # sigmoid
            scores = torch.sigmoid(logits_fp32)
            dense_distribution = scores / scores.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-20)
            scores_for_choice = scores
            if bias_fp32 is not None:
                # DeepSeek-style correction bias for expert choice.
                scores_for_choice = scores_for_choice + bias_fp32
            choice_scores = scores_for_choice
            _, topk_idx = torch.topk(choice_scores, self.top_k, dim=-1)
            selected_scores = scores.gather(-1, topk_idx)
            # Match DeepSeek-style routing: bias affects only expert choice,
            # while the expert mixing weights come from the original sigmoid
            # affinities normalized over the selected experts.
            topk_scores = selected_scores / selected_scores.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-20)
            router_distribution = None
            if should_cache_router_distribution:
                router_distribution = dense_distribution

        if self.collect_routing_stats:
            self.accumulate_routing_stats(topk_idx)
        if (
            should_cache_router_distribution
            and router_distribution is not None
        ):
            self.last_router_distribution = router_distribution
        else:
            self.last_router_distribution = None
        if should_cache_router_logits:
            self.last_router_logits = logits_fp32
        else:
            self.last_router_logits = None
        self._update_routed_expert_stats_and_floor_loss(
            topk_idx=topk_idx,
            dense_distribution=dense_distribution,
            choice_scores=choice_scores,
        )
        if self.routing_scale != 1.0:
            topk_scores = topk_scores * self.routing_scale
        topk_scores = topk_scores.to(logits.dtype)

        # 3. Sparse Experts Computation (Grouped MM / ONNX Loop)
        sparse_out = self._compute_sparse_experts(x, topk_idx, topk_scores)

        # 4. Combine
        output = shared_out + sparse_out
        output = self.mlp_dropout(output)

        if return_routing_debug:
            return output, topk_idx, logits_fp32
        return output

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor = None,
        sin: torch.Tensor = None,
        mask: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        router_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass compatible with ONNX and Attention/Norm."""
        norm_x = self.norm1(x)
        attn_out = self.attn(norm_x, cos, sin, mask)
        x = x + attn_out
        if self.use_cross_attn and memory is not None:
            x_cross = self.norm_cross(x)
            if memory.ndim == 4:
                b, t, d_model = x_cross.shape
                _, _, n_fut, _ = memory.shape
                q = x_cross.reshape(b * t, 1, d_model)
                mem = memory.reshape(b * t, n_fut, d_model)
                mem_mask = None
                if memory_mask is not None:
                    if memory_mask.ndim != 3:
                        raise ValueError(
                            "memory_mask for 4D memory must have shape [B, T, N_fut]"
                        )
                    mem_mask = memory_mask.reshape(b * t, 1, 1, n_fut)
                cross = self.cross_attn(q, mem, mem_mask).reshape(
                    b, t, d_model
                )
            else:
                cross = self.cross_attn(x_cross, memory, memory_mask)
            x = x + cross

        h = self.norm2(x)
        ffn_out = self.compute_moe_ffn(h, router_x=router_x)
        x = x + ffn_out

        return x


class ModernTransformerBlock(nn.Module):
    """Modern Transformer block with pre-norm, SwiGLU MLP, and modern attention.

    Features:
        - Pre-normalization with RMSNorm.
        - ModernAttention (GQA, QK-Norm, RealRoPE, Gated Attention).
        - DeepseekV3MLP (SwiGLU) for feed-forward.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        ff_mult: int = 4,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        gated_attn_type: str = "headwise",
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.use_cross_attn = bool(use_cross_attn)
        self.norm1 = RMSNorm(d_model)
        self.attn = ModernAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_qk_norm=use_qk_norm,
            use_gated_attn=use_gated_attn,
            gated_attn_type=gated_attn_type,
            attn_dropout=attn_dropout,
        )
        if self.use_cross_attn:
            self.norm_cross = RMSNorm(d_model)
            self.cross_attn = ModernCrossAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                use_qk_norm=use_qk_norm,
                use_gated_attn=use_gated_attn,
                gated_attn_type=gated_attn_type,
                attn_dropout=attn_dropout,
            )
        else:
            self.norm_cross = None
            self.cross_attn = None
        self.norm2 = RMSNorm(d_model)
        self.mlp = DeepseekV3MLP(
            hidden_size=d_model, intermediate_size=d_model * ff_mult
        )
        self.mlp_dropout = (
            nn.Dropout(mlp_dropout) if mlp_dropout > 0.0 else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor [B, T, d_model].
            freqs_cis: RoPE frequencies [T, head_dim // 2].
            mask: Attention mask [T, T] or [B, T, T], True = allowed (can attend).

        Returns:
            out: Output tensor [B, T, d_model].
        """
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        if self.use_cross_attn and memory is not None:
            x_cross = self.norm_cross(x)
            if memory.ndim == 4:
                b, t, d_model = x_cross.shape
                _, _, n_fut, _ = memory.shape
                q = x_cross.reshape(b * t, 1, d_model)
                mem = memory.reshape(b * t, n_fut, d_model)
                mem_mask = None
                if memory_mask is not None:
                    if memory_mask.ndim != 3:
                        raise ValueError(
                            "memory_mask for 4D memory must have shape [B, T, N_fut]"
                        )
                    mem_mask = memory_mask.reshape(b * t, 1, 1, n_fut)
                cross = self.cross_attn(q, mem, mem_mask).reshape(
                    b, t, d_model
                )
            else:
                cross = self.cross_attn(x_cross, memory, memory_mask)
            x = x + cross
        x = x + self.mlp_dropout(self.mlp(self.norm2(x)))
        return x


class DeepseekV3MLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection for efficiency."""

    def __init__(self, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Fused gate and up projection: outputs [gate, up] concatenated
        self.gate_up_proj = nn.Linear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=True,
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_size]
        gate_up = self.gate_up_proj(x)  # [..., 2 * intermediate_size]
        gate, up = gate_up.chunk(2, dim=-1)  # each [..., intermediate_size]
        return self.down_proj(self.act_fn(gate) * up)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Llama, DeepSeek, Qwen)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return x_normed * self.weight


class ModernAttention(nn.Module):
    """Modern attention with GQA, QK-Norm, RealRoPE, Flash Attention, and Gated Attention.

    Features:
        - GQA: Grouped Query Attention (n_kv_heads < n_heads).
        - QK-Norm: RMSNorm on queries and keys for stability.
        - RealRoPE: Real-valued Rotary Positional Embeddings.
        - Flash Attention: via F.scaled_dot_product_attention.
        - Gated Attention: Headwise or element-wise sigmoid gating (Qwen3-style).
        - Fused Projections: Q separate, KV fused for efficiency.

    Reference: https://github.com/qiuzh20/gated_attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        gated_attn_type: str = "headwise",
        attn_dropout: float = 0.0,
    ):
        """Initialize ModernAttention.

        Args:
            d_model: Model dimension.
            n_heads: Number of query heads.
            n_kv_heads: Number of key/value heads (for GQA). Defaults to n_heads.
            use_qk_norm: Apply RMSNorm to Q and K.
            use_gated_attn: Enable gated attention.
            gated_attn_type: "headwise" (Qwen3-style, one gate per head) or
                             "elementwise" (one gate per element).
            attn_dropout: Dropout probability for attention.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_gated_attn = use_gated_attn
        self.gated_attn_type = gated_attn_type
        self.attn_dropout = attn_dropout

        # Fused projections: Q separate, KV fused (for GQA efficiency)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(
            d_model, 2 * self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        if self.use_gated_attn:
            if self.gated_attn_type == "headwise":
                # Qwen3-style: one gate scalar per head [B, T, n_heads]
                self.gate_proj = nn.Linear(d_model, n_heads, bias=False)
            else:
                # Element-wise: gate each element [B, T, d_model]
                self.gate_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,  # [B, T, D]
        sin: torch.Tensor,  # [B, T, D]
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, T, d_model].
            cos: RoPE cosine frequencies [B, T, head_dim].
            sin: RoPE sine frequencies [B, T, head_dim].
            mask: Attention mask [T, T] or [B, T, T], True = allowed (can attend).

        Returns:
            out: Output tensor [B, T, d_model].
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # each [B, T, n_kv_heads, head_dim]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Transpose for SDPA: [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash Attention via SDPA (handles GQA internally)
        dropout_p = self.attn_dropout if self.training else 0.0
        is_exporting = torch.onnx.is_in_onnx_export()

        if is_exporting:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
            enable_gqa = False
        else:
            enable_gqa = True
        attn_out = export_safe_scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=(mask is None),
            enable_gqa=enable_gqa,
        )

        # attn_out: [B, n_heads, T, head_dim]

        # Gated Attention (Qwen3-style)
        if self.use_gated_attn:
            if self.gated_attn_type == "headwise":
                # Headwise gating: [B, T, n_heads] -> [B, n_heads, T, 1]
                g = torch.sigmoid(self.gate_proj(x))  # [B, T, n_heads]
                g = g.transpose(1, 2)[..., None]  # [B, n_heads, T, 1]
                attn_out = attn_out * g
            else:
                # Element-wise gating: apply after reshaping
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
                g = torch.sigmoid(self.gate_proj(x))  # [B, T, d_model]
                attn_out = attn_out * g
                return self.o_proj(attn_out)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)

    def forward_single_token(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        new_len: torch.Tensor,
        insert_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward for single token with per-environment KV cache update.

        Args:
            x: Input tensor [B, 1, d_model].
            cos: RoPE cosine frequencies [B, 1, head_dim].
            sin: RoPE sine frequencies [B, 1, head_dim].
            k_cache: K cache for this layer [B, max_ctx_len, n_kv_heads, head_dim].
            v_cache: V cache for this layer [B, max_ctx_len, n_kv_heads, head_dim].
            new_len: New valid cache length per env AFTER inserting this token [B].
            insert_pos: Insert position per env [B].

        Returns:
            attn_out: [B, 1, d_model]
            k_cache: Updated K cache
            v_cache: Updated V cache
        """
        B = x.shape[0]
        q = self.q_proj(x).view(B, 1, self.n_heads, self.head_dim)
        kv = self.kv_proj(x).view(B, 1, 2, self.n_kv_heads, self.head_dim)
        k_new, v_new = kv[:, :, 0], kv[:, :, 1]  # [B, 1, n_kv_heads, head_dim]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k_new = self.k_norm(k_new)

        # Apply RoPE with per-environment position
        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)
        q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        q = q.transpose(1, 2)
        k_new = k_new.transpose(1, 2)

        # Scatter K, V into cache at per-env insert positions
        # insert_pos: [B] -> [B, 1, 1, 1] for scatter
        idx = (
            insert_pos.view(B, 1, 1, 1)
            .expand(B, 1, self.n_kv_heads, self.head_dim)
            .to(torch.int64)
        )
        if torch.onnx.is_in_onnx_export():
            # === ONNX 模式: Out-of-place (生成新 Tensor) ===
            k_cache = k_cache.scatter(1, idx, k_new.to(k_cache.dtype))
            v_cache = v_cache.scatter(1, idx, v_new.to(v_cache.dtype))
        else:
            # === Rollout 模式: In-place (原地修改) ===
            k_cache.scatter_(1, idx, k_new.to(k_cache.dtype))
            v_cache.scatter_(1, idx, v_new.to(v_cache.dtype))

        # Compute attention over cached keys/values
        # Mask out positions >= new_len (after insert)
        max_len = k_cache.shape[1]
        new_len = new_len.clamp(max=max_len)  # [B]
        # Build per-env mask: [B, max_len] where True = valid (can attend)
        pos_idx = torch.arange(max_len, device=x.device, dtype=torch.int64)
        valid_mask = pos_idx[None, :] < new_len[:, None]  # [B, max_len]
        # For SDPA bool mask: True = allowed (can attend)
        attn_mask = valid_mask[:, None, None, :]  # [B, 1, 1, max_len]

        # GQA: Use native SDPA broadcasting (no repeat_interleave)
        k_attn = k_cache.to(q.dtype)
        v_attn = v_cache.to(q.dtype)

        # Transpose for SDPA: [B, n_heads, T, head_dim]
        q_t = q.transpose(1, 2)  # [B, n_heads, 1, head_dim]
        k_t = k_attn.transpose(1, 2)  # [B, n_kv_heads, max_len, head_dim]
        v_t = v_attn.transpose(1, 2)

        dropout_p = self.attn_dropout if self.training else 0.0
        is_exporting = torch.onnx.is_in_onnx_export()

        if is_exporting:
            k_t = repeat_kv(k_t, self.n_rep)
            v_t = repeat_kv(v_t, self.n_rep)
            enable_gqa = False
        else:
            enable_gqa = True
        attn_out = export_safe_scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
        # attn_out: [B, n_heads, 1, head_dim]

        # Gated Attention
        if self.use_gated_attn:
            if self.gated_attn_type == "headwise":
                g = torch.sigmoid(self.gate_proj(x))  # [B, 1, n_heads]
                g = g.transpose(1, 2)[..., None]  # [B, n_heads, 1, 1]
                attn_out = attn_out * g
            else:
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, -1)
                g = torch.sigmoid(self.gate_proj(x))
                attn_out = attn_out * g
                return self.o_proj(attn_out), k_cache, v_cache

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, -1)
        return self.o_proj(attn_out), k_cache, v_cache


class ModernCrossAttention(nn.Module):
    """Cross-attention with GQA/QK-norm and optional gated attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        gated_attn_type: str = "headwise",
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = (
            int(n_kv_heads) if n_kv_heads is not None else int(n_heads)
        )
        self.head_dim = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_qk_norm = bool(use_qk_norm)
        self.use_gated_attn = bool(use_gated_attn)
        self.gated_attn_type = str(gated_attn_type)
        self.attn_dropout = float(attn_dropout)

        self.q_proj = nn.Linear(
            self.d_model, self.n_heads * self.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(
            self.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.d_model, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        if self.use_gated_attn:
            if self.gated_attn_type == "headwise":
                self.gate_proj = nn.Linear(
                    self.d_model, self.n_heads, bias=False
                )
            else:
                self.gate_proj = nn.Linear(
                    self.d_model, self.d_model, bias=False
                )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")
        if memory.ndim != 3:
            raise ValueError(
                f"memory must be [B, N, D], got {tuple(memory.shape)}"
            )
        b, t, _ = x.shape
        bm, n, _ = memory.shape
        if bm != b:
            raise ValueError(
                f"batch mismatch between x and memory: {b} vs {bm}"
            )

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        kv = self.kv_proj(memory).view(b, n, 2, self.n_kv_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.attn_dropout if self.training else 0.0
        is_exporting = torch.onnx.is_in_onnx_export()
        if is_exporting:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
            enable_gqa = False
        else:
            enable_gqa = True
        attn_out = export_safe_scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False,
            enable_gqa=enable_gqa,
        )

        if self.use_gated_attn:
            if self.gated_attn_type == "headwise":
                g = torch.sigmoid(self.gate_proj(x))
                g = g.transpose(1, 2)[..., None]
                attn_out = attn_out * g
            else:
                attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
                g = torch.sigmoid(self.gate_proj(x))
                attn_out = attn_out * g
                return self.o_proj(attn_out)

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn_out)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Standard LLaMA GQA replication logic, optimized for ONNX export.
    Equivalent to torch.repeat_interleave(x, dim=1, repeats=n_rep).

    Input shape:  (batch, num_key_value_heads, seqlen, head_dim)
    Output shape: (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # 1. Unsqueeze: [batch, n_kv, 1, seq, dim]
    # 2. Expand:    [batch, n_kv, n_rep, seq, dim]
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )

    # 3. Reshape:   [batch, n_kv * n_rep, seq, dim] -> [batch, n_head, seq, dim]
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


def export_safe_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if (
        not torch.onnx.is_in_onnx_export()
        or attn_mask is None
        or attn_mask.dtype != torch.bool
    ):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )

    # Use additive float bias during ONNX export so the legacy exporter
    # does not emit the bool-mask SDPA cleanup path with IsNaN.
    mask_bias = torch.zeros_like(attn_mask, dtype=q.dtype)
    mask_bias = mask_bias.masked_fill(~attn_mask, torch.finfo(q.dtype).min)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    orig_dtype = q.dtype

    # 强制转为 fp32 进行计算
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    cos_fp32 = cos.to(torch.float32)
    sin_fp32 = sin.to(torch.float32)

    q_embed = (q_fp32 * cos_fp32) + (rotate_half(q_fp32) * sin_fp32)
    k_embed = (k_fp32 * cos_fp32) + (rotate_half(k_fp32) * sin_fp32)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def _grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    offs: torch.Tensor | None = None,
) -> torch.Tensor:
    """input: [Total_Tokens, In_Dim]
    weight: [Num_Experts, In_Dim, Out_Dim]
    """
    orig_dtype = input.dtype
    if input.dtype != weight.dtype:
        input = input.to(weight.dtype)
    out = torch._grouped_mm(input, weight, offs=offs)
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    if bias is not None:
        out = out + bias
    return out
