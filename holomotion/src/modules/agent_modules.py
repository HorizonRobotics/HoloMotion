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


from __future__ import annotations
import io
import copy
import math
from pathlib import Path

import holomotion.src.modules.network_modules as NM
import torch
import torch.nn as nn
import torch.nn.functional as F
from holomotion.src.modules.network_modules import EmpiricalNormalization
from loguru import logger
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch.distributions import Normal


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def _clone_module_for_cpu_export(module: nn.Module) -> nn.Module:
    """Clone a module for CPU-side export without mutating the live module."""
    buffer = io.BytesIO()
    # Keep the training module on-device; rank-local device hops during export
    # can desynchronize DDP state and hang later collectives.
    torch.save(module, buffer)
    buffer.seek(0)
    clone = torch.load(buffer, map_location="cpu", weights_only=False)
    clone = clone.to("cpu")
    clone.eval()
    return clone


class TensorDictAssembler(torch.nn.Module):
    def __init__(self, schema_config: dict, *, output_mode: str = "flat"):
        super().__init__()
        self.schema_config = schema_config
        self.output_mode = str(output_mode).lower()
        if self.output_mode not in ("flat", "seq"):
            raise ValueError(
                f"output_mode must be one of {{'flat','seq'}}, got {output_mode}"
            )

        self.seq_len_dict: dict[str, int] = {
            str(k): int(v.get("seq_len", 1)) for k, v in schema_config.items()
        }
        _uniq_lens = sorted(set(self.seq_len_dict.values()))
        self.seq_len: int | None = (
            int(_uniq_lens[0]) if len(_uniq_lens) == 1 else None
        )
        if self.output_mode == "seq" and self.seq_len is None:
            raise ValueError(
                "TensorDictAssembler(output_mode='seq') requires a single unique seq_len "
                f"across schema groups, got seq_len_dict={self.seq_len_dict}"
            )

        self.output_dim: int | None = None

    @staticmethod
    def _get_from_data(data: TensorDict, key: str):
        # Support hierarchical keys like "latent/z"
        if key in data.keys():
            return data.get(key)
        if "/" in key:
            current = data
            for p in key.split("/"):
                if isinstance(current, TensorDict) and p in current.keys():
                    current = current.get(p)
                else:
                    return None
            return current
        return None

    def _validate_to_seq(
        self,
        tensor: torch.Tensor,
        seq_len: int,
        term: str,
    ) -> torch.Tensor:
        """Return [B, seq_len, d] tensor."""
        if tensor.ndim == 2:
            # [B, d] treat as seq_len=1
            if seq_len != 1:
                raise ValueError(
                    f"Term '{term}' expected seq_len={seq_len} but tensor is 2D {tensor.shape}"
                )
            return tensor[:, None, :]
        if tensor.ndim == 3:
            if tensor.shape[1] != seq_len:
                raise ValueError(
                    f"Term '{term}' seq_len mismatch: expected {seq_len}, got {tensor.shape[1]}"
                )
            return tensor
        raise ValueError(
            f"Term '{term}' tensor ndim must be 2 or 3, got {tensor.ndim}"
        )

    def _validate_and_flatten(
        self,
        tensor: torch.Tensor,
        seq_len: int,
        term: str,
    ) -> torch.Tensor:
        if tensor.ndim == 2:
            # [B, D] treat as seq_len=1
            if seq_len != 1:
                raise ValueError(
                    f"Term '{term}' expected seq_len={seq_len} but tensor is 2D {tensor.shape}"
                )
            return tensor
        if tensor.ndim == 3:
            if tensor.shape[1] != seq_len:
                raise ValueError(
                    f"Term '{term}' seq_len mismatch: expected {seq_len}, got {tensor.shape[1]}"
                )
            b, t, d = tensor.shape
            return tensor.reshape(b, t * d)
        raise ValueError(
            f"Term '{term}' tensor ndim must be 2 or 3, got {tensor.ndim}"
        )

    def forward(self, data: TensorDict) -> torch.Tensor:
        if not isinstance(data, TensorDict):
            raise TypeError("TensorDictAssembler expects TensorDict input.")

        if self.output_mode == "flat":
            assembled = []
            output_dim = 0
            batch_size = None

            for _, seq_cfg in self.schema_config.items():
                seq_len = int(seq_cfg.get("seq_len", 1))
                terms = seq_cfg.get("terms", [])
                for term in terms:
                    tensor = self._get_from_data(data, term)
                    if tensor is None:
                        raise KeyError(
                            f"Missing term '{term}' in TensorDict input for assembler. "
                            "Use explicit hierarchical terms (e.g. 'group/term') "
                            "for nested TensorDict keys."
                        )
                    flat = self._validate_and_flatten(tensor, seq_len, term)
                    if batch_size is None:
                        batch_size = flat.shape[0]
                    elif flat.shape[0] != batch_size:
                        raise ValueError(
                            f"Batch size mismatch for term '{term}': {flat.shape[0]} vs {batch_size}"
                        )
                    assembled.append(flat)
                    output_dim += flat.shape[-1]

            if not assembled:
                raise ValueError(
                    "Assembler received an empty schema or no tensors found"
                )

            out = torch.cat(assembled, dim=-1)

            # Cache output_dim on first successful forward
            if self.output_dim is None:
                self.output_dim = output_dim
            return out

        # output_mode == "seq"
        assembled_seq = []
        batch_size = None
        seq_len_ref = None

        for _, seq_cfg in self.schema_config.items():
            seq_len = int(seq_cfg.get("seq_len", 1))
            if seq_len_ref is None:
                seq_len_ref = seq_len
            elif seq_len != seq_len_ref:
                raise ValueError(
                    "TensorDictAssembler(output_mode='seq') requires consistent seq_len "
                    f"across schema groups, got {seq_len_ref} vs {seq_len}"
                )
            terms = seq_cfg.get("terms", [])
            for term in terms:
                tensor = self._get_from_data(data, term)
                if tensor is None:
                    raise KeyError(
                        f"Missing term '{term}' in TensorDict input for assembler. "
                        "Use explicit hierarchical terms (e.g. 'group/term') "
                        "for nested TensorDict keys."
                    )
                seq_tensor = self._validate_to_seq(tensor, seq_len, term)
                if batch_size is None:
                    batch_size = seq_tensor.shape[0]
                elif seq_tensor.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size mismatch for term '{term}': {seq_tensor.shape[0]} vs {batch_size}"
                    )
                assembled_seq.append(seq_tensor)

        if not assembled_seq:
            raise ValueError(
                "Assembler received an empty schema or no tensors found"
            )

        out = torch.cat(assembled_seq, dim=-1)
        # Expose seq_len and output_dim for sequence assembly
        if self.seq_len is None:
            self.seq_len = int(out.shape[1])
        if self.output_dim is None:
            self.output_dim = int(out.shape[-1])
        return out

    @torch.inference_mode()
    def infer_output_dim(self, sample: TensorDict) -> int:
        """Run a dry forward pass to populate output_dim without grads."""
        if self.output_dim is not None:
            return int(self.output_dim)
        _ = self.forward(sample)
        return self.output_dim


class PPOActorOnnxModule(nn.Module):
    def __init__(
        self,
        actor_module: nn.Module,
        obs_normalizer: nn.Module,
        obs_norm_enabled: bool,
        obs_norm_clip: float,
    ):
        super().__init__()
        self.actor_module = actor_module
        self.obs_normalizer = obs_normalizer
        self.obs_norm_enabled = bool(obs_norm_enabled)
        self.obs_norm_clip = float(obs_norm_clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = obs
        if self.obs_norm_enabled:
            actor_obs = self.obs_normalizer.normalize_only(actor_obs)
            if self.obs_norm_clip > 0.0:
                actor_obs = torch.clamp(
                    actor_obs, -self.obs_norm_clip, self.obs_norm_clip
                )
        return self.actor_module(actor_obs)


class PPOTFActorOnnxModule(nn.Module):
    def __init__(
        self,
        actor_module: nn.Module,
        obs_normalizer: nn.Module,
        obs_norm_enabled: bool,
        obs_norm_clip: float,
    ):
        super().__init__()
        self.actor_module = actor_module
        self.obs_normalizer = obs_normalizer
        self.obs_norm_enabled = bool(obs_norm_enabled)
        self.obs_norm_clip = float(obs_norm_clip)

    def forward(
        self,
        obs: torch.Tensor,
        past_key_values: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        actor_obs = obs
        if self.obs_norm_enabled:
            actor_obs = self.obs_normalizer.normalize_only(actor_obs)
            if self.obs_norm_clip > 0.0:
                actor_obs = torch.clamp(
                    actor_obs, -self.obs_norm_clip, self.obs_norm_clip
                )
        return self.actor_module(
            actor_obs,
            past_key_values=past_key_values,
            current_pos=step_idx,
        )


class PPOTFWoKVCacheActorOnnxModule(nn.Module):
    def __init__(
        self,
        actor_module: nn.Module,
        obs_normalizer: nn.Module,
        obs_norm_enabled: bool,
        obs_norm_clip: float,
    ):
        super().__init__()
        self.actor_module = actor_module
        self.obs_normalizer = obs_normalizer
        self.obs_norm_enabled = bool(obs_norm_enabled)
        self.obs_norm_clip = float(obs_norm_clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError(
                f"Expected obs [B, 32, D] for no-kv ONNX path, got {obs.shape}"
            )
        if obs.shape[1] != 32:
            raise ValueError(
                f"Expected fixed token length 32, got {int(obs.shape[1])}"
            )
        actor_obs = obs
        if self.obs_norm_enabled:
            actor_obs = self.obs_normalizer.normalize_only(actor_obs)
            if self.obs_norm_clip > 0.0:
                actor_obs = torch.clamp(
                    actor_obs, -self.obs_norm_clip, self.obs_norm_clip
                )
        action_seq = self.actor_module.sequence_mu(actor_obs, attn_mask=None)
        return action_seq[:, -1, :]


class PPOCondTFActorOnnxModule(nn.Module):
    def __init__(
        self,
        actor_module: nn.Module,
        state_obs_normalizer: nn.Module,
        obs_norm_enabled: bool,
        obs_norm_clip: float,
        state_dim: int,
        future_seq_len: int,
        future_token_dim: int,
        future_term_dims: list[int],
    ):
        super().__init__()
        self.actor_module = actor_module
        self.state_obs_normalizer = state_obs_normalizer
        self.obs_norm_enabled = bool(obs_norm_enabled)
        self.obs_norm_clip = float(obs_norm_clip)
        self.state_dim = int(state_dim)
        self.future_seq_len = int(future_seq_len)
        self.future_token_dim = int(future_token_dim)
        self.future_term_dims = [int(x) for x in future_term_dims]
        if any(d <= 0 for d in self.future_term_dims):
            raise ValueError(
                f"future_term_dims must be all positive, got {self.future_term_dims}"
            )
        if sum(self.future_term_dims) != self.future_token_dim:
            raise ValueError(
                "future_term_dims sum mismatch: expected "
                f"{self.future_token_dim}, got {sum(self.future_term_dims)}"
            )

    def forward(
        self,
        obs: torch.Tensor,
        past_key_values: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if obs.ndim != 2:
            raise ValueError(f"Expected obs [B, D], got {obs.shape}")
        state_obs = obs[:, : self.state_dim]
        future_flat = obs[:, self.state_dim :]
        expected_future_dim = self.future_seq_len * self.future_token_dim
        if future_flat.shape[-1] != expected_future_dim:
            raise ValueError(
                "Future obs dim mismatch for ONNX path: expected "
                f"{expected_future_dim}, got {future_flat.shape[-1]}"
            )
        if self.obs_norm_enabled:
            state_obs = self.state_obs_normalizer.normalize_only(state_obs)
            if self.obs_norm_clip > 0.0:
                state_obs = torch.clamp(
                    state_obs, -self.obs_norm_clip, self.obs_norm_clip
                )
        # Reconstruct [B, N_fut, D_fut] from term-major flattened layout:
        # [term1 (N_fut*d1), term2 (N_fut*d2), ...] -> per-step concat along last dim.
        b = int(obs.shape[0])
        offset = 0
        future_parts = []
        for d_term in self.future_term_dims:
            span = int(self.future_seq_len * d_term)
            chunk = future_flat[:, offset : offset + span]
            future_parts.append(chunk.reshape(b, self.future_seq_len, d_term))
            offset += span
        if offset != int(future_flat.shape[-1]):
            raise ValueError(
                "Future flat slicing mismatch in ONNX path: "
                f"consumed={offset}, total={int(future_flat.shape[-1])}"
            )
        future_obs = torch.cat(future_parts, dim=-1)
        return self.actor_module._forward_inference_onnx_cond(
            state_obs,
            future_obs,
            past_key_values,
            step_idx,
        )


class PPOActor(TensorDictModuleBase):
    def __init__(
        self,
        obs_schema: dict | None,
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
        *,
        obs_example: dict | None = None,
    ):
        super(PPOActor, self).__init__()

        self.use_logvar = module_config_dict.get("use_logvar", False)
        obs_norm_cfg = module_config_dict.get("obs_norm", {})
        self.obs_norm_enabled = bool(obs_norm_cfg.get("enabled", False))
        if self.obs_norm_enabled:
            self.obs_norm_clip = float(obs_norm_cfg.get("clip_range", 0.0))
            self.obs_norm_eps = float(obs_norm_cfg.get("epsilon", 1.0e-8))
            self.obs_norm_update_method = str(
                obs_norm_cfg.get(
                    "update_method", obs_norm_cfg.get("method", "cumulative")
                )
            ).lower()
            self.obs_norm_ema_momentum = float(
                obs_norm_cfg.get("ema_momentum")
            )

        module_config_dict = self._process_module_config(
            module_config_dict,
            num_actions,
        )

        self.actor_net_type = module_config_dict.get("type", "MLP")

        logger.info(f"actor_net_type: {self.actor_net_type}")

        actor_net_class = getattr(NM, self.actor_net_type, None)

        if actor_net_class is NM.MLP and obs_schema is None:
            raise ValueError(
                "PPOActor(Mlp) requires obs_schema so the agent module can assemble"
                "TensorDict observations into a flat tensor."
            )

        if obs_schema is not None:
            output_mode = "seq" if actor_net_class is NM.ConvMLP else "flat"
            self.assembler = TensorDictAssembler(
                obs_schema, output_mode=output_mode
            )
            if obs_example is not None:
                self.assembler.infer_output_dim(obs_example)
            if self.assembler.output_dim is None:
                raise ValueError(
                    "TensorDictAssembler could not infer output_dim"
                )
            input_dim_for_net = int(self.assembler.output_dim)
        else:
            raise ValueError("obs_schema can't be None!")

        actor_in_keys: list[str] = []
        for _, seq_cfg in obs_schema.items():
            if not isinstance(seq_cfg, dict):
                continue
            for term in seq_cfg.get("terms", []):
                actor_in_keys.append(str(term))
        self.in_keys = actor_in_keys
        self.out_keys = [
            "actions",
            "actions_log_prob",
            "mu",
            "sigma",
            "entropy",
        ]
        if self.obs_norm_enabled and self.assembler is not None:
            self.obs_normalizer = EmpiricalNormalization(
                shape=self.assembler.output_dim,
                eps=self.obs_norm_eps,
                update_method=self.obs_norm_update_method,
                ema_momentum=self.obs_norm_ema_momentum,
            )
        else:
            self.obs_normalizer = nn.Identity()

        # Always pass obs_example if available
        if obs_example is not None:
            self.actor_module = actor_net_class(
                input_dim=input_dim_for_net,
                output_dim=int(module_config_dict["output_dim"]),
                module_config_dict=module_config_dict,
            )
        else:
            raise ValueError("Obs example can't be None!")

        if "output_head_init_scale" in module_config_dict:
            output_head_init_scale = float(
                module_config_dict["output_head_init_scale"]
            )
            if output_head_init_scale <= 0.0:
                raise ValueError("output_head_init_scale must be > 0.")
            output_head = self.actor_module.output_head
            if not isinstance(output_head, nn.Linear):
                raise ValueError(
                    "output_head_init_scale requires actor_module.output_head to be nn.Linear."
                )
            with torch.no_grad():
                output_head.weight.mul_(output_head_init_scale)
                if output_head.bias is not None:
                    output_head.bias.mul_(output_head_init_scale)

        self._actor_schema_module = bool(
            getattr(self.actor_module, "proprio_assembler", None)
        )
        self.fix_sigma = module_config_dict.get("fix_sigma", False)
        self.max_sigma = module_config_dict.get("max_sigma", 1.0)
        self.min_sigma = module_config_dict.get("min_sigma", 0.1)

        if "noise_std_type" in module_config_dict:
            self.noise_std_type = str(
                module_config_dict["noise_std_type"]
            ).lower()
        elif self.use_logvar:
            self.noise_std_type = "log"
        else:
            self.noise_std_type = "scalar"

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
        self.actor_obs_transforms: list[callable] = []
        if self.obs_norm_enabled:
            self.actor_obs_transforms.append(self._normalize_actor_obs)

    def _process_module_config(self, module_config_dict, num_actions):
        if module_config_dict.get("output_schema", None) is not None:
            raise ValueError(
                "PPOActor no longer supports module_config_dict.output_schema. "
                "Use scalar module_config_dict.output_dim instead."
            )

        # Resolve output_dim placeholders when present.
        if "output_dim" in module_config_dict:
            output_dim = module_config_dict["output_dim"]
            if isinstance(output_dim, list):
                raise ValueError(
                    "PPOActor expects module_config_dict.output_dim to be a scalar. "
                    "List-valued output_dim is not supported."
                )
            if output_dim == "robot_action_dim":
                module_config_dict["output_dim"] = num_actions

        return module_config_dict

    def _sigma_from_params(self) -> torch.Tensor:
        if self.noise_std_type == "log":
            return torch.exp(self.log_std)
        return self.std

    def _normalize_actor_obs(
        self, obs: torch.Tensor, update: bool
    ) -> torch.Tensor:
        if not self.obs_norm_enabled:
            return obs
        clip = float(self.obs_norm_clip)
        if obs.ndim == 3:
            b, seq_len, d = obs.shape
            flat_obs = obs.reshape(b * seq_len, d)
            if update:
                self.obs_normalizer.update(flat_obs)
            flat_obs = self.obs_normalizer.normalize_only(flat_obs)
            obs = flat_obs.reshape(b, seq_len, d)
        else:
            if update:
                self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize_only(obs)
        if clip > 0.0:
            obs = torch.clamp(obs, -clip, clip)
        return obs

    def _sigma_like(self, like: torch.Tensor) -> torch.Tensor:
        sigma_vec = self._sigma_from_params()
        sigma_vec = torch.clamp(
            sigma_vec,
            min=float(self.min_sigma),
            max=float(self.max_sigma),
        )
        if sigma_vec.ndim == 1 and like.ndim >= 2:
            view_shape = [1 for _ in range(like.ndim - 1)] + [
                sigma_vec.shape[0]
            ]
            return sigma_vec.view(*view_shape).expand_as(like)
        if sigma_vec.shape != like.shape:
            return sigma_vec.expand_as(like)
        return sigma_vec

    @property
    def actor(self):
        return self.actor_module

    @property
    def flat_obs_dim(self) -> int:
        if self.assembler is None:
            raise ValueError(
                "PPOActor has no assembler; flat obs dim unavailable."
            )
        if self.assembler.output_dim is None:
            raise ValueError(
                "PPOActor assembler output_dim is not initialized."
            )
        return int(self.assembler.output_dim)

    def export_onnx(
        self,
        onnx_path: str | Path,
        *,
        opset_version: int = 17,
    ) -> str:
        if self._actor_schema_module:
            raise ValueError(
                "PPOActor export expects flat-obs actor modules, not schema-native modules."
            )
        export_path = Path(onnx_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.actor_module, "clear_router_distribution_cache"):
            self.actor_module.clear_router_distribution_cache()
        actor_module = _clone_module_for_cpu_export(self.actor_module)
        if self.obs_norm_enabled:
            obs_normalizer = _clone_module_for_cpu_export(self.obs_normalizer)
        else:
            obs_normalizer = nn.Identity()

        exporter = PPOActorOnnxModule(
            actor_module=actor_module,
            obs_normalizer=obs_normalizer,
            obs_norm_enabled=self.obs_norm_enabled,
            obs_norm_clip=self.obs_norm_clip if self.obs_norm_enabled else 0.0,
        ).to("cpu")
        exporter.eval()

        obs = torch.zeros(
            1, self.flat_obs_dim, device="cpu", dtype=torch.float32
        )
        torch.onnx.export(
            exporter,
            (obs,),
            str(export_path),
            export_params=True,
            opset_version=opset_version,
            verbose=False,
            dynamo=False,
            input_names=["obs"],
            output_names=["actions"],
        )
        return str(export_path)

    def forward(
        self,
        obs_td: TensorDict,
        actions: torch.Tensor | None = None,
        mode: str = "sampling",
        *,
        update_obs_norm: bool = True,
    ) -> TensorDict:
        """TensorDict-first forward for PPOActor.

        Returns a TensorDict with keys:
        - actions: [B, A]
        - actions_log_prob: [B] (sampling/logp only)
        - mu: [B, A]
        - sigma: [B, A]
        - entropy: [B] (sampling/logp only)
        """
        if mode not in ("sampling", "logp", "inference"):
            raise ValueError(f"Unsupported mode: {mode}")
        if not isinstance(obs_td, TensorDict):
            raise ValueError("PPOActor.forward expects TensorDict input.")

        td = obs_td.clone(
            recurse=False
        )  # this only clones the tree sturcture, not the data

        if self._actor_schema_module:
            mu = self.actor_module(obs_td)
        else:
            if self.assembler is None:
                raise ValueError(
                    "Flat-tensor actor module requires obs_schema in PPOActor init."
                )
            actor_obs = self.assembler(obs_td)
            update = bool(update_obs_norm)
            for fn in self.actor_obs_transforms:
                actor_obs = fn(actor_obs, update)
            mu = self.actor_module(actor_obs)

        sigma = self._sigma_like(mu)
        td.set("mu", mu)
        td.set("sigma", sigma)

        if mode == "inference":
            actions_out = mu
            td.set("actions", actions_out)
            return td

        self.distribution = Normal(mu, sigma)
        if mode == "sampling":
            actions_out = self.distribution.sample()
        else:
            if actions is None:
                raise ValueError("actions must be provided when mode='logp'")
            actions_out = actions

        td.set("actions", actions_out)
        td.set(
            "actions_log_prob",
            self.distribution.log_prob(actions_out).sum(dim=-1),
        )
        td.set("entropy", self.distribution.entropy().sum(dim=-1))
        return td

    def update_distribution(self, actor_obs):
        mean = self.actor(actor_obs)
        # Resolve std according to parameterization
        std_val = self._sigma_from_params()

        std_val = torch.clamp(std_val, min=self.min_sigma, max=self.max_sigma)
        self.distribution = Normal(mean, std_val)

    def override_sigma(self, sigma_override: float | torch.Tensor) -> None:
        """Override actor sigma parameters (std) explicitly.

        Args:
            sigma_override: scalar or [A] tensor for sigma_theta (std).
        """
        if self.noise_std_type not in ("scalar", "log"):
            raise ValueError(
                f"Unsupported noise_std_type for override: {self.noise_std_type}"
            )
        param = self.log_std if self.noise_std_type == "log" else self.std
        sigma_tensor = torch.as_tensor(
            sigma_override, device=param.device, dtype=param.dtype
        )
        if sigma_tensor.numel() == 1:
            sigma_tensor = sigma_tensor.expand_as(param)
        elif sigma_tensor.shape != param.shape:
            raise ValueError(
                f"sigma_override shape {tuple(sigma_tensor.shape)} does not match "
                f"actor sigma shape {tuple(param.shape)}."
            )
        if torch.any(sigma_tensor <= 0):
            raise ValueError("sigma_override must be > 0 for all dims.")
        if self.noise_std_type == "log":
            sigma_tensor = torch.log(sigma_tensor)
        with torch.no_grad():
            param.copy_(sigma_tensor)


class PPOCritic(TensorDictModuleBase):
    def __init__(
        self,
        obs_schema: dict | None,
        module_config_dict,
        *,
        obs_example: dict | None = None,
    ):
        super(PPOCritic, self).__init__()
        self.critic_net_type = module_config_dict.get("type", "MLP")
        obs_norm_cfg = module_config_dict.get("obs_norm", {})
        self.obs_norm_enabled = bool(obs_norm_cfg.get("enabled", False))

        if self.obs_norm_enabled:
            self.obs_norm_clip = float(obs_norm_cfg.get("clip_range", 0.0))
            self.obs_norm_eps = float(obs_norm_cfg.get("epsilon", 1.0e-8))

            self.obs_norm_update_method = str(
                obs_norm_cfg.get(
                    "update_method", obs_norm_cfg.get("method", "cumulative")
                )
            ).lower()
            self.obs_norm_ema_momentum = float(
                obs_norm_cfg.get("ema_momentum")
            )

        critic_net_class = getattr(NM, self.critic_net_type, None)
        if critic_net_class is None:
            critic_net_class = globals().get(self.critic_net_type, None)
        if critic_net_class is None or not isinstance(critic_net_class, type):
            available_classes = [
                name
                for name in dir(NM)
                if isinstance(getattr(NM, name, None), type)
            ] + [
                name
                for name, obj in globals().items()
                if isinstance(obj, type)
            ]
            raise NotImplementedError(
                f"Unknown critic_net_type: {self.critic_net_type}. "
                f"Available classes: {available_classes}"
            )

        if critic_net_class is NM.MLP and obs_schema is None:
            raise ValueError(
                "PPOCritic(MLP) requires obs_schema so the agent module can assemble "
                "TensorDict observations into a flat tensor."
            )

        # Build assembler for flat-tensor networks only
        # Schema-based networks (e.g., MultiTaskCritic) don't need it
        if obs_schema is not None:
            output_mode = "seq" if critic_net_class is NM.ConvMLP else "flat"
            self.assembler = TensorDictAssembler(
                obs_schema, output_mode=output_mode
            )
            if obs_example is not None:
                self.assembler.infer_output_dim(obs_example)
            if self.assembler.output_dim is None:
                raise ValueError(
                    "TensorDictAssembler could not infer output_dim; provide obs_example."
                )
            input_dim_for_net = int(self.assembler.output_dim)
        else:
            # Schema-based modules don't use wrapper's assembler
            self.assembler = None
            input_dim_for_net = 0

        critic_in_keys: list[str] = []
        if obs_schema is not None:
            for _, seq_cfg in obs_schema.items():
                if not isinstance(seq_cfg, dict):
                    continue
                for term in seq_cfg.get("terms", []):
                    critic_in_keys.append(str(term))
        self.in_keys = critic_in_keys
        self.out_keys = ["values"]

        if self.obs_norm_enabled and self.assembler is not None:
            self.obs_normalizer = EmpiricalNormalization(
                shape=self.assembler.output_dim,
                eps=self.obs_norm_eps,
                update_method=self.obs_norm_update_method,
                ema_momentum=self.obs_norm_ema_momentum,
            )
        else:
            self.obs_normalizer = nn.Identity()

        # Always pass obs_example if available
        if obs_example is not None:
            self.critic_module = critic_net_class(
                input_dim=input_dim_for_net,
                output_dim=int(module_config_dict["output_dim"]),
                module_config_dict=module_config_dict,
            )

        else:
            raise ValueError("obs_schema can't be None!")
        self._critic_schema_module = bool(
            getattr(self.critic_module, "proprio_assembler", None)
        )
        self.critic_obs_transforms: list[callable] = []
        if self.obs_norm_enabled:
            self.critic_obs_transforms.append(self._normalize_critic_obs)

    def _normalize_critic_obs(
        self, obs: torch.Tensor, update: bool
    ) -> torch.Tensor:
        if not self.obs_norm_enabled:
            return obs
        clip = float(self.obs_norm_clip)
        if obs.ndim == 3:
            b, seq_len, d = obs.shape
            flat_obs = obs.reshape(b * seq_len, d)
            if update:
                self.obs_normalizer.update(flat_obs)
            flat_obs = self.obs_normalizer.normalize_only(flat_obs)
            obs = flat_obs.reshape(b, seq_len, d)
        else:
            if update:
                self.obs_normalizer.update(obs)
            obs = self.obs_normalizer.normalize_only(obs)
        if clip > 0.0:
            obs = torch.clamp(obs, -clip, clip)
        return obs

    def forward(
        self,
        obs_td: TensorDict,
        update_obs_norm: bool = True,
        **kwargs,
    ) -> TensorDict:
        """TensorDict-first forward for PPOCritic.

        Args:
            obs_td: TensorDict observations keyed by obs terms.
            update_obs_norm: If False, skip updating running stats.

        Returns:
            TensorDict with key:
                - "values": [B, 1]
        """
        if not isinstance(obs_td, TensorDict):
            raise ValueError("PPOCritic.forward expects TensorDict input.")

        td = obs_td.clone(recurse=False)
        if self._critic_schema_module:
            values = self.critic_module(obs_td)
            if values.ndim == 1:
                values = values[..., None]
            td.set("values", values)
            return td

        if self.assembler is None:
            raise ValueError(
                "Flat-tensor critic module requires obs_schema in PPOCritic init."
            )
        critic_obs = self.assembler(obs_td)
        update = bool(update_obs_norm)
        for fn in self.critic_obs_transforms:
            critic_obs = fn(critic_obs, update)
        values = self.critic_module(critic_obs)
        if values.ndim == 1:
            values = values[..., None]
        td.set("values", values)
        return td


class PPOTFActor(PPOActor):
    """Transformer-based PPO actor wrapper compatible with PPOActor interface.

    - Uses NM.TransformerDecoderPolicy as actor_module
    - Provides KV-cache controls
    - Uses model-predicted diagonal std for distribution
    """

    def __init__(
        self,
        obs_schema: dict | None,
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
        *,
        obs_example: dict | None = None,
    ):
        super().__init__(
            obs_schema=obs_schema,
            module_config_dict=module_config_dict,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            obs_example=obs_example,
        )
        # Ensure initial std is strictly inside [min_sigma, max_sigma] to avoid boundary saturation
        init_std_val = float(init_noise_std)
        if not (self.min_sigma < init_std_val < self.max_sigma):
            # Expand bounds conservatively if needed
            if init_std_val >= self.max_sigma:
                self.max_sigma = max(self.max_sigma, init_std_val * 2.0)
            if init_std_val <= self.min_sigma:
                self.min_sigma = min(self.min_sigma, init_std_val * 0.1)
        aux_cfg = module_config_dict.get("aux_state_pred", {})
        self.aux_state_pred_enabled = bool(aux_cfg.get("enabled", False))
        aux_cmd_cfg = module_config_dict.get("aux_router_command_recon", {})
        self.aux_router_command_recon_enabled = bool(
            aux_cmd_cfg.get("enabled", False)
        )
        aux_switch_cfg = module_config_dict.get(
            "aux_router_switch_penalty", {}
        )
        self.aux_router_switch_penalty_enabled = bool(
            aux_switch_cfg.get("enabled", False)
        )
        aux_router_future_cfg = module_config_dict.get(
            "aux_router_future_recon", {}
        )
        self.aux_router_future_recon_enabled = bool(
            aux_router_future_cfg.get("enabled", False)
        )
        self.aux_router_future_recon_assembler: TensorDictAssembler | None = (
            None
        )

    def _sigma_from_params(self) -> torch.Tensor:
        # Prefer log-std if present; otherwise use softplus(linear) for positivity
        if hasattr(self, "log_std"):
            return torch.exp(self.log_std)
        return F.softplus(self.std)

    def reset_kv_cache(self, num_envs: int, device):
        if hasattr(self.actor_module, "reset_kv_cache"):
            self.actor_module.reset_kv_cache(num_envs, device)

    def clear_env_cache(self, env_ids: torch.Tensor):
        if hasattr(self.actor_module, "clear_env_cache"):
            self.actor_module.clear_env_cache(env_ids)

    def onnx_past_key_values_shape(
        self, *, batch_size: int = 1
    ) -> tuple[int, int, int, int, int, int]:
        num_kv_layers = int(
            getattr(
                self.actor_module, "onnx_kv_layers", self.actor_module.n_layers
            )
        )
        return (
            num_kv_layers,
            2,
            int(batch_size),
            int(self.actor_module.max_ctx_len),
            int(self.actor_module.n_kv_heads),
            int(self.actor_module.head_dim),
        )

    def onnx_moe_layer_indices(self) -> list[int]:
        layers = getattr(self.actor_module, "layers", None)
        if layers is None:
            return []
        return [
            layer_idx
            for layer_idx, layer in enumerate(layers)
            if isinstance(layer, NM.GroupedMoEBlock)
        ]

    def onnx_routing_output_names(self) -> list[str]:
        output_names: list[str] = []
        for layer_idx in self.onnx_moe_layer_indices():
            output_names.extend(
                [
                    f"moe_layer_{layer_idx}_expert_indices",
                    f"moe_layer_{layer_idx}_expert_logits",
                ]
            )
        return output_names

    def _maybe_update_aux_router_future_recon_norm(
        self,
        obs_td: TensorDict,
        *,
        update: bool,
    ) -> None:
        if (
            not update
            or not self.aux_router_future_recon_enabled
            or self.aux_router_future_recon_assembler is None
        ):
            return
        future_target = self.aux_router_future_recon_assembler(obs_td)
        self.actor_module.update_aux_router_future_recon_normalizer(
            future_target
        )

    def export_onnx(
        self,
        onnx_path: str | Path,
        *,
        opset_version: int = 17,
        use_kv_cache: bool = True,
    ) -> str:
        export_path = Path(onnx_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.actor_module, "clear_router_distribution_cache"):
            self.actor_module.clear_router_distribution_cache()
        actor_module = _clone_module_for_cpu_export(self.actor_module)
        if self.obs_norm_enabled:
            obs_normalizer = _clone_module_for_cpu_export(self.obs_normalizer)
        else:
            obs_normalizer = nn.Identity()

        obs = torch.zeros(
            1, self.flat_obs_dim, device="cpu", dtype=torch.float32
        )
        if use_kv_cache:
            exporter = PPOTFActorOnnxModule(
                actor_module=actor_module,
                obs_normalizer=obs_normalizer,
                obs_norm_enabled=self.obs_norm_enabled,
                obs_norm_clip=self.obs_norm_clip
                if self.obs_norm_enabled
                else 0.0,
            ).to("cpu")
            exporter.eval()

            cache_shape = self.onnx_past_key_values_shape(batch_size=1)
            past_key_values = torch.zeros(
                *cache_shape, device="cpu", dtype=torch.float32
            )
            step_idx = torch.tensor([0], dtype=torch.long, device="cpu")
            output_names = [
                "actions",
                "present_key_values",
                *self.onnx_routing_output_names(),
            ]

            torch.onnx.export(
                exporter,
                (obs, past_key_values, step_idx),
                str(export_path),
                export_params=True,
                opset_version=opset_version,
                verbose=False,
                dynamo=False,
                input_names=["obs", "past_key_values", "step_idx"],
                output_names=output_names,
            )
        else:
            exporter = PPOTFWoKVCacheActorOnnxModule(
                actor_module=actor_module,
                obs_normalizer=obs_normalizer,
                obs_norm_enabled=self.obs_norm_enabled,
                obs_norm_clip=self.obs_norm_clip
                if self.obs_norm_enabled
                else 0.0,
            ).to("cpu")
            exporter.eval()
            obs = torch.zeros(
                1, 32, self.flat_obs_dim, device="cpu", dtype=torch.float32
            )

            torch.onnx.export(
                exporter,
                (obs,),
                str(export_path),
                export_params=True,
                opset_version=opset_version,
                verbose=False,
                dynamo=False,
                input_names=["obs"],
                output_names=["actions"],
            )
        return str(export_path)

    def update_distribution(self, actor_obs):
        """Distribution using TransformerDecoderPolicy single-step mu + learnable log-std.

        Args:
            actor_obs: [B, D] normalized obs
        """
        mu = self.actor_module.single_step_mu(actor_obs)
        std = self._sigma_from_params()
        std = torch.clamp(std, min=self.min_sigma, max=self.max_sigma)
        self.distribution = Normal(mu, std)

    def forward(
        self,
        obs_td: TensorDict | torch.Tensor,
        actions: torch.Tensor | None = None,
        mode: str = "sampling",
        attn_mask: torch.Tensor | None = None,
        *,
        update_obs_norm: bool = True,
        past_key_values: torch.Tensor | None = None,
        current_pos: torch.Tensor | None = None,
    ) -> TensorDict | tuple[torch.Tensor, torch.Tensor]:
        """TensorDict-first forward for PPOTFActor.

        Modes:
        - "sampling" / "logp" / "inference": single-step policy with KV-cache-aware
          mean prediction via `actor_module.single_step_mu`.
        - "sequence_logp": sequence log-prob evaluation with attention mask support.
        """
        if past_key_values is not None:
            if isinstance(obs_td, TensorDict):
                if self.assembler is None:
                    raise ValueError(
                        "PPOTFActor requires obs_schema/assembler for ONNX cache path."
                    )
                actor_obs = self.assembler(obs_td)
            else:
                actor_obs = obs_td
            return self.actor_module(
                actor_obs,
                past_key_values=past_key_values,
                current_pos=current_pos,
            )
        if mode == "sequence_logp":
            if not isinstance(obs_td, TensorDict):
                raise ValueError(
                    "PPOTFActor.forward(mode='sequence_logp') expects TensorDict input."
                )
            if obs_td.batch_dims != 2:
                raise ValueError(
                    "PPOTFActor.forward(mode='sequence_logp') expects TensorDict with "
                    f"batch_dims=2 [B, T], got batch_size={tuple(obs_td.batch_size)}"
                )
            if self.assembler is None:
                raise ValueError(
                    "PPOTFActor requires obs_schema to assemble sequence observations."
                )
            if actions is None:
                raise ValueError(
                    "actions must be provided when mode='sequence_logp'"
                )

            b, t = int(obs_td.batch_size[0]), int(obs_td.batch_size[1])
            flat_td = obs_td.flatten(0, 1)
            actor_obs_flat = self.assembler(flat_td)
            update = bool(update_obs_norm)
            for fn in self.actor_obs_transforms:
                actor_obs_flat = fn(actor_obs_flat, update)
            self._maybe_update_aux_router_future_recon_norm(
                flat_td, update=update
            )
            actor_obs_seq = actor_obs_flat.reshape(b, t, -1)

            if actor_obs_seq.ndim != 3:
                raise ValueError(
                    "PPOTFActor forward(mode='sequence_logp') expects actor_obs "
                    f"with shape [B, T, D], got {actor_obs_seq.shape}"
                )
            mu, sigma, logp, entropy, aux_preds = self.sequence_forward_logp(
                actor_obs_seq, actions, attn_mask
            )
            td = obs_td.clone(recurse=False)
            td.set("mu", mu)
            td.set("sigma", sigma)
            td.set("actions", actions)
            td.set("actions_log_prob", logp)
            td.set("entropy", entropy)
            if aux_preds is not None:
                if "base_lin_vel_loc" in aux_preds:
                    td.set(
                        "aux_base_lin_vel_loc", aux_preds["base_lin_vel_loc"]
                    )
                    td.set(
                        "aux_base_lin_vel_log_std",
                        aux_preds["base_lin_vel_log_std"],
                    )
                    td.set("aux_root_height_loc", aux_preds["root_height_loc"])
                    td.set(
                        "aux_root_height_log_std",
                        aux_preds["root_height_log_std"],
                    )
                    td.set(
                        "aux_keybody_contact_logits",
                        aux_preds["keybody_contact_logits"],
                    )
                    td.set(
                        "aux_ref_keybody_rel_pos",
                        aux_preds["ref_keybody_rel_pos"],
                    )
                    td.set(
                        "aux_robot_keybody_rel_pos",
                        aux_preds["robot_keybody_rel_pos"],
                    )
                if "router_command_recon" in aux_preds:
                    td.set(
                        "aux_router_command_recon",
                        aux_preds["router_command_recon"],
                    )
                if "router_future_recon" in aux_preds:
                    td.set(
                        "aux_router_future_recon",
                        aux_preds["router_future_recon"],
                    )
                if "router_features" in aux_preds:
                    td.set("router_features", aux_preds["router_features"])
                if "router_temporal_features" in aux_preds:
                    td.set(
                        "router_temporal_features",
                        aux_preds["router_temporal_features"],
                    )
            return td

        if mode not in ("sampling", "logp", "inference"):
            raise ValueError(f"Unsupported mode: {mode}")
        if not isinstance(obs_td, TensorDict):
            raise ValueError("PPOTFActor.forward expects TensorDict input.")
        if self.assembler is None:
            raise ValueError(
                "Flat-tensor actor module requires obs_schema in PPOTFActor init."
            )

        td = obs_td.clone(recurse=False)
        actor_obs = self.assembler(obs_td)
        update = bool(update_obs_norm)
        for fn in self.actor_obs_transforms:
            actor_obs = fn(actor_obs, update)
        self._maybe_update_aux_router_future_recon_norm(obs_td, update=update)

        if hasattr(self.actor_module, "single_step_mu"):
            mu = self.actor_module.single_step_mu(actor_obs)
        else:
            mu = self.actor_module(actor_obs)
        sigma = self._sigma_like(mu)
        td.set("mu", mu)
        td.set("sigma", sigma)

        if mode == "inference":
            td.set("actions", mu)
            return td

        self.distribution = Normal(mu, sigma)
        if mode == "sampling":
            actions_out = self.distribution.sample()
        else:
            if actions is None:
                raise ValueError("actions must be provided when mode='logp'")
            actions_out = actions
        td.set("actions", actions_out)
        td.set(
            "actions_log_prob",
            self.distribution.log_prob(actions_out).sum(dim=-1),
        )
        td.set("entropy", self.distribution.entropy().sum(dim=-1))
        return td

    def sequence_forward_logp(
        self,
        obs_seq: torch.Tensor,
        actions: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
    ]:
        """Sequence log-prob path with learnable per-action log-std.

        Args:
            obs_seq: [B, T, D]
            actions: [B, T, A]
            attn_mask: [B, T, T] boolean (True if attend allowed)

        Returns:
            mu: [B, T, A], sigma: [B, T, A], logp: [B, T, 1], entropy: [B, T, 1]
        """
        aux_preds = None
        aux_router_future_recon_enabled = bool(
            getattr(self, "aux_router_future_recon_enabled", False)
        )
        need_pre_moe_aux = self.aux_state_pred_enabled
        need_router_features = (
            self.aux_router_command_recon_enabled
            or self.aux_router_switch_penalty_enabled
        )
        need_router_aux = (
            need_router_features or aux_router_future_recon_enabled
        )
        need_ref_aux_hidden = bool(
            (need_pre_moe_aux or aux_router_future_recon_enabled)
            and getattr(
                self.actor_module, "supports_explicit_ref_aux_hidden", False
            )
        )
        if need_pre_moe_aux and need_router_aux:
            sequence_mu_kwargs = {
                "attn_mask": attn_mask,
                "return_pre_moe_hidden": True,
                "return_router_features": need_router_features,
                "return_router_temporal_features": self.aux_router_switch_penalty_enabled,
            }
            if need_ref_aux_hidden:
                sequence_mu_kwargs["return_ref_aux_hidden"] = True
            actor_outputs = self.actor_module.sequence_mu(
                obs_seq,
                **sequence_mu_kwargs,
            )
            output_parts = list(actor_outputs)
            mu = output_parts.pop(0)
            pre_moe_hidden = output_parts.pop(0)
            ref_aux_hidden = (
                output_parts.pop(0) if need_ref_aux_hidden else None
            )
            router_features = (
                output_parts.pop(0) if need_router_features else None
            )
            router_temporal_features = (
                output_parts.pop(0)
                if self.aux_router_switch_penalty_enabled
                else None
            )
            aux_preds = self.actor_module.predict_aux_from_pre_moe(
                pre_moe_hidden,
                ref_aux_hidden=ref_aux_hidden if need_ref_aux_hidden else None,
            )
            if router_features is not None:
                aux_preds["router_features"] = router_features
            if router_temporal_features is not None:
                aux_preds["router_temporal_features"] = (
                    router_temporal_features
                )
            if self.aux_router_command_recon_enabled:
                aux_preds["router_command_recon"] = (
                    self.actor_module.predict_aux_router_command_from_router_features(
                        router_features
                    )
                )
            if aux_router_future_recon_enabled:
                aux_preds["router_future_recon"] = (
                    self.actor_module.predict_aux_router_future_recon_from_router_hidden(
                        ref_aux_hidden
                    )
                )
        elif need_pre_moe_aux:
            sequence_mu_kwargs = {
                "attn_mask": attn_mask,
                "return_pre_moe_hidden": True,
            }
            if need_ref_aux_hidden:
                sequence_mu_kwargs["return_ref_aux_hidden"] = True
            actor_outputs = self.actor_module.sequence_mu(
                obs_seq,
                **sequence_mu_kwargs,
            )
            if need_ref_aux_hidden:
                mu, pre_moe_hidden, ref_aux_hidden = actor_outputs
            else:
                mu, pre_moe_hidden = actor_outputs
            aux_preds = self.actor_module.predict_aux_from_pre_moe(
                pre_moe_hidden,
                ref_aux_hidden=ref_aux_hidden if need_ref_aux_hidden else None,
            )
        elif need_router_aux:
            sequence_mu_kwargs = {
                "attn_mask": attn_mask,
                "return_router_features": need_router_features,
                "return_router_temporal_features": self.aux_router_switch_penalty_enabled,
            }
            if need_ref_aux_hidden:
                sequence_mu_kwargs["return_ref_aux_hidden"] = True
            actor_outputs = self.actor_module.sequence_mu(
                obs_seq,
                **sequence_mu_kwargs,
            )
            output_parts = list(actor_outputs)
            mu = output_parts.pop(0)
            ref_aux_hidden = (
                output_parts.pop(0) if need_ref_aux_hidden else None
            )
            router_features = (
                output_parts.pop(0) if need_router_features else None
            )
            router_temporal_features = (
                output_parts.pop(0)
                if self.aux_router_switch_penalty_enabled
                else None
            )
            aux_preds = {}
            if router_features is not None:
                aux_preds["router_features"] = router_features
            if router_temporal_features is not None:
                aux_preds["router_temporal_features"] = (
                    router_temporal_features
                )
            if self.aux_router_command_recon_enabled:
                aux_preds["router_command_recon"] = (
                    self.actor_module.predict_aux_router_command_from_router_features(
                        router_features
                    )
                )
            if aux_router_future_recon_enabled:
                aux_preds["router_future_recon"] = (
                    self.actor_module.predict_aux_router_future_recon_from_router_hidden(
                        ref_aux_hidden
                    )
                )
        else:
            mu = self.actor_module.sequence_mu(obs_seq, attn_mask=attn_mask)
        # Match sampling-time clamping for stability and consistent KL/log-prob
        sigma_vec = self._sigma_from_params().clamp(
            self.min_sigma, self.max_sigma
        )
        sigma = sigma_vec[None, None, :].expand_as(mu)
        var = sigma * sigma
        logp = -0.5 * (
            ((actions - mu) ** 2) / (var + 1.0e-8)
            + 2.0 * torch.log(sigma + 1.0e-8)
            + math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)
        entropy = (
            0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(sigma + 1.0e-8)
        ).sum(dim=-1, keepdim=True)
        return mu, sigma, logp, entropy, aux_preds


class PPOTFRefRouterActor(PPOTFActor):
    @staticmethod
    def _leaf_obs_name(term: str) -> str:
        return str(term).rsplit("/", maxsplit=1)[-1]

    @classmethod
    def _infer_flat_term_dim(
        cls,
        *,
        obs_example: TensorDict,
        term: str,
        seq_len: int,
    ) -> int:
        tensor = TensorDictAssembler._get_from_data(obs_example, str(term))
        if tensor is None:
            raise KeyError(
                f"Missing obs term '{term}' in obs_example while inferring "
                "reference-router feature indices."
            )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Obs term '{term}' must be a torch.Tensor, got {type(tensor)}."
            )
        if tensor.ndim == 2:
            if seq_len != 1:
                raise ValueError(
                    f"Obs term '{term}' expected seq_len={seq_len} but tensor "
                    f"is 2D with shape {tuple(tensor.shape)}."
                )
            return int(tensor.shape[-1])
        if tensor.ndim == 3:
            if int(tensor.shape[1]) != seq_len:
                raise ValueError(
                    f"Obs term '{term}' seq_len mismatch: expected {seq_len}, "
                    f"got {int(tensor.shape[1])}."
                )
            return int(tensor.shape[1] * tensor.shape[-1])
        raise ValueError(
            f"Obs term '{term}' tensor ndim must be 2 or 3, got {tensor.ndim}."
        )

    @classmethod
    def infer_router_feature_indices(
        cls,
        obs_schema: dict,
        obs_example: TensorDict,
        router_obs_terms=None,
    ) -> list[int]:
        if not isinstance(obs_example, TensorDict):
            raise ValueError(
                "PPOTFRefRouterActor requires TensorDict obs_example."
            )

        router_term_names = None
        requested_router_terms: list[str] = []
        if router_obs_terms is not None:
            requested_router_terms = [str(term) for term in router_obs_terms]
            if len(requested_router_terms) == 0:
                raise ValueError(
                    "PPOTFRefRouterActor received an empty router_obs_terms "
                    "whitelist."
                )
            router_term_names = set(requested_router_terms)
            router_term_names.update(
                cls._leaf_obs_name(term) for term in requested_router_terms
            )
        router_feature_indices: list[int] = []
        offset = 0
        for _, seq_cfg in obs_schema.items():
            if not isinstance(seq_cfg, dict):
                continue
            seq_len = int(seq_cfg.get("seq_len", 1))
            for term in seq_cfg.get("terms", []):
                term_str = str(term)
                flat_dim = cls._infer_flat_term_dim(
                    obs_example=obs_example,
                    term=term_str,
                    seq_len=seq_len,
                )
                leaf_name = cls._leaf_obs_name(term_str)
                if router_term_names is None:
                    include_router_term = leaf_name.startswith("actor_ref_")
                else:
                    include_router_term = (
                        term_str in router_term_names
                        or leaf_name in router_term_names
                    )
                if include_router_term:
                    router_feature_indices.extend(
                        range(offset, offset + flat_dim)
                    )
                offset += flat_dim

        if len(router_feature_indices) == 0:
            if router_term_names is None:
                raise ValueError(
                    "PPOTFRefRouterActor could not infer any actor_ref_* "
                    "features from obs_schema."
                )
            raise ValueError(
                "PPOTFRefRouterActor could not match any configured "
                f"router_obs_terms in obs_schema: {requested_router_terms}"
            )
        return router_feature_indices

    def __init__(
        self,
        obs_schema: dict | None,
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
        *,
        obs_example: dict | None = None,
    ):
        if obs_schema is None:
            raise ValueError(
                "PPOTFRefRouterActor requires non-empty obs_schema."
            )
        if obs_example is None:
            raise ValueError("PPOTFRefRouterActor requires obs_example.")
        if bool(module_config_dict.get("use_future_cross_attn", False)):
            raise ValueError(
                "PPOTFRefRouterActor does not support use_future_cross_attn=True."
            )

        actor_module_cfg = copy.deepcopy(module_config_dict)
        aux_future_cfg = actor_module_cfg.get("aux_router_future_recon", {})
        if bool(aux_future_cfg.get("enabled", False)):
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicy does not support "
                "aux_router_future_recon."
            )
        router_feature_indices = self.infer_router_feature_indices(
            obs_schema,
            obs_example,
            router_obs_terms=actor_module_cfg.get("router_obs_terms", None),
        )
        actor_module_cfg["router_input_dim"] = int(len(router_feature_indices))
        actor_module_cfg["router_feature_indices"] = list(
            router_feature_indices
        )
        if "router_embed_mlp_hidden" not in actor_module_cfg:
            actor_module_cfg["router_embed_mlp_hidden"] = int(
                actor_module_cfg.get("obs_embed_mlp_hidden", 1024)
            )

        super().__init__(
            obs_schema=obs_schema,
            module_config_dict=actor_module_cfg,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            obs_example=obs_example,
        )
        self.router_feature_indices = list(router_feature_indices)


class PPOTFRefRouterSeqActor(PPOTFActor):
    REQUIRED_CURRENT_REF_TERMS = (
        "actor_ref_gravity_projection_cur",
        "actor_ref_base_linvel_cur",
        "actor_ref_base_angvel_cur",
        "actor_ref_dof_pos_cur",
        "actor_ref_root_height_cur",
    )
    REQUIRED_FUTURE_REF_TERMS = (
        "actor_ref_gravity_projection_fut",
        "actor_ref_base_linvel_fut",
        "actor_ref_base_angvel_fut",
        "actor_ref_dof_pos_fut",
        "actor_ref_root_height_fut",
    )
    SUPPORTED_AUX_WEIGHT_NAMES = {
        "w_base_lin_vel",
        "w_keybody_contact",
        "w_ref_keybody_rel_pos",
        "w_robot_keybody_rel_pos",
    }

    @staticmethod
    def _leaf_obs_name(term: str) -> str:
        return str(term).rsplit("/", maxsplit=1)[-1]

    @classmethod
    def _infer_flat_term_dim(
        cls,
        *,
        obs_example: TensorDict,
        term: str,
        seq_len: int,
    ) -> int:
        tensor = TensorDictAssembler._get_from_data(obs_example, str(term))
        if tensor is None:
            raise KeyError(
                f"Missing obs term '{term}' in obs_example while inferring shared ref partitions."
            )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Obs term '{term}' must be a torch.Tensor, got {type(tensor)}."
            )
        if tensor.ndim == 2:
            if seq_len != 1:
                raise ValueError(
                    f"Obs term '{term}' expected seq_len={seq_len} but tensor "
                    f"is 2D with shape {tuple(tensor.shape)}."
                )
            return int(tensor.shape[-1])
        if tensor.ndim == 3:
            if int(tensor.shape[1]) != seq_len:
                raise ValueError(
                    f"Obs term '{term}' seq_len mismatch: expected {seq_len}, "
                    f"got {int(tensor.shape[1])}."
                )
            return int(tensor.shape[-1])
        raise ValueError(
            f"Obs term '{term}' tensor ndim must be 2 or 3, got {tensor.ndim}."
        )

    @classmethod
    def _validate_v2_aux_config(cls, module_config_dict: dict) -> None:
        aux_cmd_cfg = module_config_dict.get("aux_router_command_recon", {})
        if bool(aux_cmd_cfg.get("enabled", False)):
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 does not support "
                "aux_router_command_recon."
            )
        aux_cfg = module_config_dict.get("aux_state_pred", {})
        if not bool(aux_cfg.get("enabled", False)):
            return
        for key, value in aux_cfg.items():
            if not str(key).startswith("w_"):
                continue
            if float(value) <= 0.0:
                continue
            if str(key) not in cls.SUPPORTED_AUX_WEIGHT_NAMES:
                raise ValueError(
                    "ReferenceRoutedGroupedMoETransformerPolicyV2 only supports "
                    "aux_state_pred weights for "
                    "base_lin_vel, keybody_contact, ref_keybody_rel_pos, and "
                    "robot_keybody_rel_pos. Unsupported weight: "
                    f"{key}."
                )

    @classmethod
    def _build_aux_router_future_recon_schema(
        cls, obs_schema: dict
    ) -> dict[str, dict]:
        required_terms = set(cls.REQUIRED_FUTURE_REF_TERMS)
        matched_terms: set[str] = set()
        future_schema: dict[str, dict] = {}

        for group_name, seq_cfg in obs_schema.items():
            if not isinstance(seq_cfg, dict):
                continue
            terms = [
                str(term)
                for term in seq_cfg.get("terms", [])
                if cls._leaf_obs_name(str(term)) in required_terms
            ]
            if len(terms) == 0:
                continue
            next_seq_cfg = dict(seq_cfg)
            next_seq_cfg["terms"] = terms
            future_schema[str(group_name)] = next_seq_cfg
            matched_terms.update(cls._leaf_obs_name(term) for term in terms)

        missing_terms = sorted(required_terms.difference(matched_terms))
        if missing_terms:
            raise ValueError(
                "PPOTFRefRouterSeqActor could not infer all future ref terms "
                "for aux_router_future_recon. Missing: "
                + ", ".join(missing_terms)
            )
        return future_schema

    @classmethod
    def _prepare_aux_router_future_recon(
        cls,
        *,
        actor_module_cfg: dict,
        obs_schema: dict,
        obs_example: TensorDict,
    ) -> TensorDictAssembler | None:
        aux_future_cfg = copy.deepcopy(
            actor_module_cfg.get("aux_router_future_recon", {})
        )
        if not bool(aux_future_cfg.get("enabled", False)):
            actor_module_cfg["aux_router_future_recon"] = aux_future_cfg
            return None

        future_schema = cls._build_aux_router_future_recon_schema(obs_schema)
        future_assembler = TensorDictAssembler(
            future_schema, output_mode="flat"
        )
        aux_future_cfg["output_dim"] = int(
            future_assembler.infer_output_dim(obs_example)
        )
        actor_module_cfg["aux_router_future_recon"] = aux_future_cfg
        return future_assembler

    @classmethod
    def _infer_shared_ref_layout(
        cls,
        obs_schema: dict,
        obs_example: TensorDict,
    ) -> dict[str, int | list[int] | list[tuple[int, int, int]]]:
        if not isinstance(obs_example, TensorDict):
            raise ValueError(
                "PPOTFRefRouterSeqActor requires TensorDict obs_example."
            )

        required_cur = set(cls.REQUIRED_CURRENT_REF_TERMS)
        required_fut = set(cls.REQUIRED_FUTURE_REF_TERMS)
        found_cur: dict[str, tuple[int, int]] = {}
        found_fut: dict[str, tuple[int, int, int]] = {}
        state_indices: list[int] = []
        ref_cur_indices: list[int] = []
        offset = 0
        ref_fut_seq_len: int | None = None

        for _, seq_cfg in obs_schema.items():
            if not isinstance(seq_cfg, dict):
                continue
            seq_len = int(seq_cfg.get("seq_len", 1))
            for term in seq_cfg.get("terms", []):
                term_str = str(term)
                leaf_name = cls._leaf_obs_name(term_str)
                flat_term_dim = cls._infer_flat_term_dim(
                    obs_example=obs_example,
                    term=term_str,
                    seq_len=seq_len,
                )
                flat_span = int(seq_len * flat_term_dim)
                term_range = list(range(offset, offset + flat_span))

                if leaf_name in required_cur:
                    if seq_len != 1:
                        raise ValueError(
                            "current ref term "
                            f"'{leaf_name}' must have seq_len=1, got {seq_len}."
                        )
                    if leaf_name in found_cur:
                        raise ValueError(
                            f"duplicate current ref term '{leaf_name}' in obs_schema."
                        )
                    found_cur[leaf_name] = (offset, flat_term_dim)
                    ref_cur_indices.extend(term_range)
                elif leaf_name in required_fut:
                    if leaf_name in found_fut:
                        raise ValueError(
                            f"duplicate future ref term '{leaf_name}' in obs_schema."
                        )
                    if ref_fut_seq_len is None:
                        ref_fut_seq_len = seq_len
                    elif ref_fut_seq_len != seq_len:
                        raise ValueError(
                            "future ref terms must share one seq_len, got "
                            f"{ref_fut_seq_len} and {seq_len}."
                        )
                    found_fut[leaf_name] = (
                        offset,
                        offset + flat_span,
                        flat_term_dim,
                    )
                else:
                    state_indices.extend(term_range)

                offset += flat_span

        missing_cur = sorted(required_cur.difference(found_cur.keys()))
        if missing_cur:
            raise ValueError(
                "missing required current ref term(s): "
                + ", ".join(missing_cur)
            )
        missing_fut = sorted(required_fut.difference(found_fut.keys()))
        if missing_fut:
            raise ValueError(
                "missing required future ref term(s): "
                + ", ".join(missing_fut)
            )
        if ref_fut_seq_len is None or ref_fut_seq_len <= 0:
            raise ValueError(
                "missing required future ref terms in obs_schema."
            )
        if len(state_indices) == 0:
            raise ValueError(
                "ReferenceRoutedGroupedMoETransformerPolicyV2 requires at least "
                "one non-reference actor state feature."
            )

        ordered_fut_slices = [
            found_fut[leaf_name] for leaf_name in cls.REQUIRED_FUTURE_REF_TERMS
        ]
        return {
            "full_obs_input_dim": int(offset),
            "state_obs_input_dim": int(len(state_indices)),
            "ref_cur_token_dim": int(len(ref_cur_indices)),
            "ref_fut_token_dim": int(
                sum(end - start for start, end, _ in ordered_fut_slices)
                // ref_fut_seq_len
            ),
            "ref_fut_seq_len": int(ref_fut_seq_len),
            "state_feature_indices": state_indices,
            "ref_cur_feature_indices": ref_cur_indices,
            "ref_fut_slices": ordered_fut_slices,
        }

    def __init__(
        self,
        obs_schema: dict | None,
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
        *,
        obs_example: dict | None = None,
    ):
        if obs_schema is None:
            raise ValueError(
                "PPOTFRefRouterSeqActor requires non-empty obs_schema."
            )
        if obs_example is None:
            raise ValueError("PPOTFRefRouterSeqActor requires obs_example.")
        if bool(module_config_dict.get("use_future_cross_attn", False)):
            raise ValueError(
                "PPOTFRefRouterSeqActor does not support use_future_cross_attn=True."
            )
        self._validate_v2_aux_config(module_config_dict)
        inferred_layout = self._infer_shared_ref_layout(
            obs_schema, obs_example
        )

        actor_module_cfg = copy.deepcopy(module_config_dict)
        actor_module_cfg["input_dim_override"] = int(
            inferred_layout["state_obs_input_dim"]
        )
        actor_module_cfg["state_obs_input_dim"] = int(
            inferred_layout["state_obs_input_dim"]
        )
        actor_module_cfg["ref_cur_token_dim"] = int(
            inferred_layout["ref_cur_token_dim"]
        )
        actor_module_cfg["ref_fut_token_dim"] = int(
            inferred_layout["ref_fut_token_dim"]
        )
        actor_module_cfg["ref_fut_seq_len"] = int(
            inferred_layout["ref_fut_seq_len"]
        )
        actor_module_cfg["state_feature_indices"] = list(
            inferred_layout["state_feature_indices"]
        )
        actor_module_cfg["ref_cur_feature_indices"] = list(
            inferred_layout["ref_cur_feature_indices"]
        )
        actor_module_cfg["ref_fut_slices"] = [
            list(item) for item in inferred_layout["ref_fut_slices"]
        ]
        actor_module_cfg.pop("router_hist_obs_schema", None)
        actor_module_cfg.pop("router_fut_obs_schema", None)

        super().__init__(
            obs_schema=obs_schema,
            module_config_dict=actor_module_cfg,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            obs_example=obs_example,
        )
        self.full_obs_input_dim = int(inferred_layout["full_obs_input_dim"])
        self.state_obs_input_dim = int(inferred_layout["state_obs_input_dim"])
        self.ref_cur_token_dim = int(inferred_layout["ref_cur_token_dim"])
        self.ref_fut_token_dim = int(inferred_layout["ref_fut_token_dim"])
        self.ref_fut_seq_len = int(inferred_layout["ref_fut_seq_len"])
        self.state_feature_indices = list(
            inferred_layout["state_feature_indices"]
        )
        self.ref_cur_feature_indices = list(
            inferred_layout["ref_cur_feature_indices"]
        )
        self.ref_fut_slices = [
            tuple(int(v) for v in item)
            for item in inferred_layout["ref_fut_slices"]
        ]


class PPOTFRefRouterV3Actor(PPOTFRefRouterSeqActor):
    def __init__(
        self,
        obs_schema: dict | None,
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
        *,
        obs_example: dict | None = None,
    ):
        if obs_schema is None:
            raise ValueError(
                "PPOTFRefRouterV3Actor requires non-empty obs_schema."
            )
        if obs_example is None:
            raise ValueError("PPOTFRefRouterV3Actor requires obs_example.")
        if bool(module_config_dict.get("use_future_cross_attn", False)):
            raise ValueError(
                "PPOTFRefRouterV3Actor does not support use_future_cross_attn=True."
            )
        self._validate_v2_aux_config(module_config_dict)
        inferred_layout = self._infer_shared_ref_layout(
            obs_schema, obs_example
        )

        actor_module_cfg = copy.deepcopy(module_config_dict)
        actor_module_cfg["state_obs_input_dim"] = int(
            inferred_layout["state_obs_input_dim"]
        )
        actor_module_cfg["ref_cur_token_dim"] = int(
            inferred_layout["ref_cur_token_dim"]
        )
        actor_module_cfg["ref_fut_token_dim"] = int(
            inferred_layout["ref_fut_token_dim"]
        )
        actor_module_cfg["ref_fut_seq_len"] = int(
            inferred_layout["ref_fut_seq_len"]
        )
        actor_module_cfg["state_feature_indices"] = list(
            inferred_layout["state_feature_indices"]
        )
        actor_module_cfg["ref_cur_feature_indices"] = list(
            inferred_layout["ref_cur_feature_indices"]
        )
        actor_module_cfg["ref_fut_slices"] = [
            list(item) for item in inferred_layout["ref_fut_slices"]
        ]
        actor_module_cfg.pop("router_hist_obs_schema", None)
        actor_module_cfg.pop("router_fut_obs_schema", None)
        future_recon_assembler = self._prepare_aux_router_future_recon(
            actor_module_cfg=actor_module_cfg,
            obs_schema=obs_schema,
            obs_example=obs_example,
        )

        PPOTFActor.__init__(
            self,
            obs_schema=obs_schema,
            module_config_dict=actor_module_cfg,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            obs_example=obs_example,
        )
        self.full_obs_input_dim = int(inferred_layout["full_obs_input_dim"])
        self.state_obs_input_dim = int(inferred_layout["state_obs_input_dim"])
        self.ref_cur_token_dim = int(inferred_layout["ref_cur_token_dim"])
        self.ref_fut_token_dim = int(inferred_layout["ref_fut_token_dim"])
        self.ref_fut_seq_len = int(inferred_layout["ref_fut_seq_len"])
        self.state_feature_indices = list(
            inferred_layout["state_feature_indices"]
        )
        self.ref_cur_feature_indices = list(
            inferred_layout["ref_cur_feature_indices"]
        )
        self.ref_fut_slices = [
            tuple(int(v) for v in item)
            for item in inferred_layout["ref_fut_slices"]
        ]
        self.aux_router_future_recon_assembler = future_recon_assembler


class PPOCondTFActor(PPOTFActor):
    """Transformer actor with flat state obs and seq future-token conditioning."""

    def __init__(
        self,
        obs_schema: dict | None,
        module_config_dict: dict,
        num_actions: int,
        init_noise_std: float,
        *,
        obs_example: dict | None = None,
    ):
        super().__init__(
            obs_schema=obs_schema,
            module_config_dict=module_config_dict,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            obs_example=obs_example,
        )
        if obs_schema is None:
            raise ValueError("PPOCondTFActor requires non-empty obs_schema.")
        if "flattened_obs" not in obs_schema:
            raise ValueError("obs_schema must contain 'flattened_obs'.")
        if "flattened_obs_fut" not in obs_schema:
            raise ValueError("obs_schema must contain 'flattened_obs_fut'.")
        if obs_example is None:
            raise ValueError("PPOCondTFActor requires obs_example.")

        self.state_schema = {"flattened_obs": obs_schema["flattened_obs"]}
        self.future_schema = {
            "flattened_obs_fut": obs_schema["flattened_obs_fut"]
        }
        self.state_assembler = TensorDictAssembler(
            self.state_schema, output_mode="flat"
        )
        self.future_assembler = TensorDictAssembler(
            self.future_schema, output_mode="seq"
        )
        self.state_dim = int(
            self.state_assembler.infer_output_dim(obs_example)
        )
        self.future_token_dim = int(
            self.future_assembler.infer_output_dim(obs_example)
        )
        self.future_seq_len = int(self.future_assembler.seq_len)
        self.future_term_dims = self._infer_future_term_dims(obs_example)
        self.full_obs_dim = int(self.flat_obs_dim)
        expected_full = self.state_dim + (
            self.future_seq_len * self.future_token_dim
        )
        if self.full_obs_dim != expected_full:
            raise ValueError(
                "Assembled obs dim mismatch in PPOCondTFActor: "
                f"full={self.full_obs_dim}, expected={expected_full}"
            )
        if self.obs_norm_enabled:
            self.state_obs_normalizer = EmpiricalNormalization(
                shape=self.state_dim,
                eps=self.obs_norm_eps,
                update_method=self.obs_norm_update_method,
                ema_momentum=self.obs_norm_ema_momentum,
            )
        else:
            self.state_obs_normalizer = nn.Identity()

    def _infer_future_term_dims(self, obs_example: TensorDict) -> list[int]:
        if not isinstance(obs_example, TensorDict):
            raise ValueError("PPOCondTFActor requires TensorDict obs_example.")
        fut_cfg = self.future_schema.get("flattened_obs_fut", None)
        if fut_cfg is None:
            raise ValueError(
                "Missing future schema group 'flattened_obs_fut'."
            )
        terms = fut_cfg.get("terms", [])
        if not isinstance(terms, list) or len(terms) == 0:
            raise ValueError("Future schema terms must be a non-empty list.")
        dims: list[int] = []
        for term in terms:
            tensor = TensorDictAssembler._get_from_data(obs_example, str(term))
            if tensor is None:
                raise KeyError(
                    f"Missing future term '{term}' in obs_example TensorDict."
                )
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Future term '{term}' must be a torch.Tensor, got {type(tensor)}"
                )
            if tensor.ndim == 2:
                dims.append(int(tensor.shape[-1]))
            elif tensor.ndim == 3:
                dims.append(int(tensor.shape[-1]))
            else:
                raise ValueError(
                    f"Future term '{term}' tensor ndim must be 2 or 3, got {tensor.ndim}"
                )
        if sum(dims) != int(self.future_token_dim):
            raise ValueError(
                "Inferred future_term_dims sum mismatch: expected "
                f"{int(self.future_token_dim)}, got {sum(dims)} (dims={dims})"
            )
        return dims

    @property
    def flat_obs_dim(self) -> int:
        if self.assembler is None:
            raise ValueError(
                "PPOCondTFActor requires the base flat assembler for ONNX."
            )
        if self.assembler.output_dim is None:
            raise ValueError("Base assembler output_dim is not initialized.")
        return int(self.assembler.output_dim)

    def _normalize_state_obs(
        self, state_obs: torch.Tensor, update: bool
    ) -> torch.Tensor:
        if not self.obs_norm_enabled:
            return state_obs
        if state_obs.ndim != 2:
            raise ValueError(
                f"state_obs must be [B, D_state], got {tuple(state_obs.shape)}"
            )
        if update:
            self.state_obs_normalizer.update(state_obs)
        state_obs = self.state_obs_normalizer.normalize_only(state_obs)
        if self.obs_norm_clip > 0.0:
            state_obs = torch.clamp(
                state_obs, -self.obs_norm_clip, self.obs_norm_clip
            )
        return state_obs

    def _assemble_state_future(
        self, obs_td: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(obs_td, TensorDict):
            raise ValueError(
                "PPOCondTFActor._assemble_state_future expects TensorDict input."
            )
        state_obs = self.state_assembler(obs_td)
        future_obs = self.future_assembler(obs_td)
        return state_obs, future_obs

    def _split_flat_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim != 2:
            raise ValueError(f"Expected [B, D], got {obs.shape}")
        state_obs = obs[:, : self.state_dim]
        future_flat = obs[:, self.state_dim :]
        expected_dim = self.future_seq_len * self.future_token_dim
        if future_flat.shape[-1] != expected_dim:
            raise ValueError(
                "Future flat obs dim mismatch: expected "
                f"{expected_dim}, got {future_flat.shape[-1]}"
            )
        b = int(obs.shape[0])
        offset = 0
        future_parts = []
        for d_term in self.future_term_dims:
            span = int(self.future_seq_len * d_term)
            chunk = future_flat[:, offset : offset + span]
            future_parts.append(chunk.reshape(b, self.future_seq_len, d_term))
            offset += span
        if offset != int(future_flat.shape[-1]):
            raise ValueError(
                "Future flat slicing mismatch: "
                f"consumed={offset}, total={int(future_flat.shape[-1])}"
            )
        future_obs = torch.cat(future_parts, dim=-1)
        return state_obs, future_obs

    def export_onnx(
        self,
        onnx_path: str | Path,
        *,
        opset_version: int = 17,
    ) -> str:
        export_path = Path(onnx_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.actor_module, "clear_router_distribution_cache"):
            self.actor_module.clear_router_distribution_cache()
        actor_module = _clone_module_for_cpu_export(self.actor_module)
        if self.obs_norm_enabled:
            state_obs_normalizer = _clone_module_for_cpu_export(
                self.state_obs_normalizer
            )
        else:
            state_obs_normalizer = nn.Identity()

        exporter = PPOCondTFActorOnnxModule(
            actor_module=actor_module,
            state_obs_normalizer=state_obs_normalizer,
            obs_norm_enabled=self.obs_norm_enabled,
            obs_norm_clip=self.obs_norm_clip if self.obs_norm_enabled else 0.0,
            state_dim=self.state_dim,
            future_seq_len=self.future_seq_len,
            future_token_dim=self.future_token_dim,
            future_term_dims=self.future_term_dims,
        ).to("cpu")
        exporter.eval()

        cache_shape = self.onnx_past_key_values_shape(batch_size=1)
        obs = torch.zeros(
            1, self.flat_obs_dim, device="cpu", dtype=torch.float32
        )
        past_key_values = torch.zeros(
            *cache_shape, device="cpu", dtype=torch.float32
        )
        step_idx = torch.tensor([0], dtype=torch.long, device="cpu")
        output_names = [
            "actions",
            "present_key_values",
            *self.onnx_routing_output_names(),
        ]

        torch.onnx.export(
            exporter,
            (obs, past_key_values, step_idx),
            str(export_path),
            export_params=True,
            opset_version=opset_version,
            verbose=False,
            dynamo=False,
            input_names=["obs", "past_key_values", "step_idx"],
            output_names=output_names,
        )
        return str(export_path)

    def update_distribution(self, actor_obs):
        if not isinstance(actor_obs, tuple) or len(actor_obs) != 2:
            raise ValueError(
                "PPOCondTFActor.update_distribution expects tuple(state_obs, future_obs)."
            )
        state_obs, future_obs = actor_obs
        mu = self.actor_module.single_step_mu_cond(
            state_obs,
            future_obs,
            future_mask=None,
        )
        std = self._sigma_from_params()
        std = torch.clamp(std, min=self.min_sigma, max=self.max_sigma)
        self.distribution = Normal(mu, std)

    def forward(
        self,
        obs_td: TensorDict | torch.Tensor,
        actions: torch.Tensor | None = None,
        mode: str = "sampling",
        attn_mask: torch.Tensor | None = None,
        *,
        update_obs_norm: bool = True,
        past_key_values: torch.Tensor | None = None,
        current_pos: torch.Tensor | None = None,
    ) -> TensorDict | tuple[torch.Tensor, torch.Tensor]:
        if past_key_values is not None:
            if isinstance(obs_td, TensorDict):
                state_obs, future_obs = self._assemble_state_future(obs_td)
            else:
                state_obs, future_obs = self._split_flat_obs(obs_td)
            state_obs = self._normalize_state_obs(state_obs, update=False)
            return self.actor_module._forward_inference_onnx_cond(
                state_obs,
                future_obs,
                past_key_values,
                current_pos,
            )

        if mode == "sequence_logp":
            if not isinstance(obs_td, TensorDict):
                raise ValueError(
                    "PPOCondTFActor.forward(mode='sequence_logp') expects TensorDict input."
                )
            if obs_td.batch_dims != 2:
                raise ValueError(
                    "PPOCondTFActor.forward(mode='sequence_logp') expects batch_dims=2 [B, T], "
                    f"got batch_size={tuple(obs_td.batch_size)}"
                )
            if actions is None:
                raise ValueError(
                    "actions must be provided when mode='sequence_logp'"
                )

            b, t = int(obs_td.batch_size[0]), int(obs_td.batch_size[1])
            future_mask = None
            if "future_mask" in obs_td.keys():
                future_mask = obs_td.get("future_mask")
                if future_mask.shape != (b, t, self.future_seq_len):
                    raise ValueError(
                        "future_mask shape mismatch in sequence_logp: expected "
                        f"{(b, t, self.future_seq_len)}, got {tuple(future_mask.shape)}"
                    )
                future_mask = future_mask.to(torch.bool)
            flat_td = obs_td.flatten(0, 1)
            state_flat, future_flat = self._assemble_state_future(flat_td)
            update = bool(update_obs_norm)
            state_flat = self._normalize_state_obs(state_flat, update=update)
            state_seq = state_flat.reshape(b, t, -1)
            future_seq = future_flat.reshape(
                b, t, self.future_seq_len, self.future_token_dim
            )

            (
                mu,
                sigma,
                logp,
                entropy,
                aux_preds,
            ) = self.sequence_forward_logp_cond(
                state_seq,
                future_seq,
                actions,
                attn_mask,
                future_mask,
            )
            td = obs_td.clone(recurse=False)
            td.set("mu", mu)
            td.set("sigma", sigma)
            td.set("actions", actions)
            td.set("actions_log_prob", logp)
            td.set("entropy", entropy)
            if aux_preds is not None:
                if "base_lin_vel_loc" in aux_preds:
                    td.set(
                        "aux_base_lin_vel_loc", aux_preds["base_lin_vel_loc"]
                    )
                    td.set(
                        "aux_base_lin_vel_log_std",
                        aux_preds["base_lin_vel_log_std"],
                    )
                    td.set("aux_root_height_loc", aux_preds["root_height_loc"])
                    td.set(
                        "aux_root_height_log_std",
                        aux_preds["root_height_log_std"],
                    )
                    td.set(
                        "aux_keybody_contact_logits",
                        aux_preds["keybody_contact_logits"],
                    )
                    td.set(
                        "aux_ref_keybody_rel_pos",
                        aux_preds["ref_keybody_rel_pos"],
                    )
                    td.set(
                        "aux_robot_keybody_rel_pos",
                        aux_preds["robot_keybody_rel_pos"],
                    )
                if "router_command_recon" in aux_preds:
                    td.set(
                        "aux_router_command_recon",
                        aux_preds["router_command_recon"],
                    )
                if "router_features" in aux_preds:
                    td.set("router_features", aux_preds["router_features"])
                if "router_temporal_features" in aux_preds:
                    td.set(
                        "router_temporal_features",
                        aux_preds["router_temporal_features"],
                    )
            return td

        if mode not in ("sampling", "logp", "inference"):
            raise ValueError(f"Unsupported mode: {mode}")
        if not isinstance(obs_td, TensorDict):
            raise ValueError(
                "PPOCondTFActor.forward expects TensorDict input."
            )

        td = obs_td.clone(recurse=False)
        state_obs, future_obs = self._assemble_state_future(obs_td)
        update = bool(update_obs_norm)
        state_obs = self._normalize_state_obs(state_obs, update=update)
        future_mask = None
        if "future_mask" in td.keys():
            future_mask = td.get("future_mask")
            if future_mask.shape != (state_obs.shape[0], self.future_seq_len):
                raise ValueError(
                    "future_mask shape mismatch in single-step forward: expected "
                    f"{(state_obs.shape[0], self.future_seq_len)}, got {tuple(future_mask.shape)}"
                )
            future_mask = future_mask.to(torch.bool)
        mu = self.actor_module.single_step_mu_cond(
            state_obs, future_obs, future_mask=future_mask
        )
        sigma = self._sigma_like(mu)
        td.set("mu", mu)
        td.set("sigma", sigma)

        if mode == "inference":
            td.set("actions", mu)
            return td

        self.distribution = Normal(mu, sigma)
        if mode == "sampling":
            actions_out = self.distribution.sample()
        else:
            if actions is None:
                raise ValueError("actions must be provided when mode='logp'")
            actions_out = actions
        td.set("actions", actions_out)
        td.set(
            "actions_log_prob",
            self.distribution.log_prob(actions_out).sum(dim=-1),
        )
        td.set("entropy", self.distribution.entropy().sum(dim=-1))
        return td

    def sequence_forward_logp_cond(
        self,
        state_seq: torch.Tensor,
        future_seq: torch.Tensor,
        actions: torch.Tensor,
        attn_mask: torch.Tensor | None,
        future_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
    ]:
        aux_preds = None
        need_pre_moe_aux = self.aux_state_pred_enabled
        need_router_aux = (
            self.aux_router_command_recon_enabled
            or self.aux_router_switch_penalty_enabled
        )
        if need_pre_moe_aux and need_router_aux:
            actor_outputs = self.actor_module.sequence_mu_cond(
                state_seq,
                future_seq,
                attn_mask=attn_mask,
                future_mask=future_mask,
                return_pre_moe_hidden=True,
                return_router_features=True,
                return_router_temporal_features=self.aux_router_switch_penalty_enabled,
            )
            if self.aux_router_switch_penalty_enabled:
                (
                    mu,
                    pre_moe_hidden,
                    router_features,
                    router_temporal_features,
                ) = actor_outputs
            else:
                mu, pre_moe_hidden, router_features = actor_outputs
            aux_preds = self.actor_module.predict_aux_from_pre_moe(
                pre_moe_hidden
            )
            aux_preds["router_features"] = router_features
            if self.aux_router_switch_penalty_enabled:
                aux_preds["router_temporal_features"] = (
                    router_temporal_features
                )
            if self.aux_router_command_recon_enabled:
                aux_preds["router_command_recon"] = (
                    self.actor_module.predict_aux_router_command_from_router_features(
                        router_features
                    )
                )
        elif need_pre_moe_aux:
            mu, pre_moe_hidden = self.actor_module.sequence_mu_cond(
                state_seq,
                future_seq,
                attn_mask=attn_mask,
                future_mask=future_mask,
                return_pre_moe_hidden=True,
            )
            aux_preds = self.actor_module.predict_aux_from_pre_moe(
                pre_moe_hidden
            )
        elif need_router_aux:
            actor_outputs = self.actor_module.sequence_mu_cond(
                state_seq,
                future_seq,
                attn_mask=attn_mask,
                future_mask=future_mask,
                return_router_features=True,
                return_router_temporal_features=self.aux_router_switch_penalty_enabled,
            )
            if self.aux_router_switch_penalty_enabled:
                (
                    mu,
                    router_features,
                    router_temporal_features,
                ) = actor_outputs
            else:
                mu, router_features = actor_outputs
            aux_preds = {"router_features": router_features}
            if self.aux_router_switch_penalty_enabled:
                aux_preds["router_temporal_features"] = (
                    router_temporal_features
                )
            if self.aux_router_command_recon_enabled:
                aux_preds["router_command_recon"] = (
                    self.actor_module.predict_aux_router_command_from_router_features(
                        router_features
                    )
                )
        else:
            mu = self.actor_module.sequence_mu_cond(
                state_seq,
                future_seq,
                attn_mask=attn_mask,
                future_mask=future_mask,
            )
        sigma_vec = self._sigma_from_params().clamp(
            self.min_sigma, self.max_sigma
        )
        sigma = sigma_vec[None, None, :].expand_as(mu)
        var = sigma * sigma
        logp = -0.5 * (
            ((actions - mu) ** 2) / (var + 1.0e-8)
            + 2.0 * torch.log(sigma + 1.0e-8)
            + math.log(2.0 * math.pi)
        ).sum(dim=-1, keepdim=True)
        entropy = (
            0.5 + 0.5 * math.log(2.0 * math.pi) + torch.log(sigma + 1.0e-8)
        ).sum(dim=-1, keepdim=True)
        return mu, sigma, logp, entropy, aux_preds
