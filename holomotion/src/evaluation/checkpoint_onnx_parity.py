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

"""Strict saved actor checkpoint versus deployed ONNX/KV parity."""

from __future__ import annotations
import hashlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from holomotion.src.algo.ppo_tf import PPOTF
from holomotion.src.modules.agent_modules import (
    PPOTFActorOnnxModule,
)
from holomotion.src.motion_tracking.link_actor_packing import (
    link_actor_layout,
)


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def actor_example_from_flat(flat, config=None):
    """Invert the saved atomic observation shapes."""
    import math

    if config is None:
        layout = link_actor_layout(
            {"link_names": list(range(14)), "frame_offsets": [1, 2]}
        )
    else:
        layout = link_actor_layout(
            config.link_observation,
            config.algo.config.module_dict.actor.obs_schema,
        )
    flat = torch.as_tensor(flat, dtype=torch.float32)
    if flat.ndim != 2 or flat.shape[1] != layout["actor_dim"]:
        raise ValueError(
            f"Parity expects B x {layout['actor_dim']} observations"
        )
    terms = {}
    cursor = 0
    for name in layout["atomic_terms"]:
        shape = layout["term_shapes"][name]
        width = math.prod(shape)
        terms[name] = flat[:, cursor : cursor + width].reshape(
            flat.shape[0], *shape
        )
        cursor += width
    assert cursor == layout["actor_dim"]
    return TensorDict(
        {"unified": TensorDict(terms, batch_size=[flat.shape[0]])},
        batch_size=[flat.shape[0]],
    )


def construct_checkpoint_actor(config, example):
    """Construct the saved actor without Isaac, critic or optimizer."""
    algo = config.algo.config
    actor_cfg = OmegaConf.to_container(algo.module_dict.actor, resolve=True)
    # The actor wrapper is selected by the saved config.  This keeps parity
    # model-agnostic while preserving strict checkpoint loading.
    if bool(actor_cfg.get("use_future_cross_attn", False)):
        raise ValueError(
            "Conditional ActorRef export is not supported by this branch"
        )
    actor_wrapper_cls = PPOTF._select_actor_wrapper_cls(actor_cfg)
    for key, default in [
        ("noise_std_type", "log"),
        ("min_sigma", 0.1),
        ("max_sigma", 1.5),
        ("fix_sigma", False),
    ]:
        actor_cfg[key] = algo.get(key, default)
    for key in [
        "aux_state_pred",
        "aux_router_command_recon",
        "aux_router_future_recon",
        "aux_router_switch_penalty",
        "moe_load_balance",
        "inactive_expert_margin_to_topk",
        "selected_expert_margin_to_unselected",
    ]:
        value = algo.get(key, {})
        actor_cfg[key] = (
            OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else dict(value)
        )
    if actor_cfg["aux_router_command_recon"].get(
        "enabled", False
    ) or actor_cfg["aux_router_future_recon"].get("enabled", False):
        raise ValueError(
            "Additional reconstruction-head schemas need a dedicated "
            "parity adapter"
        )
    schema = actor_cfg["obs_schema"]
    link_config = config.get(
        "link_observation",
        {"link_names": list(range(14)), "frame_offsets": [1, 2]},
    )
    layout = link_actor_layout(link_config, schema)
    actor_cfg["input_dim_override"] = layout["actor_dim"]
    actor = actor_wrapper_cls(
        obs_schema=schema,
        module_config_dict=actor_cfg,
        num_actions=29,
        init_noise_std=float(algo.init_noise_std),
        obs_example=example,
    )
    if actor.flat_obs_dim != layout["actor_dim"]:
        raise ValueError(
            "Constructed actor dimensions disagree with saved schema"
        )
    return actor


def build_actor_onnx_module(actor, actor_config):
    """Build the exact export wrapper selected by the saved actor config."""
    cfg = (
        OmegaConf.to_container(actor_config, resolve=True)
        if OmegaConf.is_config(actor_config)
        else dict(actor_config)
    )
    if bool(cfg.get("use_future_cross_attn", False)):
        raise ValueError(
            "Conditional ActorRef export is not supported by this branch"
        )
    return PPOTFActorOnnxModule(
        actor.actor_module,
        actor.obs_normalizer,
        actor.obs_norm_enabled,
        actor.obs_norm_clip if actor.obs_norm_enabled else 0,
    )


def load_actor_strict(actor, actor_directory):
    """Strictly load actor and normalizer weights, never optimizer state."""
    from accelerate.utils import load_state_dict

    directory = Path(actor_directory).resolve(strict=True)
    path = next(
        (
            directory / name
            for name in ["model.safetensors", "pytorch_model.bin"]
            if (directory / name).is_file()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            "Expected the current single-file Accelerate actor checkpoint"
        )
    state = load_state_dict(str(path))
    state = {
        key.removeprefix("_orig_mod."): value for key, value in state.items()
    }
    actor.load_state_dict(state, strict=True)
    if actor.obs_norm_enabled:
        required_normalizers = ["obs_normalizer._mean"]
        missing = [key for key in required_normalizers if key not in state]
        if missing:
            raise ValueError(
                "Saved actor is missing enabled normalizer state: "
                + ", ".join(missing)
            )
    return {
        "actor_file": str(path),
        "actor_sha256": file_sha256(path),
        "strict_actor_state": True,
        "normalizer_restored": bool(actor.obs_norm_enabled),
    }


def compare_arrays(actual, expected, atol, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "shape_actual": list(actual.shape),
            "shape_expected": list(expected.shape),
            "max_abs": None,
            "max_rel": None,
        }
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {
            "passed": False,
            "finite": False,
            "max_abs": None,
            "max_rel": None,
        }
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = delta / np.maximum(np.abs(expected), atol)
    return {
        "passed": bool(np.all(delta <= atol + rtol * np.abs(expected))),
        "finite": True,
        "shape": list(actual.shape),
        "max_abs": float(delta.max(initial=0)),
        "max_rel": float(relative.max(initial=0)),
    }


class CheckpointOnnxParity:
    """Compare checkpoint/ONNX actions and independent recurrent KV states.

    Currently supports the integrated v2 link-observation PPOTF actor schema;
    conditional ActorRef and other unsupported schemas fail explicitly.
    """

    def __init__(
        self,
        actor,
        session,
        io_names,
        *,
        steps=8,
        device="cpu",
        atol=1e-4,
        rtol=1e-4,
        checkpoint_metadata=None,
        actor_config=None,
    ):
        if steps != 8:
            raise ValueError(
                "The initial checkpoint protocol is exactly eight "
                "continuous steps"
            )
        self.device = torch.device(device)
        self.actor = actor.to(self.device).float().eval()
        if actor_config is None:
            actor_config = {
                "type": type(actor.actor_module).__name__,
                "use_future_cross_attn": False,
            }
        self.module = build_actor_onnx_module(self.actor, actor_config).eval()
        self.names = dict(io_names)
        inputs = {node.name: node for node in session.get_inputs()}
        outputs = {node.name: node for node in session.get_outputs()}
        if not all(
            self.names[key] in inputs for key in ["obs", "past", "step"]
        ) or not all(
            self.names[key] in outputs for key in ["actions", "present"]
        ):
            raise ValueError(
                "Actual ONNX I/O names do not match the requested "
                "observation/KV contract"
            )
        shape = inputs[self.names["past"]].shape
        if any(not isinstance(size, int) or size <= 0 for size in shape):
            raise ValueError(
                "Current checkpoint parity expects a concrete batch=1 KV shape"
            )
        self.cache = torch.zeros(
            shape, dtype=torch.float32, device=self.device
        )
        if tuple(self.cache.shape) != tuple(
            self.actor.onnx_past_key_values_shape(batch_size=1)
        ):
            raise ValueError(
                "Torch and ONNX KV layouts differ; do not "
                "reinterpret tuple/list ordering"
            )
        self.limit, self.atol, self.rtol = steps, atol, rtol
        self.rows = []
        self.error = None
        # MoE diagnostic buffers may change during inference; only the
        # observation normalizer must remain frozen.
        self.normalizer_before = {
            name: value.detach().cpu().clone()
            for name, value in self.actor.obs_normalizer.named_buffers()
        }
        self.metadata = dict(checkpoint_metadata or {})
        self.metadata.update(
            {
                "protocol": "link_checkpoint_pt_onnx_kv_v1",
                "io_names": self.names,
                "kv_shape": list(shape),
                "device": str(self.device),
                "onnx_providers": session.get_providers(),
                "onnx_provider_options": session.get_provider_options(),
                "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "atol": atol,
                "rtol": rtol,
                "relative_error_floor": atol,
                "expected_steps": steps,
                "precision": (
                    "FP32 eager, autocast disabled; actual "
                    "evaluation ORT session reused"
                ),
                "torch_version": torch.__version__,
                "actor_type": str(actor_config.get("type", "")),
                "use_future_cross_attn": bool(
                    actor_config.get("use_future_cross_attn", False)
                ),
            }
        )

    def check_step(self, input_feed, named_onnx_outputs):
        if len(self.rows) >= self.limit or self.error is not None:
            return
        try:
            obs = np.asarray(input_feed[self.names["obs"]], dtype=np.float32)
            step = np.asarray(input_feed[self.names["step"]], dtype=np.int64)
            if step.shape != (1,) or int(step[0]) != len(self.rows):
                raise ValueError(
                    "Parity must begin at zero cache and continue "
                    "the first eight clip steps"
                )
            if len(self.rows) == 0 and np.count_nonzero(
                input_feed[self.names["past"]]
            ):
                raise ValueError("The first ONNX cache is not zero")
            with (
                torch.inference_mode(),
                torch.autocast(device_type=self.device.type, enabled=False),
            ):
                result = self.module(
                    torch.as_tensor(obs, device=self.device),
                    self.cache,
                    torch.as_tensor(step, device=self.device),
                )
            # Outputs: actions, present KV, optional routing.
            # Align ONNX outputs by their actual names.
            if (
                not isinstance(result, (tuple, list))
                or len(result) < 2
                or not isinstance(result[1], torch.Tensor)
            ):
                raise ValueError(
                    "Unexpected Torch KV output container; refusing "
                    "implicit flatten/reorder"
                )
            actions, present = result[0], result[1]
            if tuple(present.shape) != tuple(self.cache.shape):
                raise ValueError("Unexpected Torch KV layout")
            action_metrics = compare_arrays(
                named_onnx_outputs[self.names["actions"]],
                actions.detach().cpu().numpy(),
                self.atol,
                self.rtol,
            )
            cache_metrics = compare_arrays(
                named_onnx_outputs[self.names["present"]],
                present.detach().cpu().numpy(),
                self.atol,
                self.rtol,
            )
            self.rows.append(
                {
                    "step_idx": int(step[0]),
                    "actions": action_metrics,
                    "present_key_values": cache_metrics,
                }
            )
            self.cache = present.detach()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def report(self):
        unchanged = all(
            torch.equal(
                value,
                dict(self.actor.obs_normalizer.named_buffers())[name]
                .detach()
                .cpu(),
            )
            for name, value in self.normalizer_before.items()
        )
        passed = (
            len(self.rows) == self.limit
            and self.error is None
            and unchanged
            and all(
                row["actions"]["passed"]
                and row["present_key_values"]["passed"]
                for row in self.rows
            )
        )
        return {
            **self.metadata,
            "status": "passed" if passed else "failed",
            "checked_steps": len(self.rows),
            "normalizer_unchanged": unchanged,
            "error": self.error,
            "steps": self.rows,
            "limitations": (
                "Deterministic mean actions and "
                "independent continuous KV only. This "
                "does not prove physical tracking or "
                "compare stochastic sampled training "
                "actions."
            ),
        }
