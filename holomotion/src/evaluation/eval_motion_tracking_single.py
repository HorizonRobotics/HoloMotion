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

import copy
import os
import re
import sys
from pathlib import Path
from typing import Optional

import hydra
import onnx
import torch
import torch.nn as nn
from hydra.utils import get_class
from loguru import logger
from omegaconf import OmegaConf

from holomotion.src.utils.config import compile_config


def load_training_config(
    checkpoint_path: str, eval_config: OmegaConf
) -> OmegaConf:
    """Load training config from checkpoint directory.

    Args:
        checkpoint_path: Path to the checkpoint file.
        eval_config: Full evaluation config (including command line overrides).

    Returns:
        Merged config with training config as base.
    """
    checkpoint = Path(checkpoint_path)
    config_path = checkpoint.parent / "config.yaml"

    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.exists():
            logger.warning(
                f"Training config not found at {config_path}, using evaluation config"
            )
            return eval_config

    logger.info(f"Loading training config from {config_path}")
    with open(config_path) as file:
        train_config = OmegaConf.load(file)

    # Apply eval_overrides from training config if they exist
    if train_config.get("eval_overrides") is not None:
        train_config = OmegaConf.merge(
            train_config, train_config.eval_overrides
        )

    # Set checkpoint path
    train_config.checkpoint = checkpoint_path
    train_config.algo.config.checkpoint = checkpoint_path

    # For evaluation, merge eval_config into train_config
    config = OmegaConf.merge(train_config, eval_config)

    # force set the terminations and domain rand with eval_config's
    config.env.config.terminations = eval_config.env.config.terminations
    config.env.config.domain_rand = eval_config.env.config.domain_rand

    return config


def export_policy_to_onnx(
    algo,
    checkpoint_path: str,
    onnx_name_suffix: Optional[str] = None,
):
    """Export a minimal ONNX that takes flattened obs and outputs actions only.

    - Supports obs from configs like obs_isaaclab_nose (uses obs_serializer dims)
    - Attaches metadata needed for MuJoCo sim2sim (PD params, defaults, action_scale)
    """

    checkpoint = Path(checkpoint_path)
    export_dir = checkpoint.parent / "exported"
    export_dir.mkdir(exist_ok=True)

    onnx_name = checkpoint.name.replace(".pt", ".onnx")
    if onnx_name_suffix is not None:
        onnx_name_suffix = re.sub(r"[\s+]", "_", onnx_name_suffix)
        onnx_name = onnx_name.replace(".onnx", f"_{onnx_name_suffix}.onnx")
    onnx_path = export_dir / onnx_name

    logger.info("Starting ONNX minimal policy export (actions-only)...")

    # Set models to evaluation mode
    algo.actor.eval()
    algo.critic.eval()

    class _OnnxPolicyHoloMotion(nn.Module):
        def __init__(self, ppo_algo):
            super().__init__()
            # Always use Accelerate, so check if actor is wrapped
            if hasattr(ppo_algo.actor, "module"):
                self.actor = copy.deepcopy(ppo_algo.actor.module)
            else:
                self.actor = copy.deepcopy(ppo_algo.actor)
            self.actor.to("cpu")
            self.actor.eval()

            # Copy normalizer state if enabled
            self.obs_norm_enabled = bool(
                getattr(ppo_algo, "obs_norm_enabled", False)
            )
            self.actor_obs_normalizer = None
            if (
                self.obs_norm_enabled
                and getattr(ppo_algo, "obs_normalizer", None) is not None
            ):
                self.actor_obs_normalizer = copy.deepcopy(
                    ppo_algo.obs_normalizer
                )
                self.actor_obs_normalizer.to("cpu")
                self.actor_obs_normalizer.eval()

        def forward(self, obs):
            # obs: [B, F]
            if self.obs_norm_enabled and self.actor_obs_normalizer is not None:
                if hasattr(self.actor_obs_normalizer, "normalize"):
                    obs = self.actor_obs_normalizer.normalize(obs)
                else:
                    obs = self.actor_obs_normalizer(obs)
            actions, _, _, _, _ = self.actor(
                obs, actions=None, mode="inference"
            )
            return actions

    exporter = _OnnxPolicyHoloMotion(algo).to("cpu")

    F = int(algo.obs_serializer.obs_flat_dim)
    obs = torch.zeros(1, F, device="cpu")
    torch.onnx.export(
        exporter,
        (obs,),
        onnx_path,
        export_params=True,
        opset_version=11,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )

    # Attach rich metadata needed by MuJoCo sim2sim
    attach_onnx_metadata_holomotion(
        algo.env._env,
        onnx_path=str(onnx_path),
    )
    logger.info(f"Successfully exported minimal policy to: {onnx_path}")

    return str(onnx_path)


def attach_onnx_metadata_holomotion(env, onnx_path: str):
    def list_to_csv_str(
        arr, *, decimals: int = 3, delimiter: str = ","
    ) -> str:
        fmt = f"{{:.{decimals}f}}"
        return delimiter.join(
            fmt.format(x) if isinstance(x, (int, float)) else str(x)
            for x in arr  # numbers → format, strings → as-is
        )

    metadata = {
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"]
        .data.joint_stiffness[0]
        .cpu()
        .tolist(),
        "joint_damping": env.scene["robot"]
        .data.joint_damping[0]
        .cpu()
        .tolist(),
        "default_joint_pos": env.scene["robot"]
        .data.default_joint_pos[0]
        .cpu()
        .tolist(),
        "action_scale": env.action_manager.get_term("dof_pos")
        ._scale[0]
        .cpu()
        .tolist(),
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_isaaclab",
    version_base=None,
)
def main(config: OmegaConf):
    """Evaluate the motion tracking model.

    Args:
        config: OmegaConf object containing the evaluation configuration.

    """
    # Load training config first
    if config.checkpoint is None:
        raise ValueError("Checkpoint path must be provided for evaluation")

    config = load_training_config(config.checkpoint, config)
    # Compile config without accelerator (PPO will create it)
    config = compile_config(config, accelerator=None)

    # Use checkpoint directory as log_dir for offline evaluation
    log_dir = os.path.dirname(config.checkpoint)
    headless = config.headless

    # PPO creates Accelerator, AppLauncher, and environment internally
    algo_class = get_class(config.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.config,
        log_dir=log_dir,
        headless=headless,
        is_offline_eval=True,
    )

    if (
        algo.accelerator.is_main_process
        and os.environ.get("TORCH_COMPILE_DISABLE", "0") != "1"
    ):
        logger.info(
            "Tip: If you encounter Triton/compilation errors during evaluation,"
        )
        logger.info(
            "     set environment variable: export TORCH_COMPILE_DISABLE=1"
        )

    if algo.accelerator.is_main_process:
        eval_log_dir = os.path.dirname(config.checkpoint)
        with open(os.path.join(eval_log_dir, "eval_config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    if hasattr(config, "checkpoint") and config.checkpoint is not None:
        if algo.accelerator.is_main_process:
            logger.info(
                f"Loading checkpoint for evaluation: {config.checkpoint}"
            )
        algo.load(config.checkpoint)
    else:
        if algo.accelerator.is_main_process:
            logger.warning("No checkpoint provided for evaluation!")

    command_name = list(config.env.config.commands.keys())[0]
    if command_name == "ref_motion":
        motion_cmd = algo.env._env.command_manager.get_term("ref_motion")
        algo.env._env.reset()
        motion_cmd._update_ref_motion_state()

    # Export ONNX if requested
    if config.get("export_policy", True):
        if algo.accelerator.is_main_process:
            onnx_name_suffix = config.get("onnx_name_suffix", None)
            onnx_path = export_policy_to_onnx(
                algo, config.checkpoint, onnx_name_suffix
            )
            logger.info(f"Successfully exported policy to: {onnx_path}")
        algo.accelerator.wait_for_everyone()

    # Dump NPZs only; metrics will be computed by a separate script
    result = algo.offline_evaluate_policy(config.get("dump_npzs", False))
    if algo.accelerator.is_main_process:
        logger.info("Evaluation completed successfully!")
        if isinstance(result, dict) and "output_dir" in result:
            logger.info(f"NPZs saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
