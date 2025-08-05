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
"""Evaluate the motion tracking model."""

import copy
import os
import sys
from pathlib import Path

import hydra
import isaacgym  # noqa: F401
import torch
from accelerate import Accelerator
from hydra.utils import get_class
from loguru import logger
from omegaconf import OmegaConf

from holomotion.src.utils.config import compile_config

torch.set_float32_matmul_precision("high")


def setup_logging(accelerator: Accelerator):
    """Configure logging for the evaluation process."""
    logger.remove()
    if accelerator.is_main_process:
        logger.add(
            sys.stdout,
            level=os.environ.get("LOGURU_LEVEL", "INFO").upper(),
            colorize=True,
        )


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
                f"Training config not found at {config_path}, "
                "using evaluation config"
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

    # For evaluation, merge eval_config into train_config
    # This allows all command line overrides to take effect
    config = OmegaConf.merge(train_config, eval_config)

    return config


def setup_eval_directories(config: OmegaConf, checkpoint_path: str):
    """Setup evaluation-specific directories."""
    checkpoint = Path(checkpoint_path)
    ckpt_num = checkpoint.name.split("_")[-1].split(".")[0]

    eval_log_dir = checkpoint.parent / "eval_logs" / f"ckpt_{ckpt_num}"
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    config.env.config.save_rendering_dir = str(
        checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}"
    )
    config.env.config.ckpt_dir = str(checkpoint.parent)
    config.env.config.eval_log_dir = str(eval_log_dir)

    return eval_log_dir


def export_policy_to_onnx(algo, checkpoint_path: str, device: str):
    """Export the policy to ONNX format."""
    checkpoint = Path(checkpoint_path)
    export_dir = checkpoint.parent / "exported"
    export_dir.mkdir(exist_ok=True)

    onnx_name = checkpoint.name.replace(".pt", ".onnx")
    onnx_path = export_dir / onnx_name

    class WrappedActor(torch.nn.Module):
        def __init__(self, ppo):
            super().__init__()
            # Handle DistributedDataParallel wrapper from accelerate
            if hasattr(ppo.actor, "module"):
                self.actor = ppo.actor.module.actor_module
            else:
                self.actor = ppo.actor.actor_module

        def forward(self, raw_obs):
            return self.actor(raw_obs)

    wrapped_actor = copy.deepcopy(WrappedActor(algo)).to(device)

    # Determine input dimension
    # Handle DistributedDataParallel wrapper from accelerate
    if hasattr(algo.actor, "module"):
        actor_module = algo.actor.module.actor_module
    else:
        actor_module = algo.actor.actor_module

    if "tf" in algo.actor_type.lower():
        feature_dim = actor_module.obs_serializer.obs_flat_dim
    else:
        feature_dim = actor_module.input_dim

    example_input = torch.randn(1, feature_dim, device=device)

    torch.onnx.export(
        wrapped_actor,
        example_input,
        onnx_path,
        verbose=False,
        input_names=["raw_obs"],
        output_names=["action"],
        opset_version=18,
        export_params=True,
        do_constant_folding=True,
    )

    logger.info(f"Exported policy to: {onnx_path}")


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_isaacgym",
    version_base=None,
)
def main(config: OmegaConf):
    """Evaluate the motion tracking model.

    Args:
        config: OmegaConf object containing the configuration.

    """
    if config.checkpoint is None:
        raise ValueError("Checkpoint path must be provided for evaluation")

    # Load training config and apply evaluation overrides
    config = load_training_config(config.checkpoint, config)

    # Setup accelerator and logging
    accelerator = Accelerator()
    setup_logging(accelerator)

    # Compile configuration
    config = compile_config(config, accelerator)

    # Setup evaluation directories
    eval_log_dir = setup_eval_directories(config, config.checkpoint)

    # Motion file validation
    if not config.get("robot", {}).get("motion", {}).get("motion_file", None):
        raise ValueError(
            "Motion file is not set in training config or overrides!"
        )

    if accelerator.is_main_process:
        logger.info(f"Evaluating checkpoint: {config.checkpoint}")
        logger.info(f"Saving eval logs to: {eval_log_dir}")

        # Save evaluation config
        with open(eval_log_dir / "eval_config.yaml", "w") as file:
            OmegaConf.save(config, file)

    # Create environment
    env_class = get_class(config.env._target_)
    env = env_class(
        config=config.env.config,
        device=accelerator.device,
    )

    # Create algorithm
    algo_class = get_class(config.algo.algo._target_)
    algo = algo_class(
        env=env,
        config=config.algo.algo.config,
        log_dir=str(eval_log_dir),
        device=accelerator.device,
    )

    # Setup and load checkpoint
    algo.setup()
    algo.load(config.checkpoint)

    # Export policy to ONNX if enabled
    if config.get("export_policy", True):
        export_policy_to_onnx(algo, config.checkpoint, accelerator.device)

    # Run evaluation
    algo.evaluate_policy()


if __name__ == "__main__":
    main()
