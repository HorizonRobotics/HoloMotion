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

import os
from pathlib import Path

import hydra
from hydra.utils import get_class
from loguru import logger
from omegaconf import OmegaConf

from holomotion.src.utils.config import compile_config
from holomotion.src.utils.onnx_export import export_policy_to_onnx


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
                "Training config not found at "
                f"{config_path}, using evaluation config"
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

    # For velocity tracking, always keep the robot configuration from training
    if hasattr(train_config, "robot"):
        config.robot = train_config.robot

    # foce set the terminations and domain rand with eval_config's
    config.env.config.terminations = eval_config.env.config.terminations
    config.env.config.domain_rand = eval_config.env.config.domain_rand
    config.env.config.domain_rand = eval_config.env.config.domain_rand

    return config


@hydra.main(
    config_path="../../config",
    config_name="evaluation/eval_isaaclab",
    version_base=None,
)
def main(config: OmegaConf):
    """Evaluate the velocity tracking model.

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
            "Tip: set TORCH_COMPILE_DISABLE=1 if Triton/compile errors occur"
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

    # Export ONNX if requested
    if config.get("export_policy", True):
        if algo.accelerator.is_main_process:
            onnx_name_suffix = config.get("onnx_name_suffix", None)
            onnx_path = export_policy_to_onnx(
                algo,
                config.checkpoint,
                onnx_name_suffix=onnx_name_suffix,
                use_kv_cache=config.get("use_kv_cache", True),
            )
            logger.info(f"Successfully exported policy to: {onnx_path}")
        algo.accelerator.wait_for_everyone()

    # Run indefinite velocity tracking rollout for visualization
    algo.offline_evaluate_velocity_tracking()
    if algo.accelerator.is_main_process:
        logger.info("Velocity tracking evaluation completed!")


if __name__ == "__main__":
    main()
