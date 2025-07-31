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

import os
import sys

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
    """Configure logging for the training process."""
    logger.remove()
    if accelerator.is_main_process:
        logger.add(
            sys.stdout,
            level=os.environ.get("LOGURU_LEVEL", "INFO").upper(),
            colorize=True,
        )


@hydra.main(
    config_path="../../config",
    config_name="training/train_base",
    version_base=None,
)
def main(config: OmegaConf):
    """Train the motion tracking model.

    Args:
        config: OmegaConf object containing the configuration.

    """
    accelerator = Accelerator()

    setup_logging(accelerator)
    config = compile_config(config, accelerator)

    env_class = get_class(config.env._target_)
    env = env_class(
        config=config.env.config,
        device=accelerator.device,
    )

    algo_class = get_class(config.algo.algo._target_)
    algo = algo_class(
        env=env,
        config=config.algo.algo.config,
        log_dir=config.experiment_save_dir,
        device=accelerator.device,
    )
    algo.setup()
    if config.checkpoint is not None:
        algo.load(config.checkpoint)
    algo.learn()


if __name__ == "__main__":
    main()
