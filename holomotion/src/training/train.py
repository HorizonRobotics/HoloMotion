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

import hydra
from hydra.utils import get_class
from omegaconf import OmegaConf

from holomotion.src.utils.config import compile_config


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

    config = compile_config(config, accelerator=None)

    log_dir = config.experiment_save_dir
    headless = config.headless

    algo_class = get_class(config.algo.algo._target_)
    algo = algo_class(
        env_config=config.env,
        config=config.algo.algo.config,
        log_dir=log_dir,
        headless=headless,
    )

    algo.load(config.checkpoint)
    algo.learn()


if __name__ == "__main__":
    main()
