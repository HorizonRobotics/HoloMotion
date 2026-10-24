"""Framework adapter for the shared reference-task PPO control."""

import os
from pathlib import Path

from omegaconf import OmegaConf

from .config import ReferencePPOConfig
from .training import execute


class ReferencePPO:
    def __init__(self, env_config, config, log_dir=None, headless=True):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not headless:
            raise ValueError(
                "Reference PPO requires one headless training process"
            )
        if log_dir is None:
            raise ValueError("A resolved experiment directory is required")
        environment = OmegaConf.to_container(env_config.config, resolve=True)
        options = OmegaConf.to_container(config, resolve=True)
        if {"manifest", "robot_urdf", "output"} & set(options):
            raise ValueError(
                "Data and output paths belong to the environment and runner"
            )
        self.options = ReferencePPOConfig.from_mapping(
            dict(
                options,
                manifest=environment["motion_manifest"],
                robot_urdf=environment["robot_urdf"],
                output=Path(log_dir) / "training",
            )
        )
        self.is_main_process = True
        self.report = None

    def load(self, checkpoint):
        if checkpoint is not None:
            raise ValueError(
                "Reference PPO currently supports fresh training; checkpoint recovery is not implemented"
            )

    def learn(self):
        self.report = execute(self.options)
        return self.report
