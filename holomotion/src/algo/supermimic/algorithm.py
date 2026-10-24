"""SuperMimic algorithm entry for the shared HoloMotion training lifecycle."""

import os
from dataclasses import asdict, replace
from pathlib import Path

from omegaconf import OmegaConf

from holomotion.src.algo.supermimic.config import SuperMimicConfig
from holomotion.src.algo.supermimic.training import execute


class SuperMimic:
    """Own model-based training without inheriting PPO's rollout assumptions."""

    def __init__(self, env_config, config, log_dir=None, headless=True):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(
                "This SuperMimic recipe currently supports one training process"
            )
        if not headless:
            raise ValueError(
                "The batched SuperMimic trainer currently requires headless=true"
            )
        if log_dir is None:
            raise ValueError(
                "Training requires a resolved experiment directory"
            )
        environment = OmegaConf.to_container(env_config.config, resolve=True)
        options = OmegaConf.to_container(config, resolve=True)
        overlap = {"manifest", "robot_urdf", "output"} & set(options)
        if overlap:
            raise ValueError(
                f"Data and output paths belong to the environment/runner: {sorted(overlap)}"
            )
        self.options = SuperMimicConfig.from_mapping(
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
        """Restore learning state and begin new physical episodes."""
        if checkpoint is not None:
            if self.options.resume is not None:
                raise ValueError(
                    "Specify either load checkpoint or algo.config.resume"
                )
            self.options = SuperMimicConfig.from_mapping(
                asdict(replace(self.options, resume=Path(checkpoint)))
            )

    def learn(self):
        """Run real collection, learned dynamics, imagination and full evaluation."""
        self.report = execute(self.options)
        return self.report
