import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TERMINATION_CONFIG_PATHS = [
    PROJECT_ROOT
    / "holomotion/config/env/terminations/termination_motion_tracking.yaml",
]

ROBOT_TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
REWARD_CONFIG_PATH = (
    PROJECT_ROOT
    / "holomotion/config/env/rewards/motion_tracking/rew_motion_tracking.yaml"
)


class ReferenceMotionConfigWiringTests(unittest.TestCase):
    def test_motion_tracking_termination_configs_forward_ref_prefix(self):
        for config_path in TERMINATION_CONFIG_PATHS:
            with self.subTest(config_path=str(config_path)):
                config = OmegaConf.load(config_path)
                for term_name, term_cfg in config.terminations.items():
                    if term_name == "time_out":
                        continue
                    self.assertIn("params", term_cfg)
                    self.assertIn("ref_prefix", term_cfg.params)

    def test_robot_training_config_uses_hdf5_v2_backend(self):
        config = OmegaConf.load(ROBOT_TRAINING_CONFIG_PATH)
        self.assertEqual(config.robot.motion.backend, "hdf5_v2")

    def test_termination_ref_prefix_resolves_with_reward_config(self):
        reward_config = OmegaConf.load(REWARD_CONFIG_PATH)
        for config_path in TERMINATION_CONFIG_PATHS:
            with self.subTest(config_path=str(config_path)):
                termination_config = OmegaConf.load(config_path)
                merged = OmegaConf.merge(reward_config, termination_config)
                for term_name, term_cfg in merged.terminations.items():
                    if term_name == "time_out":
                        continue
                    resolved_ref_prefix = OmegaConf.select(
                        merged,
                        f"terminations.{term_name}.params.ref_prefix",
                    )
                    self.assertIsInstance(resolved_ref_prefix, str)
                    self.assertTrue(resolved_ref_prefix.endswith("ref_"))
