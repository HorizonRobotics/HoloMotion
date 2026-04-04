import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OBS_CONFIG_PATHS = [
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motion_tracking.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motrack_tf_ref_v3.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motrack_tf_more_info.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motrack_mlp_20260210.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motrack_tf_20260210.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motrack_teacher.yaml",
]

TERMINATION_CONFIG_PATHS = [
    PROJECT_ROOT
    / "holomotion/config/env/terminations/termination_motion_tracking.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/terminations/termination_motion_tracking_simple.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/terminations/termination_motrack_with_kpe.yaml",
    PROJECT_ROOT
    / "holomotion/config/env/terminations/termination_motrack_with_kpe_jpe.yaml",
]

ROBOT_TRAINING_CONFIG_PATH = (
    PROJECT_ROOT
    / "holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"
)
REWARD_CONFIG_PATH = (
    PROJECT_ROOT
    / "holomotion/config/env/rewards/motion_tracking/rew_motrack_robust.yaml"
)


class ReferenceMotionConfigWiringTests(unittest.TestCase):
    def test_motion_tracking_observation_configs_expose_cutoff_term(self):
        for config_path in OBS_CONFIG_PATHS:
            with self.subTest(config_path=str(config_path)):
                config = OmegaConf.load(config_path)
                self.assertTrue(
                    self._config_has_obs_term(
                        config,
                        term_name="ref_motion_filter_cutoff_hz",
                    )
                )

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

    @staticmethod
    def _config_has_obs_term(config, term_name: str) -> bool:
        for group_cfg in config.obs.obs_groups.values():
            for term_cfg in group_cfg.atomic_obs_list:
                for obs_name, obs_value in term_cfg.items():
                    if obs_name == term_name:
                        return True
                    if obs_value.get("func") == term_name:
                        return True
        return False
