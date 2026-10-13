import unittest
from types import SimpleNamespace

from omegaconf import OmegaConf
from humanoid_policy.config import motion_reference_horizon
from humanoid_policy.config import DeploymentConfig
from humanoid_policy.launch_profile import drop_legacy_policy_fields


def test_v2_horizon_does_not_evaluate_training_only_resolver():
    unresolved = OmegaConf.create({
        "link_observation": {"schema_version": "g1_link_xz6d_v1", "frame_offsets": [1, 2, 3, 4, 5]},
        "obs": {"n_fut_frames": "${eval:'len(${link_observation.frame_offsets})'}"},
    })
    assert motion_reference_horizon(unresolved) == 5
    legacy = OmegaConf.create({"obs": {"n_fut_frames": 3}})
    with unittest.TestCase().assertRaisesRegex(ValueError, 'only HoloMotion v2.0'):
        motion_reference_horizon(legacy)


class FakeNode:
    def __init__(self, values):
        self.values = dict(values)

    def declare_parameter(self, name, default):
        return SimpleNamespace(value=self.values.get(name, default))


class DeploymentConfigTest(unittest.TestCase):
    def test_reads_only_user_facing_inference_backend_field(self):
        config = DeploymentConfig.from_node(
            FakeNode(
                {
                    "config_path": "/tmp/g1_29dof_holomotion.yaml",
                    "inference_backend": "tensorrt",
                }
            )
        )

        self.assertEqual(config.inference_backend, "tensorrt")

    def test_rejects_unknown_inference_backend(self):
        with self.assertRaisesRegex(ValueError, "inference_backend"):
            DeploymentConfig.from_node(
                FakeNode(
                    {
                        "config_path": "/tmp/g1_29dof_holomotion.yaml",
                        "inference_backend": "bad",
                    }
                )
            )

    def test_drops_hidden_tensorrt_policy_options(self):
        profile = {
            "policy": {
                "inference_backend": "tensorrt",
                "inference_device_id": 1,
                "tensorrt_fp16_enable": False,
                "tensorrt_engine_cache_enable": False,
                "tensorrt_engine_cache_path": "/tmp/cache",
            }
        }

        drop_legacy_policy_fields(profile)

        self.assertEqual(profile["policy"], {"inference_backend": "tensorrt"})


if __name__ == "__main__":
    unittest.main()
