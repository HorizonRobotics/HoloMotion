import unittest
from types import SimpleNamespace

from humanoid_policy.config import DeploymentConfig
from humanoid_policy.launch_profile import drop_legacy_policy_fields


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
