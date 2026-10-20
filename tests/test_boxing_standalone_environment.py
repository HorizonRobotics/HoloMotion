"""The boxing MJCF workflow must not import unrelated simulator stacks."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class BoxingStandaloneEnvironmentTest(unittest.TestCase):
    def test_motion_inputs_require_current_kinematics_and_resource_identity(
        self,
    ):
        import hashlib
        import json
        import tempfile

        import numpy as np

        from holoboxing.data.contract import ROBOT_KINEMATICS
        from holoboxing.data.records import DATA_FORMAT
        from holoboxing.evaluation.runtime.model_contract import CONTRACT
        from holoboxing.training.rsl_rl.modules.amp import motion_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "clip.npz"
            np.savez(clip, obs=np.zeros((1, 1097), dtype=np.float32))
            identity = {
                "robot_kinematics": ROBOT_KINEMATICS,
                "resource_manifest_sha256": "current-resources",
            }
            manifest = {
                "format": DATA_FORMAT,
                "policy_contract": CONTRACT,
                "control_hz": 50,
                "asset_identity": identity,
                "records": [
                    {
                        "path": clip.name,
                        "samples": 1,
                        "bytes": clip.stat().st_size,
                        "sha256": hashlib.sha256(
                            clip.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(
                motion_files(
                    root, resource_manifest_sha256="current-resources"
                ),
                [clip],
            )
            with self.assertRaisesRegex(ValueError, "training assets differ"):
                motion_files(root, resource_manifest_sha256="other-resources")
            identity["robot_kinematics"] = "competition_xml_v1"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "competition XML"):
                motion_files(
                    root, resource_manifest_sha256="current-resources"
                )

    def test_training_data_and_export_without_optional_stacks(self):
        script = textwrap.dedent("""
            import importlib.abc
            import sys

            class RejectOptionalStack(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in {
                        "isaacsim", "isaaclab", "omni", "newton", "smplx",
                        "pytorch_kinematics", "matplotlib", "trimesh",
                    }:
                        raise ImportError("Unnecessary boxing dependency: " + fullname)

            sys.meta_path.insert(0, RejectOptionalStack())
            import holoboxing.training.train_sl
            import holoboxing.training.train_rl
            import holoboxing.training.export
            import holoboxing.training.envs.mujoco_env
            from holoboxing.data.prepare import prepare_dataset, verify_dataset
            from holoboxing.data.robot import robot_file
            from holomotion.src.training.data_production.robot_h5 import RobotH5ShardWriter
            from holomotion.src.training.h5_dataloader import Hdf5RootDofDataset
            from holomotion.src.motion_tracking.link_fk import SharedLinkFK

            import mujoco
            import numpy as np
            import tempfile
            from pathlib import Path
            import torch

            xml = '<mujoco><worldbody><body name="pelvis">' + \
                  '<body name="tip" pos="0 0 1"><joint name="bend" axis="0 1 0"/>' + \
                  '<geom type="sphere" size="0.1" mass="1"/>' + \
                  '</body></body></worldbody></mujoco>'
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "robot.xml"
                path.write_text(xml)
                scene = Path(directory) / "scene.xml"
                scene.write_text(
                    '<mujoco><asset><model name="robot" file="robot.xml"/></asset>'
                    '<worldbody><attach model="robot" prefix="p1_"/></worldbody></mujoco>'
                )
                with robot_file(scene) as kinematic_xml:
                    fk = SharedLinkFK(str(kinematic_xml), "cpu")
                positions, orientations = fk(torch.tensor([[0.35]]))
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                data.qpos[:] = 0.35
                mujoco.mj_forward(model, data)
                for index, name in enumerate(fk.body_names):
                    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    np.testing.assert_allclose(positions[0, index], data.xpos[body], atol=1e-6)
                    np.testing.assert_allclose(orientations[0, index], data.xquat[body], atol=1e-6)
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
