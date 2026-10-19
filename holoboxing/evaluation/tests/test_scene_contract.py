"""Reject independent robot/arena definitions and verify URDF-derived XMLs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from holoboxing.evaluation.resources import ResourceTree
from holoboxing.evaluation.robot_models import export_xml, verify_robots
from holoboxing.evaluation.tests.resource_fixtures import (
    make_resources,
    resign,
)


class SceneContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = make_resources(temporary.name)

    def test_canonical_urdf_and_saved_xml_have_identical_physical_parameters(
        self,
    ):
        report = verify_robots(ResourceTree(self.root))
        self.assertEqual(report["red"]["max_parameter_difference"], 0)
        self.assertEqual(report["blue"]["max_parameter_difference"], 0)
        self.assertEqual(
            (self.root / "red.xml").read_text(),
            export_xml(self.root / "red.urdf"),
        )

    def test_rounded_joint_limit_is_rejected(self):
        path = self.root / "red.xml"
        root = ET.parse(path).getroot()
        root.find(".//joint[@name='left_wrist_yaw_joint']").set(
            "range", "-1.61443 1.61443"
        )
        ET.ElementTree(root).write(path)
        resign(self.root)
        with self.assertRaisesRegex(ValueError, "jnt_range"):
            verify_robots(ResourceTree(self.root))

    def test_evaluator_cannot_attach_another_robot(self):
        path = self.root / "scene.xml"
        path.write_text(
            path.read_text().replace('model="blue" body=', 'model="red" body=')
        )
        resign(self.root)
        with self.assertRaisesRegex(ValueError, "robot differ: blue"):
            ResourceTree(self.root)

    def test_replay_manifest_cannot_choose_another_robot(self):
        path = self.root / "manifest.json"
        record = json.loads(path.read_text())
        record["entries"]["blue"] = "red.xml"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "manifests differ"):
            ResourceTree(self.root)

    def test_legacy_model_and_bare_hand_mesh_cannot_be_bundled(self):
        for name, payload in (
            ("legacy.xml", b"<mujoco/>"),
            ("left_rubber_hand.STL", b"old mesh"),
        ):
            with self.subTest(name=name):
                record = json.loads((self.root / "resources.json").read_text())
                (self.root / name).write_bytes(payload)
                record["files"].append(dict(path=name))
                (self.root / "resources.json").write_text(json.dumps(record))
                resign(self.root)
                with self.assertRaisesRegex(ValueError, "must not be bundled"):
                    ResourceTree(self.root)
                record["files"].pop()
                (self.root / "resources.json").write_text(json.dumps(record))
                resign(self.root)

    def test_independent_arena_geometry_is_rejected(self):
        path = self.root / "scene.xml"
        path.write_text(
            path.read_text().replace('size="2.5 2.5 .5"', 'size="3 3 .5"')
        )
        resign(self.root)
        with self.assertRaisesRegex(ValueError, "arena differ"):
            ResourceTree(self.root)


class InstalledSceneTest(unittest.TestCase):
    def test_scene_and_player_use_identical_gloves_at_representative_poses(
        self,
    ):
        directory = Path(__file__).resolve().parents[2] / "assets"
        if not (directory / "resources.json").is_file():
            self.skipTest("The optional robot asset bundle is not installed")
        from holoboxing.evaluation.runtime.scene import BoxingMatchScene
        from holoboxing.data.robot import robot_spec

        resources = ResourceTree(directory)
        with (
            patch(
                "holoboxing.evaluation.runtime.scene.require_gpu",
                return_value="cuda:0",
            ),
            patch("holoboxing.evaluation.runtime.gpu_physics.ResidentPhysics"),
        ):
            scene = BoxingMatchScene(model_path=resources.scene)
        self.assertEqual(
            (scene.model.nq, scene.model.nv, scene.model.nu), (72, 70, 58)
        )
        self.assertEqual(len(scene.body_names), 38)
        rng = np.random.default_rng(41)
        for side, prefix in (("red", "p1_"), ("blue", "p2_")):
            model = mujoco.MjModel.from_xml_path(
                str(resources.path(resources.entries[side]))
            )
            extracted = robot_spec(resources.scene, side).compile()
            np.testing.assert_array_equal(extracted.body_pos, model.body_pos)
            np.testing.assert_array_equal(
                extracted.body_inertia, model.body_inertia
            )
            data = mujoco.MjData(model)
            for _ in range(8):
                for index in range(model.njnt):
                    joint = model.joint(index)
                    target = scene.model.joint(prefix + joint.name).id
                    width = (
                        7
                        if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
                        else 1
                    )
                    value = (
                        np.array([0.2, -0.3, 0.8, 1, 0, 0, 0])
                        if width == 7
                        else rng.uniform(-0.3, 0.3, 1)
                    )
                    data.qpos[
                        model.jnt_qposadr[index] : model.jnt_qposadr[index]
                        + width
                    ] = value
                    scene.data.qpos[
                        scene.model.jnt_qposadr[
                            target
                        ] : scene.model.jnt_qposadr[target] + width
                    ] = value
                mujoco.mj_forward(model, data)
                mujoco.mj_forward(scene.model, scene.data)
                for hand in ("left", "right"):
                    for kind in ("visual", "collision"):
                        name = f"{hand}_glove_{kind}"
                        actual = scene.model.geom(prefix + name).id
                        expected = model.geom(name).id
                        np.testing.assert_array_equal(
                            scene.data.geom_xpos[actual],
                            data.geom_xpos[expected],
                        )
                        np.testing.assert_array_equal(
                            scene.data.geom_xmat[actual],
                            data.geom_xmat[expected],
                        )
                    geom = scene.model.geom(
                        prefix + f"{hand}_glove_collision"
                    ).id
                    self.assertEqual(
                        scene._strike_geom[geom],
                        (0 if side == "red" else 1, f"{hand}_hand"),
                    )
                    wrist = scene.model.body(
                        prefix + f"{hand}_wrist_yaw_link"
                    ).id
                    self.assertEqual(
                        scene.model.body_weldid[scene.model.geom_bodyid[geom]],
                        wrist,
                    )
        verify_robots(resources)


if __name__ == "__main__":
    unittest.main()
