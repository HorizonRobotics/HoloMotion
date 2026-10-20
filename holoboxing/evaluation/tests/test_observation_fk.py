"""Current-state FK and compiled-runtime routing without policy weights."""

from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import warp as wp

from holoboxing.evaluation.runtime.observation_fk import CurrentObservationFK


class ObservationFKTest(unittest.TestCase):
    def setUp(self):
        model = mujoco.MjModel.from_xml_string("""
          <mujoco><worldbody><body pos="0 0 1"><freejoint/>
            <geom type="sphere" size=".1"/>
            <body pos=".2 0 0"><joint axis="0 1 0"/>
              <geom type="capsule" size=".03" fromto="0 0 0 .3 0 0"/>
            </body>
          </body></worldbody></mujoco>""")
        self.original = mujoco.MjData(model)
        self.original.qvel[:] = [0.8, -0.3, 0.2, 0.4, -0.6, 0.5, 1.1]
        mujoco.mj_forward(model, self.original)
        state = SimpleNamespace(
            qpos=wp.array(self.original.qpos[None], dtype=float, device="cpu"),
            qvel=wp.array(self.original.qvel[None], dtype=float, device="cpu"),
            xpos=wp.array(self.original.xpos, dtype=float, device="cpu"),
            xquat=wp.array(self.original.xquat, dtype=float, device="cpu"),
            cvel=wp.array(self.original.cvel, dtype=float, device="cpu"),
            subtree_com=wp.array(
                self.original.subtree_com, dtype=float, device="cpu"
            ),
        )
        self.scene = SimpleNamespace(
            model=model,
            data=self.original,
            _gpu_physics=SimpleNamespace(data=state),
        )
        self.fk = self.scene._observation_fk = CurrentObservationFK(self.scene)

    def reference_state(self):
        expected = mujoco.MjData(self.scene.model)
        expected.qpos[:] = self.scene._gpu_physics.data.qpos.numpy()[0]
        expected.qvel[:] = self.scene._gpu_physics.data.qvel.numpy()[0]
        mujoco.mj_kinematics(self.scene.model, expected)
        mujoco.mj_comPos(self.scene.model, expected)
        mujoco.mj_comVel(self.scene.model, expected)
        return expected

    def assert_current(self):
        expected = self.reference_state()
        np.testing.assert_allclose(
            self.fk.positions.numpy(), expected.xpos, atol=1e-6
        )
        np.testing.assert_allclose(
            self.fk.rotations.numpy(), expected.xquat, atol=1e-6
        )

    def test_current_joints_replace_stale_poses_without_writing_physics(self):
        state = self.scene._gpu_physics.data
        before = {
            n: getattr(state, n).numpy().copy()
            for n in ("xpos", "xquat", "cvel", "subtree_com")
        }
        qpos = state.qpos.numpy()
        qpos[0, 0] += 0.3
        qpos[0, -1] = 0.7
        state.qpos.assign(qpos)
        self.fk.refresh()
        self.assert_current()
        self.assertGreater(
            np.max(np.abs(self.fk.positions.numpy() - before["xpos"])), 0.1
        )
        for name, expected in before.items():
            np.testing.assert_array_equal(
                getattr(state, name).numpy(), expected
            )
        np.testing.assert_array_equal(
            self.original.qpos, self.scene.model.qpos0
        )

    def test_reset_reallocation_preserves_observation_output_addresses(self):
        buffers = ("positions", "rotations", "qvel", "cvel", "subtree_com")
        pointers = {name: getattr(self.fk, name).ptr for name in buffers}
        replacement = self.scene._gpu_physics.data.qpos.numpy()
        replacement[0, 1] += 0.4
        velocity = self.scene._gpu_physics.data.qvel.numpy() * -0.7
        self.scene._gpu_physics.data = SimpleNamespace(
            qpos=wp.array(replacement, dtype=float, device="cpu"),
            qvel=wp.array(velocity, dtype=float, device="cpu"),
        )
        self.fk.refresh()
        for name, pointer in pointers.items():
            self.assertEqual(getattr(self.fk, name).ptr, pointer)
        self.assert_current()
        np.testing.assert_array_equal(self.fk.qvel.numpy(), velocity[0])

    def test_current_velocities_match_body_origin_jacobians(self):
        state = self.scene._gpu_physics.data
        qpos = state.qpos.numpy()
        qpos[0, 3:7] = [np.cos(0.4), 0, np.sin(0.4), 0]
        qpos[0, -1] = 0.6
        state.qpos.assign(qpos)
        state.qvel.assign(state.qvel.numpy() * -1.3)
        self.fk.refresh()
        self.assert_current()
        expected = self.reference_state()
        cvel = self.fk.cvel.numpy()
        centers = self.fk.subtree_com.numpy()[self.scene.model.body_rootid]
        velocity = cvel[:, 3:] + np.cross(
            cvel[:, :3], self.fk.positions.numpy() - centers
        )
        for body in range(1, self.scene.model.nbody):
            linear = np.zeros((3, self.scene.model.nv))
            angular = np.zeros_like(linear)
            mujoco.mj_jacBody(self.scene.model, expected, linear, angular, body)
            np.testing.assert_allclose(
                velocity[body], linear @ expected.qvel, atol=2e-6, rtol=2e-6
            )
            np.testing.assert_allclose(
                cvel[body, :3], angular @ expected.qvel, atol=2e-6, rtol=2e-6
            )

    def test_velocity_refresh_does_not_depend_on_pose_changes(self):
        positions = self.fk.positions.numpy().copy()
        before = self.fk.cvel.numpy().copy()
        self.scene._gpu_physics.data.qvel.assign(
            self.scene._gpu_physics.data.qvel.numpy() * 2
        )
        self.fk.refresh()
        np.testing.assert_array_equal(self.fk.positions.numpy(), positions)
        np.testing.assert_allclose(self.fk.cvel.numpy(), 2 * before, atol=1e-6)

    def test_compiled_reference_root_uses_explicit_anchor_or_current_fk(self):
        from holoboxing.evaluation.runtime.lean_runtime import Bound

        self.fk.refresh()
        scene = SimpleNamespace(
            _gpu_physics=self.scene._gpu_physics,
            _observation_fk=self.fk,
            model=SimpleNamespace(
                body=lambda name: SimpleNamespace(
                    id=1 if name.startswith("p1_") else 2
                )
            ),
        )
        bound = Bound.__new__(Bound)
        bound.inputs = {"reference_root": wp.zeros((2, 3), device="cpu")}
        bound.physical_inputs(scene)
        expected = self.fk.positions.numpy()[[1, 2]]
        np.testing.assert_array_equal(
            bound.inputs["reference_root"].numpy(), expected
        )
        anchor = expected + np.array([0.3, -0.2, 0.1], np.float32)
        scene._source_reference_root_anchor = anchor
        bound.physical_inputs(scene)
        np.testing.assert_array_equal(
            bound.inputs["reference_root"].numpy(), anchor
        )
        del scene._source_reference_root_anchor
        bound.physical_inputs(scene)
        np.testing.assert_array_equal(
            bound.inputs["reference_root"].numpy(), expected
        )
        scene._source_reference_root_anchor = np.full((2, 3), np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            bound.physical_inputs(scene)

    def test_compiled_inputs_use_current_pose_and_velocity(self):
        from holoboxing.evaluation.runtime.lean_runtime import Bound

        state = self.scene._gpu_physics.data
        qpos = state.qpos.numpy()
        qpos[0, 0] += 0.5
        qpos[0, -1] = 0.7
        state.qpos.assign(qpos)
        state.qvel.assign(state.qvel.numpy() * -1.4)
        self.fk.refresh()
        bound = Bound.__new__(Bound)
        bound.inputs = {
            "qpos": wp.zeros(state.qpos.shape, device="cpu"),
            "qvel": wp.zeros(state.qvel.shape, device="cpu"),
            "xpos": wp.zeros(state.xpos.shape, device="cpu"),
            "xquat": wp.zeros(state.xquat.shape, device="cpu"),
            "cvel": wp.zeros(state.cvel.shape, device="cpu"),
            "com": wp.zeros(state.subtree_com.shape, device="cpu"),
        }
        bound.physical_inputs(self.scene)
        np.testing.assert_array_equal(bound.inputs["qpos"].numpy(), qpos)
        np.testing.assert_array_equal(
            bound.inputs["xpos"].numpy(), self.fk.positions.numpy()
        )
        np.testing.assert_array_equal(
            bound.inputs["xquat"].numpy(), self.fk.rotations.numpy()
        )
        self.assertFalse(
            np.array_equal(bound.inputs["xpos"].numpy(), state.xpos.numpy())
        )
        expected = self.reference_state()
        np.testing.assert_allclose(
            bound.inputs["cvel"].numpy(), expected.cvel, atol=2e-6
        )
        np.testing.assert_allclose(
            bound.inputs["com"].numpy(), expected.subtree_com, atol=2e-6
        )
        np.testing.assert_array_equal(
            bound.inputs["qvel"].numpy(), state.qvel.numpy()
        )


if __name__ == "__main__":
    unittest.main()
