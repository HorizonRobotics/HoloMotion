"""Real MuJoCo-Warp state parity; no policy weights or simulated GPU backend.

Run with the pinned evaluation build dependencies and BOXING_TEST_DEVICE=cuda:0.
CPU-only environments explicitly skip these tests.
"""

import importlib.util
import os
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

from holoboxing.evaluation.runtime.lean_runtime import Bound
from holoboxing.evaluation.runtime.observation_fk import CurrentObservationFK

XML = """
<mujoco>
  <option timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size=".1" pos=".08 .02 .03" mass="1"/>
      <body pos=".2 .03 0">
        <joint name="hinge" axis="0 1 0" damping=".2"/>
        <geom type="capsule" size=".03" fromto="0 0 0 .3 0 .1"
              mass=".3" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
</mujoco>
"""


def snapshot_arrays(value, prefix=""):
    """Include contact, constraint and solver arrays, not just joint state."""
    if isinstance(value, wp.array):
        return {prefix: value.numpy().copy()}
    arrays = {}
    for name, child in vars(value).items():
        path = prefix + "." + name
        if isinstance(child, wp.array) or (
            hasattr(child, "__dict__")
            and type(child).__module__.startswith("mujoco_warp")
        ):
            arrays.update(snapshot_arrays(child, path))
    return arrays


class ObservationStateGPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not importlib.util.find_spec("mujoco_warp"):
            raise unittest.SkipTest("MuJoCo-Warp is not installed")
        if not wp.is_cuda_available():
            raise unittest.SkipTest("A real CUDA device is required")
        import mujoco_warp
        import torch

        from holoboxing.training.envs.mujoco_physics import CompetitionPhysics

        cls.mjw = mujoco_warp
        cls.torch = torch
        cls.training_class = CompetitionPhysics
        cls.device = os.environ.get("BOXING_TEST_DEVICE", "cuda:0")

    def setUp(self):
        self.addCleanup(
            self.torch.cuda.set_device, self.torch.cuda.current_device()
        )
        self.torch.cuda.set_device(self.device)
        # Restore Warp's device and stream even when setup or a test fails.
        # Leaving Torch's default stream active breaks later Warp captures.
        self.enterContext(
            wp.ScopedStream(
                wp.stream_from_torch(
                    self.torch.cuda.current_stream(self.device)
                )
            )
        )
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.gpu_model = self.mjw.put_model(self.model)
        self.scene = SimpleNamespace(model=self.model)
        self.allocate()
        self.fk = self.scene._observation_fk = CurrentObservationFK(self.scene)
        # Exercise the production training state() on the same physical data.
        # Only task-specific initialization/rewards are outside this fixture.
        self.training = self.training_class.__new__(self.training_class)
        self.training.model = self.gpu_model
        self.training.data = self.physics
        self.training.fk = self.mjw.put_data(
            self.model, self.initial, nconmax=64, njmax=128
        )
        self.training.body_ids = self.torch.tensor(
            [[1, 2]], device=self.device
        )
        self.training.root_ids = self.torch.tensor(
            [[1, 1]], device=self.device
        )
        self.training.qi = self.torch.tensor([[7]], device=self.device)
        self.training.vi = self.torch.tensor([[6]], device=self.device)
        self.training.joint_acceleration = self.torch.zeros(
            (1, 1, 1), device=self.device
        )

    def allocate(self, velocity=None, contact=False):
        self.initial = mujoco.MjData(self.model)
        self.initial.qpos[2] = -0.02 if contact else 1.0
        quaternion = np.array([0.89, 0.2, -0.3, 0.1])
        self.initial.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        self.initial.qpos[7] = 0.6
        self.initial.qvel[:] = (
            [0.9, -0.5, 0.3, 0.2, 0.6, -0.4, 1.2]
            if velocity is None
            else velocity
        )
        mujoco.mj_forward(self.model, self.initial)
        self.physics = self.mjw.put_data(
            self.model, self.initial, nconmax=64, njmax=128
        )
        self.mjw.forward(self.gpu_model, self.physics)
        self.scene._gpu_physics = SimpleNamespace(data=self.physics)
        if hasattr(self, "training"):
            self.training.data = self.physics

    def assert_reference(self):
        self.fk.refresh()
        reference = mujoco.MjData(self.model)
        reference.qpos[:] = self.physics.qpos.numpy()[0]
        reference.qvel[:] = self.physics.qvel.numpy()[0]
        mujoco.mj_kinematics(self.model, reference)
        mujoco.mj_comPos(self.model, reference)
        positions = self.fk.positions.numpy()
        cvel = self.fk.cvel.numpy()
        centers = self.fk.subtree_com.numpy()[self.model.body_rootid]
        velocity = cvel[:, 3:] + np.cross(cvel[:, :3], positions - centers)
        trained = self.training.state()
        self.torch.cuda.synchronize(self.device)
        np.testing.assert_allclose(positions, reference.xpos, atol=3e-6)
        np.testing.assert_allclose(
            self.fk.rotations.numpy(), reference.xquat, atol=3e-6
        )
        np.testing.assert_array_equal(self.fk.qvel.numpy(), reference.qvel)
        for body in (1, 2):
            linear = np.zeros((3, self.model.nv))
            angular = np.zeros_like(linear)
            mujoco.mj_jacBody(self.model, reference, linear, angular, body)
            np.testing.assert_allclose(
                velocity[body],
                linear @ reference.qvel,
                atol=3e-5,
                rtol=3e-5,
            )
            np.testing.assert_allclose(
                cvel[body, :3],
                angular @ reference.qvel,
                atol=3e-5,
                rtol=3e-5,
            )
        for name, expected in (
            ("position", positions[1:3]),
            ("quaternion", self.fk.rotations.numpy()[1:3]),
            ("velocity", velocity[1:3]),
            ("angular", cvel[1:3, :3]),
        ):
            np.testing.assert_allclose(
                trained[name].cpu().numpy()[0, 0],
                expected,
                atol=3e-5,
                rtol=3e-5,
            )
        # The old physical caches must actually differ after integration.
        stale = self.physics.cvel.numpy()[0]
        old_com = self.physics.subtree_com.numpy()[0][self.model.body_rootid]
        mixed = stale[:, 3:] + np.cross(stale[:, :3], positions - old_com)
        return float(np.max(np.abs(mixed - velocity)))

    def test_substeps_match_training_and_independent_jacobians(self):
        largest_old_error = 0.0
        for name, velocity in (
            ("translation", [0.9, -0.5, 0.3, 0, 0, 0, 0]),
            ("rotation", [0, 0, 0, 0.2, 0.6, -0.4, 0]),
            ("joint", [0, 0, 0, 0, 0, 0, 1.2]),
            ("combined", [0.9, -0.5, 0.3, 0.2, 0.6, -0.4, 1.2]),
        ):
            with self.subTest(motion=name):
                self.allocate(velocity)
                largest_old_error = max(
                    largest_old_error, self.assert_reference()
                )
                for step in range(12):
                    self.mjw.step(self.gpu_model, self.physics)
                    if step in (0, 3, 11):
                        largest_old_error = max(
                            largest_old_error, self.assert_reference()
                        )
        self.assertGreater(largest_old_error, 0.01)

    def test_refresh_preserves_contact_solve_and_fixed_control_trajectory(
        self,
    ):
        self.allocate(contact=True)
        baseline = self.mjw.put_data(
            self.model, self.initial, nconmax=64, njmax=128
        )
        self.mjw.forward(self.gpu_model, baseline)
        self.assertGreater(int(self.physics.nacon.numpy()[0]), 0)
        self.physics.ctrl.fill_(0.18)
        baseline.ctrl.fill_(0.18)
        for _ in range(8):
            self.mjw.step(self.gpu_model, self.physics)
            self.mjw.step(self.gpu_model, baseline)
            before = snapshot_arrays(self.physics)
            self.fk.refresh()
            after = snapshot_arrays(self.physics)
            self.assertGreater(len(before), 50)
            self.assertEqual(before.keys(), after.keys())
            for name in before:
                np.testing.assert_array_equal(
                    after[name], before[name], err_msg=name
                )
            for name in (
                "qpos",
                "qvel",
                "qacc",
                "qacc_warmstart",
                "time",
                "ctrl",
                "nefc",
            ):
                np.testing.assert_array_equal(
                    getattr(self.physics, name).numpy(),
                    getattr(baseline, name).numpy(),
                    err_msg=name,
                )
            for data, original in (
                (self.physics.contact, baseline.contact),
                (self.physics.efc, baseline.efc),
            ):
                for name in ("pos", "geom", "force"):
                    if hasattr(data, name):
                        np.testing.assert_array_equal(
                            getattr(data, name).numpy(),
                            getattr(original, name).numpy(),
                            err_msg=name,
                        )

    def test_reset_reallocation_keeps_captured_and_bound_inputs_current(self):
        names = ("positions", "rotations", "qvel", "cvel", "subtree_com")
        pointers = {name: getattr(self.fk, name).ptr for name in names}
        outputs = {
            name: wp.empty_like(getattr(self.fk, name)) for name in names
        }
        stream = wp.Stream(self.device)
        with wp.ScopedStream(stream):
            with wp.ScopedCapture() as capture:
                for name in names:
                    wp.copy(outputs[name], getattr(self.fk, name))
        destinations = [self.device]
        if wp.get_cuda_device_count() > 1:
            destinations.append(
                "cuda:0" if self.device != "cuda:0" else "cuda:1"
            )
        for reset in (False, True):
            if reset:
                old_pointer = self.physics.qpos.ptr
                self.allocate([-0.3, 0.8, -0.4, -0.7, 0.1, 0.9, -1.1])
                self.assertNotEqual(self.physics.qpos.ptr, old_pointer)
            self.mjw.step(self.gpu_model, self.physics)
            self.assert_reference()
            wp.capture_launch(capture.graph, stream=stream)
            wp.synchronize_stream(stream)
            for name in names:
                self.assertEqual(getattr(self.fk, name).ptr, pointers[name])
                np.testing.assert_array_equal(
                    outputs[name].numpy(), getattr(self.fk, name).numpy()
                )
            for device in destinations:
                bound = Bound.__new__(Bound)
                bound.inputs = {
                    name: wp.zeros(shape, device=device)
                    for name, shape in (
                        ("qpos", (1, self.model.nq)),
                        ("qvel", (1, self.model.nv)),
                        ("xpos", (self.model.nbody, 3)),
                        ("xquat", (self.model.nbody, 4)),
                        ("cvel", (self.model.nbody, 6)),
                        ("com", (self.model.nbody, 3)),
                    )
                }
                bound.physical_inputs(self.scene)
                wp.synchronize_device(device)
                for name, source in (
                    ("xpos", "positions"),
                    ("xquat", "rotations"),
                    ("qvel", "qvel"),
                    ("cvel", "cvel"),
                    ("com", "subtree_com"),
                ):
                    np.testing.assert_array_equal(
                        bound.inputs[name].numpy().reshape(-1),
                        getattr(self.fk, source).numpy().reshape(-1),
                    )


if __name__ == "__main__":
    unittest.main()
