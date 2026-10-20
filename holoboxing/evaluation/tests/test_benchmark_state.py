"""Fixed-state inference timing must not initialize a contact solver."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from holoboxing.evaluation.runtime.benchmark_state import BenchmarkState


class BenchmarkStateTest(unittest.TestCase):
    def setUp(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size="0.1" mass="1"/>'
            '<body pos="0.1 0 0.2"><joint axis="0 1 0"/>'
            '<geom type="capsule" size="0.02 0.15" mass="0.3"/>'
            '</body></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        data.qpos[:] = [0.2, -0.3, 0.9, 0.8, 0.0, 0.0, 0.6, 0.4]
        data.qvel[:] = np.arange(model.nv) * 0.11 - 0.2
        self.scene = SimpleNamespace(model=model, data=data)

    def test_nonzero_state_matches_independent_full_forward(self):
        reference = mujoco.MjData(self.scene.model)
        reference.qpos[:] = self.scene.data.qpos
        reference.qvel[:] = self.scene.data.qvel
        mujoco.mj_forward(self.scene.model, reference)
        # CPU transport tests do not claim CUDA acceptance; the same fixture
        # also runs through the actual CUDA benchmark during runtime validation.
        with patch(
            'holoboxing.evaluation.runtime.benchmark_state.require_gpu',
            return_value='cpu',
        ):
            state = BenchmarkState(self.scene, 'cuda:0')
        for name in ('qpos', 'qvel', 'xpos', 'xquat', 'xipos', 'cvel', 'subtree_com'):
            np.testing.assert_allclose(
                getattr(state.data, name).numpy()[0],
                getattr(reference, name),
                atol=1e-6,
                rtol=1e-6,
            )
        self.assertGreater(np.max(np.abs(state.data.cvel.numpy())), 0.1)

    def test_reset_replaces_buffers_with_current_state_and_cannot_step(self):
        with patch(
            'holoboxing.evaluation.runtime.benchmark_state.require_gpu',
            return_value='cpu',
        ):
            state = BenchmarkState(self.scene, 'cuda:0')
        old = state.data
        self.scene.data.qpos[0] += 0.7
        self.scene.data.qvel[:] *= -2
        state.forward_from_host()
        self.assertIsNot(state.data, old)
        np.testing.assert_allclose(state.data.qpos.numpy()[0], self.scene.data.qpos)
        np.testing.assert_allclose(state.data.qvel.numpy()[0], self.scene.data.qvel)
        with self.assertRaisesRegex(RuntimeError, 'cannot step physics'):
            state.frame()
        self.assertEqual(state.steps, 0)


if __name__ == '__main__':
    unittest.main()
