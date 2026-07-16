import unittest

import numpy as np

from holomotion.src.motion_tracking.reference_observation import (
    derive_reference_kinematics_numpy,
    derive_reference_kinematics_torch,
    pack_reference_qpos,
)


class ReferenceObservationTests(unittest.TestCase):
    def test_numpy_derives_centered_velocities_and_local_state(self):
        frames = 5
        time = np.arange(frames, dtype=np.float32) * 0.02
        root_pos = np.zeros((frames, 3), dtype=np.float32)
        root_pos[:, 0] = time
        root_quat = np.zeros((frames, 4), dtype=np.float32)
        root_quat[:, 0] = 1.0
        dof_pos = np.broadcast_to(time[:, None], (frames, 29)).copy()
        qpos = pack_reference_qpos(root_pos, root_quat, dof_pos)

        result = derive_reference_kinematics_numpy(qpos, fps=50.0)

        np.testing.assert_allclose(result.dof_vel, 1.0, atol=1.0e-6)
        np.testing.assert_allclose(
            result.root_linvel_world[:, 0], 1.0, atol=1.0e-6
        )
        np.testing.assert_allclose(
            result.root_linvel_local[:, 0], 1.0, atol=1.0e-6
        )
        np.testing.assert_allclose(result.root_angvel_world, 0.0, atol=0.0)
        np.testing.assert_allclose(
            result.projected_gravity,
            np.tile(
                np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                (frames, 1),
            ),
            atol=0.0,
        )

    def test_torch_gpu_matches_numpy(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        rng = np.random.default_rng(7)
        root_pos = rng.normal(size=(2, 16, 3)).astype(np.float32)
        root_quat = rng.normal(size=(2, 16, 4)).astype(np.float32)
        root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)
        dof_pos = rng.normal(size=(2, 16, 29)).astype(np.float32)
        qpos = pack_reference_qpos(root_pos, root_quat, dof_pos)

        expected = derive_reference_kinematics_numpy(qpos, fps=50.0)
        actual = derive_reference_kinematics_torch(
            torch.from_numpy(qpos).cuda(), fps=50.0
        )
        torch.cuda.synchronize()

        for name in expected.__dataclass_fields__:
            np.testing.assert_allclose(
                getattr(actual, name).cpu().numpy(),
                getattr(expected, name),
                atol=1.0e-5,
                rtol=1.0e-5,
            )


if __name__ == "__main__":
    unittest.main()
