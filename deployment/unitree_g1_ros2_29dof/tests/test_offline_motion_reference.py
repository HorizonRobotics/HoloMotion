import unittest
from pathlib import Path

import numpy as np

from humanoid_policy.motion_clip_library import LoadedMotionClip
from humanoid_policy.offline_motion_reference import OfflineMotionReference
from humanoid_policy.offline_motion_reference import quat_rotate_inv_wxyz


def _make_clip(frames: int = 5, dofs: int = 3, bodies: int = 3) -> LoadedMotionClip:
    dof_pos = np.arange(frames * dofs, dtype=np.float32).reshape(frames, dofs)
    dof_vel = dof_pos + 100.0
    translation = np.zeros((frames, bodies, 3), dtype=np.float32)
    velocity = np.zeros((frames, bodies, 3), dtype=np.float32)
    angular_velocity = np.zeros((frames, bodies, 3), dtype=np.float32)
    rotation = np.zeros((frames, bodies, 4), dtype=np.float32)
    rotation[..., 3] = 1.0

    for frame in range(frames):
        translation[frame, 0] = [frame, 0.0, 0.5 + frame]
        translation[frame, 1] = [frame + 1.0, 2.0, 3.0]
        translation[frame, 2] = [frame + 4.0, 5.0, 6.0]
        velocity[frame, 0] = [frame + 0.1, frame + 0.2, frame + 0.3]
        angular_velocity[frame, 0] = [frame + 1.1, frame + 1.2, frame + 1.3]

    return LoadedMotionClip(
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        global_translation=translation,
        global_rotation_quat=rotation,
        global_velocity=velocity,
        global_angular_velocity=angular_velocity,
        n_frames=frames,
    )


class OfflineMotionReferenceTest(unittest.TestCase):
    def test_policy_node_initializes_future_frame_count_before_offline_reference(self):
        evaluator_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "humanoid_policy"
            / "observation_evaluator.py"
        )
        source = evaluator_path.read_text()

        n_fut_init = source.index("self.n_fut_frames_int = int(")
        offline_ref_init = source.index("self._offline_reference = OfflineMotionReference(")

        self.assertLess(n_fut_init, offline_ref_init)

    def test_current_and_future_dof_reference_match_phase_3b_layout(self):
        clip = _make_clip()
        ref_to_onnx = [2, 0, 1]
        reference = OfflineMotionReference(
            n_fut_frames=3,
            num_actions=3,
            ref_to_onnx=ref_to_onnx,
        )
        reference.set_clip(clip)

        self.assertEqual(reference.current_frame_idx(1), 1)
        np.testing.assert_array_equal(reference.future_frame_indices(1), [2, 3, 4])
        np.testing.assert_array_equal(reference.ref_dof_pos_onnx_order(1), [5, 3, 4])
        np.testing.assert_array_equal(reference.ref_dof_vel_onnx_order(1), [105, 103, 104])

        expected_pos_fut = clip.dof_pos[[2, 3, 4]].T
        expected_pos_fut = expected_pos_fut[ref_to_onnx, :].transpose(1, 0).reshape(-1)
        np.testing.assert_array_equal(
            reference.obs_ref_dof_pos_fut(1),
            expected_pos_fut.astype(np.float32),
        )

    def test_reference_dof_count_does_not_depend_on_action_count(self):
        clip = _make_clip(dofs=4)
        ref_to_onnx = [3, 1, 2]
        reference = OfflineMotionReference(
            n_fut_frames=2,
            num_actions=3,
            ref_to_onnx=ref_to_onnx,
            reference_dof_count=4,
        )
        reference.set_clip(clip)

        expected_pos_fut = clip.dof_pos[[1, 2]].T
        expected_pos_fut = expected_pos_fut[ref_to_onnx, :].transpose(1, 0).reshape(-1)
        np.testing.assert_array_equal(
            reference.obs_ref_dof_pos_fut(0),
            expected_pos_fut.astype(np.float32),
        )

    def test_root_velocity_gravity_and_keybody_reference(self):
        clip = _make_clip()
        reference = OfflineMotionReference(
            n_fut_frames=2,
            num_actions=3,
            ref_to_onnx=[0, 1, 2],
        )
        reference.set_clip(clip)

        np.testing.assert_allclose(reference.obs_ref_root_pos_cur(1), [1.0, 0.0, 1.5])
        np.testing.assert_allclose(reference.obs_ref_root_height_cur(1), 1.5)
        np.testing.assert_allclose(reference.obs_ref_gravity_projection_cur(1), [0.0, 0.0, -1.0])
        np.testing.assert_allclose(reference.obs_ref_base_linvel_cur(1), [1.1, 1.2, 1.3])
        np.testing.assert_allclose(reference.obs_ref_base_angvel_cur(1), [2.1, 2.2, 2.3])

        np.testing.assert_allclose(reference.obs_ref_root_height_fut(1), [2.5, 3.5])
        np.testing.assert_allclose(
            reference.obs_ref_root_pos_fut(1),
            [2.0, 0.0, 2.5, 3.0, 0.0, 3.5],
        )
        np.testing.assert_allclose(
            reference.obs_ref_keybody_rel_pos_cur(1, np.array([1, 2], dtype=np.int64)),
            [1.0, 2.0, 1.5, 4.0, 5.0, 4.5],
        )

    def test_future_reference_clamps_to_last_frame(self):
        clip = _make_clip()
        reference = OfflineMotionReference(
            n_fut_frames=4,
            num_actions=3,
            ref_to_onnx=[0, 1, 2],
        )
        reference.set_clip(clip)

        np.testing.assert_array_equal(reference.future_frame_indices(3), [4, 4, 4, 4])
        np.testing.assert_allclose(reference.obs_ref_root_height_fut(3), [4.5, 4.5, 4.5, 4.5])

    def test_zero_future_frames_return_empty_arrays(self):
        clip = _make_clip()
        reference = OfflineMotionReference(
            n_fut_frames=0,
            num_actions=3,
            ref_to_onnx=[0, 1, 2],
        )
        reference.set_clip(clip)

        self.assertEqual(reference.obs_ref_dof_pos_fut(1).shape, (0,))
        self.assertEqual(reference.obs_ref_root_pos_fut(1).shape, (0,))
        self.assertEqual(reference.obs_ref_keybody_rel_pos_fut(1, np.array([1])).shape, (0,))

    def test_inverse_quaternion_rotation_uses_wxyz_order(self):
        angle = np.pi / 2.0
        q_wxyz = np.array(
            [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
            dtype=np.float32,
        )
        v_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        np.testing.assert_allclose(
            quat_rotate_inv_wxyz(q_wxyz, v_world),
            [0.0, -1.0, 0.0],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
