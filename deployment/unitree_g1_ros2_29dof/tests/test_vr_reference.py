import unittest

import numpy as np

from humanoid_policy.reference_contract import LatestObsReferenceQueues
from humanoid_policy.reference_contract import flatten_vr_future_dof_pos_onnx
from humanoid_policy.vr_reference import VrLatestObsReference


def _obs(frame: int) -> np.ndarray:
    arr = np.zeros(65, dtype=np.float32)
    arr[:29] = frame + np.arange(29, dtype=np.float32)
    arr[29:58] = 100.0 + frame + np.arange(29, dtype=np.float32)
    arr[58:61] = [frame + 0.1, frame + 0.2, frame + 0.3]
    arr[61:65] = [1.0, frame + 0.4, frame + 0.5, frame + 0.6]
    return arr


class VrLatestObsReferenceTest(unittest.TestCase):
    def test_store_matches_phase_3b_contract_queues(self):
        runtime = VrLatestObsReference(n_fut_frames=3, num_actions=29, expected_dim=65)
        contract = LatestObsReferenceQueues(n_fut_frames=3, num_actions=29, expected_dim=65)

        for frame in range(5):
            arr = _obs(frame)
            self.assertTrue(runtime.store(arr, current_time=10.0 + frame, frame_index=frame))
            self.assertTrue(contract.store(arr, current_time=10.0 + frame, frame_index=frame))

        self.assertTrue(runtime.received)
        self.assertEqual(runtime.seen_frames, contract.seen_frames)
        self.assertEqual(runtime.last_obs_time, contract.last_obs_time)
        np.testing.assert_allclose(runtime.latest_obs, contract.latest_obs)
        np.testing.assert_allclose(runtime.dof_pos_queue, contract.dof_pos_queue)
        np.testing.assert_allclose(runtime.dof_vel_queue, contract.dof_vel_queue)
        np.testing.assert_allclose(runtime.root_pos_queue, contract.root_pos_queue)
        np.testing.assert_allclose(runtime.root_rot_queue, contract.root_rot_queue)
        np.testing.assert_array_equal(runtime.frame_idx_queue, contract.frame_idx_queue)
        np.testing.assert_allclose(runtime.prev_dof_pos, contract.prev_dof_pos)
        np.testing.assert_allclose(runtime.prev_dof_vel, contract.prev_dof_vel)
        np.testing.assert_allclose(runtime.prev_root_pos, contract.prev_root_pos)
        np.testing.assert_allclose(runtime.prev_root_rot, contract.prev_root_rot)
        self.assertEqual(runtime.prev_frame_idx, contract.prev_frame_idx)

    def test_current_reference_uses_previous_future_queue_frame(self):
        runtime = VrLatestObsReference(n_fut_frames=2, num_actions=29, expected_dim=65)
        runtime.store(_obs(0), current_time=1.0, frame_index=0)
        runtime.store(_obs(1), current_time=2.0, frame_index=1)
        runtime.store(_obs(2), current_time=3.0, frame_index=2)

        np.testing.assert_allclose(runtime.current_dof_pos(), _obs(0)[:29])
        np.testing.assert_allclose(runtime.current_dof_vel(), _obs(0)[29:58])
        np.testing.assert_allclose(runtime.current_root_pos(), _obs(0)[58:61])
        np.testing.assert_allclose(runtime.current_root_rot(), _obs(0)[61:65])

    def test_no_future_queue_uses_latest_obs_or_fallbacks(self):
        runtime = VrLatestObsReference(n_fut_frames=0, num_actions=29, expected_dim=65)
        fallback_pos = np.ones(29, dtype=np.float32)
        fallback_vel = np.ones(29, dtype=np.float32) * 2.0

        np.testing.assert_allclose(runtime.current_dof_pos(fallback_pos), fallback_pos)
        np.testing.assert_allclose(runtime.current_dof_vel(fallback_vel), fallback_vel)
        np.testing.assert_allclose(runtime.current_root_pos(), np.zeros(3, dtype=np.float32))
        self.assertIsNone(runtime.current_root_rot())

        runtime.store(_obs(4), current_time=4.0, frame_index=4)
        np.testing.assert_allclose(runtime.current_dof_pos(fallback_pos), _obs(4)[:29])
        np.testing.assert_allclose(runtime.current_dof_vel(fallback_vel), _obs(4)[29:58])
        np.testing.assert_allclose(runtime.current_root_pos(), _obs(4)[58:61])
        np.testing.assert_allclose(runtime.current_root_rot(), _obs(4)[61:65])

    def test_future_dof_and_root_obs_match_phase_3b_layout(self):
        runtime = VrLatestObsReference(n_fut_frames=3, num_actions=29, expected_dim=65)
        contract = LatestObsReferenceQueues(n_fut_frames=3, num_actions=29, expected_dim=65)
        for frame in range(5):
            runtime.store(_obs(frame), current_time=float(frame), frame_index=frame)
            contract.store(_obs(frame), current_time=float(frame), frame_index=frame)

        ref_to_onnx = np.arange(28, -1, -1, dtype=np.int64)
        pos_fut_buffer = np.zeros((29, 3), dtype=np.float32)

        dof_queue = np.stack([_obs(2)[:29], _obs(3)[:29], _obs(4)[:29]])
        root_queue = np.stack([_obs(2)[58:61], _obs(3)[58:61], _obs(4)[58:61]])

        np.testing.assert_allclose(
            runtime.obs_ref_dof_pos_fut(
                ref_to_onnx=ref_to_onnx,
                pos_fut_buffer=pos_fut_buffer,
                n_frames=3,
            ),
            flatten_vr_future_dof_pos_onnx(contract, 3, ref_to_onnx.tolist()),
        )
        np.testing.assert_allclose(
            runtime.obs_ref_root_height_fut(n_frames=3),
            root_queue[:, 2].reshape(-1).astype(np.float32),
        )
        np.testing.assert_allclose(
            runtime.obs_ref_root_pos_fut(n_frames=3),
            root_queue.reshape(-1).astype(np.float32),
        )
        np.testing.assert_allclose(
            runtime.obs_ref_dof_pos_fut(
                ref_to_onnx=ref_to_onnx,
                pos_fut_buffer=pos_fut_buffer,
                n_frames=2,
            ),
            dof_queue[:2, ref_to_onnx].reshape(-1).astype(np.float32),
        )

    def test_copy_fk_sequence_inputs_uses_reference_queues_directly(self):
        runtime = VrLatestObsReference(n_fut_frames=2, num_actions=29, expected_dim=65)
        for frame in range(3):
            runtime.store(_obs(frame), current_time=float(frame), frame_index=frame)

        root_pos_seq = np.zeros((1, 3, 3), dtype=np.float32)
        root_rot_seq = np.zeros((1, 3, 4), dtype=np.float32)
        dof_pos_seq = np.zeros((1, 3, 29), dtype=np.float32)
        cur_root_pos = np.array([9.1, 9.2, 9.3], dtype=np.float32)
        cur_root_rot = np.array([1.0, 0.1, 0.2, 0.3], dtype=np.float32)
        cur_dof_pos = np.arange(29, dtype=np.float32) + 9.0

        self.assertTrue(
            runtime.copy_fk_sequence_inputs(
                root_pos_seq=root_pos_seq,
                root_rot_seq=root_rot_seq,
                dof_pos_seq=dof_pos_seq,
                cur_root_pos=cur_root_pos,
                cur_root_rot=cur_root_rot,
                cur_dof_pos=cur_dof_pos,
                n_frames=2,
            )
        )

        np.testing.assert_allclose(root_pos_seq[0, 0], cur_root_pos)
        np.testing.assert_allclose(root_rot_seq[0, 0], cur_root_rot)
        np.testing.assert_allclose(dof_pos_seq[0, 0], cur_dof_pos)
        np.testing.assert_allclose(
            root_pos_seq[0, 1:],
            np.stack([_obs(1)[58:61], _obs(2)[58:61]]),
        )
        np.testing.assert_allclose(
            root_rot_seq[0, 1:],
            np.stack([_obs(1)[61:65], _obs(2)[61:65]]),
        )
        np.testing.assert_allclose(
            dof_pos_seq[0, 1:],
            np.stack([_obs(1)[:29], _obs(2)[:29]]),
        )

    def test_ready_for_motion_matches_phase_3b_rule(self):
        runtime = VrLatestObsReference(n_fut_frames=3, num_actions=29, expected_dim=65)
        for frame in range(8):
            runtime.store(_obs(frame), current_time=float(frame), frame_index=frame)
        self.assertFalse(runtime.is_ready_for_motion(False, delay_frames=5))
        self.assertFalse(runtime.is_ready_for_motion(True, delay_frames=5))

        runtime.store(_obs(8), current_time=8.0, frame_index=8)
        self.assertTrue(runtime.is_ready_for_motion(True, delay_frames=5))

    def test_rejects_short_latest_obs_without_mutating_state(self):
        runtime = VrLatestObsReference(n_fut_frames=2, num_actions=29, expected_dim=65)

        self.assertFalse(runtime.store(np.zeros(64, dtype=np.float32), current_time=1.0))

        self.assertFalse(runtime.received)
        self.assertIsNone(runtime.latest_obs)
        self.assertEqual(runtime.seen_frames, 0)


if __name__ == "__main__":
    unittest.main()
