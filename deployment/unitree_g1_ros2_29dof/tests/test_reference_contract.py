import unittest

import numpy as np

from humanoid_policy.reference_contract import LatestObsReferenceQueues
from humanoid_policy.reference_contract import flatten_future_dof_pos_onnx
from humanoid_policy.reference_contract import flatten_vr_future_dof_pos_onnx
from humanoid_policy.reference_contract import future_frame_indices
from humanoid_policy.reference_contract import vr_current_dof_pos
from humanoid_policy.reference_contract import vr_current_dof_vel
from humanoid_policy.reference_contract import vr_current_root_pos


def _latest_obs_frame(value: float) -> np.ndarray:
    obs = np.zeros(65, dtype=np.float32)
    obs[:29] = value
    obs[29:58] = value + 100.0
    obs[58:61] = value + 200.0
    obs[61:65] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return obs


class ReferenceContractTest(unittest.TestCase):
    def test_latest_obs_future_queue_matches_phase_3b_shift_order(self):
        queues = LatestObsReferenceQueues(n_fut_frames=2)

        for frame_idx, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            self.assertTrue(
                queues.store(
                    _latest_obs_frame(value),
                    current_time=100.0 + frame_idx,
                    frame_index=frame_idx,
                )
            )

        self.assertEqual(queues.seen_frames, 4)
        self.assertTrue(queues.is_ready_for_motion(True, delay_frames=1))
        np.testing.assert_allclose(queues.latest_obs[0, :3], np.array([4.0, 4.0, 4.0]))
        np.testing.assert_allclose(queues.dof_pos_queue[:, 0], np.array([3.0, 4.0]))
        np.testing.assert_allclose(queues.dof_vel_queue[:, 0], np.array([103.0, 104.0]))
        np.testing.assert_allclose(queues.root_pos_queue[:, 0], np.array([203.0, 204.0]))
        np.testing.assert_allclose(queues.prev_dof_pos[:3], np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(queues.prev_dof_vel[:3], np.array([102.0, 102.0, 102.0]))
        np.testing.assert_allclose(queues.prev_root_pos, np.array([202.0, 202.0, 202.0]))
        self.assertEqual(queues.prev_frame_idx, 1)

    def test_current_vr_reference_uses_previous_future_queue_head(self):
        queues = LatestObsReferenceQueues(n_fut_frames=2)
        for frame_idx, value in enumerate([1.0, 2.0, 3.0]):
            queues.store(_latest_obs_frame(value), current_time=10.0 + frame_idx)

        offline_pos = np.full((5, 29), 9.0, dtype=np.float32)
        offline_vel = np.full((5, 29), 19.0, dtype=np.float32)

        np.testing.assert_allclose(vr_current_dof_pos(queues, offline_pos, 0)[:3], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(vr_current_dof_vel(queues, offline_vel, 0)[:3], [101.0, 101.0, 101.0])
        np.testing.assert_allclose(vr_current_root_pos(queues), [201.0, 201.0, 201.0])

    def test_current_vr_reference_without_future_queue_uses_latest_obs(self):
        queues = LatestObsReferenceQueues(n_fut_frames=0)
        queues.store(_latest_obs_frame(7.0), current_time=10.0)
        offline_pos = np.full((5, 29), 9.0, dtype=np.float32)
        offline_vel = np.full((5, 29), 19.0, dtype=np.float32)

        np.testing.assert_allclose(vr_current_dof_pos(queues, offline_pos, 0)[:3], [7.0, 7.0, 7.0])
        np.testing.assert_allclose(vr_current_dof_vel(queues, offline_vel, 0)[:3], [107.0, 107.0, 107.0])
        np.testing.assert_allclose(vr_current_root_pos(queues), [207.0, 207.0, 207.0])

    def test_future_indices_and_offline_flatten_clamp_at_last_frame(self):
        ref = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
        self.assertEqual(future_frame_indices(2, n_motion_frames=4, n_fut_frames=3).tolist(), [3, 3, 3])
        np.testing.assert_allclose(
            flatten_future_dof_pos_onnx(
                ref_dof_pos=ref,
                frame_idx=2,
                n_motion_frames=4,
                n_fut_frames=3,
                ref_to_onnx=[2, 0, 1],
            ),
            np.array([11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0], dtype=np.float32),
        )

    def test_vr_future_flatten_uses_queue_order_and_onnx_mapping(self):
        queues = LatestObsReferenceQueues(n_fut_frames=3, num_actions=3)
        queues.dof_pos_queue[:] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            flatten_vr_future_dof_pos_onnx(
                queues,
                n_fut_frames=3,
                ref_to_onnx=[2, 0, 1],
            ),
            np.array([3.0, 1.0, 2.0, 6.0, 4.0, 5.0, 9.0, 7.0, 8.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
