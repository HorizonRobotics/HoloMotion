import unittest

import numpy as np

from humanoid_policy.vr_reference import VrReference


def _qpos(frame: float) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[:3] = [frame, frame + 0.1, frame + 0.2]
    qpos[3] = 1.0
    qpos[7:] = frame
    return qpos


class VrReferenceTest(unittest.TestCase):
    def test_queue_exposes_current_and_future_qpos(self):
        reference = VrReference(n_fut_frames=2)
        for frame in range(6):
            self.assertTrue(
                reference.store(
                    _qpos(float(frame)),
                    current_time=10.0 + frame,
                    sample_time=100.0 + 0.02 * frame,
                    frame_index=frame,
                )
            )

        np.testing.assert_array_equal(reference.current_dof_pos(), 2.0)
        np.testing.assert_array_equal(reference.dof_pos_queue[:, 0], [3.0, 4.0])
        np.testing.assert_allclose(reference.current_root_pos(), [2.0, 2.1, 2.2])
        np.testing.assert_array_equal(reference.root_pos_queue[:, 0], [3.0, 4.0])
        sequence, _ = reference.observation_sequence(n_frames=2)
        np.testing.assert_array_equal(sequence[:, 7], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_shared_kinematics_uses_sender_timestamps(self):
        reference = VrReference(n_fut_frames=2)
        sample_times = [100.00, 100.01, 100.03, 100.06, 100.10, 100.15]
        values = [0.0, 1.0, 3.0, 6.0, 10.0, 15.0]
        for frame, (sample_time, value) in enumerate(zip(sample_times, values)):
            reference.store(
                _qpos(value),
                current_time=10.0 + frame,
                sample_time=sample_time,
                frame_index=frame,
            )

        kinematics = reference.update_kinematics(n_frames=2, fps=50.0)
        np.testing.assert_allclose(
            kinematics.dof_vel[:, 0],
            [
                (6.0 - 1.0) / 0.05,
                (10.0 - 3.0) / 0.07,
                (15.0 - 6.0) / 0.09,
            ],
            atol=1.0e-3,
        )
        np.testing.assert_allclose(
            kinematics.root_linvel_world[:, 0],
            kinematics.dof_vel[:, 0],
            atol=1.0e-5,
        )

    def test_motion_readiness_requires_past_current_and_future(self):
        reference = VrReference(n_fut_frames=3)
        for frame in range(4):
            reference.store(_qpos(frame), current_time=float(frame))
        self.assertFalse(reference.is_ready_for_motion(True, 0))
        reference.store(_qpos(4.0), current_time=4.0)
        self.assertFalse(reference.is_ready_for_motion(True, 0))
        reference.store(_qpos(5.0), current_time=5.0)
        self.assertTrue(reference.is_ready_for_motion(True, 0))
        self.assertFalse(reference.is_ready_for_motion(False, 0))

    def test_rejects_wrong_reference_dimension_without_mutating_state(self):
        reference = VrReference(n_fut_frames=2)
        self.assertFalse(reference.store(np.zeros(35), current_time=1.0))
        self.assertFalse(reference.has_reference)
        self.assertEqual(reference.seen_frames, 0)


if __name__ == "__main__":
    unittest.main()
