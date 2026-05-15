import unittest
from types import SimpleNamespace

import numpy as np

from humanoid_policy.observation_evaluator import PolicyObservationEvaluator


class FakeLogger:
    def __init__(self):
        self.warns = []

    def info(self, msg):
        del msg

    def warn(self, msg):
        self.warns.append(str(msg))

    def error(self, msg):
        raise AssertionError(str(msg))


class FakeNode:
    def __init__(self):
        self.real_dof_names = ["hip", "knee", "ankle"]
        self.dof_names_ref_motion = ["hip", "knee", "ankle"]
        self.velocity_dof_names_onnx = ["knee", "hip", "ankle"]
        self.motion_dof_names_onnx = ["ankle", "hip", "knee"]
        self.velocity_default_angles_onnx = np.array([0.2, 0.1, 0.3], dtype=np.float32)
        self.motion_default_angles_onnx = np.array([0.3, 0.1, 0.2], dtype=np.float32)
        self.n_fut_frames = 2
        self.num_actions = 3
        self.actions_dim = 3
        self.current_policy_mode = "velocity"
        self.latest_obs_flag = False
        self.motion_frame_idx = 0
        self.n_motion_frames = 1
        self.actions_onnx = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.actor_place_holder_ndim = 4
        self._vr_reference = None
        self._lowstate_msg = SimpleNamespace(
            imu_state=SimpleNamespace(
                quaternion=[1.0, 0.0, 0.0, 0.0],
                gyroscope=[0.1, 0.2, 0.3],
            ),
            motor_state=[
                SimpleNamespace(q=1.0, dq=0.1),
                SimpleNamespace(q=2.0, dq=0.2),
                SimpleNamespace(q=3.0, dq=0.3),
            ],
        )
        self.logger = FakeLogger()

    def get_logger(self):
        return self.logger


class PolicyObservationEvaluatorTest(unittest.TestCase):
    def test_initializes_observation_state_and_offline_reference(self):
        node = FakeNode()
        evaluator = PolicyObservationEvaluator(node)

        evaluator.initialize_observation_state()

        self.assertEqual(evaluator.n_fut_frames_int, 2)
        self.assertEqual(evaluator.ref_to_onnx, [2, 0, 1])
        self.assertEqual(evaluator._offline_reference.n_fut_frames, 2)
        self.assertEqual(evaluator._pos_fut_buffer.shape, (3, 2))
        self.assertEqual(evaluator._fk_quat_fut_wxyz.shape, (2, 4))

    def test_robot_and_velocity_observation_terms_match_node_contract(self):
        node = FakeNode()
        evaluator = PolicyObservationEvaluator(node)
        evaluator.initialize_observation_state()

        np.testing.assert_allclose(
            evaluator._get_obs_dof_pos(),
            np.array([1.8, 0.9, 2.7], dtype=np.float32),
        )
        np.testing.assert_allclose(
            evaluator._get_obs_dof_vel(),
            np.array([0.2, 0.1, 0.3], dtype=np.float32),
        )
        np.testing.assert_allclose(
            evaluator._get_obs_rel_robot_root_ang_vel(),
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
        np.testing.assert_allclose(
            evaluator._get_obs_last_action(),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_placeholder_term_uses_configured_dimension(self):
        node = FakeNode()
        evaluator = PolicyObservationEvaluator(node)

        np.testing.assert_array_equal(
            evaluator._get_obs_place_holder(),
            np.zeros(4, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
