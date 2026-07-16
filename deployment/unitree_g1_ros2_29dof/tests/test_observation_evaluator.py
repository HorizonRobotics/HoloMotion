import unittest
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from humanoid_policy.observation_evaluator import PolicyObservationEvaluator


def _quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.asarray(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float32,
    )


def _quat_xyzw_from_yaw(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.asarray(
        [0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)],
        dtype=np.float32,
    )


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
        self.velocity_default_angles_onnx = np.array(
            [0.2, 0.1, 0.3], dtype=np.float32
        )
        self.motion_default_angles_onnx = np.array(
            [0.3, 0.1, 0.2], dtype=np.float32
        )
        self.n_fut_frames = 2
        self.num_actions = 3
        self.actions_dim = 3
        self.current_policy_mode = "velocity"
        self.reference_stream_active = False
        self.motion_frame_idx = 0
        self.n_motion_frames = 1
        self.actions_onnx = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.actor_place_holder_ndim = 4
        self.root_body_idx = 0
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
        self.assertEqual(evaluator._future_root_quat_wxyz_buffer.shape, (2, 4))

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

    def test_yaw_reference_observation_terms_are_deployable(self):
        node = FakeNode()
        evaluator = PolicyObservationEvaluator(node)
        evaluator.initialize_observation_state()
        node.n_motion_frames = 3
        node.ref_raw_bodylink_rot = np.zeros((3, 1, 4), dtype=np.float32)
        node.ref_raw_bodylink_rot[:, 0] = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            evaluator._get_obs_ref_future_yaw_delta_sin_cos(),
            [1.0, 0.0, 0.0, -1.0],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            evaluator._get_obs_ref_robot_yaw_error_sin_cos(),
            [0.0, 1.0],
            atol=1.0e-6,
        )
        self.assertEqual(
            evaluator._get_obs_ref_future_root_ori_robot_frame_6d().shape,
            (12,),
        )

    def test_motion_yaw_alignment_uses_motion_entry_yaw(self):
        node = FakeNode()
        evaluator = PolicyObservationEvaluator(node)
        evaluator.initialize_observation_state()
        node.n_motion_frames = 3
        node.ref_raw_bodylink_rot = np.zeros((3, 1, 4), dtype=np.float32)
        node.ref_raw_bodylink_rot[:, 0] = np.array(
            [
                _quat_xyzw_from_yaw(np.pi / 2.0),
                _quat_xyzw_from_yaw(np.pi / 2.0),
                _quat_xyzw_from_yaw(np.pi / 2.0),
            ],
            dtype=np.float32,
        )
        node._lowstate_msg.imu_state.quaternion = _quat_wxyz_from_yaw(
            np.pi / 6.0
        )
        evaluator.cache_lowstate(node._lowstate_msg, force=True)

        evaluator.begin_motion_yaw_alignment()

        np.testing.assert_allclose(
            evaluator._get_obs_ref_robot_yaw_error_sin_cos(),
            [0.0, 1.0],
            atol=1.0e-6,
        )

    def test_motion_yaw_alignment_tracks_post_entry_robot_yaw_error(self):
        node = FakeNode()
        evaluator = PolicyObservationEvaluator(node)
        evaluator.initialize_observation_state()
        node.n_motion_frames = 3
        node.ref_raw_bodylink_rot = np.zeros((3, 1, 4), dtype=np.float32)
        node.ref_raw_bodylink_rot[:, 0] = np.array(
            [
                _quat_xyzw_from_yaw(np.pi / 2.0),
                _quat_xyzw_from_yaw(np.pi / 2.0),
                _quat_xyzw_from_yaw(np.pi / 2.0),
            ],
            dtype=np.float32,
        )
        node._lowstate_msg.imu_state.quaternion = _quat_wxyz_from_yaw(
            np.pi / 6.0
        )
        evaluator.cache_lowstate(node._lowstate_msg, force=True)
        evaluator.begin_motion_yaw_alignment()

        yaw_delta = np.deg2rad(10.0)
        node._lowstate_msg.imu_state.quaternion = _quat_wxyz_from_yaw(
            np.pi / 6.0 + yaw_delta
        )
        evaluator.cache_lowstate(node._lowstate_msg, force=True)

        np.testing.assert_allclose(
            evaluator._get_obs_ref_robot_yaw_error_sin_cos(),
            [np.sin(-yaw_delta), np.cos(-yaw_delta)],
            atol=1.0e-6,
        )

    def test_policy_obs_list_includes_additional_atomic_terms(self):
        node = FakeNode()
        evaluator = PolicyObservationEvaluator(node)
        config = OmegaConf.create(
            {
                "obs": {
                    "obs_groups": {
                        "unified": {
                            "atomic_obs_list": [
                                {"actor_a": {"func": "place_holder"}}
                            ],
                            "additional_atomic_obs_list": [
                                {"actor_b": {"func": "place_holder"}}
                            ],
                        }
                    }
                },
                "modules": {
                    "actor": {
                        "obs_schema": {
                            "flattened_obs": {
                                "terms": ["unified/actor_b", "unified/actor_a"]
                            }
                        }
                    }
                },
            }
        )

        atomic_obs_list = evaluator._get_policy_atomic_obs_list(config)[
            "atomic_obs_list"
        ]

        self.assertEqual(
            [list(item.keys())[0] for item in atomic_obs_list],
            ["actor_b", "actor_a"],
        )


if __name__ == "__main__":
    unittest.main()
