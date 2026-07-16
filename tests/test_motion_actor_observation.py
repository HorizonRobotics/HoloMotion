import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from holomotion.src.training.h5_dataloader import _WorldFrameNormalizeTransform
from holomotion.src.motion_tracking.actor_observation import (
    MotionActorObservationInput,
    build_motion_actor_observation_torch,
    derive_motion_actor_terms_torch,
    motion_actor_observation_dim,
)
from holomotion.src.motion_tracking.reference_observation import (
    derive_reference_kinematics_numpy,
)


def _make_inputs(device: str, *, batch: int = 2, future: int = 10):
    frames = future + 2
    time = torch.arange(frames, device=device, dtype=torch.float32) / 50.0
    qpos = torch.zeros((batch, frames, 36), device=device)
    qpos[..., 0] = time
    qpos[..., 2] = 0.8 + time
    qpos[..., 3] = 1.0
    qpos[..., 7:] = time[None, :, None]
    robot_quat = torch.zeros((batch, 4), device=device)
    robot_quat[:, 0] = 1.0
    robot_angvel = torch.zeros((batch, 3), device=device)
    robot_dof_pos = torch.arange(29, device=device, dtype=torch.float32).expand(
        batch, -1
    )
    robot_dof_vel = robot_dof_pos + 100.0
    last_action = robot_dof_pos + 200.0
    default = torch.arange(29, device=device, dtype=torch.float32)
    return MotionActorObservationInput(
        reference_qpos=qpos,
        robot_root_quat_wxyz=robot_quat,
        robot_root_angvel_local=robot_angvel,
        robot_dof_pos=robot_dof_pos,
        robot_dof_vel=robot_dof_vel,
        last_action=last_action,
        default_dof_pos=default,
    )


class MotionActorObservationTests(unittest.TestCase):
    def test_training_world_normalization_matches_deployment_entry_alignment(self):
        rng = np.random.default_rng(20260711)
        frames = 13
        sample_time = np.arange(frames, dtype=np.float32) / 50.0
        initial_yaw = 1.13
        yaw = initial_yaw + 0.45 * sample_time
        euler = np.stack(
            [
                0.08 * np.sin(1.7 * sample_time),
                -0.06 * np.cos(1.2 * sample_time),
                yaw,
            ],
            axis=-1,
        )
        quat_xyzw = Rotation.from_euler("xyz", euler).as_quat().astype(
            np.float32
        )
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
        root_pos = np.stack(
            [
                2.3 + 0.7 * sample_time,
                -1.7 - 0.3 * sample_time,
                0.82 + 0.03 * np.sin(2.0 * sample_time),
            ],
            axis=-1,
        ).astype(np.float32)
        dof_pos = rng.normal(size=(frames, 29)).astype(np.float32) * 0.2
        qpos = np.concatenate([root_pos, quat_wxyz, dof_pos], axis=-1)
        kinematics = derive_reference_kinematics_numpy(
            qpos, sample_time=sample_time
        )

        training_arrays = {
            "ref_rg_pos": torch.from_numpy(root_pos[:, None].copy()),
            "ref_rb_rot": torch.from_numpy(quat_xyzw[:, None].copy()),
            "ref_body_vel": torch.from_numpy(
                kinematics.root_linvel_world[:, None].copy()
            ),
            "ref_body_ang_vel": torch.from_numpy(
                kinematics.root_angvel_world[:, None].copy()
            ),
        }
        _WorldFrameNormalizeTransform()(training_arrays)
        normalized_qpos = np.concatenate(
            [
                training_arrays["ref_rg_pos"][:, 0].numpy(),
                training_arrays["ref_rb_rot"][:, 0].numpy()[:, [3, 0, 1, 2]],
                dof_pos,
            ],
            axis=-1,
        )

        deployment_entry_yaw = -0.72
        alignment_xyzw = Rotation.from_euler(
            "z", deployment_entry_yaw - initial_yaw
        ).as_quat().astype(np.float32)
        alignment_wxyz = alignment_xyzw[[3, 0, 1, 2]]
        current_index = 1
        tracking_yaw_error = 0.09
        canonical_robot_rotation = Rotation.from_euler(
            "xyz",
            [
                0.03,
                -0.02,
                yaw[current_index] - initial_yaw + tracking_yaw_error,
            ],
        )
        deployment_robot_rotation = Rotation.from_euler(
            "z", deployment_entry_yaw
        ) * canonical_robot_rotation
        canonical_robot_wxyz = canonical_robot_rotation.as_quat()[
            [3, 0, 1, 2]
        ].astype(np.float32)
        deployment_robot_wxyz = deployment_robot_rotation.as_quat()[
            [3, 0, 1, 2]
        ].astype(np.float32)

        common = {
            "reference_sample_time": torch.from_numpy(sample_time[None]),
            "robot_root_angvel_local": torch.tensor([[0.1, -0.2, 0.3]]),
            "robot_dof_pos": torch.from_numpy(
                rng.normal(size=(1, 29)).astype(np.float32)
            ),
            "robot_dof_vel": torch.from_numpy(
                rng.normal(size=(1, 29)).astype(np.float32)
            ),
            "last_action": torch.from_numpy(
                rng.normal(size=(1, 29)).astype(np.float32)
            ),
            "default_dof_pos": torch.from_numpy(
                rng.normal(size=29).astype(np.float32)
            ),
        }
        training = build_motion_actor_observation_torch(
            MotionActorObservationInput(
                reference_qpos=torch.from_numpy(normalized_qpos[None]),
                robot_root_quat_wxyz=torch.from_numpy(
                    canonical_robot_wxyz[None]
                ),
                **common,
            ),
            current_index=current_index,
            num_future_frames=10,
        )
        deployment = build_motion_actor_observation_torch(
            MotionActorObservationInput(
                reference_qpos=torch.from_numpy(qpos[None]),
                robot_root_quat_wxyz=torch.from_numpy(
                    deployment_robot_wxyz[None]
                ),
                reference_yaw_alignment_wxyz=torch.from_numpy(
                    alignment_wxyz[None]
                ),
                **common,
            ),
            current_index=current_index,
            num_future_frames=10,
        )

        np.testing.assert_allclose(
            training_arrays["ref_rg_pos"][0, 0, :2], 0.0, atol=1.0e-6
        )
        torch.testing.assert_close(
            training, deployment, rtol=1.0e-5, atol=1.0e-5
        )

    def test_release_shape_and_named_term_layout(self):
        inputs = _make_inputs("cpu")
        observation = build_motion_actor_observation_torch(inputs)
        terms = derive_motion_actor_terms_torch(inputs)

        self.assertEqual(tuple(observation.shape), (2, 604))
        self.assertEqual(motion_actor_observation_dim(10), 604)
        self.assertEqual(tuple(terms["actor_ref_dof_pos_fut"].shape), (2, 10, 29))
        self.assertEqual(
            tuple(terms["actor_ref_future_root_ori_robot_frame_6d"].shape),
            (2, 10, 6),
        )
        np.testing.assert_allclose(
            terms["actor_ref_gravity_projection_cur"].numpy(),
            [[0.0, 0.0, -1.0]] * 2,
            atol=0.0,
        )
        np.testing.assert_allclose(
            terms["actor_ref_base_linvel_cur"].numpy(),
            [[1.0, 0.0, 1.0]] * 2,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            terms["actor_ref_robot_yaw_error_sin_cos"].numpy(),
            [[0.0, 1.0]] * 2,
            atol=0.0,
        )
        np.testing.assert_allclose(terms["actor_dof_pos"].numpy(), 0.0, atol=0.0)

    def test_reference_dof_reordering(self):
        inputs = _make_inputs("cpu", batch=1)
        qpos = inputs.reference_qpos.clone()
        qpos[..., 7:] = torch.arange(29, dtype=torch.float32)
        reverse = torch.arange(28, -1, -1, dtype=torch.int32)
        reordered = MotionActorObservationInput(
            **{
                **inputs.__dict__,
                "reference_qpos": qpos,
                "reference_dof_indices": reverse,
            }
        )
        terms = derive_motion_actor_terms_torch(reordered)
        torch.testing.assert_close(
            terms["actor_ref_dof_pos_cur"][0],
            torch.arange(28, -1, -1, dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_cuda_matches_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        expected = build_motion_actor_observation_torch(_make_inputs("cpu"))
        actual = build_motion_actor_observation_torch(_make_inputs("cuda"))
        torch.cuda.synchronize()
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=1.0e-5, atol=1.0e-5
        )


if __name__ == "__main__":
    unittest.main()
