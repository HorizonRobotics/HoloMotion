import math
from dataclasses import replace

import pytest
import torch

from holomotion.src.motion_tracking.link_observation import (
    LinkObservationInput,
    build_link_observation,
    quat_multiply_wxyz,
)


def make_inputs() -> LinkObservationInput:
    roll_90 = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
    reference_pos = torch.tensor(
        [
            [
                [[10.0, 20.0, 1.0], [10.0, 20.0, 2.0]],
                [[10.0, 20.0, 1.0], [10.0, 20.0, 3.0]],
            ]
        ]
    )
    actual_quat = torch.zeros(1, 2, 4)
    actual_quat[..., 0] = 1.0
    return LinkObservationInput(
        reference_link_pos=reference_pos,
        reference_link_quat_wxyz=roll_90.reshape(1, 1, 1, 4)
        .expand(1, 2, 2, 4)
        .clone(),
        reference_root_pos=torch.tensor([[10.0, 20.0, 1.0]]),
        robot_root_quat_wxyz=roll_90.reshape(1, 4),
        robot_link_pos_local=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        ),
        robot_link_quat_local_wxyz=actual_quat,
        frame_offsets=torch.tensor([1.0, 2.0]),
        reference_root_height=torch.tensor([[0.8]]),
    )


def test_future_targets_use_current_root_with_roll_and_frame_offsets():
    observation = build_link_observation(make_inputs())

    torch.testing.assert_close(
        observation.target_pos[0, :, 1],
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        observation.position_error[0, :, 1],
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        observation.target_rot6d[0, 0, 1],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert observation.time_offsets[0, :, 0].tolist() == [1.0, 2.0]
    torch.testing.assert_close(
        observation.reference_height,
        torch.tensor([[0.8]]),
    )
    assert observation.task.shape == (1, 2, 37)


def test_orientation_error_uses_target_times_inverse_actual():
    inputs = make_inputs()
    half = math.sqrt(0.5)
    pitch_90 = torch.tensor([half, 0.0, half, 0.0])
    yaw_90 = torch.tensor([half, 0.0, 0.0, half])
    reference_quat = inputs.reference_link_quat_wxyz.clone()
    reference_quat[:, :, 1] = quat_multiply_wxyz(
        inputs.robot_root_quat_wxyz[:, None, :], pitch_90
    )
    actual_quat = inputs.robot_link_quat_local_wxyz.clone()
    actual_quat[:, 1] = yaw_90
    observation = build_link_observation(
        replace(
            inputs,
            reference_link_quat_wxyz=reference_quat,
            robot_link_quat_local_wxyz=actual_quat,
        )
    )

    torch.testing.assert_close(
        observation.target_rot6d[0, 0, 1],
        torch.tensor([0.0, 0.0, -1.0, 1.0, 0.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        observation.rotation_error6d[0, 0, 1],
        torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_masks_clear_invalid_time_and_link_without_nan_leakage():
    inputs = make_inputs()
    positions = inputs.reference_link_pos.clone()
    positions[0, 1, 0] = float("nan")
    positions[0, :, 1] = float("nan")
    observation = build_link_observation(
        replace(
            inputs,
            reference_link_pos=positions,
            link_mask=torch.tensor([1.0, 0.0]),
            time_valid=torch.tensor([1.0, 0.0]),
            frame_offsets=torch.tensor([1.0, -2.0]),
        )
    )

    assert observation.link_mask.tolist() == [[1.0, 0.0]]
    assert observation.time_valid.tolist() == [[1.0, 0.0]]
    assert observation.effective_mask.tolist() == [[[1.0, 0.0], [0.0, 0.0]]]
    assert observation.time_offsets[0, :, 0].tolist() == [1.0, 0.0]
    assert torch.count_nonzero(observation.target_pos[0, :, 1]) == 0
    assert torch.count_nonzero(observation.target_pos[0, 1]) == 0
    assert torch.isfinite(observation.flat_command).all()

    active_positions = inputs.reference_link_pos.clone()
    active_positions[0, 0, 1] = float("nan")
    active = build_link_observation(
        replace(inputs, reference_link_pos=active_positions)
    )
    assert active.link_mask.tolist() == [[1.0, 0.0]]
    assert torch.count_nonzero(active.target_pos[0, :, 1]) == 0


def test_reference_only_and_height_policy():
    inputs = make_inputs()
    no_fk = replace(
        inputs,
        robot_link_pos_local=None,
        robot_link_quat_local_wxyz=None,
    )
    reference_only = build_link_observation(no_fk, include_error_terms=False)
    full = build_link_observation(inputs)

    torch.testing.assert_close(reference_only.target_pos, full.target_pos)
    assert reference_only.position_error is None
    assert reference_only.rotation_error6d is None
    assert reference_only.task.shape == (1, 2, 19)

    invalid_height = replace(
        no_fk, reference_root_height=torch.tensor([[float("nan")]])
    )
    optional_height = build_link_observation(
        invalid_height, include_error_terms=False
    )
    mandatory_height = build_link_observation(
        invalid_height, root_link_index=None, include_error_terms=False
    )
    assert optional_height.link_mask.tolist() == [[0.0, 1.0]]
    assert optional_height.reference_height.tolist() == [[0.0]]
    assert mandatory_height.link_mask.tolist() == [[0.0, 0.0]]


def test_masked_link_has_zero_gradient_and_active_link_has_finite_gradient():
    inputs = make_inputs()
    positions = inputs.reference_link_pos.clone().requires_grad_(True)
    observation = build_link_observation(
        replace(
            inputs,
            reference_link_pos=positions,
            link_mask=torch.tensor([1.0, 0.0]),
        )
    )
    observation.target_pos.sum().backward()

    assert torch.isfinite(positions.grad).all()
    assert torch.count_nonzero(positions.grad[0, :, 0]) > 0
    assert torch.count_nonzero(positions.grad[0, :, 1]) == 0


def test_bad_builder_arguments_fail_before_packing():
    inputs = make_inputs()
    with pytest.raises(TypeError, match="include_error_terms"):
        build_link_observation(inputs, include_error_terms=1)
    with pytest.raises(ValueError, match="root_link_index"):
        build_link_observation(inputs, root_link_index=2)
    with pytest.raises(ValueError, match="Expected"):
        build_link_observation(
            replace(inputs, frame_offsets=torch.tensor([1.0]))
        )
