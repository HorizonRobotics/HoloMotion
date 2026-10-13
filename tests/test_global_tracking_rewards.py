"""CPU checks for clean immediate-next world-frame tracking rewards."""

import math
from types import SimpleNamespace

import pytest
import torch

from holomotion.src.motion_tracking.global_tracking_rewards import (
    anchor_height_tracking_exp,
    body_rotation_tracking_exp_from_quaternions,
    global_body_rotation_tracking_exp,
    height_tracking_exp_from_positions,
    quaternion_angle_error_wxyz,
)


def _quat(axis: int, angle: float) -> torch.Tensor:
    result = torch.zeros(4, dtype=torch.float64)
    result[0] = math.cos(angle / 2)
    result[axis + 1] = math.sin(angle / 2)
    return result


def _fake_env():
    identity = _quat(2, 0.0)
    ref_pos = torch.tensor(
        [
            [[10.0, 20.0, 1.5], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[10.0, 20.0, 1.7], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ],
        dtype=torch.float64,
    )
    robot_pos = ref_pos.clone()
    robot_pos[0, 0] = torch.tensor([100.0, -100.0, 1.8])
    ref_quat = identity.repeat(2, 3, 1)
    robot_quat = ref_quat.clone()
    robot_quat[:, 0] = _quat(0, math.pi)  # Excluded pelvis rotation.
    robot_quat[0, 2] = _quat(2, math.pi / 2)  # Selected right yaw.
    calls = []

    class FakeCommand:
        urdf_body_names = ("right", "pelvis", "left")
        robot = SimpleNamespace(
            body_names=("pelvis", "left", "right"),
            data=SimpleNamespace(body_pos_w=robot_pos, body_quat_w=robot_quat),
        )

        def get_ref_motion_bodylink_global_pos_immediate_next(self, prefix):
            calls.append(("position", prefix))
            return ref_pos

        def get_ref_motion_bodylink_global_rot_wxyz_immediate_next(
            self, prefix
        ):
            calls.append(("rotation", prefix))
            return ref_quat

    command = FakeCommand()

    def get_term(name):
        assert name == "ref_motion"
        return command

    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=get_term))
    return env, command, calls


def test_height_tracks_world_z_only_and_preserves_inputs():
    reference = torch.tensor([[12.0, -7.0, 1.5], [0.0, 0.0, 1.7]])
    robot = torch.tensor([[-5.0, 19.0, 1.8], [9.0, -4.0, 1.7]])
    reference_before = reference.clone()
    robot_before = robot.clone()

    result = height_tracking_exp_from_positions(reference, robot, 0.3)

    torch.testing.assert_close(result, torch.tensor([math.exp(-1), 1.0]))
    torch.testing.assert_close(reference, reference_before)
    torch.testing.assert_close(robot, robot_before)
    shifted_origin = torch.tensor([4.0, -9.0, 13.0])
    torch.testing.assert_close(
        height_tracking_exp_from_positions(
            reference + shifted_origin, robot + shifted_origin, 0.3
        ),
        result,
    )


@pytest.mark.parametrize("bad_std", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_invalid_standard_deviation_is_rejected(bad_std):
    with pytest.raises(ValueError, match="finite and positive"):
        height_tracking_exp_from_positions(
            torch.zeros(1, 3), torch.zeros(1, 3), bad_std
        )
    with pytest.raises(ValueError, match="finite and positive"):
        body_rotation_tracking_exp_from_quaternions(
            _quat(2, 0.0).reshape(1, 1, 4),
            _quat(2, 0.0).reshape(1, 1, 4),
            bad_std,
        )


def test_malformed_position_and_quaternion_shapes_are_rejected():
    with pytest.raises(ValueError, match="matching"):
        height_tracking_exp_from_positions(
            torch.zeros(1, 3), torch.zeros(2, 3), 1.0
        )
    with pytest.raises(ValueError, match="matching"):
        height_tracking_exp_from_positions(
            torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), 1.0
        )
    with pytest.raises(ValueError, match="matching"):
        quaternion_angle_error_wxyz(torch.zeros(1, 4), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="nonempty"):
        body_rotation_tracking_exp_from_quaternions(
            torch.zeros(1, 0, 4), torch.zeros(1, 0, 4), 1.0
        )


def test_quaternion_angle_uses_full_rotation_and_shortest_arc():
    identity = _quat(2, 0.0)
    reference = identity.repeat(3, 1)
    robot = torch.stack(
        (-3 * identity, _quat(2, math.pi / 2), _quat(1, math.pi))
    )

    torch.testing.assert_close(
        quaternion_angle_error_wxyz(reference, robot),
        torch.tensor([0.0, math.pi / 2, math.pi], dtype=torch.float64),
    )
    torch.testing.assert_close(
        quaternion_angle_error_wxyz(reference, -robot),
        torch.tensor([0.0, math.pi / 2, math.pi], dtype=torch.float64),
    )


def test_invalid_quaternions_do_not_earn_a_good_reward():
    identity = _quat(2, 0.0).reshape(1, 1, 4)
    zero = torch.zeros_like(identity)
    result = body_rotation_tracking_exp_from_quaternions(identity, zero, 0.4)
    assert torch.isnan(result).all()


def test_body_rotation_squares_then_averages_before_exponentiating():
    identity = _quat(2, 0.0)
    reference = identity.repeat(1, 2, 1)
    robot = torch.stack((identity, _quat(2, math.pi / 2))).reshape(1, 2, 4)

    result = body_rotation_tracking_exp_from_quaternions(reference, robot, 1.0)

    expected = math.exp(-((math.pi / 2) ** 2) / 2)
    torch.testing.assert_close(
        result, torch.tensor([expected], dtype=torch.float64)
    )
    per_body_exp_mean = (1 + math.exp(-((math.pi / 2) ** 2))) / 2
    assert result.item() != pytest.approx(per_body_exp_mean)


def test_adapters_use_clean_immediate_next_and_named_simulator_bodies():
    env, _command, calls = _fake_env()

    height = anchor_height_tracking_exp(env, std=0.3)
    rotation = global_body_rotation_tracking_exp(
        env, std=1.0, body_names=["right", "left"]
    )

    torch.testing.assert_close(
        height, torch.tensor([math.exp(-1), 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        rotation,
        torch.tensor(
            [math.exp(-((math.pi / 2) ** 2) / 2), 1.0], dtype=torch.float64
        ),
    )
    assert calls == [("position", "ref_"), ("rotation", "ref_")]


@pytest.mark.parametrize("names", [[], ["left", "left"], "left", ["missing"]])
def test_rotation_adapter_rejects_invalid_body_selection(names):
    env, _command, calls = _fake_env()
    with pytest.raises(ValueError):
        global_body_rotation_tracking_exp(env, std=0.4, body_names=names)
    assert calls == []


def test_anchor_adapter_checks_reference_body_membership():
    env, command, calls = _fake_env()
    command.urdf_body_names = ("right", "left")
    with pytest.raises(ValueError, match="Body names missing"):
        anchor_height_tracking_exp(env, std=0.3)
    assert calls == []
