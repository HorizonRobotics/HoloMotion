"""CUDA FK and source-independent actor parity; never start ROS/controllers."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp
from humanoid_policy.observation import PolicyObservationBuilder
from humanoid_policy.link_reference import DeviceLinkReference
from humanoid_policy.vr_reference_adapter import VRJointReferenceAdapter
from test_observation_contract import (
    ASSET,
    _fixture,
    _training_actor,
    _training_reference_from_vr,
)

from holomotion.src.motion_tracking.link_fk import SharedLinkFK
from humanoid_policy.fk_warp import WarpLinkFK

DEVICE = os.environ.get("HOLOMOTION_WARP_TEST_DEVICE", "cuda:0")
MEASURED = (
    "robot_root_quat_wxyz",
    "robot_dof_pos_onnx",
    "robot_dof_vel_onnx",
    "projected_gravity",
    "root_ang_vel_local",
    "last_action_onnx",
)


def _cuda_device():
    try:
        device = wp.get_device(DEVICE)
    except RuntimeError:
        pytest.skip("CUDA device unavailable; no CPU deployment fallback")
    if not device.is_cuda:
        pytest.skip("CUDA test requires a CUDA device, not CPU FK")
    return device


@pytest.mark.parametrize("batch", [1, 5, 6, 32])
def test_fk_same_asset_joint_order_position_and_quaternion(batch):
    device = _cuda_device()
    oracle = SharedLinkFK(
        str(ASSET),
        joint_names=tuple(reversed(SharedLinkFK(str(ASSET)).dof_names)),
    )
    gpu = WarpLinkFK(
        ASSET,
        joint_names=oracle.input_joint_names,
        batch_size=batch,
        device=device,
    )
    q = (
        np.random.default_rng(batch)
        .uniform(-1.5, 1.5, (batch, 29))
        .astype(np.float32)
    )
    q[0] = 0
    expected_pos, expected_rot = oracle(torch.from_numpy(q))
    positions, rotations = gpu.launch(
        wp.array(q, dtype=wp.float32, device=device)
    )
    np.testing.assert_allclose(
        positions.numpy(), expected_pos.numpy(), atol=3e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        rotations.numpy()[..., [3, 0, 1, 2]],
        expected_rot.numpy(),
        atol=3e-6,
        rtol=2e-5,
    )


def _gpu_fixture():
    device = _cuda_device()
    contract, inputs, modes = _fixture()
    source_names = tuple(reversed(contract["config_joint_names"]))
    return (
        contract,
        PolicyObservationBuilder(contract, ASSET, device=device),
        VRJointReferenceAdapter(contract, ASSET, source_names, device=device),
        inputs,
        modes,
        source_names,
    )


def _direct_reference(contract, inputs, *, time_valid=None):
    device = _cuda_device()

    def array(value):
        return wp.array(
            np.ascontiguousarray(value, dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )

    return DeviceLinkReference(
        link_names=tuple(contract["link_names"]),
        frame_offsets=tuple(contract["frame_offsets"]),
        positions=array(inputs["reference_link_pos"]),
        rotations_wxyz=array(inputs["reference_link_quat_wxyz"]),
        root_position=array(inputs["reference_root_pos"]),
        time_valid=array(np.ones(5) if time_valid is None else time_valid),
    )


def test_four_modes_accept_link_reference_without_vr_adapter():
    contract, gpu, _, inputs, masks, _ = _gpu_fixture()
    measured = {key: inputs[key] for key in MEASURED}
    reference = _direct_reference(contract, inputs)
    for mode, mask in masks.items():
        actual = gpu.build(
            mode=mode, reference=reference, **measured
        ).array.numpy()
        expected = _training_actor(contract, mode, inputs, measured)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        assert actual.shape == (1, 1378)
        np.testing.assert_array_equal(actual[0, 94:108], mask)


def test_vr_adapter_and_four_mode_actor_parity_with_random_joint_orders():
    contract, gpu, adapter, inputs, modes, source_names = _gpu_fixture()
    rng = np.random.default_rng(741)
    for sample in range(16):
        seq = rng.uniform(-1, 1, (8, 36)).astype(np.float32)
        seq[:, 3:7] /= np.linalg.norm(seq[:, 3:7], axis=-1, keepdims=True)
        root = rng.uniform(-1, 1, 4).astype(np.float32)
        root /= np.linalg.norm(root)
        yaw = np.array(
            [np.cos(sample * 0.1), 0, 0, np.sin(sample * 0.1)],
            dtype=np.float32,
        )
        measured = dict(
            robot_root_quat_wxyz=root,
            robot_dof_pos_onnx=rng.uniform(-0.9, 0.9, 29).astype(np.float32),
            robot_dof_vel_onnx=rng.uniform(-1, 1, 29).astype(np.float32),
            projected_gravity=inputs["projected_gravity"],
            root_ang_vel_local=np.array([0.1, -0.2, 0.3], dtype=np.float32),
            last_action_onnx=rng.uniform(-0.5, 0.5, 29).astype(np.float32),
        )
        reference = adapter.from_device_sequence(
            wp.array(seq[None], dtype=wp.float32, device=gpu.device),
            yaw_alignment_wxyz=yaw,
        )
        oracle_reference = _training_reference_from_vr(
            contract, seq, source_names, yaw
        )
        for mode in modes:
            actual = gpu.build(
                mode=mode, reference=reference, **measured
            ).array.numpy()
            expected = _training_actor(
                contract, mode, oracle_reference, measured
            )
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_host_origin_vr_queue_and_device_queue_have_identical_gpu_fk():
    _, gpu, adapter, inputs, _, _ = _gpu_fixture()
    seq = np.zeros((8, 36), dtype=np.float32)
    seq[:, 2] = 0.8
    seq[:, 3] = 1
    seq[:, 7:] = adapter.reference_defaults
    measured = {key: inputs[key] for key in MEASURED}
    a = gpu.build(
        mode="Keypoint-5", reference=adapter.from_host_sequence(seq), **measured
    ).array.numpy()
    b = gpu.build(
        mode="Keypoint-5",
        reference=adapter.from_device_sequence(
            wp.array(seq[None], dtype=wp.float32, device=gpu.device)
        ),
        **measured,
    ).array.numpy()
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "corrupt", ["nan", "zero_quaternion", "measured_nan", "invalid_tail"]
)
def test_gpu_invalid_vr_reference_or_state_is_rejected(corrupt):
    _, gpu, adapter, inputs, _, _ = _gpu_fixture()
    seq = np.zeros((8, 36), dtype=np.float32)
    seq[:, 2] = 0.8
    seq[:, 3] = 1
    measured = {key: inputs[key] for key in MEASURED}
    if corrupt == "nan":
        seq[2, 7] = np.nan
    elif corrupt == "zero_quaternion":
        seq[2, 3:7] = 0
    elif corrupt == "invalid_tail":
        seq[-1, 7] = np.nan
    else:
        measured["robot_dof_pos_onnx"] = np.full(29, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid"):
        gpu.build(
            mode="Keypoint-5", reference=adapter.from_host_sequence(seq), **measured
        )


@pytest.mark.parametrize(
    "time_valid", [[1, 1, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 2, 1, 1]]
)
def test_time_valid_mask_has_training_semantics(time_valid):
    contract, gpu, _, inputs, _, _ = _gpu_fixture()
    measured = {key: inputs[key] for key in MEASURED}
    reference = _direct_reference(contract, inputs, time_valid=time_valid)
    if time_valid != [1, 1, 1, 1, 0]:
        with pytest.raises(ValueError, match="Invalid"):
            gpu.build(mode="WholeBody-14", reference=reference, **measured)
    else:
        actual = gpu.build(
            mode="WholeBody-14", reference=reference, **measured
        ).array.numpy()
        np.testing.assert_allclose(
            actual,
            _training_actor(
                contract, "WholeBody-14", inputs, measured, time_valid
            ),
            atol=2e-5,
            rtol=2e-5,
        )


def test_invalid_unselected_link_is_masked_but_selected_link_is_rejected():
    contract, gpu, _, inputs, _, _ = _gpu_fixture()
    measured = {key: inputs[key] for key in MEASURED}
    inputs["reference_link_pos"][:, 1, :] = np.nan
    reference = _direct_reference(contract, inputs)
    gpu.build(mode="Pelvis-1", reference=reference, **measured)
    with pytest.raises(ValueError, match="Invalid"):
        gpu.build(mode="WholeBody-14", reference=reference, **measured)


@pytest.mark.parametrize("field", ["link_names", "frame_offsets"])
def test_wrong_reference_order_is_rejected_before_inference(field):
    contract, gpu, _, inputs, _, _ = _gpu_fixture()
    reference = _direct_reference(contract, inputs)
    reference = replace(
        reference, **{field: tuple(reversed(getattr(reference, field)))}
    )
    with pytest.raises(ValueError, match="differ"):
        gpu.build(
            mode="Keypoint-5",
            reference=reference,
            **{key: inputs[key] for key in MEASURED},
        )


def test_gpu_fk_utility_rejects_cpu_device():
    with pytest.raises(ValueError, match="CUDA"):
        WarpLinkFK(ASSET, joint_names=(), batch_size=1, device="cpu")
