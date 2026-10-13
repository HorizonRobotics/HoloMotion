"""Training-side numerical oracle and source-independent deployment contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from humanoid_policy.observation import PolicyObservationBuilder
from humanoid_policy.vr_reference_adapter import VRJointReferenceAdapter

from holomotion.src.motion_tracking.link_actor_packing import (
    link_actor_layout,
    pack_link_actor_observation,
)
from holomotion.src.motion_tracking.link_fk import SharedLinkFK
from holomotion.src.motion_tracking.link_observation import (
    DEFAULT_COMMAND_LINK_NAMES,
    LinkObservationInput,
    build_link_observation,
    quat_apply_wxyz,
    quat_multiply_wxyz,
)

REPO = Path(__file__).resolve().parents[3]
ASSET = REPO / "assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf"
OFFSETS = (1, 2, 3, 4, 5)
LINKS = DEFAULT_COMMAND_LINK_NAMES


def _fixture():
    fk = SharedLinkFK(str(ASSET), device="cpu")
    config_joints = tuple(fk.dof_names)
    onnx_joints = tuple(reversed(config_joints))
    config_from_onnx = [onnx_joints.index(name) for name in config_joints]
    q_onnx = np.linspace(-0.18, 0.18, 29, dtype=np.float32)
    local_pos, local_quat = fk(torch.as_tensor(q_onnx[config_from_onnx])[None])
    indices = [fk.body_names.index(name) for name in LINKS]
    root = np.array([0, 0, 1], dtype=np.float32)
    mode_links = {
        "Pelvis-1": {"pelvis"},
        "Keypoint-3": {"pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"},
        "Keypoint-5": {
            "pelvis",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        },
        "WholeBody-14": set(LINKS),
    }
    modes = {
        name: [int(link in selected) for link in LINKS]
        for name, selected in mode_links.items()
    }
    contract = {
        "link_names": list(LINKS),
        "frame_offsets": list(OFFSETS),
        "modes": modes,
        "actor_dim": link_actor_layout(
            {
                "link_names": LINKS,
                "frame_offsets": OFFSETS,
                "include_error_terms": True,
            }
        )["actor_dim"],
        "include_error_terms": True,
        "include_future_link_mask": False,
        "default_joint_pos_onnx": [0.0] * 29,
        "onnx_index_for_config_joint": config_from_onnx,
        "config_joint_names": list(config_joints),
        "asset_sha256": hashlib.sha256(ASSET.read_bytes()).hexdigest(),
    }
    inputs = {
        "reference_link_pos": np.tile(
            (local_pos[0, indices].numpy() + root)[None], (5, 1, 1)
        ),
        "reference_link_quat_wxyz": np.tile(
            local_quat[0, indices].numpy()[None], (5, 1, 1)
        ),
        "reference_root_pos": root,
        "robot_root_quat_wxyz": np.array([1, 0, 0, 0], dtype=np.float32),
        "robot_dof_pos_onnx": q_onnx,
        "robot_dof_vel_onnx": np.zeros(29, dtype=np.float32),
        "projected_gravity": np.array([0, 0, -1], dtype=np.float32),
        "root_ang_vel_local": np.zeros(3, dtype=np.float32),
        "last_action_onnx": np.zeros(29, dtype=np.float32),
    }
    return contract, inputs, modes


def _training_actor(contract, mode, reference, measured, time_valid=None):
    """Reuse training FK/packing as a test oracle, not a CPU deployment backend."""
    fk = SharedLinkFK(str(ASSET), joint_names=contract["config_joint_names"])
    indices = [fk.body_names.index(name) for name in contract["link_names"]]
    q = torch.as_tensor(measured["robot_dof_pos_onnx"])[
        contract["onnx_index_for_config_joint"]
    ][None]
    actual_pos, actual_quat = fk(q)
    command = build_link_observation(
        LinkObservationInput(
            reference_link_pos=torch.as_tensor(
                reference["reference_link_pos"]
            )[None],
            reference_link_quat_wxyz=torch.as_tensor(
                reference["reference_link_quat_wxyz"]
            )[None],
            reference_root_pos=torch.as_tensor(
                reference["reference_root_pos"]
            )[None],
            robot_root_quat_wxyz=torch.as_tensor(
                measured["robot_root_quat_wxyz"]
            )[None],
            robot_link_pos_local=actual_pos[:, indices],
            robot_link_quat_local_wxyz=actual_quat[:, indices],
            frame_offsets=torch.tensor(
                contract["frame_offsets"], dtype=torch.float32
            ),
            link_mask=torch.tensor(
                contract["modes"][mode], dtype=torch.float32
            )[None],
            time_valid=torch.tensor(
                [1] * 5 if time_valid is None else time_valid,
                dtype=torch.float32,
            )[None],
            reference_root_height=torch.as_tensor(
                reference["reference_root_pos"]
            )[2].reshape(1, 1),
        ),
        root_link_index=contract["link_names"].index("pelvis"),
        include_error_terms=True,
    )
    proprio = torch.cat(
        [
            torch.as_tensor(measured["projected_gravity"])[None],
            torch.as_tensor(measured["root_ang_vel_local"])[None],
            torch.as_tensor(measured["robot_dof_pos_onnx"])[None]
            - torch.tensor(contract["default_joint_pos_onnx"])[None],
            torch.as_tensor(measured["robot_dof_vel_onnx"])[None],
            torch.as_tensor(measured["last_action_onnx"])[None],
        ],
        dim=-1,
    )
    return pack_link_actor_observation(proprio, command).numpy()


def _training_reference_from_vr(contract, sequence, source_names, yaw):
    """Numerical oracle built from existing training utilities only."""
    fk = SharedLinkFK(str(ASSET), joint_names=contract["config_joint_names"])
    future = sequence[2:7]
    yaw = torch.as_tensor(yaw / np.linalg.norm(yaw))[None]
    root_quat = torch.as_tensor(future[:, 3:7])
    root_quat = quat_multiply_wxyz(
        yaw,
        root_quat / torch.linalg.vector_norm(root_quat, dim=-1, keepdim=True),
    )
    q = torch.as_tensor(
        future[:, 7:][
            :,
            [
                source_names.index(name)
                for name in contract["config_joint_names"]
            ],
        ]
    )
    local_pos, local_quat = fk(q)
    indices = [fk.body_names.index(name) for name in contract["link_names"]]
    current = sequence[1, :3]
    aligned = torch.as_tensor(current)[None] + quat_apply_wxyz(
        yaw, torch.as_tensor(future[:, :3] - current)
    )
    return {
        "reference_link_pos": (
            aligned[:, None]
            + quat_apply_wxyz(root_quat[:, None], local_pos[:, indices])
        ).numpy(),
        "reference_link_quat_wxyz": quat_multiply_wxyz(
            root_quat[:, None], local_quat[:, indices]
        ).numpy(),
        "reference_root_pos": current,
    }


@pytest.mark.parametrize(
    "backend", [PolicyObservationBuilder, VRJointReferenceAdapter]
)
def test_cpu_fk_deployment_backend_is_rejected(backend):
    args = (
        ({}, ASSET)
        if backend is PolicyObservationBuilder
        else ({}, ASSET, ())
    )
    with pytest.raises(ValueError, match="CUDA"):
        backend(*args, device="cpu")


def test_lower_policy_interface_has_no_vr_joint_schema_dependency():
    import inspect
    import humanoid_policy.observation as lower

    assert "holoretarget" not in inspect.getsource(lower)
    assert (
        "qpos_sequence"
        not in inspect.signature(
            lower.PolicyObservationBuilder.build
        ).parameters
    )
    assert (
        "reference"
        in inspect.signature(lower.PolicyObservationBuilder.build).parameters
    )
