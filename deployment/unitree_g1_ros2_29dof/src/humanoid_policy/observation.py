"""HoloMotion v2.0 policy observation assembly (CUDA-only; source-independent)."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from humanoid_policy.fk_warp import WarpLinkFK
from humanoid_policy.link_reference import DeviceLinkReference


class WarpCudaObservation:
    """Expose a device buffer without copying it or knowing its producer."""

    is_cuda = True

    def __init__(self, array) -> None:
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def buffer_ptr(self) -> int:
        return int(self.array.ptr)

    def synchronize(self) -> None:
        wp.synchronize_device(self.array.device)


@wp.func
def _quat_at(source: wp.array(dtype=wp.float32), start: int):
    return wp.normalize(
        wp.quat(
            source[start + 1],
            source[start + 2],
            source[start + 3],
            source[start],
        )
    )


@wp.func
def _reference_rotation(
    source: wp.array3d(dtype=wp.float32), frame: int, link: int
):
    return wp.normalize(
        wp.quat(
            source[frame, link, 1],
            source[frame, link, 2],
            source[frame, link, 3],
            source[frame, link, 0],
        )
    )


@wp.kernel
def _gather_measured_joints(
    state: wp.array(dtype=wp.float32),
    indices: wp.array(dtype=wp.int32),
    joints: wp.array2d(dtype=wp.float32),
):
    joint = wp.tid()
    joints[0, joint] = state[4 + indices[joint]]


@wp.kernel
def _validate_link_reference(
    positions: wp.array3d(dtype=wp.float32),
    rotations: wp.array3d(dtype=wp.float32),
    root: wp.array(dtype=wp.float32),
    time_valid: wp.array(dtype=wp.float32),
    source_valid: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
):
    frame, link = wp.tid()
    if time_valid[frame] != 0.0 and time_valid[frame] != 1.0:
        wp.atomic_min(valid, 0, 0)
    if frame == 0 and link == 0:
        count = float(0.0)
        for i in range(time_valid.shape[0]):
            count = count + time_valid[i]
        if count <= 0.0 or source_valid[0] != 1:
            wp.atomic_min(valid, 0, 0)
        for axis in range(3):
            if not wp.isfinite(root[axis]):
                wp.atomic_min(valid, 0, 0)
    if time_valid[frame] == 1.0 and mask[link] == 1.0:
        for axis in range(3):
            if not wp.isfinite(positions[frame, link, axis]):
                wp.atomic_min(valid, 0, 0)
        length_squared = float(0.0)
        for axis in range(4):
            value = rotations[frame, link, axis]
            if not wp.isfinite(value):
                wp.atomic_min(valid, 0, 0)
            length_squared = length_squared + value * value
        if length_squared < 1.0e-12:
            wp.atomic_min(valid, 0, 0)


@wp.kernel
def _pack_links(
    reference_positions: wp.array3d(dtype=wp.float32),
    reference_rotations: wp.array3d(dtype=wp.float32),
    reference_root: wp.array(dtype=wp.float32),
    time_valid: wp.array(dtype=wp.float32),
    offsets: wp.array(dtype=wp.float32),
    state: wp.array(dtype=wp.float32),
    measured_positions: wp.array2d(dtype=wp.vec3),
    measured_rotations: wp.array2d(dtype=wp.quat),
    link_indices: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    defaults: wp.array(dtype=wp.float32),
    frames: int,
    links: int,
    pelvis: int,
    output: wp.array2d(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    current = 94 + links + frames
    value = float(0.0)
    if index < 3:
        value = state[62 + index]
    elif index < 6:
        value = state[65 + index - 3]
    elif index < 35:
        value = state[4 + index - 6] - defaults[index - 6]
    elif index < 64:
        value = state[33 + index - 35]
    elif index < 93:
        value = state[68 + index - 64]
    elif index == 93:
        if mask[pelvis] == 1.0:
            value = reference_root[2]
    elif index < 94 + links:
        value = mask[index - 94]
    elif index < current:
        value = time_valid[index - 94 - links]
    elif index < current + frames * links * 18:
        local = index - current
        term = int(0)
        dimension = int(3)
        if local >= frames * links * 6:
            dimension = 6
            term = 2
            local = local - frames * links * 6
            if local >= frames * links * 6:
                term = 3
                local = local - frames * links * 6
        elif local >= frames * links * 3:
            term = 1
            local = local - frames * links * 3
        frame = local // (links * dimension)
        link = (local // dimension) % links
        axis = local % dimension
        if mask[link] == 1.0 and time_valid[frame] == 1.0:
            body = link_indices[link]
            robot_inv = wp.quat_inverse(_quat_at(state, 0))
            world = wp.vec3(
                reference_positions[frame, link, 0],
                reference_positions[frame, link, 1],
                reference_positions[frame, link, 2],
            )
            root = wp.vec3(
                reference_root[0], reference_root[1], reference_root[2]
            )
            target = wp.quat_rotate(robot_inv, world - root)
            target_rotation = robot_inv * _reference_rotation(
                reference_rotations, frame, link
            )
            if term == 0:
                value = target[axis]
            elif term == 1:
                value = (target - measured_positions[0, body])[axis]
            else:
                if term == 3:
                    target_rotation = target_rotation * wp.quat_inverse(
                        measured_rotations[0, body]
                    )
                target_rotation = wp.normalize(target_rotation)
                basis = wp.vec3(1.0, 0.0, 0.0)
                if axis >= 3:
                    basis = wp.vec3(0.0, 0.0, 1.0)
                value = wp.quat_rotate(target_rotation, basis)[axis % 3]
    else:
        offset_index = index - current - frames * links * 18
        value = offsets[offset_index] * time_valid[offset_index]
    output[0, index] = value
    if not wp.isfinite(value):
        wp.atomic_min(valid, 0, 0)


class PolicyObservationBuilder:
    """One CUDA interface for all masks, independent of the reference source.

    Reference FK belongs to the source adapter. This class only runs measured
    robot-state FK and packs the saved actor layout. Only measured host state
    and a one-integer validity receipt cross the host boundary; reference and
    actor tensors remain on CUDA. There is no CPU FK deployment backend.
    """

    def __init__(self, contract, asset_path, *, device="cuda:0"):
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError(
                "Link deployment requires CUDA; CPU FK fallback is unsupported"
            )
        asset_path = Path(asset_path)
        if (
            hashlib.sha256(asset_path.read_bytes()).hexdigest()
            != contract["asset_sha256"]
        ):
            raise ValueError("FK asset changed since bundle preflight")
        if (
            not contract["include_error_terms"]
            or contract["include_future_link_mask"]
        ):
            raise ValueError("Unsupported GPU link actor schema")
        self.offsets = tuple(contract["frame_offsets"])
        self.links = tuple(contract["link_names"])
        self.frames = len(self.offsets)
        self.actor_dim = (
            94
            + len(self.links)
            + self.frames
            + 18 * self.frames * len(self.links)
            + self.frames
        )
        if self.actor_dim != contract["actor_dim"]:
            raise ValueError("GPU actor schema width mismatch")
        self.modes = contract["modes"]
        self.fk = WarpLinkFK(
            asset_path,
            joint_names=contract["config_joint_names"],
            batch_size=1,
            device=self.device,
        )

        def array(value, dtype):
            return wp.array(
                np.ascontiguousarray(value), dtype=dtype, device=self.device
            )

        self.measured_indices = array(
            np.array(contract["onnx_index_for_config_joint"], dtype=np.int32),
            wp.int32,
        )
        self.link_indices = array(
            np.array(
                [self.fk.body_names.index(name) for name in self.links],
                dtype=np.int32,
            ),
            wp.int32,
        )
        self.defaults = array(
            np.array(contract["default_joint_pos_onnx"], dtype=np.float32),
            wp.float32,
        )
        self.frame_offsets = array(
            np.array(self.offsets, dtype=np.float32), wp.float32
        )
        self.mode_masks = {
            name: array(np.array(mask, dtype=np.float32), wp.float32)
            for name, mask in self.modes.items()
        }
        self.q = wp.empty((1, 29), dtype=wp.float32, device=self.device)
        self.output = wp.empty(
            (1, self.actor_dim), dtype=wp.float32, device=self.device
        )
        self.observation = WarpCudaObservation(self.output)
        self.host_state = wp.empty(
            97, dtype=wp.float32, device="cpu", pinned=True
        )
        self.host_numpy = np.ctypeslib.as_array(
            (ctypes.c_float * 97).from_address(self.host_state.ptr)
        )
        self.state = wp.empty(97, dtype=wp.float32, device=self.device)
        self.valid = wp.empty(1, dtype=wp.int32, device=self.device)
        self.source_valid_default = wp.ones(
            1, dtype=wp.int32, device=self.device
        )
        self.valid_host = wp.empty(
            1, dtype=wp.int32, device="cpu", pinned=True
        )
        self.valid_numpy = np.ctypeslib.as_array(
            (ctypes.c_int32 * 1).from_address(self.valid_host.ptr)
        )
        self.pelvis = self.links.index("pelvis")
        self.default_numpy = np.array(
            contract["default_joint_pos_onnx"], dtype=np.float32
        )

    def warmup(self, reference: DeviceLinkReference):
        """Compile all masks before live entry; do not run the policy."""
        for mode in self.modes:
            self.build(
                mode=mode,
                reference=reference,
                robot_root_quat_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
                robot_dof_pos_onnx=self.default_numpy,
                robot_dof_vel_onnx=np.zeros(29, dtype=np.float32),
                projected_gravity=np.array([0, 0, -1], dtype=np.float32),
                root_ang_vel_local=np.zeros(3, dtype=np.float32),
                last_action_onnx=np.zeros(29, dtype=np.float32),
            )

    def build(
        self,
        *,
        mode,
        reference: DeviceLinkReference,
        robot_root_quat_wxyz,
        robot_dof_pos_onnx,
        robot_dof_vel_onnx,
        projected_gravity,
        root_ang_vel_local,
        last_action_onnx,
    ):
        if mode not in self.modes:
            raise ValueError("Unknown GPU link mode")
        reference.validate_layout(
            link_names=self.links,
            frame_offsets=self.offsets,
            device=self.device,
        )
        values = (
            (robot_root_quat_wxyz, 4),
            (robot_dof_pos_onnx, 29),
            (robot_dof_vel_onnx, 29),
            (projected_gravity, 3),
            (root_ang_vel_local, 3),
            (last_action_onnx, 29),
        )
        start = 0
        for value, width in values:
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError("Invalid measured robot state")
            self.host_numpy[start : start + width] = value
            start += width
        if np.linalg.norm(self.host_numpy[:4]) < 1.0e-6:
            raise ValueError("Invalid measured root quaternion")
        wp.copy(self.state, self.host_state)
        self.valid.fill_(1)
        source_valid = (
            reference.source_valid
            if reference.source_valid is not None
            else self.source_valid_default
        )
        wp.launch(
            _validate_link_reference,
            dim=(self.frames, len(self.links)),
            inputs=[
                reference.positions,
                reference.rotations_wxyz,
                reference.root_position,
                reference.time_valid,
                source_valid,
                self.mode_masks[mode],
                self.valid,
            ],
            device=self.device,
        )
        wp.launch(
            _gather_measured_joints,
            dim=29,
            inputs=[self.state, self.measured_indices],
            outputs=[self.q],
            device=self.device,
        )
        positions, rotations = self.fk.launch(self.q)
        wp.launch(
            _pack_links,
            dim=self.actor_dim,
            inputs=[
                reference.positions,
                reference.rotations_wxyz,
                reference.root_position,
                reference.time_valid,
                self.frame_offsets,
                self.state,
                positions,
                rotations,
                self.link_indices,
                self.mode_masks[mode],
                self.defaults,
                self.frames,
                len(self.links),
                self.pelvis,
            ],
            outputs=[self.output, self.valid],
            device=self.device,
        )
        wp.copy(self.valid_host, self.valid)
        wp.synchronize_device(self.device)
        if self.valid_numpy[0] != 1:
            raise ValueError("Invalid GPU reference/FK observation")
        return self.observation
