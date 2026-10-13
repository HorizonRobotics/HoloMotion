"""GPU joint-to-link adapter for the existing VR reference application.

This compatibility layer consumes the existing 36D VR/retarget queue. The
policy interface itself consumes DeviceLinkReference and does not depend on
Pico, HoloRetarget, or this queue format. FK is CUDA-only; there is no CPU FK
fallback. Joint names are supplied by the application rather than imported
from a particular retarget implementation.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import numpy as np
import warp as wp

from humanoid_policy.fk_warp import WarpLinkFK
from humanoid_policy.link_reference import DeviceLinkReference


@wp.kernel
def _gather_reference_joints(
    sequence: wp.array3d(dtype=wp.float32),
    joint_indices: wp.array(dtype=wp.int32),
    joints: wp.array2d(dtype=wp.float32),
):
    frame, joint = wp.tid()
    joints[frame, joint] = sequence[0, frame + 2, 7 + joint_indices[joint]]


@wp.kernel
def _validate_vr_sequence(
    sequence: wp.array3d(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
):
    frame, column = wp.tid()
    if not wp.isfinite(sequence[0, frame, column]):
        wp.atomic_min(valid, 0, 0)
    if column == 3 and frame >= 2 and frame < sequence.shape[1] - 1:
        quat = wp.quat(
            sequence[0, frame, 4],
            sequence[0, frame, 5],
            sequence[0, frame, 6],
            sequence[0, frame, 3],
        )
        if wp.length(quat) < 1.0e-6:
            wp.atomic_min(valid, 0, 0)


@wp.kernel
def _reference_links_in_world(
    sequence: wp.array3d(dtype=wp.float32),
    yaw: wp.array(dtype=wp.float32),
    local_positions: wp.array2d(dtype=wp.vec3),
    local_rotations: wp.array2d(dtype=wp.quat),
    link_indices: wp.array(dtype=wp.int32),
    positions: wp.array3d(dtype=wp.float32),
    rotations: wp.array3d(dtype=wp.float32),
    root_position: wp.array(dtype=wp.float32),
):
    frame, link = wp.tid()
    body = link_indices[link]
    alignment = wp.normalize(wp.quat(yaw[1], yaw[2], yaw[3], yaw[0]))
    current = wp.vec3(sequence[0, 1, 0], sequence[0, 1, 1], sequence[0, 1, 2])
    future = wp.vec3(
        sequence[0, frame + 2, 0],
        sequence[0, frame + 2, 1],
        sequence[0, frame + 2, 2],
    )
    rotation = alignment * wp.normalize(
        wp.quat(
            sequence[0, frame + 2, 4],
            sequence[0, frame + 2, 5],
            sequence[0, frame + 2, 6],
            sequence[0, frame + 2, 3],
        )
    )
    aligned_root = current + wp.quat_rotate(alignment, future - current)
    world_pos = aligned_root + wp.quat_rotate(
        rotation, local_positions[frame, body]
    )
    world_rot = rotation * local_rotations[frame, body]
    for axis in range(3):
        positions[frame, link, axis] = world_pos[axis]
        if frame == 0 and link == 0:
            root_position[axis] = current[axis]
    rotations[frame, link, 0] = world_rot[3]
    rotations[frame, link, 1] = world_rot[0]
    rotations[frame, link, 2] = world_rot[1]
    rotations[frame, link, 3] = world_rot[2]


class VRJointReferenceAdapter:
    """Adapt the existing [past,current,future...,tail] queue on CUDA.

    Each 36D frame contains root world XYZ, WXYZ, and 29 absolute joint
    angles in ``reference_joint_names`` order. Motion-entry yaw calibration
    is applied here; robot-frame observation transforms remain downstream.
    """

    def __init__(
        self, contract, asset_path, reference_joint_names, *, device="cuda:0"
    ):
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError(
                "VR joint-to-link FK requires CUDA; CPU fallback is unsupported"
            )
        asset_path = Path(asset_path)
        if (
            hashlib.sha256(asset_path.read_bytes()).hexdigest()
            != contract["asset_sha256"]
        ):
            raise ValueError("FK asset changed since bundle preflight")
        self.offsets = tuple(contract["frame_offsets"])
        self.frames = len(self.offsets)
        if self.offsets != tuple(range(1, self.frames + 1)):
            raise ValueError(
                "Existing VR queue requires consecutive future offsets"
            )
        self.links = tuple(contract["link_names"])
        config_names = tuple(contract["config_joint_names"])
        source_names = tuple(reference_joint_names)
        if (
            len(source_names) != 29
            or len(set(source_names)) != 29
            or set(source_names) != set(config_names)
        ):
            raise ValueError("VR reference and model joint-name sets differ")
        self.fk = WarpLinkFK(
            asset_path,
            joint_names=config_names,
            batch_size=self.frames,
            device=self.device,
        )
        self.joint_indices = wp.array(
            np.array(
                [source_names.index(name) for name in config_names],
                dtype=np.int32,
            ),
            dtype=wp.int32,
            device=self.device,
        )
        self.link_indices = wp.array(
            np.array(
                [self.fk.body_names.index(name) for name in self.links],
                dtype=np.int32,
            ),
            dtype=wp.int32,
            device=self.device,
        )
        self.joints = wp.empty(
            (self.frames, 29), dtype=wp.float32, device=self.device
        )
        self.yaw = wp.empty(4, dtype=wp.float32, device=self.device)
        self.yaw_host = wp.empty(
            4, dtype=wp.float32, device="cpu", pinned=True
        )
        self.yaw_numpy = np.ctypeslib.as_array(
            (ctypes.c_float * 4).from_address(self.yaw_host.ptr)
        )
        shape = (1, self.frames + 3, 36)
        self.sequence = wp.empty(shape, dtype=wp.float32, device=self.device)
        self.sequence_host = wp.empty(
            shape, dtype=wp.float32, device="cpu", pinned=True
        )
        self.sequence_numpy = np.ctypeslib.as_array(
            (ctypes.c_float * ((self.frames + 3) * 36)).from_address(
                self.sequence_host.ptr
            )
        ).reshape(shape)
        self.reference = DeviceLinkReference(
            link_names=self.links,
            frame_offsets=self.offsets,
            positions=wp.empty(
                (self.frames, len(self.links), 3),
                dtype=wp.float32,
                device=self.device,
            ),
            rotations_wxyz=wp.empty(
                (self.frames, len(self.links), 4),
                dtype=wp.float32,
                device=self.device,
            ),
            root_position=wp.empty(3, dtype=wp.float32, device=self.device),
            time_valid=wp.ones(
                self.frames, dtype=wp.float32, device=self.device
            ),
            source_valid=wp.ones(1, dtype=wp.int32, device=self.device),
        )
        config_defaults = np.asarray(
            contract["default_joint_pos_onnx"], dtype=np.float32
        )[list(contract["onnx_index_for_config_joint"])]
        self.reference_defaults = config_defaults[
            [config_names.index(name) for name in source_names]
        ]

    def from_host_sequence(self, sequence, **kwargs) -> DeviceLinkReference:
        """Upload a host-origin VR queue; FK and subsequent transforms stay on CUDA."""
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.shape != (self.frames + 3, 36):
            raise ValueError("Invalid VR reference shape")
        self.sequence_numpy[0] = sequence
        wp.copy(self.sequence, self.sequence_host)
        return self.from_device_sequence(self.sequence, **kwargs)

    def from_device_sequence(
        self,
        sequence,
        *,
        yaw_alignment_wxyz=(1.0, 0.0, 0.0, 0.0),
    ) -> DeviceLinkReference:
        if (
            sequence.shape != (1, self.frames + 3, 36)
            or sequence.dtype != wp.float32
        ):
            raise ValueError(
                "Invalid VR reference; expected float32 [1,F+3,36]"
            )
        if sequence.device != self.device:
            raise ValueError("VR reference must stay on the FK CUDA device")
        yaw = np.asarray(yaw_alignment_wxyz, dtype=np.float32)
        if (
            yaw.shape != (4,)
            or not np.isfinite(yaw).all()
            or np.linalg.norm(yaw) < 1.0e-6
        ):
            raise ValueError("Invalid VR yaw alignment")
        self.yaw_numpy[:] = yaw
        wp.copy(self.yaw, self.yaw_host)
        self.reference.source_valid.fill_(1)
        wp.launch(
            _validate_vr_sequence,
            dim=(self.frames + 3, 36),
            inputs=[sequence, self.reference.source_valid],
            device=self.device,
        )
        wp.launch(
            _gather_reference_joints,
            dim=(self.frames, 29),
            inputs=[sequence, self.joint_indices],
            outputs=[self.joints],
            device=self.device,
        )
        positions, rotations = self.fk.launch(self.joints)
        wp.launch(
            _reference_links_in_world,
            dim=(self.frames, len(self.links)),
            inputs=[
                sequence,
                self.yaw,
                positions,
                rotations,
                self.link_indices,
            ],
            outputs=[
                self.reference.positions,
                self.reference.rotations_wxyz,
                self.reference.root_position,
            ],
            device=self.device,
        )
        return self.reference

    def warmup_reference(self) -> DeviceLinkReference:
        """Compile the adapter on a synthetic reference; do not run a policy."""
        sequence = np.zeros((self.frames + 3, 36), dtype=np.float32)
        sequence[:, 2] = 0.78
        sequence[:, 3] = 1.0
        sequence[:, 7:] = self.reference_defaults
        return self.from_host_sequence(sequence)
