"""Warp FK with the exact asset topology and joint order of SharedLinkFK.

CPU asset parsing is initialization only, not a forward backend. Forward
inputs/outputs remain on CUDA; root-relative positions are metres, quaternions are XYZW
(Warp convention). Training's WXYZ convention is converted only at boundaries.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from holomotion.src.motion_tracking.link_fk import SharedLinkFK


@wp.kernel
def _link_fk(
    q: wp.array2d(dtype=wp.float32),
    input_indices: wp.array(dtype=wp.int32),
    translations: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quat),
    axes: wp.array(dtype=wp.vec3),
    body_dofs: wp.array2d(dtype=wp.int32),
    chains: wp.array2d(dtype=wp.int32),
    depths: wp.array(dtype=wp.int32),
    slots: int,
    positions: wp.array2d(dtype=wp.vec3),
    orientations: wp.array2d(dtype=wp.quat),
):
    batch, body = wp.tid()
    position = wp.vec3(0.0, 0.0, 0.0)
    rotation = wp.quat(0.0, 0.0, 0.0, 1.0)
    for level in range(depths[body]):
        current = chains[body, level]
        local_rotation = rotations[current]
        for slot in range(slots):
            joint = body_dofs[current, slot]
            if joint >= 0:
                local_rotation = local_rotation * wp.quat_from_axis_angle(
                    axes[joint], q[batch, input_indices[joint]]
                )
        position = position + wp.quat_rotate(rotation, translations[current])
        rotation = rotation * local_rotation
    positions[batch, body] = position
    orientations[batch, body] = wp.normalize(rotation)


class WarpLinkFK:
    """Preallocated batched device FK, using SharedLinkFK's validated parser.

    q is [batch,29] in joint_names order, absolute joint angles in radians.
    Output buffers are owned by this instance and reused on the next launch.
    No CPU transfer or synchronization is performed by launch().
    """

    def __init__(
        self, asset_path, *, joint_names, batch_size, device="cuda:0"
    ):
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError(
                "Deployment FK requires CUDA; CPU FK fallback is unsupported"
            )
        source = SharedLinkFK(
            str(asset_path), device="cpu", joint_names=joint_names
        )
        self.body_names = source.body_names
        self.dof_names = source.dof_names
        self.input_joint_names = source.input_joint_names
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.num_bodies = source.num_bodies
        self.num_dof = source.num_dof
        paths = []
        for body in range(source.num_bodies):
            path = []
            current = body
            while current > 0:
                path.append(current)
                current = source._parents_cpu[current]
            paths.append(list(reversed(path)))
        chain = np.zeros(
            (source.num_bodies, max(map(len, paths))), dtype=np.int32
        )
        for body, path in enumerate(paths):
            chain[body, : len(path)] = path
        dofs = source._body_dof_indices.numpy().astype(np.int32)
        dofs[~source._body_dof_valid.numpy()] = -1
        self.slots = source._max_body_dofs

        def array(value, dtype):
            return wp.array(
                np.ascontiguousarray(value), dtype=dtype, device=self.device
            )

        self.indices = array(source._input_indices.numpy(), wp.int32)
        self.translations = array(source._translations.numpy(), wp.vec3)
        self.rotations = array(
            source._rotations.numpy()[:, [1, 2, 3, 0]], wp.quat
        )
        self.axes = array(source._dof_axis.numpy(), wp.vec3)
        self.body_dofs = array(dofs, wp.int32)
        self.chains = array(chain, wp.int32)
        self.depths = array(
            np.array(list(map(len, paths)), dtype=np.int32), wp.int32
        )
        self.positions = wp.empty(
            (self.batch_size, self.num_bodies),
            dtype=wp.vec3,
            device=self.device,
        )
        self.orientations = wp.empty(
            (self.batch_size, self.num_bodies),
            dtype=wp.quat,
            device=self.device,
        )

    def launch(self, q):
        if q.shape != (self.batch_size, self.num_dof) or q.dtype != wp.float32:
            raise ValueError(
                "FK requires preallocated float32 [batch,29] joint angles"
            )
        if q.device != self.device:
            raise ValueError("FK input must be on the selected device")
        wp.launch(
            _link_fk,
            dim=(self.batch_size, self.num_bodies),
            inputs=[
                q,
                self.indices,
                self.translations,
                self.rotations,
                self.axes,
                self.body_dofs,
                self.chains,
                self.depths,
                self.slots,
            ],
            outputs=[self.positions, self.orientations],
            device=self.device,
        )
        return self.positions, self.orientations
