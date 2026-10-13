"""Batched measured-state FK shared by training and simulation deployment.

Reuses TrainingReferenceFK's asset parsers, but
computes only pelvis-local positions and WXYZ orientations. Tree topology is
resolved on CPU once; forward has no tensor-to-Python reads or derivatives.
"""

from collections.abc import Sequence
from pathlib import Path

import torch

from holomotion.src.motion_tracking.link_observation import (
    quat_apply_wxyz,
    quat_multiply_wxyz,
)


class SharedLinkFK(torch.nn.Module):
    """Forward kinematics from absolute joint angles in radians.

    Args:
        robot_file_path: The same MJCF/URDF used for reference FK.
        device: Device for static buffers and the returned tensors.
        joint_names: Optional order of input q. Names are mapped once during
            construction; otherwise q is already ordered as self.dof_names.

    Returns from forward:
        (positions[B,K,3], quaternions_wxyz[B,K,4]) in self.body_names order.
        K=30 and q has 29 coordinates for the standard G1 asset. The root
        always has position zero and identity orientation. Actual world root
        translation/orientation and temporal samples are neither used nor
        needed.

    Input q must be ABSOLUTE angles, not q minus the default posture.
    Non-finite joint inputs are not hidden here: the observation builder must
    propagate their affected links into its explicit effective validity.
    """

    def __init__(
        self,
        robot_file_path: str,
        device: torch.device | str = "cpu",
        *,
        joint_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        # Lazy import: XML/URDF parsing dependencies are initialization only.
        from holomotion.src.training.reference_fk import (
            MJCFParser,
            URDFParser,
        )

        suffix = Path(robot_file_path).suffix.lower()
        if suffix == ".urdf":
            parser = URDFParser(robot_file_path)
        elif suffix in (".xml", ".mjcf"):
            parser = MJCFParser(robot_file_path)
        else:
            raise ValueError(f"Unsupported robot file extension: {suffix}")
        (
            body_names,
            parents,
            translations,
            rotations,
            _body_joint_order,
            _joint_body_index,
            _joint_axis,
            dof_names,
            dof_axis,
            body_dof_indices,
        ) = parser.parse()
        self.robot_file_path = str(robot_file_path)
        self.body_names = tuple(body_names)
        self.dof_names = tuple(dof_names)
        self.num_bodies = len(body_names)
        self.num_dof = len(dof_names)
        self._parents_cpu = tuple(parents.tolist())
        self.input_joint_names = (
            self.dof_names if joint_names is None else tuple(joint_names)
        )
        if (
            len(self.input_joint_names) != self.num_dof
            or len(set(self.input_joint_names)) != self.num_dof
            or set(self.input_joint_names) != set(self.dof_names)
        ):
            raise ValueError("joint_names must exactly cover each model DOF")
        input_indices = [
            self.input_joint_names.index(name) for name in self.dof_names
        ]
        self.register_buffer(
            "_input_indices", torch.tensor(input_indices, dtype=torch.long)
        )
        axis_norm = dof_axis.norm(dim=-1, keepdim=True)
        if not torch.isfinite(dof_axis).all() or (axis_norm <= 0).any():
            raise ValueError("Joint axes must be finite and nonzero")
        rotations = rotations / rotations.norm(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        self.register_buffer("_translations", translations)
        self.register_buffer("_rotations", rotations)
        self.register_buffer("_dof_axis", dof_axis / axis_norm)

        max_body_dofs = max(map(len, body_dof_indices), default=0)
        dof_indices = torch.zeros(
            self.num_bodies, max_body_dofs, dtype=torch.long
        )
        dof_valid = torch.zeros(
            self.num_bodies, max_body_dofs, dtype=torch.bool
        )
        for body, indices in enumerate(body_dof_indices):
            if indices:
                dof_indices[body, : len(indices)] = torch.tensor(indices)
                dof_valid[body, : len(indices)] = True
        self._max_body_dofs = max_body_dofs
        self.register_buffer("_body_dof_indices", dof_indices)
        self.register_buffer("_body_dof_valid", dof_valid)

        # Each depth's parents are entirely in the preceding depth. All body
        # index lookups below are CPU construction, never GPU scalar reads.
        depth = []
        for body, parent in enumerate(self._parents_cpu):
            if parent < 0:
                if body != 0:
                    raise ValueError("Expected one root as the first body")
                depth.append(0)
            else:
                if parent >= body:
                    raise ValueError("Body topology must be parent-first")
                depth.append(depth[parent] + 1)
        levels = [
            [body for body, d in enumerate(depth) if d == level]
            for level in range(max(depth) + 1)
        ]
        self._num_levels = len(levels)
        for level in range(1, self._num_levels):
            bodies = levels[level]
            parent_slots = [
                levels[level - 1].index(self._parents_cpu[body])
                for body in bodies
            ]
            self.register_buffer(
                f"_level_bodies_{level}",
                torch.tensor(bodies, dtype=torch.long),
            )
            self.register_buffer(
                f"_level_parent_slots_{level}",
                torch.tensor(parent_slots, dtype=torch.long),
            )
        depth_order = [body for level in levels for body in level]
        self.register_buffer(
            "_restore_body_order",
            torch.tensor(
                [depth_order.index(body) for body in range(self.num_bodies)],
                dtype=torch.long,
            ),
        )
        self.to(device=device)

    def forward(
        self, absolute_q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pelvis-local body poses from [B,D] absolute joint angles."""
        if absolute_q.ndim != 2 or absolute_q.shape[-1] != self.num_dof:
            raise ValueError(f"absolute_q must have shape [B,{self.num_dof}]")
        q = absolute_q.to(
            device=self._translations.device,
            dtype=self._translations.dtype,
        ).index_select(1, self._input_indices)
        half_q = q.unsqueeze(-1) * 0.5
        dof_quat = torch.cat(
            (torch.cos(half_q), torch.sin(half_q) * self._dof_axis),
            dim=-1,
        )
        local_rot = self._rotations.unsqueeze(0).expand(q.shape[0], -1, -1)
        identity = torch.cat(
            (
                torch.ones_like(local_rot[..., :1]),
                torch.zeros_like(local_rot[..., 1:]),
            ),
            dim=-1,
        )
        for slot in range(self._max_body_dofs):
            joint_rot = dof_quat.index_select(
                1, self._body_dof_indices[:, slot]
            )
            joint_rot = torch.where(
                self._body_dof_valid[:, slot][None, :, None],
                joint_rot,
                identity,
            )
            local_rot = quat_multiply_wxyz(local_rot, joint_rot)

        positions = [q.new_zeros((q.shape[0], 1, 3))]
        rotations = [identity[:, :1]]
        for level in range(1, self._num_levels):
            bodies = getattr(self, f"_level_bodies_{level}")
            parent_slots = getattr(self, f"_level_parent_slots_{level}")
            parent_pos = positions[-1].index_select(1, parent_slots)
            parent_quat = rotations[-1].index_select(1, parent_slots)
            offset = self._translations.index_select(0, bodies)
            positions.append(parent_pos + quat_apply_wxyz(parent_quat, offset))
            rotations.append(
                quat_multiply_wxyz(
                    parent_quat, local_rot.index_select(1, bodies)
                )
            )
        pos = torch.cat(positions, dim=1).index_select(
            1, self._restore_body_order
        )
        quat = torch.cat(rotations, dim=1).index_select(
            1, self._restore_body_order
        )
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return pos, quat
