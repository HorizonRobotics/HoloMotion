# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from __future__ import annotations

import os
import xml.etree.ElementTree as ETree
from typing import Dict, List, Tuple

import torch
import pytorch_kinematics as pk

from loguru import logger
from holomotion.src.utils import torch_utils


class MJCFParser:
    def __init__(self, robot_file_path: str) -> None:
        self._robot_file_path = robot_file_path

    @staticmethod
    def parse_vec(
        text: str | None, size: int, default: List[float]
    ) -> List[float]:
        if text is None:
            return list(default)
        values = [float(v) for v in text.strip().split()]
        if len(values) != size:
            raise ValueError(
                f"Expected {size} values, got {len(values)} in '{text}'"
            )
        return values

    @staticmethod
    def _find_parent(
        root: ETree.Element, child: ETree.Element
    ) -> ETree.Element | None:
        for parent in root.iter():
            for node in list(parent):
                if node is child:
                    return parent
        return None

    @staticmethod
    def _select_include_children(
        parent: ETree.Element, inc_root: ETree.Element
    ) -> List[ETree.Element]:
        if inc_root.tag == "mujoco":
            if parent.tag != "mujoco":
                sub = inc_root.find(parent.tag)
                if sub is not None:
                    return list(sub)
            return list(inc_root)
        if inc_root.tag == parent.tag:
            return list(inc_root)
        return list(inc_root)

    @classmethod
    def _resolve_includes(cls, root: ETree.Element, base_dir: str) -> None:
        includes = root.findall(".//include")
        while includes:
            for inc in includes:
                inc_file = inc.attrib.get("file")
                if inc_file is None:
                    raise ValueError("Include tag missing 'file' attribute")
                inc_path = os.path.join(base_dir, inc_file)
                inc_root = ETree.parse(inc_path).getroot()
                cls._resolve_includes(inc_root, os.path.dirname(inc_path))
                parent = cls._find_parent(root, inc)
                if parent is None:
                    raise ValueError("Failed to resolve include parent")
                insert_children = cls._select_include_children(
                    parent, inc_root
                )
                insert_index = list(parent).index(inc)
                for child in list(insert_children):
                    parent.insert(insert_index, child)
                    insert_index += 1
                parent.remove(inc)
            includes = root.findall(".//include")

    def load_root(self) -> ETree.Element:
        root = ETree.parse(self._robot_file_path).getroot()
        self._resolve_includes(root, os.path.dirname(self._robot_file_path))
        return root

    def parse(
        self,
    ) -> Tuple[
        List[str],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[List[str]],
        Dict[str, int],
        Dict[str, List[float]],
        List[str],
        torch.Tensor,
        List[List[int]],
    ]:
        root = self.load_root()
        xml_world = root.find("worldbody")
        if xml_world is None:
            raise ValueError("MJCF missing worldbody")
        xml_body_root = xml_world.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF missing root body")

        body_names: List[str] = []
        parents: List[int] = []
        local_translation: List[List[float]] = []
        local_rotation: List[List[float]] = []
        body_joint_order: List[List[str]] = []
        joint_body_index: Dict[str, int] = {}
        joint_axis: Dict[str, List[float]] = {}

        def _add_body(xml_body: ETree.Element, parent_index: int) -> None:
            body_idx = len(body_names)
            body_names.append(xml_body.attrib.get("name", ""))
            parents.append(parent_index)
            local_translation.append(
                self.parse_vec(xml_body.attrib.get("pos"), 3, [0.0, 0.0, 0.0])
            )
            local_rotation.append(
                self.parse_vec(
                    xml_body.attrib.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]
                )
            )
            joints_in_body: List[str] = []
            for joint in xml_body.findall("joint"):
                joint_name = joint.attrib.get("name")
                if joint_name is None:
                    raise ValueError("Joint missing name")
                joint_type = joint.attrib.get("type", "hinge")
                if joint_type == "free":
                    continue
                if joint_type != "hinge":
                    raise ValueError(f"Unsupported joint type: {joint_type}")
                axis = self.parse_vec(
                    joint.attrib.get("axis"), 3, [0.0, 0.0, 1.0]
                )
                joint_body_index[joint_name] = body_idx
                joint_axis[joint_name] = axis
                joints_in_body.append(joint_name)
            body_joint_order.append(joints_in_body)
            for child in xml_body.findall("body"):
                _add_body(child, body_idx)

        _add_body(xml_body_root, -1)
        if local_translation:
            local_translation[0] = [0.0, 0.0, 0.0]
            local_rotation[0] = [1.0, 0.0, 0.0, 0.0]

        dof_names: List[str] = []
        for elem in root.iter():
            if elem.tag == "actuator":
                for child in list(elem):
                    joint_name = child.attrib.get("joint")
                    if joint_name is not None:
                        dof_names.append(joint_name)
        if len(dof_names) == 0:
            raise ValueError("No actuated joints found in MJCF")
        dof_axis: List[List[float]] = []
        for joint_name in dof_names:
            if joint_name not in joint_body_index:
                raise ValueError(f"Actuator joint not found: {joint_name}")
            dof_axis.append(joint_axis[joint_name])

        dof_name_to_index = {name: idx for idx, name in enumerate(dof_names)}
        body_dof_indices: List[List[int]] = []
        for joints in body_joint_order:
            indices: List[int] = []
            for name in joints:
                if name in dof_name_to_index:
                    indices.append(dof_name_to_index[name])
            body_dof_indices.append(indices)

        return (
            body_names,
            torch.tensor(parents, dtype=torch.long),
            torch.tensor(local_translation, dtype=torch.float32),
            torch.tensor(local_rotation, dtype=torch.float32),
            body_joint_order,
            joint_body_index,
            joint_axis,
            dof_names,
            torch.tensor(dof_axis, dtype=torch.float32),
            body_dof_indices,
        )


class URDFParser:
    def __init__(self, urdf_path: str) -> None:
        self._urdf_path = urdf_path

    @staticmethod
    def _as_tf(
        tf: torch.Tensor | None, identity: torch.Tensor
    ) -> torch.Tensor:
        if tf is None:
            return identity
        if tf.ndim == 3:
            return tf[0]
        return tf

    def _load_chain(self) -> pk.Chain:
        with open(self._urdf_path, mode="r", encoding="utf-8") as f:
            urdf_text = f.read()
        return pk.build_chain_from_urdf(urdf_text)

    def parse(
        self,
    ) -> Tuple[
        List[str],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[List[str]],
        Dict[str, int],
        Dict[str, List[float]],
        List[str],
        torch.Tensor,
        List[List[int]],
    ]:
        pk_chain = self._load_chain()
        dof_names = pk_chain.get_joint_parameter_names()
        if len(dof_names) == 0:
            raise ValueError("No actuated joints found in URDF")
        dof_axis = pk_chain.axes.to(dtype=torch.float32)

        root_name = pk_chain._root.name
        moving_frames = pk_chain.get_frame_names(exclude_fixed=True)
        body_names = [root_name] + [
            name for name in moving_frames if name != root_name
        ]
        body_name_to_index = {name: idx for idx, name in enumerate(body_names)}

        num_frames = len(pk_chain.idx_to_frame)
        frame_name_to_index = {
            name: idx for idx, name in pk_chain.idx_to_frame.items()
        }

        full_parent_indices: List[int] = []
        for i in range(num_frames):
            chain_indices = pk_chain.parents_indices[i]
            if chain_indices.numel() <= 1:
                full_parent_indices.append(-1)
            else:
                full_parent_indices.append(int(chain_indices[-2].item()))

        identity = torch.eye(4, dtype=torch.float32)
        frame_transforms: List[torch.Tensor] = [identity] * num_frames
        for i in range(num_frames):
            link_offset = self._as_tf(pk_chain.link_offsets[i], identity)
            joint_offset = self._as_tf(pk_chain.joint_offsets[i], identity)
            if i == 0:
                link_offset = identity
                joint_offset = identity
            parent = full_parent_indices[i]
            if parent < 0:
                frame_tf = identity
            else:
                frame_tf = (
                    frame_transforms[parent] @ link_offset @ joint_offset
                )
            frame_transforms[i] = frame_tf

        parents: List[int] = []
        local_translation: List[List[float]] = []
        local_rotation_mat: List[torch.Tensor] = []
        body_joint_order: List[List[str]] = []
        joint_body_index: Dict[str, int] = {}
        joint_axis: Dict[str, List[float]] = {}

        for body_name in body_names:
            frame_idx = frame_name_to_index[body_name]
            parent_frame_idx = full_parent_indices[frame_idx]
            parent_body_idx = -1
            while parent_frame_idx >= 0:
                parent_name = pk_chain.idx_to_frame[parent_frame_idx]
                if parent_name in body_name_to_index:
                    parent_body_idx = body_name_to_index[parent_name]
                    break
                parent_frame_idx = full_parent_indices[parent_frame_idx]
            parents.append(parent_body_idx)

            if parent_body_idx < 0:
                local_tf = identity
            else:
                local_tf = (
                    torch.linalg.inv(frame_transforms[parent_frame_idx])
                    @ frame_transforms[frame_idx]
                )
            local_translation.append(local_tf[:3, 3].tolist())
            local_rotation_mat.append(local_tf[:3, :3])

            joints_in_body: List[str] = []
            joint_index = int(pk_chain.joint_indices[frame_idx].item())
            if joint_index >= 0:
                joint_type = int(pk_chain.joint_type_indices[frame_idx].item())
                if joint_type != 1:
                    raise ValueError(
                        f"Unsupported joint type index: {joint_type}"
                    )
                joint_name = dof_names[joint_index]
                joints_in_body.append(joint_name)
                joint_body_index[joint_name] = body_name_to_index[body_name]
                joint_axis[joint_name] = dof_axis[joint_index].tolist()
            body_joint_order.append(joints_in_body)

        local_rotation = torch_utils.quat_from_matrix(
            torch.stack(local_rotation_mat, dim=0)
        )

        dof_name_to_index = {name: idx for idx, name in enumerate(dof_names)}
        body_dof_indices: List[List[int]] = []
        for joints in body_joint_order:
            indices: List[int] = []
            for name in joints:
                if name in dof_name_to_index:
                    indices.append(dof_name_to_index[name])
            body_dof_indices.append(indices)

        return (
            body_names,
            torch.tensor(parents, dtype=torch.long),
            torch.tensor(local_translation, dtype=torch.float32),
            local_rotation.to(dtype=torch.float32),
            body_joint_order,
            joint_body_index,
            joint_axis,
            dof_names,
            dof_axis,
            body_dof_indices,
        )


# @torch.compile(dynamic=True)
class HoloMotionFK(torch.nn.Module):
    def __init__(
        self,
        robot_file_path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.robot_file_path = robot_file_path
        _, ext = os.path.splitext(robot_file_path)
        ext = ext.lower()
        if ext == ".urdf":
            parser = URDFParser(robot_file_path)
        elif ext in [".xml", ".mjcf"]:
            parser = MJCFParser(robot_file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        logger.info(
            f"Parsing robot file for online forward kinematics: {robot_file_path}..."
        )

        (
            body_names,
            parents,
            local_translation,
            local_rotation,
            body_joint_order,
            joint_body_index,
            joint_axis,
            dof_names,
            dof_axis,
            body_dof_indices,
        ) = parser.parse()
        self.body_names = body_names
        self.dof_names = dof_names
        self.num_bodies = len(body_names)
        self.num_dof = len(dof_names)
        parents = parents.to(device=device)
        local_translation = local_translation.to(device=device, dtype=dtype)
        local_rotation = local_rotation.to(device=device, dtype=dtype)
        local_rotation_mat = torch_utils.matrix_from_quat(local_rotation)
        dof_axis = dof_axis.to(device=device, dtype=dtype)
        max_body_dofs = max(
            (len(indices) for indices in body_dof_indices), default=0
        )
        body_dof_index_tensor = torch.full(
            (self.num_bodies, max_body_dofs),
            -1,
            dtype=torch.long,
        )
        body_dof_mask = torch.zeros(
            (self.num_bodies, max_body_dofs), dtype=torch.bool
        )
        for body_idx, indices in enumerate(body_dof_indices):
            if not indices:
                continue
            body_dof_index_tensor[body_idx, : len(indices)] = torch.tensor(
                indices, dtype=torch.long
            )
            body_dof_mask[body_idx, : len(indices)] = True
        self.register_buffer("_parents", parents)
        self.register_buffer("_local_translation", local_translation)
        self.register_buffer("_local_rotation_mat", local_rotation_mat)
        self.register_buffer("_dof_axis", dof_axis)
        self.register_buffer("_body_dof_index_tensor", body_dof_index_tensor)
        self.register_buffer("_body_dof_mask", body_dof_mask)
        self._body_joint_order = body_joint_order
        self._joint_body_index = joint_body_index
        self._joint_axis = joint_axis
        self._body_dof_indices = body_dof_indices

    @torch.no_grad()
    def forward(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        dof_pos: torch.Tensor,
        fps: float,
        quat_format: str = "xyzw",
        sub_batch_size: int = 64,
        vel_smoothing_sigma: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward kinematics and smoothed velocities.

        Args:
            root_pos: (B, T, 3)
            root_quat: (B, T, 4), XYZW by default
            dof_pos: (B, T, ndof)
            fps: frames per second
            sub_batch_size: split batch into chunks to reduce peak memory
            vel_smoothing_sigma: Gaussian sigma for smoothing velocity signals
                along the time axis (set <= 0 to disable).

        Returns:
            Dict with global_translation/global_rotation_quat/global_velocity/
            global_angular_velocity/dof_pos/dof_vel.
        """
        if fps <= 0.0:
            raise ValueError(f"Invalid fps: {fps}")
        if root_pos.ndim != 3 or root_quat.ndim != 3 or dof_pos.ndim != 3:
            raise ValueError("Inputs must be (B, T, ...)")
        if (
            root_pos.shape[:2] != root_quat.shape[:2]
            or root_pos.shape[:2] != dof_pos.shape[:2]
        ):
            raise ValueError("Mismatched batch/time shapes among inputs")
        if root_pos.shape[-1] != 3 or root_quat.shape[-1] != 4:
            raise ValueError(
                "root_pos must be (B,T,3) and root_quat must be (B,T,4)"
            )
        if dof_pos.shape[-1] != self.num_dof:
            raise ValueError(
                f"dof_pos last dim {dof_pos.shape[-1]} does not match "
                f"{self.num_dof}"
            )

        device = self._local_translation.device
        dtype = self._local_translation.dtype
        root_pos = root_pos.to(device=device, dtype=dtype)
        root_quat = root_quat.to(device=device, dtype=dtype)
        dof_pos = dof_pos.to(device=device, dtype=dtype)

        batch_size, seq_len = root_pos.shape[:2]
        if (
            sub_batch_size is None
            or sub_batch_size <= 0
            or sub_batch_size >= batch_size
        ):
            return self._forward_impl(
                root_pos=root_pos,
                root_quat=root_quat,
                dof_pos=dof_pos,
                fps=fps,
                quat_format=quat_format,
                vel_smoothing_sigma=float(vel_smoothing_sigma),
            )

        global_translation = torch.empty(
            (batch_size, seq_len, self.num_bodies, 3),
            device=device,
            dtype=dtype,
        )
        global_rotation_quat = torch.empty(
            (batch_size, seq_len, self.num_bodies, 4),
            device=device,
            dtype=dtype,
        )
        global_velocity = torch.empty_like(global_translation)
        global_angular_velocity = torch.empty_like(global_translation)
        dof_pos_out = torch.empty_like(dof_pos)
        dof_vel = torch.empty_like(dof_pos)

        for start in range(0, batch_size, sub_batch_size):
            end = min(start + sub_batch_size, batch_size)
            out = self._forward_impl(
                root_pos=root_pos[start:end],
                root_quat=root_quat[start:end],
                dof_pos=dof_pos[start:end],
                fps=fps,
                quat_format=quat_format,
                vel_smoothing_sigma=float(vel_smoothing_sigma),
            )
            global_translation[start:end] = out["global_translation"]
            global_rotation_quat[start:end] = out["global_rotation_quat"]
            global_velocity[start:end] = out["global_velocity"]
            global_angular_velocity[start:end] = out["global_angular_velocity"]
            dof_pos_out[start:end] = out["dof_pos"]
            dof_vel[start:end] = out["dof_vel"]

        return {
            "global_translation": global_translation,
            "global_rotation_quat": global_rotation_quat,
            "global_velocity": global_velocity,
            "global_angular_velocity": global_angular_velocity,
            "dof_pos": dof_pos_out,
            "dof_vel": dof_vel,
        }

    def _forward_impl(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        dof_pos: torch.Tensor,
        fps: float,
        quat_format: str,
        vel_smoothing_sigma: float,
    ) -> Dict[str, torch.Tensor]:
        device = self._local_translation.device
        dtype = self._local_translation.dtype
        if quat_format == "xyzw":
            root_quat_wxyz = torch_utils.xyzw_to_wxyz(root_quat)
        elif quat_format == "wxyz":
            root_quat_wxyz = root_quat
        else:
            raise ValueError(f"Unsupported quat_format: {quat_format}")

        root_rotmat = torch_utils.matrix_from_quat(root_quat_wxyz)
        dof_rotmats = torch_utils.axis_angle_to_matrix(dof_pos, self._dof_axis)

        positions_world = torch.empty(
            (dof_pos.shape[0], dof_pos.shape[1], self.num_bodies, 3),
            device=device,
            dtype=dtype,
        )
        rotations_world = torch.empty(
            (dof_pos.shape[0], dof_pos.shape[1], self.num_bodies, 3, 3),
            device=device,
            dtype=dtype,
        )

        for i in range(self.num_bodies):
            parent = int(self._parents[i].item())
            if parent < 0:
                positions_world[:, :, i] = root_pos
                rotations_world[:, :, i] = root_rotmat
                continue
            parent_pos = positions_world[:, :, parent]
            parent_rot = rotations_world[:, :, parent]
            offset = self._local_translation[i]
            pos = parent_pos + torch.einsum("btij,j->bti", parent_rot, offset)
            rot = torch.matmul(parent_rot, self._local_rotation_mat[i])
            body_dof_indices = self._body_dof_indices[i]
            for dof_idx in body_dof_indices:
                rot = torch.matmul(rot, dof_rotmats[:, :, dof_idx])
            positions_world[:, :, i] = pos
            rotations_world[:, :, i] = rot

        global_translation = positions_world
        global_rotation_mat = rotations_world
        global_quat_wxyz = torch_utils.quat_from_matrix(global_rotation_mat)
        global_quat_xyzw = torch_utils.wxyz_to_xyzw(global_quat_wxyz)

        dt = 1.0 / fps
        if dof_pos.shape[1] < 2:
            dof_vel = torch.zeros_like(dof_pos)
        else:
            diff = (dof_pos[:, 1:] - dof_pos[:, :-1]) / dt
            pad = diff[:, -2:-1] if diff.shape[1] >= 2 else diff[:, -1:]
            dof_vel = torch.cat([diff, pad], dim=1)
        dof_vel = torch_utils.smooth_time_series(
            dof_vel, sigma=float(vel_smoothing_sigma), dim=1
        )

        global_velocity = torch_utils.grad_t(global_translation, dt)
        global_velocity = torch_utils.smooth_time_series(
            global_velocity, sigma=float(vel_smoothing_sigma), dim=1
        )

        if global_quat_xyzw.shape[1] < 2:
            global_angular_velocity = torch.zeros_like(global_translation)
        else:
            q1 = torch_utils.xyzw_to_wxyz(global_quat_xyzw[:, 1:])
            q0_inv = torch_utils.quat_conjugate(
                torch_utils.xyzw_to_wxyz(global_quat_xyzw[:, :-1])
            )
            q_rel = torch_utils.quat_mul(q1, q0_inv)
            q_rel = q_rel / torch.linalg.norm(q_rel, dim=-1, keepdim=True)
            q_rel = torch_utils.standardize_quaternion(q_rel)

            identity = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
            )[None, None, None]
            q_rel_full = identity.expand(
                global_quat_xyzw.shape[0],
                global_quat_xyzw.shape[1],
                global_quat_xyzw.shape[2],
                4,
            ).clone()
            q_rel_full[:, :-1] = q_rel
            global_angular_velocity = (
                torch_utils.axis_angle_from_quat(q_rel_full, w_last=False) / dt
            )
            global_angular_velocity = torch_utils.smooth_time_series(
                global_angular_velocity,
                sigma=float(vel_smoothing_sigma),
                dim=1,
            )

        return {
            "global_translation": global_translation,
            "global_rotation_quat": global_quat_xyzw,
            "global_velocity": global_velocity,
            "global_angular_velocity": global_angular_velocity,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
        }


# class HoloMotionFK_V2(torch.nn.Module):
#     """
#     Use pytorch_kinematics to compute FK.
#     """

#     def __init__(
#         self,
#         robot_file_path: str,
#         device: torch.device | str = "cpu",
#         dtype: torch.dtype = torch.float32,
#     ) -> None:
#         super().__init__()
#         self.robot_file_path = robot_file_path
#         urdf_path = os.path.splitext(robot_file_path)[0] + ".urdf"
#         if not os.path.isfile(urdf_path):
#             raise FileNotFoundError(f"URDF not found: {urdf_path}")
#         with open(urdf_path, mode="r", encoding="utf-8") as f:
#             urdf_text = f.read()

#         pk_chain = pk.build_chain_from_urdf(urdf_text)
#         pk_chain = pk_chain.to(dtype=dtype, device=device)

#         self.dof_names = pk_chain.get_joint_parameter_names()
#         self.num_dof = len(self.dof_names)
#         root_name = pk_chain._root.name
#         moving_frames = pk_chain.get_frame_names(exclude_fixed=True)
#         self.body_names = [root_name] + [
#             name for name in moving_frames if name != root_name
#         ]
#         self.num_bodies = len(self.body_names)

#         body_frame_indices = pk_chain.get_frame_indices(*self.body_names)
#         self.register_buffer("_body_frame_indices", body_frame_indices)

#         num_frames = len(pk_chain.idx_to_frame)
#         identity = torch.eye(4, device=device, dtype=dtype)
#         link_offsets = []
#         joint_offsets = []
#         for i in range(num_frames):
#             link_offset = pk_chain.link_offsets[i]
#             joint_offset = pk_chain.joint_offsets[i]
#             if link_offset is None:
#                 link_offset = identity
#             if joint_offset is None:
#                 joint_offset = identity
#             if link_offset.ndim == 3:
#                 link_offset = link_offset[0]
#             if joint_offset.ndim == 3:
#                 joint_offset = joint_offset[0]
#             link_offsets.append(link_offset)
#             joint_offsets.append(joint_offset)
#         if num_frames > 0:
#             link_offsets[0] = identity
#             joint_offsets[0] = identity

#         parent_indices: List[int] = []
#         for i in range(num_frames):
#             chain_indices = pk_chain.parents_indices[i]
#             if chain_indices.numel() <= 1:
#                 parent_indices.append(-1)
#             else:
#                 parent_indices.append(int(chain_indices[-2].item()))

#         self.register_buffer("_pk_axes", pk_chain.axes)
#         self.register_buffer(
#             "_pk_joint_type_indices", pk_chain.joint_type_indices
#         )
#         self.register_buffer("_pk_joint_indices", pk_chain.joint_indices)
#         self.register_buffer(
#             "_pk_link_offsets", torch.stack(link_offsets, dim=0)
#         )
#         self.register_buffer(
#             "_pk_joint_offsets", torch.stack(joint_offsets, dim=0)
#         )
#         self.register_buffer(
#             "_pk_parent_indices",
#             torch.tensor(parent_indices, dtype=torch.long, device=device),
#         )
#         self._num_frames = num_frames

#     def forward(
#         self,
#         root_pos: torch.Tensor,
#         root_quat: torch.Tensor,
#         dof_pos: torch.Tensor,
#         fps: float,
#         quat_format: str = "xyzw",
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Args:
#             root_pos: (B, T, 3)
#             root_quat: (B, T, 4), XYZW by default
#             dof_pos: (B, T, ndof)
#             fps: frames per second
#         Returns:
#             Dict with global_translation/global_rotation_quat/global_velocity/
#             global_angular_velocity/dof_pos/dof_vel.
#         """
#         if fps <= 0.0:
#             raise ValueError(f"Invalid fps: {fps}")
#         if root_pos.ndim != 3 or root_quat.ndim != 3 or dof_pos.ndim != 3:
#             raise ValueError("Inputs must be (B, T, ...)")
#         if (
#             root_pos.shape[:2] != root_quat.shape[:2]
#             or root_pos.shape[:2] != dof_pos.shape[:2]
#         ):
#             raise ValueError("Mismatched batch/time shapes among inputs")
#         if root_pos.shape[-1] != 3 or root_quat.shape[-1] != 4:
#             raise ValueError(
#                 "root_pos must be (B,T,3) and root_quat must be (B,T,4)"
#             )
#         if dof_pos.shape[-1] != self.num_dof:
#             raise ValueError(
#                 f"dof_pos last dim {dof_pos.shape[-1]} does not match {self.num_dof}"
#             )

#         device = self._pk_axes.device
#         dtype = self._pk_axes.dtype
#         root_pos = root_pos.to(device=device, dtype=dtype)
#         root_quat = root_quat.to(device=device, dtype=dtype)
#         dof_pos = dof_pos.to(device=device, dtype=dtype)

#         if quat_format == "xyzw":
#             root_quat_wxyz = torch_utils.xyzw_to_wxyz(root_quat)
#         elif quat_format == "wxyz":
#             root_quat_wxyz = root_quat
#         else:
#             raise ValueError(f"Unsupported quat_format: {quat_format}")

#         batch_size, seq_len = root_pos.shape[:2]
#         flat_size = batch_size * seq_len
#         root_pos_flat = root_pos.reshape(flat_size, 3)
#         root_quat_flat = root_quat_wxyz.reshape(flat_size, 4)
#         dof_pos_flat = dof_pos.reshape(flat_size, self.num_dof)

#         axes_expanded = self._pk_axes[None].expand(flat_size, -1, -1)
#         revolute_tf = axis_and_angle_to_matrix_44(axes_expanded, dof_pos_flat)
#         prismatic_tf = axis_and_d_to_pris_matrix(axes_expanded, dof_pos_flat)

#         frame_transforms = torch.empty(
#             (flat_size, self._num_frames, 4, 4), device=device, dtype=dtype
#         )
#         identity = torch.eye(4, device=device, dtype=dtype).repeat(
#             flat_size, 1, 1
#         )

#         for i in range(self._num_frames):
#             parent = int(self._pk_parent_indices[i].item())
#             if parent < 0:
#                 frame_tf = identity
#             else:
#                 frame_tf = frame_transforms[:, parent]
#             frame_tf = frame_tf @ self._pk_link_offsets[i]
#             frame_tf = frame_tf @ self._pk_joint_offsets[i]
#             joint_type = int(self._pk_joint_type_indices[i].item())
#             if joint_type == 1:
#                 joint_index = int(self._pk_joint_indices[i].item())
#                 frame_tf = frame_tf @ revolute_tf[:, joint_index]
#             elif joint_type == 2:
#                 joint_index = int(self._pk_joint_indices[i].item())
#                 frame_tf = frame_tf @ prismatic_tf[:, joint_index]
#             frame_transforms[:, i] = frame_tf

#         chain_tf = torch.index_select(
#             frame_transforms, 1, self._body_frame_indices
#         )

#         root_rotmat = torch_utils.matrix_from_quat(root_quat_flat)
#         root_tf = torch.eye(4, device=device, dtype=dtype).repeat(
#             flat_size, 1, 1
#         )
#         root_tf[:, :3, :3] = root_rotmat
#         root_tf[:, :3, 3] = root_pos_flat

#         world_tf = root_tf[:, None] @ chain_tf
#         world_tf = world_tf.reshape(batch_size, seq_len, self.num_bodies, 4, 4)
#         global_translation = world_tf[:, :, :, :3, 3]
#         global_rotation_mat = world_tf[:, :, :, :3, :3]
#         global_quat_wxyz = torch_utils.quat_from_matrix(global_rotation_mat)
#         global_quat_xyzw = torch_utils.wxyz_to_xyzw(global_quat_wxyz)

#         dt = 1.0 / fps
#         if dof_pos.shape[1] < 2:
#             dof_vel = torch.zeros_like(dof_pos)
#         else:
#             diff = (dof_pos[:, 1:] - dof_pos[:, :-1]) / dt
#             pad = diff[:, -2:-1] if diff.shape[1] >= 2 else diff[:, -1:]
#             dof_vel = torch.cat([diff, pad], dim=1)

#         global_velocity = torch_utils.grad_t(global_translation, dt)
#         global_velocity = torch_utils.gaussian_filter1d(
#             global_velocity, sigma=2.0, dim=1
#         )

#         if global_quat_xyzw.shape[1] < 2:
#             global_angular_velocity = torch.zeros_like(global_translation)
#         else:
#             q1 = torch_utils.xyzw_to_wxyz(global_quat_xyzw[:, 1:])
#             q0_inv = torch_utils.quat_conjugate(
#                 torch_utils.xyzw_to_wxyz(global_quat_xyzw[:, :-1])
#             )
#             q_rel = torch_utils.quat_mul(q1, q0_inv)
#             q_rel = q_rel / torch.linalg.norm(q_rel, dim=-1, keepdim=True)
#             q_rel = torch_utils.standardize_quaternion(q_rel)

#             identity = torch.tensor(
#                 [1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
#             )[None, None, None]
#             q_rel_full = identity.expand(
#                 global_quat_xyzw.shape[0],
#                 global_quat_xyzw.shape[1],
#                 global_quat_xyzw.shape[2],
#                 4,
#             ).clone()
#             q_rel_full[:, :-1] = q_rel
#             global_angular_velocity = (
#                 torch_utils.axis_angle_from_quat(q_rel_full, w_last=False) / dt
#             )
#             global_angular_velocity = torch_utils.gaussian_filter1d(
#                 global_angular_velocity,
#                 sigma=2.0,
#                 dim=1,
#             )

#         return {
#             "global_translation": global_translation,
#             "global_rotation_quat": global_quat_xyzw,
#             "global_velocity": global_velocity,
#             "global_angular_velocity": global_angular_velocity,
#             "dof_pos": dof_pos,
#             "dof_vel": dof_vel,
#         }
