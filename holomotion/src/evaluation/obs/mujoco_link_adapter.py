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

"""Saved link-policy observation adapter using the shared training packing."""

import numpy as np
import torch

from holomotion.src.motion_tracking.link_actor_packing import (
    LINK_CURRENT_TERMS,
    link_actor_atomic_terms,
    link_actor_layout,
)
from holomotion.src.motion_tracking.link_fk import SharedLinkFK
from holomotion.src.motion_tracking.link_mode_mask import resolve_control_links
from holomotion.src.motion_tracking.link_observation import (
    LinkObservationInput,
    build_link_observation,
    quat_apply_wxyz,
    quat_multiply_wxyz,
)


class MujocoLinkObsAdapter:
    """Measured-FK adapter, with no actor world-position input.

    Uses the saved schema and reference FK asset. One command snapshot is
    prepared per policy step, so atomic terms cannot disagree within a frame.
    """

    def __init__(self, evaluator, link_config, eval_options, term_specs):
        self.evaluator = evaluator
        version = link_config.get("schema_version", "g1_link_xz6d_v1")
        if version != "g1_link_xz6d_v1":
            raise ValueError(
                f"Unsupported saved link schema version: {version}"
            )
        self.link_names = tuple(link_config["link_names"])
        self.offsets = tuple(int(v) for v in link_config["frame_offsets"])
        saved_schema = (
            evaluator.config.get("algo", {})
            .get("config", {})
            .get("module_dict", {})
            .get("actor", {})
            .get("obs_schema")
        )
        self.layout = link_actor_layout(link_config, saved_schema)
        self.include_error_terms = self.layout["include_error_terms"]
        self.include_future_link_mask = self.layout["include_future_link_mask"]
        self.actor_dim = self.layout["actor_dim"]
        if (
            len(self.link_names) != 14
            or len(set(self.link_names)) != 14
            or "pelvis" not in self.link_names
        ):
            raise ValueError(
                "The current link actor schema requires 14 unique "
                "named links including pelvis"
            )
        if (
            len(self.offsets) < 1
            or len(set(self.offsets)) != len(self.offsets)
            or min(self.offsets) <= 0
        ):
            raise ValueError(
                "The link actor requires distinct positive future "
                "frame offsets"
            )
        if (
            evaluator.num_actions != 29
            or abs(evaluator.policy_dt - 0.02) > 1e-8
        ):
            raise ValueError(
                "The current link actor schema requires 29 DOF and "
                "50 Hz control"
            )
        names = tuple(spec["name"] for spec in term_specs)
        if names != self.layout["atomic_terms"]:
            raise ValueError(
                f"Saved link actor atomic order is incompatible: {names}"
            )
        for spec in term_specs:
            if (
                int(spec.get("history_length", 0)) > 1
                or float(spec.get("scale", 1.0)) != 1.0
            ):
                raise ValueError(
                    "Link actor v1 expects atomic history <= 1 and "
                    "unit scale; KV history is in the model"
                )
        _, mask = resolve_control_links(self.link_names, **eval_options)
        self.requested_mask = torch.tensor(mask, dtype=torch.float32)[None]
        self.root_link_index = self.link_names.index("pelvis")
        self.reference_body_names = tuple(evaluator.config.robot.body_names)
        self.ref_indices = [
            self.reference_body_names.index(n) for n in self.link_names
        ]
        self.ref_root_index = self.reference_body_names.index("pelvis")
        self.fk = None
        self.fk_indices = []
        if self.include_error_terms:
            fk_path = str(evaluator.config.robot.motion.fk_robot_file_path)
            self.fk = SharedLinkFK(
                fk_path, device="cpu", joint_names=evaluator.dof_names_onnx
            )
            self.fk_indices = [
                self.fk.body_names.index(n) for n in self.link_names
            ]
        self.atomic = {}

    def reset(self):
        self.atomic = {}

    def prepare(self):
        ev = self.evaluator
        if (
            ev.n_motion_frames <= 0
            or getattr(ev, "ref_global_translation", None) is None
            or getattr(ev, "ref_global_rotation_quat_xyzw", None) is None
        ):
            raise ValueError(
                "Link control requires a reference with global link "
                "positions and XYZW rotations"
            )
        ev._ensure_ref_to_sim_transform_rigid()
        current = int(ev.motion_frame_idx)
        if not 0 <= current < ev.n_motion_frames:
            raise ValueError("Current reference frame outside the motion clip")
        requested = current + np.asarray(self.offsets, dtype=np.int64)
        valid = requested < ev.n_motion_frames
        indices = np.minimum(requested, ev.n_motion_frames - 1)
        ref_pos = torch.as_tensor(
            ev.ref_global_translation[indices][:, self.ref_indices],
            dtype=torch.float32,
        )[None]
        xyzw = torch.as_tensor(
            ev.ref_global_rotation_quat_xyzw[indices][:, self.ref_indices],
            dtype=torch.float32,
        )[None]
        ref_quat = xyzw[..., [3, 0, 1, 2]]
        align_q = torch.as_tensor(ev._ref_to_sim_q_wxyz, dtype=torch.float32)
        align_t = torch.as_tensor(ev._ref_to_sim_t, dtype=torch.float32)
        ref_pos = quat_apply_wxyz(align_q, ref_pos) + align_t
        ref_quat = quat_multiply_wxyz(align_q, ref_quat)
        root_raw = torch.as_tensor(
            ev.ref_global_translation[current, self.ref_root_index],
            dtype=torch.float32,
        )[None]
        ref_root = quat_apply_wxyz(align_q, root_raw) + align_t
        proprio = [
            np.asarray(
                getattr(ev, f"_get_obs_{name}")(), dtype=np.float32
            ).reshape(-1)
            for name in LINK_CURRENT_TERMS[:5]
        ]
        local_pos = local_quat = None
        if self.include_error_terms:
            absolute_q = torch.as_tensor(
                proprio[2] + ev.default_angles_onnx, dtype=torch.float32
            )[None]
            local_pos, local_quat = self.fk(absolute_q)
            local_pos = local_pos[:, self.fk_indices]
            local_quat = local_quat[:, self.fk_indices]
        # IMU-equivalent orientation, without root translation.
        root_quat = torch.as_tensor(
            ev.robot_global_bodylink_rot[ev.root_body_idx], dtype=torch.float32
        )[None]
        command = build_link_observation(
            LinkObservationInput(
                reference_link_pos=ref_pos,
                reference_link_quat_wxyz=ref_quat,
                reference_root_pos=ref_root,
                robot_root_quat_wxyz=root_quat,
                robot_link_pos_local=local_pos,
                robot_link_quat_local_wxyz=local_quat,
                frame_offsets=torch.tensor(self.offsets, dtype=torch.float32),
                link_mask=self.requested_mask,
                time_valid=torch.as_tensor(valid, dtype=torch.float32)[None],
                reference_root_height=root_raw[:, 2:3],
            ),
            root_link_index=self.root_link_index,
            include_error_terms=self.include_error_terms,
        )
        self.atomic = {
            name: tensor[0].detach().cpu().numpy().reshape(-1)
            for name, tensor in link_actor_atomic_terms(
                command,
                self.include_error_terms,
                self.include_future_link_mask,
            ).items()
        }
        self.atomic.update(
            {
                name: item
                for name, item in zip(
                    LINK_CURRENT_TERMS[:5], proprio, strict=True
                )
            }
        )

    def validate_packed(self, packed):
        if packed.shape != (self.actor_dim,) or not np.isfinite(packed).all():
            raise ValueError(
                f"Input {packed.shape}: expected finite {self.actor_dim}D"
            )
