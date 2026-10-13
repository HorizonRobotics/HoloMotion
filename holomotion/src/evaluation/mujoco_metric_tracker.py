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

"""Passive, same-timestamp body metrics for the quantitative model comparison.

The live MuJoCo data is never forwarded or changed. Kinematics are recomputed
from its current qpos on an independent MjData; no policy inputs are returned.
"""

from __future__ import annotations

import numpy as np

from holomotion.src.evaluation.metrics import (
    DEFAULT_ROOT_HEIGHT_THRESHOLD_M,
    compute_position_error,
    compute_root_height_error,
    root_height_success,
)
from holomotion.src.motion_tracking.link_mode_mask import resolve_control_links

PROTOCOL_ID = "common_current_qpos_fk_precontrol_all14_v1_20260914"
ROOT_HEIGHT_SR_STANDARD_ID = "evaluation_v2_root_height_clip_micro_sr_v2"
ROOT_HEIGHT_SR_THRESHOLD_M = DEFAULT_ROOT_HEIGHT_THRESHOLD_M


def normalized_quaternions(values):
    values = np.array(values, dtype=np.float64, copy=True)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(lengths < 1e-12):
        raise ValueError("Common metrics require finite nonzero quaternions")
    return values / lengths


class MujocoMetricTracker:
    def __init__(
        self,
        model,
        all14_link_names,
        active_link_names,
        failure_pos_err_thresh_m=DEFAULT_ROOT_HEIGHT_THRESHOLD_M,
    ):
        import mujoco

        root_height_success(0.0, failure_pos_err_thresh_m)
        self.failure_pos_err_thresh_m = float(failure_pos_err_thresh_m)
        self.model = model
        self.names = list(all14_link_names)
        self.active_names = list(active_link_names)
        if (
            len(self.names) != 14
            or len(set(self.names)) != 14
            or "pelvis" not in self.names
        ):
            raise ValueError(
                "Common metrics require exactly fourteen "
                "unique links including pelvis"
            )
        if not self.active_names or len(set(self.active_names)) != len(
            self.active_names
        ):
            raise ValueError("Active links must be nonempty and unique")
        if not set(self.active_names).issubset(self.names):
            raise ValueError("Active links must belong to all14")
        self.control_mask = np.asarray(
            [name in self.active_names for name in self.names], dtype=bool
        )
        self.body_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in self.names
            ],
            dtype=np.int64,
        )
        if np.any(self.body_ids < 1):
            raise ValueError("MuJoCo plant is missing a common metric link")
        if int(model.jnt_type[0]) != 0 or int(model.jnt_qposadr[0]) != 0:
            raise ValueError(
                "Common root-height metric requires root freejoint at qpos0"
            )
        self.root_index = self.names.index("pelvis")
        self.active_indices = np.asarray(
            [self.names.index(name) for name in self.active_names]
        )
        self.scratch = mujoco.MjData(model)
        self.rows = []

    @classmethod
    def from_config(cls, model, all14_link_names, config):
        """Use the named-point configuration shared with control inputs."""
        active, _ = resolve_control_links(
            all14_link_names, **config.get("link_eval", {})
        )
        return cls(
            model,
            all14_link_names,
            active,
            failure_pos_err_thresh_m=config.get(
                "failure_pos_err_thresh_m", DEFAULT_ROOT_HEIGHT_THRESHOLD_M
            ),
        )

    def observe(
        self,
        live_data,
        reference_positions,
        reference_quaternions_wxyz,
        frame_index,
    ):
        import mujoco

        if frame_index != len(self.rows):
            raise ValueError(
                "Common metric frames must be consecutive starting from zero"
            )
        reference_positions = np.array(
            reference_positions, dtype=np.float64, copy=True
        )
        reference_quaternions_wxyz = np.array(
            reference_quaternions_wxyz, dtype=np.float64, copy=True
        )
        if reference_positions.shape != (
            14,
            3,
        ) or reference_quaternions_wxyz.shape != (14, 4):
            raise ValueError(
                "Common reference must have positions[14,3] "
                "and quaternions[14,4]"
            )
        if (
            not np.isfinite(reference_positions).all()
            or not np.isfinite(live_data.qpos).all()
        ):
            raise ValueError("Common metrics require finite positions")
        self.scratch.qpos[:] = live_data.qpos
        mujoco.mj_kinematics(self.model, self.scratch)
        actual_positions = np.array(
            self.scratch.xpos[self.body_ids], dtype=np.float64, copy=True
        )
        actual_quaternions = normalized_quaternions(
            self.scratch.xquat[self.body_ids]
        )
        target_quaternions = normalized_quaternions(reference_quaternions_wxyz)
        local_errors = compute_position_error(
            actual_positions, reference_positions, root_index=self.root_index
        )
        world_errors = compute_position_error(
            actual_positions, reference_positions
        )
        rotation_errors = 2 * np.arccos(
            np.clip(
                np.abs(
                    np.sum(actual_quaternions * target_quaternions, axis=-1)
                ),
                0,
                1,
            )
        )
        row = {
            "all14_local_link_pos_error_m": float(local_errors.mean()),
            "all14_world_link_pos_error_m": float(world_errors.mean()),
            "all14_link_rot_error_rad": float(rotation_errors.mean()),
            "active_local_link_pos_error_m": float(
                local_errors[self.active_indices].mean()
            ),
            "active_world_link_pos_error_m": float(
                world_errors[self.active_indices].mean()
            ),
            "active_link_rot_error_rad": float(
                rotation_errors[self.active_indices].mean()
            ),
            "root_height_error_m": float(
                compute_root_height_error(
                    live_data.qpos[2], reference_positions[self.root_index, 2]
                )
            ),
        }
        if not all(np.isfinite(v) for v in row.values()):
            raise ValueError("Common metric result is nonfinite")
        self.rows.append(row)

    def report(self, expected_steps):
        metrics = (
            {
                name: float(np.mean([row[name] for row in self.rows]))
                for name in self.rows[0]
            }
            if self.rows
            else {}
        )
        coverage_complete = len(self.rows) == expected_steps
        max_root_height_error_m = (
            max(row["root_height_error_m"] for row in self.rows)
            if self.rows
            else None
        )
        return {
            "protocol_id": PROTOCOL_ID,
            "coverage_complete": coverage_complete,
            "expected_steps": int(expected_steps),
            "observed_steps": len(self.rows),
            "metrics": metrics,
            "all14_link_names": self.names,
            "active_link_names": self.active_names,
            "root_height_SR": {
                "metric_standard_id": ROOT_HEIGHT_SR_STANDARD_ID,
                "threshold_m": self.failure_pos_err_thresh_m,
                "max_root_height_error_m": max_root_height_error_m,
                "height_tracking_success": bool(
                    coverage_complete
                    and max_root_height_error_m is not None
                    and root_height_success(
                        max_root_height_error_m, self.failure_pos_err_thresh_m
                    )
                ),
            },
        }
