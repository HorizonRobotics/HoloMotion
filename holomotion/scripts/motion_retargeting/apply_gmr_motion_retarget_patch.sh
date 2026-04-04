#!/usr/bin/env bash
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
#
# This file was originally adapted from the [GMR] repository:
# https://github.com/YanjieZe/GMR/blob/master/general_motion_retargeting/motion_retarget.py
#

set -euo pipefail

REPO_ROOT="$(pwd)"
TARGET_FILE="${1:-$REPO_ROOT/thirdparties/GMR/general_motion_retargeting/motion_retarget.py}"

if [[ ! -f "$TARGET_FILE" ]]; then
    echo "Target file not found: $TARGET_FILE" >&2
    exit 1
fi

python - "$TARGET_FILE" <<'PY'
from pathlib import Path
import ast
import sys
import textwrap


PATCH_MARKERS = (
    "self.first_frame_damping = max(float(damping), 2.0)",
    "self.prev_posture_task = mink.PostureTask(self.model, cost=1e-3)",
    "def _solve_task_group(",
)


PATCHED_INIT = """
def __init__(
    self,
    src_human: str,
    tgt_robot: str,
    actual_human_height: float = None,
    solver: str="daqp", # change from "quadprog" to "daqp".
    damping: float=5e-1, # change from 1e-1 to 1e-2.
    verbose: bool=True,
    use_velocity_limit: bool=False,
) -> None:

    # load the robot model
    self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
    if verbose:
        print("Use robot model: ", self.xml_file)
    self.model = mj.MjModel.from_xml_path(self.xml_file)

    # Print DoF names in order
    print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
    self.robot_dof_names = {}
    for i in range(self.model.nv):  # 'nv' is the number of DoFs
        dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
        self.robot_dof_names[dof_name] = i
        if verbose:
            print(f"DoF {i}: {dof_name}")

    print("[GMR] Robot Body names and their IDs:")
    self.robot_body_names = {}
    for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
        self.robot_body_names[body_name] = i
        if verbose:
            print(f"Body ID {i}: {body_name}")

    print("[GMR] Robot Motor (Actuator) names and their IDs:")
    self.robot_motor_names = {}
    for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
        motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        self.robot_motor_names[motor_name] = i
        if verbose:
            print(f"Motor ID {i}: {motor_name}")

    # Load the IK config
    with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
        ik_config = json.load(f)
    if verbose:
        print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])

    # compute the scale ratio based on given human height and the assumption in the IK config
    if actual_human_height is not None:
        ratio = actual_human_height / ik_config["human_height_assumption"]
    else:
        ratio = 1.0

    # adjust the human scale table
    for key in ik_config["human_scale_table"].keys():
        ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio

    # used for retargeting
    self.ik_match_table1 = ik_config["ik_match_table1"]
    self.ik_match_table2 = ik_config["ik_match_table2"]
    self.human_root_name = ik_config["human_root_name"]
    self.robot_root_name = ik_config["robot_root_name"]
    self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
    self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
    self.human_scale_table = ik_config["human_scale_table"]
    self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

    self.max_iter = 10

    self.solver = solver
    self.damping = damping
    self.first_frame_damping = max(float(damping), 2.0)
    self.first_frame_max_iter = max(int(self.max_iter), 10)
    self._is_first_frame = True

    self.human_body_to_task1 = {}
    self.human_body_to_task2 = {}
    self.pos_offsets1 = {}
    self.rot_offsets1 = {}
    self.pos_offsets2 = {}
    self.rot_offsets2 = {}
    self._arm_task_original_orientation_costs = {}
    self._first_frame_arm_orientation_cost = 1.0

    self.task_errors1 = {}
    self.task_errors2 = {}

    self.ik_limits = [mink.ConfigurationLimit(self.model)]
    if use_velocity_limit:
        VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
        self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

    self.setup_retarget_configuration()

    self.ground_offset = 0.0
"""


PATCHED_SETUP = """
def setup_retarget_configuration(self):
    self.configuration = mink.Configuration(self.model)
    self._default_qpos = self.configuration.data.qpos.copy()
    self.posture_task = mink.PostureTask(self.model, cost=1e-2)
    self.posture_task.set_target(self._default_qpos)
    self.prev_posture_task = mink.PostureTask(self.model, cost=1e-3)
    self.prev_posture_task.set_target(self._default_qpos)

    self.tasks1 = []
    self.tasks2 = []

    for frame_name, entry in self.ik_match_table1.items():
        body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
        if pos_weight != 0 or rot_weight != 0:
            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            self.human_body_to_task1[body_name] = task
            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets1[body_name] = R.from_quat(
                rot_offset, scalar_first=True
            )
            self.tasks1.append(task)
            self.task_errors1[task] = []
            if self._is_arm_body(body_name):
                self._arm_task_original_orientation_costs[task] = float(
                    rot_weight
                )

    for frame_name, entry in self.ik_match_table2.items():
        body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
        if pos_weight != 0 or rot_weight != 0:
            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            self.human_body_to_task2[body_name] = task
            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets2[body_name] = R.from_quat(
                rot_offset, scalar_first=True
            )
            self.tasks2.append(task)
            self.task_errors2[task] = []
            if self._is_arm_body(body_name):
                self._arm_task_original_orientation_costs[task] = float(
                    rot_weight
                )
"""


PATCHED_RETARGET_BLOCK = """
@staticmethod
def _is_arm_body(body_name):
    return any(
        token in body_name
        for token in (
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        )
    )

def _set_first_frame_arm_task_costs(self, enabled):
    for task, original_orientation_cost in (
        self._arm_task_original_orientation_costs.items()
    ):
        orientation_cost = (
            self._first_frame_arm_orientation_cost
            if enabled
            else original_orientation_cost
        )
        task.set_orientation_cost(orientation_cost)

def _solve_task_group(
    self,
    tasks,
    error_fn,
    *,
    damping,
    max_iter,
    include_posture,
    include_prev_posture,
):
    solve_tasks = list(tasks)
    if include_posture:
        solve_tasks.append(self.posture_task)
    if include_prev_posture:
        solve_tasks.append(self.prev_posture_task)

    curr_error = error_fn()
    dt = self.configuration.model.opt.timestep
    vel = mink.solve_ik(
        self.configuration,
        solve_tasks,
        dt,
        self.solver,
        damping,
        limits=self.ik_limits,
    )
    self.configuration.integrate_inplace(vel, dt)
    next_error = error_fn()
    num_iter = 0
    while curr_error - next_error > 0.001 and num_iter < max_iter:
        curr_error = next_error
        dt = self.configuration.model.opt.timestep
        vel = mink.solve_ik(
            self.configuration,
            solve_tasks,
            dt,
            self.solver,
            damping,
            limits=self.ik_limits,
        )
        self.configuration.integrate_inplace(vel, dt)
        next_error = error_fn()
        num_iter += 1

def retarget(self, human_data, offset_to_ground=False):
    prev_q = self.configuration.data.qpos.copy()
    # Update the task targets
    self.update_targets(human_data, offset_to_ground)
    include_posture = self._is_first_frame
    include_prev_posture = True
    solve_damping = (
        self.first_frame_damping if self._is_first_frame else self.damping
    )
    solve_max_iter = (
        self.first_frame_max_iter if self._is_first_frame else self.max_iter
    )
    self.prev_posture_task.set_target(prev_q)
    if self._is_first_frame:
        self._set_first_frame_arm_task_costs(True)

    if self.use_ik_match_table1:
        self._solve_task_group(
            self.tasks1,
            self.error1,
            damping=solve_damping,
            max_iter=solve_max_iter,
            include_posture=include_posture,
            include_prev_posture=include_prev_posture,
        )

    if self.use_ik_match_table2:
        self._solve_task_group(
            self.tasks2,
            self.error2,
            damping=solve_damping,
            max_iter=solve_max_iter,
            include_posture=include_posture,
            include_prev_posture=include_prev_posture,
        )

    if self._is_first_frame:
        self._set_first_frame_arm_task_costs(False)
    self._is_first_frame = False
    return self.configuration.data.qpos.copy()
"""


def indent_block(src: str, indent: str = "    ") -> str:
    body = textwrap.dedent(src).strip("\n")
    return "\n".join(indent + line if line else "" for line in body.splitlines()) + "\n"


def find_class(module: ast.Module, class_name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise SystemExit(f"Class {class_name!r} not found in target file.")


def find_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise SystemExit(f"Method {method_name!r} not found in class {class_node.name}.")


def apply_replacement(lines, node: ast.AST, replacement: str):
    start = node.lineno - 1
    end = node.end_lineno
    return lines[:start] + [indent_block(replacement)] + lines[end:]


path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

if all(marker in text for marker in PATCH_MARKERS):
    print(f"Patch already present: {path}")
    raise SystemExit(0)

module = ast.parse(text)
class_node = find_class(module, "GeneralMotionRetargeting")
init_node = find_method(class_node, "__init__")
setup_node = find_method(class_node, "setup_retarget_configuration")
retarget_node = find_method(class_node, "retarget")

lines = text.splitlines(keepends=True)
for node, replacement in sorted(
    [
        (retarget_node, PATCHED_RETARGET_BLOCK),
        (setup_node, PATCHED_SETUP),
        (init_node, PATCHED_INIT),
    ],
    key=lambda item: item[0].lineno,
    reverse=True,
):
    lines = apply_replacement(lines, node, replacement)

new_text = "".join(lines)
ast.parse(new_text)
path.write_text(new_text, encoding="utf-8")
print(f"Patch applied to: {path}")
PY