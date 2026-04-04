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

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MotorFamily:
    name: str
    armature: float
    x1: float
    x2: float
    y1: float
    y2: float
    fs: float
    fd: float
    va: float = 0.01


@dataclass(frozen=True)
class JointSpec:
    joint_expr: str
    motor: MotorFamily
    effort_limit: float
    velocity_limit: float
    servo_scale: float = 1.0
    envelope_scale: float = 1.0
    friction_scale: float = 1.0


# -----------------------------------------------------------------------------
# Base actuator families
# -----------------------------------------------------------------------------

N5020_16 = MotorFamily(
    name="N5020_16",
    armature=0.003609725,
    x1=30.86,
    x2=40.13,
    y1=24.8,
    y2=31.9,
    fs=0.6,
    fd=0.06,
)

N7520_14P3 = MotorFamily(
    name="N7520_14P3",
    armature=0.010177520,
    x1=22.63,
    x2=35.52,
    y1=71.0,
    y2=83.3,
    fs=1.6,
    fd=0.16,
)

N7520_22P5 = MotorFamily(
    name="N7520_22P5",
    armature=0.025101925,
    x1=14.5,
    x2=22.7,
    y1=111.0,
    y2=131.0,
    fs=2.4,
    fd=0.24,
)

W4010_25 = MotorFamily(
    name="W4010_25",
    armature=0.00425,
    x1=15.3,
    x2=24.76,
    y1=4.8,
    y2=8.6,
    fs=0.6,
    fd=0.06,
)


# -----------------------------------------------------------------------------
# Design constants
# -----------------------------------------------------------------------------

NATURAL_FREQ_HZ = 10.0
DAMPING_RATIO = 2.0

# Set this to your actual physics dt before running the generator.
PHYSICS_DT = 1.0 / 200.0

# Desired action delay budget: at most 2 * (1 / 50) = 0.04 s.
MIN_DELAY_SECONDS = 0.0
MAX_DELAY_SECONDS = 2.0 / 50.0


def seconds_to_delay_steps(delay_seconds: float, physics_dt: float) -> int:
    return int(math.floor(delay_seconds / physics_dt + 1e-12))


MIN_DELAY = seconds_to_delay_steps(MIN_DELAY_SECONDS, PHYSICS_DT)
MAX_DELAY = seconds_to_delay_steps(MAX_DELAY_SECONDS, PHYSICS_DT)


# -----------------------------------------------------------------------------
# Single-group mapping
#
# ankle / waist:
# - servo-side armature/gains are doubled
# - torque envelope is NOT doubled
# - friction is NOT doubled
# -----------------------------------------------------------------------------

ALL_JOINT_SPECS: list[JointSpec] = [
    # legs
    JointSpec(
        ".*_hip_yaw_joint", N7520_14P3, effort_limit=88.0, velocity_limit=32.0
    ),
    JointSpec(
        ".*_hip_roll_joint",
        N7520_22P5,
        effort_limit=139.0,
        velocity_limit=20.0,
    ),
    JointSpec(
        ".*_hip_pitch_joint",
        N7520_14P3,
        effort_limit=88.0,
        velocity_limit=32.0,
    ),
    JointSpec(
        ".*_knee_joint", N7520_22P5, effort_limit=139.0, velocity_limit=20.0
    ),
    # feet
    JointSpec(
        ".*_ankle_pitch_joint",
        N5020_16,
        effort_limit=50.0,
        velocity_limit=37.0,
        servo_scale=2.0,
    ),
    JointSpec(
        ".*_ankle_roll_joint",
        N5020_16,
        effort_limit=50.0,
        velocity_limit=37.0,
        servo_scale=2.0,
    ),
    # waist
    JointSpec(
        "waist_roll_joint",
        N5020_16,
        effort_limit=50.0,
        velocity_limit=37.0,
        servo_scale=2.0,
    ),
    JointSpec(
        "waist_pitch_joint",
        N5020_16,
        effort_limit=50.0,
        velocity_limit=37.0,
        servo_scale=2.0,
    ),
    JointSpec(
        "waist_yaw_joint", N7520_14P3, effort_limit=88.0, velocity_limit=32.0
    ),
    # arms
    JointSpec(
        ".*_shoulder_pitch_joint",
        N5020_16,
        effort_limit=25.0,
        velocity_limit=37.0,
    ),
    JointSpec(
        ".*_shoulder_roll_joint",
        N5020_16,
        effort_limit=25.0,
        velocity_limit=37.0,
    ),
    JointSpec(
        ".*_shoulder_yaw_joint",
        N5020_16,
        effort_limit=25.0,
        velocity_limit=37.0,
    ),
    JointSpec(
        ".*_elbow_joint", N5020_16, effort_limit=25.0, velocity_limit=37.0
    ),
    JointSpec(
        ".*_wrist_roll_joint", N5020_16, effort_limit=25.0, velocity_limit=37.0
    ),
    JointSpec(
        ".*_wrist_pitch_joint", W4010_25, effort_limit=5.0, velocity_limit=22.0
    ),
    JointSpec(
        ".*_wrist_yaw_joint", W4010_25, effort_limit=5.0, velocity_limit=22.0
    ),
]


def compute_pd_gains(
    armature: float, natural_freq_hz: float, damping_ratio: float
) -> tuple[float, float]:
    wn = natural_freq_hz * 2.0 * math.pi
    kp = armature * wn * wn
    kd = 2.0 * damping_ratio * armature * wn
    return kp, kd


def fmt_float(x: float) -> str:
    return format(float(x), ".12g")


def fmt_value(value: Any, indent: int = 0) -> str:
    sp = " " * indent

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        for k, v in value.items():
            lines.append(f"{sp}    {k!r}: {fmt_value(v, indent + 4)},")
        lines.append(f"{sp}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        lines = ["["]
        for item in value:
            lines.append(f"{sp}    {fmt_value(item, indent + 4)},")
        lines.append(f"{sp}]")
        return "\n".join(lines)

    if isinstance(value, float):
        return fmt_float(value)

    return repr(value)


def build_single_group_cfg(
    specs: list[JointSpec],
    natural_freq_hz: float = NATURAL_FREQ_HZ,
    damping_ratio: float = DAMPING_RATIO,
    min_delay: int = MIN_DELAY,
    max_delay: int = MAX_DELAY,
) -> dict[str, Any]:
    joint_names_expr = [spec.joint_expr for spec in specs]

    effort_limit: dict[str, float] = {}
    velocity_limit: dict[str, float] = {}
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    armature: dict[str, float] = {}
    x1: dict[str, float] = {}
    x2: dict[str, float] = {}
    y1: dict[str, float] = {}
    y2: dict[str, float] = {}
    fs: dict[str, float] = {}
    fd: dict[str, float] = {}
    va: dict[str, float] = {}
    action_scale: dict[str, float] = {}

    for spec in specs:
        name = spec.joint_expr
        servo_armature = spec.motor.armature * spec.servo_scale
        kp, kd = compute_pd_gains(
            servo_armature, natural_freq_hz, damping_ratio
        )

        effort_limit[name] = spec.effort_limit
        velocity_limit[name] = spec.velocity_limit
        stiffness[name] = kp
        damping[name] = kd
        armature[name] = servo_armature

        x1[name] = spec.motor.x1
        x2[name] = spec.motor.x2
        y1[name] = spec.motor.y1 * spec.envelope_scale
        y2[name] = spec.motor.y2 * spec.envelope_scale
        fs[name] = spec.motor.fs * spec.friction_scale
        fd[name] = spec.motor.fd * spec.friction_scale
        va[name] = spec.motor.va

        action_scale[name] = 0.25 * spec.effort_limit / kp

    return {
        "joint_names_expr": joint_names_expr,
        "min_delay": min_delay,
        "max_delay": max_delay,
        "effort_limit": effort_limit,
        "velocity_limit": velocity_limit,
        "stiffness": stiffness,
        "damping": damping,
        "armature": armature,
        "friction": 0.0,
        "dynamic_friction": 0.0,
        "viscous_friction": 0.0,
        "X1": x1,
        "X2": x2,
        "Y1": y1,
        "Y2": y2,
        "Fs": fs,
        "Fd": fd,
        "Va": va,
        "action_scale": action_scale,
    }


def render_single_group_cfg(
    cfg: dict[str, Any], group_name: str = "all_joints"
) -> str:
    ordered_keys = [
        "joint_names_expr",
        "min_delay",
        "max_delay",
        "effort_limit",
        "velocity_limit",
        "stiffness",
        "damping",
        "armature",
        "friction",
        "dynamic_friction",
        "viscous_friction",
        "X1",
        "X2",
        "Y1",
        "Y2",
        "Fs",
        "Fd",
        "Va",
    ]

    lines = [
        "from unitree_actuators import UnitreeActuatorCfg",
        "",
        "G1_HIFI_ACTUATORS = {",
        f"    {group_name!r}: UnitreeActuatorCfg(",
    ]
    for key in ordered_keys:
        rendered = fmt_value(cfg[key], indent=8)
        lines.append(f"        {key}={rendered},")
    lines.append("    )")
    lines.append("}")
    lines.append("")
    lines.append("G1_HIFI_ACTION_SCALE = {")
    for joint_expr in cfg["joint_names_expr"]:
        lines.append(
            f"    {joint_expr!r}: {fmt_float(cfg['action_scale'][joint_expr])},"
        )
    lines.append("}")
    return "\n".join(lines)


def print_summary(cfg: dict[str, Any]) -> None:
    print("# === SUMMARY ===")
    print(f"# physics_dt = {fmt_float(PHYSICS_DT)}")
    print(f"# min_delay  = {cfg['min_delay']}")
    print(f"# max_delay  = {cfg['max_delay']}")
    print(
        "# joint_expr | effort_limit | velocity_limit | armature | kp | kd | "
        "X1 | X2 | Y1 | Y2 | Fs | Fd | action_scale"
    )
    for joint_expr in cfg["joint_names_expr"]:
        print(
            f"# {joint_expr} | "
            f"{fmt_float(cfg['effort_limit'][joint_expr])} | "
            f"{fmt_float(cfg['velocity_limit'][joint_expr])} | "
            f"{fmt_float(cfg['armature'][joint_expr])} | "
            f"{fmt_float(cfg['stiffness'][joint_expr])} | "
            f"{fmt_float(cfg['damping'][joint_expr])} | "
            f"{fmt_float(cfg['X1'][joint_expr])} | "
            f"{fmt_float(cfg['X2'][joint_expr])} | "
            f"{fmt_float(cfg['Y1'][joint_expr])} | "
            f"{fmt_float(cfg['Y2'][joint_expr])} | "
            f"{fmt_float(cfg['Fs'][joint_expr])} | "
            f"{fmt_float(cfg['Fd'][joint_expr])} | "
            f"{fmt_float(cfg['action_scale'][joint_expr])}"
        )
    print()


def main() -> None:
    cfg = build_single_group_cfg(ALL_JOINT_SPECS)
    print_summary(cfg)
    print(render_single_group_cfg(cfg, group_name="all_joints"))


if __name__ == "__main__":
    main()
