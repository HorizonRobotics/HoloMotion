# This file is copied from:
#   https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/common/command_helper.py
# Copyright (c) Unitree Robotics
# Licensed under the BSD 3-Clause License (see https://github.com/unitreerobotics/unitree_rl_gym/blob/main/LICENSE)
#
# Unmodified copy. All credit to the original authors.

from typing import Union

from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG


class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def create_damping_cmd(cmd: Union[LowCmdGo, LowCmdHG]):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 8
        cmd.motor_cmd[i].tau = 0


def create_zero_cmd(cmd: Union[LowCmdGo, LowCmdHG]):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def init_cmd_hg(cmd: LowCmdHG, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def init_cmd_go(cmd: LowCmdGo, weak_motor: list):
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    pos_stop_f = 2.146e9
    vel_stop_f = 16000.0
    size = len(cmd.motor_cmd)
    for i in range(size):
        if i in weak_motor:
            cmd.motor_cmd[i].mode = 1
        else:
            cmd.motor_cmd[i].mode = 0x0A
        cmd.motor_cmd[i].q = pos_stop_f
        cmd.motor_cmd[i].qd = vel_stop_f
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0
