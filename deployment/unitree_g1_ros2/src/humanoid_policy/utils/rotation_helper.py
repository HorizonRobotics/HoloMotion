# This file is copied from:
#   https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/common/rotation_helper.py
# Copyright (c) Unitree Robotics
# Licensed under the BSD 3-Clause License (see https://github.com/unitreerobotics/unitree_rl_gym/blob/main/LICENSE)
#
# Unmodified copy. All credit to the original authors.
import numpy as np
from scipy.spatial.transform import Rotation


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    rz_waist = Rotation.from_euler("z", waist_yaw).as_matrix()
    r_torso = Rotation.from_quat(
        [imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]
    ).as_matrix()
    r_pelvis = np.dot(r_torso, rz_waist.T)
    w = np.dot(rz_waist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return Rotation.from_matrix(r_pelvis).as_quat()[[3, 0, 1, 2]], w
