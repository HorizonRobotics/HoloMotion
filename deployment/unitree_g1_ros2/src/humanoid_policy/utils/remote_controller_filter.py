# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
# This file was originally copied from the [unitree_rl_gym] repository:
# https://github.com/unitreerobotics/unitree_rl_gym/tree/main
# Modifications have been made to fit the needs of this project.

import struct


class KeyMap:
    """Mapping from button names to indices for Unitree wireless remote.

    Reference:
        https://github.com/unitreerobotics/unitree_rl_gym/blob/main/unitree_rl_gym/envs/common/remote_controller.py
    """

    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    """Process and filter wireless remote data for robot teleoperation.

    Provides deadzone and smoothing filters for joystick axes, and parses
    button states.

    Reference:
        https://github.com/unitreerobotics/unitree_rl_gym/blob/main/unitree_rl_gym/envs/common/remote_controller.py
    """

    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16
        # Filter parameters
        self.alpha = 0.3  # Smoothing coefficient (0-1, smaller is smoother)
        self.deadzone = 0.05  # Deadzone threshold
        # Previous state for filtering
        self.lx_prev = 0
        self.ly_prev = 0
        self.rx_prev = 0
        self.ry_prev = 0
        self.smooth = 0.03  # Smoothing coefficient
        self.dead_zone = 0.01  # Deadzone threshold
        # Velocity mapping parameters
        self.max_linear_speed = 1.0  # Maximum linear speed (m/s)
        self.max_angular_speed = 1.0  # Maximum angular speed (rad/s)

    def apply_filter_and_deadzone(self, value, prev_value):
        """Apply deadzone and smoothing filter to joystick axis value.

        Args:
            value (float): Current raw value.
            prev_value (float): Previous filtered value.

        Returns:
            float: Filtered value.
        """
        if abs(value) < self.dead_zone:
            value = 0.0
        return prev_value * (1 - self.smooth) + value * self.smooth

    def set(self, data):
        """Parse wireless remote data and update button and joystick states.

        Args:
            data (bytes or bytearray): Raw wireless remote data.
        Reference:
            https://github.com/unitreerobotics/unitree_rl_gym/blob/main/
            unitree_rl_gym/envs/common/remote_controller.py
        """
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        # Read raw joystick values
        lx_raw = struct.unpack("f", data[4:8])[0]
        rx_raw = struct.unpack("f", data[8:12])[0]
        ry_raw = struct.unpack("f", data[12:16])[0]
        ly_raw = struct.unpack("f", data[20:24])[0]
        # Apply filter and deadzone
        self.lx = self.apply_filter_and_deadzone(lx_raw, self.lx_prev)
        self.ly = self.apply_filter_and_deadzone(ly_raw, self.ly_prev)
        self.rx = self.apply_filter_and_deadzone(rx_raw, self.rx_prev)
        self.ry = self.apply_filter_and_deadzone(ry_raw, self.ry_prev)
        # Update previous values
        self.lx_prev = self.lx
        self.ly_prev = self.ly
        self.rx_prev = self.rx
        self.ry_prev = self.ry

    def get_velocity_commands(self):
        """Convert joystick values to velocity commands for teleoperation.

        Returns:
            tuple: (vx, vy, vyaw)
                - vx: Forward/backward speed (m/s), left stick Y (ly)
                - vy: Lateral speed (m/s), left stick X (lx)
                - vyaw: Yaw rate (rad/s), right stick X (rx)
        """
        vx = self.ly * self.max_linear_speed  # Forward/backward speed
        vy = 0.5 * self.lx * self.max_linear_speed  # Lateral speed
        vyaw = -self.rx * self.max_angular_speed  # Yaw rate
        return vx, vy, vyaw
