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
import struct


def crc32_core(data_array, length):
    crc32 = 0xFFFFFFFF
    dw_polynomial = 0x04C11DB7

    for i in range(length):
        data = data_array[i]
        for bit in range(32):  # Process all 32 bits
            if (crc32 >> 31) & 1:  # Check MSB before shift
                crc32 = ((crc32 << 1) & 0xFFFFFFFF) ^ dw_polynomial
            else:
                crc32 = (crc32 << 1) & 0xFFFFFFFF

            if (data >> (31 - bit)) & 1:  # Match C++ bit processing order
                crc32 ^= dw_polynomial

    return crc32


def calc_crc(cmd) -> int:
    """Calculate CRC for LowCmd message."""
    buffer = bytearray()

    # Pack header (mode_pr, mode_machine + 2 padding)
    buffer.extend(struct.pack("<BBxx", cmd.mode_pr, cmd.mode_machine))

    # Pack motor commands
    for motor in cmd.motor_cmd:
        buffer.extend(
            struct.pack(
                "<B3xfffffI",
                motor.mode,
                motor.q,
                motor.dq,
                motor.tau,
                motor.kp,
                motor.kd,
                motor.reserve,
            )
        )

    # Pack reserve (4 bytes)
    buffer.extend(struct.pack("<4B", *cmd.reserve))

    # Convert to uint32 array (little-endian)
    uint32_array = struct.unpack(f"<{len(buffer) // 4}I", buffer)

    # Calculate with fixed length (246 for LowCmd struct size)
    return crc32_core(uint32_array, 246)
