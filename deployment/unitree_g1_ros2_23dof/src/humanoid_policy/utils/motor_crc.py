import struct
import numpy as np
from ctypes import Structure, c_uint8, c_float, c_uint32, Array


def crc32_core(data_array, length):
    CRC32 = 0xFFFFFFFF
    dwPolynomial = 0x04C11DB7

    for i in range(length):
        data = data_array[i]
        for bit in range(32):  # Process all 32 bits
            if (CRC32 >> 31) & 1:  # Check MSB before shift
                CRC32 = ((CRC32 << 1) & 0xFFFFFFFF) ^ dwPolynomial
            else:
                CRC32 = (CRC32 << 1) & 0xFFFFFFFF
                
            if (data >> (31 - bit)) & 1:  # Match C++ bit processing order
                CRC32 ^= dwPolynomial

    return CRC32


def calc_crc(cmd) -> int:
    """Calculate CRC for LowCmd message"""
    buffer = bytearray()
    
    # Pack header (mode_pr, mode_machine + 2 padding)
    buffer.extend(struct.pack('<BBxx', cmd.mode_pr, cmd.mode_machine))
    
    # Pack motor commands
    for motor in cmd.motor_cmd:
        buffer.extend(struct.pack('<B3xfffffI',
            motor.mode,
            motor.q,
            motor.dq,
            motor.tau,
            motor.kp,
            motor.kd,
            motor.reserve
        ))
    
    # Pack reserve (4 bytes)
    buffer.extend(struct.pack('<4B', *cmd.reserve))
    
    # Convert to uint32 array (little-endian)
    uint32_array = struct.unpack(f'<{len(buffer)//4}I', buffer)
    
    # Calculate with fixed length (246 for LowCmd struct size)
    return crc32_core(uint32_array, 246)