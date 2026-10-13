#include "motor_crc_hg.h"
#include <cstring>

void get_crc(unitree_hg::msg::LowCmd &msg)
{
    // Encode the official HG wire layout into initialized integer words.
    // Do not CRC C++ object padding or read float objects through uint32_t*.
    std::array<uint32_t, 250> words{};
    words[0] = static_cast<uint32_t>(msg.mode_pr)
             | (static_cast<uint32_t>(msg.mode_machine) << 8);
    for (size_t i = 0; i < 35; ++i) {
        const auto &motor = msg.motor_cmd[i];
        const size_t base = 1 + i * 7;
        words[base] = motor.mode;
        const float values[] = {motor.q, motor.dq, motor.tau, motor.kp, motor.kd};
        for (size_t j = 0; j < 5; ++j) {
            std::memcpy(&words[base + 1 + j], &values[j], sizeof(float));
        }
        words[base + 6] = motor.reserve;
    }
    for (size_t i = 0; i < 4; ++i) words[246 + i] = msg.reserve[i];
    msg.crc = crc32_core(words.data(), static_cast<uint32_t>(words.size()));
}

uint32_t crc32_core(uint32_t *ptr, uint32_t len)
{
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
                CRC32 <<= 1;
            if (data & xbit)
                CRC32 ^= dwPolynomial;

            xbit >>= 1;
        }
    }
    return CRC32;
}