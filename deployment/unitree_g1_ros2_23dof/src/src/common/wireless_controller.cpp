#include "common/wireless_controller.h"
#include <cstring>

// Define static constants
const int KeyMap::R1 = 0;
const int KeyMap::L1 = 1;
const int KeyMap::start = 2;
const int KeyMap::select = 3;
const int KeyMap::R2 = 4;
const int KeyMap::L2 = 5;
const int KeyMap::F1 = 6;
const int KeyMap::F2 = 7;
const int KeyMap::A = 8;
const int KeyMap::B = 9;
const int KeyMap::X = 10;
const int KeyMap::Y = 11;
const int KeyMap::up = 12;
const int KeyMap::right = 13;
const int KeyMap::down = 14;
const int KeyMap::left = 15;

// Implement RemoteController methods
RemoteController::RemoteController() {
  lx = 0;
  ly = 0;
  rx = 0;
  ry = 0;
  std::fill(button, button + 16, 0);
}

void RemoteController::set(const std::array<unsigned char, 40> &data) {
  // Debug print raw bytes
  //   printf("Raw data bytes: ");
  //   for (int i = 0; i < 40; i++) {
  //     printf("%02x ", data[i]);
  //   }
  //   printf("\n");

  // Extract keys from bytes 2-3
  uint16_t keys = (data[3] << 8) | data[2];
  //   printf("Keys value: 0x%04x\n", keys);

  for (int i = 0; i < 16; i++) {
    button[i] = (keys & (1 << i)) >> i;
  }

  // Extract and print floats before memcpy
  float lx_temp, rx_temp, ry_temp, ly_temp;
  std::memcpy(&lx_temp, &data[4], 4);  // bytes 4-7
  std::memcpy(&rx_temp, &data[8], 4);  // bytes 8-11
  std::memcpy(&ry_temp, &data[12], 4); // bytes 12-15
  std::memcpy(&ly_temp, &data[20], 4); // bytes 20-23

  //   printf("Values before assignment: lx=%f, ly=%f, rx=%f, ry=%f\n", lx_temp,
  //  ly_temp, rx_temp, ry_temp);

  // Assign to class members
  lx = lx_temp;
  rx = rx_temp;
  ry = ry_temp;
  ly = ly_temp;

  //   printf("Values after assignment: lx=%f, ly=%f, rx=%f, ry=%f\n", lx, ly,
  //   rx,
  //  ry);
}

void RemoteController::set(
    const unitree_go::msg::WirelessController::SharedPtr msg) {
  uint16_t keys = msg->keys;
  for (int i = 0; i < 16; i++) {
    button[i] = (keys & (1 << i)) >> i;
  }
  lx = msg->lx;
  rx = msg->rx;
  ry = msg->ry;
  ly = msg->ly;
}