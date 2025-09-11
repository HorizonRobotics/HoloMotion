#pragma once

#include "unitree_go/msg/wireless_controller.hpp"
#include <array>
#include <cstring>

class KeyMap {
public:
    static const int R1;
    static const int L1;
    static const int start;
    static const int select;
    static const int R2;
    static const int L2;
    static const int F1;
    static const int F2;
    static const int A;
    static const int B;
    static const int X;
    static const int Y;
    static const int up;
    static const int right;
    static const int down;
    static const int left;
};

class RemoteController {
public:
    // Constructor
    RemoteController();

    // Add overloaded set method for raw data
    void set(const std::array<unsigned char, 40>& data);
    
    // Keep original method for compatibility
    void set(const unitree_go::msg::WirelessController::SharedPtr msg);

    // Member variables
    double lx;
    double ly;
    double rx;
    double ry;
    int button[16];
};
