// Project HoloMotion
//
// Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
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
