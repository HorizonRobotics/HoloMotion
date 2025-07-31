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

/**
 * This example demonstrates how to use ROS2 to send low-level motor commands of
 * unitree g1 robot
 **/
#include "common/motor_crc_hg.h"
#include "common/wireless_controller.h"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/wireless_controller.hpp"
#include "unitree_hg/msg/low_cmd.hpp"
#include "unitree_hg/msg/low_state.hpp"
#include "unitree_hg/msg/motor_cmd.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <map>
#include <sstream>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <thread>

#define INFO_IMU 0   // Set 1 to info IMU states
#define INFO_MOTOR 0 // Set 1 to info motor states

enum PRorAB { PR = 0, AB = 1 };

using std::placeholders::_1;

const int G1_NUM_MOTOR = 29;

enum class RobotState { ZERO_TORQUE, MOVE_TO_DEFAULT, EMERGENCY_STOP, POLICY };
enum class EmergencyStopPhase { DAMPING, DISABLE };  // New enum for emergency stop phases

// Create a humanoid_controller class for low state receive
class humanoid_controller : public rclcpp::Node {
public:
  humanoid_controller() : Node("humanoid_controller") {
    // Get config path from ROS parameter
    std::string config_path =
        this->declare_parameter<std::string>("config_path", "");

    RCLCPP_INFO(this->get_logger(), "Config file path: %s",
                config_path.c_str());

    // Load configuration
    loadConfig(config_path);
    RCLCPP_INFO(this->get_logger(),
                "Entered ZERO_TORQUE state, press start to switch to "
                "MOVE_TO_DEFAULT state, press A to switch to POLICY state, "
                "press select to emergency stop. Waiting for start signal...");

    lowstate_subscriber_ = this->create_subscription<unitree_hg::msg::LowState>(
        "/lowstate", 10,
        std::bind(&humanoid_controller::LowStateHandler, this, _1));

    policy_action_subscriber_ =
        this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/humanoid/action", 10,
            std::bind(&humanoid_controller::PolicyActionHandler, this, _1));

    lowcmd_publisher_ =
        this->create_publisher<unitree_hg::msg::LowCmd>("/lowcmd", 10);

    robot_state_publisher_ = 
        this->create_publisher<std_msgs::msg::String>("/robot_state", 10);

    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(timer_dt),
                                std::bind(&humanoid_controller::Control, this));

    time_ = 0;
    duration_ = 3; // 3 s
  }

private:
  std::map<std::string, int> dof2motor_idx;
  std::map<std::string, double> default_dof_pos;
  std::map<std::string, double> kps;
  std::map<std::string, double> kds;
  std::vector<std::string> complete_dof_order;
  std::vector<std::string> policy_dof_order;
  RemoteController remote_controller;
  std::map<std::string, double> target_dof_pos;
  std::vector<float> policy_action_data;

  RobotState current_state_ = RobotState::ZERO_TORQUE;

  bool should_shutdown_ = false;

  // Add safety limit parameters using existing structure in YAML
  std::map<std::string, std::pair<double, double>> joint_position_limits; // min, max
  std::map<std::string, double> joint_velocity_limits;
  std::map<std::string, double> joint_effort_limits;
  
  // Scaling coefficients for limits
  double position_limit_scale = 1.0;
  double velocity_limit_scale = 1.0;
  double effort_limit_scale = 1.0;

  EmergencyStopPhase emergency_stop_phase_ = EmergencyStopPhase::DAMPING;
  double emergency_stop_time_ = 0.0;
  double emergency_damping_duration_ = 2.0;  // 1 second of damping before disabling

  // Add a helper function to calculate expected torque
  double calculateExpectedTorque(const std::string& dof_name, double q_des, double q, double dq) {
    double kp = kps[dof_name];
    double kd = kds[dof_name];
    // dq_des is assumed to be 0 in your control scheme
    return kp * (q_des - q) + kd * (0.0 - dq);
  }
  
  // Add a helper function to scale kp and kd to limit torque
  std::pair<double, double> limitTorque(const std::string& dof_name, double q_des, double q, double dq) {
    double kp = kps[dof_name];
    double kd = kds[dof_name];
    
    // Calculate expected torque
    double expected_torque = calculateExpectedTorque(dof_name, q_des, q, dq);
    double abs_expected_torque = std::abs(expected_torque);
    
    // Check if torque would exceed limit
    if (joint_effort_limits.find(dof_name) != joint_effort_limits.end()) {
      double max_torque = joint_effort_limits[dof_name] * effort_limit_scale;
      
      if (abs_expected_torque > max_torque && abs_expected_torque > 1e-6) {
        // Scale both kp and kd by the same factor to preserve damping characteristics
        double scale_factor = max_torque / abs_expected_torque;
        return std::make_pair(kp * scale_factor, kd * scale_factor);
      }
    }
    
    // If no scaling needed, return original values
    return std::make_pair(kp, kd);
  }

  void loadConfig(const std::string &config_path) {
    try {
      YAML::Node config = YAML::LoadFile(config_path);

      // Load motor indices
      auto indices = config["dof2motor_idx_mapping"];
      for (const auto &it : indices) {
        dof2motor_idx[it.first.as<std::string>()] = it.second.as<int>();
      }

      // Load default angles
      auto angles = config["default_joint_angles_start"];
      for (const auto &it : angles) {
        default_dof_pos[it.first.as<std::string>()] = it.second.as<double>();
      }
      // Set target dof pos to default dof pos
      for (const auto &it : default_dof_pos) {
        target_dof_pos[it.first] = it.second;
      }

      // Load stiffness values
      auto stiff = config["control_params"]["stiffness"];
      for (const auto &it : stiff) {
        kps[it.first.as<std::string>()] = it.second.as<double>();
      }

      // Load damping values
      auto damp = config["control_params"]["damping"];
      for (const auto &it : damp) {
        kds[it.first.as<std::string>()] = it.second.as<double>();
      }

      // Load dof order
      for (const auto &it : config["complete_dof_order"]) {
        complete_dof_order.push_back(it.as<std::string>());
      }
      for (const auto &it : config["policy_dof_order"]) {
        policy_dof_order.push_back(it.as<std::string>());
      }

      // Load control frequency
      control_freq_ = config["control_freq"].as<double>();
      control_dt_ = 1.0 / control_freq_;
      timer_dt = static_cast<int>(control_dt_ * 1000);
      RCLCPP_INFO(this->get_logger(), "Control frequency set to: %f Hz",
                  control_freq_);

      // Initialize policy_action vector with the correct size
      policy_action_scale = config["action_scale"].as<double>();

      // Load joint limits
      auto pos_limits = config["joint_limits"]["position"];
      for (const auto &it : pos_limits) {
        std::string dof_name = it.first.as<std::string>();
        auto limits = it.second.as<std::vector<double>>();
        joint_position_limits[dof_name] = std::make_pair(limits[0], limits[1]);
      }

      auto vel_limits = config["joint_limits"]["velocity"];
      for (const auto &it : vel_limits) {
        joint_velocity_limits[it.first.as<std::string>()] = it.second.as<double>();
      }

      auto effort_limits = config["joint_limits"]["effort"];
      for (const auto &it : effort_limits) {
        joint_effort_limits[it.first.as<std::string>()] = it.second.as<double>();
      }

      // Load joint limits scaling coefficients (optional, default to 1.0)
      position_limit_scale = config["limit_scales"]["position"].as<double>(1.0);
      velocity_limit_scale = config["limit_scales"]["velocity"].as<double>(1.0);
      effort_limit_scale = config["limit_scales"]["effort"].as<double>(1.0);
      
      RCLCPP_INFO(this->get_logger(), "Joint limit scales - Position: %f, Velocity: %f, Effort: %f",
                 position_limit_scale, velocity_limit_scale, effort_limit_scale);
    } catch (const YAML::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Error parsing config file: %s",
                   e.what());
    }
  }

  void Control() {
    // First check if we're already in emergency stop
    if (current_state_ == RobotState::EMERGENCY_STOP) {
        emergency_stop_time_ += control_dt_;
        
        if (emergency_stop_phase_ == EmergencyStopPhase::DAMPING) {
            SendDampedEmergencyStop();
            if (emergency_stop_time_ >= emergency_damping_duration_) {
                emergency_stop_phase_ = EmergencyStopPhase::DISABLE;
                RCLCPP_INFO(this->get_logger(), "Damping complete, disabling motors");
            }
        } else {
            SendFinalEmergencyStop();
            if (timer_) {
                timer_->cancel();
            }
            rclcpp::shutdown();
            return;
        }
        
        get_crc(low_command);
        lowcmd_publisher_->publish(low_command);
        return;  // Exit early, ignore all other commands
    }

    // If not in emergency stop, check for emergency stop command first
    if (remote_controller.button[KeyMap::select] == 1) {
        current_state_ = RobotState::EMERGENCY_STOP;
        should_shutdown_ = true;
        publishRobotState();
        return;
    }

    // Process other commands only if not in emergency stop
    if (remote_controller.button[KeyMap::L1] == 1 &&
        current_state_ != RobotState::ZERO_TORQUE) {
        RCLCPP_INFO(this->get_logger(), "Switching to ZERO_TORQUE state");
        current_state_ = RobotState::ZERO_TORQUE;
        publishRobotState();
    }

    // Start button only works in ZERO_TORQUE state
    if (remote_controller.button[KeyMap::start] == 1) {
        if (current_state_ == RobotState::ZERO_TORQUE) {
            RCLCPP_INFO(this->get_logger(), "Switching to MOVE_TO_DEFAULT state");
            current_state_ = RobotState::MOVE_TO_DEFAULT;
            time_ = 0.0;
            publishRobotState();
        } else {
            RCLCPP_INFO(this->get_logger(), 
                "Start button only works in ZERO_TORQUE state. Current state: %d", 
                static_cast<int>(current_state_));
        }
    }

    // A button only works in MOVE_TO_DEFAULT state
    if (remote_controller.button[KeyMap::A] == 1) {
        if (current_state_ == RobotState::MOVE_TO_DEFAULT) {
            // Check lower body joint positions before allowing transition
            bool positions_ok = true;
            std::stringstream deviation_msg;
            const double position_threshold = 0.4;

            // List of lower body joints to check
            std::vector<std::string> lower_body_joints = {
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll"
            };

            for (int i = 0; i < G1_NUM_MOTOR; ++i) {
                std::string dof_name = complete_dof_order[i];
                
                // Skip if not a lower body joint
                if (std::find(lower_body_joints.begin(), lower_body_joints.end(), dof_name) == lower_body_joints.end()) {
                    continue;
                }

                double current_pos = motor[i].q;
                double default_pos = default_dof_pos[dof_name];
                double diff = std::abs(current_pos - default_pos);

                if (diff > position_threshold) {
                    positions_ok = false;
                    deviation_msg << dof_name << "(" << diff << "), ";
                }
            }

            if (positions_ok) {
                RCLCPP_INFO(this->get_logger(), "Switching to POLICY state");
                current_state_ = RobotState::POLICY;
                time_ = 0.0;
                publishRobotState();
                
            } else {
                RCLCPP_WARN(this->get_logger(), 
                    "Cannot switch to POLICY state. Lower body joints with large deviations: %s", 
                    deviation_msg.str().c_str());
                
            }
        } else {
            RCLCPP_INFO(this->get_logger(), 
                "A button only works in MOVE_TO_DEFAULT state. Current state: %d", 
                static_cast<int>(current_state_));
        }
    }

    // Normal state machine logic
    switch (current_state_) {
        case RobotState::ZERO_TORQUE:
            SendZeroTorqueCommand();
            get_crc(low_command);
            lowcmd_publisher_->publish(low_command);
            break;

        case RobotState::MOVE_TO_DEFAULT:
            SendDefaultPositionCommand();
            get_crc(low_command);
            lowcmd_publisher_->publish(low_command);
            break;

        case RobotState::POLICY:
            SendPolicyCommand();
            get_crc(low_command);
            lowcmd_publisher_->publish(low_command);
            break;

        case RobotState::EMERGENCY_STOP:
            // Emergency stop is handled at the beginning of the function
            // This case should not be reached due to early return
            break;
    }
    
    // Publish current robot state
    publishRobotState();
  }

  void SendZeroTorqueCommand() {
    low_command.mode_pr = mode_;
    low_command.mode_machine = mode_machine;

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      low_command.motor_cmd[i].mode = 1; // Enable
      low_command.motor_cmd[i].q = 0.0;
      low_command.motor_cmd[i].dq = 0.0;
      low_command.motor_cmd[i].kp = 0.0;
      low_command.motor_cmd[i].kd = 0.0;
      low_command.motor_cmd[i].tau = 0.0;
    }
  }

  void SendDefaultPositionCommand() {
    time_ += control_dt_;
    low_command.mode_pr = mode_;
    low_command.mode_machine = mode_machine;

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      std::string dof_name = complete_dof_order[i];
      double ratio = clamp(time_ / duration_, 0.0, 1.0);
      double target_pos = (1. - ratio) * motor[i].q + ratio * default_dof_pos[dof_name];
      
      // Get current joint state
      double current_pos = motor[i].q;
      double current_vel = motor[i].dq;
      
      // Calculate torque-limited kp and kd
      auto [limited_kp, limited_kd] = limitTorque(dof_name, target_pos, current_pos, current_vel);
      
      low_command.motor_cmd[i].mode = 1;
      low_command.motor_cmd[i].tau = 0.0;
      low_command.motor_cmd[i].q = target_pos;
      low_command.motor_cmd[i].dq = 0.0;
      low_command.motor_cmd[i].kp = limited_kp;
      low_command.motor_cmd[i].kd = limited_kd;
    }
  }

  void SendPolicyCommand() {
    time_ += control_dt_;
    low_command.mode_pr = mode_;
    low_command.mode_machine = mode_machine;

    for (const auto &pair : target_dof_pos) {
      const std::string &dof_name = pair.first;
      const double &target_pos = pair.second;
      int motor_idx = dof2motor_idx[dof_name];
      
      // Get current joint state
      double current_pos = motor[motor_idx].q;
      double current_vel = motor[motor_idx].dq;
      
      // Calculate torque-limited kp and kd
      auto [limited_kp, limited_kd] = limitTorque(dof_name, target_pos, current_pos, current_vel);
      
      low_command.motor_cmd[motor_idx].mode = 1;
      low_command.motor_cmd[motor_idx].tau = 0.0;
      low_command.motor_cmd[motor_idx].q = target_pos;
      low_command.motor_cmd[motor_idx].dq = 0.0;
      low_command.motor_cmd[motor_idx].kp = limited_kp;
      low_command.motor_cmd[motor_idx].kd = limited_kd;
    }
  }

  void SendDampedEmergencyStop() {
    low_command.mode_pr = mode_;
    low_command.mode_machine = mode_machine;

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      std::string dof_name = complete_dof_order[i];
      low_command.motor_cmd[i].mode = 1; // Keep enabled
      low_command.motor_cmd[i].q = motor[i].q; // Current position
      low_command.motor_cmd[i].dq = 0.0; // Target zero velocity
      low_command.motor_cmd[i].kp = 0.0; // No position control
      low_command.motor_cmd[i].kd = kds[dof_name] * 2.0; // Higher damping for faster stopping
      low_command.motor_cmd[i].tau = 0.0;
    }
  }

  void SendFinalEmergencyStop() {
    low_command.mode_pr = mode_;
    low_command.mode_machine = mode_machine;

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      low_command.motor_cmd[i].mode = 0; // Disable
      low_command.motor_cmd[i].q = 0.0;
      low_command.motor_cmd[i].dq = 0.0;
      low_command.motor_cmd[i].kp = 0.0;
      low_command.motor_cmd[i].kd = 0.0;
      low_command.motor_cmd[i].tau = 0.0;
    }
  }

  void LowStateHandler(unitree_hg::msg::LowState::SharedPtr message) {
    mode_machine = (int)message->mode_machine;
    imu = message->imu_state;
    for (int i = 0; i < G1_NUM_MOTOR; i++) {
      motor[i] = message->motor_state[i];
    }

    // Check joint limits for all joints
    bool limits_exceeded = false;
    std::string exceeded_msg;

    for (const auto &pair : dof2motor_idx) {
      const std::string &dof_name = pair.first;
      int motor_idx = pair.second;
      
      // Check position limits with scaling
      if (joint_position_limits.find(dof_name) != joint_position_limits.end()) {
        double pos = motor[motor_idx].q;
        // Calculate the middle point of the range
        double mid_pos = (joint_position_limits[dof_name].first + joint_position_limits[dof_name].second) / 2.0;
        // Calculate the half-range and scale it
        double half_range = (joint_position_limits[dof_name].second - joint_position_limits[dof_name].first) / 2.0;
        double scaled_half_range = half_range * position_limit_scale;
        
        // Calculate scaled min and max by expanding from midpoint
        double min_pos = mid_pos - scaled_half_range;
        double max_pos = mid_pos + scaled_half_range;
        
        if (pos < min_pos || pos > max_pos) {
          limits_exceeded = true;
          exceeded_msg = "Position limit exceeded for joint " + dof_name + 
                        ": " + std::to_string(pos) + " (scaled limits: " + 
                        std::to_string(min_pos) + ", " + std::to_string(max_pos) + ")";
          break;
        }
      }
      
      // Check velocity limits with scaling
      if (joint_velocity_limits.find(dof_name) != joint_velocity_limits.end()) {
        double vel = std::abs(motor[motor_idx].dq);
        double max_vel = joint_velocity_limits[dof_name] * velocity_limit_scale;
        
        if (vel > max_vel) {
          limits_exceeded = true;
          exceeded_msg = "Velocity limit exceeded for joint " + dof_name + 
                        ": " + std::to_string(vel) + " (scaled limit: " + 
                        std::to_string(max_vel) + ")";
          break;
        }
      }
      
      // Check effort/torque limits with scaling
      if (joint_effort_limits.find(dof_name) != joint_effort_limits.end()) {
        double torque = std::abs(motor[motor_idx].tau_est);
        double max_torque = joint_effort_limits[dof_name] * effort_limit_scale;
        
        if (torque > max_torque) {
          limits_exceeded = true;
          exceeded_msg = "Torque limit exceeded for joint " + dof_name + 
                        ": " + std::to_string(torque) + " (scaled limit: " + 
                        std::to_string(max_torque) + ")";
          break;
        }
      }
    }
    
    // Trigger emergency stop if any limits are exceeded
    if (limits_exceeded) {
      RCLCPP_ERROR(this->get_logger(), "%s", exceeded_msg.c_str());
      RCLCPP_ERROR(this->get_logger(), "Joint limits exceeded! Triggering emergency stop.");
      current_state_ = RobotState::EMERGENCY_STOP;
      should_shutdown_ = true;
      publishRobotState();
    }

    remote_controller.set(message->wireless_remote);
    //   RCLCPP_INFO(this->get_logger(),
    //               "Wireless controller -- lx: %f; ly: %f; rx: %f; ry: %f;
    //               start " "pressed: %d select pressed: %d A
    //               pressed: %d B " "pressed: %d X " "pressed: %d Y pressed: %d",
    //               remote_controller.lx, remote_controller.ly,
    //               remote_controller.rx, remote_controller.ry,
    //               remote_controller.button[KeyMap::start],
    //               remote_controller.button[KeyMap::select],
    //               remote_controller.button[KeyMap::A],
    //               remote_controller.button[KeyMap::B],
    //               remote_controller.button[KeyMap::X],
    //               remote_controller.button[KeyMap::Y]);
    // }
  }

  void PolicyActionHandler(
      const std_msgs::msg::Float32MultiArray::SharedPtr message) {
    // RCLCPP_INFO(this->get_logger(), "PolicyActionHandler called!");
    policy_action_data = message->data;

    // Check if message size matches expected size
    if (policy_action_data.size() != policy_dof_order.size()) {
      RCLCPP_ERROR(this->get_logger(), 
                  "Policy action data size mismatch: got %zu, expected %zu", 
                  policy_action_data.size(), policy_dof_order.size());
      current_state_ = RobotState::EMERGENCY_STOP;
      should_shutdown_ = true;
      publishRobotState();
      return;
    }

    // set target dof pos
    for (size_t i = 0; i < policy_dof_order.size(); i++) {
      const auto &dof_name = policy_dof_order[i];
      // double calculated_pos = policy_action_data[i] * policy_action_scale +
      //                        default_dof_pos[dof_name];

      double calculated_pos = policy_action_data[i];
      
      // Check if the target position is within joint limits (with scaling)
      if (joint_position_limits.find(dof_name) != joint_position_limits.end()) {
        // Calculate the middle point of the range
        double mid_pos = (joint_position_limits[dof_name].first + joint_position_limits[dof_name].second) / 2.0;
        // Calculate the half-range and scale it
        double half_range = (joint_position_limits[dof_name].second - joint_position_limits[dof_name].first) / 2.0;
        double scaled_half_range = half_range * position_limit_scale;
        
        // Calculate scaled min and max by expanding from midpoint
        double min_pos = mid_pos - scaled_half_range;
        double max_pos = mid_pos + scaled_half_range;
        
        if (calculated_pos < min_pos || calculated_pos > max_pos) {
          RCLCPP_WARN(this->get_logger(), 
                     "Target position would exceed limit for joint %s: %f (scaled limits: %f, %f)", 
                     dof_name.c_str(), calculated_pos, min_pos, max_pos);
          // Clamp the position to within limits
          calculated_pos = std::clamp(calculated_pos, min_pos, max_pos);
        }
      }
      
      // Set the target position (clamped to safe values if needed)
      target_dof_pos[dof_name] = calculated_pos;
    }

    // Trigger emergency stop if any limits would be exceeded
    // if (limits_exceeded) {
    //   RCLCPP_ERROR(this->get_logger(), "%s", exceeded_msg.c_str());
    //   RCLCPP_ERROR(this->get_logger(), "Policy would exceed joint limits! Triggering emergency stop.");
    //   current_state_ = RobotState::EMERGENCY_STOP;
    //   should_shutdown_ = true;
    // }

    // log target dof pos
    // for (const auto& pair : target_dof_pos) {
    //     const std::string& dof_name = pair.first;
    //     const double& target_pos = pair.second;
    //     RCLCPP_INFO(this->get_logger(), "Target dof pos -- %s: %f",
    //                 dof_name.c_str(), target_pos);
    // }
  }

  double clamp(double value, double low, double high) {
    if (value < low)
      return low;
    if (value > high)
      return high;
    return value;
  }

  std::string robotStateToString(RobotState state) {
    switch (state) {
      case RobotState::ZERO_TORQUE:
        return "ZERO_TORQUE";
      case RobotState::MOVE_TO_DEFAULT:
        return "MOVE_TO_DEFAULT";
      case RobotState::EMERGENCY_STOP:
        return "EMERGENCY_STOP";
      case RobotState::POLICY:
        return "POLICY";
      default:
        return "UNKNOWN";
    }
  }

  void publishRobotState() {
    std_msgs::msg::String state_msg;
    state_msg.data = robotStateToString(current_state_);
    robot_state_publisher_->publish(state_msg);
  }

  rclcpp::TimerBase::SharedPtr timer_; // ROS2 timer
  rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr
      lowcmd_publisher_; // ROS2 Publisher
  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr
      lowstate_subscriber_; // ROS2 Subscriber
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
      policy_action_subscriber_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_state_publisher_;
  unitree_hg::msg::LowCmd low_command; // Unitree hg lowcmd message
  unitree_hg::msg::IMUState imu;       // Unitree hg IMU message
  unitree_hg::msg::MotorState
      motor[G1_NUM_MOTOR]; // Unitree hg motor state message
  double control_freq_;
  double policy_action_scale;
  double control_dt_;
  int timer_dt;
  double time_; // Running time count
  double duration_;
  PRorAB mode_ = PRorAB::PR;
  int mode_machine;
  RemoteController wireless_remote_;
}; // End of humanoid_controller class

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);                            // Initialize rclcpp
  auto node = std::make_shared<humanoid_controller>(); // Create a ROS2 node
  rclcpp::spin(node);                                  // Run ROS2 node
  rclcpp::shutdown();                                  // Exit
  return 0;
}