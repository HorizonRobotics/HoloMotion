/**
 * This example demonstrates how to use ROS2 to send low-level motor commands of
 *unitree g1 robot 29 dof
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
     RCLCPP_INFO(this->get_logger(), "Using main_node !!!");
 
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
 
     // Add subscribers for kps and kds parameters from policy node
     kps_subscriber_ =
         this->create_subscription<std_msgs::msg::Float32MultiArray>(
             "/humanoid/kps", 10,
             std::bind(&humanoid_controller::KpsHandler, this, _1));
     
     kds_subscriber_ =
         this->create_subscription<std_msgs::msg::Float32MultiArray>(
             "/humanoid/kds", 10,
             std::bind(&humanoid_controller::KdsHandler, this, _1));
 
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
   
   // Policy-subscribed control parameters
   std::vector<float> policy_kps_data;
   std::vector<float> policy_kds_data;
   bool kps_received_ = false;
   bool kds_received_ = false;
   RemoteController remote_controller;
   std::map<std::string, double> target_dof_pos;
   std::vector<float> policy_action_data;
 
   // Optional arrays from YAML for start (MOVE_TO_DEFAULT) behavior
   bool has_joint_arrays_ = false;
   std::vector<std::string> joint_names_array_;
   std::vector<double> default_position_array_;
   std::vector<double> kp_array_;
   std::vector<double> kd_array_;
   
   std::map<std::string, double> move_to_default_kps;
   std::map<std::string, double> move_to_default_kds;
 
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
   
   // Add a helper function to scale custom kp and kd to limit torque
   std::pair<double, double> limitTorqueWithCustomGains(
     const std::string& dof_name, 
     double q_des, 
     double q, 
     double dq,
     double custom_kp,
     double custom_kd) {
     
     // Calculate expected torque
     double expected_torque = custom_kp * (q_des - q) + custom_kd * (0.0 - dq);
     double abs_expected_torque = std::abs(expected_torque);
     
     // Check if torque would exceed limit
     if (joint_effort_limits.find(dof_name) != joint_effort_limits.end()) {
       double max_torque = joint_effort_limits[dof_name] * effort_limit_scale;
       
       if (abs_expected_torque > max_torque && abs_expected_torque > 1e-6) {
         // Scale both kp and kd by the same factor to preserve damping characteristics
         double scale_factor = max_torque / abs_expected_torque;
         return std::make_pair(custom_kp * scale_factor, custom_kd * scale_factor);
       }
     }
     
     // If no scaling needed, return original values
     return std::make_pair(custom_kp, custom_kd);
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
       auto angles = config["default_joint_angles"];
       for (const auto &it : angles) {
         default_dof_pos[it.first.as<std::string>()] = it.second.as<double>();
       }
       // Set target dof pos to default dof pos
       for (const auto &it : default_dof_pos) {
         target_dof_pos[it.first] = it.second;
       }
 
       // Note: kps and kds are now received from policy node via ROS topics
       // No longer loading from config file to avoid conflicts
 
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
 
       // Optional: arrays for joint configuration on Start
       // If kp and kd arrays are provided, use them with joint names and positions
       // Auto-generate joint_names and default_position from complete_dof_order and default_joint_angles if not provided
       if (config["kp"] && config["kd"]) {
         joint_names_array_.clear();
         default_position_array_.clear();
         kp_array_.clear();
         kd_array_.clear();
 
         // Auto-generate joint_names and default_position from existing config if not explicitly provided
         if (config["joint_names"] && config["default_position"]) {
           // Use explicitly provided arrays
           for (const auto &it : config["joint_names"]) {
             joint_names_array_.push_back(it.as<std::string>());
           }
           for (const auto &it : config["default_position"]) {
             default_position_array_.push_back(it.as<double>());
           }
         } else {
           // Auto-generate from complete_dof_order and default_joint_angles
           for (const auto &dof_name : complete_dof_order) {
             joint_names_array_.push_back(dof_name);
             if (default_dof_pos.find(dof_name) != default_dof_pos.end()) {
               default_position_array_.push_back(default_dof_pos[dof_name]);
             } else {
               RCLCPP_WARN(this->get_logger(), "Default position not found for joint %s, using 0.0", dof_name.c_str());
               default_position_array_.push_back(0.0);
             }
           }
           RCLCPP_INFO(this->get_logger(), "Auto-generated joint_names and default_position from complete_dof_order and default_joint_angles");
         }
 
         // Load kp and kd arrays
         for (const auto &it : config["kp"]) {
           kp_array_.push_back(it.as<double>());
         }
         for (const auto &it : config["kd"]) {
           kd_array_.push_back(it.as<double>());
         }
 
         // Basic validation
         if (joint_names_array_.size() == default_position_array_.size() &&
             joint_names_array_.size() == kp_array_.size() &&
             joint_names_array_.size() == kd_array_.size()) {
           has_joint_arrays_ = true;
 
           // Store MoveToDefault-specific kps/kds and default positions
           for (size_t i = 0; i < joint_names_array_.size(); ++i) {
             const std::string &name = joint_names_array_[i];
             double pos = default_position_array_[i];
             double kp_v = kp_array_[i];
             double kd_v = kd_array_[i];
             default_dof_pos[name] = pos;
             
             // Store MoveToDefault kp/kd
             move_to_default_kps[name] = kp_v;
             move_to_default_kds[name] = kd_v;
           }
 
           RCLCPP_INFO(this->get_logger(), "Using joint arrays for Start behavior (size: %zu)", joint_names_array_.size());
         } else {
           RCLCPP_WARN(this->get_logger(), "joint_names/default_position/kp/kd size mismatch; ignoring arrays");
         }
       }
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
             // Check if kps and kds parameters have been received from policy node
             if (!kps_received_ || !kds_received_) {
                 RCLCPP_ERROR(this->get_logger(), 
                             "Cannot switch to POLICY state. Control parameters not received from policy node! kps_received: %s, kds_received: %s", 
                             kps_received_ ? "true" : "false", 
                             kds_received_ ? "true" : "false");
                 return;
             }
             
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
 
     // Print kp/kd values on first execution
     static bool first_move_to_default = true;
     if (first_move_to_default) {
       RCLCPP_INFO(this->get_logger(), "=== First MOVE_TO_DEFAULT execution ===");
       first_move_to_default = false;
     }
 
     if (has_joint_arrays_) {
       // Use provided arrays and dof2motor mapping to command motors
       double ratio = clamp(time_ / duration_, 0.0, 1.0);
       for (size_t j = 0; j < joint_names_array_.size(); ++j) {
         const std::string &dof_name = joint_names_array_[j];
         if (dof2motor_idx.find(dof_name) == dof2motor_idx.end()) {
           continue; // skip unknown names
         }
         int motor_idx = dof2motor_idx[dof_name];
 
         double target_final = default_position_array_[j];
         double target_pos = (1. - ratio) * motor[motor_idx].q + ratio * target_final;
 
         // Current state
         double current_pos = motor[motor_idx].q;
         double current_vel = motor[motor_idx].dq;
 
         // Use MoveToDefault specialized kp/kd
         double kp_to_use = move_to_default_kps[dof_name];
         double kd_to_use = move_to_default_kds[dof_name];
         
         // Print kp/kd values on first execution
         if (time_ <= control_dt_ * 2) { // Print for first few iterations
           RCLCPP_INFO(this->get_logger(), "MoveToDefault - %s: kp=%.2f, kd=%.2f", 
                      dof_name.c_str(), kp_to_use, kd_to_use);
         }
         
         // Apply torque limiting with MoveToDefault gains
         auto [limited_kp, limited_kd] = limitTorqueWithCustomGains(
           dof_name, target_pos, current_pos, current_vel, kp_to_use, kd_to_use);
 
         low_command.motor_cmd[motor_idx].mode = 1;
         low_command.motor_cmd[motor_idx].tau = 0.0;
         low_command.motor_cmd[motor_idx].q = target_pos;
         low_command.motor_cmd[motor_idx].dq = 0.0;
         low_command.motor_cmd[motor_idx].kp = limited_kp;
         low_command.motor_cmd[motor_idx].kd = limited_kd;
       }
     } else {
       // Fall back to map-driven order with default MoveToDefault gains
       // Use default kp/kd values for MoveToDefault since policy kps/kds are not available yet
       const double default_move_kp = 50.0;  // Default stiffness for MoveToDefault
       const double default_move_kd = 5.0;   // Default damping for MoveToDefault
       
       // Print default kp/kd values on first execution
       if (time_ <= control_dt_ * 2) { // Print for first few iterations
         RCLCPP_INFO(this->get_logger(), "MoveToDefault (fallback) - Using default kp=%.2f, kd=%.2f", 
                    default_move_kp, default_move_kd);
       }
       
       for (int i = 0; i < G1_NUM_MOTOR; ++i) {
         std::string dof_name = complete_dof_order[i];
         double ratio = clamp(time_ / duration_, 0.0, 1.0);
         double target_pos = (1. - ratio) * motor[i].q + ratio * default_dof_pos[dof_name];
 
         // Current state
         double current_pos = motor[i].q;
         double current_vel = motor[i].dq;
 
         // Use default MoveToDefault gains with torque limiting
         auto [limited_kp, limited_kd] = limitTorqueWithCustomGains(
           dof_name, target_pos, current_pos, current_vel, default_move_kp, default_move_kd);
 
         low_command.motor_cmd[i].mode = 1;
         low_command.motor_cmd[i].tau = 0.0;
         low_command.motor_cmd[i].q = target_pos;
         low_command.motor_cmd[i].dq = 0.0;
         low_command.motor_cmd[i].kp = limited_kp;
         low_command.motor_cmd[i].kd = limited_kd;
       }
     }
   }
 
   void SendPolicyCommand() {
     time_ += control_dt_;
     low_command.mode_pr = mode_;
     low_command.mode_machine = mode_machine;
 
     // Print kp/kd values on first execution
     static bool first_policy_command = true;
     if (first_policy_command) {
       RCLCPP_INFO(this->get_logger(), "=== First POLICY command execution ===");
       first_policy_command = false;
     }
 
     // Check if kps and kds parameters have been received from policy node
     if (!kps_received_ || !kds_received_) {
       RCLCPP_ERROR(this->get_logger(), 
                   "Policy control parameters not received! kps_received: %s, kds_received: %s", 
                   kps_received_ ? "true" : "false", 
                   kds_received_ ? "true" : "false");
       RCLCPP_ERROR(this->get_logger(), "Cannot execute POLICY commands without control parameters. Triggering emergency stop.");
       current_state_ = RobotState::EMERGENCY_STOP;
       should_shutdown_ = true;
       publishRobotState();
       return;
     }
 
     for (const auto &pair : target_dof_pos) {
       const std::string &dof_name = pair.first;
       const double &target_pos = pair.second;
       int motor_idx = dof2motor_idx[dof_name];
       
       
       // Get policy kp/kd values
       double policy_kp = kps[dof_name];
       double policy_kd = kds[dof_name];
       
       // Print kp/kd values on first execution
       if (time_ <= control_dt_ * 2) { // Print for first few iterations
         RCLCPP_INFO(this->get_logger(), "Policy - %s: kp=%.2f, kd=%.2f", 
                    dof_name.c_str(), policy_kp, policy_kd);
       }
       
       // Use policy kp/kd values directly without torque limiting
       low_command.motor_cmd[motor_idx].mode = 1;
       low_command.motor_cmd[motor_idx].tau = 0.0;
       low_command.motor_cmd[motor_idx].q = target_pos;
       low_command.motor_cmd[motor_idx].dq = 0.0;
       low_command.motor_cmd[motor_idx].kp = policy_kp;
       low_command.motor_cmd[motor_idx].kd = policy_kd;
     }
   }
 
   void SendDampedEmergencyStop() {
     low_command.mode_pr = mode_;
     low_command.mode_machine = mode_machine;
 
     // Use default damping value for emergency stop since kds may not be available
     const double default_emergency_kd = 10.0; // Higher damping for faster stopping
 
     for (int i = 0; i < G1_NUM_MOTOR; ++i) {
       std::string dof_name = complete_dof_order[i];
       low_command.motor_cmd[i].mode = 1; // Keep enabled
       low_command.motor_cmd[i].q = motor[i].q; // Current position
       low_command.motor_cmd[i].dq = 0.0; // Target zero velocity
       low_command.motor_cmd[i].kp = 0.0; // No position control
       low_command.motor_cmd[i].kd = default_emergency_kd; // Use default damping
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
     
     // Trigger emergency stop if any limits are exceeded
     if (limits_exceeded) {
       RCLCPP_ERROR(this->get_logger(), "%s", exceeded_msg.c_str());
       RCLCPP_ERROR(this->get_logger(), "Joint limits exceeded! Triggering emergency stop.");
       // current_state_ = RobotState::EMERGENCY_STOP;
       // should_shutdown_ = true;
       // publishRobotState();
     }
 
     remote_controller.set(message->wireless_remote);
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
           // RCLCPP_WARN(this->get_logger(), 
           //            "Target position would exceed limit for joint %s: %f (scaled limits: %f, %f)", 
           //            dof_name.c_str(), calculated_pos, min_pos, max_pos);
           // Clamp the position to within limits
           calculated_pos = std::clamp(calculated_pos, min_pos, max_pos);
         }
       }
       
       // Set the target position (clamped to safe values if needed)
       target_dof_pos[dof_name] = calculated_pos;
     }
   }
 
   void KpsHandler(const std_msgs::msg::Float32MultiArray::SharedPtr message) {
     policy_kps_data = message->data;
     kps_received_ = true;
     
     // Check if message size matches expected size
     if (policy_kps_data.size() != policy_dof_order.size()) {
       RCLCPP_ERROR(this->get_logger(), 
                   "Policy kps data size mismatch: got %zu, expected %zu", 
                   policy_kps_data.size(), policy_dof_order.size());
       current_state_ = RobotState::EMERGENCY_STOP;
       should_shutdown_ = true;
       publishRobotState();
       return;
     }
     
     // Update kps map with policy data
     for (size_t i = 0; i < policy_dof_order.size(); i++) {
       const auto &dof_name = policy_dof_order[i];
       kps[dof_name] = policy_kps_data[i];
     }
     
     RCLCPP_INFO(this->get_logger(), "Received kps parameters from policy node (size: %zu)", policy_kps_data.size());
   }
 
   void KdsHandler(const std_msgs::msg::Float32MultiArray::SharedPtr message) {
     policy_kds_data = message->data;
     kds_received_ = true;
     
     // Check if message size matches expected size
     if (policy_kds_data.size() != policy_dof_order.size()) {
       RCLCPP_ERROR(this->get_logger(), 
                   "Policy kds data size mismatch: got %zu, expected %zu", 
                   policy_kds_data.size(), policy_dof_order.size());
       current_state_ = RobotState::EMERGENCY_STOP;
       should_shutdown_ = true;
       publishRobotState();
       return;
     }
     
     // Update kds map with policy data
     for (size_t i = 0; i < policy_dof_order.size(); i++) {
       const auto &dof_name = policy_dof_order[i];
       kds[dof_name] = policy_kds_data[i];
     }
     
     RCLCPP_INFO(this->get_logger(), "Received kds parameters from policy node (size: %zu)", policy_kds_data.size());
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
   rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
       kps_subscriber_;
   rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
       kds_subscriber_;
   rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robot_state_publisher_;
   unitree_hg::msg::LowCmd low_command; // Unitree hg lowcmd message
   unitree_hg::msg::IMUState imu;       // Unitree hg IMU message
   unitree_hg::msg::MotorState
       motor[G1_NUM_MOTOR]; // Unitree hg motor state message
   double control_freq_;
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