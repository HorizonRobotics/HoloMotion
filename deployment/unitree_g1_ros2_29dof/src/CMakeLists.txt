cmake_minimum_required(VERSION 3.8)
project(humanoid_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


include_directories(include include/common include/nlohmann)
link_directories(src)

set(
  DEPENDENCY_LIST
  unitree_go
  unitree_hg
  unitree_api
  rclcpp
  std_msgs
  rosbag2_cpp
  yaml-cpp
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)


# Main control executable
add_executable(
  humanoid_control
  src/main_node.cpp
  src/common/motor_crc_hg.cpp
  src/common/wireless_controller.cpp
)


ament_target_dependencies(humanoid_control ${DEPENDENCY_LIST})


# Install Python modules
ament_python_install_package(humanoid_policy)

# Install Python scripts as executables

install(PROGRAMS
  humanoid_policy/policy_node_29dof.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME policy_node_29dof
)
install(PROGRAMS
  scripts/close_ai_sport.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
# Install your models directory
install(DIRECTORY
  models/
  DESTINATION share/${PROJECT_NAME}/models
)

install(TARGETS
  humanoid_control
  DESTINATION lib/${PROJECT_NAME})

# motion folder
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY
  motion_data/
  DESTINATION share/${PROJECT_NAME}/motion_data
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
