#!/bin/bash

##############################################################################
# HoloMotion Deployment Launch Script
#
# This script sets up the complete environment and launches the HoloMotion
# humanoid robot control system for the Unitree G1 robot. It handles:
# 1. ROS2 environment setup and workspace building
# 2. Conda environment configuration for GPU/CUDA support
# 3. Library path configuration for proper linking
# 4. Launch of the complete HoloMotion control pipeline
#
# Prerequisites:
# - Unitree ROS2 SDK properly installed at ~/unitree_ros2/
# - Conda environment 'holomotion_deploy' with required packages
# - Network interface configured for robot communication
# - Proper permissions for robot hardware access
#
# Usage:
#   ./launch_holomotion_29dof_docker.sh [--record]
#   --record: Enable topic recording (optional, disabled by default)
#
# Author: HoloMotion Team
# License: See project LICENSE file
##############################################################################

# Default values
ENABLE_RECORDING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --record)
            ENABLE_RECORDING=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--record]"
            echo "  --record: Enable topic recording (optional, disabled by default)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            echo "Usage: $0 [--record]"
            echo "  --record: Enable topic recording (optional, disabled by default)"
            exit 1
            ;;
    esac
done

echo "Starting HoloMotion 29DOF Docker..."
echo "Recording enabled: $ENABLE_RECORDING"
source /root/miniconda3/etc/profile.d/conda.sh
while [[ ${CONDA_SHLVL:-0} -gt 0 ]]; do
    conda deactivate
done
rm -rf build/ install/ log/
source /opt/ros/humble/setup.sh 
source /root/unitree_ros2/setup.sh

colcon build
source install/setup.bash

# Configure conda environment paths for CUDA and library linking
# NOTE: Update this path to match your actual conda environment location
CONDA_PREFIX="/root/miniconda3/envs/humanoid_deploy"
export CYCLONEDDS_HOME=/root/cyclonedds/install
export CMAKE_PREFIX_PATH=$CYCLONEDDS_HOME:$CMAKE_PREFIX_PATH
source ../../deploy.env
export LD_LIBRARY_PATH=/host_gpu:/cuda_base:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH
# Launch with recording parameter
ros2 launch humanoid_control holomotion_29dof_launch.py enable_recording:=$ENABLE_RECORDING