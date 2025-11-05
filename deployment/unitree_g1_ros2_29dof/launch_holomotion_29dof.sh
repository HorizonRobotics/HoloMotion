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
#   ./launch_holomotion_29dof.sh [--record]
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

echo "Starting HoloMotion 29DOF..."
echo "Recording enabled: $ENABLE_RECORDING"

source ~/unitree_ros2/setup.sh

colcon build
source install/setup.bash

# Configure conda environment paths for CUDA and library linking
CONDA_PREFIX="$HOME/anaconda3/envs/holomotion_deploy"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate holomotion_deploy || true
fi
# Ensure the conda env python is preferred even if activation above is skipped
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"

export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/stubs
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH

# Launch with recording parameter
ros2 launch humanoid_control holomotion_29dof_launch.py enable_recording:=$ENABLE_RECORDING