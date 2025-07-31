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
#   ./launch_holomotion.sh
#
# Author: HoloMotion Team
# License: See project LICENSE file
##############################################################################

source ~/unitree_ros2/setup.sh

colcon build
source install/setup.bash

# Configure conda environment paths for CUDA and library linking
# NOTE: Update this path to match your actual conda environment location
CONDA_PREFIX="$HOME/miniconda3/envs/holomotion_deploy"
# Alternative for Anaconda users: CONDA_PREFIX="$HOME/anaconda3/envs/holomotion_deploy"

export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/stubs
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH

ros2 launch humanoid_control holomotion.launch.py