#!/bin/bash
##############################################################################
# HoloMotion Environment Deployment Script
#
# This script sets up the complete environment for HoloMotion humanoid robot
# system deployment. It handles:
# 1. Conda environment creation with all dependencies (CUDA, PyTorch, etc.)
# 2. Special dependencies (unitree_sdk2_python)  
# 3. ROS2 workspace compilation
#
# Prerequisites:
# - Anaconda/Miniconda installed
# - ROS2 Humble installed at /opt/ros/humble/
# - Unitree ROS2 SDK at ~/unitree_ros2/
#
# Usage:
#   chmod +x deploy_environment.sh
#   ./deploy_environment.sh [environment_name]
#
# Arguments:
#   environment_name: Optional. Name for the conda environment (default: holomotion_deploy)
#
# Examples:
#   ./deploy_environment.sh                    # Uses default name 'holomotion_deploy'
#   ./deploy_environment.sh my_robot_env      # Uses custom name 'my_robot_env'
#
# Author: HoloMotion Team
##############################################################################

set -e  # Exit on any error

# Parse command line arguments
ENV_NAME="${1:-holomotion_deploy}"

echo "🚀 Starting HoloMotion Environment Deployment..."
echo "📝 Environment name: $ENV_NAME"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Script directory: $SCRIPT_DIR"

# Step 1: Create conda environment with all dependencies
echo ""
echo "📦 Step 1: Creating conda environment with all dependencies..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists. Removing it..."
    conda env remove -n "$ENV_NAME" -y
fi

echo "🔧 Creating new environment from environment_deploy.yaml..."
echo "   This will install: PyTorch (CUDA), NumPy, SciPy, ONNX Runtime, and all other dependencies..."
cd "$PROJECT_ROOT"
conda env create -f holomotion/environment_deploy.yaml -n "$ENV_NAME"

echo "✅ Conda environment with all dependencies created successfully!"

# Step 2: Install unitree_sdk2_python
echo ""
echo "📦 Step 2: Installing unitree_sdk2_python..."

# Function to run commands in conda environment
run_in_env() {
    conda run -n "$ENV_NAME" "$@"
}

echo "🔧 Installing unitree_sdk2_python..."
if [ ! -d "$HOME/unitree_sdk2_python" ]; then
    echo "📥 Cloning unitree_sdk2_python repository..."
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git "$HOME/unitree_sdk2_python"
fi

echo "🔧 Installing unitree_sdk2_python in development mode..."
cd "$HOME/unitree_sdk2_python"
run_in_env pip install -e .

echo "✅ unitree_sdk2_python installed successfully!"

# Step 3: Setup ROS2 workspace
echo ""
echo "📦 Step 3: Setting up ROS2 workspace..."

# Ensure conda environment is completely deactivated for ROS2 compilation
echo "🔧 Ensuring conda environment is completely deactivated..."

# Initialize conda for this script
eval "$(conda shell.bash hook)"

# Deactivate any active conda environments
while [[ "$CONDA_DEFAULT_ENV" != "" && "$CONDA_DEFAULT_ENV" != "base" ]]; do
    echo "  Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
done

# If we're still in base environment, deactivate it too
if [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    echo "  Deactivating base conda environment"
    conda deactivate
fi

echo "  ✅ Conda environment fully deactivated"

# Check ROS2 installation
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
    echo "❌ ROS2 Humble not found at /opt/ros/humble/"
    echo "   Please install ROS2 Humble first: https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

# Check Unitree ROS2 SDK
if [ ! -f "$HOME/unitree_ros2/setup.sh" ]; then
    echo "❌ Unitree ROS2 SDK not found at ~/unitree_ros2/"
    echo "   Please install Unitree ROS2 SDK first"
    exit 1
fi

echo "🔧 Compiling ROS2 workspace..."
cd "$PROJECT_ROOT/holomotion/deployment/unitree_g1_ros2"

# Clean previous build
rm -rf build install log

# Source ROS2 and Unitree setup
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/setup.sh

# Build workspace
echo "🏗️  Building workspace with colcon..."
colcon build

echo "✅ ROS2 workspace compiled successfully!"

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Summary of installed packages:"
echo "   ✅ PyTorch 2.3.1 with CUDA 12.1 support"  
echo "   ✅ ONNX Runtime for neural network inference"
echo "   ✅ SMPLX for humanoid motion processing"
echo "   ✅ Scientific computing packages (NumPy, SciPy, etc.)"
echo "   ✅ Unitree SDK2 Python bindings"
echo "   ✅ ROS2 workspace compiled"
echo ""
echo "📋 To run the system:"
echo "1. Activate the conda environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Launch the system:"
echo "   cd $PROJECT_ROOT/holomotion/deployment/unitree_g1_ros2"
echo "   bash launch_holomotion.sh"
echo ""
echo "✅ Environment '$ENV_NAME' setup complete!"
echo "🚀 Ready for robot deployment!"
