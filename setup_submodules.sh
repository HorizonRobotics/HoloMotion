#!/bin/bash

# Script to set up submodules based on .gitmodules file

echo "Setting up submodules from .gitmodules file..."

# Remove existing empty submodule directories if they exist
echo "Cleaning up existing empty directories..."
rm -rf thirdparties/SMPLSim
rm -rf thirdparties/joints2smpl  
rm -rf thirdparties/omomo_release
rm -rf thirdparties/unitree_ros
rm -rf thirdparties/unitree_ros2
rm -rf thirdparties/tram

# Add each submodule according to .gitmodules
echo "Adding submodules..."

echo "Adding SMPLSim submodule..."
git submodule add -b master https://github.com/ZhengyiLuo/SMPLSim thirdparties/SMPLSim

echo "Adding joints2smpl submodule..."
git submodule add -b main https://github.com/wangsen1312/joints2smpl.git thirdparties/joints2smpl

echo "Adding omomo_release submodule..."
git submodule add -b main https://github.com/lijiaman/omomo_release.git thirdparties/omomo_release

echo "Adding unitree_ros submodule..."
git submodule add -b master https://github.com/unitreerobotics/unitree_ros thirdparties/unitree_ros

echo "Adding unitree_ros2 submodule..."
git submodule add -b master https://github.com/unitreerobotics/unitree_ros2 thirdparties/unitree_ros2

echo "Adding tram submodule..."
git submodule add -b main https://github.com/yufu-wang/tram.git thirdparties/tram

echo "Initializing and updating submodules..."
git submodule update --init --recursive

echo "Submodule setup complete!"
echo "Verifying submodule status:"
git submodule status