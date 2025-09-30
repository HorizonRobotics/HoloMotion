# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


source train.env
export CUDA_VISIBLE_DEVICES=""

lmdb_path="path_to_lmdb"
dump_dir="path_to_dump_dir"

# robot_config="unitree/G1/23dof/23dof_training"
robot_config="ZJ-Humanoid/hi2/21dof_training"


# motion_keys="["dance1_subject2_sliced-90-615_padded"]"
motion_keys="[]"

$Train_CONDA_PREFIX/bin/python \
    holomotion/src/misc/export_single_motion.py \
    robot.motion.motion_file=$lmdb_path \
    robot=${robot_config} \
    +motion_keys=${motion_keys} \
    dump_dir=${dump_dir} \
    robot.motion.min_frame_length=0
