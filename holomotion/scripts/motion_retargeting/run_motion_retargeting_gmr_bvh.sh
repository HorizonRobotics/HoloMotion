# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

bvh_src_dir="data/lafan1_bvh"
gmr_tgt_dir="data/gmr_retargeted/lafan1/"

# Step 1: retargeting to robot dataset from smplx format
# create gmr_tgt_dir if not exists
if [ ! -d "$gmr_tgt_dir" ]; then
    mkdir -p $gmr_tgt_dir
fi

$Train_CONDA_PREFIX/bin/python \
    thirdparties/GMR/scripts/bvh_to_robot_dataset.py \
    --src_folder ${bvh_src_dir}/ \
    --tgt_folder ${gmr_tgt_dir}/ \
    --robot unitree_g1
