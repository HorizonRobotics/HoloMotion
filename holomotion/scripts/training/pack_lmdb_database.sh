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

retargeted_pkl_path="data/retargeted_datasets/waltz"
dump_dir="data/lmdb_datasets/lmdb_g1_21dof_test"
robot_config="unitree/G1/21dof/21dof_training"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/training/pack_lmdb.py \
    robot=$robot_config \
    retargeted_pkl_path=$retargeted_pkl_path \
    lmdb_save_dir=$dump_dir \
    num_jobs=4
