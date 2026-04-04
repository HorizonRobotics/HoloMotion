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

holo_src_dir="src_holomotion_npz_dir"
holo_tgt_dir="output_holomotion_npz_dir"

pipeline="['filename_as_motionkey','legacy_to_ref_keys','tagging']"

robot_config="holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/motion_retargeting/holomotion_preprocess.py \
    padding.robot_config_path=${robot_config} \
    io.src_root=${holo_src_dir} \
    io.out_root=${holo_tgt_dir} \
    preprocess.pipeline=${pipeline} \
    ray.enabled=true \
    padding.stand_still_time=20.0 \
    ray.num_workers=2
