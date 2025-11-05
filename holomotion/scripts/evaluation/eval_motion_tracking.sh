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
export CUDA_VISIBLE_DEVICES="0"

config_name="eval_isaaclab"

num_envs=1

ckpt_path="your_pretrained_model_ckpt"
eval_h5_dataset_path="data/hdf5_datasets/h5_g1_29dof_amass_test"

${Train_CONDA_PREFIX}/bin/accelerate launch \
    holomotion/src/evaluation/eval_motion_tracking.py \
    --config-name=evaluation/${config_name} \
    headless=false \
    project_name="HoloMotionMotionTracking" \
    num_envs=${num_envs} \
    experiment_name=${config_name} \
    checkpoint=${ckpt_path} \
    motion_h5_path=${eval_h5_dataset_path}
