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

config_name="eval_velocity_tracking"


num_envs=1

ckpt_path="logs/HoloMotionVelocityTracking/20251105_161238-train_g1_29dof_velocity_tracking/model_0.pt"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/evaluation/eval_velocity_tracking.py \
    --config-name=evaluation/${config_name} \
    project_name="HoloMotionVelocityTracking" \
    num_envs=${num_envs} \
    headless=false \
    experiment_name=${config_name} \
    checkpoint=${ckpt_path} \
    +env.config.commands.base_velocity.params.resampling_time_range=[3,5] \
