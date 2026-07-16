#!/bin/bash

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
export CUDA_VISIBLE_DEVICES="0"

HEADLESS=true
CONFIG_NAME="eval_isaaclab"

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to a trained v1.4.0 checkpoint}"

eval_h5_dataset_path="${EVAL_H5_DATASET_PATH:-['data/h5v2_datasets/lafan1']}"

num_envs="${NUM_ENVS:-4}"


${Train_CONDA_PREFIX}/bin/accelerate launch \
    holomotion/src/evaluation/eval_motion_tracking_single.py \
    --config-name=evaluation/${CONFIG_NAME} \
    headless=${HEADLESS} \
    num_envs=${num_envs} \
    export_policy=true \
    dump_npzs=true \
    calc_per_clip_metrics=true \
    generate_report=true \
    motion_h5_path=${eval_h5_dataset_path} \
    +use_kv_cache=true \
    export_only=false \
    checkpoint=$CKPT_PATH \
    project_name="HoloMotionMoTrack"
