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

export CUDA_VISIBLE_DEVICES=1

if [[ $(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l) -eq 1 ]]; then
    USE_MULTI_GPU=false
else
    USE_MULTI_GPU=true
fi

config_name="train_g1_29dof_motion_tracking"

# ckpt_path="your_pretrained_ckpt_path"

num_envs=4096

COMMON_ARGS=(
    "holomotion/src/training/train.py"
    "--config-name=training/motion_tracking/${config_name}"
    "num_envs=${num_envs}"
    "headless=true"
    "experiment_name=${config_name}"
    # "checkpoint=${ckpt_path}"
)

trap cleanup SIGINT SIGTERM
if [[ "${USE_MULTI_GPU}" == "true" ]]; then
    ${Train_CONDA_PREFIX}/bin/accelerate launch \
        --multi_gpu \
        "${COMMON_ARGS[@]}"
else
    ${Train_CONDA_PREFIX}/bin/accelerate launch \
        "${COMMON_ARGS[@]}"
fi
wait ${TRAIN_PID}
trap - SIGINT SIGTERM
