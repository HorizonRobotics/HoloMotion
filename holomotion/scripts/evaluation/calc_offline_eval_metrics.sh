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

npz_dir="your_npz_dir"
dataset_suffix="HoloMotion_eval"
metric_calculation="per_clip"   # Options: "per_clip" or "per_frame"
dof_mode="23"  # Options: "29" for full DoF, "23" for upper body only

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/evaluation/metrics.py \
    --npz_dir=${npz_dir} \
    --dataset_suffix=${dataset_suffix} \
    --failure_pos_err_thresh_m=0.25 \
    --metric_calculation=${metric_calculation} \
    --dof_mode=${dof_mode}
