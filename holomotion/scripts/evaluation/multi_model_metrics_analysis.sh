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
metrics_json_dir="logs/Holomotion/metrics_output_dataset"

${Train_CONDA_PREFIX}/bin/python \
  holomotion/src/evaluation/multi_model_metrics_report.py \
  --json_dir="$metrics_json_dir"
