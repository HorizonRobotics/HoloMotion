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

export MUJOCO_GL="osmesa"

motion_npz_root="path_to_your_npz_dir"

export motion_name="all"


$Train_CONDA_PREFIX/bin/python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    +key_prefix="robot_" \
    +draw_ref_body_spheres=true \
    +ref_key_prefix="ref_" \
    +motion_npz_root=${motion_npz_root} \
    skip_frames=6 \
    max_workers=11 \
    +motion_name='${oc.env:motion_name}'
