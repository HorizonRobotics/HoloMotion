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

export HEADLESS=false
if $HEADLESS; then
    export MUJOCO_GL="osmesa"
    export RECORD_VIDEO=true
else
    export MUJOCO_GL="egl"
    export RECORD_VIDEO=false
fi

model_type="${model_type:-holomotion}"

robot_xml_path="assets/robots/unitree/G1/29dof/scene_29dof.xml"

ONNX_PATH="your_onnx_model.onnx"

export motion_npz_path="your_npz.npz"

${Train_CONDA_PREFIX}/bin/python holomotion/src/evaluation/eval_mujoco_sim2sim.py \
    record_video=$RECORD_VIDEO \
    headless=$HEADLESS \
    camera_tracking=true \
    camera_distance=7.0 \
    +model_type=${model_type} \
    use_gpu=true \
    dump_npzs=true \
    dump_onnx_io_npy=false \
    calc_per_clip_metrics=true \
    generate_report=true \
    ray_actors_per_gpu=12 \
    policy_action_delay_step=0 \
    action_delay_type=step \
    +ckpt_onnx_path="$ONNX_PATH" \
    +motion_npz_path='${oc.env:motion_npz_path}' \
    robot_xml_path=${robot_xml_path}
