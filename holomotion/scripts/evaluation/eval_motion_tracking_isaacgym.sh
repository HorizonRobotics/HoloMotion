#!/bin/bash
source train.env
export CUDA_VISIBLE_DEVICES="0"

# Configuration
checkpoint_path="logs/HoloMotion/20250804_205352-train_unitree_g1_21dof_student/model_0.pt" # REQUIRED: Set this to your checkpoint path
lmdb_path="data/lmdb_datasets/lmdb_g1_21dof_test"
num_envs=4

${Train_CONDA_PREFIX}/bin/accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --main_process_port=29501 \
    holomotion/src/evaluation/eval_motion_tracking.py \
    use_accelerate=True \
    num_envs=${num_envs} \
    headless=True \
    export_policy=True \
    env.config.termination.terminate_when_motion_far=False \
    +robot.motion.motion_file="${lmdb_path}" \
    checkpoint="${checkpoint_path}"
