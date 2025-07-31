#!/bin/bash
source train.env
export CUDA_VISIBLE_DEVICES="0"

# Fix for RTX 4000 series GPU P2P communication issue
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Configuration
checkpoint_path="logs/HoloMotion/20250710_143424-exp_unitree_g1_21dof_teacher/model_10000.pt" # REQUIRED: Set this to your checkpoint path
lmdb_path="data/lmdb_datasets/lmdb_g1_21dof_test"
num_envs=4

# Run evaluation with accelerate launch (similar to training)
${Train_CONDA_PREFIX}/bin/accelerate launch \
    holomotion/src/evaluation/eval_motion_tracking.py \
    use_accelerate=True \
    num_envs=${num_envs} \
    headless=True \
    export_policy=True \
    env.config.termination.terminate_when_motion_far=False \
    robot.motion.motion_file="${lmdb_path}" \
    checkpoint="${checkpoint_path}"
