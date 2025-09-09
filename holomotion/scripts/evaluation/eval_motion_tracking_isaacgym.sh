#!/bin/bash
source train.env
export CUDA_VISIBLE_DEVICES="0"

eval_config="eval_isaacgym"
# eval_config="eval_isaacgym_with_dr"

checkpoint_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/logs/HoloMotionDebug/20250909_175315-train_ZJ-Humanoid-hi2_21dof_teacher_stage2/model_0.pt"
lmdb_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/lmdb_datasets/lmdb_ZJ-Humanoid-hi2_21dof_train"

num_envs=4

${Train_CONDA_PREFIX}/bin/accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --main_process_port=29501 \
    holomotion/src/evaluation/eval_motion_tracking.py \
    --config-name=evaluation/${eval_config} \
    use_accelerate=true \
    num_envs=${num_envs} \
    env.config.align_marker_to_root=false \
    headless=false \
    export_policy=true \
    env.config.termination.terminate_by_gravity=true \
    env.config.termination.terminate_by_low_height=false \
    env.config.termination.terminate_when_motion_far=false \
    env.config.termination.terminate_when_ee_z_far=false \
    motion_lmdb_path="${lmdb_path}" \
    checkpoint="${checkpoint_path}"
