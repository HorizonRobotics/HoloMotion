#!/usr/bin/env bash

# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

source train.env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOLOMOTION_FINETUNE_CHECKPOINT="${HOLOMOTION_FINETUNE_CHECKPOINT:-$REPO_ROOT/checkpoints/holomotion_v1.4/model_14000.pt}"

checkpoint="$HOLOMOTION_FINETUNE_CHECKPOINT"
checkpoint_stem="${checkpoint%.pt}"
required_files=(
    "$checkpoint"
    "$checkpoint_stem/actor/model.safetensors"
    "$checkpoint_stem/critic/model.safetensors"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -s "$required_file" ]]; then
        echo "Missing finetune checkpoint file: $required_file" >&2
        echo "Download the complete v1.4 checkpoint package first." >&2
        exit 1
    fi
done

config_name="motion_tracking_v1_4_0_finetune"
num_envs="${NUM_ENVS:-4096}"

common_args=(
    "holomotion/src/training/train.py"
    "--config-name=training/motion_tracking/${config_name}"
    "num_envs=${num_envs}"
    "headless=true"
    "experiment_name=${config_name}"
    "$@"
)

gpu_count="$(printf '%s\n' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)"
launch_args=()
if [[ "$gpu_count" -gt 1 ]]; then
    launch_args+=(--multi_gpu)
fi

if [[ "${HOLOMOTION_FINETUNE_DRY_RUN:-0}" == "1" ]]; then
    printf 'Finetune command:'
    printf ' %q' "${Train_CONDA_PREFIX}/bin/accelerate" launch
    printf ' %q' "${launch_args[@]}" "${common_args[@]}"
    printf '\n'
    exit 0
fi

"${Train_CONDA_PREFIX}/bin/accelerate" launch \
    "${launch_args[@]}" \
    "${common_args[@]}" &
TRAIN_PID=$!
trap cleanup SIGINT SIGTERM
wait "$TRAIN_PID"
status=$?
trap - SIGINT SIGTERM
exit "$status"
