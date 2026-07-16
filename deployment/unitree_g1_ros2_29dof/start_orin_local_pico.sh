#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [MODEL_DIR]"
    echo "Without MODEL_DIR, use HoloMotion_motion_tracking_model_v1.4.0 from the repository models directory."
    exit 0
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [MODEL_DIR]"
    exit 2
fi

model_dir="${1:-${HOLOMOTION_MODEL_DIR:-}}"
if [[ -n "$model_dir" ]]; then
    model_dir="$(cd "$model_dir" && pwd -P)"
    model_root="$(dirname "$model_dir")"
    echo "Selected motion policy directory: $model_dir"
    export HOLOMOTION_MODEL_ROOT="$model_root"
    printf -v quoted_model_dir '%q' "$model_dir"
    export HOLOMOTION_CONTAINER_COMMAND="cd /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof && exec bash scripts/launch_with_motion_model.sh $quoted_model_dir"
else
    echo "Selected repository model: HoloMotion_motion_tracking_model_v1.4.0"
    export HOLOMOTION_CONTAINER_COMMAND="cd /home/unitree/holomotion/deployment/unitree_g1_ros2_29dof && exec bash launch_holomotion_29dof_docker.sh"
fi
exec bash "$SCRIPT_DIR/start_container.sh" "$REPO_ROOT"
