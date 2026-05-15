#!/bin/bash

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 SCRIPT_DIR DEFAULT_PROFILE LABEL DISPLAY_COMMAND [launch args...]"
    exit 1
fi

SCRIPT_DIR="$1"
DEFAULT_PROFILE="$2"
LAUNCH_LABEL="$3"
DISPLAY_COMMAND="$4"
shift 4

cd "$SCRIPT_DIR"

PROFILE_PATH="$DEFAULT_PROFILE"
PROFILE_OVERRIDE=""
PROFILE_READER="$SCRIPT_DIR/src/humanoid_policy/launch_profile.py"
PROFILE_PYTHON="${PROFILE_PYTHON:-python3}"

usage() {
    echo "Usage: $DISPLAY_COMMAND [--profile PATH] [--profile-override OVERRIDE] [--set OVERRIDE] [--record]"
    echo "  --profile PATH: Launch profile YAML path. Relative paths are resolved from this script directory."
    echo "  --profile-override OVERRIDE: YAML file, YAML/JSON mapping, or dotted key=value override."
    echo "  --set OVERRIDE: Alias for --profile-override."
    echo "  --record: Enable rosbag recording via recording.enabled=true override."
}

append_profile_override() {
    local item="$1"
    if [[ -z "$PROFILE_OVERRIDE" ]]; then
        PROFILE_OVERRIDE="$item"
    else
        PROFILE_OVERRIDE="${PROFILE_OVERRIDE};${item}"
    fi
}

resolve_profile_path() {
    local path="$1"
    if [[ "$path" != /* ]]; then
        path="$SCRIPT_DIR/$path"
    fi
    if command -v realpath >/dev/null 2>&1; then
        realpath -m "$path"
        return
    fi

    local dir
    local base
    dir="$(dirname "$path")"
    base="$(basename "$path")"
    if [[ -d "$dir" ]]; then
        printf "%s/%s\n" "$(cd "$dir" && pwd -P)" "$base"
    else
        printf "%s\n" "$path"
    fi
}

source_runtime_file() {
    local label="$1"
    local path="$2"
    if [[ -z "$path" ]]; then
        return
    fi
    if [[ ! -f "$path" ]]; then
        echo "$label not found: $path"
        exit 1
    fi
    echo "Sourcing $label: $path"
    local had_nounset=0
    local status=0
    if [[ $- == *u* ]]; then
        had_nounset=1
        set +u
    fi
    source "$path" || status=$?
    if [[ "$had_nounset" == "1" ]]; then
        set -u
    fi
    return "$status"
}

deactivate_conda_stack() {
    if ! command -v conda >/dev/null 2>&1; then
        return
    fi
    while [[ ${CONDA_SHLVL:-0} -gt 0 ]]; do
        conda deactivate
    done
}

clean_workspace() {
    rm -rf build/ install/ log/ 2>/dev/null || sudo rm -rf build/ install/ log/
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE_PATH="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE_PATH="${1#*=}"
            shift
            ;;
        --profile-override|--set)
            append_profile_override "$2"
            shift 2
            ;;
        --profile-override=*|--set=*)
            append_profile_override "${1#*=}"
            shift
            ;;
        --record)
            append_profile_override "recording.enabled=true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

PROFILE_PATH="$(resolve_profile_path "$PROFILE_PATH")"
if [[ ! -f "$PROFILE_PATH" ]]; then
    echo "Launch profile not found: $PROFILE_PATH"
    exit 1
fi
if [[ ! -f "$PROFILE_READER" ]]; then
    echo "Launch profile reader not found: $PROFILE_READER"
    exit 1
fi

echo "Starting $LAUNCH_LABEL..."
echo "Launch profile: $PROFILE_PATH"
echo "Profile override: ${PROFILE_OVERRIDE:-<none>}"

RUNTIME_EXPORTS="$("$PROFILE_PYTHON" "$PROFILE_READER" runtime-shell \
    --profile "$PROFILE_PATH" \
    --override "$PROFILE_OVERRIDE" \
    --root "$SCRIPT_DIR")" || {
    echo "Failed to read launch profile runtime fields with $PROFILE_PYTHON."
    exit 1
}
eval "$RUNTIME_EXPORTS"
echo "Runtime build before launch: $HLM_RUNTIME_BUILD_BEFORE_LAUNCH"

source_runtime_file "Conda setup" "$HLM_RUNTIME_CONDA_SETUP"
deactivate_conda_stack
source_runtime_file "ROS setup" "$HLM_RUNTIME_ROS_SETUP"
source_runtime_file "Unitree setup" "$HLM_RUNTIME_UNITREE_SETUP"

if [[ "$HLM_RUNTIME_BUILD_BEFORE_LAUNCH" == "true" ]]; then
    clean_workspace
    colcon build
fi
if [[ ! -f install/setup.bash ]]; then
    echo "ROS workspace install/setup.bash not found. Enable runtime.build_before_launch or run colcon build first."
    exit 1
fi
source_runtime_file "Workspace setup" "$SCRIPT_DIR/install/setup.bash"

if [[ -n "$HLM_RUNTIME_CYCLONEDDS_HOME" ]]; then
    export CYCLONEDDS_HOME="$HLM_RUNTIME_CYCLONEDDS_HOME"
    export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME:${CMAKE_PREFIX_PATH:-}"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${LIBRARY_PATH:-}"
source_runtime_file "Deploy env" "$HLM_RUNTIME_DEPLOY_ENV"
if [[ -n "$HLM_RUNTIME_EXTRA_LD_LIBRARY_PATHS" ]]; then
    export LD_LIBRARY_PATH="$HLM_RUNTIME_EXTRA_LD_LIBRARY_PATHS:$LD_LIBRARY_PATH"
fi

LAUNCH_ARGS=("launch_profile:=$PROFILE_PATH")
if [[ -n "$PROFILE_OVERRIDE" ]]; then
    LAUNCH_ARGS+=("profile_override:=$PROFILE_OVERRIDE")
fi

ros2 launch humanoid_control holomotion_29dof_launch.py "${LAUNCH_ARGS[@]}"
