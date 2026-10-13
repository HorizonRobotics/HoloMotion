#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/holomotion
DEPLOY="$ROOT/deployment/unitree_g1_ros2_29dof"
PYTHON=/root/miniconda3/envs/holomotion_deploy/bin/python
MOTION="$DEPLOY/src/models/v2_motion_model"
VELOCITY="$DEPLOY/src/models/velocity_tracking_model"
ASSET="$ROOT/assets/robots/unitree/G1/29dof/g1_29dof_rev_1_0.urdf"
LAUNCH="$DEPLOY/launch_holomotion_29dof_docker.sh"

usage() {
    printf '%s\n' \
        'Usage: holomotion check [--structure-only]' \
        '       holomotion control-check (read-only official controller status)' \
        '       holomotion ready (packaged isolated inference check)' \
        '       holomotion teleop [--mode Pelvis-1|Keypoint-5] [launcher arguments]' \
        '       holomotion offline  (unsupported for v2 live-reference policy)' \
        '       holomotion shell  (interactive shell, no services started)' \
        'check never starts ROS, Pico, or robot control.' \
        '--structure-only checks bundle/imports, not Orin GPU/TensorRT readiness.'
}

check_structure() {
    test -f "$DEPLOY/install/setup.bash"
    test -f "$VELOCITY/config.yaml"
    test -f "$DEPLOY/src/config/g1_29dof_holomotion_v2.yaml"
    shopt -s nullglob
    local motion_models=("$MOTION"/exported/*.onnx)
    local velocity_models=("$VELOCITY"/exported/*.onnx)
    [[ ${#motion_models[@]} -eq 1 && ${#velocity_models[@]} -eq 1 ]] || { echo "Expected one ONNX per policy bundle" >&2; return 1; }
    "$PYTHON" "$DEPLOY/scripts/check_v2_motion_model.py" \
        --config "$MOTION/config.yaml" \
        --mode-table "$MOTION/link_command_modes.json" \
        --onnx "${motion_models[0]}" \
        --asset "$ASSET" \
        --velocity-config "$VELOCITY/config.yaml" \
        --velocity-onnx "${velocity_models[0]}" \
        --robot-config "$DEPLOY/src/config/g1_29dof_holomotion_v2.yaml"
}

check_device() {
    "$PYTHON" -c 'import onnxruntime as ort, torch; p=ort.get_available_providers(); assert "TensorrtExecutionProvider" in p and "CUDAExecutionProvider" in p, f"Jetson ONNX providers unavailable: {p}"; print("CPU Torch FK import and ORT CUDA/TensorRT provider presence PASSED; GPU session not tested; no robot action sent.")'
}

run_control_ownership() {
    CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-/root/cyclonedds/install}" \
    LD_LIBRARY_PATH="${CYCLONEDDS_HOME:-/root/cyclonedds/install}/lib:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}" \
        "$PYTHON" "$DEPLOY/scripts/control_ownership.py" "$@"
}

case "${1:-check}" in
    check)
        shift || true
        if [[ "${1:-}" == "--structure-only" && $# -eq 1 ]]; then
            check_structure
            echo "STRUCTURE-ONLY PASS; GPU/Orin/robot behavior was NOT tested."
        elif [[ $# -eq 0 ]]; then
            check_structure
            check_device
        else
            usage >&2
            exit 2
        fi
        ;;
    ready)
        shift
        exec /usr/local/bin/holomotion-ready "$@"
        ;;
    control-check)
        shift
        run_control_ownership --check-only "$@"
        ;;
    teleop)
        shift
        mode=Keypoint-5
        if [[ "${1:-}" == "--mode" ]]; then
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            mode="$2"
            shift 2
        fi
        case "$mode" in
            Pelvis-1|Keypoint-5) ;;
            *) echo "Unsupported v2 control mode: $mode" >&2; exit 2 ;;
        esac
        if [[ -f /opt/holomotion-ready/bundle.sha256 ]]; then
            (cd / && sha256sum --quiet -c /opt/holomotion-ready/bundle.sha256)
        else
            check_structure
            check_device
        fi
        ownership_args=()
        if [[ "${HOLOMOTION_ISOLATED_TEST:-0}" == 1 ]]; then
            ownership_args=(--isolated-test)
        fi
        run_control_ownership "${ownership_args[@]}" "$@"
        exec bash "$LAUNCH" \
            --set runtime.build_before_launch=false \
            --set robot.config_file=config/g1_29dof_holomotion_v2.yaml \
            --set "policy.link_control_mode=$mode" "$@"
        ;;
    offline)
        echo "HoloMotion v2 stage 1 uses the live Pico reference; offline NPZ replay is unsupported." >&2
        exit 2
        ;;
    shell)
        shift
        exec /bin/bash "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
