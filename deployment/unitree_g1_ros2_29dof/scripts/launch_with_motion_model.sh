#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 MODEL_DIR"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$(cd "$1" && pwd -P)"
BASE_CONFIG="$SCRIPT_DIR/src/config/g1_29dof_holomotion.yaml"
RUNTIME_CONFIG="/tmp/holomotion_robot_config_$(basename "$MODEL_DIR").yaml"
TENSORRT_CACHE_DIR="$SCRIPT_DIR/.cache/tensorrt_engines"

if [[ ! -f "$MODEL_DIR/config.yaml" ]]; then
    echo "Model config not found: $MODEL_DIR/config.yaml"
    exit 1
fi

shopt -s nullglob
onnx_files=("$MODEL_DIR"/exported/*.onnx)
shopt -u nullglob
if [[ ${#onnx_files[@]} -ne 1 ]]; then
    echo "Expected exactly one ONNX file in $MODEL_DIR/exported, found ${#onnx_files[@]}."
    exit 1
fi

echo "Clearing TensorRT engine cache: $TENSORRT_CACHE_DIR"
rm -rf "$TENSORRT_CACHE_DIR"
mkdir -p "$TENSORRT_CACHE_DIR"

python3 - "$BASE_CONFIG" "$RUNTIME_CONFIG" "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
model_dir = Path(sys.argv[3])
config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
config["motion_tracking_model_folder"] = str(model_dir)
target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

echo "Motion policy directory: $MODEL_DIR"
echo "Motion policy ONNX: ${onnx_files[0]}"
exec bash "$SCRIPT_DIR/launch_holomotion_29dof_docker.sh" \
    --set "robot.config_file=$RUNTIME_CONFIG"
