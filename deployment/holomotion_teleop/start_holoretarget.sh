#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${HOLOMOTION_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENV_NAME="${HOLOMOTION_TELEOP_ENV_NAME:-holomotion_teleop}"
if [[ -n "${HOLOMOTION_PY:-}" ]]; then
  PY="$HOLOMOTION_PY"
elif command -v conda >/dev/null 2>&1; then
  PY="$(conda info --base)/envs/$ENV_NAME/bin/python"
else
  echo "Conda not found. Set HOLOMOTION_PY to the teleoperation Python executable." >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "Teleoperation Python not found or not executable: $PY" >&2
  echo "Run setup_holomotion_teleop_x86_ubuntu2204.sh or set HOLOMOTION_PY." >&2
  exit 1
fi
URI="${HOLORETARGET_ZMQ_URI:-tcp://*:6001}"
HZ="${HOLORETARGET_HZ:-50}"
TIMING_EVERY="${HOLORETARGET_TIMING_LOG_EVERY:-200}"
RECORD_ROOT="${HOLORETARGET_RECORD_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/holomotion/recordings}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

stamp="$(date +%Y%m%d_%H%M%S)"
record_dir="$RECORD_ROOT/$stamp"
mkdir -p "$record_dir"

echo "HoloRetarget running in foreground."
echo "  uri: $URI"
echo "  recording: $record_dir"
echo "Stop and save with Ctrl+C."

exec "$PY" deployment/holomotion_teleop/holomotion_teleop_node.py \
  --hz "$HZ" \
  --robot-zmq-uri "$URI" \
  --robot-zmq-mode bind \
  --timing-log-every "$TIMING_EVERY" \
  --save-reference-path "$record_dir/reference_qpos.npz" \
  --debug-retarget-dump "$record_dir/retarget_debug.npz" \
  "$@"
