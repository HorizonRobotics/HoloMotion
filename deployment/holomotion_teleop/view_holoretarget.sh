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
URI="${HOLORETARGET_VIEW_URI:-tcp://127.0.0.1:6001}"
VIEWER_FPS="${HOLORETARGET_VIEWER_FPS:-50}"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

exec "$PY" deployment/holomotion_teleop/holomotion_teleop_mjviewer.py \
  --uri "$URI" \
  --viewer-fps "$VIEWER_FPS" \
  "$@"
