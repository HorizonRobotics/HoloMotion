#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_IP="${1:-${HOLOMOTION_ROBOT_IP:-192.168.123.164}}"
if [[ $# -gt 0 ]]; then
  shift
fi

export HOLORETARGET_VIEW_URI="tcp://${ROBOT_IP}:6002"
exec bash "$SCRIPT_DIR/view_holoretarget.sh" "$@"
