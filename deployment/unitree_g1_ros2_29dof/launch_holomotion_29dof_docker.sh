#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROFILE="$SCRIPT_DIR/launch_profiles/orin_docker.yaml"

exec bash "$SCRIPT_DIR/scripts/launch_runtime.sh" \
    "$SCRIPT_DIR" \
    "$DEFAULT_PROFILE" \
    "HoloMotion 29DOF Docker" \
    "$0" \
    "$@"
