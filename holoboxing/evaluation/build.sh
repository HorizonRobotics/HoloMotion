#!/usr/bin/env bash
# Maintainer entrypoint. This only builds locally; it never publishes an image.
set -euo pipefail
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${HOLOBOXING_BUILD_PYTHON:-python3}" "$base/scripts/build_image.py" "$@"
