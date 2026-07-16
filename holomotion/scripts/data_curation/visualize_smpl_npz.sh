#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TOOL_DIR="${REPO_ROOT}/holomotion/src/data_curation"

if [[ -n "${GVHMR_PYTHON:-}" ]]; then
  PYTHON_BIN="${GVHMR_PYTHON}"
elif [[ -n "${GVHMR_CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="${GVHMR_CONDA_PREFIX}/bin/python"
elif command -v conda >/dev/null 2>&1; then
  PYTHON_BIN="$(conda info --base)/envs/gvhmr/bin/python"
else
  echo "Unable to locate the GVHMR Python interpreter." >&2
  echo "Set GVHMR_CONDA_PREFIX or GVHMR_PYTHON explicitly." >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "GVHMR Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${TOOL_DIR}"
exec "${PYTHON_BIN}" visualize_smpl_npz.py "$@"
