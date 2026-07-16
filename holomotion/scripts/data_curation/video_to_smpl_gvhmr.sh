#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CONDA_BASE="$(conda info --base)"
Train_CONDA_PREFIX="${GVHMR_CONDA_PREFIX:-${CONDA_BASE}/envs/gvhmr}"

video_folder_root="${VIDEO_FOLDER_ROOT:-${REPO_ROOT}/data/video_data}"
npz_data_root="${NPZ_DATA_ROOT:-${REPO_ROOT}/data/gvhmr_converted/gvhmr_result}"
out_dir="${SMPL_OUTPUT_ROOT:-${REPO_ROOT}/data/gvhmr_converted/collected_smpl}"

cd "${REPO_ROOT}/thirdparties/GVHMR"

"${Train_CONDA_PREFIX}/bin/python" "${REPO_ROOT}/holomotion/src/data_curation/video_to_smpl_gvhmr.py" \
    --folder="${video_folder_root}" \
    --output_root="${npz_data_root}" \
    -s

mkdir -p "${out_dir}"
for subdir in "${npz_data_root}"/*; do
    if [[ ! -d "${subdir}" ]]; then
        continue
    fi

    sub_name=$(basename "${subdir}")
    src_npz="${subdir}/smpl.npz"

    if [[ ! -f "${src_npz}" ]]; then
        echo "[SKIP] ${sub_name}: smpl.npz not found"
        continue
    fi

    dst_npz="${out_dir}/${sub_name}_smpl.npz"

    cp -f "${src_npz}" "${dst_npz}"
    echo "[COPY] ${src_npz} -> ${dst_npz}"
done
