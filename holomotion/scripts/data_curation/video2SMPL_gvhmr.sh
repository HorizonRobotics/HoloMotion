export CONDA_BASE=$(conda info --base)
export Train_CONDA_PREFIX="$CONDA_BASE/envs/gvhmr"

video_folder_root="holomotion_abs_path/data/video_data"
npz_data_root="holomotion_abs_path/data/gvhmr_converted/gvhmr_result"
out_dir="holomotion_abs_path/data/gvhmr_converted/collected_smpl"

cd thirdparties/GVHMR/

$Train_CONDA_PREFIX/bin/python ../../holomotion/src/data_curation/video2SMPL_gvhmr.py \
    --folder=${video_folder_root} \
    --output_root=${npz_data_root} \
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