export CONDA_BASE=$(conda info --base)
export Train_CONDA_PREFIX="$CONDA_BASE/envs/gvhmr"

$Train_CONDA_PREFIX/bin/python ../../holomotion/src/data_curation/video_to_smpl_gvhmr.py