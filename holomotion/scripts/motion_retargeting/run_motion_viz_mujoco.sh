source train.env

export MUJOCO_GL="osmesa"

motion_npz_root="data/holomotion_retargeted/processed_datasets/AMASS_test"

# "all" for visualizing all motions, or set a specific motion name
export motion_name="all"

$Train_CONDA_PREFIX/bin/python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    +key_prefix="ref_" \
    +motion_npz_root=${motion_npz_root} \
    skip_frames=1 \
    max_workers=16 \
    +motion_name='${oc.env:motion_name}'
