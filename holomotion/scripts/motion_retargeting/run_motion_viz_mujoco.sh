source train.env

# export MUJOCO_GL="egl"
export MUJOCO_GL="osmesa"

motion_npz_root="data/holomotion_retargeted/AMASS_test"

# "all" for visualizing all motions, or set a specific motion name
export motion_name="all"
# export motion_name="your_motion_name"

$Train_CONDA_PREFIX/bin/python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    +motion_npz_root=${motion_npz_root} \
    +motion_name='${oc.env:motion_name}'
