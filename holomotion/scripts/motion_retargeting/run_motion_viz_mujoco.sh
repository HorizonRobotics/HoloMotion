export MUJOCO_GL="osmesa"
# export MUJOCO_GL="egl"

# retargeted path mosified by user
motion_pkl_root="assets/test_data/motion_retargeting/retargeted_datasets/phc_retargeted"
# "all" for default 
motion_name="all"
# modify end

python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    +motion_pkl_root=${motion_pkl_root} \
    +motion_name=${motion_name}