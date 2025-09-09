source train.env

export MUJOCO_GL="osmesa"


motion_pkl_root="assets/test_data/motion_retargeting/retargeted_datasets/phc_retargeted"

# "all" for default 
motion_name="all"

robot_config="ZJ-Humanoid_hi2_retargeting_config"

${Train_CONDA_PREFIX}/bin/python holomotion/src/motion_retargeting/utils/visualize_with_mujoco.py \
    --config-name=${robot_config} \
    motion_pkl_root=${motion_pkl_root} \
    motion_name=${motion_name}