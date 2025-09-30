source train.env

# robot_config="unitree_g1_23dof_retargeting_config"
robot_config="ZJ-Humanoid_hi2_retargeting_config"

${Train_CONDA_PREFIX}/bin/python holomotion/src/motion_retargeting/robot_smpl_shape_fitting.py \
    --config-name=${robot_config}