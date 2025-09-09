source train.env

robot_config="zjrx_hi2_retargeting_config"

${Train_CONDA_PREFIX}/bin/python holomotion/src/motion_retargeting/robot_smpl_shape_fitting.py \
    --config-name=${robot_config}