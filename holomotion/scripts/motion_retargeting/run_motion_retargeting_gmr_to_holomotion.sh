source train.env

gmr_tgt_dir="data/gmr_retargeted/AMASS_test"
holo_retargeted_dir="data/holomotion_retargeted/AMASS_test"

$Train_CONDA_PREFIX/bin/python \
    holomotion/src/motion_retargeting/gmr_to_holomotion.py \
    --src_dir=${gmr_tgt_dir} \
    --out_root=${holo_retargeted_dir} \
    --ref_dir="holomotion/src/motion_retargeting/utils" \
    --num_workers=1 \
    --robot_config="holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"