source train.env

holo_src_dir="data/holomotion_retargeted/AMASS_test"
holo_tgt_dir="data/holomotion_retargeted/processed_datasets/AMASS_test"

pipeline="['filename_as_motionkey','legacy_to_ref_keys','slicing','add_padding','tagging']"

robot_config="holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/motion_retargeting/holomotion_preprocess.py \
    padding.robot_config_path=${robot_config} \
    io.src_root=${holo_src_dir} \
    io.out_root=${holo_tgt_dir} \
    preprocess.pipeline=${pipeline} \
    ray.enabled=true \
    ray.num_workers=2
