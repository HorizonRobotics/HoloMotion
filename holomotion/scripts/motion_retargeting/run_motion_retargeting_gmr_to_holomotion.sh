source train.env

dir_name="AMASS_test"
gmr_tgt_dir="data/gmr_retargeted/${dir_name}"
holo_retargeted_dir="data/holomotion_retargeted/processed_datasets/${dir_name}"

robot_cfg="holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml"

preprocess_pipeline="['filename_as_motionkey','legacy_to_ref_keys','slicing','add_padding','tagging']"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/motion_retargeting/gmr_to_holomotion.py \
    io.robot_config=${robot_cfg} \
    io.src_dir=${gmr_tgt_dir} \
    io.out_root=${holo_retargeted_dir} \
    processing.target_fps=50 \
    preprocess.pipeline=${preprocess_pipeline} \
    ray.num_workers=16
