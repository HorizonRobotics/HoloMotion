source train.env
export CUDA_VISIBLE_DEVICES=""

holomotion_retargeted_dirs='["data/holomotion_retargeted/processed_datasets/AMASS_test"]'
hdf5_root="data/hdf5_datasets/processed_datasets/h5_AMASS_test"

robot_config="unitree/G1/29dof/29dof_training_isaaclab"
${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/motion_retargeting/pack_hdf5.py \
    robot=$robot_config \
    +holomotion_retargeted_dirs=${holomotion_retargeted_dirs} \
    +hdf5_root=$hdf5_root
