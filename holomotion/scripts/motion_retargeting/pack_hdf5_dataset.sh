source train.env
export CUDA_VISIBLE_DEVICES=""


holomotion_npz_root="data/holomotion_retargeted/AMASS_test"
hdf5_root="data/hdf5_datasets/h5_g1_29dof_amass_test"

robot_config="unitree/G1/29dof/29dof_training_isaaclab"
${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/motion_retargeting/pack_hdf5.py \
    robot=$robot_config \
    +precomputed_npz_root=${holomotion_npz_root} \
    +hdf5_root=$hdf5_root \
    chunks_t=1024 \
    compression=lzf \
    shard_target_gb=2.0 \
    num_jobs=16 \
    debug_local_mode=false
