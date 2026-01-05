source train.env

dataset_root="data/holomotion_retargeted/processed_datasets/AMASS_test"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/motion_retargeting/kinematic_filter.py \
    io.dataset_root=${dataset_root}