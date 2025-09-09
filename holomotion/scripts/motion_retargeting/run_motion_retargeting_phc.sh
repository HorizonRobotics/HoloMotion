source train.env

amass_dir="assets/test_data/motion_retargeting/Male2Walking_c3d"
dump_dir="assets/test_data/motion_retargeting/retargeted_datasets/phc_retargeted" # path for testing

robot_config="ZJ-Humanoid_hi2_retargeting_config"

${Train_CONDA_PREFIX}/bin/python holomotion/src/motion_retargeting/phc_fitting.py \
       --config-name=${robot_config} \
       +dump_dir=${dump_dir} \
       +amass_root=${amass_dir} \
       +num_jobs=12
