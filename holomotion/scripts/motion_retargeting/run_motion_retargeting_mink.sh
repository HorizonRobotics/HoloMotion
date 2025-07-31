amass_dir="assets/test_data/motion_retargeting/Male2Walking_c3d"
dump_dir="assets/test_data/motion_retargeting/retargeted_datasets/mink_retargeted" # path for testing
# dump_dir="data/retargeted_datasets/g1_21dof_test" # path for training

python holomotion/src/motion_retargeting/mink_fitting.py \
       +dump_dir=${dump_dir} \
       +amass_root=${amass_dir} \
       +num_jobs=12
