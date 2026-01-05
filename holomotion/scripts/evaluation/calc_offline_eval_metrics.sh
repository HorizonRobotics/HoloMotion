source train.env


npz_dir="logs/HoloMotionMotionTracking/20260104_150135-train_g1_29dof_motion_tracking/isaaclab_eval_output_model_3000_h5_AMASS_test"
dataset_suffix="AMASS_test"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/evaluation/metrics.py \
    --npz_dir=${npz_dir} \
    --dataset_suffix=${dataset_suffix} \
    --failure_pos_err_thresh_m=0.25
