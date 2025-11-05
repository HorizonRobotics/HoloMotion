source train.env

npz_dir="logs/zhangbo/dump_ckpt_20251104/isaaclab_eval_output_model_100000"

${Train_CONDA_PREFIX}/bin/python \
    holomotion/src/evaluation/metrics.py \
    --npz_dir=${npz_dir} \
    --failure_pos_err_thresh_m=0.25
