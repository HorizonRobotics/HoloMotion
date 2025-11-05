source train.env

bvh_src_dir="data/lafan1_bvh"
gmr_tgt_dir="data/gmr_retargeted/lafan1/"

# Step 1: retargeting to robot dataset from smplx format
# create gmr_tgt_dir if not exists
if [ ! -d "$gmr_tgt_dir" ]; then
    mkdir -p $gmr_tgt_dir
fi

$Train_CONDA_PREFIX/bin/python \
    thirdparties/GMR/scripts/bvh_to_robot_dataset.py \
    --src_folder ${bvh_src_dir}/ \
    --tgt_folder ${gmr_tgt_dir}/ \
    --robot unitree_g1
