import joblib
import os
from tqdm import tqdm


def remove_motion_clip_root_trans_xy(src_pkl_path, dst_dir):
    pkl_name = os.path.basename(src_pkl_path)
    motion = joblib.load(src_pkl_path)
    motion_key = list(motion.keys())[0]
    motion_data = motion[motion_key]
    motion_data["root_trans_offset"][:, :2] = 0
    motion[motion_key] = motion_data
    joblib.dump(motion, os.path.join(dst_dir, pkl_name))


if __name__ == "__main__":
    retargeted_pkl_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/robodance100"
    dst_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/robodance100_no_global_translation"
    os.makedirs(dst_dir, exist_ok=True)
    for pkl_name in tqdm(os.listdir(retargeted_pkl_dir)):
        if pkl_name.endswith(".pkl"):
            src_pkl_path = os.path.join(retargeted_pkl_dir, pkl_name)
            remove_motion_clip_root_trans_xy(src_pkl_path, dst_dir)
