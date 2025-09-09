import joblib
from loguru import logger
import os


def reverse_motion(motion_data_dict):
    """
    Assume the "pose_aa" "root_rot" "root_trans_offset" are all in the same order as the original motion.
    """
    rev_motion_data_dict = {}
    rev_motion_data_dict["pose_aa"] = motion_data_dict["pose_aa"][::-1]
    rev_motion_data_dict["root_rot"] = motion_data_dict["root_rot"][::-1]
    rev_motion_data_dict["root_trans_offset"] = motion_data_dict[
        "root_trans_offset"
    ][::-1]
    rev_motion_data_dict["dof"] = motion_data_dict["dof"][::-1]
    rev_motion_data_dict["fps"] = motion_data_dict["fps"]
    return rev_motion_data_dict


def main():
    # rtg_pkl_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_pkls/0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded_normalized.pkl"
    rtg_pkl_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/full_amass_23dof_lockwrist_asap/0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949.pkl"
    target_dump_dir = "data/retargeted_datasets/20250825_chengdu_demo_pkls"
    os.makedirs(target_dump_dir, exist_ok=True)
    motion = joblib.load(rtg_pkl_path)
    motion_name = list(motion.keys())[0]
    rev_motion_key = f"{motion_name}_rev"
    rev_motion_data = reverse_motion(motion[motion_name])
    joblib.dump(
        {rev_motion_key: rev_motion_data},
        os.path.join(target_dump_dir, f"{rev_motion_key}.pkl"),
    )
    logger.info(
        f"Reversed motion saved to {os.path.join(target_dump_dir, f'{rev_motion_key}.pkl')}"
    )


if __name__ == "__main__":
    main()
