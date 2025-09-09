import joblib
import os


def slice_motion(motion_path, s_time, e_time):
    motion = joblib.load(motion_path)
    motion_name = list(motion.keys())[0]
    motion = motion[motion_name]
    fps = motion["fps"]
    s_frame = int(s_time * fps)
    e_frame = int(e_time * fps)
    n_frames = motion["root_trans_offset"].shape[0]
    print(
        f"Number of frames: {n_frames}, start frame: {s_frame}, end frame: {e_frame}"
    )
    slice = (s_frame, e_frame)
    sliced_motion = {}
    for key, value in motion.items():
        if key != "fps":
            sliced_motion[key] = value[slice[0] : slice[1]]
        else:
            sliced_motion[key] = value
    sliced_motion = {
        f"{motion_name}_sliced-{s_frame}-{e_frame}": sliced_motion
    }
    # save sliced motion
    motion_root_dir = os.path.dirname(motion_path)
    sliced_motion_path = os.path.join(
        motion_root_dir, f"{motion_name}_sliced-{s_frame}-{e_frame}.pkl"
    )
    with open(sliced_motion_path, "wb") as f:
        joblib.dump(sliced_motion, f)
        print(f"Sliced motion saved to {sliced_motion_path}")


if __name__ == "__main__":
    # motion_path = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/0630_demo_selected/0-CMU_94_94_16_stageii.pkl"
    # motion_path = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/0630_demo_selected/0-SFU_0008_0008_ChaCha001_stageii_slowdown.pkl"
    # motion_path = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/0630_demo_selected/0-SFU_0008_0008_ChaCha001_stageii_zero_vel.pkl"

    # motion_path = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/lafan1_23dof/dance1_subject2.pkl"
    # s_time = 3.0
    # e_time = 20.5

    # motion_path = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/full_amass_23dof_lockwrist_asap/0-SFU_0008_0008_ChaCha001_stageii.pkl"
    # s_time = 0.0
    # e_time = 22.0

    # motion_path = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/full_amass_23dof_lockwrist_asap/0-Transitions_mazen_c3d_dance_stand_stageii.pkl"
    # s_time = 0.0
    # e_time = 6.5

    # motion_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance/dance2_subject3.pkl"
    # s_time = 0.0
    # e_time = 66.0

    # motion_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/full_amass_23dof_lockwrist_asap/0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii.pkl"
    # s_time = 13.0
    # e_time = 31.66

    # motion_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_salsa_shines_phc_0825/0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded_normalized.pkl"
    # s_time = 0.0
    # e_time = 0.5

    # motion_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance/dance1_subject2.pkl"
    # s_time=39.0
    # e_time=100.0

    # motion_path="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance/dance2_subject3.pkl"
    # s_time = 0.0
    # e_time = 66.0

    # motion_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/full_amass_23dof_lockwrist_asap/0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii.pkl"
    # s_time = 4.3
    # e_time = 56.0

    motion_path = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250826_chengdu_demo_train_v3/dance1_subject2_sliced-1170-3000_padded_normalized.pkl"
    s_time = 2.0
    e_time = 27.0

    slice_motion(motion_path, s_time, e_time)
