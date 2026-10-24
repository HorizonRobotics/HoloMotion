"""Load the shared physical task for training and saved-policy evaluation."""

import torch

from holomotion.src.env.reference_tracking.kinematics import G1Kinematics
from holomotion.src.env.reference_tracking.mdp import MDP
from holomotion.src.env.reference_tracking.model import build_model
from holomotion.src.env.reference_tracking.motion import MotionLib
from holomotion.src.env.reference_tracking.robot import default_joint_pos_list


def load_task(*, robot_urdf, manifest, output, device, profile):
    model = build_model(
        robot_urdf, output / "g1.xml", self_collisions=profile == "paper"
    )
    kinematics = G1Kinematics(device, model)
    motion = MotionLib.from_hdf5(manifest, kinematics, robot_urdf, device)
    mdp = MDP(
        kinematics,
        motion,
        torch.tensor(default_joint_pos_list(), device=device),
        reference_orientation=profile == "vibed",
    )
    return model, motion, mdp
