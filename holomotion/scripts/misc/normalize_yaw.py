from ast import dump
import os
import sys

import joblib
import numpy as np
import torch
from holomotion.src.motion_retargeting.utils.torch_humanoid_batch import (
    HumanoidBatch,
)
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import argparse
from loguru import logger


def extract_yaw_only_quaternion(quaternion):
    # Convert from [w, x, y, z] to [x, y, z, w] for SciPy
    quat_scipy = quaternion

    # Convert quaternion to rotation object
    rot = Rotation.from_quat(quat_scipy)

    # Get Euler angles in XYZ order
    euler_angles = rot.as_euler("xyz", degrees=False)

    # Keep only yaw (rotation around Z axis), zero out roll and pitch
    yaw_only_euler = np.array([0.0, 0.0, euler_angles[2]])

    # Convert back to quaternion
    yaw_only_rot = Rotation.from_euler("xyz", yaw_only_euler, degrees=False)
    yaw_only_quat_scipy = (
        yaw_only_rot.as_quat()
    )  # Returns in [x, y, z, w] format

    # Convert back to [w, x, y, z] format
    yaw_only_quat = np.array(
        [
            yaw_only_quat_scipy[0],  # x
            yaw_only_quat_scipy[1],  # y
            yaw_only_quat_scipy[2],  # z
            yaw_only_quat_scipy[3],  # w
        ]
    )

    return yaw_only_quat


def normalize_yaw_and_translation(
    root_rotations,
    root_translations,
    target_init_yaw=0.0,
):
    """Normalize the motion yaw.

    so that the first frame's yaw is set to target_init_yaw and translation
    starts at (0, 0, original_z).
    Only normalizes X and Y translation to 0, keeping the original Z (height)
    value.
    Adjusts both root rotations and root translations accordingly.

    Parameters:
    -----------
    root_rotations : numpy.ndarray
        Root rotations in quaternion format [x, y, z, w], shape (N, 4)
    root_translations : numpy.ndarray
        Root translations, shape (N, 3)
    target_init_yaw : float
        Target yaw angle in radians for the first frame (default: 0.0)

    Returns:
    --------
    tuple
        (normalized_root_rotations, normalized_root_translations)
    """
    # Step 1: Normalize yaw to start at target_init_yaw
    first_frame_quat = root_rotations[0]

    # Extract yaw from the first frame (returns [x, y, z, w] format)
    first_frame_yaw_quat = extract_yaw_only_quaternion(first_frame_quat)

    # Create target yaw quaternion
    target_yaw_rot = Rotation.from_euler("z", target_init_yaw, degrees=False)
    target_yaw_quat = target_yaw_rot.as_quat()  # [x, y, z, w] format

    # Create the transformation rotation: target_yaw * inverse(first_frame_yaw)
    first_yaw_rot = Rotation.from_quat(first_frame_yaw_quat)
    target_yaw_rot_obj = Rotation.from_quat(target_yaw_quat)
    yaw_transformation = target_yaw_rot_obj * first_yaw_rot.inv()

    # Apply yaw transformation to all root rotations
    yaw_normalized_rotations = np.zeros_like(root_rotations)
    for i in range(len(root_rotations)):
        # Current rotation is already in [x, y, z, w] format
        current_rot = Rotation.from_quat(root_rotations[i])

        # Apply yaw transformation
        normalized_rot = yaw_transformation * current_rot
        yaw_normalized_rotations[i] = (
            normalized_rot.as_quat()
        )  # Returns [x, y, z, w]

    # Apply yaw transformation to root translations
    yaw_normalized_translations = np.zeros_like(root_translations)
    for i in range(len(root_translations)):
        # Apply yaw transformation to the translation
        rotated_trans = yaw_transformation.apply(root_translations[i])
        yaw_normalized_translations[i] = rotated_trans

    # Step 2: Normalize translation to start at (0, 0, original_z)
    first_frame_translation = yaw_normalized_translations[0].copy()

    # Subtract the first frame's XY translation from all frames, but keep original Z
    translation_normalized = yaw_normalized_translations.copy()
    translation_normalized[:, 0] -= first_frame_translation[
        0
    ]  # Normalize X to 0
    translation_normalized[:, 1] -= first_frame_translation[
        1
    ]  # Normalize Y to 0

    return yaw_normalized_rotations, translation_normalized


def normalize_motion(motion_dict: dict, target_init_yaw=0.0):
    root_rotations = motion_dict["root_rot"]
    root_translations = motion_dict["root_trans_offset"]
    normalized_root_rotations, normalized_root_translations = (
        normalize_yaw_and_translation(
            root_rotations, root_translations, target_init_yaw
        )
    )
    motion_dict["root_rot"] = normalized_root_rotations
    motion_dict["root_trans_offset"] = normalized_root_translations
    return motion_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", type=str, required=True)
    parser.add_argument("--target_init_yaw", type=float, default=0.0)
    args = parser.parse_args()
    motion_dict = joblib.load(args.motion_path)
    raw_motion_name = list(motion_dict.keys())[0]
    normalized_motion_dict = normalize_motion(
        motion_dict[raw_motion_name], args.target_init_yaw
    )

    # give normalized motion a new name
    normalized_motion_name = f"{raw_motion_name}_normalized"
    normalized_motion_dict = {normalized_motion_name: normalized_motion_dict}

    dump_path = os.path.join(
        os.path.dirname(args.motion_path), f"{normalized_motion_name}.pkl"
    )
    joblib.dump(
        normalized_motion_dict,
        dump_path,
    )
    logger.info(f"Normalized motion saved to {dump_path}")
