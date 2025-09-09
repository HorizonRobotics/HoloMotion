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

DEFAULT_JOINT_ANGLES = [
    # left lower body
    -0.25,
    0.0,
    0.0,
    0.5,
    -0.25,
    0.0,
    # right lower body
    -0.25,
    0.0,
    0.0,
    0.5,
    -0.25,
    0.0,
    # waist
    0.0,
    0.0,
    0.0,
    # left upper body
    0.3,
    0.6,
    0.0,
    1.0,
    # right upper body
    0.3,
    -0.6,
    0.0,
    1.0,
]


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


def dof_to_pose_aa(
    dof_pos,
    robot_config_path,
    root_rot=None,
):
    # Add project root to path if needed
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Load robot configuration
    # robot_cfg = OmegaConf.create({"robot": OmegaConf.load(robot_config_path)})
    robot_cfg = OmegaConf.load(robot_config_path)

    # Initialize the forward kinematics model
    # import ipdb; ipdb.set_trace()
    humanoid_fk = HumanoidBatch(robot_cfg.robot)
    num_augment_joint = len(robot_cfg.robot.extend_config)

    # Convert input to torch tensor if needed
    if not isinstance(dof_pos, torch.Tensor):
        dof_pos = torch.tensor(dof_pos, dtype=torch.float32)

    # Ensure correct shape
    if dof_pos.dim() == 3 and dof_pos.shape[2] == 1:
        dof_pos = dof_pos.squeeze(-1)  # Remove last dimension if it's 1

    N = dof_pos.shape[0]

    # Handle root rotation
    if root_rot is None:
        # If no root rotation provided, use identity quaternion
        root_rot_aa = torch.zeros((N, 3))
    else:
        # Convert input to torch tensor if needed
        if not isinstance(root_rot, torch.Tensor):
            root_rot = torch.tensor(root_rot, dtype=torch.float32)

        # Check if root_rot is quaternion (shape [N, 4])
        if root_rot.shape[-1] == 4:
            # Convert quaternion to axis-angle
            root_rot_aa = quaternion_to_axis_angle(root_rot)
        elif root_rot.shape[-1] == 3:
            # Already in axis-angle format
            root_rot_aa = root_rot
        else:
            raise ValueError(
                f"Root rotation should have shape [N, 4] for quaternion or [N, 3] for axis-angle, got {root_rot.shape}"
            )

    # Compute joint axis-angle representations
    joint_aa = humanoid_fk.dof_axis * dof_pos.unsqueeze(-1)  # [N, num_dof, 3]

    # Combine root rotation and joint rotations
    pose_aa = torch.cat(
        [
            root_rot_aa.unsqueeze(1),  # Root rotation [N, 1, 3]
            joint_aa,  # Joint rotations [N, num_dof, 3]
            torch.zeros(
                (N, num_augment_joint, 3)
            ),  # Augmented joints [N, num_augment_joint, 3]
        ],
        dim=1,
    )

    # Convert back to numpy for output
    return pose_aa.numpy()


def quaternion_to_axis_angle(quaternions):
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)

    # Extract the components
    x, y, z, w = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )

    # Compute the rotation angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    # Compute the axis
    # Handle the case where angle is close to zero
    sin_half_angle = torch.sqrt(1.0 - w * w)
    mask = sin_half_angle < 1e-6

    # Safe division
    sin_half_angle = torch.clamp(sin_half_angle, min=1e-6)
    axis_x = torch.where(mask, x, x / sin_half_angle)
    axis_y = torch.where(mask, y, y / sin_half_angle)
    axis_z = torch.where(mask, z, z / sin_half_angle)

    # Combine angle and axis
    axis_angles = torch.stack(
        [axis_x * angle, axis_y * angle, axis_z * angle], dim=-1
    )

    return axis_angles


def interpolate_to_default_pose(start_dof_pos, end_dof_pos, transition_frames):
    """
    Interpolate the dof positions to the default pose.
    """
    dof_pos_diff = end_dof_pos - start_dof_pos
    delta_dof_pos = dof_pos_diff / transition_frames
    dof_pos = (
        start_dof_pos + delta_dof_pos * np.arange(transition_frames)[:, None]
    )
    return dof_pos


def interpolate_quaternions(start_quat, end_quat, steps):
    """
    Interpolate between two quaternions using SciPy's Slerp.

    Parameters:
    -----------
    start_quat : numpy.ndarray
        Starting quaternion in [w, x, y, z] format, shape (4,)
    end_quat : numpy.ndarray
        Ending quaternion in [w, x, y, z] format, shape (4,)
    steps : int
        Number of interpolation steps (including start and end points)

    Returns:
    --------
    numpy.ndarray
        Interpolated quaternions in [w, x, y, z] format, shape (steps, 4)
    """
    if steps < 2:
        raise ValueError(
            "Steps must be at least 2 to include start and end points"
        )

    # Convert from [w, x, y, z] to [x, y, z, w] for SciPy
    start_scipy = np.array(
        [start_quat[1], start_quat[2], start_quat[3], start_quat[0]]
    )
    end_scipy = np.array([end_quat[1], end_quat[2], end_quat[3], end_quat[0]])

    # Create rotation objects
    rotations = Rotation.from_quat([start_scipy, end_scipy])

    # Create a slerp object
    times = [0, 1]  # Normalized time points
    slerp = Slerp(times, rotations)

    # Generate interpolation points
    interp_times = np.linspace(0, 1, steps)
    interp_rots = slerp(interp_times)

    # Get quaternions in scipy format [x, y, z, w]
    interp_quats_scipy = interp_rots.as_quat()

    # Convert back to [w, x, y, z] format
    interp_quats = np.column_stack(
        (
            interp_quats_scipy[:, 3],  # w
            interp_quats_scipy[:, 0],  # x
            interp_quats_scipy[:, 1],  # y
            interp_quats_scipy[:, 2],  # z
        )
    )

    return interp_quats


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


def interpolate_root_trans(
    start_root_trans, end_root_trans, transition_frames
):
    root_trans_diff = end_root_trans - start_root_trans
    delta_root_trans = root_trans_diff / transition_frames
    root_trans = (
        start_root_trans
        + delta_root_trans * np.arange(transition_frames)[:, None]
    )
    return root_trans


def smooth_root_translations(
    root_translations: np.ndarray, sigma: float = 2.0
) -> np.ndarray:
    """
    Apply Gaussian filtering to root translations for smoothing.

    Parameters:
    -----------
    root_translations : np.ndarray
        Root translations array of shape (N, 3)
    sigma : float
        Standard deviation for Gaussian kernel (higher = more smoothing)

    Returns:
    --------
    np.ndarray
        Smoothed root translations
    """
    smoothed = np.zeros_like(root_translations)
    for i in range(3):  # Smooth each axis separately
        smoothed[:, i] = gaussian_filter1d(
            root_translations[:, i], sigma=sigma
        )
    return smoothed


def smooth_root_rotations(
    root_rotations: np.ndarray, sigma: float = 2.0
) -> np.ndarray:
    """
    Apply Gaussian filtering to root rotations for smoothing.
    Uses quaternion interpolation to maintain valid rotations.

    Parameters:
    -----------
    root_rotations : np.ndarray
        Root rotations array of shape (N, 4) in quaternion format [x, y, z, w]
    sigma : float
        Standard deviation for Gaussian kernel (higher = more smoothing)

    Returns:
    --------
    np.ndarray
        Smoothed root rotations
    """
    # Convert to Euler angles for smoothing
    euler_angles = np.zeros((len(root_rotations), 3))
    for i, quat in enumerate(root_rotations):
        rot = Rotation.from_quat(quat)
        euler_angles[i] = rot.as_euler("xyz", degrees=False)

    # Handle angle wrapping for continuous smoothing
    # Unwrap angles to avoid discontinuities at ±π
    for axis in range(3):
        euler_angles[:, axis] = np.unwrap(euler_angles[:, axis])

    # Apply Gaussian smoothing to each Euler angle
    smoothed_euler = np.zeros_like(euler_angles)
    for i in range(3):
        smoothed_euler[:, i] = gaussian_filter1d(
            euler_angles[:, i], sigma=sigma
        )

    # Convert back to quaternions
    smoothed_rotations = np.zeros_like(root_rotations)
    for i, euler in enumerate(smoothed_euler):
        rot = Rotation.from_euler("xyz", euler, degrees=False)
        smoothed_rotations[i] = rot.as_quat()  # [x, y, z, w]

    return smoothed_rotations


def normalize_yaw_and_translation(
    root_rotations, root_translations, target_init_yaw=0.0
):
    """
    Normalize the motion so that the first frame's yaw is set to target_init_yaw and translation starts at (0, 0, original_z).
    Only normalizes X and Y translation to 0, keeping the original Z (height) value.
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
    # Keep Z unchanged (don't subtract first_frame_translation[2])

    return yaw_normalized_rotations, translation_normalized


def main(
    motion_path,
    dump_path,
    robot_config_path,
    target_init_yaw=0.0,
    add_padding=True,
    normalize_yaw_trans=True,
):
    motion = joblib.load(motion_path)

    padded_motions = {}

    for motion_name in tqdm(motion.keys()):
        single_motion = motion[motion_name]
        dof_positions = single_motion["dof"]  # Shape: [Timestep, num_dof]
        root_rotations = single_motion["root_rot"]  # Shape: [Timestep, 4]
        root_translations = single_motion[
            "root_trans_offset"
        ]  # Shape: [Timestep, 3]

        # Step 1: Conditionally normalize yaw to start at target_init_yaw and XY translation to start at (0, 0)
        if normalize_yaw_trans:
            print(
                f"Normalizing yaw to {target_init_yaw:.2f} rad ({np.degrees(target_init_yaw):.1f}°) and translation for motion: {motion_name}"
            )
            normalized_root_rotations, normalized_root_translations = (
                normalize_yaw_and_translation(
                    root_rotations, root_translations, target_init_yaw
                )
            )

            # Update the motion data with normalized values
            root_rotations = normalized_root_rotations
            root_translations = normalized_root_translations
        else:
            print(
                f"Skipping yaw and translation normalization for motion: {motion_name}"
            )

        # Step 2: Conditionally add padding
        if add_padding:
            print(f"Adding padding to motion: {motion_name}")

            start_root_rot = root_rotations[0:1, ...]
            end_root_rot = root_rotations[-1:, ...]

            default_dof_pos = np.array([DEFAULT_JOINT_ANGLES])

            stand_still_time = 1.0
            transition_time = 1.5
            stand_still_frames = int(stand_still_time * single_motion["fps"])
            transition_frames = int(transition_time * single_motion["fps"])
            first_frame_dof_pos = dof_positions[0:1, ...]
            last_frame_dof_pos = dof_positions[-1:, ...]

            start_stand_still_dof_pos = default_dof_pos.repeat(
                stand_still_frames, axis=0
            )
            end_stand_still_dof_pos = default_dof_pos.repeat(
                stand_still_frames, axis=0
            )

            start_transition_dof_pos = interpolate_to_default_pose(
                start_dof_pos=default_dof_pos,
                end_dof_pos=first_frame_dof_pos,
                transition_frames=transition_frames,
            )
            end_transition_dof_pos = interpolate_to_default_pose(
                start_dof_pos=last_frame_dof_pos,
                end_dof_pos=default_dof_pos,
                transition_frames=transition_frames,
            )

            start_root_rot_2 = root_rotations[0, ...]
            # Create target yaw quaternion for padding sections
            target_yaw_rot = Rotation.from_euler(
                "z", target_init_yaw, degrees=False
            )
            start_root_rot_0 = target_yaw_rot.as_quat()  # [x, y, z, w] format
            start_root_rot_1 = interpolate_quaternions(
                start_root_rot_0, start_root_rot_2, transition_frames
            )
            start_root_rot = np.concatenate(
                [
                    start_root_rot_0[None, :].repeat(
                        stand_still_frames, axis=0
                    ),
                    start_root_rot_1,
                ],
                axis=0,
            )

            end_root_rot_0 = root_rotations[-1, ...]
            end_root_rot_2 = target_yaw_rot.as_quat()  # [x, y, z, w] format
            end_root_rot_1 = interpolate_quaternions(
                end_root_rot_0, end_root_rot_2, transition_frames
            )
            end_root_rot = np.concatenate(
                [
                    end_root_rot_1,
                    end_root_rot_2[None, :].repeat(stand_still_frames, axis=0),
                ],
                axis=0,
            )

            start_root_trans_0 = root_translations[0:1, ...].repeat(
                stand_still_frames, axis=0
            )
            start_root_trans_0[:, 2] = 0.793
            start_root_trans_1 = interpolate_root_trans(
                start_root_trans_0[0:1, ...],
                root_translations[1:2, ...],
                transition_frames,
            )
            start_root_trans = np.concatenate(
                [
                    start_root_trans_0,
                    start_root_trans_1,
                ],
                axis=0,
            )

            end_root_trans_1 = root_translations[-1:, ...].repeat(
                stand_still_frames, axis=0
            )
            end_root_trans_1[:, 2] = 0.793
            end_root_trans_0 = interpolate_root_trans(
                root_translations[-2:-1, ...],
                end_root_trans_1[0:1, ...],
                transition_frames,
            )
            end_root_trans = np.concatenate(
                [end_root_trans_0, end_root_trans_1], axis=0
            )

            entire_motion_dof_pos = np.concatenate(
                [
                    start_stand_still_dof_pos,
                    start_transition_dof_pos,
                    dof_positions,
                    end_transition_dof_pos,
                    end_stand_still_dof_pos,
                ],
                axis=0,
            )

            entire_motion_root_rot = np.concatenate(
                [
                    start_root_rot,
                    root_rotations,
                    end_root_rot,
                ],
                axis=0,
            )

            entire_motion_root_trans = np.concatenate(
                [
                    start_root_trans,
                    root_translations,
                    end_root_trans,
                ],
                axis=0,
            )
        else:
            print(f"Skipping padding for motion: {motion_name}")
            # Use original motion data without padding
            entire_motion_dof_pos = dof_positions
            entire_motion_root_rot = root_rotations
            entire_motion_root_trans = root_translations

        entire_motion_pose_aa = dof_to_pose_aa(
            entire_motion_dof_pos,
            robot_config_path=robot_config_path,
            root_rot=entire_motion_root_rot,
        )

        padded_motion_dict = {}
        padded_motion_dict["pose_aa"] = entire_motion_pose_aa
        padded_motion_dict["dof"] = entire_motion_dof_pos
        padded_motion_dict["root_rot"] = entire_motion_root_rot
        padded_motion_dict["root_trans_offset"] = entire_motion_root_trans
        padded_motion_dict["fps"] = single_motion["fps"]
        # padded_motion_dict["smpl_joints"] = motion["smpl_joints"]

        # Update motion name suffix based on flags
        suffix = ""
        if add_padding and normalize_yaw_trans:
            suffix = "_padded_normalized"
        elif add_padding:
            suffix = "_padded"
        elif normalize_yaw_trans:
            suffix = "_normalized"
        else:
            suffix = "_processed"

        padded_motion_name = f"{motion_name}{suffix}"
        padded_motions[padded_motion_name] = padded_motion_dict

    joblib.dump(
        padded_motions,
        dump_path,
    )

    print(f"Processed motion saved to {dump_path}")


def combine_clips_with_yaw_continuity(
    motion_paths: list,
    dump_path: str,
    robot_config_path: str,
    stand_still_time: float = 1.0,
    transition_time: float = 1.5,
    enable_root_smoothing: bool = True,
    smoothing_sigma: float = 2.0,
    add_padding: bool = True,
    normalize_yaw_trans: bool = True,
):
    """
    Combine multiple motion clips into a single continuous motion with optional yaw and translation continuity and padding.

    This function:
    1. Optionally adds stand still padding to all motions
    2. Optionally records the last step yaw and translation of each clip
    3. Optionally transforms the next clip to start with the same yaw and translation as the previous clip's end
    4. Combines all clips into a single continuous motion
    5. Optionally applies Gaussian smoothing to root translations and rotations

    Parameters:
    -----------
    motion_paths : list
        List of paths to motion files to combine
    dump_path : str
        Path to save the combined motion
    robot_config_path : str
        Path to robot configuration file
    stand_still_time : float
        Duration of stand still padding in seconds
    transition_time : float
        Duration of transition between clips in seconds
    enable_root_smoothing : bool
        Whether to apply Gaussian smoothing to root translations and rotations
    smoothing_sigma : float
        Standard deviation for Gaussian smoothing kernel (higher = more smoothing)
    add_padding : bool
        Whether to add stand still and transition padding to clips
    normalize_yaw_trans : bool
        Whether to normalize yaw and translation for continuity between clips
    """
    combined_motion = {}
    current_yaw = 0.0  # Starting yaw for the first clip
    current_translation_xy = np.array(
        [0.0, 0.0]
    )  # Starting XY translation position

    # Default pose for padding
    default_dof_pos = np.array(DEFAULT_JOINT_ANGLES)

    # Storage for combined motion data
    all_dof_positions = []
    all_root_rotations = []
    all_root_translations = []
    combined_fps = None

    operation_desc = []
    if add_padding:
        operation_desc.append("padding")
    if normalize_yaw_trans:
        operation_desc.append("yaw continuity")
    operation_str = (
        " and ".join(operation_desc) if operation_desc else "basic combination"
    )

    print(
        f"Combining {len(motion_paths)} motion clips with {operation_str}..."
    )

    for i, motion_path in enumerate(
        tqdm(motion_paths, desc="Processing clips")
    ):
        print(
            f"\nProcessing clip {i + 1}/{len(motion_paths)}: {os.path.basename(motion_path)}"
        )

        # Load motion data
        motion = joblib.load(motion_path)
        motion_name = list(motion.keys())[0]
        single_motion = motion[motion_name]

        # Extract motion data
        dof_positions = single_motion["dof"]
        root_rotations = single_motion["root_rot"]
        root_translations = single_motion["root_trans_offset"]
        fps = single_motion["fps"]

        if combined_fps is None:
            combined_fps = fps
        elif combined_fps != fps:
            print(f"Warning: FPS mismatch. Expected {combined_fps}, got {fps}")

        # Step 1: Conditionally normalize this clip to start with the current target yaw and translation
        if normalize_yaw_trans:
            print(
                f"  Normalizing clip to start with yaw: {current_yaw:.2f} rad ({np.degrees(current_yaw):.1f}°)"
            )
            print(
                f"  Target XY translation: [{current_translation_xy[0]:.3f}, {current_translation_xy[1]:.3f}]"
            )
            normalized_root_rotations, normalized_root_translations = (
                normalize_yaw_and_translation(
                    root_rotations,
                    root_translations,
                    target_init_yaw=current_yaw,
                )
            )

            # Additional step: align XY translation to continue from previous clip's end position
            if (
                i > 0
            ):  # Skip for first clip as it's already normalized to (0,0,z)
                # Get the XY translation offset needed to align with target position
                first_frame_translation_xy = normalized_root_translations[
                    0, :2
                ]  # Only X, Y
                translation_offset_xy = (
                    current_translation_xy - first_frame_translation_xy
                )

                # Apply XY translation offset to all frames, keeping original Z values
                normalized_root_translations[:, :2] = (
                    normalized_root_translations[:, :2]
                    + translation_offset_xy[None, :]
                )

                print(
                    f"  Applied XY translation offset: [{translation_offset_xy[0]:.3f}, {translation_offset_xy[1]:.3f}]"
                )
        else:
            print(f"  Skipping yaw and translation normalization for clip")
            # Use original data without normalization
            normalized_root_rotations = root_rotations
            normalized_root_translations = root_translations

        # Step 2: Conditionally add padding for this clip
        if add_padding:
            print(f"  Adding padding to clip")

            stand_still_frames = int(stand_still_time * fps)
            transition_frames = int(transition_time * fps)

            # Create stand still and transition sequences
            start_stand_still_dof_pos = default_dof_pos[None, :].repeat(
                stand_still_frames, axis=0
            )
            end_stand_still_dof_pos = default_dof_pos[None, :].repeat(
                stand_still_frames, axis=0
            )

            # Transition from default pose to first frame
            start_transition_dof_pos = interpolate_to_default_pose(
                start_dof_pos=default_dof_pos[None, :],
                end_dof_pos=dof_positions[0:1, :],
                transition_frames=transition_frames,
            )

            # Transition from last frame to default pose
            end_transition_dof_pos = interpolate_to_default_pose(
                start_dof_pos=dof_positions[-1:, :],
                end_dof_pos=default_dof_pos[None, :],
                transition_frames=transition_frames,
            )

            # Create root rotation sequences
            # Start padding uses only yaw rotation (zero pitch and roll for upright standing)
            target_yaw_rot = Rotation.from_euler(
                "z", current_yaw, degrees=False
            )
            start_root_rot_target = (
                target_yaw_rot.as_quat()
            )  # [x, y, z, w] format

            print(
                f"  Start padding will use yaw: {current_yaw:.3f} rad ({np.degrees(current_yaw):.1f}°) with zero pitch/roll"
            )

            # Stand still at target yaw with zero pitch and roll (upright standing)
            start_stand_still_rot = start_root_rot_target[None, :].repeat(
                stand_still_frames, axis=0
            )

            # Transition from target yaw to first frame rotation
            start_transition_rot = interpolate_quaternions(
                start_root_rot_target,
                normalized_root_rotations[0],
                transition_frames,
            )

            # For end padding, extract final yaw but use zero pitch and roll for upright standing
            final_motion_rot = normalized_root_rotations[-1]
            final_motion_rot_obj = Rotation.from_quat(final_motion_rot)
            final_motion_euler = final_motion_rot_obj.as_euler(
                "xyz", degrees=False
            )
            final_yaw = final_motion_euler[2]

            # Create end padding rotation with final yaw but zero pitch and roll for upright standing
            end_padding_rot = Rotation.from_euler(
                "xyz", [0.0, 0.0, final_yaw], degrees=False
            )
            end_padding_rot_quat = (
                end_padding_rot.as_quat()
            )  # [x, y, z, w] format

            print(
                f"  End padding will use yaw: {final_yaw:.3f} rad ({np.degrees(final_yaw):.1f}°) with zero pitch/roll"
            )

            # Transition from final motion rotation to upright pose with same yaw
            end_transition_rot = interpolate_quaternions(
                final_motion_rot, end_padding_rot_quat, transition_frames
            )

            # Stand still at the final yaw with zero pitch and roll (upright standing)
            end_stand_still_rot = end_padding_rot_quat[None, :].repeat(
                stand_still_frames, axis=0
            )

            # Create root translation sequences
            start_trans_height = 0.793  # Default standing height

            # Stand still at target position (from previous clip's end or initial position)
            if i == 0:
                # For first clip, use the normalized starting position
                start_stand_still_trans = normalized_root_translations[
                    0:1, :
                ].repeat(stand_still_frames, axis=0)
                start_stand_still_trans[:, 2] = start_trans_height
            else:
                # For subsequent clips, use the target XY translation from previous clip
                start_stand_still_trans = normalized_root_translations[
                    0:1, :
                ].repeat(stand_still_frames, axis=0)
                start_stand_still_trans[:, :2] = current_translation_xy[
                    None, :
                ]  # Set XY from target
                start_stand_still_trans[:, 2] = (
                    start_trans_height  # Set Z to standard height
                )

            # Transition from stand still to first frame
            start_transition_trans = interpolate_root_trans(
                start_stand_still_trans[0:1, :],
                normalized_root_translations[0:1, :],
                transition_frames,
            )

            # Transition from last frame to stand still
            end_stand_still_trans_target = normalized_root_translations[
                -1:, :
            ].copy()
            end_stand_still_trans_target[:, 2] = start_trans_height

            end_transition_trans = interpolate_root_trans(
                normalized_root_translations[-1:, :],
                end_stand_still_trans_target,
                transition_frames,
            )

            # Stand still at the end
            end_stand_still_trans = end_stand_still_trans_target.repeat(
                stand_still_frames, axis=0
            )

            # Combine all sequences for this clip
            clip_dof_positions = np.concatenate(
                [
                    start_stand_still_dof_pos,
                    start_transition_dof_pos,
                    dof_positions,
                    end_transition_dof_pos,
                    end_stand_still_dof_pos,
                ],
                axis=0,
            )

            clip_root_rotations = np.concatenate(
                [
                    start_stand_still_rot,
                    start_transition_rot,
                    normalized_root_rotations,
                    end_transition_rot,
                    end_stand_still_rot,
                ],
                axis=0,
            )

            clip_root_translations = np.concatenate(
                [
                    start_stand_still_trans,
                    start_transition_trans,
                    normalized_root_translations,
                    end_transition_trans,
                    end_stand_still_trans,
                ],
                axis=0,
            )
        else:
            print(f"  Skipping padding for clip")
            # Use the motion data without padding
            clip_dof_positions = dof_positions
            clip_root_rotations = normalized_root_rotations
            clip_root_translations = normalized_root_translations

        # Add to combined motion
        all_dof_positions.append(clip_dof_positions)
        all_root_rotations.append(clip_root_rotations)
        all_root_translations.append(clip_root_translations)

        # Step 3: Update continuity info for next clip (only if normalization is enabled)
        if normalize_yaw_trans:
            # Extract final yaw and translation for next clip continuity
            # Extract yaw from the normalized (final transformed) rotation
            final_rotation = normalized_root_rotations[-1]
            final_rot_obj = Rotation.from_quat(final_rotation)
            final_euler = final_rot_obj.as_euler("xyz", degrees=False)

            # Also check the original final yaw and the transformation applied
            original_final_rotation = root_rotations[-1]
            original_final_rot_obj = Rotation.from_quat(
                original_final_rotation
            )
            original_final_euler = original_final_rot_obj.as_euler(
                "xyz", degrees=False
            )

            # Calculate the yaw change within this clip
            original_first_rotation = root_rotations[0]
            original_first_rot_obj = Rotation.from_quat(
                original_first_rotation
            )
            original_first_euler = original_first_rot_obj.as_euler(
                "xyz", degrees=False
            )

            # Handle angle wrapping for yaw change calculation
            yaw_change_in_clip = (
                original_final_euler[2] - original_first_euler[2]
            )

            # Wrap yaw change to [-π, π] to handle angle wrapping
            while yaw_change_in_clip > np.pi:
                yaw_change_in_clip -= 2 * np.pi
            while yaw_change_in_clip < -np.pi:
                yaw_change_in_clip += 2 * np.pi

            # The final yaw should be the target initial yaw plus the yaw change in the clip
            expected_final_yaw = current_yaw + yaw_change_in_clip
            actual_final_yaw = final_euler[2]

            print(
                f"  Original first yaw: {original_first_euler[2]:.3f} rad ({np.degrees(original_first_euler[2]):.1f}°)"
            )
            print(
                f"  Original final yaw: {original_final_euler[2]:.3f} rad ({np.degrees(original_final_euler[2]):.1f}°)"
            )
            print(
                f"  Yaw change in clip: {yaw_change_in_clip:.3f} rad ({np.degrees(yaw_change_in_clip):.1f}°)"
            )
            print(
                f"  Target initial yaw: {current_yaw:.3f} rad ({np.degrees(current_yaw):.1f}°)"
            )
            print(
                f"  Expected final yaw: {expected_final_yaw:.3f} rad ({np.degrees(expected_final_yaw):.1f}°)"
            )
            print(
                f"  Actual final yaw: {actual_final_yaw:.3f} rad ({np.degrees(actual_final_yaw):.1f}°)"
            )

            # Use the expected final yaw for better continuity (calculated from yaw change)
            current_yaw = expected_final_yaw

            # Wrap the current yaw to [-π, π] for consistency
            while current_yaw > np.pi:
                current_yaw -= 2 * np.pi
            while current_yaw < -np.pi:
                current_yaw += 2 * np.pi

            # Update target XY translation for next clip to continue from this clip's end
            current_translation_xy = normalized_root_translations[
                -1, :2
            ].copy()  # Only X, Y

            print(
                f"  Final yaw of this clip: {current_yaw:.2f} rad ({np.degrees(current_yaw):.1f}°)"
            )
            print(
                f"  Final XY translation: [{current_translation_xy[0]:.3f}, {current_translation_xy[1]:.3f}]"
            )

    # Combine all clips
    print("\nCombining all clips...")
    combined_dof_positions = np.concatenate(all_dof_positions, axis=0)
    combined_root_rotations = np.concatenate(all_root_rotations, axis=0)
    combined_root_translations = np.concatenate(all_root_translations, axis=0)

    # Apply smoothing if enabled
    if enable_root_smoothing:
        print(
            f"Applying Gaussian smoothing with sigma={smoothing_sigma:.1f}..."
        )

        # Smooth root translations
        print("  Smoothing root translations...")
        combined_root_translations = smooth_root_translations(
            combined_root_translations, sigma=smoothing_sigma
        )

        # Smooth root rotations
        print("  Smoothing root rotations...")
        combined_root_rotations = smooth_root_rotations(
            combined_root_rotations, sigma=smoothing_sigma
        )

        print("  Root smoothing completed.")
    else:
        print("Root smoothing disabled.")

    # Generate pose_aa representation
    print("Generating pose_aa representation...")
    combined_pose_aa = dof_to_pose_aa(
        combined_dof_positions,
        robot_config_path=robot_config_path,
        root_rot=combined_root_rotations,
    )

    # Create combined motion dictionary
    combined_motion_name = os.path.basename(dump_path).split(".")[0]
    combined_motion[combined_motion_name] = {
        "pose_aa": combined_pose_aa,
        "dof": combined_dof_positions,
        "root_rot": combined_root_rotations,
        "root_trans_offset": combined_root_translations,
        "fps": combined_fps,
    }

    # Save combined motion
    joblib.dump(combined_motion, dump_path)

    print(f"\nCombined motion saved to {dump_path}")
    print(f"Total frames: {combined_dof_positions.shape[0]}")
    print(
        f"Total duration: {combined_dof_positions.shape[0] / combined_fps:.2f} seconds"
    )

    return combined_motion


if __name__ == "__main__":
    robot_config_path = (
        # "holomotion/config/robot/unitree_g1/retargeting_23dof_anneal_21dof.yaml"
        # "holomotion/config/robot/unitree_g1/asap_retargeting_23dof_lockwrist.yaml"
        "holomotion/config/motion_retargeting/unitree_G1_23dof_retargeting_v0.yaml"
    )

    # retargeted_root = "data/retargeted_datasets/robodance100"
    # dump_dir = "/home/maiyue01.chen/projects/humanoid_locomotion/data/retargeted_datasets/combined_clips_robodance100/"
    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/douyinhot10v0814"
    # dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/douyinhot10v0814_combined10"
    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/robodance100_no_global_translation"
    # dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/robodance100_no_global_translation_combined10"
    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/lafan1_23dof/"
    # dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/lafan1_23dof_dance_padded/"

    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance"
    # dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance"

    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_stand_squat_norm_yaw"
    # dump_dir = retargeted_root

    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_0825retargeting_processed"
    # dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_0825retargeting_normed"

    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/full_amass_23dof_lockwrist_asap"
    # dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_pkls"

    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_pkls"
    # dump_dir = retargeted_root

    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_seg_lanfan_dance"
    # dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250825_chengdu_demo_pkls"

    # retargeted_root = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250826_chengdu_demo_train_v2"
    # retargeted_root="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/20250826_chengdu_demo_train_v3"
    # dump_dir = retargeted_root
    
    retargeted_root="/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/lafan1_23dof"
    dump_dir = "/home/maiyue01.chen/project3/humanoid_locomotion/holomotion/data/retargeted_datasets/23dof_bydmimic_lafan_dance"

    os.makedirs(dump_dir, exist_ok=True)

    # Configuration flags
    ADD_PADDING = False  # Set to False to skip padding
    NORMALIZE_YAW_TRANS = True  # Set to False to skip normalization
    ENABLE_ROOT_SMOOTHING = False  # Set to True to enable smoothing
    STAND_STILL_TIME = 1.0  # Duration of stand still padding in seconds
    TRANSITION_TIME = 1.0  # Duration of transition between clips in seconds
    SMOOTHING_SIGMA = 5.0  # Smoothing strength (higher = more smoothing)

    # selected_keys = [
    #     "0-hps_douyin031_filter_btws",
    #     "0-hps_douyin173_filter_btws",
    #     # "0-hps_douyin123_filter_btws",
    #     "0-hps_douyin193_filter_btws",
    #     "0-hps_douyin330_filter_btws",
    #     "0-hps_douyin430_filter_btws",
    #     "0-hps_douyin694_filter_btws",
    #     "0-hps_douyin1285_filter_btws",
    #     "0-hps_douyin1044_filter_btws",
    #     "0-hps_douyin1378_filter_btws",
    # ]
    selected_keys = [
        # "0-20250811_douyin143_hps_douyin031_filter",
        # "0-20250811_douyin143_hps_douyin173_filter",
        # "0-20250811_douyin143_hps_douyin193_filter",
        # "0-20250811_douyin143_hps_douyin330_filter",
        # "0-20250811_douyin143_hps_douyin430_filter",
        # "0-20250811_douyin143_hps_douyin694_filter",
        # "0-20250811_douyin143_hps_douyin1285_filter",
        # "0-20250811_douyin143_hps_douyin1044_filter",
        # "0-20250811_douyin143_hps_douyin1378_filter",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949",
        # "dance1_subject2_sliced-1170-3000",
        # "dance2_subject3_sliced-0-1980",
        # "rosbag2_2025_08_22-11_38_24_rosbag2_2025_08_22-11_38_24_0_segment_1_l1y"
        # "rosbag2_2025_08_22-11_38_24_rosbag2_2025_08_22-11_38_24_0_segment_2_l1y"
        # "rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_1_l1y"
        # "rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_2_l1y"
        # "rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_3_l1y"
        # "rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_4_l1y"
        # "rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_5_l1y"
        # "rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_full"
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded_normalized",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded_normalized_rev",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded_normalized"
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_rev",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949__normalized_combined",
        # "dance1_subject2_sliced-1170-3000",
        # "dance2_subject3_sliced-0-1980",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-129-1680",
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949",
        # "dance1_subject2_sliced-1170-3000_padded_normalized_sliced-1251-1890",
        # "dance1_subject2_sliced-1170-3000_padded_normalized_sliced-60-810",
        "dance1_subject2_sliced-90-615_padded",
    ]

    # List of motion files to combine (in order)
    motion_paths_to_combine = [
        os.path.join(retargeted_root, f"{k}.pkl") for k in selected_keys
    ]

    # Generate output filename based on flags
    operation_suffix = []
    if ADD_PADDING:
        operation_suffix.append("padded")
    if NORMALIZE_YAW_TRANS:
        operation_suffix.append("normalized")
    if ENABLE_ROOT_SMOOTHING:
        operation_suffix.append("smoothed")
    if len(selected_keys) > 1:
        operation_suffix.append("combined")

    suffix_str = "_" + "_".join(operation_suffix) if operation_suffix else ""

    # Combine clips with yaw continuity
    combined_dump_path = os.path.join(
        dump_dir,
        # "0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949_padded.pkl",
        # "lafan1_dance_combined_padded.pkl",
        # "dance1_subject2_sliced-1170-3000_padded.pkl",
        # f"dance2_subject3_sliced-0-1980{suffix_str}.pkl",
        # f"rosbag2_2025_08_22-11_38_24_rosbag2_2025_08_22-11_38_24_0_segment_1_l1y{suffix_str}.pkl",
        # f"rosbag2_2025_08_22-11_38_24_rosbag2_2025_08_22-11_38_24_0_segment_2_l1y{suffix_str}.pkl",
        # f"rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_1_l1y{suffix_str}.pkl",
        # f"rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_2_l1y{suffix_str}.pkl",
        #     f"rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_3_l1y{suffix_str}.pkl",
        #     f"rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_4_l1y{suffix_str}.pkl",
        #     f"rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_segment_5_l1y{suffix_str}.pkl",
        # f"rosbag2_2025_08_25-15_45_00_rosbag2_2025_08_25-15_45_00_0_full{suffix_str}.pkl",
        # f"0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949{suffix_str}.pkl",
        # f"0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949__normalized_combined_{suffix_str}.pkl",
        # f"dance1_subject2_sliced-1170-3000{suffix_str}.pkl",
        # f"dance2_subject3_sliced-0-1980{suffix_str}.pkl",
        # f"0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-129-1680{suffix_str}.pkl",
        # f"0-DanceDB_20120807_VasoAristeidou_Vasso_Salsa_Shines_01_stageii_sliced-390-949{suffix_str}.pkl",
        # f"dance1_subject2_sliced-1170-3000_padded_normalized_sliced-1251-1890{suffix_str}.pkl",
        # f"dance1_subject2_sliced-1170-3000_padded_normalized_sliced-60-810{suffix_str}.pkl",
        f"dance1_subject2_sliced-90-615_padded{suffix_str}.pkl",
    )

    print(f"\nProcessing configuration:")
    print(f"  Add padding: {ADD_PADDING}")
    print(f"  Normalize yaw and translation: {NORMALIZE_YAW_TRANS}")
    print(f"  Enable root smoothing: {ENABLE_ROOT_SMOOTHING}")
    if ADD_PADDING:
        print(f"  Stand still time: {STAND_STILL_TIME}s")
        print(f"  Transition time: {TRANSITION_TIME}s")
    if ENABLE_ROOT_SMOOTHING:
        print(f"  Smoothing sigma: {SMOOTHING_SIGMA}")
    print(f"  Output file: {os.path.basename(combined_dump_path)}\n")

    combine_clips_with_yaw_continuity(
        motion_paths=motion_paths_to_combine,
        dump_path=combined_dump_path,
        robot_config_path=robot_config_path,
        stand_still_time=STAND_STILL_TIME,
        transition_time=TRANSITION_TIME,
        enable_root_smoothing=ENABLE_ROOT_SMOOTHING,
        smoothing_sigma=SMOOTHING_SIGMA,
        add_padding=ADD_PADDING,
        normalize_yaw_trans=NORMALIZE_YAW_TRANS,
    )
