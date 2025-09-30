# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import fcntl
import os
import pickle
import shutil
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List

import hydra
import joblib
import lmdb
import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from holomotion.src.utils import torch_utils
from holomotion.src.motion_retargeting.utils.torch_humanoid_batch import HumanoidBatch

max_threads = 1
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
torch.set_num_threads(max_threads)


class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor.copy())


def fix_trans_height(pose_aa, trans, humanoid_fk, fix_height_mode):
    if fix_height_mode == FixHeightMode.no_fix:
        return trans, 0

    with torch.no_grad():
        mesh_obj = humanoid_fk.mesh_fk(pose_aa[None, :1], trans[None, :1])
        height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
        # trans[..., 2] -= height_diff

        return trans, height_diff


def interpolate_motion(motion_dict, source_fps, target_fps):
    """Interpolate motion data from source_fps to target_fps.

    Args:
        motion_dict: Dictionary containing motion data
        source_fps: Original frames per second
        target_fps: Target frames per second

    Returns:
        Dictionary with interpolated motion data

    """
    # Calculate timestamps for original and target motions
    orig_dt = 1.0 / source_fps
    target_dt = 1.0 / target_fps

    # Find the first tensor to determine number of frames
    for v in motion_dict.values():
        if torch.is_tensor(v):
            num_frames = v.shape[0]
            device = v.device
            break
    else:
        return motion_dict  # No tensor data to interpolate

    orig_times = torch.arange(0, num_frames, device=device) * orig_dt
    wallclock_len = orig_dt * (num_frames - 1)
    target_num_frames = int(wallclock_len * target_fps) + 1
    target_times = (
        torch.arange(0, target_num_frames, device=device) * target_dt
    )

    # Create interpolated motion dictionary
    interp_motion = {}

    for k, v in motion_dict.items():
        if not torch.is_tensor(v):
            interp_motion[k] = v
            continue
        is_quat = "quat" in k
        interp_motion[k] = interpolate_tensor(
            v, orig_times, target_times, is_quat
        )

    return interp_motion


def interpolate_tensor(tensor, orig_times, target_times, use_slerp=False):
    """Interpolate a tensor based on original and target timestamps.

    Args:
        tensor: Tensor to interpolate [num_frames, ...]
        orig_times: Original timestamps
        target_times: Target timestamps
        use_slerp: Whether to use SLERP for quaternion interpolation

    Returns:
        Interpolated tensor [target_num_frames, ...]

    """
    target_num_frames = len(target_times)
    shape = list(tensor.shape)
    shape[0] = target_num_frames
    interp_data = torch.zeros(shape, device=tensor.device)

    # Handle different tensor dimensions
    if len(tensor.shape) == 2:
        interp_data = interpolate_frames(
            tensor, orig_times, target_times, use_slerp
        )
    elif len(tensor.shape) == 3:
        # For tensors with multiple sequences (e.g., joint data)
        for j in range(tensor.shape[1]):
            interp_data[:, j] = interpolate_frames(
                tensor[:, j], orig_times, target_times, use_slerp
            )

    return interp_data


def interpolate_frames(frames, orig_times, target_times, use_slerp=False):
    """Interpolate a sequence of frames.

    Args:
        frames: Frame data [num_frames, dim]
        orig_times: Original timestamps
        target_times: Target timestamps
        use_slerp: Whether to use SLERP for quaternion interpolation

    Returns:
        Interpolated frames [target_num_frames, dim]

    """
    target_num_frames = len(target_times)
    interp_data = torch.zeros(
        (target_num_frames, frames.shape[1]), device=frames.device
    )

    for i in range(target_num_frames):
        t = target_times[i]

        # Handle edge cases
        if t <= orig_times[0]:
            interp_data[i] = frames[0]
        elif t >= orig_times[-1]:
            interp_data[i] = frames[-1]
        else:
            # Find surrounding frames
            idx = torch.searchsorted(orig_times, t) - 1
            next_idx = idx + 1

            # Calculate interpolation factor
            alpha = (t - orig_times[idx]) / (
                orig_times[next_idx] - orig_times[idx]
            )

            if use_slerp and frames.shape[1] == 4:  # Quaternion data
                interp_data[i] = torch_utils.slerp(
                    frames[idx : idx + 1],
                    frames[next_idx : next_idx + 1],
                    alpha,
                ).squeeze(0)
            else:  # Linear interpolation
                interp_data[i] = (
                    frames[idx] * (1 - alpha) + frames[next_idx] * alpha
                )

    return interp_data


def fast_interpolate_motion(motion_dict, source_fps, target_fps):
    """Optimized motion interpolation that preserves correctness"""
    # Early return if no interpolation needed
    if source_fps == target_fps:
        return motion_dict

    # Calculate timestamps
    orig_dt = 1.0 / source_fps
    target_dt = 1.0 / target_fps

    # Find the first tensor to determine number of frames
    for v in motion_dict.values():
        if torch.is_tensor(v):
            num_frames = v.shape[0]
            device = v.device
            break
    else:
        return motion_dict  # No tensor data to interpolate

    orig_times = torch.arange(0, num_frames, device=device) * orig_dt
    wallclock_len = orig_dt * (num_frames - 1)
    target_num_frames = int(wallclock_len * target_fps) + 1
    target_times = (
        torch.arange(0, target_num_frames, device=device) * target_dt
    )

    # Create interpolated motion dictionary
    interp_motion = {}

    for k, v in motion_dict.items():
        if not torch.is_tensor(v):
            interp_motion[k] = v
            continue

        is_quat = "quat" in k
        interp_motion[k] = batch_interpolate_tensor(
            v, orig_times, target_times, is_quat
        )

    return interp_motion


def batch_interpolate_tensor(
    tensor, orig_times, target_times, use_slerp=False
):
    """Optimized tensor interpolation with batch processing"""
    target_num_frames = len(target_times)
    shape = list(tensor.shape)
    shape[0] = target_num_frames

    # Create empty output tensor
    result = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)

    if len(tensor.shape) == 2:
        # For 2D tensors - process all frames at once
        # Create masks for the three cases
        before_mask = target_times <= orig_times[0]
        after_mask = target_times >= orig_times[-1]
        valid_mask = ~(before_mask | after_mask)

        # Handle edge cases
        if before_mask.any():
            result[before_mask] = tensor[0]
        if after_mask.any():
            result[after_mask] = tensor[-1]

        # Process interpolation for valid times
        if valid_mask.any():
            valid_times = target_times[valid_mask]
            # Get indices for lower frames
            indices = torch.searchsorted(orig_times, valid_times) - 1
            # Ensure indices are valid
            indices = torch.clamp(indices, 0, len(orig_times) - 2)
            next_indices = indices + 1

            # Calculate weights
            alphas = (valid_times - orig_times[indices]) / (
                orig_times[next_indices] - orig_times[indices]
            )
            alphas = alphas.unsqueeze(-1)  # Add dimension for broadcasting

            if use_slerp and tensor.shape[1] == 4:  # Quaternion data
                # Process in smaller batches to avoid memory issues
                batch_size = 1000  # Adjust based on available memory
                num_valid = valid_mask.sum()

                for i in range(0, num_valid, batch_size):
                    end_idx = min(i + batch_size, num_valid)
                    batch_indices = torch.where(valid_mask)[0][i:end_idx]
                    batch_alphas = alphas[i:end_idx]
                    batch_lower_indices = indices[i:end_idx]
                    batch_upper_indices = next_indices[i:end_idx]

                    # Get frame data for this batch
                    frames_low = tensor[batch_lower_indices]
                    frames_high = tensor[batch_upper_indices]

                    # Apply SLERP to this batch
                    result[batch_indices] = torch_utils.slerp(
                        frames_low, frames_high, batch_alphas
                    )
            else:
                # Standard linear interpolation - can be done in one batch
                frames_low = tensor[indices]
                frames_high = tensor[next_indices]
                result[valid_mask] = (
                    frames_low * (1 - alphas) + frames_high * alphas
                )

    elif len(tensor.shape) == 3:
        # For 3D tensors - process each joint sequence
        for j in range(tensor.shape[1]):
            result[:, j] = batch_interpolate_tensor(
                tensor[:, j], orig_times, target_times, use_slerp
            )

    return result


def process_single_motion(
    robot_cfg: dict,
    all_samples,  # Can be dict or LazyMotionLoader
    curr_key: str,
    target_fps: int = 50,
    fast_interpolate: bool = True,
):
    humanoid_fk = HumanoidBatch(robot_cfg.motion)
    motion_sample_dict = all_samples[curr_key]
    seq_len = motion_sample_dict["root_trans_offset"].shape[0]
    start, end = 0, seq_len
    trans = to_torch(motion_sample_dict["root_trans_offset"]).clone()[
        start:end
    ]
    pose_aa = to_torch(motion_sample_dict["pose_aa"][start:end]).clone()
    dt = 1 / motion_sample_dict["fps"]
    trans, _ = fix_trans_height(
        pose_aa,
        trans,
        humanoid_fk,
        fix_height_mode=FixHeightMode.full_fix,
    )
    curr_motion = humanoid_fk.fk_batch(
        pose_aa[None,],
        trans[None,],
        return_full=True,
        dt=dt,
    )
    curr_motion = dict(
        {
            k: v.squeeze() if torch.is_tensor(v) else v
            for k, v in curr_motion.items()
        }
    )
    motion_fps = curr_motion["fps"]
    motion_dt = 1.0 / motion_fps
    num_frames = curr_motion["global_rotation"].shape[0]
    wallclock_len = motion_dt * (num_frames - 1)
    num_dofs = len(robot_cfg.motion.dof_names)
    num_bodies = len(robot_cfg.motion.body_names)
    num_extended_bodies = num_bodies + len(robot_cfg.motion.extend_config)

    # logger.info(f"Number of frames: {num_frames}")
    # logger.info(f"Number of DOFs: {num_dofs}")
    # logger.info(f"Number of bodies: {num_bodies}")
    # logger.info(f"Number of extended bodies: {num_extended_bodies}")

    # build a frame_flag array to indicate three status:
    # start_of_motion: 0, middle_of_motion: 1, end_of_motion: 2
    frame_flag = torch.ones(num_frames).int()
    frame_flag[0] = 0
    frame_flag[-1] = 2
    curr_motion["frame_flag"] = frame_flag

    # rename and pop some keys
    curr_motion["global_rotation_quat"] = curr_motion.pop("global_rotation")
    curr_motion["local_rotation_quat"] = curr_motion.pop("local_rotation")
    if "global_translation_extend" in curr_motion:
        curr_motion["global_rotation_quat_extend"] = curr_motion.pop(
            "global_rotation_extend"
        )
    curr_motion.pop("fps")
    curr_motion.pop("global_rotation_mat")
    curr_motion.pop("global_rotation_mat_extend")

    # add some keys
    curr_motion["global_root_translation"] = curr_motion["global_translation"][
        :, 0
    ]
    curr_motion["global_root_rotation_quat"] = curr_motion[
        "global_rotation_quat"
    ][:, 0]

    # Interpolate to target_fps if different from original fps
    if target_fps != motion_fps:
        if fast_interpolate:
            curr_motion = fast_interpolate_motion(
                curr_motion, motion_fps, target_fps
            )
        else:
            curr_motion = interpolate_motion(
                curr_motion, motion_fps, target_fps
            )
        motion_fps = target_fps
        motion_dt = 1.0 / target_fps
        num_frames = (
            next(iter(curr_motion.values())).shape[0]
            if curr_motion
            else num_frames
        )
        wallclock_len = motion_dt * (num_frames - 1)

    sample_dict = {
        "motion_name": curr_key,
        "motion_fps": motion_fps,
        "num_frames": num_frames,
        "wallclock_len": wallclock_len,
        "num_dofs": num_dofs,
        "num_bodies": num_bodies,
        "num_extended_bodies": num_extended_bodies,
    }
    sample_dict.update(
        {
            k: curr_motion[k].float().cpu().numpy()
            for k in sorted(curr_motion.keys())
        }
    )

    return sample_dict


@ray.remote(num_cpus=1)
def process_single_motion_remote(
    robot_cfg, all_samples, curr_key, target_fps=50, fast_interpolate=True
):
    return process_single_motion(
        robot_cfg, all_samples, curr_key, target_fps, fast_interpolate
    )


@ray.remote(num_cpus=1)
class MotionProcessorActor:
    def __init__(self, robot_cfg, all_samples):
        """Initialize actor with robot config and all motion samples.

        Args:
            robot_cfg: Robot configuration
            all_samples: Dictionary of all motion samples to process

        """
        self.robot_cfg = robot_cfg
        self.all_samples = all_samples

    def process_motion(self, curr_key, target_fps=50, fast_interpolate=True):
        """Process a single motion by key.

        Args:
            curr_key: Key of the motion to process
            target_fps: Target frames per second
            fast_interpolate: Whether to use fast interpolation

        Returns:
            Processed motion sample dictionary

        """
        return process_single_motion(
            self.robot_cfg,
            self.all_samples,
            curr_key,
            target_fps,
            fast_interpolate,
        )


@ray.remote(num_cpus=1)
class MotionLmdbWriterActor:
    """Ray Actor for parallel writing of motion data to LMDB"""

    def __init__(self, db_path: str):
        """Initialize the writer actor with an LMDB environment

        Args:
            db_path: Path to the LMDB database

        """
        self.env = lmdb.open(
            db_path,
            map_size=1024 * 1024 * 1024 * 1024,  # 1 TB
            subdir=True,
            map_async=True,
            writemap=True,
            meminit=False,
            max_readers=126,
            lock=True,  # Enable lock for concurrent access
        )

    def add_motion_data(self, motion_key: str, motion_data: Dict) -> bool:
        """Add motion data to LMDB database in a thread-safe manner

        Args:
            motion_key: Key identifier for the motion
            motion_data: Dictionary containing the motion data

        Returns:
            bool: True if successful, False otherwise

        """
        try:
            with self.env.begin(write=True) as txn:
                # Store the motion data
                txn.put(
                    f"motion/{motion_key}".encode(),
                    pickle.dumps(motion_data),
                )

                # Update the UUID tracking
                all_uuids_key = b"all_uuids"
                all_uuids = set()
                existing_uuids = txn.get(all_uuids_key)
                if existing_uuids:
                    all_uuids = set(pickle.loads(existing_uuids))
                all_uuids.add(motion_key)
                txn.put(all_uuids_key, pickle.dumps(list(all_uuids)))

                # Update motion count
                motion_count_key = b"motion_count"
                motion_count = 0
                existing_count = txn.get(motion_count_key)
                if existing_count:
                    motion_count = pickle.loads(existing_count)
                txn.put(motion_count_key, pickle.dumps(motion_count + 1))

            return True
        except Exception as e:
            logger.error(f"Failed to add motion data {motion_key}: {str(e)}")
            return False

    def close(self):
        """Close the LMDB environment"""
        if self.env:
            self.env.sync()
            self.env.close()


@ray.remote(num_cpus=1)
class ProcessAndWriteActor:
    """Actor that both processes motion data and writes it to LMDB,
    handling dynamic map size adjustments with proper synchronization.
    """

    def __init__(
        self,
        robot_cfg,
        all_samples,
        db_path: str,
        initial_map_size: int,
        map_growth_factor: float = 2.0,
        max_retries: int = 10,
    ):
        """Initialize with robot config, motion samples, db path, and map size parameters.

        Args:
            robot_cfg: Robot configuration
            all_samples: Dictionary of motion samples
            db_path: Path to LMDB database
            initial_map_size: Initial size of the LMDB map in bytes
            map_growth_factor: Factor by which to increase map size when full
            max_retries: Maximum number of resize attempts

        """
        self.robot_cfg = robot_cfg
        self.all_samples = all_samples
        self.db_path = db_path
        self.current_map_size = initial_map_size
        self.map_growth_factor = map_growth_factor
        self.max_retries = max_retries
        self.resize_lock_path = os.path.join(db_path, "resize.lock")
        self.env = self._open_env()

    def _acquire_resize_lock(self, timeout=30):
        """Acquire a file-based lock for LMDB resizing operations."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.resize_lock_path), exist_ok=True)

            # Open lock file
            self.lock_file = open(self.resize_lock_path, "w")

            # Try to acquire exclusive lock with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(
                        self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    return True
                except BlockingIOError:
                    time.sleep(0.1)

            # Timeout reached
            self.lock_file.close()
            return False
        except Exception as e:
            logger.error(f"Failed to acquire resize lock: {e}")
            if (
                hasattr(self, "lock_file")
                and self.lock_file
                and not self.lock_file.closed
            ):
                try:
                    self.lock_file.close()
                except:
                    pass
            return False

    def _release_resize_lock(self):
        """Release the file-based lock."""
        try:
            if (
                hasattr(self, "lock_file")
                and self.lock_file
                and not self.lock_file.closed
            ):
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
        except Exception as e:
            logger.warning(f"Error releasing resize lock: {e}")
        finally:
            # Ensure lock file is always properly closed
            if hasattr(self, "lock_file"):
                try:
                    if not self.lock_file.closed:
                        self.lock_file.close()
                except:
                    pass

    def _get_current_map_size(self):
        """Check the current map size from the database stats."""
        try:
            # First try to get size from existing environment if available and valid
            if self.env and hasattr(self.env, "stat"):
                try:
                    stat = self.env.stat()
                    return stat["map_size"]
                except:
                    pass

            # If that fails, try opening a temporary read-only connection
            if os.path.exists(self.db_path):
                try:
                    temp_env = lmdb.open(
                        self.db_path, readonly=True, lock=False, subdir=True
                    )
                    stat = temp_env.stat()
                    current_size = stat["map_size"]
                    temp_env.close()
                    return current_size
                except Exception as e:
                    logger.debug(f"Could not read current map size: {e}")
                    pass
        except Exception as e:
            logger.debug(f"Error getting current map size: {e}")

        return self.current_map_size

    def _open_env(self):
        """Opens or reopens the LMDB environment with the current map size."""
        try:
            # Check if database already exists and get its current map size
            if os.path.exists(self.db_path):
                try:
                    temp_env = lmdb.open(
                        self.db_path, readonly=True, lock=False, subdir=True
                    )
                    stat = temp_env.stat()
                    current_db_size = stat["map_size"]
                    temp_env.close()

                    # Use the larger of current size or our expected size
                    self.current_map_size = max(
                        self.current_map_size, current_db_size
                    )
                except Exception:
                    pass  # If we can't read it, use our current size

            logger.debug(
                f"Opening LMDB env at {self.db_path} with map size {self.current_map_size / (1024**3):.2f} GB"
            )
            return lmdb.open(
                self.db_path,
                map_size=self.current_map_size,
                subdir=True,
                map_async=True,
                writemap=False,  # Disable writemap to avoid SIGBUS issues with concurrent access
                meminit=False,
                max_readers=126,
                lock=True,  # Use file locking for safe concurrent writes/resizes
            )
        except Exception as e:
            logger.error(f"Failed to open LMDB environment: {e}")
            raise

    def _resize_env(self):
        """Increases the map size and reopens the environment with global synchronization."""
        # Acquire global resize lock
        if not self._acquire_resize_lock():
            logger.warning(
                "Failed to acquire resize lock, trying to reopen environment"
            )
            self._handle_map_resized_error()
            return

        try:
            # Close existing environment safely
            if self.env:
                try:
                    self.env.sync()  # Ensure all data is written
                    self.env.close()
                except Exception as e:
                    logger.warning(
                        f"Error closing environment during resize: {e}"
                    )
                finally:
                    self.env = None

            # Check current database size in case another process already resized it
            old_size = self.current_map_size
            current_db_size = self._get_current_map_size()

            if current_db_size > self.current_map_size:
                # Another process already resized, just update our size and reopen
                self.current_map_size = current_db_size
                logger.info(
                    f"Database already resized by another process to {self.current_map_size / (1024**3):.2f} GB"
                )
            else:
                # We need to resize
                self.current_map_size = int(
                    self.current_map_size * self.map_growth_factor
                )
                logger.warning(
                    f"LMDB map full. Resizing map_size from {old_size / (1024**3):.2f} GB to {self.current_map_size / (1024**3):.2f} GB"
                )

            # Add a small delay to let other processes catch up
            time.sleep(0.5)

            # Reopen environment with new size
            max_reopen_attempts = 3
            for attempt in range(max_reopen_attempts):
                try:
                    self.env = self._open_env()
                    break
                except Exception as e:
                    if attempt == max_reopen_attempts - 1:
                        raise
                    logger.warning(
                        f"Failed to reopen environment on attempt {attempt + 1}: {e}"
                    )
                    time.sleep(0.5)

        finally:
            self._release_resize_lock()

    def _handle_map_resized_error(self):
        """Handle MDB_MAP_RESIZED error by reopening the environment."""
        try:
            # Close existing environment safely
            if self.env:
                try:
                    self.env.sync()  # Ensure all data is written
                    self.env.close()
                except Exception as e:
                    logger.warning(
                        f"Error closing environment during map resize handling: {e}"
                    )
                finally:
                    self.env = None

            # Update our map size to match the database
            current_db_size = self._get_current_map_size()
            if current_db_size > self.current_map_size:
                self.current_map_size = current_db_size

            # Don't increase map size, just reopen with current size
            # The resize was done by another process
            logger.info(
                "Reopening LMDB environment due to map resize by another process"
            )
            time.sleep(0.2)  # Brief delay

            # Attempt to reopen with retry logic
            max_reopen_attempts = 3
            for attempt in range(max_reopen_attempts):
                try:
                    self.env = self._open_env()
                    break
                except Exception as e:
                    if attempt == max_reopen_attempts - 1:
                        raise
                    logger.warning(
                        f"Failed to reopen environment on attempt {attempt + 1}: {e}"
                    )
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to handle map resized error: {e}")
            raise

    def process_and_write(
        self, motion_key: str, target_fps=50, fast_interpolate=True
    ) -> bool:
        """Process a motion and write it directly to LMDB, handling map size errors.

        Args:
            motion_key: Key of the motion to process
            target_fps: Target frames per second
            fast_interpolate: Whether to use fast interpolation

        Returns:
            bool: True if successful, False otherwise

        """
        retries = 0
        motion_data = None
        while retries <= self.max_retries:
            try:
                # Process the motion (only if not already processed in a previous attempt)
                if retries == 0:  # Avoid reprocessing on retry
                    motion_data = process_single_motion(
                        self.robot_cfg,
                        self.all_samples,
                        motion_key,
                        target_fps,
                        fast_interpolate,
                    )

                # Ensure environment is valid before attempting transaction
                if not self.env:
                    logger.warning(
                        f"LMDB environment is None for {motion_key}, attempting to reopen"
                    )
                    self.env = self._open_env()

                # Write to LMDB with additional safety checks
                try:
                    with self.env.begin(write=True) as txn:
                        # --- Begin Transaction ---
                        # Store metadata separately (non-array data)
                        metadata = {
                            k: v
                            for k, v in motion_data.items()
                            if not isinstance(v, (np.ndarray, list))
                            or k
                            in [
                                "motion_name",
                                "motion_fps",
                                "num_frames",
                                "wallclock_len",
                                "num_dofs",
                                "num_bodies",
                                "num_extended_bodies",
                            ]
                        }
                        txn.put(
                            f"motion/{motion_key}/metadata".encode(),
                            pickle.dumps(metadata),
                        )

                        # Store each array separately for direct access
                        for key, value in motion_data.items():
                            if isinstance(value, np.ndarray):
                                # Store numpy array directly
                                buffer_bytes = value.tobytes()
                                txn.put(
                                    f"motion/{motion_key}/{key}".encode(),
                                    buffer_bytes,
                                )

                                # Also store shape and dtype for easy reconstruction
                                txn.put(
                                    f"motion/{motion_key}/{key}_shape".encode(),
                                    pickle.dumps(value.shape),
                                )
                                txn.put(
                                    f"motion/{motion_key}/{key}_dtype".encode(),
                                    pickle.dumps(value.dtype),
                                )

                        # --- Atomic Updates for Tracking ---
                        # Update the UUID tracking (read-modify-write within transaction)
                        all_uuids_key = b"all_uuids"
                        all_uuids = set()
                        existing_uuids_data = txn.get(all_uuids_key)
                        if existing_uuids_data:
                            all_uuids = set(pickle.loads(existing_uuids_data))
                        all_uuids.add(motion_key)
                        txn.put(all_uuids_key, pickle.dumps(list(all_uuids)))

                        # Store available keys for this motion
                        motion_keys_list = list(
                            motion_data.keys()
                        )  # Ensure it's a list
                        txn.put(
                            f"motion/{motion_key}/available_keys".encode(),
                            pickle.dumps(motion_keys_list),
                        )

                        # Update motion count (read-modify-write within transaction)
                        motion_count_key = b"motion_count"
                        motion_count = 0
                        existing_count_data = txn.get(motion_count_key)
                        if existing_count_data:
                            motion_count = pickle.loads(existing_count_data)
                        txn.put(
                            motion_count_key, pickle.dumps(motion_count + 1)
                        )
                        # --- End Transaction ---
                except OSError as e:
                    # Handle SIGBUS and other system-level errors by treating as environment corruption
                    error_msg = str(e)
                    if (
                        "Bus error" in error_msg
                        or "Segmentation fault" in error_msg
                        or e.errno in [7, 11]
                    ):  # SIGBUS/SIGSEGV
                        logger.error(
                            f"System error (possible SIGBUS/SIGSEGV) for {motion_key}: {e}"
                        )
                        if retries < self.max_retries:
                            # Force environment recreation
                            try:
                                if self.env:
                                    self.env.close()
                            except:
                                pass
                            self.env = None
                            time.sleep(0.5)
                            retries += 1
                            continue
                        else:
                            logger.error(
                                f"Failed to recover from system error for {motion_key}"
                            )
                            return False
                    else:
                        raise  # Re-raise if not a system error we can handle

                return True  # Success

            except lmdb.MapFullError:
                logger.warning(
                    f"MapFullError encountered for key {motion_key} on attempt {retries + 1}."
                )
                if retries < self.max_retries:
                    try:
                        self._resize_env()
                    except Exception as resize_error:
                        logger.error(
                            f"Failed to resize environment for {motion_key}: {resize_error}"
                        )
                        if retries == self.max_retries:
                            return False
                    retries += 1
                    time.sleep(0.1 * retries)  # Small delay before retrying
                else:
                    logger.error(
                        f"LMDB MapFullError: Failed to write {motion_key} after {self.max_retries} resize attempts. Max size reached: {self.current_map_size / (1024**3):.2f} GB"
                    )
                    return False  # Failed after max retries

            except lmdb.Error as e:  # Catch other LMDB errors
                error_msg = str(e)
                if "MDB_MAP_RESIZED" in error_msg:
                    logger.warning(
                        f"Map resized by another process for key {motion_key} on attempt {retries + 1}"
                    )
                    if retries < self.max_retries:
                        try:
                            self._handle_map_resized_error()
                        except Exception as resize_error:
                            logger.error(
                                f"Failed to handle map resize for {motion_key}: {resize_error}"
                            )
                            if retries == self.max_retries:
                                return False
                        retries += 1
                        time.sleep(
                            0.1 * retries
                        )  # Small delay before retrying
                        continue
                    else:
                        logger.error(
                            f"Failed to handle map resize for {motion_key} after {self.max_retries} attempts"
                        )
                        return False
                elif (
                    "MDB_BAD_RSLOT" in error_msg
                    or "MDB_CORRUPTED" in error_msg
                ):
                    # Handle database corruption by recreating environment
                    logger.warning(
                        f"Database corruption detected for {motion_key}: {error_msg}"
                    )
                    if retries < self.max_retries:
                        try:
                            if self.env:
                                self.env.close()
                        except:
                            pass
                        self.env = None
                        retries += 1
                        time.sleep(0.5)
                        continue
                    else:
                        logger.error(
                            f"Failed to recover from corruption for {motion_key}"
                        )
                        return False
                else:
                    logger.error(
                        f"LMDB Error during write for {motion_key}: {e}"
                    )
                    return False
            except Exception as e:
                # Catch processing errors or unexpected issues
                logger.error(
                    f"Failed to process and write motion {motion_key}: {str(e)}",
                    exc_info=True,
                )
                return False

        return False  # Should not be reached if loop finishes normally

    def write_split_keys(
        self, train_uuids: List[str], val_uuids: List[str]
    ) -> bool:
        """Writes the train and validation UUID lists to the LMDB database.

        Args:
            train_uuids: List of motion keys for the training set.
            val_uuids: List of motion keys for the validation set.

        Returns:
            bool: True if successful, False otherwise.

        """
        retries = 0
        while retries <= self.max_retries:
            try:
                with self.env.begin(write=True) as txn:
                    txn.put(b"train_uuids", pickle.dumps(train_uuids))
                    txn.put(b"val_uuids", pickle.dumps(val_uuids))
                logger.info(
                    f"Successfully wrote train ({len(train_uuids)}) and val ({len(val_uuids)}) UUID keys."
                )
                return True  # Success

            except lmdb.MapFullError:
                logger.warning(
                    f"MapFullError encountered while writing split keys on attempt {retries + 1}."
                )
                if retries < self.max_retries:
                    self._resize_env()
                    retries += 1
                    time.sleep(0.1 * retries)  # Small delay before retrying
                else:
                    logger.error(
                        f"LMDB MapFullError: Failed to write split keys after {self.max_retries} resize attempts. Max size reached: {self.current_map_size / (1024**3):.2f} GB"
                    )
                    return False  # Failed after max retries
            except lmdb.Error as e:  # Catch other LMDB errors
                error_msg = str(e)
                if "MDB_MAP_RESIZED" in error_msg:
                    logger.warning(
                        f"Map resized by another process while writing split keys on attempt {retries + 1}"
                    )
                    if retries < self.max_retries:
                        self._handle_map_resized_error()
                        retries += 1
                        time.sleep(
                            0.1 * retries
                        )  # Small delay before retrying
                        continue
                    else:
                        logger.error(
                            f"Failed to handle map resize for split keys after {self.max_retries} attempts"
                        )
                        return False
                else:
                    logger.error(f"LMDB Error during split key write: {e}")
                    return False
            except Exception as e:
                logger.error(
                    f"Unexpected error writing split keys: {str(e)}",
                    exc_info=True,
                )
                return False
        return False  # Should not be reached

    def close(self):
        """Close the LMDB environment and clean up resources"""
        try:
            # Release any held locks first
            self._release_resize_lock()

            if self.env:
                try:
                    logger.debug(
                        f"Syncing and closing LMDB env at {self.db_path}"
                    )
                    # Force sync to ensure all data is written
                    self.env.sync()
                    self.env.close()
                except Exception as e:
                    logger.warning(f"Error during LMDB environment close: {e}")
                finally:
                    self.env = None  # Explicitly set to None after closing

            # Clean up lock file if it exists
            try:
                if os.path.exists(self.resize_lock_path):
                    os.remove(self.resize_lock_path)
            except Exception as e:
                logger.debug(f"Could not remove lock file: {e}")

        except Exception as e:
            logger.warning(f"Error during actor cleanup: {e}")
        finally:
            # Ensure environment is always set to None
            self.env = None


# Add a utility function to read partial data from LMDB
def read_motion_array_slice(
    env, motion_key, array_name, start_idx=None, end_idx=None
):
    """Read a slice of an array from the LMDB without loading the entire array (first dimension only)

    Args:
        env: LMDB environment
        motion_key: Key of the motion
        array_name: Name of the array (e.g., 'dof_pos')
        start_idx: Start index for slicing (optional)
        end_idx: End index for slicing (optional)

    Returns:
        Sliced numpy array

    """
    # For backward compatibility, call the more flexible function
    if start_idx is not None or end_idx is not None:
        # Convert to a slice object for the first dimension
        first_dim_slice = slice(start_idx, end_idx)
        return read_motion_array(env, motion_key, array_name, first_dim_slice)
    else:
        return read_motion_array(env, motion_key, array_name)


def read_motion_array(env, motion_key, array_name, slices=None):
    """Read an array from LMDB with support for multi-dimensional slicing

    Args:
        env: LMDB environment
        motion_key: Key of the motion
        array_name: Name of the array (e.g., 'dof_pos')
        slices: Slice specification, which can be:
               - None: Return the entire array
               - slice object: Slice only the first dimension
               - tuple of slices/indices: Multi-dimensional slicing
               - int or list of ints: Index or indices to access

    Examples:
        # Get entire array
        read_motion_array(env, "walk_01", "dof_pos")

        # Slice time dimension (frames 10 through 50)
        read_motion_array(env, "walk_01", "dof_pos", slice(10, 50))

        # Multi-dimensional slicing (frames 10-50, joints 2-5)
        read_motion_array(env, "walk_01", "dof_pos", (slice(10, 50), slice(2, 5)))

        # Get specific frame and all joints
        read_motion_array(env, "walk_01", "dof_pos", 10)

        # Get specific joints for a range of frames
        read_motion_array(env, "walk_01", "dof_pos", (slice(10, 50), [0, 2, 5]))

    Returns:
        Sliced numpy array

    """
    with env.begin(write=False) as txn:
        # Get the array data
        array_data = txn.get(f"motion/{motion_key}/{array_name}".encode())
        if array_data is None:
            raise KeyError(
                f"Array {array_name} not found for motion {motion_key}"
            )

        # Get the shape information
        shape_data = txn.get(
            f"motion/{motion_key}/{array_name}_shape".encode()
        )
        if shape_data is None:
            raise KeyError(f"Shape information for {array_name} not found")
        shape = pickle.loads(shape_data)

        # Get the dtype information
        dtype_data = txn.get(
            f"motion/{motion_key}/{array_name}_dtype".encode()
        )
        if dtype_data is None:
            raise KeyError(f"Dtype information for {array_name} not found")
        dtype = pickle.loads(dtype_data)

        # Convert directly from bytes to numpy array
        flat_array = np.frombuffer(array_data, dtype=dtype)
        array = flat_array.reshape(shape)

        # Apply slicing if requested
        if slices is not None:
            if isinstance(slices, (int, slice)):
                # Single dimension slicing
                return array[slices]
            elif isinstance(slices, tuple):
                # Multi-dimensional slicing
                return array[slices]
            elif isinstance(slices, list):
                # List of indices
                return array[slices]
            else:
                raise ValueError(f"Unsupported slice type: {type(slices)}")

        return array


# Example usage function to demonstrate multi-dimensional slicing
def example_motion_slicing(db_path):
    """Demonstrate different ways to slice motion data"""
    env = lmdb.open(db_path, readonly=True)

    # Let's assume we have a motion called "walk_01" with arrays:
    # - dof_pos shape (120, 25) - 120 frames, 25 joints
    # - global_rotation_quat shape (120, 15, 4) - 120 frames, 15 bodies, 4 quat components

    # Get full arrays
    dof_pos = read_motion_array(env, "walk_01", "dof_pos")
    rotations = read_motion_array(env, "walk_01", "global_rotation_quat")

    # Slice just the time dimension (frames 10-30)
    dof_pos_frames_10_to_30 = read_motion_array(
        env, "walk_01", "dof_pos", slice(10, 30)
    )

    # Multi-dimensional slicing - frames 10-30, joints 0-5
    dof_pos_subset = read_motion_array(
        env, "walk_01", "dof_pos", (slice(10, 30), slice(0, 5))
    )

    # Get a single frame
    frame_20 = read_motion_array(env, "walk_01", "dof_pos", 20)

    # Get specific joints
    selected_joints = read_motion_array(
        env, "walk_01", "dof_pos", (slice(None), [0, 5, 10])
    )

    # Complex multi-dimensional slicing for quaternions
    # Get frames 30-60, bodies 2-5, all quaternion components
    quat_subset = read_motion_array(
        env,
        "walk_01",
        "global_rotation_quat",
        (slice(30, 60), slice(2, 5), slice(None)),
    )

    env.close()
    return {
        "dof_pos_shape": dof_pos.shape,
        "rotation_shape": rotations.shape,
        "time_slice_shape": dof_pos_frames_10_to_30.shape,
        "multi_dim_slice_shape": dof_pos_subset.shape,
        "single_frame_shape": frame_20.shape,
        "selected_joints_shape": selected_joints.shape,
        "quat_subset_shape": quat_subset.shape,
    }


class LazyMotionLoader:
    """Lazy loader for motion data that can handle both single files and directories.
    Only loads individual motion data when requested to save memory.
    """

    def __init__(self, pkl_path: str):
        """Initialize the lazy loader.

        Args:
            pkl_path: Path to either a single PKL file or directory containing PKL files

        """
        self.pkl_path = Path(pkl_path)
        self._motion_keys = None
        self._motion_file_map = {}  # motion_key -> file_path
        self._loaded_data = {}  # Cache for single file case
        self._is_single_file = False

        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.pkl_path}")

        self._initialize()

    def _initialize(self):
        """Initialize the loader by scanning for available motions."""
        if self.pkl_path.is_file():
            # Single file case - load once and cache motion keys
            logger.info(
                f"Initializing lazy loader for single file: {self.pkl_path}"
            )
            self._is_single_file = True
            self._loaded_data = joblib.load(self.pkl_path)
            self._motion_keys = list(self._loaded_data.keys())
            logger.info(
                f"Found {len(self._motion_keys)} motions in single file"
            )

        elif self.pkl_path.is_dir():
            # Directory case - use filenames as motion keys (no loading needed)
            logger.info(
                f"Initializing lazy loader for directory: {self.pkl_path}"
            )
            pkl_files = list(self.pkl_path.glob("*.pkl"))

            if not pkl_files:
                raise ValueError(
                    f"No PKL files found in directory: {self.pkl_path}"
                )

            logger.info(
                f"Found {len(pkl_files)} PKL files, using filenames as motion keys"
            )

            # Use filename (without extension) as motion key
            for pkl_file in pkl_files:
                motion_key = pkl_file.stem
                if motion_key in self._motion_file_map:
                    logger.warning(
                        f"Duplicate motion key found: {motion_key}, using latest file"
                    )
                self._motion_file_map[motion_key] = pkl_file

            self._motion_keys = list(self._motion_file_map.keys())
            logger.info(
                f"Mapped {len(self._motion_keys)} motion keys from {len(pkl_files)} PKL files"
            )

        else:
            raise ValueError(
                f"Path is neither a file nor a directory: {self.pkl_path}"
            )

        if not self._motion_keys:
            raise ValueError("No motion data found")

    def keys(self):
        """Get all available motion keys."""
        return self._motion_keys

    def __getitem__(self, motion_key: str):
        """Lazy load and return motion data for the given key.

        Args:
            motion_key: The motion key to load

        Returns:
            Motion data for the requested key

        Raises:
            KeyError: If the motion key is not found

        """
        if motion_key not in self._motion_keys:
            raise KeyError(f"Motion key not found: {motion_key}")

        if self._is_single_file:
            # Single file case - data is already loaded
            return self._loaded_data[motion_key]
        else:
            # Directory case - load the specific file containing this motion
            pkl_file = self._motion_file_map[motion_key]

            try:
                loaded_data = joblib.load(pkl_file)

                # Check if the loaded data is a dictionary with the motion key
                if isinstance(loaded_data, dict):
                    if motion_key in loaded_data:
                        # PKL file contains {motion_key: motion_data}
                        return loaded_data[motion_key]
                    else:
                        # PKL file contains motion data directly
                        return loaded_data
                else:
                    # PKL file contains raw motion data
                    return loaded_data

            except Exception as e:
                logger.error(
                    f"Failed to load motion {motion_key} from {pkl_file}: {e}"
                )
                raise

    def __contains__(self, motion_key: str):
        """Check if a motion key exists."""
        return motion_key in self._motion_keys

    def __len__(self):
        """Get the number of available motions."""
        return len(self._motion_keys)

    def validate_loader(self):
        """Validate that the loader can access a few sample motions.
        Useful for debugging and ensuring the loader is working correctly.

        Returns:
            bool: True if validation passes, False otherwise

        """
        try:
            if not self._motion_keys:
                logger.error("No motion keys available")
                return False

            # Test loading a few sample motions
            test_keys = self._motion_keys[: min(3, len(self._motion_keys))]

            for test_key in test_keys:
                try:
                    motion_data = self[test_key]
                    if motion_data is None:
                        logger.error(
                            f"Motion data is None for key: {test_key}"
                        )
                        return False

                    # Check if motion data has expected structure
                    if isinstance(motion_data, dict):
                        required_keys = ["root_trans_offset", "pose_aa", "fps"]
                        missing_keys = [
                            k for k in required_keys if k not in motion_data
                        ]
                        if missing_keys:
                            logger.warning(
                                f"Motion {test_key} missing keys: {missing_keys}"
                            )
                            logger.info(
                                f"Available keys in motion {test_key}: {list(motion_data.keys())}"
                            )
                        else:
                            logger.info(
                                f"Motion {test_key} has all required keys"
                            )

                    logger.info(
                        f"Successfully validated motion key: {test_key}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load test motion {test_key}: {e}")
                    return False

            logger.info(
                f"LazyMotionLoader validation passed for {len(test_keys)} test motions"
            )
            return True

        except Exception as e:
            logger.error(f"LazyMotionLoader validation failed: {e}")
            return False

    def __getstate__(self):
        """Custom serialization for Ray actors."""
        return {
            "pkl_path": str(self.pkl_path),
            "motion_keys": self._motion_keys,
            "motion_file_map": {
                k: str(v) for k, v in self._motion_file_map.items()
            },
            "loaded_data": self._loaded_data,
            "is_single_file": self._is_single_file,
        }

    def __setstate__(self, state):
        """Custom deserialization for Ray actors."""
        self.pkl_path = Path(state["pkl_path"])
        self._motion_keys = state["motion_keys"]
        self._motion_file_map = {
            k: Path(v) for k, v in state["motion_file_map"].items()
        }
        self._loaded_data = state["loaded_data"]
        self._is_single_file = state["is_single_file"]


def load_motion_data(pkl_path: str) -> LazyMotionLoader:
    """Create a lazy loader for motion data from either a single PKL file or directory.

    Args:
        pkl_path: Path to either a single PKL file or directory containing PKL files

    Returns:
        LazyMotionLoader instance that can load motions on demand

    Raises:
        FileNotFoundError: If the path doesn't exist
        ValueError: If no valid motion data is found

    """
    return LazyMotionLoader(pkl_path)


def generate_lmdb_readme(db_path):
    """Generate a README.md file documenting the structure of the LMDB dataset.

    Args:
        db_path: Path to the LMDB database

    """
    env = lmdb.open(db_path, readonly=True, lock=False)
    readme_path = os.path.join(db_path, "README.md")

    try:
        # Structure to hold all the information
        structure = {
            "metadata": {},
            "motion_keys": {},
            "array_shapes": {},
            "examples": {},
        }

        with env.begin() as txn:
            # Get basic stats
            all_uuids_data = txn.get(b"all_uuids")
            train_uuids_data = txn.get(b"train_uuids")
            val_uuids_data = txn.get(b"val_uuids")

            all_motion_ids = []
            if all_uuids_data:
                all_motion_ids = pickle.loads(all_uuids_data)
                structure["metadata"]["total_motions"] = len(all_motion_ids)

            if train_uuids_data:
                structure["metadata"]["train_count"] = len(
                    pickle.loads(train_uuids_data)
                )

            if val_uuids_data:
                structure["metadata"]["val_count"] = len(
                    pickle.loads(val_uuids_data)
                )

            # Collect all keys and categorize them
            cursor = txn.cursor()
            key_categories = {}

            for key, _ in cursor:
                key_str = key.decode("utf-8", errors="ignore")
                parts = key_str.split("/")

                # Categorize by top level
                top_level = parts[0] if parts else key_str
                if top_level not in key_categories:
                    key_categories[top_level] = []
                key_categories[top_level].append(key_str)

            structure["key_categories"] = key_categories

            # Analyze a few example motions
            sample_size = min(5, len(all_motion_ids))
            for i in range(sample_size):
                if i < len(all_motion_ids):
                    motion_id = all_motion_ids[i]

                    # Get available keys
                    available_keys_data = txn.get(
                        f"motion/{motion_id}/available_keys".encode()
                    )
                    if available_keys_data:
                        available_keys = pickle.loads(available_keys_data)

                        # Get shape for each array
                        shapes = {}
                        for key in available_keys:
                            if isinstance(key, str):  # Ensure key is a string
                                shape_data = txn.get(
                                    f"motion/{motion_id}/{key}_shape".encode()
                                )
                                if shape_data:
                                    shapes[key] = pickle.loads(shape_data)

                        structure["examples"][motion_id] = {
                            "arrays": available_keys,
                            "shapes": shapes,
                        }

                        # Store all unique array shapes
                        for key, shape in shapes.items():
                            if key not in structure["array_shapes"]:
                                structure["array_shapes"][key] = shape

        # Generate the README content
        readme_content = "# Motion LMDB Dataset Documentation\n\n"

        # Add metadata section
        readme_content += "## Dataset Statistics\n\n"
        readme_content += "| Statistic | Value |\n"
        readme_content += "|-----------|-------|\n"
        for key, value in structure["metadata"].items():
            readme_content += f"| {key} | {value} |\n"

        # Add key structure section
        readme_content += "\n## Key Structure\n\n"
        for category, keys in structure["key_categories"].items():
            readme_content += f"### {category}\n\n"
            if len(keys) > 10:
                readme_content += f"Total keys: {len(keys)}\n\n"
                readme_content += "Sample keys:\n"
                for key in keys[:10]:
                    readme_content += f"- `{key}`\n"
            else:
                for key in keys:
                    readme_content += f"- `{key}`\n"
            readme_content += "\n"

        # Add array shapes section
        readme_content += "## Array Shapes\n\n"
        readme_content += "| Array Name | Shape | Description |\n"
        readme_content += "|------------|-------|-------------|\n"
        for key, shape in structure["array_shapes"].items():
            description = ""
            if "quat" in key:
                description = "Quaternion rotation data"
            elif "pos" in key:
                description = "Position data"
            elif "trans" in key:
                description = "Translation data"
            elif "root" in key:
                description = "Root joint data"

            readme_content += f"| {key} | {shape} | {description} |\n"

        # Add example motion section
        readme_content += "\n## Example Motion\n\n"
        if structure["examples"]:
            example_id = next(iter(structure["examples"].keys()))
            example = structure["examples"][example_id]

            readme_content += f"Example motion ID: `{example_id}`\n\n"
            readme_content += "Available arrays:\n"

            for array_name in example["arrays"]:
                if isinstance(
                    array_name, str
                ):  # Ensure array_name is a string
                    shape_str = str(
                        example["shapes"].get(array_name, "unknown")
                    )
                    readme_content += f"- `{array_name}`: Shape {shape_str}\n"

        # Add usage examples section
        readme_content += "\n## Usage Examples\n\n"
        readme_content += "```python\n"
        readme_content += "import lmdb\n"
        readme_content += "import pickle\n"
        readme_content += "import numpy as np\n\n"

        readme_content += "# Open the LMDB environment\n"
        readme_content += (
            "env = lmdb.open('path/to/this/lmdb', readonly=True)\n\n"
        )

        readme_content += "# Get all motion IDs\n"
        readme_content += "with env.begin() as txn:\n"
        readme_content += (
            "    all_motion_ids = pickle.loads(txn.get(b'all_uuids'))\n\n"
        )

        readme_content += "# Read a specific motion array\n"
        readme_content += (
            "def read_motion_array(env, motion_id, array_name, slices=None):\n"
        )
        readme_content += "    with env.begin() as txn:\n"
        readme_content += "        # Get array data\n"
        readme_content += "        array_data = txn.get(f'motion/{motion_id}/{array_name}'.encode())\n"
        readme_content += "        # Get shape and dtype\n"
        readme_content += "        shape = pickle.loads(txn.get(f'motion/{motion_id}/{array_name}_shape'.encode()))\n"
        readme_content += "        dtype = pickle.loads(txn.get(f'motion/{motion_id}/{array_name}_dtype'.encode()))\n"
        readme_content += "        # Convert to numpy array\n"
        readme_content += "        array = np.frombuffer(array_data, dtype=dtype).reshape(shape)\n"
        readme_content += "        # Apply slicing if requested\n"
        readme_content += "        if slices is not None:\n"
        readme_content += "            return array[slices]\n"
        readme_content += "        return array\n\n"

        readme_content += "# Example: Get positions for frames 10-20\n"
        readme_content += "motion_id = all_motion_ids[0]  # First motion\n"
        readme_content += "dof_pos_slice = read_motion_array(env, motion_id, 'dof_pos', slice(10, 20))\n\n"

        readme_content += (
            "# Example: Multi-dimensional slicing (frames 10-20, joints 0-5)\n"
        )
        readme_content += "dof_pos_subset = read_motion_array(env, motion_id, 'dof_pos', (slice(10, 20), slice(0, 5)))\n\n"

        readme_content += "# Don't forget to close when done\n"
        readme_content += "env.close()\n"
        readme_content += "```\n"

        # Write the README file
        with open(readme_path, "w") as f:
            f.write(readme_content)

        logger.info(f"README.md file generated at {readme_path}")

    except Exception as e:
        logger.error(f"Failed to generate README.md: {str(e)}")
    finally:
        env.close()


@hydra.main(
    config_path="../../config",
    config_name="training/pack_lmdb_database",
    version_base=None,
)
def main(config: OmegaConf):
    if not ray.is_initialized():
        ray.init(num_cpus=config.num_jobs)

    # Create the LMDB database directory
    db_path = config.lmdb_save_dir
    if os.path.exists(db_path):
        # Check if it's a directory and not empty, clear if necessary
        if os.path.isdir(db_path) and os.listdir(db_path):
            logger.warning(
                f"Output directory {db_path} already exists and is not empty. Clearing it."
            )
            shutil.rmtree(
                db_path
            )  # Remove existing directory and its contents
            os.makedirs(db_path, exist_ok=True)
        elif os.path.isfile(db_path):
            logger.warning(
                f"Output path {db_path} exists and is a file. Removing it."
            )
            os.remove(db_path)
            os.makedirs(db_path, exist_ok=True)
        else:
            # Directory exists but is empty, or doesn't exist yet
            os.makedirs(db_path, exist_ok=True)

    # No longer need to open env here, actors manage their own
    # Get map size config
    initial_map_size = config.get(
        "initial_map_size", 1 * 1024**3
    )  # Default 1 GB
    map_growth_factor = config.get("map_growth_factor", 1.5)
    max_retries = config.get("max_retries", 5)

    # Load motion data - handle both single file and directory of files
    try:
        test_all_samples = load_motion_data(config.retargeted_pkl_path)
    except FileNotFoundError:
        logger.error(
            f"Retargeted PKL path not found at: {config.retargeted_pkl_path}"
        )
        ray.shutdown()
        return
    except Exception as e:
        logger.error(f"Error loading motion data: {e}", exc_info=True)
        ray.shutdown()
        return

    # Validate the lazy loader
    logger.info("Validating lazy motion loader...")
    if not test_all_samples.validate_loader():
        logger.error("Lazy motion loader validation failed. Exiting.")
        ray.shutdown()
        return

    motion_keys = (
        test_all_samples.keys()
    )  # LazyMotionLoader.keys() returns the list
    total_motions = len(test_all_samples)

    if total_motions == 0:
        logger.warning("No motions found in the PKL file. Exiting.")
        ray.shutdown()
        return

    # Create actors that both process and write
    # Reduce concurrent actors to prevent memory pressure with small map sizes
    num_actors = min(
        max(2, config.num_jobs // 2), total_motions, 4
    )  # Cap at 4 actors
    logger.info(
        f"Creating {num_actors} ProcessAndWrite Actors (reduced for memory efficiency)."
    )
    actors = [
        ProcessAndWriteActor.remote(
            config.robot,
            test_all_samples,
            db_path,
            initial_map_size=initial_map_size,
            map_growth_factor=map_growth_factor,
            max_retries=max_retries,
        )
        for _ in range(num_actors)
    ]

    # Distribute work to actors
    pending_keys = list(motion_keys)  # Ensure it's a list for proper copying
    tasks = []
    task_to_key = {}  # Maps task IDs to motion keys
    task_to_actor = {}  # Maps task IDs to the actors that process them
    failed_actors = set()  # Track failed actors

    # Setup progress bar
    pbar = tqdm(total=total_motions, desc="Processing and writing motions")

    # Initial batch of tasks - start with fewer tasks per actor
    initial_tasks_per_actor = max(
        1, min(2, total_motions // num_actors)
    )  # Reduce initial load
    actor_idx = 0

    # Distribute initial tasks among actors
    while pending_keys and len(tasks) < num_actors * initial_tasks_per_actor:
        motion_key = pending_keys.pop(0)
        actor = actors[actor_idx]
        if actor not in failed_actors:  # Skip failed actors
            task = actor.process_and_write.remote(motion_key)
            tasks.append(task)
            task_to_key[task] = motion_key
            task_to_actor[task] = actor

        # Move to next actor in a round-robin fashion
        actor_idx = (actor_idx + 1) % num_actors

    # Process results and assign new tasks
    successful_motions = 0
    failed_motions = 0
    completed_tasks = {}  # Initialize dict to store task completion status
    all_successful_keys = []  # Initialize list to store keys of successful tasks

    while tasks:
        # Wait for the next completed task
        done_id, tasks = ray.wait(tasks, num_returns=1)
        done_task = done_id[0]

        # Get the actor that completed this task
        actor = task_to_actor.pop(done_task, None)

        # Get the motion key associated with this task
        motion_key = task_to_key.get(done_task, None)  # Use get for safety

        # Get the result (True/False for success)
        try:
            success = ray.get(done_task)
            completed_tasks[done_task] = (
                success  # Store completion status for potential debugging/stats
            )
        except ray.exceptions.RayActorError as e:
            logger.error(f"Ray actor error for task {motion_key}: {e}")
            success = False
            completed_tasks[done_task] = success  # Mark as failed
            # Mark this actor as failed
            if actor:
                failed_actors.add(actor)
                logger.warning("Actor marked as failed due to crash")
        except ray.exceptions.RayTaskError as e:
            logger.error(f"Ray task error for task {motion_key}: {e}")
            success = False
            completed_tasks[done_task] = success  # Mark as failed
        except Exception as e:
            logger.error(
                f"Unexpected error getting result for task {motion_key}: {e}",
                exc_info=True,
            )
            success = False
            completed_tasks[done_task] = success  # Mark as failed

        # Remove the key from the tracking dict now that we're done with this task
        task_to_key.pop(done_task, None)

        if success:
            successful_motions += 1
            if motion_key is not None:
                all_successful_keys.append(motion_key)  # Append successful key
        else:
            failed_motions += 1
            # Log warning only if processing/writing failed, not if ray.get failed (already logged)
            if motion_key is not None:
                logger.warning(f"Failed to process motion: {motion_key}")

        # If any pending keys remain, assign a new task to a working actor
        if pending_keys and actor is not None and actor not in failed_actors:
            next_key = pending_keys.pop(0)
            try:
                new_task = actor.process_and_write.remote(next_key)
                tasks.append(new_task)
                task_to_key[new_task] = next_key
                task_to_actor[new_task] = actor
            except Exception as e:
                logger.error(f"Failed to assign new task to actor: {e}")
                # Put the key back for another actor to handle
                pending_keys.insert(0, next_key)

        pbar.update(1)

    pbar.close()

    # Clean up actors
    logger.info("Closing actor LMDB environments.")

    # Filter out failed actors to avoid hanging on dead actors
    working_actors = [actor for actor in actors if actor not in failed_actors]

    if working_actors:
        actor_close_tasks = []
        for actor in working_actors:
            try:
                close_task = actor.close.remote()
                actor_close_tasks.append(close_task)
            except Exception as e:
                logger.warning(f"Failed to create close task for actor: {e}")

        if actor_close_tasks:
            try:
                ray.get(actor_close_tasks, timeout=30)  # Reduce timeout to 30s
                logger.info("All working actor environments closed.")
            except ray.exceptions.RayTaskError as e:
                logger.error(f"Ray error closing actors: {e}")
            except ray.exceptions.GetTimeoutError:
                logger.error(
                    "Timeout waiting for actors to close, continuing anyway."
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error closing actors: {e}", exc_info=True
                )

    # For failed actors, just log that they won't be closed properly
    if failed_actors:
        logger.warning(
            f"Skipping cleanup for {len(failed_actors)} failed actors"
        )

    logger.info("Actor cleanup completed.")

    # Generate README.md documentation
    logger.info("Generating README.md for the database.")
    try:
        generate_lmdb_readme(db_path)
    except Exception as e:
        logger.error(f"Failed to generate README.md: {e}", exc_info=True)

    ray.shutdown()

    logger.info(
        f"Motion processing complete. Successful: {successful_motions}, Failed: {failed_motions}"
    )


if __name__ == "__main__":
    main()
