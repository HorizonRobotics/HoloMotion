import json, os, sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import joblib
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
from tqdm import tqdm

from loguru import logger
from holomotion.src.utils import torch_utils
from holomotion.src.motion_retargeting.utils.torch_humanoid_batch import (
    HumanoidBatch,
)
from holomotion.src.motion_retargeting.utils import (
    rotation_conversions as rot_conv,
)
from holomotion.src.motion_retargeting.holomotion_preprocess import (
    HoloMotionPreprocessor,
    ProcessedClip,
)
import ray
import logging


def quaternion_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.norm(q, dim=-1, keepdim=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    s = torch.clamp(s, min=1e-6)
    ax = x / s
    ay = y / s
    az = z / s
    axis_angles = torch.stack([ax * angle, ay * angle, az * angle], dim=-1)
    return axis_angles


def dof_to_pose_aa(
    dof_pos: np.ndarray,
    robot_config_path: Optional[str],
    root_rot: Optional[np.ndarray],
) -> np.ndarray:
    """Compute pose_aa via FK; if no config is provided, return zero placeholders."""
    if not robot_config_path:
        T = dof_pos.shape[0]
        return np.zeros((T, 27, 3), dtype=np.float32)

    robot_cfg = OmegaConf.load(robot_config_path)
    logger.info(f"Loaded robot config for FK from: {robot_config_path}")
    fk = HumanoidBatch(robot_cfg.robot)
    num_aug = len(robot_cfg.robot.extend_config)

    dof_t = torch.as_tensor(dof_pos, dtype=torch.float32)
    if dof_t.dim() == 3 and dof_t.shape[-1] == 1:
        dof_t = dof_t.squeeze(-1)
    T = dof_t.shape[0]

    if root_rot is None:
        root_aa = torch.zeros((T, 3), dtype=torch.float32)
    else:
        rr = torch.as_tensor(root_rot, dtype=torch.float32)
        root_aa = quaternion_to_axis_angle(rr) if rr.shape[-1] == 4 else rr

    joint_aa = fk.dof_axis * dof_t.unsqueeze(-1)
    pose_aa = torch.cat(
        [root_aa.unsqueeze(1), joint_aa, torch.zeros((T, num_aug, 3))], dim=1
    )
    return pose_aa.numpy().astype(np.float32, copy=False)


def load_any_pkl(p: Path):
    with open(p, "rb") as f:
        return joblib.load(f)


def unwrap_source(obj) -> Dict[str, np.ndarray]:
    """Accept {top_key: inner} or flat dict (early GMR)."""
    if isinstance(obj, dict) and len(obj) == 1:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict):
            return inner
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported PKL structure")


def make_motion_key(p: Path, src_dir: Path) -> str:
    rel = p.relative_to(src_dir).with_suffix("")
    return "/".join(rel.parts)


def key_to_filename(key: str) -> str:
    return key.replace("/", "_") + ".npz"


def get_ref_schema(
    ref_dir: Path,
) -> Tuple[Dict[str, Tuple[Tuple[int, ...], np.dtype]], str]:
    """
    Read schema only from ref_dir/_schema.json.
    Expected JSON structure:
    {
      "schema": {
        "root_trans_offset": {"shape": [T, 3], "dtype": "float64"},
        "pose_aa": {"shape": [T, 27, 3], "dtype": "float32"},
        ...
      },
      "sample_top_key": "xxx"
    }
    """
    ref_dir = Path(ref_dir)
    cache_path = ref_dir / "_schema.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Schema JSON not found: {cache_path}")
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse _schema.json: {e}")

    schema: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
    raw = obj.get("schema", {})
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Schema JSON missing 'schema' object or it's empty.")

    for k, v in raw.items():
        if not isinstance(v, dict) or "shape" not in v or "dtype" not in v:
            raise ValueError(f"Bad schema entry for key '{k}': {v}")
        shape = tuple(int(x) for x in v["shape"])
        dtype = np.dtype(v["dtype"])
        schema[k] = (shape, dtype)

    sample_top_key = obj.get("sample_top_key", "")
    return schema, sample_top_key


def infer_T(src_inner: Dict[str, np.ndarray]) -> Optional[int]:
    for key in [
        "root_trans_offset",
        "root_pos",
        "pose_aa",
        "dof",
        "dof_pos",
        "root_rot",
        "smpl_joints",
    ]:
        v = src_inner.get(key)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > 0:
            return int(v.shape[0])
    T = 0
    for v in src_inner.values():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > T:
            T = int(v.shape[0])
    return T or None


def build_inner_from_source(
    src_inner: Dict[str, np.ndarray],
    schema: Dict[str, Tuple[Tuple[int, ...], np.dtype]],
    T_default: int,
) -> Dict[str, object]:
    alt_map = {
        "root_trans_offset": [
            "root_trans_offset",
            "root_pos",
            "trans",
            "root_trans",
        ],
        "pose_aa": ["pose_aa"],
        "dof": ["dof", "dof_pos"],
        "root_rot": ["root_rot", "root_orient", "root_quat"],
        "smpl_joints": ["smpl_joints", "joints", "smpljoints"],
        "fps": ["fps", "mocap_framerate", "mocap_frame_rate"],
    }
    out: Dict[str, object] = {}
    T = infer_T(src_inner) or T_default

    for key, (shape, dtype) in schema.items():
        if key == "fps":
            fps = None
            for cand in alt_map["fps"]:
                v = src_inner.get(cand)
                if isinstance(v, (int, np.integer)):
                    fps = int(v)
                    break
            out["fps"] = int(fps) if fps is not None else 30
            continue

        src_arr = None
        for cand in alt_map.get(key, []):
            v = src_inner.get(cand)
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                src_arr = v
                break

        # Target shape: override leading T; keep source column count for DOF
        if key == "dof" and isinstance(src_arr, np.ndarray):
            target_shape = (T, src_arr.shape[1] if src_arr.ndim >= 2 else 1)
        else:
            ts = list(shape)
            if ts:
                ts[0] = T
            target_shape = tuple(ts)

        if src_arr is None:
            out[key] = np.zeros(target_shape, dtype=dtype)
            continue

        arr = src_arr.astype(dtype, copy=False)

        if key == "dof" and arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] > T:
            arr = arr[:T]
        elif arr.shape[0] < T:
            pad = np.repeat(arr[-1:], T - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)

        if (
            key != "dof"
            and len(target_shape) == 2
            and arr.shape[1] != target_shape[1]
        ):
            if arr.shape[1] > target_shape[1]:
                arr = arr[:, : target_shape[1]]
            else:
                pad = np.zeros(
                    (T, target_shape[1] - arr.shape[1]), dtype=arr.dtype
                )
                arr = np.concatenate([arr, pad], axis=1)

        if len(target_shape) == 3:
            d1 = min(arr.shape[1], target_shape[1])
            d2 = min(arr.shape[2], target_shape[2])
            arr = arr[:, :d1, :d2]
            if (arr.shape[1], arr.shape[2]) != (
                target_shape[1],
                target_shape[2],
            ):
                pad = np.zeros(
                    (
                        T,
                        target_shape[1] - arr.shape[1],
                        target_shape[2] - arr.shape[2],
                    ),
                    dtype=arr.dtype,
                )
                arr = np.concatenate([arr, pad], axis=1)
                if arr.shape[2] != target_shape[2]:
                    pad2 = np.zeros(
                        (T, target_shape[1], target_shape[2] - arr.shape[2]),
                        dtype=arr.dtype,
                    )
                    arr = np.concatenate([arr, pad2], axis=2)

        out[key] = arr.astype(dtype, copy=False)
    return out


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor.copy())


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


def process_single_motion(
    robot_cfg: dict,
    all_samples,  # Can be dict or LazyMotionLoader
    curr_key: str,
    target_fps: int = 50,
    fast_interpolate: bool = True,
    debug_mode: bool = False,
):
    logger.debug(f"Starting process_single_motion for key: {curr_key}")

    humanoid_fk = HumanoidBatch(robot_cfg)

    motion_sample_dict = all_samples[curr_key]

    if len(motion_sample_dict) == 1:
        motion_sample_dict = motion_sample_dict[
            list(motion_sample_dict.keys())[0]
        ]

    logger.debug("Step 3: Extracting sequence length")
    if debug_mode:
        # In debug mode, let exceptions bubble up naturally
        if "root_trans_offset" not in motion_sample_dict:
            available_keys = list(motion_sample_dict.keys())
            raise KeyError(
                f"'root_trans_offset' not found in motion data. Available keys: {available_keys}"
            )
        seq_len = motion_sample_dict["root_trans_offset"].shape[0]
        start, end = 0, seq_len
        logger.debug(f"Step 3 completed - seq_len: {seq_len}")
    else:
        try:
            if "root_trans_offset" not in motion_sample_dict:
                available_keys = list(motion_sample_dict.keys())
                raise KeyError(
                    f"'root_trans_offset' not found in motion data. Available keys: {available_keys}"
                )
            seq_len = motion_sample_dict["root_trans_offset"].shape[0]
            start, end = 0, seq_len
            logger.debug(f"Step 3 completed - seq_len: {seq_len}")
        except Exception as e:
            logger.error(
                f"Step 3 failed - Extracting sequence length: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to extract sequence length: {e}"
            ) from e

    logger.debug("Step 4: Processing root translation")
    if debug_mode:
        # In debug mode, let exceptions bubble up naturally
        trans = to_torch(motion_sample_dict["root_trans_offset"]).clone()[
            start:end
        ]
        logger.debug(f"Step 4 completed - trans shape: {trans.shape}")
    else:
        try:
            trans = to_torch(motion_sample_dict["root_trans_offset"]).clone()[
                start:end
            ]
            logger.debug(f"Step 4 completed - trans shape: {trans.shape}")
        except Exception as e:
            logger.error(
                f"Step 4 failed - Processing root translation: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed to process root translation: {e}"
            ) from e

    logger.debug("Step 5: Processing pose_aa")
    if debug_mode:
        # In debug mode, let exceptions bubble up naturally
        if "pose_aa" not in motion_sample_dict:
            available_keys = list(motion_sample_dict.keys())
            raise KeyError(
                f"'pose_aa' not found in motion data. Available keys: {available_keys}"
            )
        pose_aa = to_torch(motion_sample_dict["pose_aa"][start:end]).clone()
        # If available, enforce root rotation from input quaternions (XYZW)
        if "root_rot" in motion_sample_dict:
            root_quat_xyzw = to_torch(
                motion_sample_dict["root_rot"][start:end]
            ).clone()
            root_quat_wxyz = rot_conv.xyzw_to_wxyz(root_quat_xyzw)
            root_axis_angle = rot_conv.quaternion_to_axis_angle(root_quat_wxyz)
            pose_aa[:, 0, :] = root_axis_angle
        logger.debug(f"Step 5 completed - pose_aa shape: {pose_aa.shape}")
    else:
        try:
            if "pose_aa" not in motion_sample_dict:
                available_keys = list(motion_sample_dict.keys())
                raise KeyError(
                    f"'pose_aa' not found in motion data. Available keys: {available_keys}"
                )
            pose_aa = to_torch(
                motion_sample_dict["pose_aa"][start:end]
            ).clone()
            # If available, enforce root rotation from input quaternions (XYZW)
            if "root_rot" in motion_sample_dict:
                root_quat_xyzw = to_torch(
                    motion_sample_dict["root_rot"][start:end]
                ).clone()
                root_quat_wxyz = rot_conv.xyzw_to_wxyz(root_quat_xyzw)
                root_axis_angle = rot_conv.quaternion_to_axis_angle(
                    root_quat_wxyz
                )
                pose_aa[:, 0, :] = root_axis_angle
            logger.debug(f"Step 5 completed - pose_aa shape: {pose_aa.shape}")
        except Exception as e:
            logger.error(
                f"Step 5 failed - Processing pose_aa: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to process pose_aa: {e}") from e

    logger.debug("Step 6: Calculating dt")
    if debug_mode:
        # In debug mode, let exceptions bubble up naturally
        if "fps" not in motion_sample_dict:
            available_keys = list(motion_sample_dict.keys())
            raise KeyError(
                f"'fps' not found in motion data. Available keys: {available_keys}"
            )
        fps = motion_sample_dict["fps"]
        if fps <= 0:
            raise ValueError(f"Invalid fps value: {fps}")
        dt = 1 / fps
        logger.debug(f"Step 6 completed - fps: {fps}, dt: {dt}")
    else:
        try:
            if "fps" not in motion_sample_dict:
                available_keys = list(motion_sample_dict.keys())
                raise KeyError(
                    f"'fps' not found in motion data. Available keys: {available_keys}"
                )
            fps = motion_sample_dict["fps"]
            if fps <= 0:
                raise ValueError(f"Invalid fps value: {fps}")
            dt = 1 / fps
            logger.debug(f"Step 6 completed - fps: {fps}, dt: {dt}")
        except Exception as e:
            logger.error(f"Step 6 failed - Calculating dt: {e}", exc_info=True)
            raise RuntimeError(f"Failed to calculate dt: {e}") from e

    logger.debug("Step 8: Running forward kinematics")
    if debug_mode:
        # In debug mode, let exceptions bubble up naturally
        curr_motion = humanoid_fk.fk_batch(
            pose_aa[None,],
            trans[None,],
            return_full=True,
            dt=dt,
        )
        logger.debug("Step 8 completed")
    else:
        try:
            curr_motion = humanoid_fk.fk_batch(
                pose_aa[None,],
                trans[None,],
                return_full=True,
                dt=dt,
            )
            logger.debug("Step 8 completed")
        except Exception as e:
            logger.error(
                f"Step 8 failed - Forward kinematics: {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to run forward kinematics: {e}") from e
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
    num_extended_bodies = num_bodies + len(
        robot_cfg.motion.get("extend_config", [])
    )

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
    if "global_rotation_mat_extend" in curr_motion:
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
        curr_motion = fast_interpolate_motion(
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

    if debug_mode:
        for k, v in sample_dict.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                logger.debug(f"{k}: {v.shape}")
            else:
                logger.debug(f"{k}: {v}")

    return sample_dict


class InMemoryAlignedLoader:
    """Minimal Loader interface: compatible with process_single_motion sample access."""

    def __init__(self, mapping: Dict[str, Dict[str, object]]):
        self._map = mapping

    def keys(self) -> List[str]:
        return list(self._map.keys())

    def __len__(self):
        return len(self._map)

    def __getitem__(self, k: str):
        return self._map[k]

    def load(self, k: str):
        return self._map[k]

    def get(self, k: str, default=None):
        return self._map.get(k, default)


def arrays_for_npz(
    sample: Dict, emit_prefixed: bool = True, emit_legacy: bool = False
) -> Dict[str, np.ndarray]:
    """
    Build NPZ arrays:
    - Always include frame_flag if present
    - If emit_prefixed: write ref_* arrays mapped from base keys
    - If emit_legacy: also include legacy, unprefixed keys for compatibility
    """
    base_to_ref = {
        "dof_pos": "ref_dof_pos",
        "dof_vel": "ref_dof_vel",
        "global_translation": "ref_global_translation",
        "global_rotation_quat": "ref_global_rotation_quat",
        "global_velocity": "ref_global_velocity",
        "global_angular_velocity": "ref_global_angular_velocity",
    }
    out: Dict[str, np.ndarray] = {}
    if isinstance(sample.get("frame_flag"), np.ndarray):
        out["frame_flag"] = sample["frame_flag"]
    for base, ref_name in base_to_ref.items():
        v = sample.get(base, None)
        if isinstance(v, np.ndarray):
            if emit_prefixed:
                out[ref_name] = v
            if emit_legacy:
                out[base] = v
    return out


@ray.remote
class MotionProcessorActor:
    """
    Persistent Ray actor that loads robot config once and processes PKLs asynchronously.
    """

    def __init__(
        self,
        robot_cfg_path: str,
        schema: Dict[str, Tuple[Tuple[int, ...], np.dtype]],
    ):
        cfg = OmegaConf.load(robot_cfg_path)
        self.robot_cfg = cfg.robot
        self.schema = schema
        # Cached FK holder for DOF → axis-angle conversion (uses dof_axis)
        self._fk_for_dof = HumanoidBatch(self.robot_cfg)

    def _dof_to_pose_aa_cached(
        self, dof_pos: np.ndarray, root_rot: Optional[np.ndarray]
    ) -> np.ndarray:
        dof_t = torch.as_tensor(dof_pos, dtype=torch.float32)
        if dof_t.dim() == 3 and dof_t.shape[-1] == 1:
            dof_t = dof_t.squeeze(-1)
        T = int(dof_t.shape[0])

        if root_rot is None:
            root_aa = torch.zeros((T, 3), dtype=torch.float32)
        else:
            rr = torch.as_tensor(root_rot, dtype=torch.float32)
            root_aa = quaternion_to_axis_angle(rr) if rr.shape[-1] == 4 else rr

        num_aug = len(self.robot_cfg.extend_config)
        joint_aa = self._fk_for_dof.dof_axis * dof_t[:, :, None]
        pose_aa = torch.cat(
            [root_aa[:, None, :], joint_aa, torch.zeros((T, num_aug, 3))],
            dim=1,
        )
        return pose_aa.numpy().astype(np.float32, copy=False)

    def process_pkl(
        self,
        p_str: str,
        src_dir_str: str,
        target_fps: int,
        fast_interpolate: bool,
        debug_mode: bool,
    ) -> Tuple[bool, Dict[str, object]]:
        """
        Returns (success, payload). On success, payload contains:
          { "flat_key": str, "sample": Dict[str, np.ndarray|scalar] }
        """
        p = Path(p_str)
        src_dir = Path(src_dir_str)
        motion_key_rel = make_motion_key(p, src_dir)
        flat_key = motion_key_rel.replace("/", "_")

        obj = load_any_pkl(p)
        inner = unwrap_source(obj)
        T_default = infer_T(inner) or 1
        aligned = build_inner_from_source(inner, self.schema, T_default)

        dof = aligned.get("dof")
        if isinstance(dof, np.ndarray) and dof.size > 0:
            root_rot = aligned.get("root_rot", None)
            aligned["pose_aa"] = self._dof_to_pose_aa_cached(dof, root_rot)

        loader = InMemoryAlignedLoader({flat_key: aligned})
        sample = process_single_motion(
            self.robot_cfg,
            loader,
            flat_key,
            int(target_fps),
            bool(fast_interpolate),
            bool(debug_mode),
        )
        payload: Dict[str, object] = {"flat_key": flat_key, "sample": sample}
        return True, payload


@hydra.main(
    config_path="../../config",
    config_name="motion_retargeting/gmr_to_holomotion",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Setup logging
    logger.remove()
    log_level = "DEBUG" if bool(cfg.processing.debug_mode) else "INFO"
    logger.add(sys.stderr, level=log_level, colorize=True)

    src_path = Path(str(cfg.io.src_dir)).expanduser().resolve()
    ref_dir = Path(str(cfg.io.ref_dir)).expanduser().resolve()
    out_root = Path(str(cfg.io.out_root)).expanduser().resolve()
    clips_dir = out_root / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # dump resolved config used
    (out_root).mkdir(parents=True, exist_ok=True)
    with open(out_root / "config_used.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 1) schema from _schema.json
    schema, _ = get_ref_schema(ref_dir)

    # 2) gather PKLs
    if src_path.is_file() and src_path.suffix == ".pkl":
        src_pkls = [src_path]
        root_for_keys = src_path.parent
    else:
        src_pkls = []
        for dirpath, _, filenames in os.walk(src_path, followlinks=True):
            for filename in filenames:
                if filename.endswith(".pkl"):
                    p = Path(dirpath) / filename
                    if p.is_file():
                        src_pkls.append(p)
        src_pkls = sorted(src_pkls)
        root_for_keys = src_path

    # 3) quiet third-party DEBUG logs (e.g., filelock/Ray)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("ray").setLevel(logging.ERROR)
    os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "error")

    # 4) initialize Ray
    if str(cfg.ray.ray_address):
        ray.init(
            address=str(cfg.ray.ray_address),
            ignore_reinit_error=True,
            log_to_driver=False,
            include_dashboard=False,
            logging_level=logging.ERROR,
        )
    else:
        num_cpus = (
            None if int(cfg.ray.num_workers) <= 0 else int(cfg.ray.num_workers)
        )
        ray.init(
            num_cpus=num_cpus,
            ignore_reinit_error=True,
            log_to_driver=False,
            include_dashboard=False,
            logging_level=logging.ERROR,
        )

    # 5) build work list (skip existing if requested)
    skip_existing = bool(cfg.processing.skip_existing)
    work_list: List[Path] = []
    for p in src_pkls:
        motion_key = make_motion_key(p, root_for_keys)
        out_name = key_to_filename(motion_key)
        if skip_existing and (clips_dir / out_name).exists():
            continue
        work_list.append(p)

    if not work_list:
        logger.info("No tasks to run (all outputs exist or no PKLs found).")
        ray.shutdown()
        return

    # 6) create persistent actors (each loads robot config once)
    if int(cfg.ray.num_workers) > 0:
        num_actors = min(len(work_list), int(cfg.ray.num_workers))
    else:
        available_cpus = int(ray.available_resources().get("CPU", 1))
        num_actors = min(len(work_list), max(1, available_cpus))
    actors = [
        MotionProcessorActor.remote(str(cfg.io.robot_config), schema)
        for _ in range(num_actors)
    ]

    # Parse pipeline config
    pipeline_cfg = cfg.get("preprocess", None)
    pipeline = None
    if pipeline_cfg is not None:
        pipeline_val = pipeline_cfg.get("pipeline", None)
        if pipeline_val is not None:
            if isinstance(pipeline_val, (list, tuple, ListConfig)):
                pipeline = [str(s) for s in pipeline_val]
            elif isinstance(pipeline_val, str):
                import ast
                pipeline = ast.literal_eval(pipeline_val)
            else:
                logger.warning(f"Unexpected pipeline type: {type(pipeline_val)}, value: {pipeline_val}")
                pipeline = []
        else:
            pipeline = []
    else:
        pipeline = []

    # Separate per-clip stages from dataset-level stages
    per_clip_pipeline = [s for s in pipeline if s != "tagging"] if pipeline else []
    tagging_enabled = pipeline and "tagging" in pipeline

    logger.info("=" * 80)
    logger.info("Preprocessing Configuration:")
    if pipeline:
        logger.info(f"  Pipeline stages: {pipeline}")
        logger.info(f"  Number of stages: {len(pipeline)}")
        for i, stage in enumerate(pipeline, 1):
            logger.info(f"    {i}. {stage}")
        if tagging_enabled:
            logger.info("  Note: 'tagging' is a dataset-level operation and will run after all clips are processed")
    else:
        logger.info("  No preprocessing pipeline specified - no processors will be applied")
    logger.info("=" * 80)

    preprocessor = HoloMotionPreprocessor(
        slicing_cfg=cfg.slicing,
        filtering_cfg=cfg.filtering,
        tagging_cfg=cfg.tagging,
        padding_cfg=cfg.get("padding", None),
        pipeline=per_clip_pipeline if per_clip_pipeline else None,
    )

    # 7) asynchronously schedule PKLs to actors (round-robin)
    pending = {}
    next_idx = 0
    # prime the queue
    for i in range(min(num_actors, len(work_list))):
        p = work_list[next_idx]
        next_idx += 1
        ref = actors[i].process_pkl.remote(
            str(p),
            str(root_for_keys),
            int(cfg.processing.target_fps),
            bool(cfg.processing.fast_interpolate),
            bool(cfg.processing.debug_mode),
        )
        pending[ref] = i

    # 8) collect results and keep feeding new tasks (post-process in-memory, then write)
    total_outputs = 0
    with tqdm(total=len(work_list), desc="Ray: PKL→NPZ (Hydra)") as pbar:
        while pending:
            done, _ = ray.wait(list(pending.keys()), num_returns=1)
            ref = done[0]
            actor_idx = pending.pop(ref)
            ok, payload = ray.get(ref)
            if ok:
                flat_key: str = payload["flat_key"]  # type: ignore[assignment]
                sample: Dict = payload["sample"]  # type: ignore[assignment]
                arrays_ref = arrays_for_npz(
                    sample,
                    emit_prefixed=bool(cfg.naming.emit_prefixed),
                    emit_legacy=bool(cfg.naming.emit_legacy),
                )
                base_meta = {
                    "motion_key": flat_key,
                    "raw_motion_key": flat_key,
                    "motion_fps": float(sample["motion_fps"]),
                    "num_frames": int(sample["num_frames"]),
                    "wallclock_len": float(sample["wallclock_len"]),
                    "num_dofs": int(sample["num_dofs"]),
                    "num_bodies": int(sample["num_bodies"]),
                    "num_extended_bodies": int(sample["num_extended_bodies"]),
                    "slice_start": 0,
                    "slice_end": int(sample["num_frames"]),
                }
                base_clip = ProcessedClip(
                    motion_key=flat_key,
                    metadata=base_meta,
                    arrays=arrays_ref,
                )
                clips = preprocessor.process_clip(base_clip)
                for clip in clips:
                    out_name = f"{clip.motion_key}.npz"
                    out_path = clips_dir / out_name
                    np.savez_compressed(
                        out_path,
                        metadata=json.dumps(clip.metadata),
                        **clip.arrays,
                    )
                    total_outputs += 1
            else:
                logger.warning(f"Processing failed: {payload}")
            pbar.update(1)
            if next_idx < len(work_list):
                p = work_list[next_idx]
                next_idx += 1
                new_ref = actors[actor_idx].process_pkl.remote(
                    str(p),
                    str(root_for_keys),
                    int(cfg.processing.target_fps),
                    bool(cfg.processing.fast_interpolate),
                    bool(cfg.processing.debug_mode),
                )
                pending[new_ref] = actor_idx

    # 9) Optional kinematic tagging (write to out_root level)
    if tagging_enabled:
        tags_path = (
            Path(str(cfg.tagging.output_json_path)).expanduser().resolve()
            if str(cfg.tagging.output_json_path)
            else (out_root / "kinematic_tags.json")
        )
        preprocessor.tag_directory(clips_dir, tags_path)

    logger.info(
        f"Done. NPZ written to: {clips_dir} (total clips: {total_outputs})"
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
