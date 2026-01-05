import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from holomotion.src.motion_retargeting.utils.torch_humanoid_batch import (
    HumanoidBatch,
)
from holomotion.src.motion_retargeting.utils import (
    rotation_conversions as rot_conv,
)


def compute_slices(
    sequence_len: int, window_size: int, overlap: int
) -> List[Tuple[int, int]]:
    step = window_size - overlap
    if step <= 0:
        raise ValueError("window_size must be > overlap")
    slices: List[Tuple[int, int]] = []
    start = 0
    length = int(sequence_len)
    while start < length:
        end = min(start + window_size, length)
        slices.append((start, end))
        if end == length:
            break
        start += step
    return slices


def _reshape_time_flat(a: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    shape = a.shape
    t = shape[0]
    return a.reshape(t, -1), shape


def _butterworth_lowpass_smooth_time(
    a: np.ndarray, fps: float, cutoff_hz: float, order: int
) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    t = a.shape[0]
    if t < 3:
        return a.astype(np.float32, copy=True)
    if fps <= 0.0 or cutoff_hz <= 0.0:
        return a.astype(np.float32, copy=True)
    nyquist = 0.5 * float(fps)
    wn = float(cutoff_hz) / nyquist
    if wn >= 1.0:
        wn = 0.999
    if wn <= 0.0:
        return a.astype(np.float32, copy=True)
    flat, shape = _reshape_time_flat(a.astype(np.float64, copy=False))
    b, a_coefs = butter(int(order), wn, btype="low", analog=False)
    maxlen = max(len(b), len(a_coefs))
    padlen_required = max(3 * (maxlen - 1), 3 * maxlen)
    if t <= padlen_required:
        return a.astype(np.float32, copy=True)
    filtered = filtfilt(b, a_coefs, flat, axis=0, method="pad")
    return filtered.reshape(shape).astype(np.float32, copy=False)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return (q / norm).astype(np.float32, copy=False)


def _quat_hemisphere_align(q: np.ndarray) -> np.ndarray:
    if q.shape[0] == 0:
        return q
    aligned = q.copy()
    prev = aligned[0]
    for t in range(1, aligned.shape[0]):
        dots = np.sum(prev * aligned[t], axis=-1)
        mask = dots < 0.0
        if np.any(mask):
            aligned[t, mask] = -aligned[t, mask]
        prev = aligned[t]
    return aligned


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    conj = q.copy()
    conj[..., :3] = -conj[..., :3]
    return conj


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av = a[..., :3]
    aw = a[..., 3:4]
    bv = b[..., :3]
    bw = b[..., 3:4]
    cross = np.cross(av, bv)
    vec = aw * bv + bw * av + cross
    scalar = aw * bw - np.sum(av * bv, axis=-1, keepdims=True)
    return np.concatenate([vec, scalar], axis=-1)


def _finite_difference_time(a: np.ndarray, dt: float) -> np.ndarray:
    t = a.shape[0]
    if t < 2 or dt <= 0.0:
        return np.zeros_like(a, dtype=np.float32)
    deriv = np.gradient(
        a.astype(np.float64, copy=False),
        dt,
        axis=0,
        edge_order=2 if t >= 3 else 1,
    )
    return deriv.astype(np.float32, copy=False)


def _angular_velocity_from_quat(
    q: np.ndarray, q_dot: np.ndarray
) -> np.ndarray:
    q_conj = _quat_conjugate(q)
    prod = _quat_multiply(q_conj, q_dot)
    omega = 2.0 * prod[..., :3]
    return omega.astype(np.float32, copy=False)


def butterworth_filter_ref_arrays(
    arrays: Dict[str, np.ndarray], fps: float, cutoff_hz: float, order: int
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    dt = 1.0 / float(fps) if float(fps) > 0.0 else 0.0
    if "ref_dof_pos" in arrays:
        a = arrays["ref_dof_pos"].astype(np.float32, copy=True)
        s = _butterworth_lowpass_smooth_time(a, fps, cutoff_hz, order)
        v = _finite_difference_time(s, dt)
        out["ft_ref_dof_pos"] = s
        out["ft_ref_dof_vel"] = v
    if "ref_global_translation" in arrays:
        a = arrays["ref_global_translation"].astype(np.float32, copy=True)
        s = _butterworth_lowpass_smooth_time(a, fps, cutoff_hz, order)
        v = _finite_difference_time(s, dt)
        out["ft_ref_global_translation"] = s
        out["ft_ref_global_velocity"] = v
    if "ref_global_rotation_quat" in arrays:
        q = arrays["ref_global_rotation_quat"].astype(np.float32, copy=True)
        q = _quat_normalize(q)
        q = _quat_hemisphere_align(q)
        qs = _butterworth_lowpass_smooth_time(q, fps, cutoff_hz, order)
        qs = _quat_normalize(qs)
        q_dot = _finite_difference_time(qs, dt)
        out["ft_ref_global_rotation_quat"] = _quat_normalize(qs)
        out["ft_ref_global_angular_velocity"] = _angular_velocity_from_quat(
            qs, q_dot
        )
    return out


def _summary(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q25": 0.0,
            "q75": 0.0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def _ds_summary(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {
            "DS_mean": 0.0,
            "DS_std": 0.0,
            "DS_median": 0.0,
            "DS_min": 0.0,
            "DS_max": 0.0,
            "DS_q25": 0.0,
            "DS_q75": 0.0,
        }
    return {
        "DS_mean": float(arr.mean()),
        "DS_std": float(arr.std()),
        "DS_median": float(np.median(arr)),
        "DS_min": float(arr.min()),
        "DS_max": float(arr.max()),
        "DS_q25": float(np.quantile(arr, 0.25)),
        "DS_q75": float(np.quantile(arr, 0.75)),
    }


def _interpolate_linear(
    start: np.ndarray, end: np.ndarray, num_frames: int
) -> np.ndarray:
    """Linear interpolation between start and end over num_frames.

    Returns array where result[0] == start and result[-1] == end.
    """
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    if num_frames <= 1:
        return start[None, ...]
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    for _ in range(start.ndim):
        t = t[..., None]
    result = ((1.0 - t) * start + t * end).astype(np.float32)
    result[0] = start
    result[-1] = end
    return result


def _interpolate_quaternions_slerp(
    start_quat: np.ndarray, end_quat: np.ndarray, num_frames: int
) -> np.ndarray:
    """SLERP interpolation between two quaternions (XYZW format) over num_frames.

    Args:
        start_quat: shape [4] in XYZW format
        end_quat: shape [4] in XYZW format
        num_frames: number of interpolation frames

    Returns:
        shape [num_frames, 4] in XYZW format, with result[0] == start_quat
        and result[-1] == end_quat.
    """
    start_quat = np.asarray(start_quat, dtype=np.float32)
    end_quat = np.asarray(end_quat, dtype=np.float32)
    if num_frames <= 1:
        return start_quat[None, ...]
    rotations = sRot.from_quat([start_quat, end_quat])
    slerp = Slerp([0.0, 1.0], rotations)
    t = np.linspace(0.0, 1.0, num_frames)
    result = slerp(t).as_quat().astype(np.float32)
    result[0] = start_quat
    result[-1] = end_quat
    return result


def _extract_yaw_only_quat(quat: np.ndarray) -> np.ndarray:
    """Extract yaw-only quaternion (XYZW format) from a full quaternion.

    Args:
        quat: shape [4] in XYZW format

    Returns:
        shape [4] in XYZW format with only yaw rotation
    """
    rot = sRot.from_quat(quat)
    euler = rot.as_euler("xyz", degrees=False)
    yaw_only_euler = np.array([0.0, 0.0, euler[2]])
    yaw_only_rot = sRot.from_euler("xyz", yaw_only_euler, degrees=False)
    return yaw_only_rot.as_quat().astype(np.float32)


def _dof_to_pose_aa(
    dof_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    humanoid_fk: "HumanoidBatch",
    num_augment_joints: int,
) -> np.ndarray:
    """Convert DOF positions and root rotation to pose axis-angle.

    Args:
        dof_pos: shape [T, num_dofs]
        root_rot_xyzw: shape [T, 4] in XYZW format
        humanoid_fk: HumanoidBatch instance
        num_augment_joints: number of augmented joints

    Returns:
        pose_aa: shape [T, num_bodies + num_augment_joints, 3]
    """
    dof_t = torch.as_tensor(dof_pos, dtype=torch.float32)
    T = dof_t.shape[0]
    root_quat_wxyz = rot_conv.xyzw_to_wxyz(
        torch.as_tensor(root_rot_xyzw, dtype=torch.float32)
    )
    root_aa = rot_conv.quaternion_to_axis_angle(root_quat_wxyz)
    joint_aa = humanoid_fk.dof_axis * dof_t[:, :, None]
    pose_aa = torch.cat(
        [
            root_aa[:, None, :],
            joint_aa,
            torch.zeros((T, num_augment_joints, 3), dtype=torch.float32),
        ],
        dim=1,
    )
    return pose_aa.numpy().astype(np.float32)


def _compute_fk_motion(
    dof_pos: np.ndarray,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    humanoid_fk: "HumanoidBatch",
    num_augment_joints: int,
    fps: float,
) -> Dict[str, np.ndarray]:
    """Compute all motion arrays from dof_pos, root_pos, and root_rot via FK.

    Args:
        dof_pos: shape [T, num_dofs]
        root_pos: shape [T, 3]
        root_rot_xyzw: shape [T, 4] in XYZW format
        humanoid_fk: HumanoidBatch instance
        num_augment_joints: number of augmented joints
        fps: frames per second

    Returns:
        Dict with ref_dof_pos, ref_dof_vel, ref_global_translation,
        ref_global_rotation_quat, ref_global_velocity, ref_global_angular_velocity,
        frame_flag
    """
    T = dof_pos.shape[0]
    dt = 1.0 / fps
    pose_aa = _dof_to_pose_aa(
        dof_pos, root_rot_xyzw, humanoid_fk, num_augment_joints
    )
    pose_aa_t = torch.as_tensor(pose_aa, dtype=torch.float32)
    root_pos_t = torch.as_tensor(root_pos, dtype=torch.float32)
    fk_result = humanoid_fk.fk_batch(
        pose_aa_t[None, ...],
        root_pos_t[None, ...],
        return_full=True,
        dt=dt,
    )
    frame_flag = np.ones(T, dtype=np.int32)
    frame_flag[0] = 0
    frame_flag[-1] = 2
    return {
        "ref_dof_pos": fk_result.dof_pos.squeeze(0).numpy().astype(np.float32),
        "ref_dof_vel": fk_result.dof_vels.squeeze(0).numpy().astype(np.float32),
        "ref_global_translation": fk_result.global_translation.squeeze(0)
        .numpy()
        .astype(np.float32),
        "ref_global_rotation_quat": fk_result.global_rotation.squeeze(0)
        .numpy()
        .astype(np.float32),
        "ref_global_velocity": fk_result.global_velocity.squeeze(0)
        .numpy()
        .astype(np.float32),
        "ref_global_angular_velocity": fk_result.global_angular_velocity.squeeze(
            0
        )
        .numpy()
        .astype(np.float32),
        "frame_flag": frame_flag,
    }


@dataclass
class ProcessedClip:
    motion_key: str
    metadata: Dict[str, Any]
    arrays: Dict[str, np.ndarray]


class HoloMotionPreprocessor:
    """
    Composable preprocessing pipeline operating on standardized HoloMotion NPZ clips.

    Supports per-clip stages like slicing and Butterworth filtering,
    plus dataset-level kinematic tagging.
    """

    def __init__(
        self,
        slicing_cfg: Optional[DictConfig] = None,
        filtering_cfg: Optional[DictConfig] = None,
        tagging_cfg: Optional[DictConfig] = None,
        padding_cfg: Optional[DictConfig] = None,
        pipeline: Optional[List[str]] = None,
    ) -> None:
        self.slicing_cfg = slicing_cfg
        self.filtering_cfg = filtering_cfg
        self.tagging_cfg = tagging_cfg
        self.padding_cfg = padding_cfg
        self.pipeline = self._resolve_pipeline(pipeline)
        self._humanoid_fk: Optional[HumanoidBatch] = None
        self._robot_cfg: Optional[DictConfig] = None

    def _resolve_pipeline(self, pipeline: Optional[List[str]]) -> List[str]:
        if pipeline is not None:
            return list(pipeline)
        return []

    def process_clip(self, clip: ProcessedClip) -> List[ProcessedClip]:
        clips = [clip]
        logger.debug(
            f"Processing clip '{clip.motion_key}' with pipeline: {self.pipeline}"
        )
        for stage in self.pipeline:
            logger.debug(f"Applying stage: {stage}")
            if stage in ("slicing", "slice"):
                next_clips: List[ProcessedClip] = []
                for c in clips:
                    next_clips.extend(self._apply_slicing(c))
                clips = next_clips
                logger.debug(f"After slicing: {len(clips)} clips")
            elif stage in (
                "apply_butterworth_filter",
                "filtering",
                "butterworth_filter",
            ):
                clips = [self._apply_filtering(c) for c in clips]
                logger.debug(
                    f"After apply_butterworth_filter: {len(clips)} clips"
                )
            elif stage == "filename_as_motionkey":
                clips = [self._apply_filename_as_motionkey(c) for c in clips]
                logger.debug(
                    f"After filename_as_motionkey: {len(clips)} clips"
                )
            elif stage == "legacy_to_ref_keys":
                clips = [self._apply_legacy_to_ref_keys(c) for c in clips]
                logger.debug(f"After legacy_to_ref_keys: {len(clips)} clips")
            elif stage == "add_legacy_keys":
                clips = [self._apply_add_legacy_keys(c) for c in clips]
                logger.debug(f"After add_legacy_keys: {len(clips)} clips")
            elif stage == "add_padding":
                clips = [self._apply_add_padding(c) for c in clips]
                logger.debug(f"After add_padding: {len(clips)} clips")
            else:
                logger.warning(
                    f"Unknown preprocessing stage '{stage}' ignored."
                )
        return clips

    def _apply_slicing(self, clip: ProcessedClip) -> List[ProcessedClip]:
        cfg = self.slicing_cfg
        if cfg is None:
            logger.warning(
                "Slicing requested but slicing_cfg is None - skipping slicing"
            )
            return [clip]

        window_size = int(getattr(cfg, "window_size", 0))
        overlap = int(getattr(cfg, "overlap", 0))
        seq_len = int(clip.metadata.get("num_frames", 0))
        if seq_len <= 0:
            return [clip]

        slice_specs = compute_slices(seq_len, window_size, overlap)
        if not slice_specs:
            return [clip]

        fps = float(clip.metadata.get("motion_fps", 0.0))
        raw_motion_key = str(
            clip.metadata.get(
                "raw_motion_key", clip.metadata.get("motion_key", "")
            )
        )
        base_motion_key = str(clip.metadata.get("motion_key", raw_motion_key))
        arrays = clip.arrays

        out_clips: List[ProcessedClip] = []
        for s, e in slice_specs:
            arrays_window: Dict[str, np.ndarray] = {}
            for k, v in arrays.items():
                if (
                    isinstance(v, np.ndarray)
                    and v.ndim >= 1
                    and v.shape[0] == seq_len
                ):
                    arrays_window[k] = v[s:e]
                else:
                    arrays_window[k] = v

            num_frames = int(e - s)
            if num_frames <= 0:
                continue

            wallclock_len = float(num_frames - 1) / fps if fps > 0.0 else 0.0
            if s == 0 and e == seq_len:
                motion_key = base_motion_key
            else:
                motion_key = f"{base_motion_key}_s{s}_e{e}"

            meta = dict(clip.metadata)
            meta["motion_key"] = motion_key
            meta["raw_motion_key"] = raw_motion_key
            meta["num_frames"] = num_frames
            meta["wallclock_len"] = wallclock_len
            meta["slice_start"] = int(s)
            meta["slice_end"] = int(e)
            out_clips.append(
                ProcessedClip(
                    motion_key=motion_key,
                    metadata=meta,
                    arrays=arrays_window,
                )
            )
        return out_clips

    def _apply_filtering(self, clip: ProcessedClip) -> ProcessedClip:
        cfg = self.filtering_cfg
        if cfg is None:
            logger.warning(
                "Filtering requested but filtering_cfg is None - skipping filtering"
            )
            return clip

        fps = float(clip.metadata.get("motion_fps", 0.0))
        cutoff = float(getattr(cfg, "butter_cutoff_hz", 0.0))
        order = int(getattr(cfg, "butter_order", 4))
        ft = butterworth_filter_ref_arrays(
            clip.arrays, fps=fps, cutoff_hz=cutoff, order=order
        )
        arrays = dict(clip.arrays)
        arrays.update(ft)
        return ProcessedClip(
            motion_key=clip.motion_key,
            metadata=clip.metadata,
            arrays=arrays,
        )

    def _apply_filename_as_motionkey(
        self, clip: ProcessedClip
    ) -> ProcessedClip:
        filename = clip.metadata.get("source_filename", None)
        if filename is None:
            logger.warning(
                "filename_as_motionkey requested but source_filename not found in metadata - skipping"
            )
            return clip

        new_motion_key = str(filename)
        meta = dict(clip.metadata)
        meta["motion_key"] = new_motion_key
        if "raw_motion_key" not in meta:
            meta["raw_motion_key"] = clip.motion_key

        return ProcessedClip(
            motion_key=new_motion_key,
            metadata=meta,
            arrays=clip.arrays,
        )

    def _apply_add_legacy_keys(self, clip: ProcessedClip) -> ProcessedClip:
        """Add deprecated legacy keys for backward compatibility.

        Maps ref_* keys to legacy unprefixed keys according to spec:
        - ref_dof_pos -> dof_pos
        - ref_dof_vel -> dof_vels
        - ref_global_translation -> global_translation
        - ref_global_rotation_quat -> global_rotation_quat
        - ref_global_velocity -> global_velocity
        - ref_global_angular_velocity -> global_angular_velocity
        """
        ref_to_legacy = {
            "ref_dof_pos": "dof_pos",
            "ref_dof_vel": "dof_vels",
            "ref_global_translation": "global_translation",
            "ref_global_rotation_quat": "global_rotation_quat",
            "ref_global_velocity": "global_velocity",
            "ref_global_angular_velocity": "global_angular_velocity",
        }

        arrays = dict(clip.arrays)
        for ref_key, legacy_key in ref_to_legacy.items():
            if ref_key in arrays:
                if legacy_key not in arrays:
                    arrays[legacy_key] = arrays[ref_key].copy()
                    logger.debug(
                        f"Added legacy key '{legacy_key}' from '{ref_key}'"
                    )
                else:
                    logger.debug(
                        f"Legacy key '{legacy_key}' already exists, skipping"
                    )

        return ProcessedClip(
            motion_key=clip.motion_key,
            metadata=clip.metadata,
            arrays=arrays,
        )

    def _apply_legacy_to_ref_keys(self, clip: ProcessedClip) -> ProcessedClip:
        """Add new ref_* keys from legacy unprefixed keys.

        Maps legacy keys to ref_* keys according to spec while keeping the
        original legacy arrays:
        - dof_pos -> ref_dof_pos
        - dof_vels -> ref_dof_vel
        - global_translation -> ref_global_translation
        - global_rotation_quat -> ref_global_rotation_quat
        - global_velocity -> ref_global_velocity
        - global_angular_velocity -> ref_global_angular_velocity
        """
        legacy_to_ref = {
            "dof_pos": "ref_dof_pos",
            "dof_vels": "ref_dof_vel",
            "global_translation": "ref_global_translation",
            "global_rotation_quat": "ref_global_rotation_quat",
            "global_velocity": "ref_global_velocity",
            "global_angular_velocity": "ref_global_angular_velocity",
        }

        arrays = dict(clip.arrays)
        for legacy_key, ref_key in legacy_to_ref.items():
            if legacy_key in arrays:
                if ref_key not in arrays:
                    arrays[ref_key] = arrays[legacy_key].copy()
                    logger.debug(
                        f"Added ref key '{ref_key}' from legacy key '{legacy_key}'"
                    )
                else:
                    logger.debug(
                        f"Ref key '{ref_key}' already exists, skipping"
                    )

        return ProcessedClip(
            motion_key=clip.motion_key,
            metadata=clip.metadata,
            arrays=arrays,
        )

    def _get_humanoid_fk(self) -> HumanoidBatch:
        """Lazy-load and cache HumanoidBatch for FK computation."""
        if self._humanoid_fk is not None:
            return self._humanoid_fk
        cfg = self.padding_cfg
        robot_config_path = str(getattr(cfg, "robot_config_path", ""))
        self._robot_cfg = OmegaConf.load(robot_config_path)
        self._humanoid_fk = HumanoidBatch(self._robot_cfg.robot)
        return self._humanoid_fk

    def _get_default_dof_pos(self) -> np.ndarray:
        """Get default DOF positions from robot config."""
        robot_cfg = self._robot_cfg.robot
        dof_names = list(robot_cfg.dof_names)
        init_state = robot_cfg.get("init_state", {})
        default_angles = init_state.get("default_joint_angles", {})
        default_dof = np.zeros(len(dof_names), dtype=np.float32)
        for i, name in enumerate(dof_names):
            default_dof[i] = float(default_angles.get(name, 0.0))
        return default_dof

    def _apply_add_padding(self, clip: ProcessedClip) -> ProcessedClip:
        """Add transition and static padding to the motion clip.

        Adds stand-still padding at default pose before and after the motion,
        with smooth transitions between default pose and the motion's first/last
        frames. Recalculates all states from root pos, rot and dof pos via FK.
        """
        cfg = self.padding_cfg
        if cfg is None:
            logger.warning(
                "Padding requested but padding_cfg is None - skipping padding"
            )
            return clip

        fps = float(clip.metadata.get("motion_fps", 50.0))
        stand_still_time = float(getattr(cfg, "stand_still_time", 1.0))
        transition_time = float(getattr(cfg, "transition_time", 1.5))
        robot_config_path = str(getattr(cfg, "robot_config_path", ""))
        if not robot_config_path:
            raise ValueError(
                "robot_config_path must be specified in padding_cfg"
            )

        stand_still_frames = max(1, int(stand_still_time * fps))
        transition_frames = max(1, int(transition_time * fps))

        humanoid_fk = self._get_humanoid_fk()
        default_dof = self._get_default_dof_pos()
        extend_config = self._robot_cfg.robot.get("extend_config", [])
        num_augment = len(extend_config) if extend_config else 0

        # Get root offset from HumanoidBatch (usually from MJCF root body pos)
        # self._offsets is [1, num_bodies, 3]
        # root_offset = humanoid_fk._offsets[0, 0].cpu().numpy()

        arrays = clip.arrays
        dof_pos = arrays.get("ref_dof_pos", arrays.get("dof_pos"))
        global_trans = arrays.get(
            "ref_global_translation", arrays.get("global_translation")
        )
        global_rot = arrays.get(
            "ref_global_rotation_quat", arrays.get("global_rotation_quat")
        )
        if dof_pos is None or global_trans is None or global_rot is None:
            raise ValueError(
                "Missing required arrays for padding: ref_dof_pos, "
                "ref_global_translation, or ref_global_rotation_quat"
            )

        T_orig = dof_pos.shape[0]
        dof_pos = dof_pos.astype(np.float32, copy=True)
        root_pos = global_trans[:, 0, :].astype(np.float32, copy=True)
        root_rot = global_rot[:, 0, :].astype(np.float32, copy=True)

        first_dof = dof_pos[0].copy()
        last_dof = dof_pos[-1].copy()
        first_root_pos = root_pos[0].copy()
        last_root_pos = root_pos[-1].copy()
        first_root_rot = root_rot[0].copy()
        last_root_rot = root_rot[-1].copy()

        logger.debug(
            f"Padding: T_orig={T_orig}, first_root_pos={first_root_pos}, "
            f"last_root_pos={last_root_pos}"
        )
        logger.debug(
            f"Padding: first_dof[:3]={first_dof[:3]}, last_dof[:3]={last_dof[:3]}"
        )
        logger.debug(
            f"Padding: original dof_pos[-1][:3]={dof_pos[-1][:3]}, "
            f"original root_pos[-1]={root_pos[-1]}, original root_rot[-1]={root_rot[-1]}"
        )

        first_yaw_quat = _extract_yaw_only_quat(first_root_rot)
        last_yaw_quat = _extract_yaw_only_quat(last_root_rot)

        start_stand_dof = np.tile(default_dof, (stand_still_frames, 1))
        start_trans_dof = _interpolate_linear(
            default_dof, first_dof, transition_frames
        )
        end_trans_dof = _interpolate_linear(
            last_dof, default_dof, transition_frames
        )
        end_stand_dof = np.tile(default_dof, (stand_still_frames, 1))

        start_stand_root_pos = np.tile(first_root_pos, (stand_still_frames, 1))
        start_trans_root_pos = _interpolate_linear(
            first_root_pos, first_root_pos, transition_frames
        )
        end_trans_root_pos = _interpolate_linear(
            last_root_pos, last_root_pos, transition_frames
        )
        end_stand_root_pos = np.tile(last_root_pos, (stand_still_frames, 1))

        start_stand_root_rot = np.tile(first_yaw_quat, (stand_still_frames, 1))
        start_trans_root_rot = _interpolate_quaternions_slerp(
            first_yaw_quat, first_root_rot, transition_frames
        )
        end_trans_root_rot = _interpolate_quaternions_slerp(
            last_root_rot, last_yaw_quat, transition_frames
        )
        end_stand_root_rot = np.tile(last_yaw_quat, (stand_still_frames, 1))

        # Construct full sequence of inputs
        full_dof = np.concatenate(
            [
                start_stand_dof,
                start_trans_dof,
                dof_pos,
                end_trans_dof,
                end_stand_dof,
            ],
            axis=0,
        )
        full_root_pos = np.concatenate(
            [
                start_stand_root_pos,
                start_trans_root_pos,
                root_pos,
                end_trans_root_pos,
                end_stand_root_pos,
            ],
            axis=0,
        )
        full_root_rot = np.concatenate(
            [
                start_stand_root_rot,
                start_trans_root_rot,
                root_rot,
                end_trans_root_rot,
                end_stand_root_rot,
            ],
            axis=0,
        )

        # Compute FK for the entire sequence to ensure continuity
        new_arrays = _compute_fk_motion(
            full_dof,
            full_root_pos,
            full_root_rot,
            humanoid_fk,
            num_augment,
            fps,
        )

        T_new = full_dof.shape[0]
        wallclock_len = float(T_new - 1) / fps if fps > 0.0 else 0.0
        meta = dict(clip.metadata)
        meta["num_frames"] = T_new
        meta["wallclock_len"] = wallclock_len
        meta["padding_stand_still_frames"] = stand_still_frames
        meta["padding_transition_frames"] = transition_frames
        meta["original_num_frames"] = T_orig

        return ProcessedClip(
            motion_key=clip.motion_key,
            metadata=meta,
            arrays=new_arrays,
        )

    def process_npz_file(self, npz_path: Path) -> List[ProcessedClip]:
        with np.load(npz_path, allow_pickle=False) as data:
            if "metadata" not in data:
                raise KeyError(f"'metadata' missing in NPZ: {npz_path}")
            meta_text = str(data["metadata"])
            metadata = json.loads(meta_text)
            motion_key = str(metadata["motion_key"])
            arrays: Dict[str, np.ndarray] = {}
            for k in data.files:
                if k == "metadata":
                    continue
                arrays[k] = np.array(data[k], copy=False)

        filename_without_ext = npz_path.stem
        metadata["source_filename"] = filename_without_ext

        base_clip = ProcessedClip(
            motion_key=motion_key,
            metadata=metadata,
            arrays=arrays,
        )
        return self.process_clip(base_clip)

    def run_on_directory(
        self,
        src_root: Path,
        out_root: Path,
        use_ray: bool = False,
        num_workers: int = 0,
    ) -> None:
        if src_root.is_dir():
            if (src_root / "clips").is_dir():
                clips_src = src_root / "clips"
            else:
                clips_src = src_root
        else:
            raise ValueError(f"Source root is not a directory: {src_root}")

        clips_dst = out_root / "clips"
        clips_dst.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in clips_src.rglob("*.npz") if p.is_file()])
        if not files:
            logger.info("No NPZ files found to process.")
            return

        if use_ray:
            if num_workers <= 0:
                available_cpus = int(ray.available_resources().get("CPU", 1))
                effective_workers = max(1, available_cpus)
            else:
                effective_workers = num_workers
            self._run_on_directory_ray(files, clips_dst, effective_workers)
        else:
            self._run_on_directory_sequential(files, clips_dst)

    def _run_on_directory_sequential(
        self, files: List[Path], clips_dst: Path
    ) -> None:
        logger.info(f"Processing {len(files)} NPZ files sequentially")
        logger.info(f"Pipeline stages to apply: {self.pipeline}")
        total_input_clips = 0
        total_output_clips = 0
        for p in tqdm(files, desc="HoloMotion preprocess NPZ", unit="file"):
            clips = self.process_npz_file(p)
            total_input_clips += 1
            for clip in clips:
                total_output_clips += 1
                out_name = f"{clip.motion_key}.npz"
                out_path = clips_dst / out_name
                metadata_json = json.dumps(clip.metadata)
                np.savez_compressed(
                    out_path, metadata=metadata_json, **clip.arrays
                )
        logger.info(
            f"Processed {total_input_clips} input files into {total_output_clips} output clips"
        )

    def _run_on_directory_ray(
        self, files: List[Path], clips_dst: Path, num_workers: int
    ) -> None:
        if num_workers <= 0:
            available_cpus = int(ray.available_resources().get("CPU", 1))
            num_actors = min(len(files), max(1, available_cpus))
        else:
            num_actors = min(len(files), num_workers)
        actors = [
            PreprocessorActor.remote(
                slicing_cfg=self.slicing_cfg,
                filtering_cfg=self.filtering_cfg,
                tagging_cfg=self.tagging_cfg,
                padding_cfg=self.padding_cfg,
                pipeline=self.pipeline,
            )
            for _ in range(num_actors)
        ]

        pending = {}
        next_idx = 0
        for i in range(min(num_actors, len(files))):
            p = files[next_idx]
            next_idx += 1
            ref = actors[i].process_npz_file.remote(str(p))
            pending[ref] = i

        total_outputs = 0
        with tqdm(
            total=len(files), desc="Ray: HoloMotion preprocess NPZ"
        ) as pbar:
            while pending:
                done, _ = ray.wait(list(pending.keys()), num_returns=1)
                ref = done[0]
                actor_idx = pending.pop(ref)
                clips = ray.get(ref)
                for clip in clips:
                    out_name = f"{clip.motion_key}.npz"
                    out_path = clips_dst / out_name
                    metadata_json = json.dumps(clip.metadata)
                    np.savez_compressed(
                        out_path, metadata=metadata_json, **clip.arrays
                    )
                    total_outputs += 1
                pbar.update(1)
                if next_idx < len(files):
                    p = files[next_idx]
                    next_idx += 1
                    new_ref = actors[actor_idx].process_npz_file.remote(str(p))
                    pending[new_ref] = actor_idx

        logger.info(f"Processed {total_outputs} clips total.")

    def tag_directory(self, clips_dir: Path, tags_path: Path) -> None:
        files = sorted([p for p in clips_dir.rglob("*.npz") if p.is_file()])

        clip_info: Dict[str, Dict[str, Dict[str, float]]] = {}
        all_speed: List[np.ndarray] = []
        all_wnorm: List[np.ndarray] = []
        all_zrel: List[np.ndarray] = []
        all_jerk: List[np.ndarray] = []

        for f in tqdm(files, desc="Tagging kinematics", unit="file"):
            with np.load(f, allow_pickle=True) as data:
                meta_text = str(data["metadata"])
                meta = json.loads(meta_text)
                key = str(meta["motion_key"])
                fps = float(meta["motion_fps"])

                def pick(name: str) -> np.ndarray:
                    if f"ft_ref_{name}" in data:
                        return np.array(data[f"ft_ref_{name}"], copy=False)
                    if f"ref_{name}" in data:
                        return np.array(data[f"ref_{name}"], copy=False)
                    return np.array([], dtype=np.float32)

                gv = pick("global_velocity")
                ga = pick("global_angular_velocity")
                gt = pick("global_translation")

                if gv.size > 0:
                    root_vel = gv[:, 0, :]
                    speed = np.linalg.norm(root_vel, axis=1)
                else:
                    speed = np.array([], dtype=float)

                if ga.size > 0:
                    root_w = ga[:, 0, :]
                    wnorm = np.linalg.norm(root_w, axis=1)
                else:
                    wnorm = np.array([], dtype=float)

                if gt.size > 0:
                    root_pos_z = gt[:, 0, 2]
                    z_rel = np.abs(root_pos_z - float(root_pos_z[0]))
                else:
                    z_rel = np.array([], dtype=float)

                if gv.shape[0] >= 3:
                    dt = 1.0 / fps if fps > 0.0 else 0.0
                    a = (
                        np.diff(gv, axis=0) / dt
                        if dt > 0.0
                        else np.zeros_like(gv)
                    )
                    j = (
                        np.diff(a, axis=0) / dt
                        if dt > 0.0
                        else np.zeros_like(a)
                    )
                    jn = np.linalg.norm(j, axis=2)
                else:
                    jn = np.array([], dtype=float)

                clip_info[key] = {
                    "root_linear_speed": _summary(speed),
                    "root_angular_speed": _summary(wnorm),
                    "root_delta_z": _summary(z_rel),
                    "jerk": _summary(jn),
                }

                if speed.size > 0:
                    all_speed.append(speed.astype(float))
                if wnorm.size > 0:
                    all_wnorm.append(wnorm.astype(float))
                if z_rel.size > 0:
                    all_zrel.append(z_rel.astype(float))
                if jn.size > 0:
                    all_jerk.append(jn.astype(float))

        speed_cat = (
            np.concatenate([a for a in all_speed if a.size > 0], axis=0)
            if len(all_speed) > 0
            else np.array([], dtype=float)
        )
        wnorm_cat = (
            np.concatenate([a for a in all_wnorm if a.size > 0], axis=0)
            if len(all_wnorm) > 0
            else np.array([], dtype=float)
        )
        zrel_cat = (
            np.concatenate([a for a in all_zrel if a.size > 0], axis=0)
            if len(all_zrel) > 0
            else np.array([], dtype=float)
        )
        jerk_cat = (
            np.concatenate([a for a in all_jerk if a.size > 0], axis=0)
            if len(all_jerk) > 0
            else np.array([], dtype=float)
        )

        result = {
            "dataset_stats": {
                "root_linear_speed": _ds_summary(speed_cat),
                "root_angular_speed": _ds_summary(wnorm_cat),
                "root_delta_z": _ds_summary(zrel_cat),
                "jerk": _ds_summary(jerk_cat),
            },
            "clip_info": clip_info,
        }
        with open(tags_path, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        logger.info(f"Wrote kinematic tags JSON to: {tags_path}")


@ray.remote
class PreprocessorActor:
    """Ray actor that holds a HoloMotionPreprocessor instance for parallel processing."""

    def __init__(
        self,
        slicing_cfg: Optional[DictConfig] = None,
        filtering_cfg: Optional[DictConfig] = None,
        tagging_cfg: Optional[DictConfig] = None,
        padding_cfg: Optional[DictConfig] = None,
        pipeline: Optional[List[str]] = None,
    ) -> None:
        self.preprocessor = HoloMotionPreprocessor(
            slicing_cfg=slicing_cfg,
            filtering_cfg=filtering_cfg,
            tagging_cfg=tagging_cfg,
            padding_cfg=padding_cfg,
            pipeline=pipeline,
        )
        logger.debug(
            f"PreprocessorActor initialized with pipeline: {self.preprocessor.pipeline}"
        )

    def process_npz_file(self, npz_path_str: str) -> List[ProcessedClip]:
        npz_path = Path(npz_path_str)
        return self.preprocessor.process_npz_file(npz_path)


@hydra.main(
    config_path="../../config",
    config_name="motion_retargeting/holomotion_preprocess",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)

    src_root = Path(str(cfg.io.src_root)).expanduser().resolve()
    out_root = Path(str(cfg.io.out_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Dump resolved config used
    with open(out_root / "config_used.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Parse pipeline
    pipeline_cfg = cfg.get("preprocess", None)
    logger.debug(f"Raw preprocess config: {pipeline_cfg}")
    pipeline = None
    if pipeline_cfg is not None:
        pipeline_val = pipeline_cfg.get("pipeline", None)
        logger.debug(
            f"Raw pipeline value: {pipeline_val} (type: {type(pipeline_val)})"
        )
        if pipeline_val is not None:
            if isinstance(pipeline_val, (list, tuple, ListConfig)):
                pipeline = [str(s) for s in pipeline_val]
            elif isinstance(pipeline_val, str):
                import ast

                pipeline = ast.literal_eval(pipeline_val)
            else:
                logger.warning(
                    f"Unexpected pipeline type: {type(pipeline_val)}, value: {pipeline_val}"
                )
                pipeline = []
        else:
            logger.debug("pipeline_val is None")
    else:
        logger.debug("preprocess config is None")

    # Separate per-clip stages from dataset-level stages
    per_clip_pipeline = (
        [s for s in pipeline if s != "tagging"] if pipeline else []
    )
    tagging_enabled = pipeline and "tagging" in pipeline

    logger.info("=" * 80)
    logger.info("Preprocessing Configuration:")
    logger.info(f"  Source directory: {src_root}")
    logger.info(f"  Output directory: {out_root}")
    if pipeline:
        logger.info(f"  Pipeline stages: {pipeline}")
        logger.info(f"  Number of stages: {len(pipeline)}")
        for i, stage in enumerate(pipeline, 1):
            logger.info(f"    {i}. {stage}")
        if tagging_enabled:
            logger.info(
                "  Note: 'tagging' is a dataset-level operation and will run after all clips are processed"
            )
    else:
        logger.warning(
            "  No preprocessing pipeline specified - no processors will be applied!"
        )
    logger.info("=" * 80)

    use_ray = bool(cfg.get("ray", {}).get("enabled", False))
    num_workers = int(cfg.get("ray", {}).get("num_workers", 0))
    ray_address = str(cfg.get("ray", {}).get("ray_address", ""))

    if use_ray:
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.ERROR)
        os.environ.setdefault("RAY_BACKEND_LOG_LEVEL", "error")

        if ray_address:
            ray.init(
                address=ray_address,
                ignore_reinit_error=True,
                log_to_driver=False,
                include_dashboard=False,
                logging_level=logging.ERROR,
            )
            if num_workers <= 0:
                num_workers = int(ray.available_resources().get("CPU", 1))
        else:
            num_cpus = None if num_workers <= 0 else num_workers
            ray.init(
                num_cpus=num_cpus,
                ignore_reinit_error=True,
                log_to_driver=False,
                include_dashboard=False,
                logging_level=logging.ERROR,
            )
            if num_workers <= 0:
                num_workers = int(ray.available_resources().get("CPU", 1))

    preprocessor = HoloMotionPreprocessor(
        slicing_cfg=cfg.slicing,
        filtering_cfg=cfg.filtering,
        tagging_cfg=cfg.tagging,
        padding_cfg=cfg.get("padding", None),
        pipeline=per_clip_pipeline if per_clip_pipeline else None,
    )

    logger.info(
        f"Preprocessor initialized with pipeline: {preprocessor.pipeline}"
    )
    logger.info(
        f"  Slicing config present: {preprocessor.slicing_cfg is not None}"
    )
    logger.info(
        f"  Filtering config present: {preprocessor.filtering_cfg is not None}"
    )
    logger.info(
        f"  Tagging config present: {preprocessor.tagging_cfg is not None}"
    )

    preprocessor.run_on_directory(
        src_root, out_root, use_ray=use_ray, num_workers=num_workers
    )

    if use_ray:
        ray.shutdown()

    if tagging_enabled:
        if str(cfg.tagging.output_json_path):
            tags_path = (
                Path(str(cfg.tagging.output_json_path)).expanduser().resolve()
            )
        else:
            tags_path = out_root / "kinematic_tags.json"
        clips_dir = out_root / "clips"
        preprocessor.tag_directory(clips_dir, tags_path)


if __name__ == "__main__":
    main()
