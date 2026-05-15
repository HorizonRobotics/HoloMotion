"""Offline motion clip loading helpers for the 29DOF policy node."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class LoadedMotionClip:
    """Validated view of one legacy motion clip dict."""

    dof_pos: np.ndarray
    dof_vel: np.ndarray
    global_translation: np.ndarray
    global_rotation_quat: np.ndarray
    global_velocity: np.ndarray
    global_angular_velocity: np.ndarray
    n_frames: int


def list_motion_clip_files(motion_clips_dir: str) -> list[str]:
    """Return sorted .npz motion clip file names, matching Phase 3B behavior."""
    return sorted(
        filename
        for filename in os.listdir(motion_clips_dir)
        if filename.endswith(".npz")
    )


def load_motion_clip(motion_path: str) -> dict:
    """Load one .npz file into the legacy policy_node motion dict shape."""
    motion_data_dict = dict(np.load(motion_path, allow_pickle=True))
    return {
        "dof_pos": motion_data_dict["ref_dof_pos"],
        "dof_vel": motion_data_dict["ref_dof_vel"],
        "global_translation": motion_data_dict["ref_global_translation"],
        "global_rotation_quat": motion_data_dict["ref_global_rotation_quat"],
        "global_velocity": motion_data_dict["ref_global_velocity"],
        "global_angular_velocity": motion_data_dict["ref_global_angular_velocity"],
        "n_frames": motion_data_dict["ref_dof_pos"].shape[0],
    }


def load_motion_clips(motion_clips_dir: str, motion_clip_files: list[str]) -> list[dict]:
    """Load sorted clip files from a directory into legacy motion dicts."""
    return [
        load_motion_clip(os.path.join(motion_clips_dir, motion_clip_file))
        for motion_clip_file in motion_clip_files
    ]


def validate_loaded_motion_clip(
    motion_clip: dict,
    *,
    expected_dof_count: int,
    expected_body_count: int,
) -> LoadedMotionClip:
    """Validate one legacy motion dict and return a typed view without copying arrays."""
    loaded = LoadedMotionClip(
        dof_pos=motion_clip["dof_pos"],
        dof_vel=motion_clip["dof_vel"],
        global_translation=motion_clip["global_translation"],
        global_rotation_quat=motion_clip["global_rotation_quat"],
        global_velocity=motion_clip["global_velocity"],
        global_angular_velocity=motion_clip["global_angular_velocity"],
        n_frames=motion_clip["n_frames"],
    )

    if loaded.dof_pos is None or loaded.dof_vel is None:
        raise ValueError("Motion clip is missing ref_dof_pos/ref_dof_vel arrays")
    if loaded.global_translation is None or loaded.global_rotation_quat is None:
        raise ValueError(
            "Motion clip is missing ref_global_translation/ref_global_rotation_quat arrays"
        )
    if int(loaded.dof_pos.shape[1]) != int(expected_dof_count):
        raise ValueError(
            "ref_dof_pos DOF dimension mismatch: "
            f"ref_dof_pos.shape[1]={int(loaded.dof_pos.shape[1])} "
            f"but len(dof_names_ref_motion)={int(expected_dof_count)}"
        )
    if int(loaded.global_translation.shape[1]) != int(expected_body_count):
        raise ValueError(
            "ref_global_translation body dimension mismatch: "
            f"ref_raw_bodylink_pos.shape[1]={int(loaded.global_translation.shape[1])} "
            f"but len(motion_config.robot.body_names)={int(expected_body_count)}"
        )
    return loaded


def select_motion_clip_index(
    current_index: int,
    clip_count: int,
    command: Literal["previous", "next", "first", "last"],
) -> int:
    """Return the selected clip index using the Phase 3B button semantics."""
    if clip_count <= 0:
        return current_index
    if command == "previous":
        return (current_index - 1) % clip_count
    if command == "next":
        return (current_index + 1) % clip_count
    if command == "first":
        return 0
    if command == "last":
        return clip_count - 1
    raise ValueError(f"Unsupported motion clip selection command: {command}")
