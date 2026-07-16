"""Shared online and training API for HoloRetarget."""

from __future__ import annotations

import time

import numpy as np

from ._engine_impl import HoloRetargetRunner
from .config import DEFAULT_CONFIG, HoloRetargetConfig
from .schema import UNITREE_G1_29DOF_NAMES


class HoloRetargeter:
    """HoloMotion production retargeter.

    The only retarget path is ``Pico/XRoboToolkit body_poses[24,7] -> qpos[36]``.
    """

    def __init__(self, config: HoloRetargetConfig | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        # Training-side HoloSMPL conversion initializes this lazily when needed.
        self._smpl_adapter = None
        self._runner = self._create_runner(src_human="smplx")
        from ._gpu_targets import HoloPicoGpuTargetRunner

        self._gpu_target_runner = HoloPicoGpuTargetRunner(self)
        self.last_timing: dict[str, float] = {}
        self.last_body_poses: np.ndarray | None = None

    def _create_runner(self, src_human: str) -> HoloRetargetRunner:
        return HoloRetargetRunner(
            asset_root=str(self.config.resolved_asset_root),
            robot=self.config.robot,
            src_human=src_human,
            offset_to_ground=False,
            newton_iterations=self.config.newton_iterations,
            newton_cuda_graph=self.config.use_cuda_graph,
            newton_joint_limit_weight=self.config.joint_limit_weight,
            newton_max_joint_step=self.config.max_joint_step,
            newton_target_table=self.config.target_table,
            newton_robot_asset=self.config.robot_asset,
            newton_root_seed_mode=self.config.root_seed_mode,
            ground_calibration_frames=self.config.ground_calibration_frames,
            ground_height=self.config.ground_height,
            ground_lift_only=self.config.ground_lift_only,
            ground_calibration_mode=self.config.ground_calibration_mode,
            ground_target_scope=self.config.ground_target_scope,
            profile_timing=self.config.profile_timing,
        )

    @property
    def last_ground_offset_z(self) -> float:
        return float(getattr(self._runner, "last_ground_offset_z", 0.0))

    @property
    def dof_names(self) -> tuple[str, ...]:
        """Joint order used by the 29 values at ``qpos[7:]``."""

        return UNITREE_G1_29DOF_NAMES

    def reset_sequence(self) -> None:
        self._runner.reset_sequence()
        self._gpu_target_runner.reset_sequence()
        self.last_timing = {}
        self.last_body_poses = None

    def retarget_qpos_from_body_poses(
        self,
        body_poses: np.ndarray,
    ) -> np.ndarray:
        body_poses = np.asarray(body_poses, dtype=np.float32)
        if body_poses.shape != (24, 7):
            raise ValueError(
                f"body_poses must have shape (24, 7), got {body_poses.shape}"
            )
        t0 = time.perf_counter()
        reference_qpos = self._gpu_target_runner.retarget_qpos_from_body_poses(
            body_poses
        )
        self.last_body_poses = body_poses.copy()
        self.last_timing = dict(getattr(self._runner, "last_timing", {}))
        self.last_timing["holoretarget.total"] = time.perf_counter() - t0
        return reference_qpos

    def retarget_qpos_device_from_body_poses(self, body_poses: np.ndarray):
        """Retarget into a Warp CUDA array without copying qpos back to CPU."""
        body_poses = np.asarray(body_poses, dtype=np.float32)
        if body_poses.shape != (24, 7):
            raise ValueError(
                f"body_poses must have shape (24, 7), got {body_poses.shape}"
            )
        reference_qpos = (
            self._gpu_target_runner.retarget_qpos_device_from_body_poses(
                body_poses
            )
        )
        self.last_body_poses = body_poses.copy()
        self.last_timing = dict(getattr(self._runner, "last_timing", {}))
        return reference_qpos


__all__ = ["HoloRetargeter"]
