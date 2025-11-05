from dataclasses import MISSING
from typing import Sequence
import time

from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils


import isaaclab.utils.math as isaaclab_math
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.envs.mdp.actions import JointEffortActionCfg
from isaaclab.managers import (
    ActionTermCfg,
    CommandTerm,
    CommandTermCfg,
    EventTermCfg as EventTerm,
    ObservationGroupCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.markers import (
    VisualizationMarkers,
    VisualizationMarkersCfg,
)
from holomotion.src.training.h5_dataloader import (
    Hdf5MotionDataset,
    MotionClipBatchCache,
)
import os
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omegaconf import OmegaConf

from holomotion.src.utils.isaac_utils.rotations import (
    calc_heading_quat_inv,
    get_euler_xyz,
    my_quat_rotate,
    quat_inverse,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_to_matrix,
    wrap_to_pi,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from loguru import logger


class RefMotionCommand(CommandTerm):
    cfg: CommandTermCfg

    def __init__(
        self,
        cfg,
        env: ManagerBasedRLEnv,
    ):
        # print(cfg)
        super().__init__(cfg, env)
        self._env = env
        self._is_evaluating = self.cfg.is_evaluating

        self._init_robot_handle()
        self._init_buffers()
        self._init_motion_lib()

    #     # self._init_tracking_config()

    def _init_tracking_config(self, config):
        self.log_dict_holomotion = {}
        self.log_dict_nonreduced_holomotion = {}
        self.log_dict_nonreduced = {}
        self.log_dict = {}
        if "head_hand_bodies" in config:
            self.motion_tracking_id = [
                self.robot.body_names.index(link)
                for link in config.head_hand_bodies
            ]
        if "leg_body_names" in config:
            self.lower_body_id = [
                self.robot.body_names.index(link)
                for link in config.leg_body_names
            ]
        if "arm_body_names" in config:
            self.upper_body_id = [
                self.robot.body_names.index(link)
                for link in config.arm_body_names
            ]
        if "leg_dof_names" in config:
            self.lower_body_joint_ids = [
                config.dof_names.index(link) for link in config.leg_dof_names
            ]
        if "arm_dof_names" in config:
            self.upper_body_joint_ids = [
                config.dof_names.index(link) for link in config.arm_dof_names
            ]

        if "waist_dof_names" in config:
            self.waist_dof_indices = [
                config.dof_names.index(link) for link in config.waist_dof_names
            ]

    def _init_motion_lib(self):
        mcfg = OmegaConf.create(self.cfg.motion_lib_cfg)
        backend = mcfg.get("backend", "hdf5_simple")
        self._motion_cache = None
        if backend == "hdf5_simple":
            hdf5_root = mcfg.get("hdf5_root")
            if hdf5_root is None:
                raise ValueError("hdf5_root is required")
            val_hdf5_root = mcfg.get("val_hdf5_root")
            if val_hdf5_root is None:
                val_hdf5_root = hdf5_root

            manifest_path = os.path.join(hdf5_root, "manifest.json")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"HDF5 manifest not found at {manifest_path}. "
                    f"Please set robot.motion.hdf5_root to the correct path!"
                )

            val_manifest_path = os.path.join(val_hdf5_root, "manifest.json")
            max_frame_length = int(mcfg.get("max_frame_length", 500))
            min_frame_length = int(mcfg.get("min_frame_length", 1))

            logger.info(f"Loading HDF5 training dataset from {manifest_path}")
            train_dataset = Hdf5MotionDataset(
                manifest_path=manifest_path,
                max_frame_length=max_frame_length,
                min_window_length=min_frame_length,
                handpicked_motion_names=mcfg.get(
                    "handpicked_motion_names", None
                ),
                excluded_motion_names=mcfg.get("excluded_motion_names", None),
            )
            if len(train_dataset) == 0:
                raise ValueError(
                    f"Training dataset is empty. Check that {manifest_path} "
                    f"contains valid clips with length >= {min_frame_length}"
                )
            logger.info(f"Loaded {len(train_dataset)} training motion windows")
            train_num_clips = len(train_dataset.clips)
            train_total_frames = sum(
                int(meta.get("length", 0))
                for meta in train_dataset.clips.values()
            )
            fps_used = int(self.cfg.target_fps)
            train_duration_s = (
                float(train_total_frames) / float(fps_used)
                if fps_used > 0
                else 0.0
            )
            logger.info(
                f"Train dataset: root={hdf5_root}, manifest={manifest_path}"
            )
            logger.info(
                f"Train clips={train_num_clips}, frames={train_total_frames}, "
                f"duration={train_duration_s / 3600:.2f}h @ {fps_used} fps"
            )

            val_dataset = None
            if (
                os.path.exists(val_manifest_path)
                and val_hdf5_root != hdf5_root
            ):
                logger.info(
                    f"Loading HDF5 validation dataset from {val_manifest_path}"
                )
                val_dataset = Hdf5MotionDataset(
                    manifest_path=val_manifest_path,
                    max_frame_length=max_frame_length,
                    min_window_length=min_frame_length,
                )
                logger.info(
                    f"Loaded {len(val_dataset)} validation motion windows"
                )
                val_num_clips = len(val_dataset.clips)
                val_total_frames = sum(
                    int(meta.get("length", 0))
                    for meta in val_dataset.clips.values()
                )
                val_duration_s = (
                    float(val_total_frames) / float(fps_used)
                    if fps_used > 0
                    else 0.0
                )
                logger.info(
                    f"Val dataset: root={val_hdf5_root}, "
                    f"manifest={val_manifest_path}"
                )
                logger.info(
                    f"Val clips={val_num_clips}, frames={val_total_frames}, "
                    f"duration={val_duration_s / 3600:.1f}h @ {fps_used} fps"
                )
            else:
                logger.info(
                    "Validation dataset: using training dataset "
                    "(no separate val manifest found)"
                )

            cache_cfg = mcfg.get("cache", {})
            dataloader_cfg = mcfg.get("dataloader", {})

            stage_device = cache_cfg.get("device", "cuda")
            if stage_device == "cuda":
                stage_device = self.device
            else:
                stage_device = None

            self._motion_cache = MotionClipBatchCache(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=int(cache_cfg.get("max_num_clips", 1024)),
                stage_device=stage_device,
                num_workers=int(dataloader_cfg.get("num_workers", 4)),
                prefetch_factor=dataloader_cfg.get("prefetch_factor", None),
                pin_memory=bool(dataloader_cfg.get("pin_memory", True)),
                persistent_workers=bool(
                    dataloader_cfg.get("persistent_workers", True)
                ),
                sampler_rank=int(self.cfg.process_id),
                sampler_world_size=int(self.cfg.num_processes),
                swap_interval_steps=int(
                    cache_cfg.get("swap_interval_steps", max_frame_length)
                ),
            )
            cache = self._motion_cache
            logger.info(
                "DataLoader params: "
                f"batch_size={cache._batch_size}, "
                f"num_workers={cache._num_workers}, "
                f"prefetch_factor={cache._prefetch_factor}, "
                f"pin_memory={cache._pin_memory}, "
                f"persistent_workers={cache._persistent_workers}"
            )
            logger.info(
                "Sampler/Cache params: "
                f"rank={cache._sampler_rank}/{cache._sampler_world_size}, "
                f"device={cache._stage_device}, "
                f"swap_interval_steps={cache.swap_interval_steps}"
            )
            self._motion_lib = None

        self._init_per_env_cache()

    def _init_per_env_cache(self):
        """Initialize per-env cache for motion tracking."""
        self._clip_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._frame_indices = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._swap_pending = False
        self._swap_step_counter = 0

        # Initial assignment
        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            self.num_envs,
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(self._is_evaluating),
        )
        self._clip_indices[:] = clip_idx
        self._frame_indices[:] = frame_idx
        self._update_ref_motion_state_from_cache()

    def _init_robot_handle(self):
        self.robot: Articulation = self._env.scene[self.cfg.asset_name]
        self.anchor_bodylink_name = self.cfg.anchor_bodylink_name
        self.anchor_bodylink_idx = self.robot.body_names.index(
            self.anchor_bodylink_name
        )
        self.urdf_dof_names = self.cfg.urdf_dof_names
        self.urdf_body_names = self.cfg.urdf_body_names
        self.simulator_dof_names = self.robot.joint_names
        self.simulator_body_names = self.robot.body_names
        self.urdf2sim_dof_idx = [
            self.urdf_dof_names.index(dof) for dof in self.simulator_dof_names
        ]
        self.urdf2sim_body_idx = [
            self.urdf_body_names.index(body)
            for body in self.simulator_body_names
        ]
        self.sim2urdf_dof_idx = [
            self.simulator_dof_names.index(dof) for dof in self.urdf_dof_names
        ]
        self.sim2urdf_body_idx = [
            self.simulator_body_names.index(body)
            for body in self.urdf_body_names
        ]

        self.arm_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.arm_dof_names
        ]
        self.torso_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.torso_dof_names
        ]
        self.leg_dof_indices = [
            self.simulator_dof_names.index(dof)
            for dof in self.cfg.leg_dof_names
        ]

        # Body indices for mpkpe metrics using unified naming
        self.arm_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.arm_body_names
        ]
        self.torso_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.torso_body_names
        ]
        self.leg_body_indices = [
            self.simulator_body_names.index(body)
            for body in self.cfg.leg_body_names
        ]

    def _init_buffers(self):
        self.ref_motion_global_frame_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # mark envs that timed out (frame id exceeded end frame) in current step
        self._motion_end_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        # counter for number of motion ends per environment
        self.motion_end_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # per-environment cached motion indices
        self._cached_motion_ids = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        # env -> cache row indirection (starts as identity mapping)
        self._env_to_cache_row = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.pos_history_buffer = None
        self.rot_history_buffer = None
        self.ref_pos_history_buffer = None
        self.current_accel = None
        self.ref_body_accel = None
        self.current_ang_accel = None  # Placeholder for angular acceleration

    def _init_single_motion(self):
        single_id = self._motion_lib.sample_motion_ids_only(1, eval=True)
        motion_key = self._motion_lib.motion_id2key[int(single_id[0].item())]
        self.single_ref_motion = self._motion_lib.export_motion_clip(
            motion_key
        )

    @property
    def command(
        self,
    ) -> torch.Tensor:
        # call the corresponding method based on configured command_obs_name
        return getattr(self, f"_get_obs_{self.cfg.command_obs_name}")()

    @property
    def command_fut(
        self,
    ) -> torch.Tensor:
        # call the corresponding method based on configured command_obs_name
        return getattr(self, f"_get_obs_{self.cfg.command_obs_name}_fut")()

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> dict[str, float]:
        extras = super().reset(env_ids)

        if env_ids is None:
            env_ids = slice(None)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device)
        self._motion_end_mask[env_ids] = False
        self.motion_end_counter[env_ids] = 0

        if self._swap_pending:
            self._motion_cache.advance()
            self._swap_pending = False
            self._swap_step_counter = 0
            # Reassign all envs to new cache
            clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
                self.num_envs,
                self.cfg.n_fut_frames,
                self.device,
                deterministic_start=(self._is_evaluating),
            )
            self._clip_indices[:] = clip_idx
            self._frame_indices[:] = frame_idx
            self._update_ref_motion_state_from_cache()
            # Align all envs
            all_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._align_root_to_ref(all_ids)
            self._align_dof_to_ref(all_ids)
        else:
            self._resample_command(env_ids, eval=self._is_evaluating)

        return extras

    def compute(self, dt: float):
        self._update_metrics()
        self._update_command()

    def _update_ref_motion_state(self):
        """Update reference motion state (unified API)."""
        return self._update_ref_motion_state_from_cache()

    def _update_ref_motion_state_from_cache(
        self, env_ids: torch.Tensor | None = None
    ):
        """Update reference motion state from simple cache."""
        if env_ids is None:
            state = self._motion_cache.gather_state(
                self._clip_indices,
                self._frame_indices,
                n_future_frames=self.cfg.n_fut_frames,
            )
            self.ref_motion_state = state
        else:
            if not isinstance(env_ids, torch.Tensor):
                idxs = torch.tensor(
                    env_ids, device=self.device, dtype=torch.long
                )
            else:
                idxs = env_ids.to(self.device).long()
            sub_state = self._motion_cache.gather_state(
                self._clip_indices[idxs],
                self._frame_indices[idxs],
                n_future_frames=self.cfg.n_fut_frames,
            )
            if (
                not hasattr(self, "ref_motion_state")
                or self.ref_motion_state is None
            ):
                self.ref_motion_state = {
                    k: torch.zeros(
                        (self.num_envs,) + v.shape[1:],
                        device=v.device,
                        dtype=v.dtype,
                    )
                    for k, v in sub_state.items()
                }
            for k, v in sub_state.items():
                self.ref_motion_state[k][idxs] = v

    def _uniform_sample_ref_start_frames(self, env_ids: torch.Tensor):
        """Uniformly sample start frames within cached windows for env_ids.

        Sampling range is [start, end - 1 - n_fut_frames] to ensure required
        future frames exist. If that upper bound is < start, it falls back to start.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        else:
            env_ids = env_ids.to(self.device).long()

        starts = self.ref_motion_global_start_frame_ids[env_ids]
        ends = self.ref_motion_global_end_frame_ids[env_ids]

        # Ensure room for future frames if requested
        n_fut = (
            int(self.cfg.n_fut_frames)
            if hasattr(self.cfg, "n_fut_frames")
            else 0
        )
        max_start = ends - 1 - n_fut
        max_start = torch.maximum(max_start, starts)

        num_choices = (max_start - starts + 1).clamp(min=1)
        # Sample offsets uniformly
        rand = torch.rand_like(starts, dtype=torch.float32)
        offsets = torch.floor(rand * num_choices.float()).long()
        sampled = starts + offsets

        self.ref_motion_global_frame_ids[env_ids] = sampled

    @property
    def ref_motion_dof_pos_fut(self):
        return self.ref_motion_state["dof_pos"][:, 1:, ...][
            ..., self.urdf2sim_dof_idx
        ]

    @property
    def ref_motion_dof_vel_fut(self):
        return self.ref_motion_state["dof_vel"][:, 1:, ...][
            ..., self.urdf2sim_dof_idx
        ]

    @property
    def ref_motion_root_global_pos_fut(self):
        return self.ref_motion_state["root_pos"][:, 1:, ...]

    @property
    def ref_motion_root_global_rot_quat_xyzw_fut(self):
        return self.ref_motion_state["root_rot"][:, 1:, ...]

    @property
    def ref_motion_root_global_rot_quat_wxyz_fut(self):
        return self.ref_motion_root_global_rot_quat_xyzw_fut[..., [3, 0, 1, 2]]

    @property
    def ref_motion_root_global_rot_mat_fut(self):
        orginal_shape = self.ref_motion_root_global_rot_quat_xyzw_fut.shape
        return quaternion_to_matrix(
            self.ref_motion_root_global_rot_quat_xyzw_fut.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_root_global_lin_vel_fut(self):
        return self.ref_motion_state["root_vel"][:, 1:, ...]

    @property
    def ref_motion_root_global_ang_vel_fut(self):
        return self.ref_motion_state["root_ang_vel"][:, 1:, ...]

    @property
    def ref_motion_bodylink_global_pos_fut(self):
        return self.ref_motion_state["rg_pos"][:, 1:, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_rot_xyzw_fut(self):
        return self.ref_motion_state["rb_rot"][:, 1:, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_rot_wxyz_fut(self):
        return self.ref_motion_bodylink_global_rot_xyzw_fut[..., [3, 0, 1, 2]]

    @property
    def ref_motion_bodylink_global_rot_mat_fut(self):
        orginal_shape = self.ref_motion_bodylink_global_rot_xyzw_fut.shape
        return quaternion_to_matrix(
            self.ref_motion_bodylink_global_rot_xyzw_fut.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_bodylink_global_lin_vel_fut(self):
        return self.ref_motion_state["body_vel"][:, 1:, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_ang_vel_fut(self):
        return self.ref_motion_state["body_ang_vel"][:, 1:, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_dof_pos_cur(self):
        return self.ref_motion_state["dof_pos"][:, 0, ...][
            ..., self.urdf2sim_dof_idx
        ]

    @property
    def ref_motion_dof_pos_cur_urdf_order(self):
        return self.ref_motion_state["dof_pos"][:, 0, ...]

    @property
    def robot_dof_pos_cur_urdf_order(self):
        return self.robot.data.joint_pos[..., self.sim2urdf_dof_idx]

    @property
    def ref_motion_dof_vel_cur(self):
        return self.ref_motion_state["dof_vel"][:, 0, ...][
            ..., self.urdf2sim_dof_idx
        ]

    @property
    def robot_dof_vel_cur_urdf_order(self):
        return self.robot.data.joint_vel[..., self.sim2urdf_dof_idx]

    @property
    def ref_motion_dof_vel_cur_urdf_order(self):
        return self.ref_motion_state["dof_vel"][:, 0, ...]

    @property
    def ref_motion_root_global_pos_cur(self):
        return self.ref_motion_state["root_pos"][:, 0, ...]

    @property
    def ref_motion_root_global_rot_quat_xyzw_cur(self):
        return self.ref_motion_state["root_rot"][:, 0, ...]

    @property
    def ref_motion_root_global_rot_quat_wxyz_cur(self):
        return self.ref_motion_root_global_rot_quat_xyzw_cur[..., [3, 0, 1, 2]]

    @property
    def ref_motion_root_global_rot_mat_cur(self):
        orginal_shape = self.ref_motion_root_global_rot_quat_xyzw_cur.shape
        return quaternion_to_matrix(
            self.ref_motion_root_global_rot_quat_xyzw_cur.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_root_global_lin_vel_cur(self):
        return self.ref_motion_state["root_vel"][:, 0, ...]

    @property
    def ref_motion_root_global_ang_vel_cur(self):
        return self.ref_motion_state["root_ang_vel"][:, 0, ...]

    @property
    def ref_motion_bodylink_global_pos_cur(self):
        return self.ref_motion_state["rg_pos"][:, 0, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_pos_cur_urdf_order(self):
        return self.ref_motion_state["rg_pos"][:, 0, ...]

    @property
    def robot_bodylink_global_pos_cur_urdf_order(self):
        return self.robot.data.body_pos_w[:, self.sim2urdf_body_idx]

    @property
    def ref_motion_bodylink_global_rot_xyzw_cur(self):
        return self.ref_motion_state["rb_rot"][:, 0, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_rot_xyzw_cur_urdf_order(self):
        return self.ref_motion_state["rb_rot"][:, 0, ...]

    @property
    def ref_motion_bodylink_global_rot_wxyz_cur_urdf_order(self):
        return self.ref_motion_bodylink_global_rot_xyzw_cur_urdf_order[
            ..., [3, 0, 1, 2]
        ]

    @property
    def ref_motion_bodylink_global_rot_wxyz_cur(self):
        return self.ref_motion_bodylink_global_rot_xyzw_cur[..., [3, 0, 1, 2]]

    @property
    def robot_bodylink_global_rot_wxyz_cur_urdf_order(self):
        return self.robot.data.body_quat_w[:, self.sim2urdf_body_idx]

    @property
    def robot_bodylink_global_rot_xyzw_cur_urdf_order(self):
        return self.robot_bodylink_global_rot_wxyz_cur_urdf_order[
            ..., [1, 2, 3, 0]
        ]

    @property
    def ref_motion_bodylink_global_rot_mat_cur(self):
        orginal_shape = self.ref_motion_bodylink_global_rot_xyzw_cur.shape
        return quaternion_to_matrix(
            self.ref_motion_bodylink_global_rot_xyzw_cur.reshape(-1, 4),
            w_last=True,
        ).reshape(orginal_shape)

    @property
    def ref_motion_bodylink_global_lin_vel_cur(self):
        return self.ref_motion_state["body_vel"][:, 0, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_lin_vel_cur_urdf_order(self):
        return self.ref_motion_state["body_vel"][:, 0, ...]

    @property
    def robot_bodylink_global_lin_vel_cur_urdf_order(self):
        return self.robot.data.body_lin_vel_w[:, self.sim2urdf_body_idx]

    @property
    def ref_motion_bodylink_global_ang_vel_cur(self):
        return self.ref_motion_state["body_ang_vel"][:, 0, ...][
            ..., self.urdf2sim_body_idx, :
        ]

    @property
    def ref_motion_bodylink_global_ang_vel_cur_urdf_order(self):
        return self.ref_motion_state["body_ang_vel"][:, 0, ...]

    @property
    def robot_bodylink_global_ang_vel_cur_urdf_order(self):
        return self.robot.data.body_ang_vel_w[:, self.sim2urdf_body_idx]

    @property
    def ref_motion_global_start_frame_ids(self):
        frames = self._motion_lib.cache.cached_motion_global_start_frames.to(
            self.device
        )
        return frames[self._env_to_cache_row]

    @property
    def ref_motion_global_end_frame_ids(self):
        frames = self._motion_lib.cache.cached_motion_global_end_frames.to(
            self.device
        )
        return frames[self._env_to_cache_row]

    @property
    def motion_end_mask(self) -> torch.Tensor:
        """[B] bool: per-step timeout mask.

        Uses the per-step `motion_end_mask` set before resampling so the
        event is observable within the same step, and falls back to a
        direct comparison if not available.
        """
        return self._motion_end_mask

    @property
    def ref_motion_anchor_bodylink_global_pos_cur(self):
        return self.ref_motion_bodylink_global_pos_cur[
            :, self.anchor_bodylink_idx
        ]

    @property
    def ref_motion_anchor_bodylink_global_rot_cur_wxyz(self):
        return self.ref_motion_bodylink_global_rot_wxyz_cur[
            :, self.anchor_bodylink_idx
        ]

    @property
    def global_robot_anchor_pos_cur(self):
        return self.robot.data.body_pos_w[:, self.anchor_bodylink_idx]

    def _get_obs_bydmmc_ref_motion(self) -> torch.Tensor:
        num_envs = self.ref_motion_dof_pos_cur.shape[0]

        cur_ref_dof_pos_flat = self.ref_motion_dof_pos_cur.reshape(
            num_envs, -1
        )
        cur_ref_dof_vel_flat = self.ref_motion_dof_vel_cur.reshape(
            num_envs, -1
        )
        return torch.cat([cur_ref_dof_pos_flat, cur_ref_dof_vel_flat], dim=-1)

    def _get_obs_bydmmc_ref_motion_fut(self) -> torch.Tensor:
        num_envs = self.ref_motion_dof_pos_cur.shape[0]
        n_fut_frames = self.cfg.n_fut_frames
        fut_ref_dof_pos_flat = self.ref_motion_dof_pos_fut.reshape(
            num_envs, n_fut_frames, -1
        )
        fut_ref_dof_vel_flat = self.ref_motion_dof_vel_fut.reshape(
            num_envs, n_fut_frames, -1
        )
        rel_fut_ref_motion_state_seq = torch.cat(
            [fut_ref_dof_pos_flat, fut_ref_dof_vel_flat], dim=-1
        )
        return rel_fut_ref_motion_state_seq.reshape(num_envs, -1)

    def _get_obs_holomotion_rel_ref_motion_flat(self) -> torch.Tensor:
        num_envs, num_fut_timesteps, num_bodies, _ = (
            self.ref_motion_bodylink_global_pos_fut.shape
        )
        assert num_envs == self.num_envs
        assert num_fut_timesteps == self.cfg.n_fut_frames

        fut_ref_root_rot_quat = (
            self.ref_motion_root_global_rot_quat_xyzw_fut
        )  # [B, T, 4]
        fut_ref_root_rot_quat_inv = quat_inverse(
            fut_ref_root_rot_quat, w_last=True
        )  # [B, T, 4]
        fut_ref_root_rot_quat_body_flat = (
            fut_ref_root_rot_quat[:, :, None, :]
            .repeat(1, 1, num_bodies, 1)
            .reshape(-1, 4)
        )
        fut_ref_root_rot_quat_body_flat_inv = quat_inverse(
            fut_ref_root_rot_quat_body_flat, w_last=True
        )

        ref_fut_heading_quat_inv = calc_heading_quat_inv(
            fut_ref_root_rot_quat.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]
        ref_fut_quat_rp = quat_mul(
            ref_fut_heading_quat_inv,
            fut_ref_root_rot_quat.reshape(-1, 4),
            w_last=True,
        )  # [B*T, 4]

        ref_fut_roll, ref_fut_pitch, _ = get_euler_xyz(
            ref_fut_quat_rp,
            w_last=True,
        )
        ref_fut_roll = wrap_to_pi(ref_fut_roll).reshape(
            num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_pitch = wrap_to_pi(ref_fut_pitch).reshape(
            num_envs, num_fut_timesteps, -1
        )  # [B, T, 1]
        ref_fut_rp = torch.cat(
            [ref_fut_roll, ref_fut_pitch], dim=-1
        )  # [B, T, 2]
        ref_fut_rp_flat = ref_fut_rp.reshape(num_envs, -1)  # [B, T * 2]
        # ---

        fut_ref_root_quat_inv_fut_flat = fut_ref_root_rot_quat_inv.reshape(
            -1, 4
        )
        fut_ref_cur_root_rel_base_lin_vel = quat_rotate(
            fut_ref_root_quat_inv_fut_flat,  # [B*T, 4]
            self.ref_motion_root_global_lin_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(num_envs, -1)  # [B, num_fut_timesteps * 3]
        fut_ref_cur_root_rel_base_ang_vel = quat_rotate(
            fut_ref_root_quat_inv_fut_flat,  # [B*T, 4]
            self.ref_motion_root_global_ang_vel_fut.reshape(-1, 3),  # [B*T, 3]
            w_last=True,
        ).reshape(num_envs, -1)  # [B, num_fut_timesteps * 3]
        # ---

        # --- calculate the absolute DoF position and velocity ---
        fut_ref_dof_pos_flat = self.ref_motion_dof_pos_fut.reshape(
            num_envs, -1
        )
        fut_ref_dof_vel_flat = self.ref_motion_dof_vel_fut.reshape(
            num_envs, -1
        )
        # ---

        # --- calculate the future per frame bodylink position and rotation ---
        fut_ref_global_bodylink_pos = (
            self.ref_motion_bodylink_global_pos_fut
        )  # [B, T, num_bodies, 3]
        fut_ref_global_bodylink_rot = (
            self.ref_motion_bodylink_global_rot_xyzw_fut
        )  # [B, T, num_bodies, 4]

        # get root-relative bodylink position
        fut_ref_root_rel_bodylink_pos = quat_rotate(
            fut_ref_root_rot_quat_body_flat_inv,
            (
                fut_ref_global_bodylink_pos
                - fut_ref_global_bodylink_pos[:, :, 0:1, :]
            ).reshape(-1, 3),
            w_last=True,
        ).reshape(
            num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 3]

        # get root-relative bodylink rotation
        fut_ref_root_rel_bodylink_rot = quat_mul(
            fut_ref_root_rot_quat_body_flat_inv,
            fut_ref_global_bodylink_rot.reshape(-1, 4),
            w_last=True,
        )
        fut_ref_root_rel_bodylink_rot_mat = quaternion_to_matrix(
            fut_ref_root_rel_bodylink_rot,
            w_last=True,
        )[:, :, :2].reshape(
            num_envs, num_fut_timesteps, num_bodies, -1
        )  # [B, num_fut_timesteps, num_bodies, 6]

        rel_fut_ref_motion_state_seq = torch.cat(
            [
                ref_fut_rp_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 2]
                fut_ref_cur_root_rel_base_lin_vel.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_cur_root_rel_base_ang_vel.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, 3]
                fut_ref_dof_pos_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_dof_vel_flat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_dofs]
                fut_ref_root_rel_bodylink_pos.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*3]
                fut_ref_root_rel_bodylink_rot_mat.reshape(
                    num_envs, num_fut_timesteps, -1
                ),  # [B, T, num_bodies*6]
            ],
            dim=-1,
        )  # [B, T, 2 + 3 + 3 + num_dofs * 2 + num_bodies * (3 + 6)]
        return rel_fut_ref_motion_state_seq.reshape(self.num_envs, -1)

    def _resample_command(self, env_ids: Sequence[int], eval=False):
        """Resample command for specified environments."""
        if len(env_ids) == 0:
            return

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)
        else:
            env_ids = env_ids.to(self.device)

        if isinstance(env_ids, torch.Tensor):
            idxs = env_ids
        elif isinstance(env_ids, slice):
            idxs = torch.arange(self.num_envs, device=self.device)
        else:
            idxs = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
            len(idxs),
            self.cfg.n_fut_frames,
            self.device,
            deterministic_start=(eval or self._is_evaluating),
        )
        self._clip_indices[idxs] = clip_idx
        self._frame_indices[idxs] = frame_idx
        self._update_ref_motion_state_from_cache(env_ids=idxs)
        self._align_root_to_ref(idxs)
        self._align_dof_to_ref(idxs)

    def _align_root_to_ref(self, env_ids):
        root_pos = self.ref_motion_root_global_pos_cur.clone()
        root_rot = self.ref_motion_root_global_rot_quat_wxyz_cur.clone()
        root_lin_vel = self.ref_motion_root_global_lin_vel_cur.clone()
        root_ang_vel = self.ref_motion_root_global_ang_vel_cur.clone()

        pos_rot_range_list = [
            self.cfg.root_pose_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        pos_rot_ranges = torch.tensor(pos_rot_range_list, device=self.device)
        pos_rot_rand_deltas = isaaclab_math.sample_uniform(
            pos_rot_ranges[:, 0],
            pos_rot_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        translation_delta = pos_rot_rand_deltas[:, 0:3]
        rotation_delta = isaaclab_math.quat_from_euler_xyz(
            pos_rot_rand_deltas[:, 3],
            pos_rot_rand_deltas[:, 4],
            pos_rot_rand_deltas[:, 5],
        )

        root_pos[env_ids] += translation_delta
        root_rot[env_ids] = isaaclab_math.quat_mul(
            rotation_delta,
            root_rot[env_ids],
        )

        lin_ang_vel_range_list = [
            self.cfg.root_vel_perturb_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        lin_ang_vel_ranges = torch.tensor(
            lin_ang_vel_range_list, device=self.device
        )

        lin_ang_vel_rand_deltas = isaaclab_math.sample_uniform(
            lin_ang_vel_ranges[:, 0],
            lin_ang_vel_ranges[:, 1],
            (len(env_ids), 6),
            device=self.device,
        )
        root_lin_vel[env_ids] += lin_ang_vel_rand_deltas[:, :3]
        root_ang_vel[env_ids] += lin_ang_vel_rand_deltas[:, 3:]

        self.robot.write_root_state_to_sim(
            torch.cat(
                [
                    root_pos[env_ids],
                    root_rot[env_ids],
                    root_lin_vel[env_ids],
                    root_ang_vel[env_ids],
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )

    def _align_dof_to_ref(self, env_ids):
        dof_pos = self.ref_motion_dof_pos_cur.clone()
        dof_vel = self.ref_motion_dof_vel_cur.clone()

        dof_pos += isaaclab_math.sample_uniform(
            *self.cfg.dof_pos_perturb_range,
            dof_pos.shape,
            dof_pos.device,
        )
        soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        dof_pos[env_ids] = torch.clip(
            dof_pos[env_ids],
            soft_dof_pos_limits[:, :, 0],
            soft_dof_pos_limits[:, :, 1],
        )

        self.robot.write_joint_state_to_sim(
            dof_pos[env_ids],
            dof_vel[env_ids],
            env_ids=env_ids,
        )

    def _update_command(self):
        self._frame_indices += 1
        self._swap_step_counter += 1

        if self._swap_step_counter >= self._motion_cache.swap_interval_steps:
            self._swap_pending = True

        # Resample when motion ends
        self._resample_when_motion_end_cache()
        self._update_ref_motion_state_from_cache()

    def _resample_when_motion_end_cache(self):
        """Resample environments when motion ends (simple cache mode)."""
        lengths = self._motion_cache.lengths_for_indices(self._clip_indices)
        max_valid_frame = torch.clamp(
            lengths - 1 - self.cfg.n_fut_frames, min=0
        )
        need_resample = self._frame_indices > max_valid_frame

        if torch.any(need_resample):
            resample_ids = torch.nonzero(need_resample).squeeze(-1)
            # Resample these envs
            clip_idx, frame_idx = self._motion_cache.sample_env_assignments(
                len(resample_ids),
                self.cfg.n_fut_frames,
                self.device,
                deterministic_start=self._is_evaluating,
            )
            self._clip_indices[resample_ids] = clip_idx
            self._frame_indices[resample_ids] = frame_idx
            # Realign robot state
            self._update_ref_motion_state_from_cache(env_ids=resample_ids)
            self._align_root_to_ref(resample_ids)
            self._align_dof_to_ref(resample_ids)
            # Mark motion end
            self._motion_end_mask[:] = False
            self._motion_end_mask[resample_ids] = True
            self.motion_end_counter[resample_ids] += 1

    def _update_metrics(self):
        """Update metrics for command progress tracking."""
        if not hasattr(self, "metrics"):
            self.metrics = {}

        self._update_motion_progress_metrics()
        self._update_mpjpe_metrics()
        self._update_mpkpe_metrics()

    def _update_motion_progress_metrics(self):
        """Track motion progress as percentage."""
        lengths = self._motion_cache.lengths_for_indices(self._clip_indices)
        denom = torch.clamp(lengths.float() - 1.0, min=1.0)
        numer = torch.clamp(self._frame_indices.float(), min=0.0)
        motion_progress = torch.clamp(numer / denom, 0.0, 1.0)

        if "Task/Motion_Progress" not in self.metrics:
            self.metrics["Task/Motion_Progress"] = torch.zeros(
                self.num_envs, device=self.device
            )
        self.metrics["Task/Motion_Progress"][:] = motion_progress

    def _update_mpjpe_metrics(self):
        """Update MPJPE (Mean Per Joint Position Error) metrics."""
        # Get current and reference joint positions
        current_dof_pos = self.robot.data.joint_pos  # [B, num_dofs]
        ref_dof_pos = self.ref_motion_dof_pos_cur  # [B, num_dofs]

        # Compute joint position errors
        dof_pos_error = torch.abs(
            current_dof_pos - ref_dof_pos
        )  # [B, num_dofs]

        # MPJPE whole body
        mpjpe_wholebody = torch.mean(dof_pos_error, dim=-1)  # [B]

        # MPJPE arms (using unified naming)
        mpjpe_arms = torch.mean(
            dof_pos_error[:, self.arm_dof_indices], dim=-1
        )  # [B]

        # MPJPE torso (using unified naming)
        mpjpe_waist = torch.mean(
            dof_pos_error[:, self.torso_dof_indices], dim=-1
        )  # [B]

        # MPJPE legs
        mpjpe_legs = torch.mean(
            dof_pos_error[:, self.leg_dof_indices], dim=-1
        )  # [B]

        # Initialize metric tensors if needed
        for metric_name in [
            "Task/MPJPE_WholeBody",
            "Task/MPJPE_Arms",
            "Task/MPJPE_Waist",
            "Task/MPJPE_Legs",
        ]:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # Update metric values
        self.metrics["Task/MPJPE_WholeBody"][:] = mpjpe_wholebody
        self.metrics["Task/MPJPE_Arms"][:] = mpjpe_arms
        self.metrics["Task/MPJPE_Waist"][:] = mpjpe_waist
        self.metrics["Task/MPJPE_Legs"][:] = mpjpe_legs

    def _update_mpkpe_metrics(self):
        """Update MPKPE (Mean Per Keybody Position Error) metrics."""
        # Get current and reference body positions
        current_body_pos = self.robot.data.body_pos_w  # [B, num_bodies, 3]
        ref_body_pos = (
            self.ref_motion_bodylink_global_pos_cur
        )  # [B, num_bodies, 3]

        # Compute body position errors (L2 norm)
        body_pos_error = torch.norm(
            current_body_pos - ref_body_pos, dim=-1
        )  # [B, num_bodies]

        # MPKPE whole body
        mpkpe_wholebody = torch.mean(body_pos_error, dim=-1)  # [B]

        # MPKPE arms (using unified naming)
        mpkpe_arms = torch.mean(
            body_pos_error[:, self.arm_body_indices], dim=-1
        )  # [B]

        # MPKPE torso (using unified naming)
        mpkpe_waist = torch.mean(
            body_pos_error[:, self.torso_body_indices], dim=-1
        )  # [B]

        # MPKPE legs
        mpkpe_legs = torch.mean(
            body_pos_error[:, self.leg_body_indices], dim=-1
        )  # [B]

        # Initialize metric tensors if needed
        for metric_name in [
            "Task/MPKPE_WholeBody",
            "Task/MPKPE_Arms",
            "Task/MPKPE_Waist",
            "Task/MPKPE_Legs",
        ]:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = torch.zeros(
                    self.num_envs, device=self.device
                )

        # Update metric values
        self.metrics["Task/MPKPE_WholeBody"][:] = mpkpe_wholebody
        self.metrics["Task/MPKPE_Arms"][:] = mpkpe_arms
        self.metrics["Task/MPKPE_Waist"][:] = mpkpe_waist
        self.metrics["Task/MPKPE_Legs"][:] = mpkpe_legs

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            # Just enable debug mode - visualizers will be created lazily in callback
            self._debug_vis_enabled = True
            # Set visibility if visualizers already exist
            if hasattr(self, "ref_body_visualizers"):
                for visualizer in self.ref_body_visualizers:
                    visualizer.set_visibility(True)
        else:
            self._debug_vis_enabled = False
            # Set visibility to false
            if hasattr(self, "ref_body_visualizers"):
                for visualizer in self.ref_body_visualizers:
                    visualizer.set_visibility(False)

    def setup_offline_eval_from_frame_zero(self):
        """Setup environment for offline evaluation starting from frame 0."""

        self._frame_indices[:] = 0

        self._update_ref_motion_state()

        all_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._align_root_to_ref(all_env_ids)
        self._align_dof_to_ref(all_env_ids)

        logger.info(
            f"Offline evaluation setup complete: all {self.num_envs} "
            f"environments aligned to frame 0"
        )

    def setup_offline_eval_deterministic(self, apply_pending_swap: bool = True) -> None:
        """Deterministic multi-env setup for offline evaluation.

        - Optionally apply a pending cache swap.
        - Set env i -> cache row i mapping for active clips, frame 0.
        - Update reference state and align robot states.
        """
        if apply_pending_swap and getattr(self, "_swap_pending", False):
            self._motion_cache.advance()
            self._swap_pending = False
            self._swap_step_counter = 0

        clip_count = int(self._motion_cache.clip_count)
        active_count = min(int(self.num_envs), clip_count)

        # Reset indices
        self._clip_indices[:] = 0
        self._frame_indices[:] = 0

        if active_count > 0:
            active_ids = torch.arange(active_count, dtype=torch.long, device=self.device)
            self._clip_indices[active_ids] = torch.arange(
                active_count, dtype=torch.long, device=self.device
            )
        else:
            active_ids = None

        self._update_ref_motion_state_from_cache()

        if active_ids is not None:
            self._align_root_to_ref(active_ids)
            self._align_dof_to_ref(active_ids)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        # Check if debug visualization is enabled
        if not getattr(self, "_debug_vis_enabled", False):
            return

        # Check if reference motion state is available
        if (
            not hasattr(self, "ref_motion_state")
            or self.ref_motion_state is None
        ):
            return

        # Create visualizers lazily if they don't exist
        if not hasattr(self, "ref_body_visualizers"):
            self.ref_body_visualizers = []
            # Get number of bodies from the reference motion data
            num_bodies = self.ref_motion_bodylink_global_pos_cur.shape[-2]
            for i in range(num_bodies):
                # Reference bodylinks as red spheres
                self.ref_body_visualizers.append(
                    VisualizationMarkers(
                        self.cfg.body_keypoint_visualizer_cfg.replace(
                            prim_path=f"/Visuals/Command/ref_body_{i}"
                        )
                    )
                )

        # Visualize reference body keypoints
        if len(self.ref_body_visualizers) > 0:
            ref_body_pos = (
                self.ref_motion_bodylink_global_pos_cur
            )  # [B, num_bodies, 3]

            num_bodies = min(
                len(self.ref_body_visualizers), ref_body_pos.shape[1]
            )

            for i in range(num_bodies):
                # Visualize reference bodylinks as spheres (position only)
                self.ref_body_visualizers[i].visualize(
                    ref_body_pos[:, i],  # [B, 3]
                )


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = RefMotionCommand

    command_obs_name: str = MISSING
    urdf_dof_names: list[str] = MISSING
    urdf_body_names: list[str] = MISSING

    # DOF name groupings for mpjpe metrics (using unified naming)
    arm_dof_names: list[str] = MISSING
    torso_dof_names: list[str] = MISSING
    leg_dof_names: list[str] = MISSING

    # Body name groupings for mpkpe metrics (using unified naming)
    arm_body_names: list[str] = MISSING
    torso_body_names: list[str] = MISSING
    leg_body_names: list[str] = MISSING

    motion_lib_cfg: dict = MISSING
    process_id: int = MISSING
    num_processes: int = MISSING
    is_evaluating: bool = MISSING
    resample_time_interval_s: float = MISSING

    n_fut_frames: int = MISSING
    target_fps: int = MISSING

    anchor_bodylink_name: str = "pelvis"

    asset_name: str = MISSING
    debug_vis: bool = False

    root_pose_perturb_range: dict[str, tuple[float, float]] = {}
    root_vel_perturb_range: dict[str, tuple[float, float]] = {}
    dof_pos_perturb_range: tuple[float, float] = (-0.1, 0.1)
    dof_vel_perturb_range: tuple[float, float] = (-1.0, 1.0)

    body_keypoint_visualizer_cfg: VisualizationMarkersCfg = (
        SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/ref_keypoint")
    )
    body_keypoint_visualizer_cfg.markers["sphere"].radius = 0.03
    body_keypoint_visualizer_cfg.markers[
        "sphere"
    ].visual_material = PreviewSurfaceCfg(
        diffuse_color=(0.0, 0.0, 1.0)  # blue
    )

    resampling_time_range: tuple[float, float] = (1.0, 1.0)


@configclass
class MoTrack_CommandsCfg:
    pass


def build_motion_tracking_commands_config(command_config_dict: dict):
    """Build isaaclab-compatible CommandsCfg from a config dictionary.

    Args:
        command_config_dict: Dictionary mapping command names to command configurations.
                           Each command config should contain the type and parameters.

    Example:
        command_config_dict = {
            "ref_motion": {
                "type": "MotionCommandCfg",
                "params": {
                    "command_obs_name": "bydmmc_ref_motion",
                    "motion_lib_cfg": {...},
                    "process_id": 0,
                    "num_processes": 1,
                    # ... other parameters
                }
            }
        }
    """

    commands_cfg = MoTrack_CommandsCfg()

    # Add command terms dynamically
    for command_name, command_config in command_config_dict.items():
        command_type = command_config.get("type", "MotionCommandCfg")
        command_params = command_config.get("params", {})

        # Get the command class type
        if command_type == "MotionCommandCfg":
            command_cfg = MotionCommandCfg(**command_params)
        else:
            raise ValueError(f"Unknown command type: {command_type}")

        # Add command to config
        setattr(commands_cfg, command_name, command_cfg)

    return commands_cfg
