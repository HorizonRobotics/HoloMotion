"""Observation and FK-derived observation formulas for the 29DOF policy node."""

from __future__ import annotations

import math

import numpy as np

from humanoid_policy.offline_motion_reference import OfflineMotionReference
from humanoid_policy.vr_reference import VrLatestObsReference


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Calculate gravity orientation from a [w, x, y, z] quaternion."""
    qw = float(quaternion[0])
    qx = float(quaternion[1])
    qy = float(quaternion[2])
    qz = float(quaternion[3])

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


class PolicyObservationEvaluator:
    """Build observation terms while keeping ROS node glue out of PolicyObsBuilder."""

    def __init__(self, node):
        self._node = node

    def __getattr__(self, name):
        return getattr(self._node, name)

    def get_logger(self):
        return self._node.get_logger()

    def clear_vr_fk_cache(self) -> None:
        self._fk_vr_out = None
        self._use_fk_vr = False
        self._fk_vr_cache_ready = False

    def initialize_vr_fk_buffers(self, n_fut: int, num_actions: int) -> None:
        import torch

        n_fut = int(n_fut)
        num_actions = int(num_actions)
        if n_fut > 0:
            self._fk_root_pos_seq_np = np.zeros((1, n_fut + 1, 3), dtype=np.float32)
            self._fk_root_rot_seq_np = np.zeros((1, n_fut + 1, 4), dtype=np.float32)
            self._fk_dof_pos_seq_np = np.zeros(
                (1, n_fut + 1, num_actions), dtype=np.float32
            )
            self._fk_root_pos_seq_tensor = torch.from_numpy(self._fk_root_pos_seq_np)
            self._fk_root_rot_seq_tensor = torch.from_numpy(self._fk_root_rot_seq_np)
            self._fk_dof_pos_seq_tensor = torch.from_numpy(self._fk_dof_pos_seq_np)
            self.get_logger().info(
                f"Initialized VR future frame queues: n_fut_frames={n_fut}, num_actions={num_actions}"
            )
            return

        self._fk_root_pos_seq_np = None
        self._fk_root_rot_seq_np = None
        self._fk_dof_pos_seq_np = None
        self._fk_root_pos_seq_tensor = None
        self._fk_root_rot_seq_tensor = None
        self._fk_dof_pos_seq_tensor = None

    def initialize_observation_state(self) -> None:
        node = self._node
        self.ref_to_onnx = [
            node.dof_names_ref_motion.index(name) for name in node.motion_dof_names_onnx
        ]

        self.velocity_default_angles_dict = {
            name: float(node.velocity_default_angles_onnx[idx])
            for idx, name in enumerate(node.velocity_dof_names_onnx)
        }
        self.motion_default_angles_dict = {
            name: float(node.motion_default_angles_onnx[idx])
            for idx, name in enumerate(node.motion_dof_names_onnx)
        }
        self.velocity_dof_names_onnx_array = np.array(node.velocity_dof_names_onnx)
        self.motion_dof_names_onnx_array = np.array(node.motion_dof_names_onnx)
        self.motion_dof_real_indices = [
            node.real_dof_names.index(name) for name in node.motion_dof_names_onnx
        ]
        self.velocity_dof_real_indices = [
            node.real_dof_names.index(name) for name in node.velocity_dof_names_onnx
        ]
        self.motion_dof_real_indices_np = np.asarray(
            self.motion_dof_real_indices,
            dtype=np.int64,
        )
        self.velocity_dof_real_indices_np = np.asarray(
            self.velocity_dof_real_indices,
            dtype=np.int64,
        )

        n_dof = max(len(node.motion_dof_names_onnx), len(node.velocity_dof_names_onnx))
        self._dof_pos_obs_buffer = np.zeros(n_dof, dtype=np.float32)
        self._dof_vel_obs_buffer = np.zeros(n_dof, dtype=np.float32)
        self._real_dof_pos_buffer = np.zeros(node.actions_dim, dtype=np.float32)
        self._real_dof_vel_buffer = np.zeros(node.actions_dim, dtype=np.float32)
        self._robot_root_quat_wxyz_buffer = np.zeros(4, dtype=np.float32)
        self._robot_root_ang_vel_buffer = np.zeros(3, dtype=np.float32)
        self._lowstate_cache_msg = None

        self.n_fut_frames_int = int(getattr(node, "n_fut_frames", 0) or 0)
        if self.n_fut_frames_int <= 0:
            self.n_fut_frames_int = 0

        if self.n_fut_frames_int > 0:
            self._pos_fut_buffer = np.zeros(
                (len(node.dof_names_ref_motion), self.n_fut_frames_int),
                dtype=np.float32,
            )
            self._h_fut_buffer = np.zeros((1, self.n_fut_frames_int), dtype=np.float32)
            self._root_pos_fut_buffer = np.zeros(
                (self.n_fut_frames_int, 3), dtype=np.float32
            )
        else:
            self._pos_fut_buffer = np.zeros(
                (len(node.dof_names_ref_motion), 0), dtype=np.float32
            )
            self._h_fut_buffer = np.zeros((1, 0), dtype=np.float32)
            self._root_pos_fut_buffer = np.zeros((0, 3), dtype=np.float32)

        self._offline_reference = OfflineMotionReference(
            n_fut_frames=self.n_fut_frames_int,
            num_actions=node.num_actions,
            ref_to_onnx=self.ref_to_onnx,
            root_body_idx=self.root_body_idx,
            reference_dof_count=len(node.dof_names_ref_motion),
        )

        self._future_frame_offsets = np.arange(
            1, self.n_fut_frames_int + 1, dtype=np.int64
        )
        self._future_frame_indices_buffer = np.zeros(
            self.n_fut_frames_int, dtype=np.int64
        )
        self._future_root_quat_wxyz_buffer = np.zeros(
            (self.n_fut_frames_int, 4), dtype=np.float32
        )
        self._gravity_fut_buffer = np.zeros(
            (self.n_fut_frames_int, 3), dtype=np.float32
        )
        self._base_linvel_fut_buffer = np.zeros(
            (self.n_fut_frames_int, 3), dtype=np.float32
        )
        self._base_angvel_fut_buffer = np.zeros(
            (self.n_fut_frames_int, 3), dtype=np.float32
        )
        self._gravity_cur_buffer = np.zeros(3, dtype=np.float32)
        self._base_linvel_cur_buffer = np.zeros(3, dtype=np.float32)
        self._base_angvel_cur_buffer = np.zeros(3, dtype=np.float32)
        self._keybody_rel_pos_fut_buffer = np.zeros(
            (self.n_fut_frames_int, 0, 3), dtype=np.float32
        )
        self._keybody_rel_pos_w_buffer = None
        max_t = max(1, self.n_fut_frames_int)
        self._vel_fut_T6 = np.zeros((max_t, 6), dtype=np.float32)
        self._rot_t_buffer = np.zeros((max_t, 3), dtype=np.float32)
        self._rot_cross_buffer = np.zeros((max_t, 3), dtype=np.float32)
        self._fk_root_vel_seq_np = np.zeros((max_t + 1, 3), dtype=np.float32)
        self._fk_root_angvel_seq_np = np.zeros((max_t + 1, 3), dtype=np.float32)
        self._fk_q_rel_buffer = np.zeros((max_t, 4), dtype=np.float32)
        self._fk_axis_angle_buffer = np.zeros((max_t, 3), dtype=np.float32)
        self._use_fk_vr = False
        self._fk_vr_cache_ready = False
        self._fk_vel_0_root = np.zeros(3, dtype=np.float32)
        self._fk_angvel_0_root = np.zeros(3, dtype=np.float32)
        self._fk_quat_0_root = np.zeros(4, dtype=np.float32)
        self._fk_trans_0 = None
        self._fk_vel_fut = np.zeros((max_t, 3), dtype=np.float32)
        self._fk_angvel_fut = np.zeros((max_t, 3), dtype=np.float32)
        self._fk_quat_fut = np.zeros((max_t, 4), dtype=np.float32)
        self._fk_trans_fut = None
        self._q_conj_buffer = np.zeros((max_t + 1, 4), dtype=np.float32)
        self._rotated_3vec_buffer = np.zeros(3, dtype=np.float32)
        self._rotated_angvel_cur_buffer = np.zeros(3, dtype=np.float32)
        self._cross_t_buffer = np.zeros(3, dtype=np.float32)
        self._fk_quat_0_root_wxyz = np.zeros(4, dtype=np.float32)
        self._fk_quat_fut_wxyz = np.zeros((max_t, 4), dtype=np.float32)
        self._velocity_cmd_obs = np.zeros(4, dtype=np.float32)
        self._projected_gravity_buffer = np.zeros(3, dtype=np.float32)
        self._ref_motion_states_buffer = np.zeros(
            node.num_actions * 2,
            dtype=np.float32,
        )
        self._place_holder_buffer = np.zeros(
            int(getattr(node, "actor_place_holder_ndim", 0) or 0),
            dtype=np.float32,
        )

    def initialize_fk(self) -> None:
        import torch

        from humanoid_policy.holomotion_fk_root_only import HoloMotionFKRootOnly

        node = self._node
        try:
            self.get_logger().info(
                "Initializing root-only FK (no URDF, sync main-thread mode)"
            )
            self.fk = HoloMotionFKRootOnly(
                dof_names=node.dof_names_ref_motion,
                device="cpu",
                timing_logger_enabled=True,
                timing_log_interval_sec=5.0,
                timing_log_per_call=False,
                timing_name="FKRootOnlyVR",
                timing_log_fn=self.get_logger().info,
            )
            try:
                ndof = len(self.fk.dof_names)
                root_pos_dummy = torch.zeros((1, 4, 3), dtype=torch.float32)
                root_quat_dummy = torch.zeros((1, 4, 4), dtype=torch.float32)
                root_quat_dummy[..., 0] = 1.0
                dof_pos_dummy = torch.zeros((1, 4, ndof), dtype=torch.float32)
                _ = self.fk(
                    root_pos=root_pos_dummy,
                    root_quat=root_quat_dummy,
                    dof_pos=dof_pos_dummy,
                    fps=float(1.0 / node.dt),
                    quat_format="wxyz",
                    vel_smoothing_sigma=0.0,
                    compute_velocity=False,
                )
                self.get_logger().info("[FK] Root-only warmup completed (B=1,T=4)")
            except Exception as exc:
                self.get_logger().warn(f"[FK] Root-only warmup failed (ignored): {exc}")

            self.fk_initialized = True
            self.get_logger().info(
                f"Root-only FK initialized successfully with {len(self.fk.dof_names)} dofs"
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to initialize root-only FK: {exc}")
            self.fk = None
            self.fk_initialized = False

    def _init_keybody_indices_cache(self):
        if self.motion_config is None:
            raise ValueError("motion_config is not loaded; cannot init keybody index cache")

        atomic_list = self._get_policy_atomic_obs_list(self.motion_config)["atomic_obs_list"]
        body_names = [str(name) for name in self.motion_config.robot.body_names]
        body_name_to_idx = {body_name: idx for idx, body_name in enumerate(body_names)}

        cache = {}
        for term_dict in atomic_list:
            term_name = str(list(term_dict.keys())[0])
            term_cfg = term_dict[term_name]
            params = {}
            if isinstance(term_cfg, dict):
                params = term_cfg.get("params", {}) or {}
                if not isinstance(params, dict):
                    raise ValueError(
                        f"Observation term '{term_name}' params must be a dict, got {type(params)}"
                    )
            needs_keybody = ("keybody" in term_name) or ("keybody_names" in params)
            if not needs_keybody:
                continue

            keybody_names = params.get("keybody_names", None)
            if keybody_names is None:
                keybody_idxs = np.arange(len(body_names), dtype=np.int64)
            else:
                keybody_names = [str(name) for name in keybody_names]
                missing_names = [
                    name for name in keybody_names if name not in body_name_to_idx
                ]
                if len(missing_names) > 0:
                    raise ValueError(
                        f"Unknown keybody_names in '{term_name}': {missing_names}. "
                        f"Available body names: {body_names}"
                    )
                keybody_idxs = np.asarray(
                    [body_name_to_idx[name] for name in keybody_names],
                    dtype=np.int64,
                )

            cache[term_name] = keybody_idxs

        self._keybody_indices_by_term_name = cache

    def _get_policy_atomic_obs_list(self, config):
        """Resolve the atomic obs list used to build the ONNX policy input.

        Aligns with MuJoCo sim2sim eval ordering by honoring modules.actor.obs_schema
        when available, to guarantee the policy input term order matches training/export.
        """
        from omegaconf import OmegaConf

        def _to_plain_obs_cfg(cfg):
            if OmegaConf.is_config(cfg):
                plain_cfg = OmegaConf.to_container(cfg, resolve=True)
            elif cfg is None:
                plain_cfg = {}
            else:
                plain_cfg = dict(cfg)
            if plain_cfg is None:
                plain_cfg = {}
            if not isinstance(plain_cfg, dict):
                raise ValueError(
                    f"Observation term config must be a mapping, got {type(plain_cfg)}"
                )
            return plain_cfg

        def _get_actor_atomic_obs_entries():
            obs_cfg = config.get("obs", None)
            if obs_cfg is None:
                raise ValueError("Missing config.obs for policy obs")
            obs_groups = obs_cfg.get("obs_groups", None)
            if obs_groups is None:
                raise ValueError("Missing config.obs.obs_groups for policy obs")

            if obs_groups.get("policy", None) is not None:
                entries = []
                for term_dict in obs_groups.policy.atomic_obs_list:
                    term_name = str(list(term_dict.keys())[0])
                    entries.append(
                        (
                            "policy",
                            term_name,
                            _to_plain_obs_cfg(term_dict[term_name]),
                        )
                    )
                return entries

            if obs_groups.get("unified", None) is not None:
                entries = []
                for term_dict in obs_groups.unified.atomic_obs_list:
                    term_name = str(list(term_dict.keys())[0])
                    if term_name.startswith("actor_"):
                        entries.append(
                            (
                                "unified",
                                term_name,
                                _to_plain_obs_cfg(term_dict[term_name]),
                            )
                        )
                if not entries:
                    raise ValueError(
                        "obs_groups.unified found but contains no actor_* terms."
                    )
                return entries

            raise ValueError(
                "Unsupported obs config : expected obs_groups.policy or obs_groups.unified."
            )

        def _get_actor_obs_schema_terms():
            modules_cfg = config.get("modules", None)
            if modules_cfg is None:
                return []
            actor_cfg = modules_cfg.get("actor", None)
            if actor_cfg is None:
                return []
            obs_schema = actor_cfg.get("obs_schema", None)
            if obs_schema is None:
                return []

            if OmegaConf.is_config(obs_schema):
                obs_schema_plain = OmegaConf.to_container(obs_schema, resolve=True)
            else:
                obs_schema_plain = obs_schema
            if not isinstance(obs_schema_plain, dict):
                return []

            ordered_terms = []

            def _collect_terms(node):
                if node is None:
                    return
                if isinstance(node, dict):
                    if "terms" in node and isinstance(node["terms"], list):
                        ordered_terms.extend(str(term) for term in node["terms"])
                        return
                    for v in node.values():
                        _collect_terms(v)
                    return
                if isinstance(node, list):
                    for v in node:
                        _collect_terms(v)
                    return

            _collect_terms(obs_schema_plain)
            return ordered_terms

        actor_atomic_entries = _get_actor_atomic_obs_entries()
        schema_terms = _get_actor_obs_schema_terms()

        if len(schema_terms) == 0:
            return {
                "atomic_obs_list": [
                    {term_name: term_cfg}
                    for _, term_name, term_cfg in actor_atomic_entries
                ]
            }

        by_full_key = {}
        by_leaf_key = {}
        ambiguous_leaf_keys = set()
        for group_name, term_name, term_cfg in actor_atomic_entries:
            full_key = f"{group_name}/{term_name}"
            by_full_key[full_key] = (term_name, term_cfg)
            if term_name in by_leaf_key:
                ambiguous_leaf_keys.add(term_name)
            else:
                by_leaf_key[term_name] = (term_name, term_cfg)

        ordered_atomic_list = []
        for schema_term in schema_terms:
            schema_term_key = str(schema_term)
            if schema_term_key in by_full_key:
                term_name, term_cfg = by_full_key[schema_term_key]
                ordered_atomic_list.append({term_name: term_cfg})
                continue

            leaf_key = schema_term_key.split("/")[-1]
            if leaf_key in ambiguous_leaf_keys:
                raise ValueError(
                    f"Ambiguous obs_schema term '{schema_term_key}': "
                    f"multiple atomic obs share leaf key '{leaf_key}'."
                )
            if leaf_key not in by_leaf_key:
                available = sorted(list(by_leaf_key.keys()))
                raise ValueError(
                    f"obs_schema term '{schema_term_key}' not found in atomic_obs_list. "
                    f"Available terms: {available}"
                )
            term_name, term_cfg = by_leaf_key[leaf_key]
            ordered_atomic_list.append({term_name: term_cfg})

        return {"atomic_obs_list": ordered_atomic_list}

    def _find_actor_place_holder_ndim(self):
        n_dim = 0
        atomic_list = self._get_policy_atomic_obs_list(self.motion_config)[
            "atomic_obs_list"
        ]
        for obs_dict in atomic_list:
            name = str(list(obs_dict.keys())[0])
            if name == "place_holder" or name == "actor_place_holder":
                cfg = obs_dict[name]
                params = cfg.get("params", {}) if isinstance(cfg, dict) else {}
                n_dim = int(params.get("n_dim", 0))
        return n_dim

    # =========== Properties ===========

    def cache_lowstate(self, lowstate_msg, *, force: bool = False) -> None:
        if lowstate_msg is None:
            return
        if not force and lowstate_msg is self._lowstate_cache_msg:
            return
        imu = lowstate_msg.imu_state
        self._robot_root_quat_wxyz_buffer[:] = imu.quaternion
        self._robot_root_ang_vel_buffer[:] = imu.gyroscope
        motor_state = lowstate_msg.motor_state
        for i in range(self.actions_dim):
            state = motor_state[i]
            self._real_dof_pos_buffer[i] = state.q
            self._real_dof_vel_buffer[i] = state.dq
        self._lowstate_cache_msg = lowstate_msg

    @property
    def robot_root_rot_quat_wxyz(self):
        self.cache_lowstate(self._lowstate_msg)
        return self._robot_root_quat_wxyz_buffer

    @property
    def robot_root_ang_vel(self):
        self.cache_lowstate(self._lowstate_msg)
        return self._robot_root_ang_vel_buffer

    @property
    def robot_dof_pos_by_name(self):
        """Get DOF positions by name."""
        if self._lowstate_msg is None:
            return {}
        self.cache_lowstate(self._lowstate_msg)
        return {
            self.real_dof_names[i]: float(self._real_dof_pos_buffer[i])
            for i in range(self.actions_dim)
        }

    @property
    def robot_dof_vel_by_name(self):
        """Get DOF velocities by name."""
        if self._lowstate_msg is None:
            return {}
        self.cache_lowstate(self._lowstate_msg)
        return {
            self.real_dof_names[i]: float(self._real_dof_vel_buffer[i])
            for i in range(self.actions_dim)
        }

    @property
    def ref_motion_frame_idx(self):
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.current_frame_idx(self.motion_frame_idx)
        return min(self.motion_frame_idx, self.n_motion_frames - 1)

    @property
    def ref_dof_pos_raw(self):
        if not self.latest_obs_flag:
            if self._offline_reference is not None and self._offline_reference.has_clip:
                return self._offline_reference.ref_dof_pos_raw(self.motion_frame_idx)
            return self.ref_dof_pos[self.ref_motion_frame_idx]
        if self._vr_reference is not None:
            fallback = (
                self.ref_dof_pos[self.ref_motion_frame_idx]
                if not self._vr_reference.has_latest_obs
                else None
            )
            return self._vr_reference.current_dof_pos(fallback)
        return self.ref_dof_pos[self.ref_motion_frame_idx]

    @property
    def ref_dof_vel_raw(self):
        if not self.latest_obs_flag:
            if self._offline_reference is not None and self._offline_reference.has_clip:
                return self._offline_reference.ref_dof_vel_raw(self.motion_frame_idx)
            return self.ref_dof_vel[self.ref_motion_frame_idx]
        if self._vr_reference is not None:
            fallback = (
                self.ref_dof_vel[self.ref_motion_frame_idx]
                if not self._vr_reference.has_latest_obs
                else None
            )
            return self._vr_reference.current_dof_vel(fallback)
        return self.ref_dof_vel[self.ref_motion_frame_idx]

    @property
    def ref_dof_pos_onnx_order(self):
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.ref_dof_pos_onnx_order(self.motion_frame_idx)
        return self.ref_dof_pos_raw[self.ref_to_onnx]

    @property
    def ref_dof_vel_onnx_order(self):
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.ref_dof_vel_onnx_order(self.motion_frame_idx)
        return self.ref_dof_vel_raw[self.ref_to_onnx]

    @property
    def ref_root_pos_raw(self):
        if not self.latest_obs_flag:
            if self._offline_reference is not None and self._offline_reference.has_clip:
                return self._offline_reference.ref_root_pos_raw(self.motion_frame_idx)
            return np.asarray(
                self.ref_raw_bodylink_pos[self.ref_motion_frame_idx, self.root_body_idx],
                dtype=np.float32,
            )
        if self._vr_reference is not None:
            return self._vr_reference.current_root_pos()
        return np.zeros(3, dtype=np.float32)

    @property
    def root_body_idx(self):
        return 0

    @property
    def last_valid_ref_motion_frame_idx(self):
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.last_valid_frame_idx
        return self.n_motion_frames - 1

    # =========== Policy Obeservation Methods ===========
    def _xyzw_to_wxyz(self, q_xyzw: np.ndarray) -> np.ndarray:
        """Convert quaternions from xyzw to wxyz order."""
        q_xyzw = np.asarray(q_xyzw, dtype=np.float32)
        if q_xyzw.shape[-1] != 4:
            raise ValueError(f"_xyzw_to_wxyz expects (...,4) but got shape {q_xyzw.shape}")
        # q_xyzw[..., 0:3] -> xyz, q_xyzw[..., 3:4] -> w
        w = q_xyzw[..., 3:4]
        xyz = q_xyzw[..., 0:3]
        return np.concatenate([w, xyz], axis=-1)

    def _standardize_quaternion_wxyz(self, q_wxyz: np.ndarray) -> np.ndarray:
        """Standardize quaternion sign so that w >= 0."""
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        if q_wxyz.shape[-1] != 4:
            raise ValueError(f"_standardize_quaternion_wxyz expects (...,4) but got shape {q_wxyz.shape}")
        mask = q_wxyz[..., 0:1] < 0.0
        q_wxyz = np.where(mask, -q_wxyz, q_wxyz)
        return q_wxyz

    def _quat_rotate_wxyz(self, q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        qvec = q_wxyz[..., 1:4]
        w = q_wxyz[..., 0:1]
        t = 2.0 * np.cross(qvec, v)
        return v + w * t + np.cross(qvec, t)

    def _quat_rotate_inv_wxyz(self, q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        n = int(np.prod(q_wxyz.shape[:-1])) if q_wxyz.ndim > 1 else 1
        q_conj = self._q_conj_buffer[:n].reshape(q_wxyz.shape)
        q_conj[..., 0] = q_wxyz[..., 0]
        q_conj[..., 1:4] = -q_wxyz[..., 1:4]
        return self._quat_rotate_wxyz(q_conj, v)

    def _quat_rotate_inv_wxyz_single(
        self, q_wxyz: np.ndarray, v: np.ndarray, out: np.ndarray
    ) -> np.ndarray:
        """Rotate one 3D vector by the inverse quaternion into a preallocated output."""
        qx = -q_wxyz[1]
        qy = -q_wxyz[2]
        qz = -q_wxyz[3]
        qw = q_wxyz[0]
        tx = 2.0 * (qy * v[2] - qz * v[1])
        ty = 2.0 * (qz * v[0] - qx * v[2])
        tz = 2.0 * (qx * v[1] - qy * v[0])
        out[0] = v[0] + qw * tx + (qy * tz - qz * ty)
        out[1] = v[1] + qw * ty + (qz * tx - qx * tz)
        out[2] = v[2] + qw * tz + (qx * ty - qy * tx)
        return out

    @staticmethod
    def _fill_gravity_wxyz(q_wxyz: np.ndarray, out: np.ndarray) -> None:
        qw = q_wxyz[0]
        qx = q_wxyz[1]
        qy = q_wxyz[2]
        qz = q_wxyz[3]
        out[0] = 2.0 * (-qz * qx + qw * qy)
        out[1] = -2.0 * (qz * qy + qw * qx)
        out[2] = 1.0 - 2.0 * (qw * qw + qz * qz)

    @staticmethod
    def _fill_gravity_wxyz_batch(q_wxyz: np.ndarray, out: np.ndarray) -> None:
        qw = q_wxyz[:, 0]
        qx = q_wxyz[:, 1]
        qy = q_wxyz[:, 2]
        qz = q_wxyz[:, 3]
        out[:, 0] = 2.0 * (-qz * qx + qw * qy)
        out[:, 1] = -2.0 * (qz * qy + qw * qx)
        out[:, 2] = 1.0 - 2.0 * (qw * qw + qz * qz)

    def _get_future_frame_indices(self) -> np.ndarray:
        frame_idx = self.ref_motion_frame_idx
        last_valid = self.last_valid_ref_motion_frame_idx
        np.minimum(
            frame_idx + self._future_frame_offsets,
            last_valid,
            out=self._future_frame_indices_buffer,
        )
        return self._future_frame_indices_buffer

    def _cache_fk_vr_for_obs(self):
        """Cache FK outputs used repeatedly during observation construction."""
        fk = getattr(self, "_fk_vr_out", None)
        if not getattr(self, "latest_obs_flag", False) or fk is None:
            self._use_fk_vr = False
            self._fk_vr_cache_ready = False
            return
        if self._fk_vr_cache_ready:
            return
        self._use_fk_vr = True
        T = self.n_fut_frames_int
        rb = self.root_body_idx
        np.copyto(self._fk_vel_0_root, fk["global_velocity"][0, 0, rb])
        np.copyto(self._fk_angvel_0_root, fk["global_angular_velocity"][0, 0, rb])
        np.copyto(self._fk_quat_0_root, fk["global_rotation_quat"][0, 0, rb])
        self._fk_quat_0_root_wxyz[0] = self._fk_quat_0_root[3]
        self._fk_quat_0_root_wxyz[1:4] = self._fk_quat_0_root[:3]
        if self._fk_quat_0_root_wxyz[0] < 0.0:
            self._fk_quat_0_root_wxyz *= -1.0
        self._fill_gravity_wxyz(self._fk_quat_0_root_wxyz, self._gravity_cur_buffer)
        self._quat_rotate_inv_wxyz_single(
            self._fk_quat_0_root_wxyz,
            self._fk_vel_0_root,
            self._base_linvel_cur_buffer,
        )
        self._quat_rotate_inv_wxyz_single(
            self._fk_quat_0_root_wxyz,
            self._fk_angvel_0_root,
            self._base_angvel_cur_buffer,
        )
        trans_0 = fk["global_translation"][0, 0]
        if self._fk_trans_0 is None or self._fk_trans_0.shape != trans_0.shape:
            self._fk_trans_0 = np.empty_like(trans_0)
        np.copyto(self._fk_trans_0, trans_0)
        if T > 0:
            np.copyto(self._fk_vel_fut[:T], fk["global_velocity"][0, 1 : 1 + T, rb])
            np.copyto(self._fk_angvel_fut[:T], fk["global_angular_velocity"][0, 1 : 1 + T, rb])
            np.copyto(self._fk_quat_fut[:T], fk["global_rotation_quat"][0, 1 : 1 + T, rb])
            self._fk_quat_fut_wxyz[:T, 0] = self._fk_quat_fut[:T, 3]
            self._fk_quat_fut_wxyz[:T, 1:4] = self._fk_quat_fut[:T, :3]
            neg = self._fk_quat_fut_wxyz[:T, 0] < 0.0
            self._fk_quat_fut_wxyz[:T][neg] *= -1.0
            trans_fut = fk["global_translation"][0, 1 : 1 + T]
            if self._fk_trans_fut is None or self._fk_trans_fut.shape != trans_fut.shape:
                self._fk_trans_fut = np.empty_like(trans_fut)
            np.copyto(self._fk_trans_fut, trans_fut)
            self._fill_vr_base_linvel_angvel_fut()
        self._fk_vr_cache_ready = True

    def _fill_vr_base_linvel_angvel_fut(self):
        """Rotate future linear and angular velocity buffers in one pass."""
        T = self.n_fut_frames_int
        if T <= 0:
            return
        if T <= 16:
            self._fill_vr_base_linvel_angvel_fut_small(T)
            return
        vel_T6 = self._vel_fut_T6[:T]
        vel_T6[:, :3] = self._fk_vel_fut[:T]
        vel_T6[:, 3:6] = self._fk_angvel_fut[:T]
        q = self._fk_quat_fut_wxyz[:T]
        q_conj = self._q_conj_buffer[:T].reshape(T, 4)
        q_conj[:, 0] = q[:, 0]
        q_conj[:, 1:4] = -q[:, 1:4]
        qvec = q_conj[:, 1:4]
        w = q_conj[:, 0:1]
        rt = self._rot_t_buffer[:T]
        rc = self._rot_cross_buffer[:T]
        rt[:, 0] = 2.0 * (qvec[:, 1] * vel_T6[:, 2] - qvec[:, 2] * vel_T6[:, 1])
        rt[:, 1] = 2.0 * (qvec[:, 2] * vel_T6[:, 0] - qvec[:, 0] * vel_T6[:, 2])
        rt[:, 2] = 2.0 * (qvec[:, 0] * vel_T6[:, 1] - qvec[:, 1] * vel_T6[:, 0])
        rc[:, 0] = qvec[:, 1] * rt[:, 2] - qvec[:, 2] * rt[:, 1]
        rc[:, 1] = qvec[:, 2] * rt[:, 0] - qvec[:, 0] * rt[:, 2]
        rc[:, 2] = qvec[:, 0] * rt[:, 1] - qvec[:, 1] * rt[:, 0]
        self._base_linvel_fut_buffer[:T] = vel_T6[:, :3] + w * rt + rc
        rt[:, 0] = 2.0 * (qvec[:, 1] * vel_T6[:, 5] - qvec[:, 2] * vel_T6[:, 4])
        rt[:, 1] = 2.0 * (qvec[:, 2] * vel_T6[:, 3] - qvec[:, 0] * vel_T6[:, 5])
        rt[:, 2] = 2.0 * (qvec[:, 0] * vel_T6[:, 4] - qvec[:, 1] * vel_T6[:, 3])
        rc[:, 0] = qvec[:, 1] * rt[:, 2] - qvec[:, 2] * rt[:, 1]
        rc[:, 1] = qvec[:, 2] * rt[:, 0] - qvec[:, 0] * rt[:, 2]
        rc[:, 2] = qvec[:, 0] * rt[:, 1] - qvec[:, 1] * rt[:, 0]
        self._base_angvel_fut_buffer[:T] = vel_T6[:, 3:6] + w * rt + rc
        self._fill_gravity_wxyz_batch(q, self._gravity_fut_buffer[:T])

    def _fill_vr_base_linvel_angvel_fut_small(self, T: int) -> None:
        for i in range(T):
            qw = float(self._fk_quat_fut_wxyz[i, 0])
            qx = float(self._fk_quat_fut_wxyz[i, 1])
            qy = float(self._fk_quat_fut_wxyz[i, 2])
            qz = float(self._fk_quat_fut_wxyz[i, 3])
            vx = float(self._fk_vel_fut[i, 0])
            vy = float(self._fk_vel_fut[i, 1])
            vz = float(self._fk_vel_fut[i, 2])
            (
                self._base_linvel_fut_buffer[i, 0],
                self._base_linvel_fut_buffer[i, 1],
                self._base_linvel_fut_buffer[i, 2],
            ) = self._quat_rotate_inv_wxyz_values(qw, qx, qy, qz, vx, vy, vz)

            wx = float(self._fk_angvel_fut[i, 0])
            wy = float(self._fk_angvel_fut[i, 1])
            wz = float(self._fk_angvel_fut[i, 2])
            (
                self._base_angvel_fut_buffer[i, 0],
                self._base_angvel_fut_buffer[i, 1],
                self._base_angvel_fut_buffer[i, 2],
            ) = self._quat_rotate_inv_wxyz_values(qw, qx, qy, qz, wx, wy, wz)

            self._gravity_fut_buffer[i, 0] = 2.0 * (-qz * qx + qw * qy)
            self._gravity_fut_buffer[i, 1] = -2.0 * (qz * qy + qw * qx)
            self._gravity_fut_buffer[i, 2] = 1.0 - 2.0 * (qw * qw + qz * qz)

    @staticmethod
    def _quat_rotate_inv_wxyz_values(
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        vx: float,
        vy: float,
        vz: float,
    ) -> tuple[float, float, float]:
        qx = -qx
        qy = -qy
        qz = -qz
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return (
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )

    def _prepare_vr_fk_tensors(
        self,
        vr_reference: VrLatestObsReference,
        cur_root_pos: np.ndarray,
        cur_root_rot: np.ndarray,
        cur_dof_pos: np.ndarray,
        n_fut: int,
    ):
        """Fill preallocated FK input buffers and return torch views without reallocation."""
        if (
            n_fut <= 0
            or self._fk_root_pos_seq_np is None
            or self._fk_root_rot_seq_np is None
            or self._fk_dof_pos_seq_np is None
        ):
            raise ValueError("VR FK sequence buffers are not initialized")

        copied = vr_reference.copy_fk_sequence_inputs(
            root_pos_seq=self._fk_root_pos_seq_np,
            root_rot_seq=self._fk_root_rot_seq_np,
            dof_pos_seq=self._fk_dof_pos_seq_np,
            cur_root_pos=cur_root_pos,
            cur_root_rot=cur_root_rot,
            cur_dof_pos=cur_dof_pos,
            n_frames=n_fut,
        )
        if not copied:
            raise ValueError("VR latest_obs future sequence is not ready for FK")
        return (
            self._fk_root_pos_seq_tensor,
            self._fk_root_rot_seq_tensor,
            self._fk_dof_pos_seq_tensor,
        )

    def _prepare_vr_fk_arrays(
        self,
        vr_reference: VrLatestObsReference,
        cur_root_pos: np.ndarray,
        cur_root_rot: np.ndarray,
        cur_dof_pos: np.ndarray,
        n_fut: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            n_fut <= 0
            or self._fk_root_pos_seq_np is None
            or self._fk_root_rot_seq_np is None
            or self._fk_dof_pos_seq_np is None
        ):
            raise ValueError("VR FK sequence buffers are not initialized")

        copied = vr_reference.copy_fk_sequence_inputs(
            root_pos_seq=self._fk_root_pos_seq_np,
            root_rot_seq=self._fk_root_rot_seq_np,
            dof_pos_seq=self._fk_dof_pos_seq_np,
            cur_root_pos=cur_root_pos,
            cur_root_rot=cur_root_rot,
            cur_dof_pos=cur_dof_pos,
            n_frames=n_fut,
        )
        if not copied:
            raise ValueError("VR latest_obs future sequence is not ready for FK")
        return (
            self._fk_root_pos_seq_np,
            self._fk_root_rot_seq_np,
            self._fk_dof_pos_seq_np,
        )

    def _compute_and_cache_vr_root_fk(
        self,
        *,
        vr_reference: VrLatestObsReference,
        cur_root_pos: np.ndarray,
        cur_root_rot: np.ndarray,
        cur_dof_pos: np.ndarray | None = None,
        n_fut: int,
        fps: float,
    ) -> None:
        """Compute root-only FK directly into observation caches."""
        if fps <= 0.0:
            raise ValueError(f"Invalid fps: {fps}")

        if (
            n_fut <= 0
            or self._fk_root_pos_seq_np is None
            or self._fk_root_rot_seq_np is None
            or not vr_reference.has_future_sequence(n_fut)
        ):
            raise ValueError("VR latest_obs future sequence is not ready for FK")
        root_pos_seq = self._fk_root_pos_seq_np
        root_rot_seq = self._fk_root_rot_seq_np
        np.copyto(root_pos_seq[0, 0], cur_root_pos)
        np.copyto(root_rot_seq[0, 0], cur_root_rot)
        np.copyto(root_pos_seq[0, 1 : 1 + n_fut], vr_reference.root_pos_queue[:n_fut])
        np.copyto(root_rot_seq[0, 1 : 1 + n_fut], vr_reference.root_rot_queue[:n_fut])
        pos = root_pos_seq[0, : n_fut + 1]
        quat = root_rot_seq[0, : n_fut + 1]

        self._standardize_quat_sequence(quat, n_fut)

        dt = 1.0 / float(fps)
        inv_dt = float(fps)

        self._fill_root_linear_velocity_seq(pos, n_fut, inv_dt)
        self._fill_root_angular_velocity_seq(quat, n_fut, dt)

        self._use_fk_vr = True
        T = int(n_fut)
        np.copyto(self._fk_vel_0_root, self._fk_root_vel_seq_np[0])
        np.copyto(self._fk_angvel_0_root, self._fk_root_angvel_seq_np[0])
        self._fk_quat_0_root_wxyz[:] = quat[0]
        self._fill_gravity_wxyz(self._fk_quat_0_root_wxyz, self._gravity_cur_buffer)
        self._quat_rotate_inv_wxyz_single(
            self._fk_quat_0_root_wxyz,
            self._fk_vel_0_root,
            self._base_linvel_cur_buffer,
        )
        self._quat_rotate_inv_wxyz_single(
            self._fk_quat_0_root_wxyz,
            self._fk_angvel_0_root,
            self._base_angvel_cur_buffer,
        )

        if self._fk_trans_0 is None or self._fk_trans_0.shape != (1, 3):
            self._fk_trans_0 = np.empty((1, 3), dtype=np.float32)
        self._fk_trans_0[0] = pos[0]
        if T > 0:
            np.copyto(self._fk_vel_fut[:T], self._fk_root_vel_seq_np[1 : 1 + T])
            np.copyto(
                self._fk_angvel_fut[:T],
                self._fk_root_angvel_seq_np[1 : 1 + T],
            )
            self._fk_quat_fut_wxyz[:T] = quat[1 : 1 + T]
            if self._fk_trans_fut is None or self._fk_trans_fut.shape != (T, 1, 3):
                self._fk_trans_fut = np.empty((T, 1, 3), dtype=np.float32)
            self._fk_trans_fut[:, 0, :] = pos[1 : 1 + T]
            self._fill_vr_base_linvel_angvel_fut()
        self._fk_vr_cache_ready = True
        self._fk_vr_out = None

    def _standardize_quat_sequence(self, quat_wxyz: np.ndarray, n_fut: int) -> None:
        q = quat_wxyz[: n_fut + 1]
        neg = q[:, 0] < 0.0
        q[neg] *= -1.0

    def _fill_root_linear_velocity_seq(
        self, pos: np.ndarray, n_fut: int, inv_dt: float
    ) -> None:
        vel = self._fk_root_vel_seq_np[: n_fut + 1]
        if n_fut <= 0:
            vel[0].fill(0.0)
            return
        vel[0] = (pos[1] - pos[0]) * inv_dt
        if n_fut > 1:
            vel[1:n_fut] = (pos[2 : n_fut + 1] - pos[: n_fut - 1]) * (
                0.5 * inv_dt
            )
        vel[n_fut] = (pos[n_fut] - pos[n_fut - 1]) * inv_dt

    def _fill_root_angular_velocity_seq(
        self, quat_wxyz: np.ndarray, n_fut: int, dt: float
    ) -> None:
        angvel = self._fk_root_angvel_seq_np[: n_fut + 1]
        angvel.fill(0.0)
        if n_fut <= 0:
            return
        if n_fut <= 16:
            self._fill_root_angular_velocity_seq_small(quat_wxyz, n_fut, dt, angvel)
            return

        q0 = quat_wxyz[:n_fut]
        q1 = quat_wxyz[1 : n_fut + 1]
        rel = self._fk_q_rel_buffer[:n_fut]
        w1 = q1[:, 0]
        x1 = q1[:, 1]
        y1 = q1[:, 2]
        z1 = q1[:, 3]
        w0 = q0[:, 0]
        x0 = q0[:, 1]
        y0 = q0[:, 2]
        z0 = q0[:, 3]
        rel[:, 0] = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
        rel[:, 1] = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
        rel[:, 2] = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
        rel[:, 3] = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
        self._axis_angle_from_wxyz_inplace(rel, self._fk_axis_angle_buffer[:n_fut])
        angvel[:n_fut] = self._fk_axis_angle_buffer[:n_fut] / dt

    @staticmethod
    def _fill_root_angular_velocity_seq_small(
        quat_wxyz: np.ndarray,
        n_fut: int,
        dt: float,
        angvel: np.ndarray,
    ) -> None:
        for i in range(n_fut):
            w1 = float(quat_wxyz[i + 1, 0])
            x1 = float(quat_wxyz[i + 1, 1])
            y1 = float(quat_wxyz[i + 1, 2])
            z1 = float(quat_wxyz[i + 1, 3])
            w0 = float(quat_wxyz[i, 0])
            x0 = float(quat_wxyz[i, 1])
            y0 = float(quat_wxyz[i, 2])
            z0 = float(quat_wxyz[i, 3])
            rw = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
            rx = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
            ry = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
            rz = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
            if rw < 0.0:
                rw = -rw
                rx = -rx
                ry = -ry
                rz = -rz
            norm = math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
            norm = max(norm, 1.0e-9)
            rw /= norm
            rx /= norm
            ry /= norm
            rz /= norm
            mag = math.sqrt(rx * rx + ry * ry + rz * rz)
            half_angle = math.atan2(mag, rw)
            angle = 2.0 * half_angle
            if abs(angle) <= 1.0e-6:
                sin_half_over_angle = 0.5 - angle * angle / 48.0
            else:
                sin_half_over_angle = math.sin(half_angle) / angle
            scale = 1.0 / (sin_half_over_angle * dt)
            angvel[i, 0] = rx * scale
            angvel[i, 1] = ry * scale
            angvel[i, 2] = rz * scale

    @staticmethod
    def _axis_angle_from_wxyz_inplace(q_wxyz: np.ndarray, out: np.ndarray) -> None:
        neg = q_wxyz[:, 0] < 0.0
        q_wxyz[neg] *= -1.0
        norm = np.sqrt(np.sum(q_wxyz * q_wxyz, axis=1))
        norm = np.maximum(norm, 1.0e-9)
        q_wxyz /= norm[:, None]

        quat_w = q_wxyz[:, 0]
        quat_xyz = q_wxyz[:, 1:4]
        mag = np.sqrt(np.sum(quat_xyz * quat_xyz, axis=1))
        half_angle = np.arctan2(mag, quat_w)
        angle = 2.0 * half_angle
        use_taylor = np.abs(angle) <= 1.0e-6
        angle_safe = np.where(use_taylor, 1.0, angle)
        sin_half_over_angle = np.where(
            use_taylor,
            0.5 - angle * angle / 48.0,
            np.sin(half_angle) / angle_safe,
        )
        out[:] = quat_xyz / sin_half_over_angle[:, None]

    def _get_future_root_quat_wxyz(self) -> np.ndarray:
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable; future_root_quat_wxyz will return zeros."
            )
            return self._future_root_quat_wxyz_buffer

        fut_idx = self._get_future_frame_indices()
        q_root_xyzw = np.asarray(
            self.ref_raw_bodylink_rot[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        q_root_wxyz = self._future_root_quat_wxyz_buffer
        q_root_wxyz[:, 0] = q_root_xyzw[:, 3]
        q_root_wxyz[:, 1] = q_root_xyzw[:, 0]
        q_root_wxyz[:, 2] = q_root_xyzw[:, 1]
        q_root_wxyz[:, 3] = q_root_xyzw[:, 2]
        neg_mask = q_root_wxyz[:, 0] < 0.0
        q_root_wxyz[neg_mask] *= -1.0
        return self._future_root_quat_wxyz_buffer

    def _get_ref_keybody_indices(self, term_name: str) -> np.ndarray:
        keybody_idxs = self._keybody_indices_by_term_name.get(term_name, None)
        if keybody_idxs is None:
            raise ValueError(
                f"Keybody indices for term '{term_name}' were not cached. "
                "Ensure the term exists in motion policy obs and cache is initialized."
            )
        return keybody_idxs

    def _get_obs_actor_velocity_command(self):
        return self._get_obs_velocity_command()

    def _get_obs_actor_projected_gravity(self):
        return self._get_obs_projected_gravity()

    def _get_obs_actor_rel_robot_root_ang_vel(self):
        return self._get_obs_rel_robot_root_ang_vel()

    def _get_obs_actor_dof_pos(self):
        return self._get_obs_dof_pos()

    def _get_obs_actor_dof_vel(self):
        return self._get_obs_dof_vel()

    def _get_obs_actor_last_action(self):
        return self._get_obs_last_action()

    def _get_obs_actor_ref_gravity_projection_cur(self):
        return self._get_obs_ref_gravity_projection_cur()

    def _get_obs_actor_ref_gravity_projection_fut(self):
        return self._get_obs_ref_gravity_projection_fut()

    def _get_obs_actor_ref_base_linvel_cur(self):
        return self._get_obs_ref_base_linvel_cur()

    def _get_obs_actor_ref_base_linvel_fut(self):
        return self._get_obs_ref_base_linvel_fut()

    def _get_obs_actor_ref_base_angvel_cur(self):
        return self._get_obs_ref_base_angvel_cur()

    def _get_obs_actor_ref_base_angvel_fut(self):
        return self._get_obs_ref_base_angvel_fut()

    def _get_obs_actor_ref_dof_pos_cur(self):
        return self._get_obs_ref_dof_pos_cur()

    def _get_obs_actor_ref_dof_pos_fut(self):
        return self._get_obs_ref_dof_pos_fut()

    def _get_obs_actor_ref_root_height_cur(self):
        return self._get_obs_ref_root_height_cur()

    def _get_obs_actor_ref_root_height_fut(self):
        return self._get_obs_ref_root_height_fut()

    def _get_obs_actor_ref_keybody_rel_pos_cur(self):
        return self._get_obs_ref_keybody_rel_pos_cur()

    def _get_obs_actor_ref_keybody_rel_pos_fut(self):
        return self._get_obs_ref_keybody_rel_pos_fut()



    def _get_obs_velocity_command(self):
        """Get velocity command observation (reuses pre-allocated array)."""
        vx = float(self.vx)
        vy = float(self.vy)
        vyaw = float(self.vyaw)
        self._velocity_cmd_obs[1] = vx
        self._velocity_cmd_obs[2] = vy
        self._velocity_cmd_obs[3] = vyaw
        self._velocity_cmd_obs[0] = float(vx * vx + vy * vy + vyaw * vyaw > 0.01)
        return self._velocity_cmd_obs

    def _get_obs_projected_gravity(self):
        self._fill_gravity_wxyz(
            self.robot_root_rot_quat_wxyz,
            self._projected_gravity_buffer,
        )
        return self._projected_gravity_buffer

    def _get_obs_rel_robot_root_ang_vel(self):
        return self.robot_root_ang_vel

    def _get_obs_dof_pos(self):
        """Get DOF position observation (pre-allocated buffer + index lookup, no dict/list)."""
        if self._lowstate_msg is None:
            return self._dof_pos_obs_buffer[: len(self.motion_dof_names_onnx)]
        self.cache_lowstate(self._lowstate_msg)
        if self.current_policy_mode == "motion":
            buf = self._dof_pos_obs_buffer
            def_angles = self.motion_default_angles_onnx
            n = len(self.motion_dof_names_onnx)
            out = buf[:n]
            np.take(self._real_dof_pos_buffer, self.motion_dof_real_indices_np, out=out)
            np.subtract(out, def_angles, out=out)
            return out
        def_angles = self.velocity_default_angles_onnx
        n = len(self.velocity_dof_names_onnx)
        out = self._dof_pos_obs_buffer[:n]
        np.take(self._real_dof_pos_buffer, self.velocity_dof_real_indices_np, out=out)
        np.subtract(out, def_angles, out=out)
        return out

    def _get_obs_dof_vel(self):
        """Get DOF velocity observation (pre-allocated buffer + index lookup, no dict/list)."""
        if self._lowstate_msg is None:
            return self._dof_vel_obs_buffer[: len(self.motion_dof_names_onnx)]
        self.cache_lowstate(self._lowstate_msg)
        if self.current_policy_mode == "motion":
            buf = self._dof_vel_obs_buffer
            n = len(self.motion_dof_names_onnx)
            out = buf[:n]
            np.take(self._real_dof_vel_buffer, self.motion_dof_real_indices_np, out=out)
            return out
        n = len(self.velocity_dof_names_onnx)
        out = self._dof_vel_obs_buffer[:n]
        np.take(self._real_dof_vel_buffer, self.velocity_dof_real_indices_np, out=out)
        return out

    def _get_obs_last_action(self):
        return self.actions_onnx

    def _get_obs_ref_motion_states(self):
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_motion_states(self.motion_frame_idx)
        n = self.num_actions
        self._ref_motion_states_buffer[:n] = self.ref_dof_pos_onnx_order
        self._ref_motion_states_buffer[n:] = self.ref_dof_vel_onnx_order
        return self._ref_motion_states_buffer

    def _get_obs_ref_dof_pos_fut(self):
        """Get future DOF position observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "latest_obs_flag", False) and self._vr_reference is not None:
            return self._vr_reference.obs_ref_dof_pos_fut(
                ref_to_onnx=self.ref_to_onnx,
                pos_fut_buffer=self._pos_fut_buffer,
                n_frames=T,
            )
        if getattr(self, "latest_obs_flag", False):
            return np.zeros(self.num_actions * T, dtype=np.float32)
        if self._offline_reference is not None and self._offline_reference.has_clip:
            return self._offline_reference.obs_ref_dof_pos_fut(self.motion_frame_idx)
        if not hasattr(self, "ref_dof_pos") or self.ref_dof_pos is None:
            self.get_logger().warn(
                "[VR] ref_dof_pos is unavailable and latest_obs is not active; returning zeros for ref_dof_pos_fut."
            )
            return np.zeros(self.num_actions * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        pos_fut = self._pos_fut_buffer
        pos_fut[:, :] = self.ref_dof_pos[fut_idx].T
        # Reorder to ONNX and flatten per training layout
        pos_fut_onnx = pos_fut[self.ref_to_onnx, :].transpose(1, 0)  # [N, T]
        return pos_fut_onnx.reshape(-1).astype(np.float32)

    def _get_obs_ref_root_height_fut(self):
        """Get future root height observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.latest_obs_flag and self._vr_reference is not None:
            return self._vr_reference.obs_ref_root_height_fut(n_frames=T)
        if self.latest_obs_flag:
            return np.zeros(T, dtype=np.float32)
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_root_height_fut(self.motion_frame_idx)
        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_root_height_fut."
            )
            return np.zeros(T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        h_fut = self._h_fut_buffer
        h_fut[0, :] = self.ref_raw_bodylink_pos[fut_idx, self.root_body_idx, 2]
        return h_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_root_pos_fut(self):
        """Get future root position observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.latest_obs_flag and self._vr_reference is not None:
            return self._vr_reference.obs_ref_root_pos_fut(n_frames=T)
        if self.latest_obs_flag:
            return np.zeros(3 * T, dtype=np.float32)
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_root_pos_fut(self.motion_frame_idx)
        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_root_pos_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        pos_fut = self._root_pos_fut_buffer
        pos_fut[:, :] = self.ref_raw_bodylink_pos[fut_idx, self.root_body_idx, :]
        return pos_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_dof_pos_cur(self):
        return self.ref_dof_pos_onnx_order

    def _get_obs_ref_dof_vel_cur(self):
        return self.ref_dof_vel_onnx_order

    def _get_obs_ref_root_height_cur(self):
        if not self.latest_obs_flag:
            if self._offline_reference is not None and self._offline_reference.has_clip:
                return self._offline_reference.obs_ref_root_height_cur(self.motion_frame_idx)
            return self.ref_raw_bodylink_pos[
                self.ref_motion_frame_idx, self.root_body_idx, 2
            ]
        return float(self.ref_root_pos_raw[2])

    def _get_obs_ref_root_pos_cur(self):
        return self.ref_root_pos_raw.astype(np.float32)

    def _get_obs_ref_gravity_projection_cur(self):
        if getattr(self, "_use_fk_vr", False):
            return self._gravity_cur_buffer
        if (
            getattr(self, "latest_obs_flag", False)
            and self._vr_reference is not None
            and self._vr_reference.has_latest_obs
        ):
            q_root_wxyz = self._vr_reference.latest_root_rot()
            if q_root_wxyz is None:
                return np.zeros(3, dtype=np.float32)
            q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
            return get_gravity_orientation(q_root_wxyz)
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_gravity_projection_cur(
                self.motion_frame_idx
            )
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for gravity_projection_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[self.ref_motion_frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        return get_gravity_orientation(q_root_wxyz)

    def _get_obs_ref_gravity_projection_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False):
            return self._gravity_fut_buffer[:T].reshape(-1)
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_gravity_projection_fut(
                self.motion_frame_idx
            )
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_gravity_projection_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        q_root_wxyz = self._get_future_root_quat_wxyz()
        gravity_fut = self._gravity_fut_buffer
        qw = q_root_wxyz[:, 0]
        qx = q_root_wxyz[:, 1]
        qy = q_root_wxyz[:, 2]
        qz = q_root_wxyz[:, 3]
        gravity_fut[:, 0] = 2.0 * (-qz * qx + qw * qy)
        gravity_fut[:, 1] = -2.0 * (qz * qy + qw * qx)
        gravity_fut[:, 2] = 1.0 - 2.0 * (qw * qw + qz * qz)
        return gravity_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_base_linvel_cur(self):
        if getattr(self, "_use_fk_vr", False):
            return self._base_linvel_cur_buffer
        if (
            getattr(self, "latest_obs_flag", False)
            and self._vr_reference is not None
            and self._vr_reference.has_latest_obs
        ):
            return np.zeros(3, dtype=np.float32)
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_linvel_cur(self.motion_frame_idx)
        if not hasattr(self, "ref_global_velocity") or self.ref_global_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[self.ref_motion_frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        v_root_w = np.asarray(
            self.ref_global_velocity[self.ref_motion_frame_idx, self.root_body_idx],
            dtype=np.float32,
        )
        v_root = self._quat_rotate_inv_wxyz(q_root_wxyz, v_root_w)
        return np.asarray(v_root, dtype=np.float32).reshape(3)

    def _get_obs_ref_base_linvel_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False):
            return self._base_linvel_fut_buffer[:T].reshape(-1)

        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_linvel_fut(self.motion_frame_idx)
        if not hasattr(self, "ref_global_velocity") or self.ref_global_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_linvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        v_root_w = np.asarray(
            self.ref_global_velocity[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        base_linvel_fut = self._base_linvel_fut_buffer
        base_linvel_fut[:, :] = self._quat_rotate_inv_wxyz(q_root_wxyz, v_root_w)
        return base_linvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_base_angvel_cur(self):
        if getattr(self, "_use_fk_vr", False):
            return self._base_angvel_cur_buffer
        if (
            getattr(self, "latest_obs_flag", False)
            and self._vr_reference is not None
            and self._vr_reference.has_latest_obs
        ):
            return np.zeros(3, dtype=np.float32)
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_angvel_cur(self.motion_frame_idx)
        if not hasattr(self, "ref_global_angular_velocity") or self.ref_global_angular_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_angular_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[self.ref_motion_frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        w_root_w = np.asarray(
            self.ref_global_angular_velocity[self.ref_motion_frame_idx, self.root_body_idx],
            dtype=np.float32,
        )
        w_root = self._quat_rotate_inv_wxyz(q_root_wxyz, w_root_w)
        return np.asarray(w_root, dtype=np.float32).reshape(3)

    def _get_obs_ref_base_angvel_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False):
            return self._base_angvel_fut_buffer[:T].reshape(-1)

        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_angvel_fut(self.motion_frame_idx)
        if not hasattr(self, "ref_global_angular_velocity") or self.ref_global_angular_velocity is None:
            self.get_logger().warn(
                "[VR] ref_global_angular_velocity is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_base_angvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        w_root_w = np.asarray(
            self.ref_global_angular_velocity[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        base_angvel_fut = self._base_angvel_fut_buffer
        base_angvel_fut[:, :] = self._quat_rotate_inv_wxyz(q_root_wxyz, w_root_w)
        return base_angvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_keybody_rel_pos_cur(self):
        if getattr(self, "_use_fk_vr", False) and self._fk_trans_0 is not None:
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            if not self._root_only_fk_has_required_keybodies(keybody_idxs):
                return np.zeros(3 * n_keybodies, dtype=np.float32)
            root_pos = self._fk_trans_0[self.root_body_idx]
            keybody_pos = self._fk_trans_0[keybody_idxs]
            rel_pos_w = keybody_pos - root_pos[None, :]
            rel_pos_root = self._quat_rotate_inv_wxyz(self._fk_quat_0_root_wxyz, rel_pos_w)
            return np.asarray(rel_pos_root, dtype=np.float32).reshape(-1)

        if (
            getattr(self, "latest_obs_flag", False)
            and self._vr_reference is not None
            and self._vr_reference.has_latest_obs
        ):
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)

        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            return self._offline_reference.obs_ref_keybody_rel_pos_cur(
                self.motion_frame_idx,
                keybody_idxs,
            )
        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_cur."
            )
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_cur."
            )
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)

        keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_cur")
        n_keybodies = int(keybody_idxs.shape[0])
        if n_keybodies == 0:
            return np.zeros(0, dtype=np.float32)

        frame_idx = self.ref_motion_frame_idx
        ref_body_global_pos = np.asarray(self.ref_raw_bodylink_pos[frame_idx], dtype=np.float32)
        ref_root_global_pos = ref_body_global_pos[self.root_body_idx]
        q_root_xyzw = self.ref_raw_bodylink_rot[frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)

        rel_pos_w = ref_body_global_pos[keybody_idxs] - ref_root_global_pos[None, :]
        rel_pos_root = self._quat_rotate_inv_wxyz(q_root_wxyz, rel_pos_w)
        return np.asarray(rel_pos_root, dtype=np.float32).reshape(-1)

    def _get_obs_ref_keybody_rel_pos_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if getattr(self, "_use_fk_vr", False) and self._fk_trans_fut is not None:
            keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_fut")
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            if not self._root_only_fk_has_required_keybodies(keybody_idxs):
                return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)
            ref_body = self._fk_trans_fut[:T]  # (T, num_bodies, 3)
            ref_root = ref_body[:, self.root_body_idx, :]  # (T, 3)
            if self._keybody_rel_pos_fut_buffer.shape[1] != n_keybodies:
                self._keybody_rel_pos_fut_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
                self._keybody_rel_pos_w_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
            elif (
                self._keybody_rel_pos_w_buffer is None
                or self._keybody_rel_pos_w_buffer.shape[0] < T
                or self._keybody_rel_pos_w_buffer.shape[1] != n_keybodies
            ):
                self._keybody_rel_pos_w_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
            rel_pos_fut = self._keybody_rel_pos_fut_buffer
            np.subtract(
                ref_body[:, keybody_idxs, :],
                ref_root[:, None, :],
                out=self._keybody_rel_pos_w_buffer[:T, :n_keybodies, :],
            )
            rel_pos_fut[:, :, :] = self._quat_rotate_inv_wxyz(
                self._fk_quat_fut_wxyz[:T, None, :],
                self._keybody_rel_pos_w_buffer[:T, :n_keybodies, :],
            )
            return rel_pos_fut.reshape(-1).astype(np.float32)
        keybody_idxs = self._get_ref_keybody_indices("actor_ref_keybody_rel_pos_fut")
        n_keybodies = int(keybody_idxs.shape[0])
        if (
            not getattr(self, "latest_obs_flag", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_keybody_rel_pos_fut(
                self.motion_frame_idx,
                keybody_idxs,
            )
        if not hasattr(self, "ref_raw_bodylink_pos") or self.ref_raw_bodylink_pos is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_fut."
            )
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)
        if not hasattr(self, "ref_raw_bodylink_rot") or self.ref_raw_bodylink_rot is None:
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and latest_obs is not active; returning zeros for ref_keybody_rel_pos_fut."
            )
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)

        if n_keybodies == 0:
            return np.zeros((T, 0), dtype=np.float32).reshape(-1)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        ref_body_global_pos = np.asarray(self.ref_raw_bodylink_pos[fut_idx], dtype=np.float32)
        ref_root_global_pos = ref_body_global_pos[:, self.root_body_idx, :]
        rel_pos_w = ref_body_global_pos[:, keybody_idxs, :] - ref_root_global_pos[:, None, :]
        if self._keybody_rel_pos_fut_buffer.shape[1] != n_keybodies:
            self._keybody_rel_pos_fut_buffer = np.zeros((T, n_keybodies, 3), dtype=np.float32)
        rel_pos_fut = self._keybody_rel_pos_fut_buffer
        rel_pos_fut[:, :, :] = self._quat_rotate_inv_wxyz(q_root_wxyz[:, None, :], rel_pos_w)
        return rel_pos_fut.reshape(-1).astype(np.float32)

    def _get_obs_place_holder(self):
        expected_dim = int(getattr(self, "actor_place_holder_ndim", 0) or 0)
        if (
            not hasattr(self, "_place_holder_buffer")
            or self._place_holder_buffer.shape[0] != expected_dim
        ):
            self._place_holder_buffer = np.zeros(expected_dim, dtype=np.float32)
        return self._place_holder_buffer

    # =========== Policy Obeservation Methods ===========

    def _warmup_fk_for_vr(self):
        """Run one FK warmup step when entering VR motion mode."""
        try:
            if (
                getattr(self, "fk", None) is None
                or not getattr(self, "fk_initialized", False)
            ):
                return
            vr_reference = self._vr_reference
            if vr_reference is None or not vr_reference.has_latest_obs:
                return

            n_fut = int(getattr(self, "n_fut_frames", 0))
            if n_fut <= 0 or not vr_reference.has_future_sequence(n_fut):
                return

            cur_root_pos = vr_reference.latest_root_pos()
            cur_root_rot = vr_reference.latest_root_rot()
            cur_dof_pos = vr_reference.latest_dof_pos()
            if cur_root_pos is None or cur_root_rot is None or cur_dof_pos is None:
                return
            root_pos_tensor, root_rot_tensor, dof_pos_tensor = (
                self._prepare_vr_fk_tensors(
                    vr_reference=vr_reference,
                    cur_root_pos=cur_root_pos,
                    cur_root_rot=cur_root_rot,
                    cur_dof_pos=cur_dof_pos,
                    n_fut=n_fut,
                )
            )

            fk_out = self.fk(
                root_pos=root_pos_tensor,
                root_quat=root_rot_tensor,
                dof_pos=dof_pos_tensor,
                fps=float(1.0 / self.dt),
                quat_format="wxyz",
                vel_smoothing_sigma=0.0,
                compute_velocity=False,
            )
            self._fk_vr_out = {
                k: v.detach().cpu().numpy() for k, v in fk_out.items()
            }
        except Exception as e:
            self.get_logger().warn(f"[VR] FK warmup failed, fallback to zeros: {e}")


    def _root_only_fk_has_required_keybodies(self, keybody_idxs: np.ndarray) -> bool:
        if keybody_idxs.size == 0:
            return True
        available_bodies = 0 if self._fk_trans_0 is None else int(self._fk_trans_0.shape[0])
        if available_bodies <= int(np.max(keybody_idxs)):
            if not self._root_only_fk_keybody_warned:
                self.get_logger().warn(
                    "[RootOnlyFK] FK output only contains root body, but obs schema still "
                    "requests non-root keybody positions. Returning zeros for keybody obs."
                )
                self._root_only_fk_keybody_warned = True
            return False
        return True
