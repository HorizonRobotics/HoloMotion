"""Observation formulas for the 29DOF policy node."""

from __future__ import annotations

import numpy as np

from humanoid_policy.offline_motion_reference import OfflineMotionReference
from humanoid_policy.offline_motion_reference import yaw_from_quat_wxyz


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

    def clear_vr_reference_cache(self) -> None:
        if getattr(self, "_vr_reference", None) is not None:
            self._vr_reference.kinematics = None

    def initialize_vr_reference_buffers(self, n_fut: int, num_actions: int) -> None:
        n_fut = int(n_fut)
        num_actions = int(num_actions)
        if n_fut > 0:
            self.get_logger().info(
                f"Initialized VR future frame queues: n_fut_frames={n_fut}, num_actions={num_actions}"
            )

    def initialize_observation_state(self) -> None:
        node = self._node
        self.ref_to_onnx = [
            node.dof_names_ref_motion.index(name)
            for name in node.motion_dof_names_onnx
        ]

        self.velocity_default_angles_dict = {
            name: float(node.velocity_default_angles_onnx[idx])
            for idx, name in enumerate(node.velocity_dof_names_onnx)
        }
        self.motion_default_angles_dict = {
            name: float(node.motion_default_angles_onnx[idx])
            for idx, name in enumerate(node.motion_dof_names_onnx)
        }
        self.velocity_dof_names_onnx_array = np.array(
            node.velocity_dof_names_onnx
        )
        self.motion_dof_names_onnx_array = np.array(node.motion_dof_names_onnx)
        self.motion_dof_real_indices = [
            node.real_dof_names.index(name)
            for name in node.motion_dof_names_onnx
        ]
        self.velocity_dof_real_indices = [
            node.real_dof_names.index(name)
            for name in node.velocity_dof_names_onnx
        ]
        self.motion_dof_real_indices_np = np.asarray(
            self.motion_dof_real_indices,
            dtype=np.int64,
        )
        self.velocity_dof_real_indices_np = np.asarray(
            self.velocity_dof_real_indices,
            dtype=np.int64,
        )

        n_dof = max(
            len(node.motion_dof_names_onnx), len(node.velocity_dof_names_onnx)
        )
        self._dof_pos_obs_buffer = np.zeros(n_dof, dtype=np.float32)
        self._dof_vel_obs_buffer = np.zeros(n_dof, dtype=np.float32)
        self._real_dof_pos_buffer = np.zeros(
            node.actions_dim, dtype=np.float32
        )
        self._real_dof_vel_buffer = np.zeros(
            node.actions_dim, dtype=np.float32
        )
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
            self._h_fut_buffer = np.zeros(
                (1, self.n_fut_frames_int), dtype=np.float32
            )
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
        self._future_yaw_delta_sin_cos_buffer = np.zeros(
            (self.n_fut_frames_int, 2), dtype=np.float32
        )
        self._future_root_ori_robot_frame_6d_buffer = np.zeros(
            (self.n_fut_frames_int, 6), dtype=np.float32
        )
        self._future_root_rel_quat_buffer = np.zeros(
            (self.n_fut_frames_int, 4), dtype=np.float32
        )
        self._motion_ref_yaw_alignment_quat_wxyz = np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self._motion_ref_yaw_alignment_ready = False
        self._keybody_rel_pos_fut_buffer = np.zeros(
            (self.n_fut_frames_int, 0, 3), dtype=np.float32
        )
        self._keybody_rel_pos_w_buffer = None
        max_t = max(1, self.n_fut_frames_int)
        self._q_conj_buffer = np.zeros((max_t + 1, 4), dtype=np.float32)
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

    def _init_keybody_indices_cache(self):
        if self.motion_config is None:
            raise ValueError(
                "motion_config is not loaded; cannot init keybody index cache"
            )

        atomic_list = self._get_policy_atomic_obs_list(self.motion_config)[
            "atomic_obs_list"
        ]
        body_names = [
            str(name) for name in self.motion_config.robot.body_names
        ]
        body_name_to_idx = {
            body_name: idx for idx, body_name in enumerate(body_names)
        }

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
            needs_keybody = ("keybody" in term_name) or (
                "keybody_names" in params
            )
            if not needs_keybody:
                continue

            keybody_names = params.get("keybody_names", None)
            if keybody_names is None:
                keybody_idxs = np.arange(len(body_names), dtype=np.int64)
            else:
                keybody_names = [str(name) for name in keybody_names]
                missing_names = [
                    name
                    for name in keybody_names
                    if name not in body_name_to_idx
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
                raise ValueError(
                    "Missing config.obs.obs_groups for policy obs"
                )

            if obs_groups.get("policy", None) is not None:
                entries = []
                atomic_obs_list = list(obs_groups.policy.atomic_obs_list)
                atomic_obs_list.extend(
                    getattr(
                        obs_groups.policy, "additional_atomic_obs_list", []
                    )
                )
                for term_dict in atomic_obs_list:
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
                atomic_obs_list = list(obs_groups.unified.atomic_obs_list)
                atomic_obs_list.extend(
                    getattr(
                        obs_groups.unified, "additional_atomic_obs_list", []
                    )
                )
                for term_dict in atomic_obs_list:
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
                obs_schema_plain = OmegaConf.to_container(
                    obs_schema, resolve=True
                )
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
                        ordered_terms.extend(
                            str(term) for term in node["terms"]
                        )
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
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.current_frame_idx(
                self.motion_frame_idx
            )
        return min(self.motion_frame_idx, self.n_motion_frames - 1)

    @property
    def ref_dof_pos_raw(self):
        if not self.reference_stream_active:
            if (
                self._offline_reference is not None
                and self._offline_reference.has_clip
            ):
                return self._offline_reference.ref_dof_pos_raw(
                    self.motion_frame_idx
                )
            return self.ref_dof_pos[self.ref_motion_frame_idx]
        if self._vr_reference is not None:
            fallback = (
                self.ref_dof_pos[self.ref_motion_frame_idx]
                if not self._vr_reference.has_reference
                else None
            )
            return self._vr_reference.current_dof_pos(fallback)
        return self.ref_dof_pos[self.ref_motion_frame_idx]

    @property
    def ref_dof_vel_raw(self):
        if not self.reference_stream_active:
            if (
                self._offline_reference is not None
                and self._offline_reference.has_clip
            ):
                return self._offline_reference.ref_dof_vel_raw(
                    self.motion_frame_idx
                )
            return self.ref_dof_vel[self.ref_motion_frame_idx]
        if self._vr_reference is not None:
            fallback = (
                self.ref_dof_vel[self.ref_motion_frame_idx]
                if not self._vr_reference.has_reference
                else None
            )
            return self._vr_reference.current_dof_vel(fallback)
        return self.ref_dof_vel[self.ref_motion_frame_idx]

    @property
    def ref_dof_pos_onnx_order(self):
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.ref_dof_pos_onnx_order(
                self.motion_frame_idx
            )
        return self.ref_dof_pos_raw[self.ref_to_onnx]

    @property
    def ref_dof_vel_onnx_order(self):
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.ref_dof_vel_onnx_order(
                self.motion_frame_idx
            )
        return self.ref_dof_vel_raw[self.ref_to_onnx]

    @property
    def ref_root_pos_raw(self):
        if not self.reference_stream_active:
            if (
                self._offline_reference is not None
                and self._offline_reference.has_clip
            ):
                return self._offline_reference.ref_root_pos_raw(
                    self.motion_frame_idx
                )
            return np.asarray(
                self.ref_raw_bodylink_pos[
                    self.ref_motion_frame_idx, self.root_body_idx
                ],
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
            not getattr(self, "reference_stream_active", False)
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
            raise ValueError(
                f"_xyzw_to_wxyz expects (...,4) but got shape {q_xyzw.shape}"
            )
        # q_xyzw[..., 0:3] -> xyz, q_xyzw[..., 3:4] -> w
        w = q_xyzw[..., 3:4]
        xyz = q_xyzw[..., 0:3]
        return np.concatenate([w, xyz], axis=-1)

    def _standardize_quaternion_wxyz(self, q_wxyz: np.ndarray) -> np.ndarray:
        """Standardize quaternion sign so that w >= 0."""
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        if q_wxyz.shape[-1] != 4:
            raise ValueError(
                f"_standardize_quaternion_wxyz expects (...,4) but got shape {q_wxyz.shape}"
            )
        mask = q_wxyz[..., 0:1] < 0.0
        q_wxyz = np.where(mask, -q_wxyz, q_wxyz)
        return q_wxyz

    @staticmethod
    def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
        half_yaw = 0.5 * np.float32(yaw)
        return np.asarray(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float32,
        )

    def clear_motion_yaw_alignment(self) -> None:
        if not hasattr(self, "_motion_ref_yaw_alignment_quat_wxyz"):
            return
        self._motion_ref_yaw_alignment_quat_wxyz[:] = np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self._motion_ref_yaw_alignment_ready = False

    def begin_motion_yaw_alignment(self) -> None:
        if not hasattr(self, "_motion_ref_yaw_alignment_quat_wxyz"):
            self._motion_ref_yaw_alignment_quat_wxyz = np.asarray(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            self._motion_ref_yaw_alignment_ready = False

        q_ref = self._standardize_quaternion_wxyz(
            self._get_ref_current_root_quat_wxyz()
        )
        q_robot = self._standardize_quaternion_wxyz(
            self.robot_root_rot_quat_wxyz
        )
        if np.linalg.norm(q_ref) < 1.0e-6 or np.linalg.norm(q_robot) < 1.0e-6:
            self.clear_motion_yaw_alignment()
            self.get_logger().warn(
                "Motion yaw alignment skipped because ref or robot root quaternion is invalid."
            )
            return

        ref_yaw = float(yaw_from_quat_wxyz(q_ref))
        robot_yaw = float(yaw_from_quat_wxyz(q_robot))
        yaw_offset = robot_yaw - ref_yaw
        self._motion_ref_yaw_alignment_quat_wxyz[:] = self._yaw_quat_wxyz(
            yaw_offset
        )
        self._motion_ref_yaw_alignment_ready = True
        self.get_logger().info(
            "Motion yaw alignment captured at motion entry: "
            f"ref_yaw={ref_yaw:.4f}, robot_yaw={robot_yaw:.4f}, "
            f"offset={yaw_offset:.4f} rad"
        )

    def _align_ref_quat_for_motion_entry(
        self,
        q_ref_wxyz: np.ndarray,
    ) -> np.ndarray:
        if not getattr(self, "_motion_ref_yaw_alignment_ready", False):
            return q_ref_wxyz
        q_aligned = self._quat_mul_wxyz(
            self._motion_ref_yaw_alignment_quat_wxyz,
            q_ref_wxyz,
        )
        return self._standardize_quaternion_wxyz(q_aligned)

    @staticmethod
    def _quat_inv_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        out = np.empty_like(q_wxyz)
        out[..., 0] = q_wxyz[..., 0]
        out[..., 1:4] = -q_wxyz[..., 1:4]
        return out

    @staticmethod
    def _quat_mul_wxyz(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        q0 = np.asarray(q0, dtype=np.float32)
        q1 = np.asarray(q1, dtype=np.float32)
        w0, x0, y0, z0 = np.moveaxis(q0, -1, 0)
        w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
        out = np.empty(
            np.broadcast_shapes(q0.shape, q1.shape), dtype=np.float32
        )
        out[..., 0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        out[..., 1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        out[..., 2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        out[..., 3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        return out

    @staticmethod
    def _rot6d_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        qw = q_wxyz[..., 0]
        qx = q_wxyz[..., 1]
        qy = q_wxyz[..., 2]
        qz = q_wxyz[..., 3]
        mat = np.empty(q_wxyz.shape[:-1] + (3, 3), dtype=np.float32)
        mat[..., 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
        mat[..., 0, 1] = 2.0 * (qx * qy - qz * qw)
        mat[..., 0, 2] = 2.0 * (qx * qz + qy * qw)
        mat[..., 1, 0] = 2.0 * (qx * qy + qz * qw)
        mat[..., 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
        mat[..., 1, 2] = 2.0 * (qy * qz - qx * qw)
        mat[..., 2, 0] = 2.0 * (qx * qz - qy * qw)
        mat[..., 2, 1] = 2.0 * (qy * qz + qx * qw)
        mat[..., 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
        return mat[..., :2].reshape(q_wxyz.shape[:-1] + (6,))

    def _quat_rotate_wxyz(
        self, q_wxyz: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        qvec = q_wxyz[..., 1:4]
        w = q_wxyz[..., 0:1]
        t = 2.0 * np.cross(qvec, v)
        return v + w * t + np.cross(qvec, t)

    def _quat_rotate_inv_wxyz(
        self, q_wxyz: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
        n = int(np.prod(q_wxyz.shape[:-1])) if q_wxyz.ndim > 1 else 1
        q_conj = self._q_conj_buffer[:n].reshape(q_wxyz.shape)
        q_conj[..., 0] = q_wxyz[..., 0]
        q_conj[..., 1:4] = -q_wxyz[..., 1:4]
        return self._quat_rotate_wxyz(q_conj, v)

    @staticmethod
    def _fill_gravity_wxyz(q_wxyz: np.ndarray, out: np.ndarray) -> None:
        qw = q_wxyz[0]
        qx = q_wxyz[1]
        qy = q_wxyz[2]
        qz = q_wxyz[3]
        out[0] = 2.0 * (-qz * qx + qw * qy)
        out[1] = -2.0 * (qz * qy + qw * qx)
        out[2] = 1.0 - 2.0 * (qw * qw + qz * qz)

    def _get_future_frame_indices(self) -> np.ndarray:
        frame_idx = self.ref_motion_frame_idx
        last_valid = self.last_valid_ref_motion_frame_idx
        np.minimum(
            frame_idx + self._future_frame_offsets,
            last_valid,
            out=self._future_frame_indices_buffer,
        )
        return self._future_frame_indices_buffer

    def _get_future_root_quat_wxyz(self) -> np.ndarray:
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
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

    def _identity_future_root_quat_wxyz(self) -> np.ndarray:
        q_root_wxyz = self._future_root_quat_wxyz_buffer
        q_root_wxyz[:, :] = 0.0
        q_root_wxyz[:, 0] = 1.0
        return q_root_wxyz

    def _get_ref_current_root_quat_wxyz(self) -> np.ndarray:
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.has_reference
        ):
            q_root_wxyz = self._vr_reference.current_root_rot()
            if q_root_wxyz is not None:
                return self._standardize_quaternion_wxyz(q_root_wxyz)
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.ref_root_quat_wxyz_cur(
                self.motion_frame_idx
            )
        if (
            hasattr(self, "ref_raw_bodylink_rot")
            and self.ref_raw_bodylink_rot is not None
        ):
            q_root_xyzw = self.ref_raw_bodylink_rot[
                self.ref_motion_frame_idx,
                self.root_body_idx,
            ]
            return self._standardize_quaternion_wxyz(
                self._xyzw_to_wxyz(q_root_xyzw)
            )
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _get_ref_future_root_quat_wxyz(self) -> np.ndarray:
        T = self.n_fut_frames_int
        if T <= 0:
            return self._future_root_quat_wxyz_buffer[:0]
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.has_reference
            and self._vr_reference.root_rot_queue is not None
            and self._vr_reference.root_rot_queue.shape[0] >= T
        ):
            return self._standardize_quaternion_wxyz(
                self._vr_reference.root_rot_queue[:T]
            )
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.ref_root_quat_wxyz_fut(
                self.motion_frame_idx
            )
        if (
            hasattr(self, "ref_raw_bodylink_rot")
            and self.ref_raw_bodylink_rot is not None
        ):
            return self._get_future_root_quat_wxyz()
        return self._identity_future_root_quat_wxyz()

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

    def _get_obs_actor_ref_future_yaw_delta_sin_cos(self):
        return self._get_obs_ref_future_yaw_delta_sin_cos()

    def _get_obs_actor_ref_robot_yaw_error_sin_cos(self):
        return self._get_obs_ref_robot_yaw_error_sin_cos()

    def _get_obs_actor_ref_future_root_ori_robot_frame_6d(self):
        return self._get_obs_ref_future_root_ori_robot_frame_6d()

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
        self._velocity_cmd_obs[0] = float(
            vx * vx + vy * vy + vyaw * vyaw > 0.01
        )
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
            np.take(
                self._real_dof_pos_buffer,
                self.motion_dof_real_indices_np,
                out=out,
            )
            np.subtract(out, def_angles, out=out)
            return out
        def_angles = self.velocity_default_angles_onnx
        n = len(self.velocity_dof_names_onnx)
        out = self._dof_pos_obs_buffer[:n]
        np.take(
            self._real_dof_pos_buffer,
            self.velocity_dof_real_indices_np,
            out=out,
        )
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
            np.take(
                self._real_dof_vel_buffer,
                self.motion_dof_real_indices_np,
                out=out,
            )
            return out
        n = len(self.velocity_dof_names_onnx)
        out = self._dof_vel_obs_buffer[:n]
        np.take(
            self._real_dof_vel_buffer,
            self.velocity_dof_real_indices_np,
            out=out,
        )
        return out

    def _get_obs_last_action(self):
        return self.actions_onnx

    def _get_obs_ref_motion_states(self):
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_motion_states(
                self.motion_frame_idx
            )
        n = self.num_actions
        self._ref_motion_states_buffer[:n] = self.ref_dof_pos_onnx_order
        self._ref_motion_states_buffer[n:] = self.ref_dof_vel_onnx_order
        return self._ref_motion_states_buffer

    def _get_obs_ref_dof_pos_fut(self):
        """Get future DOF position observation (reuses pre-allocated buffer)."""
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
        ):
            return self._vr_reference.obs_ref_dof_pos_fut(
                ref_to_onnx=self.ref_to_onnx,
                pos_fut_buffer=self._pos_fut_buffer,
                n_frames=T,
            )
        if getattr(self, "reference_stream_active", False):
            return np.zeros(self.num_actions * T, dtype=np.float32)
        if (
            self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_dof_pos_fut(
                self.motion_frame_idx
            )
        if not hasattr(self, "ref_dof_pos") or self.ref_dof_pos is None:
            self.get_logger().warn(
                "[VR] ref_dof_pos is unavailable and reference_qpos is not active; returning zeros for ref_dof_pos_fut."
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
        if self.reference_stream_active and self._vr_reference is not None:
            return self._vr_reference.obs_ref_root_height_fut(n_frames=T)
        if self.reference_stream_active:
            return np.zeros(T, dtype=np.float32)
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_root_height_fut(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_raw_bodylink_pos")
            or self.ref_raw_bodylink_pos is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and reference_qpos is not active; returning zeros for ref_root_height_fut."
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
        if self.reference_stream_active and self._vr_reference is not None:
            return self._vr_reference.obs_ref_root_pos_fut(n_frames=T)
        if self.reference_stream_active:
            return np.zeros(3 * T, dtype=np.float32)
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_root_pos_fut(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_raw_bodylink_pos")
            or self.ref_raw_bodylink_pos is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and reference_qpos is not active; returning zeros for ref_root_pos_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        pos_fut = self._root_pos_fut_buffer
        pos_fut[:, :] = self.ref_raw_bodylink_pos[
            fut_idx, self.root_body_idx, :
        ]
        return pos_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_dof_pos_cur(self):
        return self.ref_dof_pos_onnx_order

    def _get_obs_ref_dof_vel_cur(self):
        return self.ref_dof_vel_onnx_order

    def _get_obs_ref_root_height_cur(self):
        if not self.reference_stream_active:
            if (
                self._offline_reference is not None
                and self._offline_reference.has_clip
            ):
                return self._offline_reference.obs_ref_root_height_cur(
                    self.motion_frame_idx
                )
            return self.ref_raw_bodylink_pos[
                self.ref_motion_frame_idx, self.root_body_idx, 2
            ]
        return float(self.ref_root_pos_raw[2])

    def _get_obs_ref_root_pos_cur(self):
        return self.ref_root_pos_raw.astype(np.float32)

    def _get_obs_ref_gravity_projection_cur(self):
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.kinematics is not None
        ):
            return self._vr_reference.kinematics.projected_gravity[0]
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_gravity_projection_cur(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for gravity_projection_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[
            self.ref_motion_frame_idx, self.root_body_idx
        ]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        return get_gravity_orientation(q_root_wxyz)

    def _get_obs_ref_gravity_projection_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.kinematics is not None
        ):
            return self._vr_reference.kinematics.projected_gravity[1 : 1 + T].reshape(-1)
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_gravity_projection_fut(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for ref_gravity_projection_fut."
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
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.kinematics is not None
        ):
            return self._vr_reference.kinematics.root_linvel_local[0]
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_linvel_cur(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_global_velocity")
            or self.ref_global_velocity is None
        ):
            self.get_logger().warn(
                "[VR] ref_global_velocity is unavailable and reference_qpos is not active; returning zeros for ref_base_linvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for ref_base_linvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[
            self.ref_motion_frame_idx, self.root_body_idx
        ]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        v_root_w = np.asarray(
            self.ref_global_velocity[
                self.ref_motion_frame_idx, self.root_body_idx
            ],
            dtype=np.float32,
        )
        v_root = self._quat_rotate_inv_wxyz(q_root_wxyz, v_root_w)
        return np.asarray(v_root, dtype=np.float32).reshape(3)

    def _get_obs_ref_base_linvel_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.kinematics is not None
        ):
            return self._vr_reference.kinematics.root_linvel_local[1 : 1 + T].reshape(-1)

        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_linvel_fut(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_global_velocity")
            or self.ref_global_velocity is None
        ):
            self.get_logger().warn(
                "[VR] ref_global_velocity is unavailable and reference_qpos is not active; returning zeros for ref_base_linvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for ref_base_linvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        v_root_w = np.asarray(
            self.ref_global_velocity[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        base_linvel_fut = self._base_linvel_fut_buffer
        base_linvel_fut[:, :] = self._quat_rotate_inv_wxyz(
            q_root_wxyz, v_root_w
        )
        return base_linvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_base_angvel_cur(self):
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.kinematics is not None
        ):
            return self._vr_reference.kinematics.root_angvel_local[0]
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_angvel_cur(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_global_angular_velocity")
            or self.ref_global_angular_velocity is None
        ):
            self.get_logger().warn(
                "[VR] ref_global_angular_velocity is unavailable and reference_qpos is not active; returning zeros for ref_base_angvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for ref_base_angvel_cur."
            )
            return np.zeros(3, dtype=np.float32)
        q_root_xyzw = self.ref_raw_bodylink_rot[
            self.ref_motion_frame_idx, self.root_body_idx
        ]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)
        w_root_w = np.asarray(
            self.ref_global_angular_velocity[
                self.ref_motion_frame_idx, self.root_body_idx
            ],
            dtype=np.float32,
        )
        w_root = self._quat_rotate_inv_wxyz(q_root_wxyz, w_root_w)
        return np.asarray(w_root, dtype=np.float32).reshape(3)

    def _get_obs_ref_base_angvel_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.kinematics is not None
        ):
            return self._vr_reference.kinematics.root_angvel_local[1 : 1 + T].reshape(-1)

        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_base_angvel_fut(
                self.motion_frame_idx
            )
        if (
            not hasattr(self, "ref_global_angular_velocity")
            or self.ref_global_angular_velocity is None
        ):
            self.get_logger().warn(
                "[VR] ref_global_angular_velocity is unavailable and reference_qpos is not active; returning zeros for ref_base_angvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for ref_base_angvel_fut."
            )
            return np.zeros(3 * T, dtype=np.float32)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        w_root_w = np.asarray(
            self.ref_global_angular_velocity[fut_idx, self.root_body_idx],
            dtype=np.float32,
        )
        base_angvel_fut = self._base_angvel_fut_buffer
        base_angvel_fut[:, :] = self._quat_rotate_inv_wxyz(
            q_root_wxyz, w_root_w
        )
        return base_angvel_fut.reshape(-1).astype(np.float32)

    def _get_obs_ref_future_yaw_delta_sin_cos(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_future_yaw_delta_sin_cos(
                self.motion_frame_idx
            )
        q_cur = self._get_ref_current_root_quat_wxyz()
        q_fut = self._get_ref_future_root_quat_wxyz()
        yaw_delta = yaw_from_quat_wxyz(q_fut) - yaw_from_quat_wxyz(q_cur)
        yaw_delta_sin_cos = self._future_yaw_delta_sin_cos_buffer
        yaw_delta_sin_cos[:, 0] = np.sin(yaw_delta)
        yaw_delta_sin_cos[:, 1] = np.cos(yaw_delta)
        return yaw_delta_sin_cos.reshape(-1).astype(np.float32)

    def _get_obs_ref_robot_yaw_error_sin_cos(self):
        q_ref = self._align_ref_quat_for_motion_entry(
            self._get_ref_current_root_quat_wxyz()
        )
        q_robot = self._standardize_quaternion_wxyz(
            self.robot_root_rot_quat_wxyz
        )
        yaw_error = yaw_from_quat_wxyz(q_ref) - yaw_from_quat_wxyz(q_robot)
        return np.asarray(
            [np.sin(yaw_error), np.cos(yaw_error)],
            dtype=np.float32,
        ).reshape(2)

    def _get_obs_ref_future_root_ori_robot_frame_6d(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        q_ref_fut = self._align_ref_quat_for_motion_entry(
            self._get_ref_future_root_quat_wxyz()
        )
        q_robot_inv = self._quat_inv_wxyz(
            self._standardize_quaternion_wxyz(self.robot_root_rot_quat_wxyz)
        )
        q_rel = self._future_root_rel_quat_buffer
        q_rel[:, :] = self._quat_mul_wxyz(q_robot_inv[None, :], q_ref_fut)
        q_rel[:, :] = self._standardize_quaternion_wxyz(q_rel)
        self._future_root_ori_robot_frame_6d_buffer[:, :] = (
            self._rot6d_from_quat_wxyz(q_rel)
        )
        return self._future_root_ori_robot_frame_6d_buffer.reshape(-1).astype(
            np.float32
        )

    def _get_obs_ref_keybody_rel_pos_cur(self):
        if (
            getattr(self, "reference_stream_active", False)
            and self._vr_reference is not None
            and self._vr_reference.has_reference
        ):
            keybody_idxs = self._get_ref_keybody_indices(
                "actor_ref_keybody_rel_pos_cur"
            )
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)

        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            keybody_idxs = self._get_ref_keybody_indices(
                "actor_ref_keybody_rel_pos_cur"
            )
            return self._offline_reference.obs_ref_keybody_rel_pos_cur(
                self.motion_frame_idx,
                keybody_idxs,
            )
        if (
            not hasattr(self, "ref_raw_bodylink_pos")
            or self.ref_raw_bodylink_pos is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and reference_qpos is not active; returning zeros for ref_keybody_rel_pos_cur."
            )
            keybody_idxs = self._get_ref_keybody_indices(
                "actor_ref_keybody_rel_pos_cur"
            )
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for ref_keybody_rel_pos_cur."
            )
            keybody_idxs = self._get_ref_keybody_indices(
                "actor_ref_keybody_rel_pos_cur"
            )
            n_keybodies = int(keybody_idxs.shape[0])
            if n_keybodies == 0:
                return np.zeros(0, dtype=np.float32)
            return np.zeros(3 * n_keybodies, dtype=np.float32)

        keybody_idxs = self._get_ref_keybody_indices(
            "actor_ref_keybody_rel_pos_cur"
        )
        n_keybodies = int(keybody_idxs.shape[0])
        if n_keybodies == 0:
            return np.zeros(0, dtype=np.float32)

        frame_idx = self.ref_motion_frame_idx
        ref_body_global_pos = np.asarray(
            self.ref_raw_bodylink_pos[frame_idx], dtype=np.float32
        )
        ref_root_global_pos = ref_body_global_pos[self.root_body_idx]
        q_root_xyzw = self.ref_raw_bodylink_rot[frame_idx, self.root_body_idx]
        q_root_wxyz = self._xyzw_to_wxyz(q_root_xyzw)
        q_root_wxyz = self._standardize_quaternion_wxyz(q_root_wxyz)

        rel_pos_w = (
            ref_body_global_pos[keybody_idxs] - ref_root_global_pos[None, :]
        )
        rel_pos_root = self._quat_rotate_inv_wxyz(q_root_wxyz, rel_pos_w)
        return np.asarray(rel_pos_root, dtype=np.float32).reshape(-1)

    def _get_obs_ref_keybody_rel_pos_fut(self):
        T = self.n_fut_frames_int
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        keybody_idxs = self._get_ref_keybody_indices(
            "actor_ref_keybody_rel_pos_fut"
        )
        n_keybodies = int(keybody_idxs.shape[0])
        if (
            not getattr(self, "reference_stream_active", False)
            and self._offline_reference is not None
            and self._offline_reference.has_clip
        ):
            return self._offline_reference.obs_ref_keybody_rel_pos_fut(
                self.motion_frame_idx,
                keybody_idxs,
            )
        if (
            not hasattr(self, "ref_raw_bodylink_pos")
            or self.ref_raw_bodylink_pos is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_pos is unavailable and reference_qpos is not active; returning zeros for ref_keybody_rel_pos_fut."
            )
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)
        if (
            not hasattr(self, "ref_raw_bodylink_rot")
            or self.ref_raw_bodylink_rot is None
        ):
            self.get_logger().warn(
                "[VR] ref_raw_bodylink_rot is unavailable and reference_qpos is not active; returning zeros for ref_keybody_rel_pos_fut."
            )
            if n_keybodies == 0:
                return np.zeros((T, 0), dtype=np.float32).reshape(-1)
            return np.zeros((T, n_keybodies, 3), dtype=np.float32).reshape(-1)

        if n_keybodies == 0:
            return np.zeros((T, 0), dtype=np.float32).reshape(-1)
        fut_idx = self._get_future_frame_indices()
        q_root_wxyz = self._get_future_root_quat_wxyz()
        ref_body_global_pos = np.asarray(
            self.ref_raw_bodylink_pos[fut_idx], dtype=np.float32
        )
        ref_root_global_pos = ref_body_global_pos[:, self.root_body_idx, :]
        rel_pos_w = (
            ref_body_global_pos[:, keybody_idxs, :]
            - ref_root_global_pos[:, None, :]
        )
        if self._keybody_rel_pos_fut_buffer.shape[1] != n_keybodies:
            self._keybody_rel_pos_fut_buffer = np.zeros(
                (T, n_keybodies, 3), dtype=np.float32
            )
        rel_pos_fut = self._keybody_rel_pos_fut_buffer
        rel_pos_fut[:, :, :] = self._quat_rotate_inv_wxyz(
            q_root_wxyz[:, None, :], rel_pos_w
        )
        return rel_pos_fut.reshape(-1).astype(np.float32)

    def _get_obs_place_holder(self):
        expected_dim = int(getattr(self, "actor_place_holder_ndim", 0) or 0)
        if (
            not hasattr(self, "_place_holder_buffer")
            or self._place_holder_buffer.shape[0] != expected_dim
        ):
            self._place_holder_buffer = np.zeros(
                expected_dim, dtype=np.float32
            )
        return self._place_holder_buffer
