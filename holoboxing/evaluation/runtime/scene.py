"""The shared two-fighter arena; every physical step executes on the GPU."""

from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .onnx_device import require_gpu


@dataclass(frozen=True)
class SceneKinematics:
    root_position: np.ndarray
    root_quaternion_wxyz: np.ndarray
    root_linear_velocity: np.ndarray
    root_angular_velocity_local: np.ndarray
    root_angular_velocity_world: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position: np.ndarray
    body_linear_velocity: np.ndarray


def _quat_to_matrix_wxyz(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()


def load_scene_model(model_path):
    """Load the replay XMLs through MJCF attachments and add 200 Hz control."""
    path = Path(model_path).resolve(strict=True)
    spec = mujoco.MjSpec.from_file(str(path))
    spec.option.timestep = 0.005
    # Replay models already supply geometry, inertia and joint force limits.
    # Add only evaluator dynamics and torque controls; never copy robot trees.
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint.damping = 0.05
        joint.armature = 0.01
        wrist = "_wrist_pitch_" in joint.name or "_wrist_yaw_" in joint.name
        joint.frictionloss = 0.1 if wrist else 0.2
        actuator = spec.add_actuator(name=joint.name.removesuffix("_joint"))
        actuator.set_to_motor()
        actuator.target = joint.name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.ctrllimited = True
        actuator.ctrlrange = joint.actfrcrange
    return spec.compile()


class BoxingMatchScene:
    def __init__(
        self,
        *,
        model_path,
        mesh_root=None,
        simulation_frequency=200,
        control_decimation=4,
        contact_force_threshold=1.0,
        show_viewer=False,
        realtime=False,
        physics_device="cuda:0",
        inference_benchmark=False,
    ):
        require_gpu(physics_device)
        if simulation_frequency != 200 or control_decimation != 4:
            raise ValueError(
                "Official boxing uses 200 Hz physics and 50 Hz control"
            )
        if show_viewer:
            raise ValueError(
                "Use the independent saved-trajectory player to view the match"
            )
        self.model = load_scene_model(model_path)
        if (self.model.nq, self.model.nv, self.model.nu) != (72, 70, 58):
            raise ValueError(
                "The shared scene requires nq=72, nv=70 and nu=58"
            )
        if not np.isclose(
            self.model.opt.timestep,
            1 / simulation_frequency,
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("The supplied scene timestep differs from 200 Hz")
        self.data = mujoco.MjData(self.model)
        self.physics_device = str(require_gpu(physics_device))
        self.physics_backend = "mujoco-warp"
        self.control_decimation = control_decimation
        self.minimum_contact_force = float(contact_force_threshold)
        self._viewer, self.realtime = None, realtime
        self._pending_contacts = None
        self._stream_frame_index = 0
        self._gpu_referee = None
        self._live_policies = []
        self._upper_pair = None
        self._targets_on_gpu = False
        self._strike_geom = {}
        self._root_qpos, self._root_dof = [], []
        self._joint_qpos, self._joint_dof, self._actuator = [], [], []
        self.body_names = []
        joint_names = None
        for fighter, prefix in enumerate(("p1_", "p2_")):
            root_body = self.model.body(prefix + "pelvis")
            root_joint = int(root_body.jntadr[0])
            self._root_qpos.append(int(self.model.jnt_qposadr[root_joint]))
            self._root_dof.append(int(self.model.jnt_dofadr[root_joint]))
            joints = [
                index
                for index in range(self.model.njnt)
                if self.model.joint(index).name.startswith(prefix)
                and self.model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
            ]
            names = tuple(
                self.model.joint(index).name[len(prefix) :] for index in joints
            )
            if len(names) != 29 or (
                joint_names is not None and names != joint_names
            ):
                raise ValueError(
                    "The scene must contain two matching 29-joint fighters"
                )
            joint_names = names
            self._joint_qpos.append(
                np.asarray(
                    [self.model.jnt_qposadr[index] for index in joints],
                    dtype=np.int64,
                )
            )
            self._joint_dof.append(
                np.asarray(
                    [self.model.jnt_dofadr[index] for index in joints],
                    dtype=np.int64,
                )
            )
            actuator_by_joint = {
                int(self.model.actuator_trnid[index, 0]): index
                for index in range(self.model.nu)
            }
            try:
                self._actuator.append(
                    np.asarray(
                        [actuator_by_joint[index] for index in joints],
                        dtype=np.int64,
                    )
                )
            except KeyError as error:
                raise ValueError(
                    "Every fighter joint requires a torque actuator"
                ) from error
            if fighter == 0:
                self.body_names = [
                    self.model.body(index).name[len(prefix) :]
                    for index in range(self.model.nbody)
                    if self.model.body(index).name
                    and self.model.body(index).name.startswith(prefix)
                ]
        self.articulation_joint_names = joint_names
        # Separate the unloaded fighters before the first GPU forward pass.
        # The versioned reset resource replaces this state before any step.
        for fighter, qi in enumerate(self._root_qpos):
            self.data.qpos[qi] = -1.0 if fighter == 0 else 1.0
        self._kp, self._kd = np.ones(29), np.ones(29)
        limits = []
        for fighter in range(2):
            selected = self._actuator[fighter]
            if not self.model.actuator_ctrllimited[selected].all():
                raise ValueError(
                    "Scene torque actuators must have explicit limits"
                )
            lower, upper = self.model.actuator_ctrlrange[selected].T
            if not np.allclose(-lower, upper):
                raise ValueError("Scene torque limits must be symmetric")
            limits.append(upper)
        if not np.allclose(limits[0], limits[1]):
            raise ValueError("Both fighters must use the same torque limits")
        self._torque_limit = limits[0]
        self._targets = np.zeros((2, 29), dtype=np.float64)
        aliases = {
            "left_wrist_yaw_link": "left_hand",
            "right_wrist_yaw_link": "right_hand",
            "left_rubber_hand": "left_hand",
            "right_rubber_hand": "right_hand",
            "left_ankle_roll_link": "left_foot",
            "right_ankle_roll_link": "right_foot",
        }
        for geom in range(self.model.ngeom):
            body = (
                self.model.body(int(self.model.geom_bodyid[geom])).name or ""
            )
            for fighter, prefix in enumerate(("p1_", "p2_")):
                if body.startswith(prefix) and body[len(prefix) :] in aliases:
                    self._strike_geom[geom] = (
                        fighter,
                        aliases[body[len(prefix) :]],
                    )
        if inference_benchmark:
            from .benchmark_state import BenchmarkState

            self._gpu_physics = BenchmarkState(self, physics_device)
        else:
            from .gpu_physics import ResidentPhysics

            self._gpu_physics = ResidentPhysics(self, physics_device)
        self._started = time.monotonic()

    @property
    def viewer_running(self):
        return True

    def _opponent_target_semantic(self, attacker, geom):
        body_id = int(self.model.geom_bodyid[geom])
        body = self.model.body(body_id).name or ""
        prefix = ("p1_", "p2_")[1 - attacker]
        if not body.startswith(prefix):
            return None
        link = body[len(prefix) :]
        semantic = {
            "torso_link": "torso",
            "pelvis": "pelvis",
            "head_link": "head",
            "logo_link": "chest",
        }.get(link, "unscored")
        mesh_id = (
            int(self.model.geom_dataid[geom])
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            else -1
        )
        mesh_name = self.model.mesh(mesh_id).name if mesh_id >= 0 else ""
        # Attached models prefix mesh names as well as body names.
        if mesh_name.startswith(prefix):
            mesh_name = mesh_name[len(prefix) :]
        semantic = {
            "head_link": "head",
            "logo_link": "chest",
            "pelvis_contour_link": "pelvis",
            "pelvis_contour_collision": "pelvis",
        }.get(mesh_name, semantic)
        return body_id, link, semantic

    def write_pair_state(self, qpos, qvel=None):
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape == (1, 2, 36):
            values = values[0]
        if values.shape != (2, 36) or not np.isfinite(values).all():
            raise ValueError("Reset pose must contain finite [2,36] states")
        velocities = (
            np.zeros((2, 35))
            if qvel is None
            else np.asarray(qvel, dtype=np.float64).reshape(2, 35)
        )
        if not np.isfinite(velocities).all() or not np.allclose(
            np.linalg.norm(values[:, 3:7], axis=-1), 1, atol=1e-5
        ):
            raise ValueError("Reset pose has invalid quaternion or velocity")
        self.data.time = 0.0
        for fighter in range(2):
            qi, vi = self._root_qpos[fighter], self._root_dof[fighter]
            self.data.qpos[qi : qi + 7] = values[fighter, :7]
            self.data.qpos[self._joint_qpos[fighter]] = values[fighter, 7:]
            self.data.qvel[vi : vi + 6] = velocities[fighter, :6]
            self.data.qvel[self._joint_dof[fighter]] = velocities[fighter, 6:]
        self._pending_contacts = None
        self._stream_frame_index = 0
        self._targets_on_gpu = False
        self._gpu_physics.forward_from_host()

    def stage_pair_joint_targets(self, targets):
        values = np.asarray(targets, dtype=np.float64).reshape(2, 29)
        if not np.isfinite(values).all():
            raise ValueError(
                "Joint targets must be finite; lower output is not sanitized"
            )
        self._targets[:] = values

    def drain_reference_frame_qpos(self):
        return None

    def step_reference_frame(self):
        if self._pending_contacts is not None:
            raise RuntimeError(
                "Consume contact_samples before the next control frame"
            )
        if self.realtime:
            remaining = (
                self._started
                + (self._stream_frame_index + 1) / 50
                - time.monotonic()
            )
            if remaining > 0:
                time.sleep(remaining)
        self._pending_contacts = self._gpu_physics.frame()
        self._stream_frame_index += 1

    def contact_samples(self, *, sim_time_s):
        if self._pending_contacts is None:
            raise RuntimeError("No completed physics frame is available")
        pending, self._pending_contacts = self._pending_contacts, None
        contacts = []
        for (attacker, attack, _), values in pending.items():
            contacts.append(
                dict(
                    attacker=attacker,
                    defender=1 - attacker,
                    attacking_body=attack,
                    **values,
                )
            )
        return [contacts]

    def read_kinematics(self):
        data = self.data
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isfinite(data.xpos).all()
        ):
            raise ValueError(
                "Nonfinite measured physics state; no sanitation is allowed"
            )
        root = np.stack(
            [data.qpos[index : index + 3] for index in self._root_qpos]
        ).astype(np.float32)[None]
        quat = np.stack(
            [data.qpos[index + 3 : index + 7] for index in self._root_qpos]
        ).astype(np.float32)[None]
        linear = np.stack(
            [data.qvel[index : index + 3] for index in self._root_dof]
        ).astype(np.float32)[None]
        angular = np.stack(
            [data.qvel[index + 3 : index + 6] for index in self._root_dof]
        ).astype(np.float32)[None]
        world = np.stack(
            [
                _quat_to_matrix_wxyz(q) @ value
                for q, value in zip(quat[0], angular[0])
            ]
        ).astype(np.float32)[None]
        joint = data.qpos[np.asarray(self._joint_qpos)].astype(np.float32)[
            None
        ]
        velocity = data.qvel[np.asarray(self._joint_dof)].astype(np.float32)[
            None
        ]
        ids = [
            [self.model.body(prefix + name).id for name in self.body_names]
            for prefix in ("p1_", "p2_")
        ]
        body = data.xpos[np.asarray(ids)].astype(np.float32)[None]
        return SceneKinematics(
            root,
            quat,
            linear,
            angular,
            world,
            joint,
            velocity,
            body,
            np.zeros_like(body),
        )

    def close(self):
        self._gpu_physics.close()
