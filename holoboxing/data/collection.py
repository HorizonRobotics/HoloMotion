"""Measured paired supervision using the frozen lower controller, without an upper model."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .contract import CompetitionHistory, target_from_world
from .common import sha256

MEASURED_SOURCE = "measured_frozen_lower_rollout_current_state"
RECIPE = {
    "version": "measured_pair_short_window_v1",
    "observation_source": MEASURED_SOURCE,
    "core_frames": 100,
    "warmup_frames": 10,
    "segment_frames": 3000,
    "future_offsets": [1, 2, 3, 4, 5],
    "initial_velocity": "reference_world_linear_local_angular_named_joints",
    "target_translation": "pair_mean_displacement_error_xy_from_reset",
    "velocity": "same_instant_link_origin",
    "target_frame": "actual_current_full_pelvis",
    "cutoff": "first_role_pelvis_or_torso_below_floor_plus_0.3m",
    "early_terminal_tail": "exclude_labels_reaching_failure",
    "remaining_time": "source_global_segment",
}


def windows(frames):
    for segment in range(0, frames, 3000):
        end = min(segment + 3000, frames) - 5
        for core in range(segment, end, 100):
            yield max(segment, core - 10), core, min(core + 100, end)


def sample_indices(frames, roles=2):
    current = np.arange(max(0, frames - 5), dtype=np.int64)
    current = current[current // 3000 == (current + 5) // 3000]
    return np.concatenate(
        [
            np.column_stack((current, np.full_like(current, role)))
            for role in range(roles)
        ]
    )


def reference_velocity(trajectory, frame):
    q = trajectory["root_quaternion"][frame]
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    angular = np.einsum(
        "rji,rj->ri",
        rotation,
        trajectory["root_angular_velocity_world"][frame],
    )
    return np.concatenate(
        (
            trajectory["root_origin_velocity"][frame],
            angular,
            trajectory["joint_velocity"][frame],
        ),
        axis=-1,
    )


def shared_translation(actual, reference, initial_actual, initial_reference):
    delta = ((actual - initial_actual) - (reference - initial_reference)).mean(
        0
    )
    delta[2] = 0
    return delta


def retained_frames(seen, core, stop):
    if not seen:
        return []
    terminal = seen[-1] + 1
    return [
        f for f in seen if f >= core and (terminal == stop or f + 5 < terminal)
    ]


def expected_retained_indices(frames, failures, roles=2):
    plan = {core: (start, stop) for start, core, stop in windows(frames)}
    cutoffs = {}
    for event in failures:
        core, frame = event["core"], event["frame"]
        if core not in plan or core in cutoffs:
            raise ValueError("Unknown or duplicate failed window")
        start, stop = plan[core]
        if (
            type(frame) is not int
            or not start <= frame < stop
            or not event["roles"]
            or any(r not in (0, 1) for r in event["roles"])
        ):
            raise ValueError("Invalid early-failure record")
        cutoffs[core] = frame
    accepted = []
    for core, (_, stop) in plan.items():
        accepted.extend(
            range(
                core, min(stop, cutoffs[core] - 5) if core in cutoffs else stop
            )
        )
    indices = sample_indices(frames, roles)
    return indices[np.isin(indices[:, 0], accepted)]


def measured_state(model, data, point_ids):
    position = data.xpos[point_ids].copy()
    angular = data.cvel[point_ids, :3]
    center = data.subtree_com[model.body_rootid[point_ids]]
    velocity = data.cvel[point_ids, 3:] + np.cross(angular, position - center)
    omega = np.stack(
        [
            data.xmat[point_ids[r, 0]].reshape(3, 3).T @ angular[r, 0]
            for r in range(2)
        ]
    )
    return position, data.xquat[point_ids].copy(), velocity, omega


def pack_reference(schema, target, height, *, out=None):
    """Fill the existing lower schema, retaining masks and validity bits."""
    points = target.reshape(5, 5, 9)
    template = schema.template.to(target.device)
    value = template.clone() if out is None else out.copy_(template)
    task = value[: schema.task_dim].reshape(5, schema.width)
    task[:, : schema.num_links * 3].reshape(5, schema.num_links, 3)[
        :, schema.active
    ] = points[..., :3]
    task[:, schema.num_links * 3 : -1].reshape(5, schema.num_links, 6)[
        :, schema.active
    ] = points[..., 3:]
    value[schema.task_dim] = height
    return value


def lower_identity(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    return {
        name: sha256(root / name)
        for name in ("exported/model_10500.onnx", "config.yaml")
    }


class MeasuredCollector:
    """One GPU scene per worker; reuse it only through the official reset."""

    def __init__(self, assets, lower, device="cuda:0"):
        from holoboxing.evaluation.runtime.holomotion_v2_runtime import (
            install_runtime,
        )
        from holoboxing.evaluation.runtime.compiled_artifacts import (
            reference_schema,
            available,
        )
        from holoboxing.evaluation.runtime.lower import LinkPolicy
        from .records import scene_layout

        if not str(device).startswith("cuda"):
            raise ValueError("Measured collection requires a CUDA device")
        torch.cuda.set_device(device)
        if available():
            raise ValueError(
                "Measured preparation requires the source runtime, not compiled evaluation-only artifacts"
            )
        self.runtime = install_runtime(lower)
        self.device = device
        self.scene = self.runtime.BoxingMatchScene(
            model_path=assets.resources.scene,
            mesh_root=assets.resources.root / "meshes",
            physics_device=device,
        )
        self.names, self.roots, self.joints, self.point_ids = scene_layout(
            self.scene.model
        )
        self.policies = tuple(
            LinkPolicy(
                Path(lower) / "exported/model_10500.onnx",
                device=device,
                articulation_joint_names=self.names,
                retarget_dof_names=self.names,
            )
            for _ in range(2)
        )
        self.scene._kp = (
            self.policies[0]
            .kp[self.policies[0].model_to_articulation]
            .astype(np.float64)
        )
        self.scene._kd = (
            self.policies[0]
            .kd[self.policies[0].model_to_articulation]
            .astype(np.float64)
        )
        for policy in self.policies:
            policy.attach_scene(self.scene)
            policy.retarget_to_model = np.arange(29, dtype=np.int64)
        self.schema = reference_schema()
        # The lower CUDA graph captures these addresses, not Python arguments.
        self.references = torch.stack(
            [self.schema.template.to(device) for _ in range(2)]
        )
        body_ids = torch.as_tensor(self.point_ids, device=device)
        for policy in self.policies:
            policy.schema = self.schema
            policy.upper_body = body_ids
            policy.upper_body_host = self.point_ids
        self.cpu = mujoco.MjData(self.scene.model)
        self.torso = [
            self.scene.model.body(p + "torso_link").id for p in ("p1_", "p2_")
        ]
        geom = self.scene.model.geom("smpl_ring_baseLink")
        self.floor = geom.pos[2] + geom.size[2]
        self.recipe = {
            **RECIPE,
            "lower_identity": lower_identity(lower),
        }

    def close(self):
        self.scene.close()

    def reset(self, trajectory, frame):
        qpos = np.concatenate(
            [
                trajectory[key][frame]
                for key in (
                    "root_position",
                    "root_quaternion",
                    "joint_position",
                )
            ],
            -1,
        )
        qvel = reference_velocity(trajectory, frame)
        if trajectory["roles"] == 1:
            qpos = np.concatenate((qpos, trajectory["standing_qpos"][None]))
            qvel = np.concatenate((qvel, np.zeros_like(qvel)))
        writer = self.scene.write_pair_state
        # Preserve the runtime's pointer/graph rebinding; inject only velocity.
        self.scene.write_pair_state = lambda values: writer(values, qvel=qvel)
        try:
            self.runtime._reset_scene(
                self.scene, SimpleNamespace(qpos=qpos[None]), np.arange(29)
            )
        finally:
            self.scene.write_pair_state = writer
        for role, ((_, vi), (_, joint_vi)) in enumerate(
            zip(self.roots, self.joints)
        ):
            actual = np.concatenate(
                (
                    self.scene.data.qvel[vi : vi + 6],
                    self.scene.data.qvel[joint_vi],
                )
            )
            np.testing.assert_allclose(
                actual, qvel[role], atol=2e-6, rtol=2e-6
            )
        for policy in self.policies:
            policy.reset()

    def collect(self, arrays, trajectory):
        if tuple(trajectory["joint_names"]) != tuple(self.names):
            raise ValueError("Reference and collector joint order differ")
        count = len(arrays["position"])
        obs, targets, row_window, accepted, events = {}, {}, {}, [], []
        max_velocity_error = max_target_error = 0.0
        window_count = 0
        for start, core, stop in windows(count):
            self.reset(trajectory, start)
            history = CompetitionHistory(1)
            seen = []
            initial_actual = initial_reference = None
            for frame in range(start, stop):
                self.cpu.qpos[:] = self.scene.data.qpos
                self.cpu.qvel[:] = self.scene.data.qvel
                if not (
                    np.isfinite(self.cpu.qpos).all()
                    and np.isfinite(self.cpu.qvel).all()
                ):
                    raise ValueError(
                        f"Nonfinite measured state at source frame {frame}"
                    )
                mujoco.mj_forward(self.scene.model, self.cpu)
                state = measured_state(
                    self.scene.model, self.cpu, self.point_ids
                )
                position, quaternion, velocity, _ = state
                low = (
                    np.minimum(position[:, 0, 2], self.cpu.xpos[self.torso, 2])
                    < self.floor + 0.3
                )
                if low.any():
                    events.append(
                        {
                            "frame": frame,
                            "core": core,
                            "roles": np.flatnonzero(low).tolist(),
                        }
                    )
                    break
                if frame - start < 2:
                    for role, slot in np.ndindex(2, 5):
                        jac = np.empty((3, self.scene.model.nv))
                        mujoco.mj_jacBody(
                            self.scene.model,
                            self.cpu,
                            jac,
                            None,
                            int(self.point_ids[role, slot]),
                        )
                        max_velocity_error = max(
                            max_velocity_error,
                            float(
                                np.abs(
                                    jac @ self.cpu.qvel - velocity[role, slot]
                                ).max()
                            ),
                        )
                tensors = [
                    torch.tensor(v[None], dtype=torch.float32) for v in state
                ]
                observation = history.observe(
                    *tensors,
                    torch.tensor([60.0 - (frame % 3000) / 50]),
                    torch.zeros(1, 2),
                    reset=torch.tensor([frame == start]),
                    step=frame - start,
                )
                reference_pelvis = arrays["position"][frame, :, 0].numpy()
                if initial_actual is None:
                    initial_actual = position[:, 0].copy()
                    initial_reference = reference_pelvis.copy()
                shift = shared_translation(
                    position[:, 0],
                    reference_pelvis,
                    initial_actual,
                    initial_reference,
                )
                future = torch.arange(frame + 1, frame + 6)
                world = arrays["position"][future] + torch.tensor(
                    shift, dtype=torch.float32
                )
                target = target_from_world(
                    tensors[0][0],
                    tensors[1][0],
                    world.transpose(0, 1),
                    arrays["quaternion"][future].transpose(0, 1),
                )
                if frame >= core:
                    obs[frame], targets[frame] = (
                        observation[0].numpy(),
                        target.numpy(),
                    )
                    row_window[frame] = (start, core)
                seen.append(frame)
                self.scene.prepare_actor_observation()
                joint_targets = []
                for role, policy in enumerate(self.policies):
                    reference = pack_reference(
                        self.schema,
                        target[role].to(self.device),
                        float(position[role, 0, 2]),
                        out=self.references[role],
                    )
                    if frame - start < 2:
                        restored = self.schema.to_targets(
                            reference.cpu().numpy(),
                            position[role, 0],
                            quaternion[role, 0],
                        )
                        max_target_error = max(
                            max_target_error,
                            float(
                                np.abs(
                                    restored.future_positions
                                    - world[:, role].numpy()
                                ).max()
                            ),
                        )
                    _, jt, _, _ = policy.act(
                        reference_qpos=reference, kinematics=None, fighter=role
                    )
                    joint_targets.append(jt)
                self.scene.stage_pair_joint_targets(
                    np.asarray(joint_targets)[None]
                )
                self.scene.step_reference_frame()
                self.scene.drain_reference_frame_qpos()
                self.scene.contact_samples(
                    sim_time_s=np.asarray([(frame - start + 1) / 50])
                )
            retained = retained_frames(seen, core, stop)
            accepted.extend(retained)
            window_count += 1
        if max_velocity_error >= 1e-7 or max_target_error >= 3e-5:
            raise ValueError(
                "Measured velocity/target geometry verification failed"
            )
        indices = sample_indices(count, trajectory["roles"])
        planned = len(indices)
        indices = indices[np.isin(indices[:, 0], accepted)]
        if not len(indices):
            raise ValueError(
                "Source produced zero measured SL rows; refusing to drop its AMP motion"
            )
        packet = {
            "obs": np.stack([obs[int(f)][int(r)] for f, r in indices]).astype(
                np.float32
            ),
            "target_pose": np.stack(
                [targets[int(f)][int(r)] for f, r in indices]
            ).astype(np.float32),
            "sample_index": indices,
            "window_index": np.asarray(
                [row_window[int(f)] for f, _ in indices], dtype=np.int64
            ),
        }
        return packet, {
            "observation_source": MEASURED_SOURCE,
            "samples": len(indices),
            "planned_samples": planned,
            "rejected_samples": planned - len(indices),
            "windows": window_count,
            "early_failures": events,
            "max_velocity_jacobian_error": max_velocity_error,
            "max_target_roundtrip_error": max_target_error,
            "collection_recipe": self.recipe,
        }
