"""Resident CUDA policies and observations; preserve physics and referee rules."""

import os
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import warp as wp
from holomotion.src.motion_tracking.link_observation import (
    quat_apply_wxyz as rotate,
    quat_conjugate_wxyz as conjugate,
    quat_multiply_wxyz as multiply,
    LinkObservationInput,
    build_link_observation,
)
from holomotion.src.motion_tracking.link_actor_packing import (
    pack_link_actor_observation,
)
from .observation_fk import CurrentObservationFK

SESSIONS = OrderedDict()
SCENES = {}
METRICS = {}


def world_angular_velocity(quaternions, velocities):
    from .scene import _quat_to_matrix_wxyz

    return np.stack(
        [_quat_to_matrix_wxyz(q) @ v for q, v in zip(quaternions, velocities)]
    ).astype(np.float32)


def matrix_quaternion(m):
    # Largest-component conversion, valid also at rotations near pi.
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    qabs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c),
                -1,
            ),
            min=0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack(
                (
                    qabs[..., 0] ** 2,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    qabs[..., 1] ** 2,
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    qabs[..., 2] ** 2,
                    m[..., 2, 1] + m[..., 1, 2],
                ),
                -1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 2, 0] + m[..., 0, 2],
                    m[..., 2, 1] + m[..., 1, 2],
                    qabs[..., 3] ** 2,
                ),
                -1,
            ),
        ),
        -2,
    )
    candidates = candidates / (2 * qabs.clamp_min(0.1)[..., None])
    q = torch.gather(
        candidates,
        -2,
        qabs.argmax(-1)[..., None, None].expand(*m.shape[:-2], 1, 4),
    ).squeeze(-2)
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def xz_quaternion(xz):
    x = xz[..., :3]
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    fallback = torch.zeros_like(x)
    fallback[..., 0] = 1
    x = torch.where(norm > 1e-8, x / norm.clamp_min(1e-8), fallback)
    z = xz[..., 3:] - (xz[..., 3:] * x).sum(-1, keepdim=True) * x
    fallback = torch.nn.functional.one_hot(x.abs().argmin(-1), 3).to(x.dtype)
    fallback = fallback - (fallback * x).sum(-1, keepdim=True) * x
    z = torch.where(
        torch.linalg.vector_norm(z, dim=-1, keepdim=True) > 1e-8, z, fallback
    )
    z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
    return matrix_quaternion(torch.stack((x, torch.linalg.cross(z, x), z), -1))


class BoundSession:
    """Fixed-address I/O binding. Recurrent outputs use separate ping-pong buffers."""

    def __init__(self, session, device):
        from .onnx_device import require_gpu, require_gpu_session

        device = require_gpu(device)
        require_gpu_session(session)
        self.session = session
        self.device = device
        self.inputs = {}
        self.outputs = {}
        self.binding = session.io_binding()
        for node in session.get_inputs():
            dtype = torch.int64 if "int64" in node.type else torch.float32
            shape = tuple(x if isinstance(x, int) else 1 for x in node.shape)
            self.inputs[node.name] = torch.zeros(
                shape, device=device, dtype=dtype
            )
            arr = self.inputs[node.name]
            self.binding.bind_input(
                node.name,
                "cuda",
                device.index or 0,
                np.int64 if dtype == torch.int64 else np.float32,
                shape,
                arr.data_ptr(),
            )
        for node in session.get_outputs():
            shape = tuple(x if isinstance(x, int) else 1 for x in node.shape)
            dtype = torch.int64 if "int64" in node.type else torch.float32
            arr = torch.empty(shape, device=device, dtype=dtype)
            self.outputs[node.name] = arr
            self.binding.bind_output(
                node.name,
                "cuda",
                device.index or 0,
                np.int64 if dtype == torch.int64 else np.float32,
                shape,
                arr.data_ptr(),
            )

    def run(self):
        from .onnx_device import verify_cuda_execution

        self.session.run_with_iobinding(self.binding)
        # A completion event is required for correct latency evidence and consumers.
        torch.cuda.synchronize(self.device)
        verify_cuda_execution(self.session)
        return self.outputs


def install():
    from .hand_rule_runtime import install as install_hand_rules

    install_hand_rules()
    from . import onnx_device

    original_session = onnx_device.create_device_session

    def cached_session(path, *args, **kwargs):
        p = Path(path)
        st = p.stat()
        key = (
            str(p.resolve()),
            st.st_size,
            st.st_mtime_ns,
            repr(args),
            kwargs.get("device_id", 0),
        )
        if key not in SESSIONS:
            SESSIONS[key] = original_session(path, *args, **kwargs)
            while len(SESSIONS) > 16:
                SESSIONS.popitem(last=False)
        SESSIONS.move_to_end(key)
        return SESSIONS[key]

    from . import lower
    from . import upper

    lower.create_device_session = cached_session
    upper.create_device_session = cached_session
    checked_models = OrderedDict()
    schemas = {}
    old_check = upper.check_model
    old_schema = upper.schema_for_model

    def check_model(path):
        key = (str(Path(path).resolve()), upper.sha256(Path(path)))
        if key not in checked_models:
            checked_models[key] = old_check(path)
            while len(checked_models) > 16:
                checked_models.popitem(last=False)
        return checked_models[key]

    def schema_for_model(graph):
        # Graph object identities remain valid while held by this bounded cache.
        key = id(graph)
        if key not in schemas:
            schemas[key] = (graph, old_schema(graph))
        if len(schemas) > 16:
            schemas.pop(next(iter(schemas)))
        return schemas[key][1]

    upper.check_model = check_model
    upper.schema_for_model = schema_for_model
    from . import match_runtime as match

    cls = match.BoxingMatchScene
    original_scene = cls
    current_scene = {}

    def cached_scene(**kwargs):
        key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
        if key not in SCENES:
            scene = original_scene(**kwargs)
            scene._actual_close = scene.close
            scene.close = lambda: None
            SCENES[key] = scene
        scene = SCENES[key]
        # A source anchor belongs only to an explicitly constructed offline
        # probe; a later online match must not inherit it from a cached scene.
        scene.__dict__.pop("_source_reference_root_anchor", None)
        if not hasattr(scene, "_observation_fk"):
            scene._observation_fk = CurrentObservationFK(scene)
        scene._observation_fk.refreshes = 0
        scene.prepare_actor_observation = scene._observation_fk.refresh
        scene._gpu_referee = None
        scene._live_policies = []
        scene._upper_pair = None
        scene._pending_contacts = None
        scene._gpu_physics.steps = 0
        scene._gpu_physics.gpu_frames = 0
        current_scene["scene"] = scene
        return scene

    match.BoxingMatchScene = cached_scene
    old_referee = match.MatchStateMachine

    def cuda_referee(rules):
        # The deployed competition is single-round. Preserve legacy multi-round support.
        if rules.num_rounds != 1:
            return old_referee(rules)
        from .gpu_referee import CudaReferee

        scene = current_scene["scene"]
        referee = CudaReferee(rules, scene)
        scene._gpu_referee = referee
        return referee

    match.MatchStateMachine = cuda_referee
    old_fallen = match._fallen

    def fallen(kinematics, **kwargs):
        if (
            current_scene["scene"]._gpu_referee is None
            or os.environ.get("BOXING_COMPARE_OBSERVATIONS") == "1"
        ):
            return old_fallen(kinematics, **kwargs)
        return None

    match._fallen = fallen
    old_scene_reset = match._reset_scene

    def scene_reset(scene, *args):
        old_scene_reset(scene, *args)
        scene.prepare_actor_observation()
        # MuJoCo Warp allocates new data on round reset; graphs must not retain old pointers.
        for policy in scene._live_policies:
            for name in ("qpos", "qvel"):
                setattr(
                    policy,
                    name,
                    wp.to_torch(getattr(scene._gpu_physics.data, name))[0],
                )
            policy.xpos = wp.to_torch(scene._observation_fk.positions)
            policy.xquat = wp.to_torch(scene._observation_fk.rotations)
            if hasattr(policy, "obs_graph"):
                del policy.obs_graph
        pair = scene._upper_pair
        if pair is not None:
            pair.positions = wp.to_torch(scene._observation_fk.positions)
            pair.rotations = wp.to_torch(scene._observation_fk.rotations)
            if hasattr(pair, "feature_graph"):
                del pair.feature_graph

    match._reset_scene = scene_reset

    old_init = lower.LinkPolicy.__init__

    def init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.bound = BoundSession(self.session, self.device)
        self.default_gpu = torch.as_tensor(
            self.default_joint_position, device=self.device
        )
        self.scale_gpu = torch.as_tensor(self.action_scale, device=self.device)
        self.order_gpu = torch.as_tensor(
            self.model_to_articulation, device=self.device
        )
        self.last_gpu = torch.zeros(29, device=self.device)
        self.bound.inputs["past_key_values"].zero_()
        self.bound.inputs["step_idx"].zero_()

    lower.LinkPolicy.__init__ = init
    old_reset = lower.LinkPolicy.reset

    def reset(self):
        old_reset(self)
        if hasattr(self, "bound"):
            self.bound.inputs["past_key_values"].zero_()
            self.bound.inputs["step_idx"].zero_()
            self.last_gpu.zero_()

    lower.LinkPolicy.reset = reset

    old_attach = lower.LinkPolicy.attach_scene

    def attach(self, scene):
        old_attach(self, scene)
        scene._live_policies.append(self)
        self.qpos = wp.to_torch(scene._gpu_physics.data.qpos)[0]
        self.qvel = wp.to_torch(scene._gpu_physics.data.qvel)[0]
        self.xpos = wp.to_torch(scene._observation_fk.positions)
        self.xquat = wp.to_torch(scene._observation_fk.rotations)
        self.joint_ids = [
            torch.as_tensor(
                scene._joint_qpos[f][self.articulation_to_model],
                device=self.device,
            )
            for f in range(2)
        ]
        self.vel_ids = [
            torch.as_tensor(
                scene._joint_dof[f][self.articulation_to_model],
                device=self.device,
            )
            for f in range(2)
        ]
        self.body_gpu = [
            torch.as_tensor(ids, device=self.device) for ids in self.body_ids
        ]
        from .reference import FIVE_POINT_MASK

        self.mask = torch.as_tensor(FIVE_POINT_MASK.copy(), device=self.device)
        self.offsets = torch.tensor(
            [1, 2, 3, 4, 5], device=self.device, dtype=torch.float32
        )
        self.gravity = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        self.active = torch.nonzero(self.mask).flatten()

    lower.LinkPolicy.attach_scene = attach

    def observation(self, reference, kinematics, fighter, current_index):
        schema = self.schema
        F, L = schema.frames, schema.num_links
        task = reference[: schema.task_dim].double().reshape(F, schema.width)
        pos = task[:, : L * 3].reshape(F, L, 3)[:, self.active]
        quat = xz_quaternion(
            task[:, L * 3 : -1].reshape(F, L, 6)[:, self.active]
        )
        root_body = self.upper_body_host[fighter, 0]
        root_rotation = self.xquat[root_body].double()
        root_rotation = root_rotation / torch.linalg.vector_norm(root_rotation)
        root_position = self.xpos[root_body].double().clone()
        root_position[2] = reference[schema.task_dim]
        future_pos = rotate(root_rotation[None, None], pos) + root_position
        future_quat = multiply(root_rotation[None, None], quat)
        ids = self.body_gpu[fighter]
        neutral = self.xquat[ids].double()
        neutral = neutral / torch.linalg.vector_norm(
            neutral, dim=-1, keepdim=True
        )
        refpos = self.xpos[ids].double()[None].expand(F, L, 3).clone()
        refquat = neutral[None].expand(F, L, 4).clone()
        refpos[:, self.active] = future_pos
        refquat[:, self.active] = future_quat
        qi = self.scene._root_qpos[fighter]
        vi = self.scene._root_dof[fighter]
        actual_root = self.qpos[qi + 3 : qi + 7].float()[None]
        inv = conjugate(actual_root)
        local_pos = rotate(
            inv[:, None],
            self.xpos[ids].float()[None]
            - self.qpos[qi : qi + 3].float()[None, None],
        )
        local_quat = multiply(inv[:, None], self.xquat[ids].float()[None])
        reference_root = getattr(
            self.scene, "_source_reference_root_anchor", None
        )
        reference_root = (
            root_position.float()[None]
            if reference_root is None
            else reference_root[fighter : fighter + 1]
        )
        command = build_link_observation(
            LinkObservationInput(
                reference_link_pos=refpos.float()[None],
                reference_link_quat_wxyz=refquat.float()[None],
                reference_root_pos=reference_root,
                reference_root_height=reference_root[:, 2:3],
                robot_root_quat_wxyz=actual_root,
                robot_link_pos_local=local_pos,
                robot_link_quat_local_wxyz=local_quat,
                frame_offsets=self.offsets,
                link_mask=self.mask,
            ),
            root_link_index=0,
            include_error_terms=True,
        )
        proprio = torch.cat(
            (
                rotate(inv, self.gravity),
                self.qvel[vi + 3 : vi + 6].float()[None],
                self.qpos[self.joint_ids[fighter]].float()[None]
                - self.default_gpu[None],
                self.qvel[self.vel_ids[fighter]].float()[None],
                self.last_gpu[None],
            ),
            -1,
        )
        return pack_link_actor_observation(
            proprio, command, include_error_terms=True
        )

    lower.LinkPolicy.gpu_observation = observation

    def act(
        self, *, reference_qpos, kinematics, fighter, reference_current_index=0
    ):
        if not bool(
            torch.isfinite(self.qpos).all()
            & torch.isfinite(self.qvel).all()
            & torch.isfinite(self.xpos).all()
            & torch.isfinite(self.xquat).all()
        ):
            raise ValueError(
                "Nonfinite physical state before frozen-lower observation"
            )
        if not hasattr(self, "obs_graph"):
            # The observation graph reads fixed physics/reference/cache addresses.
            torch.cuda.synchronize(self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.gpu_observation(
                        reference_qpos,
                        kinematics,
                        fighter,
                        reference_current_index,
                    )
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.obs_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.obs_graph, stream=stream):
                self.graph_observation = self.gpu_observation(
                    reference_qpos,
                    kinematics,
                    fighter,
                    reference_current_index,
                )
        self.obs_graph.replay()
        obs = self.graph_observation
        if not bool(torch.isfinite(obs).all()):
            raise ValueError(
                "Nonfinite frozen-lower observation; physical state is not sanitized"
            )
        if (
            os.environ.get("BOXING_COMPARE_OBSERVATIONS") == "1"
            and self.step_index < 5
        ):
            schema = self.schema
            data = self.scene._observation_fk.data
            data.qvel[:] = self.scene.data.qvel
            positions = data.xpos[self.upper_body.cpu().numpy()]
            rotations = data.xquat[self.upper_body.cpu().numpy()]
            targets = schema.to_targets(
                reference_qpos.cpu().numpy(),
                positions[fighter, 0],
                rotations[fighter, 0],
            )
            expected = self.observation(
                targets, kinematics, fighter, 0, data=data
            )
            error = float(np.max(np.abs(expected - obs.cpu().numpy())))
            METRICS["max_observation_error"] = max(
                METRICS.get("max_observation_error", 0), error
            )
            np.testing.assert_allclose(
                obs.cpu().numpy(), expected, atol=3e-5, rtol=3e-5
            )
        self.bound.inputs["obs"].copy_(obs)
        self.bound.inputs["step_idx"].fill_(self.step_index)
        start = time.perf_counter_ns()
        outputs = self.bound.run()
        end = time.perf_counter_ns()
        if (
            os.environ.get("BOXING_COMPARE_OBSERVATIONS") == "1"
            and self.step_index < 5
        ):
            feed = {
                name: value.cpu().numpy().copy()
                for name, value in self.bound.inputs.items()
            }
            expected = self.session.run(
                ["actions", "present_key_values"], feed
            )
            actual = outputs["actions"].cpu().numpy()
            error = float(np.max(np.abs(expected[0] - actual)))
            METRICS["max_binding_action_error"] = max(
                METRICS.get("max_binding_action_error", 0), error
            )
            np.testing.assert_allclose(
                actual, expected[0], atol=1e-6, rtol=1e-6
            )
            np.testing.assert_allclose(
                outputs["present_key_values"].cpu().numpy(),
                expected[1],
                atol=1e-6,
                rtol=1e-6,
            )
        self.bound.inputs["past_key_values"].copy_(
            outputs["present_key_values"]
        )
        self.last_gpu.copy_(outputs["actions"].reshape(29))
        if not bool(
            torch.isfinite(self.last_gpu).all()
            & torch.isfinite(outputs["present_key_values"]).all()
        ):
            raise ValueError("nonfinite CUDA policy output")
        self.step_index += 1
        if self.step_index >= self.maximum_context_length:
            self.step_index = 0
            self.bound.inputs["past_key_values"].zero_()
        target = (self.default_gpu + self.scale_gpu * self.last_gpu)[
            self.order_gpu
        ]
        wp.to_torch(self.scene._gpu_physics.targets_gpu).reshape(2, 29)[
            fighter
        ].copy_(target)
        self.scene._targets_on_gpu = True
        host = torch.stack((self.last_gpu, target)).cpu().numpy()
        action = host[0].copy()
        self.last_action = action
        return action, host[1], start, end

    lower.LinkPolicy.act = act

    old_pair_init = upper.FivePointReferencePair.__init__

    def pair_init(self, paths, scene, policies, **kwargs):
        old_pair_init(self, paths, scene, policies, **kwargs)
        scene._upper_pair = self
        self.bound = [BoundSession(s, self.device) for s in self.sessions]
        self.body_gpu = torch.as_tensor(self.body_ids, device=self.device)
        self.positions = wp.to_torch(scene._observation_fk.positions)
        self.rotations = wp.to_torch(scene._observation_fk.rotations)
        self.references = torch.stack(
            [s.template.to(self.device) for s in self.schemas]
        )
        for p, s in zip(policies, self.schemas):
            p.schema = s
            p.upper_body = self.body_gpu
            p.upper_body_host = self.body_ids
        self.metadata.update(
            upper_observation_device=str(self.device),
            lower_observation_device=str(self.device),
            optimization_version="resident-cuda-v1",
            kv_cache_device=str(self.device),
        )

    upper.FivePointReferencePair.__init__ = pair_init

    @torch.no_grad()
    def window(self, kinematics, *, fighters=(0, 1)):
        from .policy_output import sanitize_target_pose

        positions = self.positions[self.body_gpu].float()
        quaternions = self.rotations[self.body_gpu].float()
        quaternions = quaternions / torch.linalg.vector_norm(
            quaternions, dim=-1, keepdim=True
        )
        data = self.scene._gpu_physics.data
        cvel = wp.to_torch(data.cvel)[0][self.body_gpu].float()
        roots = torch.as_tensor(
            self.scene.model.body_rootid[self.body_ids], device=self.device
        )
        com = wp.to_torch(data.subtree_com)[0][roots].float()
        velocity = cvel[..., 3:] + torch.cross(
            cvel[..., :3], positions - com, dim=-1
        )
        qvel = wp.to_torch(data.qvel)[0]
        omega = torch.stack(
            [qvel[v + 3 : v + 6].float() for v in self.scene._root_dof]
        )
        observations = self.observations(
            positions, quaternions, velocity, omega
        )
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("Nonfinite measured upper-policy observation")
        results, starts, ends, sanitized, sanitation_ns = [], [], [], [], []
        for fighter in fighters:
            bound = self.bound[fighter]
            bound.inputs["obs"].copy_(observations[fighter : fighter + 1])
            start = time.perf_counter_ns()
            output = bound.run()["target_pose"]
            end = time.perf_counter_ns()
            sanitation_start = time.perf_counter_ns()
            output, statistics = sanitize_target_pose(output)
            sanitation_ns.append(time.perf_counter_ns() - sanitation_start)
            sanitized.append(statistics)
            self.sanitization_total[fighter] += statistics[
                "nonfinite_components"
            ]
            results.append(
                self.reference(output[0], fighter, positions[fighter, 0, 2])
            )
            starts.append(start)
            ends.append(end)
        self.history_steps += 1
        self.frame_data = dict(
            upper_policy_start_ns=starts,
            upper_policy_end_ns=ends,
            upper_history_step=self.history_steps,
            competitor_sanitization=sanitized,
            competitor_sanitization_total=self.sanitization_total.copy(),
            competitor_sanitization_ns=sanitation_ns,
        )
        return results

    upper.FivePointReferencePair.window = window

    # CPU evidence consumers only require the state below, not 28 per-body velocity queries.
    def read_kinematics(scene):
        from .scene import SceneKinematics

        d = scene.data
        root = np.stack([d.qpos[q : q + 3] for q in scene._root_qpos]).astype(
            np.float32
        )[None]
        quat = np.stack(
            [d.qpos[q + 3 : q + 7] for q in scene._root_qpos]
        ).astype(np.float32)[None]
        linear = np.stack([d.qvel[v : v + 3] for v in scene._root_dof]).astype(
            np.float32
        )[None]
        angular = np.stack(
            [d.qvel[v + 3 : v + 6] for v in scene._root_dof]
        ).astype(np.float32)[None]
        raw_quat = np.stack([d.qpos[q + 3 : q + 7] for q in scene._root_qpos])
        raw_angular = np.stack(
            [d.qvel[v + 3 : v + 6] for v in scene._root_dof]
        )
        world = world_angular_velocity(raw_quat, raw_angular)[None]
        jp = d.qpos[np.asarray(scene._joint_qpos)].astype(np.float32)[None]
        jv = d.qvel[np.asarray(scene._joint_dof)].astype(np.float32)[None]
        bp = np.zeros((1, 2, len(scene.body_names), 3), dtype=np.float32)
        return SceneKinematics(
            root_position=root,
            root_quaternion_wxyz=quat,
            root_linear_velocity=linear,
            root_angular_velocity_local=angular,
            root_angular_velocity_world=world,
            joint_position=jp,
            joint_velocity=jv,
            body_position=bp,
            body_linear_velocity=np.zeros_like(bp),
        )

    original_scene.read_kinematics = read_kinematics

    from .gpu_physics import ResidentPhysics

    # Keep the existing captured four substeps, capacities, contact order and solver.
    original_read = ResidentPhysics._readback

    def readback(self):
        if not hasattr(self, "stats"):
            return original_read(self)
        if not hasattr(self, "snapshot_stream"):
            self.snapshot_stream = wp.stream_to_torch(
                wp.get_stream(str(self.device))
            )
        torch.cuda.current_stream(self.device).wait_stream(
            self.snapshot_stream
        )
        # Height-only falls read pelvis AND torso xpos. Refresh body snapshots
        # even outside comparison mode; qpos alone leaves stale heights.
        names = (
            ("qpos", "qvel", "xpos", "xquat", "time")
            if os.environ.get("BOXING_COMPARE_OBSERVATIONS") == "1"
            else ("qpos", "qvel", "xpos", "time")
        )
        arrays = [
            wp.to_torch(getattr(self.data, name))[0].reshape(-1)
            for name in names
        ]
        arrays.extend(
            [
                wp.to_torch(self.stats).reshape(-1),
                wp.to_torch(self.overflow).float(),
            ]
        )
        if (
            not hasattr(self, "snapshot_host")
            or getattr(self, "snapshot_names", None) != names
        ):
            self.snapshot_names = names
            self.snapshot_sizes = [x.numel() for x in arrays]
            self.snapshot_host = torch.empty(
                sum(self.snapshot_sizes), dtype=torch.float32, pin_memory=True
            )
        # One pinned transfer instead of repeated allocating numpy() copies.
        packed = torch.cat(arrays)
        self.snapshot_host.copy_(packed, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        host = self.snapshot_host.numpy()
        offset = 0
        if not np.isfinite(host).all():
            raise ValueError(
                "Nonfinite measured physics snapshot; no sanitation is allowed"
            )
        for name, size in zip(names, self.snapshot_sizes):
            values = host[offset : offset + size]
            offset += size
            if name == "time":
                self.scene.data.time = float(values[0])
            else:
                getattr(self.scene.data, name)[:] = values.reshape(
                    getattr(self.scene.data, name).shape
                )
        size = self.snapshot_sizes[-2]
        self._snapshot_stats = host[offset : offset + size].reshape(
            len(self.keys), 6
        )
        self._snapshot_overflow = host[-1]

    ResidentPhysics._readback = readback
