"""Live boxing ONNX adapter using the canonical VR-5 schema and history."""

from pathlib import Path
import time

import os
import numpy as np
from .compiled_artifacts import available, DIRECTORY, reference_schema
from .onnx_device import (
    create_device_session,
    require_gpu,
    require_gpu_session,
)
from .model_contract import (
    CONTRACT,
    INPUT_LINK_NAMES,
    check_model,
    check_contract,
    sha256,
)
from .policy_output import (
    SANITIZATION_VERSION,
    sanitize_target_pose,
    run_competitor,
)


def schema_for_model(graph):
    check_contract(graph)
    return reference_schema()


def check_target_pose(value):
    value = np.asarray(value)
    if (
        value.dtype != np.float32
        or value.shape != (1, 225)
        or not np.isfinite(value).all()
    ):
        raise ValueError("Expected finite float32 target_pose[1,225]")
    axes = value.reshape(5, 5, 9)[..., 3:]
    x, z = axes[..., :3], axes[..., 3:]
    if (
        not np.allclose(np.linalg.norm(x, axis=-1), 1, atol=1e-3)
        or not np.allclose(np.linalg.norm(z, axis=-1), 1, atol=1e-3)
        or not np.allclose((x * z).sum(-1), 0, atol=1e-3)
    ):
        raise ValueError("target_pose X/Z must be orthogonal unit vectors")


def neutral_observation(schema=None):
    if available():
        reference_schema()
        return np.load(
            DIRECTORY / "neutral_observation.npy", allow_pickle=False
        )
    import torch

    from . import representation

    positions = torch.tensor(
        [
            [
                [-0.5, 0, 0.8],
                [-0.5, 0.3, 1],
                [-0.5, -0.3, 1],
                [-0.5, 0.1, 0.05],
                [-0.5, -0.1, 0.05],
            ],
            [
                [0.5, 0, 0.8],
                [0.5, 0.3, 1],
                [0.5, -0.3, 1],
                [0.5, 0.1, 0.05],
                [0.5, -0.1, 0.05],
            ],
        ],
        dtype=torch.float32,
    )
    quaternions = torch.zeros((2, 5, 4))
    quaternions[..., 0] = 1
    raw = representation.raw_self(
        positions,
        quaternions,
        torch.zeros_like(positions),
        torch.zeros((2, 3)),
        positions,
        quaternions,
    )
    state = representation.reframe_c(
        raw[:, None].expand(-1, 10, -1),
        positions,
        quaternions,
        positions.flip(0),
    )
    return torch.cat(
        (
            state["history"],
            state["task"][:, -15:],
            torch.full((2, 1), 60.0),
            torch.zeros((2, 1)),
        ),
        -1,
    )[:1].numpy()


class FivePointReferencePair:
    current_index = 0

    def __init__(self, paths, scene, lower_policies, *, device="cuda:0"):
        self.scene = scene
        self.device = require_gpu(device)
        self.schemas, self.sessions = [], []
        self.match = None
        self.body_ids = np.asarray(
            [
                [
                    scene.model.body(prefix + name).id
                    for name in INPUT_LINK_NAMES
                ]
                for prefix in ("p1_", "p2_")
            ]
        )
        provider = "CUDAExecutionProvider"
        for path, policy in zip(paths, lower_policies, strict=True):
            graph, total = check_model(Path(path))
            if total >= min(
                50_000_000,
                int(os.environ.get("BOXING_MAX_MODEL_BYTES", "50000000")),
            ):
                raise ValueError("Upper model size must be <50,000,000 bytes")
            schema = schema_for_model(graph)
            self.schemas.append(schema)
            self.sessions.append(
                create_device_session(
                    path, provider, device_id=self.device.index or 0
                )
            )
            require_gpu_session(self.sessions[-1])
            policy.attach_scene(scene)
        from .lower import DEPLOYMENT, VERSION

        self.metadata = dict(
            upper_policy_path=[str(path) for path in paths],
            upper_policy_sha256=[sha256(Path(path)) for path in paths],
            upper_policy_contract=CONTRACT,
            upper_policy_provider=provider,
            upper_policy_device=str(self.device),
            upper_observation_device="cpu",
            upper_policy_timing_scope="onnx_session_run",
            upper_input_history_frames=[
                schema.input_history_frames for schema in self.schemas
            ],
            upper_input_frame="10 self states in current H; opponent points in current P; seconds; score difference",
            upper_history_reset="repeat first frame after every round reset",
            reference_source="online_red_blue_five_point_onnx_not_recorded_motion",
            five_point_link_names=list(INPUT_LINK_NAMES),
            five_point_schema_source_hashes=self.schemas[0].source_hashes,
            holomotion_version=VERSION,
            lower_controller_model_sha256=DEPLOYMENT["model_sha256"],
            lower_controller_config_sha256=DEPLOYMENT["config_sha256"],
            lower_observation_dim=1378,
            lower_observation_device="cpu",
            lower_command_mode="VR-5",
            lower_reference_frame_offsets=[1, 2, 3, 4, 5],
            lower_reference_source="target_pose225 -> internal reference655 -> frozen HoloMotion",
            lower_kv_reset_frames=32,
            match_loop_source_version="2.0",
            competitor_sanitization=SANITIZATION_VERSION,
        )
        self.reset()

    def reset(self):
        self.raw_history = None
        self.previous_position = self.previous_quaternion = None
        self.history_steps = 0
        self.frame_data = {}
        self.sanitization_total = [0, 0]

    def observations(self, positions, quaternions, velocity, omega):
        import torch

        from . import representation

        previous_p = (
            positions
            if self.previous_position is None
            else self.previous_position
        )
        previous_q = (
            quaternions
            if self.previous_quaternion is None
            else self.previous_quaternion
        )
        raw = representation.raw_self(
            positions, quaternions, velocity, omega, previous_p, previous_q
        )
        self.raw_history = (
            raw[:, None].expand(-1, 10, -1).clone()
            if self.raw_history is None
            else torch.cat((self.raw_history[:, 1:], raw[:, None]), 1)
        )
        state = representation.reframe_c(
            self.raw_history, positions, quaternions, positions.flip(0)
        )
        self.previous_position, self.previous_quaternion = (
            positions.clone(),
            quaternions.clone(),
        )
        if self.match is None:
            raise RuntimeError(
                "Match context is required for remaining time and score"
            )
        remaining = max(
            0.0,
            self.match.rules.round_duration_s
            - (self.match._last_sim_time_s - self.match.round_start_s),
        )
        score = torch.as_tensor(
            self.match.score, device=positions.device, dtype=torch.float32
        )
        return torch.cat(
            (
                state["history"],
                state["task"][:, -15:],
                positions.new_full((2, 1), remaining),
                (score - score.flip(0))[:, None],
            ),
            -1,
        )

    def reference(self, target_pose, fighter, root_height):
        import torch

        # Keep all five future frames and all five point slots; lower packing
        # retains the frozen controller's masks, offsets and validity bits.
        points = target_pose.reshape(5, 5, 9)
        if not bool(torch.isfinite(points).all()):
            raise ValueError("Nonfinite target_pose")
        x, z = points[..., 3:6], points[..., 6:9]
        if not bool(
            ((x.norm(dim=-1) - 1).abs() <= 1e-3).all()
            & ((z.norm(dim=-1) - 1).abs() <= 1e-3).all()
            & ((x * z).sum(-1).abs() <= 1e-3).all()
        ):
            raise ValueError("target_pose X/Z must be orthogonal unit vectors")
        schema = self.schemas[fighter]
        if not hasattr(self, "references"):
            self.references = torch.stack(
                [s.template.to(target_pose.device) for s in self.schemas]
            )
        value = self.references[fighter]
        value.copy_(schema.template.to(target_pose.device))
        task = value[: schema.task_dim].reshape(5, schema.width)
        task[:, : schema.num_links * 3].reshape(5, schema.num_links, 3)[
            :, schema.active
        ] = points[..., :3]
        task[:, schema.num_links * 3 : -1].reshape(5, schema.num_links, 6)[
            :, schema.active
        ] = points[..., 3:]
        value[schema.task_dim] = root_height
        return value

    def window(self, kinematics, *, fighters=(0, 1)):
        import torch

        data = self.scene.data
        positions = torch.as_tensor(
            data.xpos[self.body_ids].copy(), dtype=torch.float32
        )
        rotations = torch.as_tensor(
            data.xquat[self.body_ids].copy(), dtype=torch.float32
        )
        cvel = torch.as_tensor(
            data.cvel[self.body_ids].copy(), dtype=torch.float32
        )
        roots = self.scene.model.body_rootid[self.body_ids]
        com = torch.as_tensor(
            data.subtree_com[roots].copy(), dtype=torch.float32
        )
        velocity = cvel[..., 3:] + torch.cross(
            cvel[..., :3], positions - com, dim=-1
        )
        omega = torch.tensor(
            np.stack([data.qvel[v + 3 : v + 6] for v in self.scene._root_dof]),
            dtype=torch.float32,
        )
        observations = self.observations(
            positions, rotations, velocity, omega
        ).numpy()
        targets, starts, ends, sanitized, sanitation_ns = [], [], [], [], []
        for fighter in fighters:
            schema, session = self.schemas[fighter], self.sessions[fighter]
            start = time.perf_counter_ns()
            output = run_competitor(
                session, observations[fighter : fighter + 1]
            )
            end = time.perf_counter_ns()
            sanitation_start = time.perf_counter_ns()
            output, statistics = sanitize_target_pose(output)
            sanitation_ns.append(time.perf_counter_ns() - sanitation_start)
            sanitized.append(statistics)
            self.sanitization_total[fighter] += statistics[
                "nonfinite_components"
            ]
            check_target_pose(output)
            reference = self.reference(
                torch.from_numpy(output[0]), fighter, positions[fighter, 0, 2]
            )
            targets.append(
                schema.to_targets(
                    reference.numpy(),
                    positions[fighter, 0].numpy(),
                    rotations[fighter, 0].numpy(),
                )
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
        return targets
