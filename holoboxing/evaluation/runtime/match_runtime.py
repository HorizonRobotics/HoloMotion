"""Shared HoloMotion V2 match loop and versioned initial pose."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import os
import numpy as np

from .scene import BoxingMatchScene
from .lower import LinkPolicy, verify_checkpoint, name_order as _name_order
from .model_contract import sha256
from .rules import MatchLogger, MatchRules, MatchStateMachine
from .hand_hit_rules import SCORING_VERSION, scoring_config
from .fall_rules import FALL_HEIGHT_M, fallen_from_scene
from .deterministic_physics import ORDERING_VERSION
from .ordered_topk import ORDERING_VERSION as INFERENCE_ORDERING

REFERENCE_HZ = 50
DOF_COUNT = 29


@dataclass(frozen=True)
class ResetPose:
    qpos: np.ndarray
    dof_names: tuple[str, ...]
    source_path: Path

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(strict=True)
        packet = json.loads(path.read_text())
        positions = np.asarray(packet["qpos"], dtype=np.float32)
        names = tuple(packet["dof_names"])
        if packet.get("schema_version") != 1 or packet.get("target_fps") != 50:
            raise ValueError("Unsupported versioned reset pose")
        if positions.shape != (2, 36) or not np.isfinite(positions).all():
            raise ValueError("Reset pose must contain finite qpos[2,36]")
        if len(names) != 29 or len(set(names)) != 29:
            raise ValueError("Reset pose requires 29 unique joint names")
        if not np.allclose(
            np.linalg.norm(positions[:, 3:7], axis=-1), 1, atol=1e-5
        ):
            raise ValueError("Reset pose quaternions must be normalized")
        return cls(positions[None], names, path)


@dataclass
class MatchConfig:
    retargeted: str | Path
    policy_0: str | Path
    policy_1: str | Path
    mujoco_model: str | Path
    mesh_root: str | Path
    output: str | Path
    device: str = "cuda:0"
    match_id: str = field(default_factory=lambda: f"boxing-{time.time_ns()}")
    model_id_0: str | None = None
    model_id_1: str | None = None
    num_rounds: int = 1
    round_duration_s: float = 60.0
    ko_confirmation_s: float = 3.0
    minimum_closing_speed_mps: float = 0.1
    minimum_contact_force_n: float = 1.0
    fall_height_m: float = 0.3
    fall_tilt_deg: float = 0.0
    headless: bool = True
    no_realtime: bool = True
    max_frames: int = 0
    allow_incomplete: bool = False
    log_every: int = 500


def _fallen(kinematics, *, height_m, tilt_deg, scene):
    if height_m != FALL_HEIGHT_M:
        raise ValueError("Official fall height must be 0.30 m")
    return fallen_from_scene(scene)


def _reset_scene(scene, motion, articulation_order):
    initial = motion.qpos[0]
    values = np.concatenate(
        (initial[:, :7], initial[:, 7:][:, articulation_order]), axis=-1
    )
    scene.write_pair_state(values)
    scene.drain_reference_frame_qpos()


def run_match(arguments: MatchConfig, *, reference_factory=None) -> Path:
    if reference_factory is None:
        raise ValueError("Two live competitor policies are required")
    motion = ResetPose.load(arguments.retargeted)
    model_paths = [
        Path(arguments.policy_0).expanduser().resolve(strict=True),
        Path(arguments.policy_1).expanduser().resolve(strict=True),
    ]
    release_configs = [verify_checkpoint(path) for path in model_paths]
    rules = MatchRules(
        num_rounds=arguments.num_rounds,
        round_duration_s=arguments.round_duration_s,
        ko_confirmation_s=arguments.ko_confirmation_s,
        minimum_closing_speed_mps=arguments.minimum_closing_speed_mps,
        minimum_contact_force_n=arguments.minimum_contact_force_n,
    )
    scene = BoxingMatchScene(
        model_path=arguments.mujoco_model,
        mesh_root=arguments.mesh_root,
        simulation_frequency=200,
        control_decimation=4,
        contact_force_threshold=arguments.minimum_contact_force_n,
        show_viewer=not arguments.headless,
        realtime=not arguments.no_realtime,
        physics_device=arguments.device,
    )
    try:
        policies = tuple(
            LinkPolicy(
                checkpoint,
                device=arguments.device,
                articulation_joint_names=scene.articulation_joint_names,
                retarget_dof_names=motion.dof_names,
            )
            for checkpoint in model_paths
        )
        if policies[0].joint_names != policies[1].joint_names:
            raise ValueError(
                "the two HoloMotion models use different joint schemas"
            )
        if not np.allclose(policies[0].kp, policies[1].kp) or not np.allclose(
            policies[0].kd, policies[1].kd
        ):
            raise ValueError("the two policies require different PD gains")
        scene._kp = (
            policies[0]
            .kp[policies[0].model_to_articulation]
            .astype(np.float64)
        )
        scene._kd = (
            policies[0]
            .kd[policies[0].model_to_articulation]
            .astype(np.float64)
        )
        articulation_order = _name_order(
            motion.dof_names,
            scene.articulation_joint_names,
            "retargeted motion",
        )
        _reset_scene(scene, motion, articulation_order)
        for policy in policies:
            policy.reset()

        reference_provider = (
            reference_factory(scene, policies)
            if reference_factory is not None
            else None
        )
        if reference_provider is not None:
            for policy in policies:
                policy.retarget_to_model = np.arange(DOF_COUNT, dtype=np.int64)

        model_hashes = {path: sha256(path) for path in set(model_paths)}
        model_ids = [
            arguments.model_id_0
            or f"{model_paths[0].stem}:{model_hashes[model_paths[0]][:12]}",
            arguments.model_id_1
            or f"{model_paths[1].stem}:{model_hashes[model_paths[1]][:12]}",
        ]
        logger = MatchLogger(
            model_id=model_ids,
            match_id=arguments.match_id,
            rules=rules,
            metadata={
                "control_hz": REFERENCE_HZ,
                "physics_hz": 200,
                "arena_version": "smplolympics-boxing-ring-v2",
                "scoring_version": SCORING_VERSION,
                "scoring_config": scoring_config(),
                "physics_backend": scene.physics_backend,
                "physics_ordering": ORDERING_VERSION,
                "inference_ordering": INFERENCE_ORDERING,
                "physics_device": scene.physics_device,
                "lower_policy_provider": [
                    policy.session.get_providers()[0] for policy in policies
                ],
                "lower_policy_device": [
                    str(policy.device) for policy in policies
                ],
                "lower_observation_device": str(policies[0].device),
                "frame_semantics": "50_hz_policy_control_frame",
                "policy_timing_scope": "onnx_session_run",
                "evaluation_core": "holoboxing",
                "core_source_sha256": sha256(Path(__file__)),
                "actor_observation_source": policies[
                    0
                ].actor_observation_source,
                "retargeted_path": str(motion.source_path),
                "retargeted_sha256": sha256(motion.source_path),
                "policy_path": [str(path) for path in model_paths],
                "policy_sha256": [model_hashes[path] for path in model_paths],
                "release_config": [str(path) for path in release_configs],
                "dof_names": list(scene.articulation_joint_names),
                "action_joint_names": list(policies[0].joint_names),
                "quaternion_order": "wxyz",
                **(
                    reference_provider.metadata
                    if reference_provider is not None
                    else {}
                ),
            },
        )
        scene._fall_height_m = arguments.fall_height_m
        scene._fall_tilt_deg = arguments.fall_tilt_deg
        match = MatchStateMachine(rules)
        if reference_provider is not None:
            reference_provider.match = match
        global_frame = 0
        round_frame = 0
        maximum_frames = arguments.max_frames or (
            rules.num_rounds
            * int(round(rules.round_duration_s * REFERENCE_HZ))
        )
        while (
            not match.match_over
            and global_frame < maximum_frames
            and scene.viewer_running
        ):
            control_step_start_ns = time.perf_counter_ns()
            prepare_observation = getattr(
                scene, "prepare_actor_observation", None
            )
            if prepare_observation is not None:
                prepare_observation()
            before = (
                scene.read_kinematics()
                if os.getenv("BOXING_COMPARE_OBSERVATIONS") == "1"
                else None
            )
            reference = (
                reference_provider.window(before)
                if reference_provider is not None
                else (_ for _ in ()).throw(
                    RuntimeError("Missing live reference provider")
                )
            )
            actions = np.zeros((2, DOF_COUNT), dtype=np.float32)
            targets = np.zeros((2, DOF_COUNT), dtype=np.float32)
            starts = [0, 0]
            ends = [0, 0]
            for fighter, policy in enumerate(policies):
                action, target, start_ns, end_ns = policy.act(
                    reference_qpos=reference[fighter],
                    kinematics=before,
                    fighter=fighter,
                    reference_current_index=(
                        reference_provider.current_index
                        if reference_provider is not None
                        else 0
                    ),
                )
                actions[fighter] = action
                targets[fighter] = target
                starts[fighter] = start_ns
                ends[fighter] = end_ns
            scene.stage_pair_joint_targets(targets[None])
            scene.step_reference_frame()
            scene.drain_reference_frame_qpos()
            global_frame += 1
            round_frame += 1
            sim_time_s = global_frame / float(REFERENCE_HZ)
            after = scene.read_kinematics()
            contacts = scene.contact_samples(
                sim_time_s=np.asarray((sim_time_s,), dtype=np.float64)
            )[0]
            result = match.observe_frame(
                sim_time_s=sim_time_s,
                contacts=contacts,
                root_pos=after.root_position[0],
                fallen=_fallen(
                    after,
                    height_m=arguments.fall_height_m,
                    tilt_deg=arguments.fall_tilt_deg,
                    scene=scene,
                ),
            )
            logger.record_frame(
                root_pos=after.root_position[0],
                root_quat=after.root_quaternion_wxyz[0],
                dof_pos=after.joint_position[0],
                action_raw=actions,
                policy_start_ns=starts,
                policy_end_ns=ends,
                score=result.score,
                events=result.events,
            )
            logger.frames[-1].update(
                joint_velocity=after.joint_velocity[0].tolist(),
                root_linear_velocity=after.root_linear_velocity[0].tolist(),
                root_angular_velocity_world=after.root_angular_velocity_world[
                    0
                ].tolist(),
            )
            logger.frames[-1]["contacts"] = match.contact_diagnostics
            if reference_provider is not None:
                logger.frames[-1].update(reference_provider.frame_data)
            logger.frames[-1].update(
                control_step_start_ns=control_step_start_ns,
                control_step_end_ns=0,
            )
            if arguments.log_every > 0 and (
                global_frame == 1
                or global_frame % arguments.log_every == 0
                or result.events
            ):
                print(
                    json.dumps(
                        {
                            "frame": global_frame,
                            "round": result.round_number,
                            "sim_time_s": sim_time_s,
                            "score": result.score,
                            "events": result.events,
                        },
                        default=lambda value: (
                            value.tolist()
                            if isinstance(value, np.ndarray)
                            else value
                        ),
                    ),
                    flush=True,
                )
            if result.round_ended and not result.match_over:
                _reset_scene(scene, motion, articulation_order)
                for policy in policies:
                    policy.reset()
                if reference_provider is not None:
                    reference_provider.reset()
                round_frame = 0
            # The complete frame includes in-memory replay/log preparation.
            # Final file encoding/writes and independent browser playback run
            # outside the steady-state control loop.
            logger.frames[-1]["control_step_end_ns"] = time.perf_counter_ns()
        if not match.match_over and not arguments.allow_incomplete:
            raise RuntimeError(
                "match did not finish; increase --max-frames or keep the "
                "viewer open"
            )
        from .chain_benchmark import control_loop_report

        logger.extra_metadata["control_loop_benchmark"] = control_loop_report(
            logger.frames
        )
        if reference_provider is not None:
            logger.extra_metadata["competitor_sanitization_total"] = (
                reference_provider.sanitization_total.copy()
            )
        logger.extra_metadata["match_completed"] = match.match_over
        logger.extra_metadata["frames_recorded"] = global_frame
        observation_fk = getattr(scene, "_observation_fk", None)
        if observation_fk is not None:
            logger.extra_metadata.update(
                observation_fk_source="current_qpos_independent_fk",
                observation_fk_refreshes=observation_fk.refreshes,
                observation_fk_control_hz=REFERENCE_HZ,
                observation_fk_changes_physics_state=False,
            )
        logger.extra_metadata["lower_cuda_verified"] = [
            policy.session.boxing_cuda_verified for policy in policies
        ]
        logger.extra_metadata["upper_cuda_verified"] = (
            [
                session.boxing_cuda_verified
                for session in reference_provider.sessions
            ]
            if reference_provider is not None
            else []
        )
        logger.extra_metadata["physics_cuda_steps"] = (
            scene._gpu_physics.steps if scene._gpu_physics is not None else 0
        )
        logger.extra_metadata.update(
            physics_execution="gpu_resident_cuda_graph",
            pd_device="cuda",
            contact_reduction_device="cuda",
            host_snapshot_hz=50,
            referee_device="cpu",
        )
        return logger.write(arguments.output, match.score)
    finally:
        scene.close()
