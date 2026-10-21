"""Score a legal exported candidate against a frozen native G1 opponent.

This training evaluator reuses the competition arena and referee. Only the
opponent's inference interface is exempted; competition qualification remains
a separate check of the candidate in the unmodified evaluation image.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from holoboxing.data.contract import CompetitionHistory
from holoboxing.evaluation.resources import ResourceTree
from holoboxing.evaluation.runtime import lower as lower_runtime
from holoboxing.evaluation.runtime.compiled_artifacts import (
    available as compiled_available,
)
from holoboxing.evaluation.runtime.fall_rules import fallen_from_scene
from holoboxing.evaluation.runtime.gpu_referee import CudaReferee
from holoboxing.evaluation.runtime.hand_hit_rules import (
    SCORING_VERSION,
    scoring_config,
)
from holoboxing.evaluation.runtime.hand_rule_runtime import (
    install as install_hand_rules,
)
from holoboxing.evaluation.runtime.match_runtime import ResetPose
from holoboxing.evaluation.runtime.model_contract import (
    INPUT_LINK_NAMES,
    check_contract,
    check_model,
    sha256,
)
from holoboxing.evaluation.runtime.observation_fk import CurrentObservationFK
from holoboxing.evaluation.runtime.onnx_device import (
    create_device_session,
    require_gpu,
    verify_cuda_execution,
)
from holoboxing.evaluation.runtime.policy_output import (
    run_competitor,
    sanitize_target_pose,
)
from holoboxing.evaluation.runtime.reference import (
    FivePointTargets,
    rotation_from_xz,
)
from holoboxing.evaluation.runtime.rules import (
    MatchLogger,
    MatchRules,
)
from holoboxing.evaluation.runtime.scene import BoxingMatchScene
from holoboxing.evaluation.runtime.upper import FivePointReferencePair
from holoboxing.training.native_robostriker import (
    NativeRoboController,
    NativeRoboStriker,
)
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)


class CurrentStateLinkPolicy(lower_runtime.LinkPolicy):
    def observation(self, *args, **kwargs):
        kwargs["data"] = self.scene._observation_fk.data
        return super().observation(*args, **kwargs)


def scene_and_referee(resources, device, rules, *, runtime=None):
    """Use the competition's measured 200 Hz contacts and shared referee."""
    install_hand_rules()
    factory = BoxingMatchScene if runtime is None else runtime.BoxingMatchScene
    scene = factory(model_path=resources.scene, physics_device=str(device))
    return scene, CudaReferee(rules, scene)


def world_targets(target_pose, root_position, root_quaternion):
    """Decode all five future poses with the current competition semantics."""
    points = np.asarray(target_pose, dtype=np.float32).reshape(5, 5, 9)
    root = Rotation.from_quat(np.asarray(root_quaternion)[[1, 2, 3, 0]])
    position = (
        root.apply(points[..., :3].reshape(-1, 3)).reshape(5, 5, 3)
        + root_position
    )
    rotations = (
        root
        * Rotation.from_matrix(
            rotation_from_xz(points[..., 3:]).reshape(-1, 3, 3)
        )
    ).as_quat()
    return FivePointTargets(
        np.asarray(root_position),
        np.asarray(root_quaternion),
        position,
        rotations[:, [3, 0, 1, 2]].reshape(5, 5, 4),
        np.arange(1, 6),
    )


def read_native_state(scene, data, device):
    bodies = [
        [scene.model.body(prefix + name).id for name in BODY_NAMES]
        for prefix in ("p1_", "p2_")
    ]
    joint_ids = [
        [scene.model.joint(prefix + name).id for name in JOINT_NAMES]
        for prefix in ("p1_", "p2_")
    ]
    qi, vi = (
        scene.model.jnt_qposadr[joint_ids],
        scene.model.jnt_dofadr[joint_ids],
    )
    values = (
        data.xpos[bodies],
        data.xquat[bodies],
        np.stack([data.qvel[v + 3 : v + 6] for v in scene._root_dof]),
        data.qpos[qi],
        data.qvel[vi],
    )
    return tuple(
        torch.as_tensor(
            np.array(value, copy=True), dtype=torch.float32, device=device
        )[None]
        for value in values
    )


def legal_observation(history, scene, data, match, frame, reset):
    ids = np.asarray(
        [
            [scene.model.body(prefix + name).id for name in INPUT_LINK_NAMES]
            for prefix in ("p1_", "p2_")
        ]
    )
    position = torch.tensor(data.xpos[ids], dtype=torch.float32)
    quaternion = torch.tensor(data.xquat[ids], dtype=torch.float32)
    cvel = torch.tensor(data.cvel[ids], dtype=torch.float32)
    com = torch.tensor(
        data.subtree_com[scene.model.body_rootid[ids]], dtype=torch.float32
    )
    velocity = cvel[..., 3:] + torch.cross(
        cvel[..., :3], position - com, dim=-1
    )
    omega = torch.tensor(
        np.stack([data.qvel[v + 3 : v + 6] for v in scene._root_dof]),
        dtype=torch.float32,
    )
    remaining = max(
        0,
        match.rules.round_duration_s
        - (match._last_sim_time_s - match.round_start_s),
    )
    return history.observe(
        position[None],
        quaternion[None],
        velocity[None],
        omega[None],
        torch.tensor([remaining], dtype=torch.float32),
        torch.tensor([match.score], dtype=torch.float32),
        reset=torch.tensor([reset]),
        step=frame,
    )[0].numpy()


def initial_pair(motion, articulation_names):
    """Reorder the exact competition reset without changing its root poses."""
    order = [motion.dof_names.index(name) for name in articulation_names]
    values = np.concatenate(
        (motion.qpos[0, :, :7], motion.qpos[0, :, 7:][:, order]), -1
    ).copy()
    return values


def evaluate(
    candidate,
    opponent_checkpoint,
    opponent_sha256,
    assets,
    lower,
    output,
    *,
    candidate_side=0,
    device="cuda:0",
    max_frames=0,
):
    if (
        candidate_side not in (0, 1)
        or type(max_frames) is not int
        or max_frames < 0
    ):
        raise ValueError("Invalid candidate side or frame limit")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    graph, size = check_model(Path(candidate))
    check_contract(graph)
    if size >= 50_000_000:
        raise ValueError("Candidate must be smaller than 50,000,000 bytes")
    if sha256(opponent_checkpoint) != opponent_sha256:
        raise ValueError(
            "Native opponent checkpoint differs from the pinned version"
        )
    device = torch.device(str(require_gpu(device)))
    native = NativeRoboController(
        NativeRoboStriker(
            torch.load(
                opponent_checkpoint, map_location="cpu", weights_only=False
            )
        ).to(device)
    )
    resources = ResourceTree(assets)
    motion = ResetPose.load(resources.initial_state)
    lower_runtime.configure_bundle(lower)
    compiled = compiled_available()
    runtime = None
    if compiled:
        from holoboxing.evaluation.runtime.holomotion_v2_runtime import (
            install_runtime,
        )

        runtime = install_runtime(lower)
    upper = (
        None
        if compiled
        else create_device_session(candidate, device_id=device.index or 0)
    )
    rules = MatchRules()
    scene, match = scene_and_referee(resources, device, rules, runtime=runtime)
    output.mkdir(parents=True)
    try:
        controllers = [
            CurrentStateLinkPolicy(
                lower_runtime.MODEL_ROOT / "exported/model_10500.onnx",
                device=str(device),
                articulation_joint_names=scene.articulation_joint_names,
                retarget_dof_names=motion.dof_names,
            )
            for _ in range(2 if compiled else 1)
        ]
        for policy in controllers:
            policy.attach_scene(scene)
        controller = controllers[candidate_side if compiled else 0]
        scene._kp = controller.kp[controller.model_to_articulation].astype(
            np.float64
        )
        scene._kd = controller.kd[controller.model_to_articulation].astype(
            np.float64
        )
        native_to_scene = [
            JOINT_NAMES.index(name) for name in scene.articulation_joint_names
        ]
        differences = {
            name: float(
                np.max(
                    np.abs(
                        getattr(native, name).cpu().numpy()[native_to_scene]
                        - getattr(scene, "_" + name)
                    )
                )
            )
            for name in ("kp", "kd")
        }
        if max(differences.values()) > 5e-4:
            raise ValueError(
                "Common PD gains differ from the native G1 controller"
            )
        pose = initial_pair(motion, scene.articulation_joint_names)
        scene.write_pair_state(pose)
        scene.drain_reference_frame_qpos()
        scene._observation_fk = CurrentObservationFK(scene)
        pair = None
        if compiled:
            pair = FivePointReferencePair(
                [Path(candidate)] * 2, scene, controllers, device=str(device)
            )
            pair.match = match
            upper = pair.sessions[candidate_side]
        model_ids = [sha256(candidate), opponent_sha256]
        if candidate_side:
            model_ids.reverse()
        metadata = {
            "evaluation_scope": "training_native_opponent_match",
            "competition_qualification": False,
            "native_opponent_submission_interface_exempted": True,
            "native_opponent_checkpoint_sha256": opponent_sha256,
            "native_opponent_network": "official_stage3_ego_prior_decoder",
            "native_opponent_input": "full_state_192_900_90",
            "candidate_input": "competition_causal_observation1097",
            "candidate_output": "target_pose225_frozen_HoloMotion_V2",
            "candidate_side": candidate_side,
            "candidate_runtime_backend": "published_compiled"
            if compiled
            else "source_torch",
            "compiled_observation_manifest_sha256": pair.metadata[
                "compiled_observation_manifest_sha256"
            ]
            if pair
            else None,
            "resource_manifest_sha256": resources.manifest_sha256,
            "initial_pose": pose.tolist(),
            "initial_pose_mode": "official",
            "lower_model_sha256": sha256(controller.path),
            "common_pd_gain_native_max_abs_difference": differences,
            "controller_gain_tolerance": 5e-4,
            "native_observation_noise": False,
            "physics": "competition_XML_mujoco_warp_ordered_200Hz",
            "control_hz": 50,
            "rules_sha256": sha256(
                Path(__file__).parents[1] / "evaluation/runtime/rules.py"
            ),
            "scoring_version": SCORING_VERSION,
            "scoring_config": scoring_config(),
        }
        logger = MatchLogger(
            model_id=model_ids,
            match_id=output.name,
            rules=rules,
            metadata=metadata,
        )
        history, frame, round_reset = CompetitionHistory(1), 0, True
        sanitation_total = 0
        native_side = 1 - candidate_side
        limit = max_frames or int(
            rules.num_rounds * rules.round_duration_s * 50
        )
        while not match.match_over and frame < limit:
            scene._observation_fk.refresh()
            data = scene._observation_fk.data
            before = scene.read_kinematics()
            if pair is not None:
                pair.compare = controller.compare = frame < 5
                target_pose = pair.window(before, fighters=(candidate_side,))[
                    0
                ]
                sanitation = pair.frame_data["competitor_sanitization"][0]
                target = target_pose
            else:
                obs = legal_observation(
                    history, scene, data, match, frame, round_reset
                )
                target_pose, sanitation = sanitize_target_pose(
                    run_competitor(
                        upper, obs[candidate_side : candidate_side + 1]
                    )
                )
                root = scene._root_qpos[candidate_side]
                target = world_targets(
                    target_pose,
                    data.qpos[root : root + 3],
                    data.qpos[root + 3 : root + 7],
                )
            sanitation_total += sanitation["nonfinite_components"]
            candidate_action, candidate_target, start, end = controller.act(
                reference_qpos=target,
                kinematics=before,
                fighter=candidate_side,
            )
            native_start = time.perf_counter_ns()
            native_action, native_target = native.act(
                *read_native_state(scene, data, device), native_side
            )
            torch.cuda.synchronize(device)
            native_end = time.perf_counter_ns()
            actions, targets = (
                np.zeros((2, 29), np.float32),
                np.zeros((2, 29), np.float32),
            )
            actions[candidate_side], targets[candidate_side] = (
                candidate_action,
                candidate_target,
            )
            actions[native_side] = native_action[0].cpu().numpy()
            targets[native_side] = (
                native_target[0, native_to_scene].cpu().numpy()
            )
            scene.stage_pair_joint_targets(targets[None])
            # Copy the native host targets and unchanged candidate targets
            # together before advancing the shared physics.
            scene._targets_on_gpu = False
            scene.step_reference_frame()
            scene.drain_reference_frame_qpos()
            frame += 1
            after = scene.read_kinematics()
            result = match.observe_frame(
                sim_time_s=frame / 50,
                contacts=scene.contact_samples(
                    sim_time_s=np.asarray([frame / 50])
                )[0],
                root_pos=after.root_position[0],
                fallen=fallen_from_scene(scene),
            )
            starts, ends = [0, 0], [0, 0]
            starts[candidate_side], ends[candidate_side] = start, end
            starts[native_side], ends[native_side] = native_start, native_end
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
            logger.frames[-1]["candidate_sanitization"] = sanitation
            round_reset = result.round_ended and not match.match_over
            if round_reset:
                scene.write_pair_state(pose)
                scene.drain_reference_frame_qpos()
                for policy in controllers:
                    policy.reset()
                if pair is not None:
                    pair.reset()
                native.reset()
            if frame == 1 or frame % 500 == 0 or result.round_ended:
                print(
                    json.dumps(
                        {
                            "frame": frame,
                            "score": [int(v) for v in result.score],
                            "match_over": result.match_over,
                        }
                    ),
                    flush=True,
                )
        verify_cuda_execution(upper)
        logger.extra_metadata.update(
            match_completed=match.match_over,
            frames_recorded=frame,
            candidate_sanitization_total=sanitation_total,
        )
        if pair is not None:
            from holoboxing.evaluation.runtime.lean_runtime import METRICS

            logger.extra_metadata.update(
                compiled_parity_steps=min(frame, 5),
                compiled_parity_errors={
                    name: METRICS.get(name)
                    for name in (
                        "max_observation_error",
                        "max_binding_action_error",
                    )
                },
            )
        logger.write(output / "match.json", match.score)
        report = {
            "status": "COMPLETE" if match.match_over else "INCOMPLETE",
            "candidate_side": candidate_side,
            "initial_pose_mode": "official",
            "candidate_runtime_backend": "published_compiled"
            if compiled
            else "source_torch",
            "candidate_sha256": sha256(candidate),
            "native_opponent_sha256": opponent_sha256,
            "frames": frame,
            "score": [int(v) for v in match.score],
            "candidate_won": result.winner == candidate_side
            if match.match_over
            else None,
            "winner": result.winner if match.match_over else None,
            "competition_qualification": False,
        }
        (output / "result.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report
    finally:
        scene.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate",
        "opponent-checkpoint",
        "assets",
        "lower",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--opponent-sha256", required=True)
    parser.add_argument(
        "--candidate-side", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="nonzero produces an explicitly incomplete diagnostic",
    )
    print(json.dumps(evaluate(**vars(parser.parse_args(argv)))), flush=True)


if __name__ == "__main__":
    main()
