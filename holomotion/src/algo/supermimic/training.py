"""Own SuperMimic collection, update scheduling, evaluation and checkpoints.

The command-line entrypoint delegates here; algorithm code never imports a
training script. The reference recipe and its update clock remain unchanged.
"""

import json
import math
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

from holomotion.src.utils.artifacts import atomic_json, fingerprint


def run(args, *, launched=None):
    from importlib.metadata import version

    import mujoco
    import torch
    import warp as wp

    from holomotion.src.algo.supermimic.backends import evaluator_class
    from holomotion.src.algo.supermimic.buffer import ReplayBuffer
    from holomotion.src.algo.supermimic.cache_pressure import (
        PhysicsCachePressure,
    )
    from holomotion.src.algo.supermimic.checkpoint import (
        capture_learning_state,
        restore_learning_state,
        validate_checkpoint,
    )
    from holomotion.src.algo.supermimic.diagnostics import dynamics_diagnostics
    from holomotion.src.algo.supermimic.schedule import (
        ScheduleConfig,
        UpdateSchedule,
    )

    pressure = PhysicsCachePressure(
        torch, args.device, args.release_phase_cache
    )
    evaluator_type = evaluator_class(args)
    from holomotion.src.algo.supermimic import networks
    from holomotion.src.algo.supermimic.learner import ReferenceLearner
    from holomotion.src.algo.supermimic.monitoring import (
        EpisodeMonitor,
        TrainingRewardArea,
    )
    from holomotion.src.env.reference_tracking.mdp import (
        MDP,
        normalize_xi,
        unflatten_state,
    )

    repository = Path(__file__).resolve().parents[4]
    source_paths = [
        repository / "holomotion/scripts/training/train_supermimic.py",
        Path(__file__),
        *sorted((repository / "holomotion/src/algo/supermimic").glob("*")),
        *sorted(
            (repository / "holomotion/src/env/reference_tracking").glob("*")
        ),
        *sorted(
            (repository / "holomotion/src/evaluation/reference_tracking").glob(
                "*"
            )
        ),
        repository
        / "holomotion/scripts/training/prepare_supermimic_lafan1.py",
        repository / "holomotion/src/utils/artifacts.py",
        repository / "holomotion/src/training/h5_dataloader.py",
        repository / "holomotion/src/training/reference_fk.py",
    ]
    source_paths = [path for path in source_paths if path.is_file()]
    source_hashes = {
        str(path.relative_to(repository)): fingerprint(path)
        for path in source_paths
    }
    for path in source_paths:
        target = args.output / "source_snapshot" / path.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    launched = time.monotonic() if launched is None else launched
    print("building reference model", flush=True)
    from holomotion.src.algo.supermimic.backends import backend_class
    from holomotion.src.env.reference_tracking.task import load_task

    model, motion, mdp = load_task(
        robot_urdf=args.robot_urdf,
        manifest=args.manifest,
        output=args.output,
        device=args.device,
        profile=args.profile,
    )
    evaluation_mdp = MDP(
        mdp.kin,
        motion,
        mdp.default_q,
        reference_orientation=args.profile == "vibed",
    )
    print(
        f"loaded {len(motion.names)} complete motions / {motion.num_frames} frames",
        flush=True,
    )
    metadata = json.loads(args.manifest.read_text())
    shards = [
        (args.manifest.parent / value["file"]).resolve()
        for value in metadata["hdf5_shards"]
    ]
    robot_meshes = sorted(
        {
            (args.robot_urdf.parent / mesh.get("filename")).resolve()
            for mesh in ET.parse(args.robot_urdf).getroot().findall(".//mesh")
        }
    )
    specification = dict(
        scope="SuperMimic compact-policy motion tracking",
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        source_sha256=source_hashes,
        input_sha256={
            str(path): fingerprint(path)
            for path in (
                args.manifest,
                args.robot_urdf,
                *shards,
                *robot_meshes,
            )
        },
        generated_model_sha256=fingerprint(args.output / "g1.xml"),
        runtime=dict(
            python=sys.version,
            executable=sys.executable,
            torch=str(torch.__version__),
            mujoco=mujoco.__version__,
            warp=wp.__version__,
            mujoco_warp=version("mujoco-warp")
            if args.physics_backend == "mujoco_warp"
            else None,
            torch_cuda=torch.version.cuda,
            cuda_allocator_backend=torch.cuda.memory.get_allocator_backend(),
            phase_cache_release_enabled=args.release_phase_cache,
            compiler_cache_environment={
                name: os.environ.get(name)
                for name in ("TORCHINDUCTOR_CACHE_DIR", "WARP_CACHE_PATH")
            },
            gpu=torch.cuda.get_device_name(args.device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_uuid=str(
                getattr(
                    torch.cuda.get_device_properties(args.device),
                    "uuid",
                    "unknown",
                )
            ),
        ),
        physics=dict(
            timestep=float(model.opt.timestep),
            control_substeps=4,
            solver_iterations=int(model.opt.iterations),
            solver_line_search_iterations=int(model.opt.ls_iterations),
            contact_buffer_per_world=128,
            constraint_buffer_per_world=768,
            self_collisions=(
                args.physics_backend == "isaac_physx"
                or args.profile == "paper"
            ),
        ),
        data=dict(
            clips=len(motion.names),
            names=motion.names,
            frames=motion.num_frames,
            duration_hours=float((motion.clip_len - 1).sum() / 50 / 3600),
        ),
        actor_observation_dim=mdp.obs_dim,
        goal_dim=73 * args.goal_frames,
        attribution=dict(
            reference="https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9",
            cross_check="https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/d9be385b3eeb13a638fb33fb200d3bfd9be9839f",
            paper="https://arxiv.org/pdf/2610.28480",
        ),
        declared_choices=dict(
            goal_encoding="Pinned Vibed 73 reference-only values per future frame; paper does not fully specify this encoding",
            termination="Analytic smooth height/tilt survival following Vibed; hard physical labels are retained",
            friction="Sampled at reset; actual per-world values enter the model",
            com="Actual torso COM offset sampled once per world",
            bin_updates="Mean of simultaneous returns per bin before its EMA update",
            lr_clock="Completed world optimizer events, following the pinned Vibed iteration convention",
            hardware="Measured local GPU; paper H100 wall-clock is a comparison target, not equivalent compute",
        ),
        reference_target=dict(
            dataset="LAFAN1 40 full clips, about 2.45 hours",
            normalized_return=0.82,
            reported_gpu="H100",
            reported_hours=1.2,
        ),
        promotion_accepted=False,
    )
    atomic_json(args.output / "specification.json", specification)
    env = backend_class(args)(
        model,
        motion,
        mdp,
        args.num_envs,
        device=args.device,
        gamma=args.gamma,
        capture_graph=args.capture_graph,
    )
    if args.preflight_only:
        transitions = []
        for _ in range(args.warmup_steps):
            transitions.append(
                env.step(torch.zeros(args.num_envs, 29, device=args.device))
            )
        torch.save(
            dict(
                transitions=transitions,
                context=env.xi,
                friction=env._friction,
                body_ipos=env._com,
            ),
            args.output / "transitions.pt",
        )
        return dict(
            scope="Physical execution preflight only; no policy learning or quality claim",
            collected_transitions=args.warmup_steps * args.num_envs,
            physics_status=env.status_summary(),
            elapsed_s=time.monotonic() - launched,
            specification=specification,
        )
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, args.device)
    learner = ReferenceLearner(args, motion, mdp, buffer)
    schedule = UpdateSchedule(
        ScheduleConfig(
            physical_envs=args.num_envs,
            actor_batch_size=args.pi_batch,
            imagination_horizon=args.horizon,
            world_batch_size=args.wm_batch,
            world_window_length=args.wm_window,
            world_batches_per_update=args.wm_batches,
            actor_updates_per_vector_step=args.utd,
            world_updates_per_vector_step=args.utd,
            warmup_vector_steps=args.warmup_steps,
            world_only_updates=args.world_only_updates,
        )
    )
    specification["schedule"] = asdict(schedule.config)
    specification["steady_sample_ratios"] = schedule.ratios()
    specification["parameters"] = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in (
            ("world", learner.wm),
            ("actor", learner.pi),
            ("critic", learner.critic),
        )
    }
    atomic_json(args.output / "specification.json", specification)
    resumed = None
    if args.resume is not None:
        resumed = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        validate_checkpoint(resumed, specification)
        specification["resume"] = dict(
            checkpoint_sha256=fingerprint(args.resume),
            exact_training_resume=False,
            semantics="Learning state restored; physical episodes and histories reset",
            parent_vector_steps=resumed["schedule"]["vector_steps"],
        )
        atomic_json(args.output / "specification.json", specification)
    previous_training_wall = (
        0.0 if resumed is None else resumed["training_wall_s"]
    )
    evaluations = []
    evaluation_wall = 0.0
    evaluator = None
    if not args.skip_evaluation and args.physics_backend != "isaac_physx":
        # Preparation samples and kernel setup for an independent evaluator
        # must not consume the training sampler's random sequence.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            evaluator = evaluator_type(
                model,
                motion,
                evaluation_mdp,
                device=args.device,
                history_length=args.hist,
                future_frames=args.goal_frames,
                capture_graph=args.capture_graph,
            )

    def training_wall():
        return (
            previous_training_wall
            + time.monotonic()
            - launched
            - evaluation_wall
        )

    def evaluate(label):
        nonlocal evaluation_wall
        before = time.monotonic()
        training_before = training_wall()
        progress = lambda value: print(
            json.dumps(dict(phase="evaluation", label=label, **value)),
            flush=True,
        )
        result = evaluator.evaluate(learner.pi, progress=progress)
        trace_path = args.output / f"rollout_{label}.pt"
        torch.save(evaluator.last_trace, trace_path)
        result.update(
            rollout_file=trace_path.name,
            rollout_sha256=fingerprint(trace_path),
        )
        result.update(
            label=label,
            training_wall_s=training_before,
            world_updates=schedule.world_updates,
            actor_updates=schedule.actor_updates,
            vector_steps=schedule.vector_steps,
        )
        evaluations.append(result)
        evaluation_wall += time.monotonic() - before
        atomic_json(args.output / f"evaluation_{label}.json", result)
        print(
            json.dumps(
                dict(
                    phase="evaluation_complete",
                    label=label,
                    normalized_return=result["normalized_return"],
                    failed_clips=result["failed_clips"],
                    evaluated_clips=result["evaluated_clips"],
                )
            ),
            flush=True,
        )

    def save(label):
        diagnostic = dynamics_diagnostics(
            learner, seed=args.seed + 997, horizon=args.wm_window
        )
        diagnostic.update(
            vector_steps=schedule.vector_steps,
            world_updates=schedule.world_updates,
        )
        atomic_json(args.output / f"world_diagnostic_{label}.json", diagnostic)
        bundle = capture_learning_state(learner, buffer, schedule, env)
        bundle.update(
            scope="Learning state with replay; physical episodes reset on resume",
            specification=specification,
            training_wall_s=training_wall(),
            monitor=vars(episode_monitor).copy(),
            training_reward_points=reward_area.points,
        )
        path = args.output / f"checkpoint_{label}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(bundle, temporary)
        temporary.replace(path)
        atomic_json(
            args.output / "latest_checkpoint.json",
            dict(
                path=path.name,
                sha256=fingerprint(path),
                vector_steps=schedule.vector_steps,
                actor_updates=schedule.actor_updates,
                training_wall_s=training_wall(),
            ),
        )

    episode_monitor = EpisodeMonitor()
    reward_area = TrainingRewardArea()
    if resumed is not None:
        restore_learning_state(resumed, learner, buffer, schedule, env)
        vars(episode_monitor).update(resumed["monitor"])
        for point in resumed["training_reward_points"]:
            reward_area.observe(*point)
        del resumed
    policy_progress = None
    if args.policy_snapshot_minutes > 0:
        from holomotion.src.algo.supermimic.policy_progress import (
            PolicyProgressSnapshots,
        )

        policy_progress = PolicyProgressSnapshots(
            args.output,
            learner.pi,
            specification,
            schedule,
            training_wall,
            args.policy_snapshot_minutes,
            args.max_hours * 3600,
        )
        policy_progress.initial()
    if evaluator is not None:
        if args.physics_backend != "isaac_physx":
            evaluate("resumed" if args.resume is not None else "initial")
    save("resumed" if args.resume is not None else "initial")
    history = torch.zeros(
        args.num_envs, args.hist, mdp.obs_dim, device=args.device
    )
    metrics = []
    next_checkpoint = (
        (math.floor(training_wall() / (args.checkpoint_minutes * 60)) + 1)
        * args.checkpoint_minutes
        * 60
    )
    next_evaluation = (
        (math.floor(training_wall() / (args.evaluate_minutes * 60)) + 1)
        * args.evaluate_minutes
        * 60
    )
    recent = []
    while (
        schedule.vector_steps < args.max_vector_steps
        and schedule.world_updates < args.max_updates
        and training_wall() < args.max_hours * 3600
    ):
        collection_start = time.monotonic()
        with torch.no_grad():
            observation = mdp.add_noise(
                mdp.obs(env.state(), env.a_prev, env.xi, env.f)
            )
            fresh = env.ep_step == 0
            history = torch.where(
                fresh[:, None, None],
                observation[:, None].expand_as(history),
                torch.cat((history[:, 1:], observation[:, None]), 1),
            )
            goal = motion.goal(env.f, args.goal_frames)
            learner.pi.obs_norm.update(observation)
            learner.pi.goal_norm.update(goal)
            action = learner.pi(history, goal)
            executed = (
                action + args.action_noise * torch.randn_like(action)
            ).clamp(-20, 20)
            pressure.before_launch()
            transition = env.step(executed)
            episode_monitor.add(transition)
            buffer.add(transition)
            state, next_state = (
                unflatten_state(transition["s"]),
                unflatten_state(transition["s1"]),
            )
            learner.wm.in_norm.update(
                learner.wm.inputs(
                    state, transition["a"], normalize_xi(transition["xi"])
                )
            )
            learner.wm.out_norm.update(learner.wm.targets(state, next_state))
            mean_reward = float(transition["r"].mean())
        reward_area.observe(training_wall(), mean_reward)
        world_events, actor_events = schedule.advance()
        rate = args.lr * (
            0.03
            + 0.97
            * 0.5
            * (
                1
                + math.cos(
                    math.pi
                    * min(schedule.world_updates / args.lr_decay_updates, 1.0)
                )
            )
        )
        learner.set_learning_rate(rate)
        event = dict(
            vector_step=schedule.vector_steps,
            real_mean_step_reward=mean_reward,
            learning_rate=rate,
            collection_seconds=time.monotonic() - collection_start,
        )
        for _ in range(world_events):
            update_start = time.monotonic()
            values = [learner.wm_update() for _ in range(args.wm_batches)]
            event["world_mean_window_loss"] = float(
                torch.stack([value[0] for value in values]).mean()
            )
            event["world_last_step_delta_normalized_mse"] = float(
                torch.stack([value[1] for value in values]).mean()
            )
            event["world_gradient_norm"] = float(
                learner.last_world_gradient_norm
            )
            event["world_update_seconds"] = time.monotonic() - update_start
            schedule.complete_world()
            if schedule.world_updates == 1:
                print(
                    json.dumps(
                        dict(
                            phase="first_world_update_complete",
                            vector_step=schedule.vector_steps,
                            torch_allocated_bytes=torch.cuda.memory_allocated(
                                args.device
                            ),
                            torch_reserved_bytes=torch.cuda.memory_reserved(
                                args.device
                            ),
                        )
                    ),
                    flush=True,
                )
        for _ in range(actor_events):
            update_start = time.monotonic()
            loss, value_loss, imagined, gradient, survival = (
                learner.policy_update()
            )
            event.update(
                actor_loss=float(loss),
                critic_loss=float(value_loss),
                imagined_mean_normalized_return=float(imagined),
                actor_gradient_norm=float(gradient),
                imagined_survival=float(survival),
            )
            event["actor_update_seconds"] = time.monotonic() - update_start
            schedule.complete_actor()
            if schedule.actor_updates == 1:
                print(
                    json.dumps(
                        dict(
                            phase="first_actor_update_complete",
                            vector_step=schedule.vector_steps,
                            actor_loss=float(loss),
                            actor_gradient_norm=float(gradient),
                            torch_allocated_bytes=torch.cuda.memory_allocated(
                                args.device
                            ),
                            torch_reserved_bytes=torch.cuda.memory_reserved(
                                args.device
                            ),
                        )
                    ),
                    flush=True,
                )
        recent.append(event)
        if schedule.vector_steps % args.log_steps == 0:
            summary = dict(
                phase="training",
                vector_steps=schedule.vector_steps,
                world_updates=schedule.world_updates,
                actor_updates=schedule.actor_updates,
                training_wall_s=training_wall(),
                sample_counts=schedule.ratios(),
                mean={
                    key: sum(value[key] for value in recent if key in value)
                    / sum(key in value for value in recent)
                    for key in {key for value in recent for key in value}
                    if key != "vector_step"
                },
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(
                    args.device
                ),
            )
            summary.update(
                torch_allocated_bytes=torch.cuda.memory_allocated(args.device),
                torch_reserved_bytes=torch.cuda.memory_reserved(args.device),
                warp_mempool_used_bytes=wp.get_mempool_used_mem_current(
                    args.device
                ),
            )
            summary["cache_pressure"] = pressure.summary()
            summary["physics_status"] = env.status_summary(
                include_world_bits=False
            )
            summary["training_monitor"] = episode_monitor.summary(
                vector_steps=schedule.vector_steps,
                training_seconds=training_wall(),
            )
            summary["training_reward_area"] = reward_area.summary(
                summary["training_wall_s"]
            )
            metrics.append(summary)
            recent = []
            atomic_json(args.output / "progress.json", summary)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
        if policy_progress is not None:
            policy_progress.due()
        elapsed = training_wall()
        checkpoint_due = elapsed >= next_checkpoint
        evaluation_due = (
            args.physics_backend != "isaac_physx"
            and evaluator is not None
            and elapsed >= next_evaluation
        )
        if checkpoint_due or evaluation_due:
            save(f"step_{schedule.vector_steps:07d}")
        if checkpoint_due:
            next_checkpoint += args.checkpoint_minutes * 60
        if evaluation_due:
            evaluate(f"step_{schedule.vector_steps:07d}")
            next_evaluation += args.evaluate_minutes * 60
    if policy_progress is not None:
        policy_progress.final()
    training_budget_end_wall = training_wall()
    training_reward_area = reward_area.summary(training_budget_end_wall)
    atomic_json(
        args.output / "training_reward_area.json", training_reward_area
    )
    atomic_json(
        args.output / "training_reward_curve.json",
        dict(
            columns=["training_wall_s", "real_mean_step_reward"],
            points=reward_area.points,
            summary=training_reward_area,
        ),
    )
    save("final")
    if evaluator is not None and (
        not evaluations
        or evaluations[-1]["vector_steps"] != schedule.vector_steps
    ):
        evaluate("final")
    if evaluations:
        atomic_json(
            args.output / "evaluation_final.json",
            dict(evaluations[-1], final_policy_evaluation=True),
        )
    changed = [
        name
        for name, digest in source_hashes.items()
        if fingerprint(repository / name) != digest
    ]
    if changed:
        raise RuntimeError(f"Source changed during the experiment: {changed}")
    reported_training_wall = training_wall()
    return dict(
        scope="Bounded reference-method training; reproduction quality requires the complete-clip evaluations",
        specification=specification,
        clock=schedule.state_dict(),
        sample_counts=schedule.ratios(),
        training_wall_s=reported_training_wall,
        training_budget_end_wall_s=training_budget_end_wall,
        training_reward_area=training_reward_area,
        post_training_finalization_wall_s=(
            reported_training_wall - training_budget_end_wall
        ),
        evaluation_wall_s=evaluation_wall,
        resumed_from=specification.get("resume"),
        physical_episode_restart=bool(args.resume is not None),
        process_wall_s=time.monotonic() - launched,
        cache_pressure=pressure.summary(),
        evaluations=evaluations,
        metrics=metrics,
        promotion_accepted=False,
        paper_reference_target_observed=bool(
            evaluations
            and evaluations[-1]["evaluated_clips"] == 40
            and evaluations[-1]["normalized_return"] >= 0.82
        ),
        physics_status=env.status_summary(),
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(
            args.device
        ),
    )


def execute(args):
    if args.physics_backend == "isaac_physx":
        from holomotion.src.algo.supermimic.runtime import execute_isaac

        return execute_isaac(args)
    return execute_local(args)


def execute_local(args, *, completion=None, operation=run):
    launched = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "run_status.json",
        dict(completed=False, phase="starting"),
    )

    notified = False

    def notify(succeeded):
        nonlocal notified
        if completion is not None and not notified:
            completion.send(dict(succeeded=succeeded))
            notified = True

    def record_failure(error):
        import traceback

        if hasattr(error, "physics_diagnostic"):
            import torch

            torch.save(
                error.physics_diagnostic, args.output / "physics_failure.pt"
            )
        (args.output / "error.txt").write_text(traceback.format_exc())
        atomic_json(
            args.output / "run_status.json",
            dict(completed=False, phase="failed"),
        )

    try:
        from holomotion.src.algo.supermimic.runtime import physics_runtime

        with physics_runtime(args):
            try:
                report = operation(args, launched=launched)
                atomic_json(args.output / "report.json", report)
                atomic_json(
                    args.output / "run_status.json",
                    dict(completed=True, phase="finished"),
                )
            except BaseException as error:
                # Persist failure before Kit shutdown can terminate its process.
                record_failure(error)
                notify(False)
                raise
            notify(True)
        return report
    except BaseException as error:
        record_failure(error)
        notify(False)
        raise
