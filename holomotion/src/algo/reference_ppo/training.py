"""Train PPO on either shared reference-tracking simulator backend."""

import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from holomotion.src.utils.artifacts import atomic_json, fingerprint


def run(args, *, launched=None):
    from importlib.metadata import version

    import mujoco
    import torch
    import warp

    from holomotion.src.algo.reference_ppo.learner import SimulatorPPOLearner
    from holomotion.src.algo.supermimic import networks
    from holomotion.src.algo.supermimic.backends import (
        backend_class,
        evaluator_class,
    )
    from holomotion.src.algo.supermimic.monitoring import (
        EpisodeMonitor,
        TrainingRewardArea,
    )
    from holomotion.src.env.reference_tracking.mdp import unflatten_state
    from holomotion.src.env.reference_tracking.task import load_task

    repository = Path(__file__).resolve().parents[4]
    paths = [
        Path(__file__),
        repository / "holomotion/scripts/training/train_reference_ppo.py",
        repository / "holomotion/src/utils/artifacts.py",
        repository / "holomotion/src/training/h5_dataloader.py",
        repository / "holomotion/src/training/reference_fk.py",
    ]
    for directory in (
        "algo/reference_ppo",
        "algo/supermimic",
        "env/reference_tracking",
        "evaluation/reference_tracking",
    ):
        paths.extend(
            sorted((repository / "holomotion/src" / directory).glob("*.py"))
        )
    hashes = {str(p.relative_to(repository)): fingerprint(p) for p in paths}
    for path in paths:
        destination = (
            args.output / "source_snapshot" / path.relative_to(repository)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    launched = time.monotonic() if launched is None else launched
    model, motion, mdp = load_task(
        robot_urdf=args.robot_urdf,
        manifest=args.manifest,
        output=args.output,
        device=args.device,
        profile=args.profile,
    )
    env = backend_class(args)(
        model,
        motion,
        mdp,
        args.num_envs,
        device=args.device,
        gamma=0.99,
        capture_graph=args.capture_graph,
    )
    # ReferenceLearner constructs this network before its actor and critic.
    # Consume the same CPU initialization stream, then discard the unused model.
    unused_initialization = networks.WorldModel()
    del unused_initialization
    learner = SimulatorPPOLearner(
        mdp,
        motion,
        device=args.device,
        hist=args.hist,
        goal_frames=args.goal_frames,
        learning_rate=args.lr,
        epochs=args.epochs,
        minibatches=args.minibatches,
        microbatch=args.microbatch,
    )
    if args.compile:
        learner.pi.forward = torch.compile(learner.pi.forward)
        learner.critic.forward = torch.compile(learner.critic.forward)
    metadata = json.loads(args.manifest.read_text())
    shards = [
        (args.manifest.parent / item["file"]).resolve()
        for item in metadata["hdf5_shards"]
    ]
    specification = dict(
        scope="PPO-Sim control on the shared reproduction task",
        arguments={
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
        },
        source_sha256=hashes,
        input_sha256={
            str(p): fingerprint(p)
            for p in (
                args.manifest,
                args.robot_urdf,
                *shards,
                *sorted(
                    {
                        (
                            args.robot_urdf.parent / mesh.get("filename")
                        ).resolve()
                        for mesh in ET.parse(args.robot_urdf)
                        .getroot()
                        .findall(".//mesh")
                    }
                ),
            )
        },
        generated_model_sha256=fingerprint(args.output / "g1.xml"),
        runtime=dict(
            python=sys.version,
            executable=sys.executable,
            torch=str(torch.__version__),
            cuda_allocator_backend=torch.cuda.memory.get_allocator_backend(),
            phase_cache_release_enabled=args.release_phase_cache,
            compiler_cache_environment={
                name: os.environ.get(name)
                for name in ("TORCHINDUCTOR_CACHE_DIR", "WARP_CACHE_PATH")
            },
            mujoco=mujoco.__version__,
            warp=warp.__version__,
            mujoco_warp=version("mujoco-warp")
            if args.physics_backend == "mujoco_warp"
            else None,
            gpu=torch.cuda.get_device_name(args.device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        ),
        physics=dict(
            timestep=float(model.opt.timestep),
            control_substeps=4,
            self_collisions=args.physics_backend == "isaac_physx"
            or args.profile == "paper",
        ),
        data=dict(
            clips=len(motion.names),
            names=motion.names,
            frames=motion.num_frames,
        ),
        actor_observation_dim=mdp.obs_dim,
        goal_dim=73 * args.goal_frames,
        ppo=dict(
            rollout_steps=args.rollout_steps,
            epochs=args.epochs,
            minibatches=args.minibatches,
            clip=0.2,
            value_coefficient=1.0,
            clipped_value=True,
            entropy=0.005,
            gamma=0.99,
            lam=0.95,
            initial_std=1.0,
            adaptive_kl=0.01,
            learning_rate=args.lr,
            max_grad_norm=1.0,
            optimizer="Adam",
            microbatch=args.microbatch,
        ),
        declared_choices=dict(
            normalization="Update running moments at rollout boundaries, frozen during collection and optimization",
            initialization="Same CPU WorldModel initialization prefix as the reference learner, then identical actor and critic construction",
            timeout="Bootstrap from physical pre-reset next state; cut GAE at every episode end",
            action="Store the sampled Gaussian action and log likelihood; execute its [-20,20] clipping",
            goal="Shared pinned 730-value reference goal; exact paper encoding remains underspecified",
        ),
        promotion_accepted=False,
    )
    atomic_json(args.output / "specification.json", specification)
    evaluations, metrics = [], []
    evaluation_wall = 0.0
    vector_steps = iterations = optimizer_steps = 0
    monitor = EpisodeMonitor()
    reward_area = TrainingRewardArea()
    evaluator = None
    if not args.skip_evaluation and args.physics_backend != "isaac_physx":
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            evaluator = evaluator_class(args)(
                model,
                motion,
                mdp,
                device=args.device,
                history_length=args.hist,
                future_frames=args.goal_frames,
                capture_graph=args.capture_graph,
            )

    def training_wall():
        return time.monotonic() - launched - evaluation_wall

    def save(label):
        bundle = dict(
            scope="PPO learning state; exact physics resume is not provided",
            exact_training_resume=False,
            pi=learner.pi.state_dict(),
            critic=learner.critic.state_dict(),
            log_std=learner.log_std.detach(),
            optimizer=learner.optimizer.state_dict(),
            specification=specification,
            training_wall_s=training_wall(),
            vector_steps=vector_steps,
            iterations=iterations,
            schedule=dict(
                vector_steps=vector_steps,
                world_updates=0,
                actor_updates=optimizer_steps,
            ),
            training_reward_points=reward_area.points,
            torch_rng=torch.random.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(args.device),
        )
        destination = args.output / f"checkpoint_{label}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(bundle, temporary)
        temporary.replace(destination)
        atomic_json(
            args.output / "latest_checkpoint.json",
            dict(
                path=destination.name,
                sha256=fingerprint(destination),
                vector_steps=vector_steps,
                actor_updates=optimizer_steps,
                training_wall_s=training_wall(),
            ),
        )

    def evaluate(label):
        nonlocal evaluation_wall
        before, training_before = time.monotonic(), training_wall()
        result = evaluator.evaluate(
            learner.pi,
            progress=lambda value: print(
                json.dumps(dict(phase="evaluation", label=label, **value)),
                flush=True,
            ),
        )
        path = args.output / f"rollout_{label}.pt"
        torch.save(evaluator.last_trace, path)
        result.update(
            rollout_file=path.name,
            rollout_sha256=fingerprint(path),
            label=label,
            training_wall_s=training_before,
            actor_updates=optimizer_steps,
            vector_steps=vector_steps,
            ppo_rollouts=iterations,
            world_updates=0,
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

    policy_progress = None
    if args.policy_snapshot_minutes > 0:
        from holomotion.src.algo.supermimic.policy_progress import (
            PolicyProgressSnapshots,
        )

        progress_schedule = SimpleNamespace(
            state_dict=lambda: dict(
                vector_steps=vector_steps,
                world_updates=0,
                actor_updates=optimizer_steps,
            )
        )
        policy_progress = PolicyProgressSnapshots(
            args.output,
            learner.pi,
            specification,
            progress_schedule,
            training_wall,
            args.policy_snapshot_minutes,
            args.max_hours * 3600,
        )
        policy_progress.initial()
    save("initial")
    if evaluator is not None:
        evaluate("initial")
    history = torch.zeros(
        args.num_envs, args.hist, mdp.obs_dim, device=args.device
    )
    prior = None
    next_checkpoint = args.checkpoint_minutes * 60
    next_evaluation = args.evaluate_minutes * 60
    while (
        iterations < args.max_rollouts
        and training_wall() < args.max_hours * 3600
    ):
        with torch.no_grad():
            if prior is None:
                observation = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                learner.pi.obs_norm.update(observation)
                learner.pi.goal_norm.update(
                    motion.goal(env.f, args.goal_frames)
                )
                learner.critic.norm.update(
                    learner.critic_input(env.state(), env.f, env.xi)
                )
            else:
                learner.pi.obs_norm.update(prior["history"][:, :, -1])
                learner.pi.goal_norm.update(prior["goal"])
                learner.critic.norm.update(prior["critic_input"])
                del prior
        collection_start = time.monotonic()
        records = []
        with torch.no_grad():
            for _ in range(args.rollout_steps):
                observation = mdp.add_noise(
                    mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                )
                history = torch.where(
                    (env.ep_step == 0)[:, None, None],
                    observation[:, None].expand_as(history),
                    torch.cat((history[:, 1:], observation[:, None]), 1),
                )
                goal = motion.goal(env.f, args.goal_frames)
                features = learner.critic_input(env.state(), env.f, env.xi)
                distribution = learner.distribution(history, goal)
                sampled = distribution.sample()
                value = learner.critic(features)
                transition = env.step(sampled.clamp(-20, 20))
                next_frame = torch.minimum(
                    transition["f"] + 1,
                    motion.clip_end[motion.clip_id[transition["f"]]],
                )
                next_value = learner.critic(
                    learner.critic_input(
                        unflatten_state(transition["s1"]),
                        next_frame,
                        transition["xi"],
                    )
                )
                records.append(
                    dict(
                        history=history,
                        goal=goal,
                        critic_input=features,
                        sampled_action=sampled,
                        log_probability=distribution.log_prob(sampled).sum(-1),
                        mean=distribution.loc,
                        std=distribution.scale.clone(),
                        value=value,
                        next_value=next_value,
                        reward=transition["r"],
                        terminated=transition["term"],
                        ended=transition["done"],
                    )
                )
                reward_area.observe(
                    training_wall(), float(transition["r"].mean())
                )
                monitor.add(transition)
                vector_steps += 1
            prior = {
                name: torch.stack([row[name] for row in records])
                for name in records[0]
            }
            del records
            mean_reward = float(prior["reward"].mean())
        collection_seconds = time.monotonic() - collection_start
        update_start = time.monotonic()
        result = learner.update(prior)
        optimizer_steps += result["optimizer_steps"]
        iterations += 1
        update_seconds = time.monotonic() - update_start
        if args.release_phase_cache:
            torch.cuda.empty_cache()
        summary = dict(
            phase="training",
            algorithm="PPO-Sim",
            vector_steps=vector_steps,
            actor_updates=optimizer_steps,
            ppo_rollouts=iterations,
            world_updates=0,
            training_wall_s=training_wall(),
            collection_seconds=collection_seconds,
            update_seconds=update_seconds,
            real_mean_step_reward=mean_reward,
            update=result,
            training_reward_area=reward_area.summary(training_wall()),
            training_monitor=monitor.summary(
                vector_steps=vector_steps, training_seconds=training_wall()
            ),
            physics_status=env.status_summary(include_world_bits=False),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(
                args.device
            ),
        )
        metrics.append(summary)
        atomic_json(args.output / "progress.json", summary)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        if policy_progress is not None:
            policy_progress.due()
        elapsed = training_wall()
        due_save, due_evaluate = (
            elapsed >= next_checkpoint,
            evaluator is not None and elapsed >= next_evaluation,
        )
        if due_save or due_evaluate:
            save(f"step_{vector_steps:07d}")
        if due_save:
            next_checkpoint += args.checkpoint_minutes * 60
        if due_evaluate:
            evaluate(f"step_{vector_steps:07d}")
            next_evaluation += args.evaluate_minutes * 60
    if policy_progress is not None:
        policy_progress.final()
    budget_end = training_wall()
    area = reward_area.summary(budget_end)
    atomic_json(args.output / "training_reward_area.json", area)
    atomic_json(
        args.output / "training_reward_curve.json",
        dict(
            columns=["training_wall_s", "real_mean_step_reward"],
            points=reward_area.points,
            summary=area,
        ),
    )
    save("final")
    if evaluator is not None and (
        not evaluations or evaluations[-1]["vector_steps"] != vector_steps
    ):
        evaluate("final")
    if evaluations:
        atomic_json(
            args.output / "evaluation_final.json",
            dict(evaluations[-1], final_policy_evaluation=True),
        )
    changed = [
        name
        for name, value in hashes.items()
        if fingerprint(repository / name) != value
    ]
    if changed:
        raise RuntimeError(f"Source changed during the experiment: {changed}")
    reported_training_wall = training_wall()
    return dict(
        scope="Bounded PPO-Sim control; efficiency requires the matching complete-clip model-based arm",
        specification=specification,
        evaluations=evaluations,
        metrics=metrics,
        vector_steps=vector_steps,
        actor_updates=optimizer_steps,
        ppo_rollouts=iterations,
        training_budget_end_wall_s=budget_end,
        training_reward_area=area,
        resumed_from=None,
        clock=dict(
            vector_steps=vector_steps,
            world_updates=0,
            actor_updates=optimizer_steps,
        ),
        post_training_finalization_wall_s=reported_training_wall - budget_end,
        training_wall_s=reported_training_wall,
        evaluation_wall_s=evaluation_wall,
        process_wall_s=time.monotonic() - launched,
        physics_status=env.status_summary(),
        promotion_accepted=False,
    )


def execute(options):
    from holomotion.src.algo.supermimic.training import execute_local

    if options.physics_backend == "isaac_physx":
        from holomotion.src.algo.supermimic.runtime import execute_isaac

        return execute_isaac(options, training_operation=run)
    return execute_local(options, operation=run)
