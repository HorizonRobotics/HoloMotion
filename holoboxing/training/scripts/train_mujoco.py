"""Launch the existing PPO algorithm on competition XML physics."""

from dataclasses import replace

from holoboxing.data.contract import CompetitionAssets
from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
from holoboxing.training.resources import (
    load_neutral_reference,
    resolve_resources,
)
from holoboxing.training.rsl_rl.modules.amp import load_amp_motions
from holoboxing.training.rsl_rl.runners.on_policy_runner import (
    OnPolicyRunner,
    seed_everything,
)
from holoboxing.training.scripts.common import prepare, write_json


def train(args, config, *, time_budget=None):
    seed_everything(config.seed)
    identity = prepare(args)
    assets = CompetitionAssets.load(args.assets)
    _, paths, _ = resolve_resources(args.assets, args.lower)
    reference = load_neutral_reference(
        paths["lower_onnx"], paths["arena_scene"]
    )
    policy, prior_records = None, None
    import torch

    from holoboxing.training.motion_prior import FORMAT as PRIOR_FORMAT
    from holoboxing.training.spherical_actor import (
        FORMAT as IMITATION_FORMAT,
        actor_from_prior,
        actor_from_state,
    )
    from holoboxing.training.spherical_rl import (
        FORMAT as LATENT_RL_FORMAT,
        SphericalActorCritic,
    )
    from holoboxing.training.supervised.training import load_supervised

    initialization = (
        args.resume
        or args.supervised_checkpoint
        or args.prior_checkpoint
        or args.policy_checkpoint
    )
    saved = torch.load(initialization, map_location="cpu", weights_only=False)
    if saved.get("format") in (
        PRIOR_FORMAT,
        IMITATION_FORMAT,
        LATENT_RL_FORMAT,
    ):
        if not args.config:
            config = replace(config, horizon_rho=0, init_noise_std=0.1)
        if saved["format"] == PRIOR_FORMAT:
            from holoboxing.training.supervised.data import Records

            prior_records = Records(args.amp_motions / "manifest.json")
            validation = Records(
                args.amp_motions / "manifest.json",
                split="validation",
                allow_empty=True,
            )
            actor = actor_from_prior(
                initialization, config, prior_records, validation
            )
        elif saved["format"] == IMITATION_FORMAT:
            actor, _, _ = load_supervised(initialization, config, reference)
        else:
            actor_state = {
                k.removeprefix("taskpolicy_ego."): v
                for k, v in saved["policy"].items()
                if k.startswith("taskpolicy_ego.")
            }
            actor = actor_from_state(actor_state, saved["policy_spec"], config)
        policy = SphericalActorCritic(actor, config)
        identity["frozen_representation"] = policy.policy_spec
    write_json(args.output / "config.json", config.to_dict())
    standing = None
    if args.training_mode == "standing":
        from holoboxing.training.standing_reference import (
            WorldStandingReference,
        )

        standing = WorldStandingReference(
            reference, config.num_envs, args.device
        )
    env = MujocoSandbagEnv(
        assets,
        paths,
        config.num_envs,
        args.device,
        args.output / ".runtime",
        config.seed,
        standing_reference=standing,
    )
    try:
        demo, motions = load_amp_motions(
            args.amp_motions,
            env.nominal,
            torch.zeros_like(env.nominal),
            device=args.device,
            resource_manifest_sha256=identity["resources"]["assets"],
        )
        identity["motions"] = motions
        write_json(
            args.output / "environment.json",
            {
                "physics": "mujoco-warp",
                "dt": 0.005,
                "control_dt": 0.02,
                "ordered_evaluation_physics": env.physics.ordered,
                "contact_parameters": "competition XML, no material randomization",
                "asset_identity": assets.identity(),
            },
        )
        runner = OnPolicyRunner(
            env,
            demo,
            reference,
            config,
            args.output,
            identity,
            policy=policy,
            training_mode=args.training_mode,
            opponent_pool_capacity=args.opponent_pool_capacity,
            opponent_refresh_updates=args.opponent_refresh_updates,
        )
        if args.resume:
            runner.resume(
                args.resume, allow_code_change=args.allow_resume_code_change
            )
            write_json(args.output / "resume.json", runner.resume_source)
        elif args.supervised_checkpoint:
            runner.initialize_from_supervised(args.supervised_checkpoint)
        elif args.policy_checkpoint:
            runner.initialize_from_policy(args.policy_checkpoint)
        else:
            runner.initialize_from_prior(args.prior_checkpoint, prior_records)
        runner.policy.enable_actor_cuda_graphs()
        result = runner.learn(
            args.updates,
            save_interval=1024,
            extra_save_interval=5000,
            time_budget=time_budget,
        )
        write_json(
            args.output / "complete.json",
            {
                "updates_completed": runner.iteration,
                "training_result": result,
                "transitions_per_role": runner.transitions_per_role,
                "boxing_quality_verified": False,
            },
        )
    finally:
        env.close()
