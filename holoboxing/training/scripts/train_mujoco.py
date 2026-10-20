"""Launch the existing PPO algorithm on competition XML physics."""

from holoboxing.data.contract import CompetitionAssets
from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
from holoboxing.training.resources import (
    resolve_resources,
    load_neutral_reference,
)
from holoboxing.training.rsl_rl.modules.amp import load_amp_motions
from holoboxing.training.rsl_rl.runners.on_policy_runner import (
    OnPolicyRunner,
    seed_everything,
)
from holoboxing.training.scripts.common import prepare, write_json


def train(args, config):
    seed_everything(config.seed)
    identity = prepare(args)
    assets = CompetitionAssets.load(args.assets)
    _, paths, _ = resolve_resources(args.assets, args.lower)
    write_json(args.output / "config.json", config.to_dict())
    env = MujocoSandbagEnv(
        assets,
        paths,
        config.num_envs,
        args.device,
        args.output / ".runtime",
        config.seed,
    )
    try:
        import torch

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
        reference = load_neutral_reference(
            paths["lower_onnx"], paths["arena_scene"]
        )
        runner = OnPolicyRunner(
            env, demo, reference, config, args.output, identity
        )
        if args.resume:
            runner.resume(
                args.resume, allow_code_change=args.allow_resume_code_change
            )
            write_json(args.output / "resume.json", runner.resume_source)
        else:
            runner.initialize_from_supervised(args.supervised_checkpoint)
        runner.policy.enable_actor_cuda_graphs()
        runner.learn(
            args.updates, save_interval=1024, extra_save_interval=5000
        )
        write_json(
            args.output / "complete.json",
            {
                "updates_completed": runner.iteration,
                "transitions_per_role": runner.transitions_per_role,
                "boxing_quality_verified": False,
            },
        )
    finally:
        env.close()
