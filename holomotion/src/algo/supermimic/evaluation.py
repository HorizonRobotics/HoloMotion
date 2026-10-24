"""Evaluate a saved compact policy using the independent complete-clip scorer."""

from holomotion.src.evaluation.reference_tracking.isaac_evaluation import (
    IsaacFullClipEvaluator,
)


def evaluate_checkpoint(options, *, launched=None):
    """Score the final policy in a fresh Isaac application, without learning."""
    import json
    import time
    from pathlib import Path

    import torch

    from holomotion.src.algo.supermimic import networks
    from holomotion.src.env.reference_tracking.task import load_task
    from holomotion.src.utils.artifacts import fingerprint

    torch.manual_seed(options.seed)
    torch.set_num_threads(8)
    torch.cuda.set_device(options.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16 if options.amp == "bf16" else None
    checkpoint_path = options.output.parent / "checkpoint_final.pt"
    bundle = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    repository = Path(__file__).resolve().parents[4]
    checkpoint_digest = fingerprint(checkpoint_path)

    def verify_contract():
        specification = bundle["specification"]
        paths = (
            [
                (repository / name, digest)
                for name, digest in specification["source_sha256"].items()
            ]
            + [
                (Path(name), digest)
                for name, digest in specification["input_sha256"].items()
            ]
            + [(checkpoint_path, checkpoint_digest)]
        )
        if any(
            not path.is_file() or fingerprint(path) != digest
            for path, digest in paths
        ):
            raise ValueError("Evaluation source, input or checkpoint changed")

    verify_contract()
    model, motion, mdp = load_task(
        robot_urdf=options.robot_urdf,
        manifest=options.manifest,
        output=options.output,
        device=options.device,
        profile=options.profile,
    )
    policy = networks.Policy(
        mdp.obs_dim, 73 * options.goal_frames, options.hist
    ).to(options.device)
    policy.load_state_dict(bundle["pi"], strict=True)
    evaluator = IsaacFullClipEvaluator(
        model,
        motion,
        mdp,
        device=options.device,
        history_length=options.hist,
        future_frames=options.goal_frames,
        robot_urdf=options.robot_urdf,
        cache_dir=options.output / "isaac_usd",
    )
    print(
        json.dumps(
            dict(
                phase="evaluation_started",
                label="final",
                evaluated_clips=len(motion.names),
                planned_steps=int((motion.clip_len - 1).sum()),
            )
        ),
        flush=True,
    )

    def progress(event):
        print(
            json.dumps(dict(phase="evaluation", label="final", **event)),
            flush=True,
        )

    result = evaluator.evaluate(policy, progress=progress)
    verify_contract()
    path = options.output / "rollout_final.pt"
    torch.save(evaluator.last_trace, path)
    result.update(
        label="final",
        rollout_file=path.name,
        rollout_sha256=fingerprint(path),
        checkpoint_sha256=fingerprint(checkpoint_path),
        training_wall_s=bundle["training_wall_s"],
        vector_steps=bundle["schedule"]["vector_steps"],
        world_updates=bundle["schedule"]["world_updates"],
        actor_updates=bundle["schedule"]["actor_updates"],
        evaluation_process_wall_s=time.monotonic() - launched,
    )
    print(
        json.dumps(
            dict(
                phase="evaluation_complete",
                normalized_return=result["normalized_return"],
                evaluated_clips=result["evaluated_clips"],
            )
        ),
        flush=True,
    )
    return result
