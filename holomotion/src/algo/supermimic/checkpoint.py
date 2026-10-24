"""Versioned learning-state recovery at an explicit physical episode boundary."""

import random

import numpy as np
import torch

# These affect execution placement, reporting or stopping, rather than updates.
RESUME_OVERRIDES = {
    "resume",
    "output",
    "device",
    "max_hours",
    "max_vector_steps",
    "max_updates",
    "checkpoint_minutes",
    "policy_snapshot_minutes",
    "evaluate_minutes",
    "log_steps",
    "skip_evaluation",
}
FORMAT = "supermimic_episode_boundary_v1"


def validate_checkpoint(bundle, specification):
    """Reject inference bundles and incompatible data, sources or recipes."""
    if bundle.get("format") != FORMAT:
        raise ValueError(
            "Checkpoint lacks complete episode-boundary learning state"
        )
    saved = bundle["specification"]
    before, after = saved["arguments"], specification["arguments"]
    keys = (before.keys() | after.keys()) - RESUME_OVERRIDES
    changed = sorted(key for key in keys if before.get(key) != after.get(key))
    if changed:
        raise ValueError(f"Resume recipe changed: {changed}")
    for field in (
        "source_sha256",
        "input_sha256",
        "generated_model_sha256",
        "schedule",
        "parameters",
        "actor_observation_dim",
        "goal_dim",
        "physics",
    ):
        if saved[field] != specification[field]:
            raise ValueError(f"Resume contract changed: {field}")
    for field in ("torch", "mujoco", "warp", "mujoco_warp", "torch_cuda"):
        if saved["runtime"][field] != specification["runtime"][field]:
            raise ValueError(f"Resume runtime changed: {field}")


def capture_learning_state(learner, buffer, schedule, env):
    """Keep replay on CPU and leave simulator internals outside this contract."""
    return dict(
        format=FORMAT,
        exact_training_resume=False,
        resume_semantics="Restore learning state; reset all physical episodes and observation histories",
        **{
            name: getattr(learner, name).state_dict()
            for name in ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v")
        },
        schedule=schedule.state_dict(),
        replay=buffer.state_dict(),
        curriculum={
            name: getattr(env, name).to("cpu", copy=True)
            for name in ("bin_G", "bin_w_avg")
        },
        torch_rng=torch.random.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(learner.dev),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
    )


def restore_learning_state(bundle, learner, buffer, schedule, env):
    """Restore RNG after initialization, then draw fresh physical resets."""
    # Validate schedule/replay before mutating model weights.
    schedule.load_state_dict(bundle["schedule"])
    buffer.load_state_dict(bundle["replay"])
    for name in ("bin_G", "bin_w_avg"):
        source, target = bundle["curriculum"][name], getattr(env, name)
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError("Curriculum shape or dtype changed")
        if not torch.isfinite(source).all():
            raise ValueError("Nonfinite curriculum")
        target.copy_(source)
    for name in ("wm", "pi", "critic"):
        getattr(learner, name).load_state_dict(bundle[name], strict=True)
    for name in ("opt_wm", "opt_pi", "opt_v"):
        getattr(learner, name).load_state_dict(bundle[name])
    torch.random.set_rng_state(bundle["torch_rng"])
    torch.cuda.set_rng_state(bundle["cuda_rng"], learner.dev)
    random.setstate(bundle["python_rng"])
    np.random.set_state(bundle["numpy_rng"])
    buffer.mark_episode_boundary()
    env.reset(torch.arange(env.N, device=learner.dev))
