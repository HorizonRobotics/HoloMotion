"""Learning-state restart checks; physical episodes are deliberately reset."""

import copy
import random
from pathlib import Path

import numpy as np
import torch

from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
from holoboxing.training.rsl_rl.modules.normalization import (
    validate_statistics,
)
from holoboxing.training.tasks.combat.config.sandbag import ROLES

RESTART_MODE = "reset_physics_and_episode_histories"


def validate_module_state(module, saved, label):
    """Reject incompatible tensors before PyTorch can copy or cast any of them."""
    expected = module.state_dict()
    if not isinstance(saved, dict):
        raise ValueError(f"Invalid {label} state")
    missing = expected.keys() - saved.keys()
    # Preserve the existing four-buffer legacy normalizer migration.
    for key in expected:
        if not key.endswith("_ex2"):
            continue
        prefix = key[: -len("_ex2")]
        legacy = {prefix + n for n in ("_mean", "_var", "_std", "count")}
        added = {
            prefix + n
            for n in (
                "_ex2",
                "_last_sync_mean",
                "_last_sync_var",
                "_last_sync_count",
            )
        }
        if legacy <= saved.keys() and not (added & saved.keys()):
            missing -= added
    if missing or saved.keys() - expected.keys():
        raise ValueError(f"Incomplete or unexpected {label} keys")
    for name, value in saved.items():
        template = expected[name]
        if (
            not torch.is_tensor(value)
            or value.shape != template.shape
            or value.dtype != template.dtype
            or value.layout != template.layout
        ):
            raise ValueError(f"Incompatible {label} tensor: {name}")
    for key in expected.keys() - saved.keys():
        if key.endswith("_ex2"):
            prefix = key[: -len("_ex2")]
            ex2 = saved[prefix + "_var"] + saved[prefix + "_mean"].square()
            if not torch.isfinite(ex2).all():
                raise ValueError(f"Nonfinite migrated {label} statistics")


def check_identity(saved, current, allow_code_change=False):
    code_keys = {"code", "shared_code"}
    if {k: v for k, v in saved.items() if k not in code_keys} != {
        k: v for k, v in current.items() if k not in code_keys
    }:
        raise ValueError("Resume resources, demonstrations or runtime differ")
    changes = {
        key: {"saved": saved.get(key), "current": current.get(key)}
        for key in code_keys
        if saved.get(key) != current.get(key)
    }
    if changes and not allow_code_change:
        raise ValueError(
            "Resume code differs; review the changes before explicitly using "
            "--allow-resume-code-change"
        )
    return changes


def capture_rng(env):
    device = torch.device(env.device)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None,
        "environment": env.generator.get_state(),
        "observation_noise": env.actor_noise.rng_state(),
    }


def validate_rng(rng, device):
    device = torch.device(device)
    if (rng["cuda"] is None) != (device.type != "cuda"):
        raise ValueError("Resume must use the same CPU/CUDA device type")
    # Validate without advancing any live random stream.
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    torch.Generator().set_state(rng["torch"])
    for key in ("environment", "observation_noise"):
        torch.Generator(device=device).set_state(rng[key])
    if device.type == "cuda":
        torch.Generator(device=device).set_state(rng["cuda"])


def restore_rng(rng, env):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.device(env.device).type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"], env.device)
    env.generator.set_state(rng["environment"])
    env.actor_noise.generator.set_state(rng["observation_noise"])


def validate_adam(saved, optimizer, steps):
    groups, current = saved["param_groups"], optimizer.param_groups
    if len(groups) != len(current):
        raise ValueError("Incomplete optimizer parameter groups")
    ids = []
    for group, live in zip(groups, current, strict=True):
        if len(group["params"]) != len(live["params"]):
            raise ValueError("Optimizer parameter count differs")
        for key in live.keys() - {"params", "lr"}:
            if group.get(key) != live[key]:
                raise ValueError(f"Optimizer setting differs: {key}")
        for index, parameter in zip(
            group["params"], live["params"], strict=True
        ):
            ids.append(index)
            if steps:
                moment = saved["state"].get(index, {})
                if set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("Missing Adam moments")
                if (
                    moment["step"].numel() != 1
                    or moment["step"].item() != steps
                ):
                    raise ValueError("Adam step differs from learner progress")
                for key in ("exp_avg", "exp_avg_sq"):
                    if (
                        moment[key].shape != parameter.shape
                        or moment[key].dtype != parameter.dtype
                    ):
                        raise ValueError("Adam moment layout differs")
                if (moment["exp_avg_sq"] < 0).any():
                    raise ValueError("Invalid Adam squared moment")
    if len(set(ids)) != len(ids) or set(saved["state"]) != (
        set(ids) if steps else set()
    ):
        raise ValueError("Unexpected optimizer state ownership")


def learning_state(runner):
    replay = runner.amp.replay
    return {
        "resume_version": 1,
        "restart_mode": RESTART_MODE,
        "optimizers": {
            role: optimizer.state_dict()
            for role, optimizer in runner.algorithm.optimizers.items()
        },
        "disc_optimizer": runner.amp.optimizer.state_dict(),
        "amp_replay": {
            "capacity": replay.max_length,
            "size": replay.current_size,
            "index": replay.current_idx,
            "buffer": replay.buffer,
        },
        "rng": capture_rng(runner.env),
        "invalid_transitions": runner.invalid_transitions,
        "resume_source": runner.resume_source,
    }


def validate_learning_state(runner, state):
    cfg = runner.config
    iteration = state["iteration"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Invalid completed update count")
    transitions = iteration * cfg.num_envs * cfg.num_steps_per_env
    steps = iteration * cfg.num_learning_epochs * cfg.num_mini_batches
    if (
        type(state["transitions_per_role"]) is not int
        or state["transitions_per_role"] != transitions
        or state["optimizer_steps"] != dict.fromkeys(ROLES, steps)
        or state["disc_optimizer_steps"] != steps
        or state["elapsed_seconds"] < 0
    ):
        raise ValueError("Inconsistent completed training progress")
    invalid = state["invalid_transitions"]
    if type(invalid) is not int or not 0 <= invalid <= transitions:
        raise ValueError("Invalid dropped-transition count")
    for name in ("optimizers", "learning_rates"):
        if set(state[name]) != set(ROLES):
            raise ValueError(f"Missing paired {name}")
    require_finite(state, "resume checkpoint")
    for module, key in (
        (runner.policy, "policy"),
        (runner.normalizers, "normalizers"),
        (runner.amp.discriminator, "discriminator"),
        (runner.amp.normalizer, "amp_normalizer"),
    ):
        validate_module_state(module, state[key], key)
    for role in ROLES:
        actor = getattr(runner.policy, f"taskpolicy_{role}")
        for name in ("neutral_reference", "physical_scale"):
            expected = getattr(actor, name).detach().cpu()
            actual = (
                state["policy"][f"taskpolicy_{role}.{name}"].detach().cpu()
            )
            matches = (
                torch.allclose(expected, actual, atol=1e-6, rtol=0)
                if name == "neutral_reference"
                else torch.equal(expected, actual)
            )
            if not matches:
                raise ValueError(
                    f"Checkpoint action mapping differs: {role}/{name}"
                )
    validate_statistics(state["normalizers"])
    validate_statistics(state["amp_normalizer"])
    provenance = state["supervised_initialization"]
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("checkpoint"), str)
        or not provenance["checkpoint"]
        or not isinstance(provenance.get("sha256"), str)
        or len(provenance["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in provenance["sha256"])
    ):
        raise ValueError("Missing supervised initialization provenance")
    if state["normalizers"]["actor_ego.count"].item() <= 0:
        raise ValueError("Attacker normalizer must be fitted")
    for role in ROLES:
        lr = state["learning_rates"][role]
        if not cfg.min_learning_rate <= lr <= cfg.max_learning_rate:
            raise ValueError("Invalid saved learning rate")
        if any(
            g["lr"] != lr for g in state["optimizers"][role]["param_groups"]
        ):
            raise ValueError("Optimizer and scheduler learning rates differ")
        validate_adam(
            state["optimizers"][role], runner.algorithm.optimizers[role], steps
        )
    validate_adam(state["disc_optimizer"], runner.amp.optimizer, steps)
    if any(
        g["lr"] != cfg.disc_learning_rate
        for g in state["disc_optimizer"]["param_groups"]
    ):
        raise ValueError("AMP optimizer learning rate differs")
    replay = state["amp_replay"]
    capacity = cfg.amp_replay_buffer_size
    inserted = 2 * (transitions - invalid)
    if (
        replay["capacity"] != capacity
        or type(replay["size"]) is not int
        or type(replay["index"]) is not int
        or replay["size"] != min(inserted, capacity)
        or replay["index"] != inserted % capacity
    ):
        raise ValueError("AMP replay differs from completed rollout progress")
    buffer = replay["buffer"]
    if inserted == 0:
        if buffer is not None:
            raise ValueError("Unexpected empty AMP replay data")
    elif (
        not torch.is_tensor(buffer)
        or buffer.shape != (capacity, 122)
        or buffer.dtype != torch.float32
    ):
        raise ValueError("Invalid AMP replay tensor")
    validate_rng(state["rng"], runner.device)


def resume(runner, path, checkpoint_format, allow_code_change=False):
    if (
        runner.iteration
        or runner.observations is not None
        or runner.algorithm.storage.step
        or runner.supervised_initialization is not None
        or runner.resume_source is not None
    ):
        raise ValueError("Resume requires a fresh runner")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not set(
        runner.checkpoint_state()
    ).issubset(state):
        raise ValueError(
            "Incomplete resume checkpoint; legacy weights-only checkpoints "
            "cannot restore optimizer, replay or random history"
        )
    if (
        state["format"] != checkpoint_format
        or state["resume_version"] != 1
        or state["restart_mode"] != RESTART_MODE
        or state["config"] != runner.config.to_dict()
    ):
        raise ValueError("Resume format or configuration differs")
    changes = check_identity(
        state["identity"], runner.identity, allow_code_change
    )
    validate_learning_state(runner, state)
    runner.policy.load_state_dict(state["policy"], strict=True)
    runner.normalizers.load_state_dict(state["normalizers"], strict=True)
    runner.amp.discriminator.load_state_dict(
        state["discriminator"], strict=True
    )
    runner.amp.normalizer.load_state_dict(state["amp_normalizer"], strict=True)
    for role in ROLES:
        runner.algorithm.optimizers[role].load_state_dict(
            state["optimizers"][role]
        )
    runner.amp.optimizer.load_state_dict(state["disc_optimizer"])
    runner.algorithm.learning_rates = copy.deepcopy(state["learning_rates"])
    runner.algorithm.optimizer_steps = copy.deepcopy(state["optimizer_steps"])
    runner.amp.optimizer_steps = state["disc_optimizer_steps"]
    replay = state["amp_replay"]
    runner.amp.replay.buffer = (
        None
        if replay["buffer"] is None
        else replay["buffer"].to(runner.device)
    )
    runner.amp.replay.current_size = replay["size"]
    runner.amp.replay.current_idx = replay["index"]
    for key in (
        "iteration",
        "transitions_per_role",
        "elapsed_seconds",
        "invalid_transitions",
        "supervised_initialization",
    ):
        setattr(runner, key, copy.deepcopy(state[key]))
    restore_rng(state["rng"], runner.env)
    runner.algorithm.reset_exploration()
    runner.resume_source = {
        "checkpoint": str(Path(path).resolve()),
        "iteration": state["iteration"],
        "code_changes": changes,
    }
