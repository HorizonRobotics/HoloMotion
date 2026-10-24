"""Validated options for the shared-task simulator PPO control."""

import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ReferencePPOConfig:
    manifest: Path
    robot_urdf: Path
    output: Path
    physics_backend: str = "mujoco_warp"
    device: str = "cuda:0"
    profile: str = "paper"
    num_envs: int = 4096
    rollout_steps: int = 24
    hist: int = 10
    goal_frames: int = 10
    epochs: int = 5
    minibatches: int = 4
    microbatch: int = 2048
    lr: float = 0.001
    max_hours: float = 0.5
    max_rollouts: int = 1000000
    evaluate_minutes: float = 60.0
    checkpoint_minutes: float = 60.0
    policy_snapshot_minutes: float = 0.0
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    capture_graph: bool = True
    release_phase_cache: bool = True
    skip_evaluation: bool = False

    @property
    def preflight_only(self):
        return False

    @classmethod
    def from_mapping(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown reference PPO options: {sorted(unknown)}"
            )
        values = dict(values)
        for key in ("manifest", "robot_urdf", "output"):
            values[key] = Path(values[key]).expanduser().resolve()
        result = cls(**values)
        for name in (
            "num_envs",
            "rollout_steps",
            "hist",
            "goal_frames",
            "epochs",
            "minibatches",
            "microbatch",
            "max_rollouts",
        ):
            value = getattr(result, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "lr",
            "max_hours",
            "evaluate_minutes",
            "checkpoint_minutes",
        ):
            value = getattr(result, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(result.policy_snapshot_minutes)
            or result.policy_snapshot_minutes < 0
        ):
            raise ValueError(
                "Policy snapshot minutes must be finite and nonnegative"
            )
        for name in (
            "compile",
            "capture_graph",
            "release_phase_cache",
            "skip_evaluation",
        ):
            if type(getattr(result, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if result.physics_backend not in ("mujoco_warp", "isaac_physx"):
            raise ValueError("Unknown physics backend")
        if result.amp not in ("none", "bf16") or result.profile not in (
            "paper",
            "vibed",
        ):
            raise ValueError("Unknown precision or observation profile")
        if type(result.seed) is not int:
            raise ValueError("Seed must be an integer")
        if result.num_envs * result.rollout_steps % result.minibatches:
            raise ValueError("PPO minibatches must cover the complete rollout")
        if not result.manifest.is_file() or not result.robot_urdf.is_file():
            raise ValueError("Motion manifest and original URDF must exist")
        if result.output.exists() and any(result.output.iterdir()):
            raise ValueError("Choose an empty output directory")
        return result
