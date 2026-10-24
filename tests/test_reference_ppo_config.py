"""Validate the paired PPO recipe before creating either physics runtime."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from holomotion.src.algo.reference_ppo.algorithm import ReferencePPO
from holomotion.src.algo.reference_ppo.config import ReferencePPOConfig
from holomotion.src.utils.config import compile_config


def inputs(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    robot = tmp_path / "robot.urdf"
    robot.write_text("<robot/>")
    return dict(
        manifest=manifest, robot_urdf=robot, output=tmp_path / "output"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        dict(lr=float("nan")),
        dict(num_envs=True),
        dict(compile="false"),
        dict(physics_backend="typo"),
        dict(rollout_steps=0),
        dict(minibatches=7),
        dict(policy_snapshot_minutes=-1),
        dict(policy_snapshot_minutes=float("nan")),
    ],
)
def test_invalid_recipe_fails_before_physics(tmp_path, overrides):
    with pytest.raises(ValueError):
        ReferencePPOConfig.from_mapping(dict(inputs(tmp_path), **overrides))


@pytest.mark.parametrize("engine", ["mujoco_warp", "isaac_physx"])
def test_framework_composes_shared_task_for_both_backends(tmp_path, engine):
    paths = inputs(tmp_path)
    with initialize_config_dir(
        config_dir=str(
            Path(__file__).resolve().parents[1] / "holomotion/config"
        ),
        version_base=None,
    ):
        cfg = compose(
            config_name="training/motion_tracking/reference_ppo",
            overrides=[
                "motion_h5_path=" + str(paths["manifest"]),
                "robot_urdf=" + str(paths["robot_urdf"]),
                "experiment_dir=" + str(tmp_path / "run"),
                "algo.config.physics_backend=" + engine,
            ],
        )
    cfg = compile_config(cfg)
    algo = ReferencePPO(
        cfg.env, cfg.algo.config, log_dir=str(tmp_path / "run")
    )
    algo.load(None)
    assert algo.options.policy_snapshot_minutes == 0
    assert algo.options.physics_backend == engine
    assert algo.options.max_hours == 0.5 and algo.options.num_envs == 4096
    assert algo.options.output == tmp_path / "run/training"
    with pytest.raises(ValueError, match="recovery"):
        algo.load("unimplemented.pt")
