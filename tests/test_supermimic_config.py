"""Configuration boundaries shared by CLI and the framework training entry."""

from dataclasses import asdict
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from holomotion.scripts.training.train_supermimic import build_parser
from holomotion.src.algo.supermimic.algorithm import SuperMimic
from holomotion.src.algo.supermimic.config import SuperMimicConfig
from holomotion.src.utils.config import compile_config


@pytest.fixture
def paths(tmp_path):
    manifest, robot = tmp_path / "manifest.json", tmp_path / "robot.urdf"
    manifest.write_text("{}")
    robot.write_text("<robot/>")
    return dict(
        manifest=manifest, robot_urdf=robot, output=tmp_path / "output"
    )


def test_cli_and_programmatic_recipe_have_identical_defaults(paths):
    parser = build_parser()
    namespace = parser.parse_args(
        [
            item
            for key, value in paths.items()
            for item in ("--" + key.replace("_", "-"), str(value))
        ]
    )
    actual = SuperMimicConfig.from_mapping(vars(namespace))
    wanted = SuperMimicConfig.from_mapping(paths)
    assert actual == wanted


@pytest.mark.parametrize(
    "overrides",
    [
        dict(buffer_steps=2),
        dict(warmup_steps=0),
        dict(pi_microbatch=0),
        dict(utd="nan"),
        dict(gamma=1),
        dict(action_noise=float("inf")),
        dict(compile="false"),
        dict(max_hours=float("nan")),
        dict(num_envs=True),
        dict(training_reward="unknown"),
        dict(training_reward="native_v1"),
        dict(typo_batch=8),
    ],
)
def test_reject_invalid_recipe_before_starting_physics(paths, overrides):
    with pytest.raises(ValueError):
        SuperMimicConfig.from_mapping(dict(paths, **overrides))


def test_hydra_recipe_resolves_paths_options_and_budget(paths, tmp_path):
    directory = str(Path(__file__).resolve().parents[1] / "holomotion/config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        cfg = compose(
            config_name="training/motion_tracking/supermimic",
            overrides=[
                "motion_h5_path=" + str(paths["manifest"]),
                "robot_urdf=" + str(paths["robot_urdf"]),
                "num_envs=2",
                "algo.config.max_hours=0.5",
                "algo.config.pi_microbatch=8",
                "experiment_dir=" + str(tmp_path / "experiment"),
            ],
        )
    cfg = compile_config(cfg)
    assert cfg.mujoco_eval.enabled is False
    algo = SuperMimic(
        cfg.env, cfg.algo.config, log_dir=str(tmp_path / "experiment")
    )
    algo.load(None)
    assert algo.options.num_envs == 2 and algo.options.max_hours == 0.5
    assert algo.options.pi_microbatch == 8 and algo.options.pi_batch == 4096
    assert algo.options.manifest == paths["manifest"]
    assert algo.options.output == tmp_path / "experiment/training"
    defaults = asdict(SuperMimicConfig.from_mapping(paths))
    for key, value in OmegaConf.to_container(
        cfg.algo.config, resolve=True
    ).items():
        if key not in ("device", "num_envs", "max_hours", "pi_microbatch"):
            assert value == defaults[key]


def test_existing_nonempty_output_is_never_overwritten(paths):
    paths["output"].mkdir()
    (paths["output"] / "checkpoint.pt").write_bytes(b"previous experiment")
    with pytest.raises(ValueError, match="empty"):
        SuperMimicConfig.from_mapping(paths)


def test_short_physics_preflight_does_not_require_training_replay_warmup(
    paths,
):
    config = SuperMimicConfig.from_mapping(
        dict(paths, preflight_only=True, warmup_steps=1)
    )
    assert config.warmup_steps == 1
