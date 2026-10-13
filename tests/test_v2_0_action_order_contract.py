"""Ensure v2.0 keeps the shared G1 motion-tracking action order."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).parents[1] / "holomotion" / "config"


def _resolved_action_config(config_name: str | None = None) -> dict:
    with initialize_config_dir(
        config_dir=str(CONFIG_ROOT.resolve()), version_base=None
    ):
        if config_name is None:
            config = compose(
                overrides=[
                    "+env=motion_tracking",
                    "+robot=unitree/G1/29dof/29dof_training_isaaclab",
                ]
            )
        else:
            config = compose(config_name=config_name)
    return OmegaConf.to_container(
        config.env.config.actions.dof_pos,
        resolve=True,
    )


def test_v2_preserves_shared_motion_tracking_action_order():
    # Resolve the reusable action/robot groups, not a historical training recipe.
    shared_action = _resolved_action_config()
    v2_action = _resolved_action_config(
        "training/motion_tracking/motion_tracking_v2_0_0"
    )

    assert v2_action == shared_action
    assert v2_action["params"]["joint_names"] == [".*"]
    assert "preserve_order" not in v2_action["params"]
