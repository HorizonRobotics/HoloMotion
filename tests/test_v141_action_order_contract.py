from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).parents[1] / "holomotion" / "config"


def _resolved_action_config(config_name: str) -> dict:
    with initialize_config_dir(
        config_dir=str(CONFIG_ROOT.resolve()), version_base=None
    ):
        config = compose(config_name=config_name)
    return OmegaConf.to_container(
        config.env.config.actions.dof_pos,
        resolve=True,
    )


def test_universal_finetune_keeps_v14_action_contract():
    v14_action = _resolved_action_config(
        "training/motion_tracking/motion_tracking_v1_4_0"
    )
    universal_action = _resolved_action_config(
        "training/motion_tracking/motion_tracking_v1_4_1_universal_finetune"
    )

    assert universal_action == v14_action
    assert universal_action["params"]["joint_names"] == [".*"]
    assert "preserve_order" not in universal_action["params"]
