import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
import isaaclab.utils.math as isaaclab_math

from holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command import (
    RefMotionCommand,
)
import isaaclab.envs.mdp as isaaclab_mdp
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from loguru import logger


def _get_dof_indices(
    robot: Articulation,
    key_dofs: list[str] | None,
) -> list[int] | None:
    if key_dofs is None:
        return list(range(len(robot.dof_names)))
    dof_indices = []
    for name in key_dofs:
        if name not in robot.joint_names:
            raise ValueError(
                f"DOF '{name}' not found in robot.joint_names: {robot.joint_names}"
            )
        dof_indices.append(robot.joint_names.index(name))
    return dof_indices


def _get_body_indices(
    robot: Articulation,
    keybody_names: list[str] | None,
) -> list[int] | None:
    """Convert body names to indices.

    Args:
        robot: Robot articulation asset
        keybody_names: List of body names. If None, returns None.

    Returns:
        List of body indices corresponding to the given names, or None if keybody_names is None
    """
    if keybody_names is None:
        return list(range(len(robot.body_names)))

    body_indices = []
    for name in keybody_names:
        if name not in robot.body_names:
            raise ValueError(
                f"Body '{name}' not found in robot.body_names: {robot.body_names}"
            )
        body_indices.append(robot.body_names.index(name))

    return body_indices


def resolve_holo_config(value):
    def _sanitize_config_object(obj):
        for attr, attr_value in vars(obj).items():
            sanitized_value = resolve_holo_config(attr_value)
            setattr(obj, attr, sanitized_value)
        return obj

    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        if "_target_" in value:
            instantiated = hydra_instantiate(value)
            if hasattr(instantiated, "__dict__") and not callable(
                instantiated
            ):
                return _sanitize_config_object(instantiated)
            return instantiated
        return {key: resolve_holo_config(item) for key, item in value.items()}

    if isinstance(value, list):
        return [resolve_holo_config(item) for item in value]

    if hasattr(value, "__dict__") and not callable(value):
        return _sanitize_config_object(value)

    return value
