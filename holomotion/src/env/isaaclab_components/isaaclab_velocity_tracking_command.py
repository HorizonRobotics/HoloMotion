from dataclasses import MISSING

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class VelocityCommandCfg(CommandTermCfg):
    """Wrapper config that builds an IsaacLab UniformVelocity/Level command term.

    Use field `class_type` to set the underlying mdp command class; by default it
    uses UniformVelocityCommandCfg. For Unitree-style progressive limits, set to
    isaaclab_mdp.UniformLevelVelocityCommandCfg and provide `limit_ranges`.
    """

    class_type: type = isaaclab_mdp.UniformVelocityCommandCfg


@configclass
class UniformLevelVelocityCommandCfg(isaaclab_mdp.UniformVelocityCommandCfg):
    """Velocity command with progressive limit ranges (Unitree RL Lab style)."""

    limit_ranges: isaaclab_mdp.UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class VelTrack_CommandsCfg:
    pass


def _convert_ranges_dict_to_object(
    ranges_dict: dict,
) -> isaaclab_mdp.UniformVelocityCommandCfg.Ranges:
    """Convert a dict of ranges to a proper Ranges object with tuples."""
    ranges_kwargs = {}
    for key, value in ranges_dict.items():
        if value is None:
            ranges_kwargs[key] = None
        elif isinstance(value, (list, tuple)):
            ranges_kwargs[key] = tuple(value)
        else:
            ranges_kwargs[key] = value
    return isaaclab_mdp.UniformVelocityCommandCfg.Ranges(**ranges_kwargs)


def build_velocity_commands_config(command_config_dict: dict) -> VelTrack_CommandsCfg:
    """Build a CommandsCfg that supports velocity commands via IsaacLab isaaclab_mdp.

    Expected format:
    {
      "base_velocity": {
        "type": "VelocityCommandCfg" | "UniformVelocityCommandCfg" | "UniformLevelVelocityCommandCfg",
        "params": { ... }  # args compatible with mdp command cfgs
      }
    }

    For ranges and limit_ranges, pass them as dicts with keys like lin_vel_x, lin_vel_y, ang_vel_z, heading.
    """
    commands_cfg = VelTrack_CommandsCfg()

    for name, cfg in command_config_dict.items():
        command_type = cfg.get("type", "VelocityCommandCfg")
        params = cfg.get("params", {}).copy()

        if "ranges" in params and isinstance(params["ranges"], dict):
            params["ranges"] = _convert_ranges_dict_to_object(params["ranges"])

        if "limit_ranges" in params and isinstance(params["limit_ranges"], dict):
            params["limit_ranges"] = _convert_ranges_dict_to_object(
                params["limit_ranges"]
            )

        if command_type == "VelocityCommandCfg":
            term_cfg = VelocityCommandCfg(**params)
        elif command_type == "UniformVelocityCommandCfg":
            term_cfg = isaaclab_mdp.UniformVelocityCommandCfg(**params)
        elif command_type == "UniformLevelVelocityCommandCfg":
            term_cfg = UniformLevelVelocityCommandCfg(**params)
        else:
            raise ValueError(f"Unknown velocity command type: {command_type}")

        setattr(commands_cfg, name, term_cfg)

    return commands_cfg
