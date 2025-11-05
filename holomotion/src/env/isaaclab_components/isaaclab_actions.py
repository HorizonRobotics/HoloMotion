from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp


class ActionFunctions:
    """Collection of action function implementations."""

    @staticmethod
    def joint_position_action(
        asset_name: str = "robot",
        joint_names: list[str] | None = None,
        use_default_offset: bool = True,
        scale: float = 1.0,
    ) -> mdp.JointPositionActionCfg:
        """Joint position control action."""
        if joint_names is None:
            joint_names = [".*"]
        return mdp.JointPositionActionCfg(
            asset_name=asset_name,
            joint_names=joint_names,
            use_default_offset=use_default_offset,
            scale=scale,
        )

    @staticmethod
    def joint_velocity_action(
        asset_name: str = "robot",
        joint_names: list[str] | None = None,
        scale: float = 1.0,
    ) -> mdp.JointVelocityActionCfg:
        """Joint velocity control action."""
        if joint_names is None:
            joint_names = [".*"]
        return mdp.JointVelocityActionCfg(
            asset_name=asset_name,
            joint_names=joint_names,
            scale=scale,
        )

    @staticmethod
    def joint_effort_action(
        asset_name: str = "robot",
        joint_names: list[str] | None = None,
        scale: float = 1.0,
    ) -> mdp.JointEffortActionCfg:
        """Joint effort control action."""
        if joint_names is None:
            joint_names = [".*"]
        return mdp.JointEffortActionCfg(
            asset_name=asset_name,
            joint_names=joint_names,
            scale=scale,
        )


@configclass
class ActionsCfg:
    """Container for action terms."""

    pass


def build_actions_config(actions_config_dict: dict) -> ActionsCfg:
    """Build IsaacLab-compatible ActionsCfg from a config dictionary."""
    actions_cfg = ActionsCfg()

    for action_name, action_config in actions_config_dict.items():
        action_type = action_config["type"]
        params = action_config.get("params", {})

        if action_type == "joint_position":
            action_term = ActionFunctions.joint_position_action(**params)
        elif action_type == "joint_velocity":
            action_term = ActionFunctions.joint_velocity_action(**params)
        elif action_type == "joint_effort":
            action_term = ActionFunctions.joint_effort_action(**params)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        setattr(actions_cfg, action_name, action_term)

    return actions_cfg
