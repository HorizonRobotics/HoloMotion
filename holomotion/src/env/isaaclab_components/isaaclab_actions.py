# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


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
