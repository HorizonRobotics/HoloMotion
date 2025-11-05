import torch
from typing import Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg, EventTermCfg
from isaaclab.utils import configclass


from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg


class DomainRandFunctions:
    @staticmethod
    def _get_dr_default_dof_pos_bias(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_name: str = "robot",
        joint_names: list[str] = (".*"),
        pos_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal[
            "uniform", "log_uniform", "gaussian"
        ] = "uniform",
    ):
        asset_cfg = SceneEntityCfg(asset_name, joint_names=joint_names)
        asset_cfg.resolve(env.scene)
        asset: Articulation = env.scene[asset_name]
        asset.data.default_joint_pos_nominal = torch.clone(
            asset.data.default_joint_pos[0]
        )

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=asset.device)

        if asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)
        else:
            joint_ids = torch.tensor(
                asset_cfg.joint_ids,
                dtype=torch.int,
                device=asset.device,
            )

        if pos_distribution_params is not None:
            pos = asset.data.default_joint_pos.to(asset.device).clone()
            pos = _randomize_prop_by_op(
                pos,
                pos_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )[env_ids][:, joint_ids]

            if env_ids != slice(None) and joint_ids != slice(None):
                env_ids = env_ids[:, None]
            asset.data.default_joint_pos[env_ids, joint_ids] = pos
            env.action_manager.get_term("dof_pos")._offset[
                env_ids, joint_ids
            ] = pos

    @staticmethod
    def _get_dr_rigid_body_com(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_name: str = "robot",
        body_names: str = "torso_link",
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        return isaaclab_mdp.events.randomize_rigid_body_com(
            env,
            env_ids,
            com_range,
            asset_cfg,
        )

    @staticmethod
    def _get_dr_rigid_body_material(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_name: str = "robot",
        body_names: str = ".*",
        static_friction_range: tuple[float, float] | None = None,
        dynamic_friction_range: tuple[float, float] | None = None,
        restitution_range: tuple[float, float] | None = None,
        num_buckets: int = 64,
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        eveent_cfg = EventTermCfg(
            func=isaaclab_mdp.events.randomize_rigid_body_material,
            params={
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets,
            },
        )
        material_randomizer = (
            isaaclab_mdp.events.randomize_rigid_body_material(eveent_cfg, env)
        )
        return material_randomizer(env, env_ids, **eveent_cfg.params)

    @staticmethod
    def _get_dr_push_by_setting_velocity(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
    ):
        return isaaclab_mdp.events.push_by_setting_velocity(
            env,
            env_ids,
            velocity_range,
        )


@configclass
class EventsCfg:
    pass


def build_domain_rand_config(domain_rand_config_dict: dict) -> EventsCfg:
    """Build IsaacLab-compatible EventsCfg from a config dictionary."""
    events_cfg = EventsCfg()

    for event_name, cfg in domain_rand_config_dict.items():
        func = getattr(DomainRandFunctions, f"_get_dr_{event_name}")
        term = EventTermCfg(
            func=func,
            **cfg,
        )
        setattr(events_cfg, event_name, term)

    return events_cfg
