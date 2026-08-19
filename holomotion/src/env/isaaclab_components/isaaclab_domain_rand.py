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
from holomotion.src.utils.g1_variants import sample_correlated_hand_payloads


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
        # Keep the all-body selector as ``slice(None)``. Isaac Lab's material
        # randomizer then operates on the padded ``max_shapes`` tensor and
        # supports articulations whose collision-shape counts differ by env.
        # Resolving the regex ``.*`` into explicit body indices instead makes
        # the upstream term infer shape counts from env_0 and reject valid
        # heterogeneous assets.
        if body_names == ".*" or body_names == [".*"]:
            asset_cfg = SceneEntityCfg(asset_name)
        else:
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

    @staticmethod
    def _get_dr_randomize_actuator_gains(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_name: str = "robot",
        body_names: str = ".*",
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal[
            "uniform", "log_uniform", "gaussian"
        ] = "uniform",
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        return isaaclab_mdp.events.randomize_actuator_gains(
            env,
            env_ids,
            asset_cfg,
            stiffness_distribution_params,
            damping_distribution_params,
            operation=operation,
            distribution=distribution,
        )

    @staticmethod
    def _get_dr_randomize_mass(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_name: str = "robot",
        body_names: str = ".*",
        mass_range: tuple[float, float] | None = None,
    ):
        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        return isaaclab_mdp.events.randomize_rigid_body_mass(
            env,
            env_ids,
            mass_distribution_params=mass_range,
            asset_cfg=asset_cfg,
            operation="add",
        )

    @staticmethod
    def _get_dr_hand_payload_mass(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_name: str = "robot",
        body_names: list[str] = (
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
        object_mass_range: tuple[float, float] = (0.0, 2.5),
        hardware_mass_range: tuple[float, float] = (0.0, 0.4),
        mode_probabilities: tuple[float, float, float, float] = (
            0.25,
            0.25,
            0.25,
            0.25,
        ),
        bimanual_split_range: tuple[float, float] = (0.3, 0.7),
        recompute_inertia: bool = True,
    ):
        """Apply bounded one-hand/bimanual payload mass to merged wrist bodies."""

        asset_cfg = SceneEntityCfg(asset_name, body_names=body_names)
        asset_cfg.resolve(env.scene)
        if (
            not isinstance(asset_cfg.body_ids, list)
            or len(asset_cfg.body_ids) != 2
        ):
            raise ValueError(
                "hand_payload_mass requires exactly two body_names in left/right order."
            )
        asset: Articulation = env.scene[asset_name]
        if env_ids is None:
            env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        body_ids = torch.tensor(
            asset_cfg.body_ids, dtype=torch.long, device="cpu"
        )

        added_mass, _ = sample_correlated_hand_payloads(
            len(env_ids_cpu),
            object_mass_range=object_mass_range,
            hardware_mass_range=hardware_mass_range,
            mode_probabilities=mode_probabilities,
            bimanual_split_range=bimanual_split_range,
            device="cpu",
        )
        default_mass = asset.data.default_mass[
            env_ids_cpu[:, None], body_ids
        ].clone()
        if bool((default_mass <= 0.0).any()):
            raise ValueError(
                "hand payload target bodies must have positive default mass."
            )
        target_mass = default_mass + added_mass

        masses = asset.root_physx_view.get_masses()
        masses[env_ids_cpu[:, None], body_ids] = target_mass
        asset.root_physx_view.set_masses(masses, env_ids_cpu)

        if recompute_inertia:
            ratios = target_mass / default_mass
            inertias = asset.root_physx_view.get_inertias()
            inertias[env_ids_cpu[:, None], body_ids] = (
                asset.data.default_inertia[env_ids_cpu[:, None], body_ids]
                * ratios[..., None]
            )
            asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


@configclass
class EventsCfg:
    pass


def build_domain_rand_config(domain_rand_config_dict: dict) -> EventsCfg:
    """Build IsaacLab-compatible EventsCfg from a config dictionary."""
    events_cfg = EventsCfg()

    for event_name, cfg in domain_rand_config_dict.items():
        # Keep non-event config under `domain_rand` available for Hydra
        # references without forcing it through the Isaac Lab event builder.
        if not (isinstance(cfg, dict) and "mode" in cfg):
            continue

        try:
            func = getattr(DomainRandFunctions, f"_get_dr_{event_name}")
        except AttributeError as exc:
            raise AttributeError(
                f"Unknown domain randomization event '{event_name}'"
            ) from exc
        term = EventTermCfg(
            func=func,
            **cfg,
        )
        setattr(events_cfg, event_name, term)

    return events_cfg
