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



import copy
import os
import time
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from loguru import logger
from holomotion.src.env.isaaclab_components.isaaclab_terrain import (
    build_terrain_config,
)
from holomotion.src.env.isaaclab_components.unitree_actuators import (
    UnitreeActuator,
    UnitreeActuatorCfg,
    UnitreeErfiActuator,
    UnitreeErfiActuatorCfg,
)


class SceneFunctions:
    """Collection of scene component builders."""

    @staticmethod
    def build_robot_config(
        config: dict,
        domain_rand_config: dict | None = None,
        main_process: bool = True,
        process_id: int = 0,
        num_processes: int = 1,
    ) -> ArticulationCfg:
        """Build robot articulation configuration.

        Args:
            config: Robot configuration dictionary
            main_process: Whether this is the main process (from compiled config)
            process_id: Process ID/rank (from compiled config)
            num_processes: Total number of processes (from compiled config)
        """
        urdf_path = config.asset.urdf_file
        init_pos = config.init_state.pos
        default_joint_positions = config.init_state.default_joint_angles
        root_link_name = config.get("root_name", "pelvis")
        prim_path = "{ENV_REGEX_NS}/Robot"

        actuator_type = config.actuators.get("actuator_type", "implicit")
        if actuator_type in {"unitree", "unitree_erfi"}:
            actuators = _build_unitree_actuator_cfg(
                config.actuators, domain_rand_config or {}
            )
        else:
            actuators = {
                "all_joints": ImplicitActuatorCfg(
                    **config.actuators.all_joints
                )
            }

        logger.info(f"Using {actuator_type} actuators")
        logger.info(f"Actuators: {actuators}")

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # Configure USD output directory. Optionally isolate per rank to avoid races.
        usd_base_dir = os.path.join(os.path.dirname(urdf_path), "usd")
        unique_usd_per_rank = True
        if num_processes > 1 and unique_usd_per_rank:
            usd_dir = os.path.join(usd_base_dir, f"rank_{process_id}")
        else:
            usd_dir = usd_base_dir
        os.makedirs(usd_dir, exist_ok=True)
        logger.info(f"Using URDF path: {urdf_path}")
        logger.info(f"Using USD directory: {usd_dir}")

        force_usd_conversion = config.asset.get("force_usd_conversion", True)
        if num_processes > 1 and unique_usd_per_rank:
            # Ensure each rank generates its own USD into its isolated directory
            force_usd_conversion = True

        # Handle DDP
        if num_processes > 1:
            logger.info(
                f"[Process {process_id}/{num_processes}] Distributed training detected"
            )

            if unique_usd_per_rank:
                logger.info(
                    f"[Process {process_id}] Using per-rank USD dir; forcing USD conversion: {force_usd_conversion}"
                )
            else:
                # Only main process should convert USD to avoid file conflicts
                if main_process:
                    logger.info(
                        f"[Process {process_id}] Main process - Force USD conversion: {force_usd_conversion}"
                    )
                else:
                    logger.info(
                        f"[Process {process_id}] Non-main process - Skipping USD conversion, waiting for main process"
                    )
                    force_usd_conversion = False

                    # Wait for USD files to be created by main process
                    urdf_basename = os.path.splitext(
                        os.path.basename(urdf_path)
                    )[0]
                    expected_usd_file = os.path.join(
                        usd_dir, f"{urdf_basename}.usd"
                    )

                    logger.info(
                        f"[Process {process_id}] Waiting for main process to create USD files at {expected_usd_file}..."
                    )
                    max_wait = 60
                    wait_interval = 1
                    waited = 0

                    while (
                        not os.path.exists(expected_usd_file)
                        and waited < max_wait
                    ):
                        time.sleep(wait_interval)
                        waited += wait_interval

                    if os.path.exists(expected_usd_file):
                        logger.info(
                            f"[Process {process_id}] USD file found, proceeding with loading"
                        )
                    else:
                        logger.warning(
                            f"[Process {process_id}] USD file not found after {max_wait}s, proceeding anyway"
                        )
        else:
            logger.info(
                f"Single process training. Force USD conversion: {force_usd_conversion}"
            )

        articulation_cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=os.path.abspath(urdf_path),
                usd_dir=os.path.abspath(usd_dir),
                force_usd_conversion=force_usd_conversion,
                fix_base=False,
                merge_fixed_joints=True,
                root_link_name=root_link_name,
                replace_cylinders_with_capsules=True,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0,
                        damping=0,
                    )
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                joint_pos=default_joint_positions,
                joint_vel={".*": 0.0},
            ),
            soft_joint_pos_limit_factor=0.9,
            actuators=actuators,
        )

        return articulation_cfg

    @staticmethod
    def build_lighting_config(
        config: dict,
    ) -> tuple[AssetBaseCfg, AssetBaseCfg]:
        """Build lighting configuration."""
        distant_light_intensity = config.get("distant_light_intensity", 3000.0)
        dome_light_intensity = config.get("dome_light_intensity", 1000.0)
        distant_light_color = config.get(
            "distant_light_color", (0.75, 0.75, 0.75)
        )
        dome_light_color = config.get("dome_light_color", (0.13, 0.13, 0.13))

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(
                color=distant_light_color, intensity=distant_light_intensity
            ),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=dome_light_color, intensity=dome_light_intensity
            ),
        )
        return light, sky_light

    @staticmethod
    def build_contact_sensor_config(config: dict) -> ContactSensorCfg:
        """Build contact sensor configuration."""
        prim_path = config.get("prim_path", "{ENV_REGEX_NS}/Robot/.*")
        history_length = config.get("history_length", 3)
        force_threshold = config.get("force_threshold", 10.0)
        track_air_time = config.get("track_air_time", True)
        debug_vis = config.get("debug_vis", False)

        return ContactSensorCfg(
            prim_path=prim_path,
            history_length=history_length,
            track_air_time=track_air_time,
            force_threshold=force_threshold,
            debug_vis=debug_vis,
        )


@configclass
class MotionTrackingSceneCfg(InteractiveSceneCfg):
    """Scene configuration for motion tracking environment."""

    pass


def build_scene_config(
    scene_config_dict: dict,
    main_process: bool = True,
    process_id: int = 0,
    num_processes: int = 1,
) -> MotionTrackingSceneCfg:
    """Build IsaacLab-compatible scene configuration from config dictionary.

    Args:
        scene_config_dict: Scene configuration dictionary
        main_process: Whether this is the main process (from compiled config)
        process_id: Process ID/rank (from compiled config)
        num_processes: Total number of processes (from compiled config)
    """
    scene_cfg = MotionTrackingSceneCfg()

    # Basic scene properties
    scene_cfg.num_envs = scene_config_dict.get("num_envs", MISSING)
    scene_cfg.env_spacing = scene_config_dict.get("env_spacing", 2.5)
    scene_cfg.replicate_physics = scene_config_dict.get(
        "replicate_physics", True
    )

    # Build robot configuration with process info
    if "robot" in scene_config_dict:
        robot_config = scene_config_dict["robot"]
        scene_cfg.robot = SceneFunctions.build_robot_config(
            robot_config,
            domain_rand_config=scene_config_dict.get("domain_rand", {}),
            main_process=main_process,
            process_id=process_id,
            num_processes=num_processes,
        )

    # Build terrain configuration
    if "terrain" in scene_config_dict:
        terrain_config = scene_config_dict["terrain"]
        scene_cfg.terrain = build_terrain_config(
            terrain_config, scene_env_spacing=scene_cfg.env_spacing
        )
        if "robot" in scene_config_dict:
            scene_cfg.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
                ray_alignment="world",
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=1.0, size=(1.0e-3, 1.0e-3)
                ),
                debug_vis=False,
                mesh_prim_paths=[str(scene_cfg.terrain.prim_path)],
                max_distance=1.0e6,
            )

    # Build lighting configuration
    if "lighting" in scene_config_dict:
        lighting_config = scene_config_dict["lighting"]
        light, sky_light = SceneFunctions.build_lighting_config(
            lighting_config
        )
        scene_cfg.light = light
        scene_cfg.sky_light = sky_light

    # Build contact sensor configuration
    if "contact_sensor" in scene_config_dict:
        contact_config = scene_config_dict["contact_sensor"]
        scene_cfg.contact_forces = SceneFunctions.build_contact_sensor_config(
            contact_config
        )

    return scene_cfg


def _cfg_to_kwargs(cfg: object) -> dict:
    return {
        key: copy.deepcopy(value)
        for key, value in vars(cfg).items()
        if not key.startswith("_")
    }


def _build_unitree_actuator_cfg(
    config: dict, domain_rand_config: dict
) -> dict[str, object]:
    base_cfg = unitree_actuator_config_hardcoded["all_joints"]
    base_kwargs = _cfg_to_kwargs(base_cfg)
    action_delay_cfg = copy.deepcopy(
        domain_rand_config.get("action_delay", {})
    )
    if action_delay_cfg.get("enabled", False):
        delay_kwargs = {
            "min_delay": int(action_delay_cfg.get("min_delay", 0)),
            "max_delay": int(action_delay_cfg.get("max_delay", 0)),
        }
    else:
        delay_kwargs = {"min_delay": 0, "max_delay": 0}

    if config.get("actuator_type", "unitree") == "unitree_erfi":
        erfi_cfg = copy.deepcopy(domain_rand_config.get("erfi", {}))
        actuator_filter_kwargs = {
            "ema_filter_enabled": bool(
                config.get("ema_filter_enabled", False)
            ),
            "ema_filter_alpha": config.get("ema_filter_alpha", 1.0),
        }
        erfi_kwargs = {
            "erfi_enabled": bool(erfi_cfg.get("enabled", False)),
            "rfi_probability": erfi_cfg.get("rfi_probability", 0.5),
            "rfi_lim": erfi_cfg.get("rfi_lim", 0.1),
            "randomize_rfi_lim": erfi_cfg.get("randomize_rfi_lim", True),
            "rfi_lim_range": erfi_cfg.get("rfi_lim_range", (0.5, 1.5)),
            "rao_lim": erfi_cfg.get("rao_lim", 0.1),
        }
        actuator_kwargs = {
            **base_kwargs,
            **delay_kwargs,
            **actuator_filter_kwargs,
            **erfi_kwargs,
        }
        actuator_cfg = UnitreeErfiActuatorCfg(**actuator_kwargs)
        actuator_cfg.class_type = UnitreeErfiActuator
    else:
        actuator_kwargs = {**base_kwargs, **delay_kwargs}
        actuator_cfg = UnitreeActuatorCfg(**actuator_kwargs)
        actuator_cfg.class_type = UnitreeActuator

    return {"all_joints": actuator_cfg}


unitree_actuator_config_hardcoded = {
    "all_joints": UnitreeActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        min_delay=0,
        max_delay=0,
        effort_limit={
            ".*_hip_yaw_joint": 88,
            ".*_hip_roll_joint": 139,
            ".*_hip_pitch_joint": 88,
            ".*_knee_joint": 139,
            ".*_ankle_pitch_joint": 50,
            ".*_ankle_roll_joint": 50,
            "waist_roll_joint": 50,
            "waist_pitch_joint": 50,
            "waist_yaw_joint": 88,
            ".*_shoulder_pitch_joint": 25,
            ".*_shoulder_roll_joint": 25,
            ".*_shoulder_yaw_joint": 25,
            ".*_elbow_joint": 25,
            ".*_wrist_roll_joint": 25,
            ".*_wrist_pitch_joint": 5,
            ".*_wrist_yaw_joint": 5,
        },
        velocity_limit={
            ".*_hip_yaw_joint": 32,
            ".*_hip_roll_joint": 20,
            ".*_hip_pitch_joint": 32,
            ".*_knee_joint": 20,
            ".*_ankle_pitch_joint": 37,
            ".*_ankle_roll_joint": 37,
            "waist_roll_joint": 37,
            "waist_pitch_joint": 37,
            "waist_yaw_joint": 32,
            ".*_shoulder_pitch_joint": 37,
            ".*_shoulder_roll_joint": 37,
            ".*_shoulder_yaw_joint": 37,
            ".*_elbow_joint": 37,
            ".*_wrist_roll_joint": 37,
            ".*_wrist_pitch_joint": 22,
            ".*_wrist_yaw_joint": 22,
        },
        stiffness={
            ".*_hip_yaw_joint": 40.1792384737,
            ".*_hip_roll_joint": 99.0984277823,
            ".*_hip_pitch_joint": 40.1792384737,
            ".*_knee_joint": 99.0984277823,
            ".*_ankle_pitch_joint": 28.5012461974,
            ".*_ankle_roll_joint": 28.5012461974,
            "waist_roll_joint": 28.5012461974,
            "waist_pitch_joint": 28.5012461974,
            "waist_yaw_joint": 40.1792384737,
            ".*_shoulder_pitch_joint": 14.2506230987,
            ".*_shoulder_roll_joint": 14.2506230987,
            ".*_shoulder_yaw_joint": 14.2506230987,
            ".*_elbow_joint": 14.2506230987,
            ".*_wrist_roll_joint": 14.2506230987,
            ".*_wrist_pitch_joint": 16.7783274819,
            ".*_wrist_yaw_joint": 16.7783274819,
        },
        damping={
            ".*_hip_yaw_joint": 2.5578897651,
            ".*_hip_roll_joint": 6.30880185368,
            ".*_hip_pitch_joint": 2.5578897651,
            ".*_knee_joint": 6.30880185368,
            ".*_ankle_pitch_joint": 1.81444568664,
            ".*_ankle_roll_joint": 1.81444568664,
            "waist_roll_joint": 1.81444568664,
            "waist_pitch_joint": 1.81444568664,
            "waist_yaw_joint": 2.5578897651,
            ".*_shoulder_pitch_joint": 0.907222843318,
            ".*_shoulder_roll_joint": 0.907222843318,
            ".*_shoulder_yaw_joint": 0.907222843318,
            ".*_elbow_joint": 0.907222843318,
            ".*_wrist_roll_joint": 0.907222843318,
            ".*_wrist_pitch_joint": 1.06814150222,
            ".*_wrist_yaw_joint": 1.06814150222,
        },
        armature={
            ".*_hip_yaw_joint": 0.01017752,
            ".*_hip_roll_joint": 0.025101925,
            ".*_hip_pitch_joint": 0.01017752,
            ".*_knee_joint": 0.025101925,
            ".*_ankle_pitch_joint": 0.00721945,
            ".*_ankle_roll_joint": 0.00721945,
            "waist_roll_joint": 0.00721945,
            "waist_pitch_joint": 0.00721945,
            "waist_yaw_joint": 0.01017752,
            ".*_shoulder_pitch_joint": 0.003609725,
            ".*_shoulder_roll_joint": 0.003609725,
            ".*_shoulder_yaw_joint": 0.003609725,
            ".*_elbow_joint": 0.003609725,
            ".*_wrist_roll_joint": 0.003609725,
            ".*_wrist_pitch_joint": 0.00425,
            ".*_wrist_yaw_joint": 0.00425,
        },
        friction=0,
        dynamic_friction=0,
        viscous_friction=0,
        X1={
            ".*_hip_yaw_joint": 22.63,
            ".*_hip_roll_joint": 14.5,
            ".*_hip_pitch_joint": 22.63,
            ".*_knee_joint": 14.5,
            ".*_ankle_pitch_joint": 30.86,
            ".*_ankle_roll_joint": 30.86,
            "waist_roll_joint": 30.86,
            "waist_pitch_joint": 30.86,
            "waist_yaw_joint": 22.63,
            ".*_shoulder_pitch_joint": 30.86,
            ".*_shoulder_roll_joint": 30.86,
            ".*_shoulder_yaw_joint": 30.86,
            ".*_elbow_joint": 30.86,
            ".*_wrist_roll_joint": 30.86,
            ".*_wrist_pitch_joint": 15.3,
            ".*_wrist_yaw_joint": 15.3,
        },
        X2={
            ".*_hip_yaw_joint": 35.52,
            ".*_hip_roll_joint": 22.7,
            ".*_hip_pitch_joint": 35.52,
            ".*_knee_joint": 22.7,
            ".*_ankle_pitch_joint": 40.13,
            ".*_ankle_roll_joint": 40.13,
            "waist_roll_joint": 40.13,
            "waist_pitch_joint": 40.13,
            "waist_yaw_joint": 35.52,
            ".*_shoulder_pitch_joint": 40.13,
            ".*_shoulder_roll_joint": 40.13,
            ".*_shoulder_yaw_joint": 40.13,
            ".*_elbow_joint": 40.13,
            ".*_wrist_roll_joint": 40.13,
            ".*_wrist_pitch_joint": 24.76,
            ".*_wrist_yaw_joint": 24.76,
        },
        Y1={
            ".*_hip_yaw_joint": 71,
            ".*_hip_roll_joint": 111,
            ".*_hip_pitch_joint": 71,
            ".*_knee_joint": 111,
            ".*_ankle_pitch_joint": 24.8,
            ".*_ankle_roll_joint": 24.8,
            "waist_roll_joint": 24.8,
            "waist_pitch_joint": 24.8,
            "waist_yaw_joint": 71,
            ".*_shoulder_pitch_joint": 24.8,
            ".*_shoulder_roll_joint": 24.8,
            ".*_shoulder_yaw_joint": 24.8,
            ".*_elbow_joint": 24.8,
            ".*_wrist_roll_joint": 24.8,
            ".*_wrist_pitch_joint": 4.8,
            ".*_wrist_yaw_joint": 4.8,
        },
        Y2={
            ".*_hip_yaw_joint": 83.3,
            ".*_hip_roll_joint": 131,
            ".*_hip_pitch_joint": 83.3,
            ".*_knee_joint": 131,
            ".*_ankle_pitch_joint": 31.9,
            ".*_ankle_roll_joint": 31.9,
            "waist_roll_joint": 31.9,
            "waist_pitch_joint": 31.9,
            "waist_yaw_joint": 83.3,
            ".*_shoulder_pitch_joint": 31.9,
            ".*_shoulder_roll_joint": 31.9,
            ".*_shoulder_yaw_joint": 31.9,
            ".*_elbow_joint": 31.9,
            ".*_wrist_roll_joint": 31.9,
            ".*_wrist_pitch_joint": 8.6,
            ".*_wrist_yaw_joint": 8.6,
        },
        Fs={
            ".*_hip_yaw_joint": 1.6,
            ".*_hip_roll_joint": 2.4,
            ".*_hip_pitch_joint": 1.6,
            ".*_knee_joint": 2.4,
            ".*_ankle_pitch_joint": 0.6,
            ".*_ankle_roll_joint": 0.6,
            "waist_roll_joint": 0.6,
            "waist_pitch_joint": 0.6,
            "waist_yaw_joint": 1.6,
            ".*_shoulder_pitch_joint": 0.6,
            ".*_shoulder_roll_joint": 0.6,
            ".*_shoulder_yaw_joint": 0.6,
            ".*_elbow_joint": 0.6,
            ".*_wrist_roll_joint": 0.6,
            ".*_wrist_pitch_joint": 0.6,
            ".*_wrist_yaw_joint": 0.6,
        },
        Fd={
            ".*_hip_yaw_joint": 0.16,
            ".*_hip_roll_joint": 0.24,
            ".*_hip_pitch_joint": 0.16,
            ".*_knee_joint": 0.24,
            ".*_ankle_pitch_joint": 0.06,
            ".*_ankle_roll_joint": 0.06,
            "waist_roll_joint": 0.06,
            "waist_pitch_joint": 0.06,
            "waist_yaw_joint": 0.16,
            ".*_shoulder_pitch_joint": 0.06,
            ".*_shoulder_roll_joint": 0.06,
            ".*_shoulder_yaw_joint": 0.06,
            ".*_elbow_joint": 0.06,
            ".*_wrist_roll_joint": 0.06,
            ".*_wrist_pitch_joint": 0.06,
            ".*_wrist_yaw_joint": 0.06,
        },
        Va={
            ".*_hip_yaw_joint": 0.01,
            ".*_hip_roll_joint": 0.01,
            ".*_hip_pitch_joint": 0.01,
            ".*_knee_joint": 0.01,
            ".*_ankle_pitch_joint": 0.01,
            ".*_ankle_roll_joint": 0.01,
            "waist_roll_joint": 0.01,
            "waist_pitch_joint": 0.01,
            "waist_yaw_joint": 0.01,
            ".*_shoulder_pitch_joint": 0.01,
            ".*_shoulder_roll_joint": 0.01,
            ".*_shoulder_yaw_joint": 0.01,
            ".*_elbow_joint": 0.01,
            ".*_wrist_roll_joint": 0.01,
            ".*_wrist_pitch_joint": 0.01,
            ".*_wrist_yaw_joint": 0.01,
        },
    )
}
