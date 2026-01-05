import os
import time
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from loguru import logger
from holomotion.src.env.isaaclab_components.isaaclab_terrain import (
    build_terrain_config,
)


class SceneFunctions:
    """Collection of scene component builders."""

    @staticmethod
    def build_robot_config(
        config: dict,
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
        actuators = {
            "all_joints": ImplicitActuatorCfg(**config.actuators.all_joints)
        }

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
