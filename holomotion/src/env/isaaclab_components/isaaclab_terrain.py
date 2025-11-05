

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg


TERRAIN_GENERATORS = {
    "cobblestone": terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=9,
        num_cols=21,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        },
    ),
}


def get_terrain_generator(
    generator_name: str,
) -> terrain_gen.TerrainGeneratorCfg:
    """Get predefined terrain generator configuration by name."""
    if generator_name not in TERRAIN_GENERATORS:
        raise ValueError(
            f"Unknown terrain generator: {generator_name}. Available: {list(TERRAIN_GENERATORS.keys())}"
        )
    return TERRAIN_GENERATORS[generator_name]


def build_terrain_config(
    config: dict, scene_env_spacing: float = None
) -> TerrainImporterCfg:
    """Build terrain configuration.

    Supports two modes:
    1. terrain_type="plane": Simple infinite plane with env_spacing grid
    2. generator_name specified: Procedural terrain generation

    Args:
        config: Terrain configuration dictionary with fields:
            - terrain_type (optional): "plane" for simple plane
            - generator_name (optional): Name of terrain generator (e.g., "cobblestone")
            - generator_params (optional): Override default generator parameters:
                - num_rows, num_cols, size, border_width, horizontal_scale,
                  vertical_scale, slope_threshold, difficulty_range
            - visual_material (optional): Visual appearance configuration
            - static_friction, dynamic_friction, restitution, etc.
        scene_env_spacing: Environment spacing from scene config (used for plane type)

    Returns:
        TerrainImporterCfg configured according to the input parameters
    """
    prim_path = config.get("prim_path", "/World/ground")
    static_friction = config.get("static_friction", 1.0)
    dynamic_friction = config.get("dynamic_friction", 1.0)
    restitution = config.get("restitution", 0.0)
    friction_combine_mode = config.get("friction_combine_mode", "multiply")
    restitution_combine_mode = config.get(
        "restitution_combine_mode", "multiply"
    )

    terrain_type = config.get("terrain_type", None)

    if terrain_type == "usd":
        usd_path = config.get("usd_path")
        if usd_path is None:
            raise ValueError(
                "'usd_path' must be specified for terrain_type 'usd'"
            )
        terrain_cfg = TerrainImporterCfg(
            prim_path=prim_path,
            terrain_type="usd",
            usd_path=usd_path,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=friction_combine_mode,
                restitution_combine_mode=restitution_combine_mode,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            ),
            debug_vis=config.get("debug_vis", False),
        )
        return terrain_cfg

    if terrain_type == "plane":
        env_spacing = (
            scene_env_spacing if scene_env_spacing is not None else 2.5
        )
        terrain_cfg = TerrainImporterCfg(
            prim_path=prim_path,
            terrain_type="plane",
            collision_group=-1,
            env_spacing=env_spacing,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=friction_combine_mode,
                restitution_combine_mode=restitution_combine_mode,
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            ),
            debug_vis=config.get("debug_vis", False),
        )
        return terrain_cfg

    generator_name = config.get("generator_name")
    if generator_name is None:
        raise ValueError(
            "Either 'terrain_type' or 'generator_name' must be specified in terrain config"
        )

    terrain_generator = get_terrain_generator(generator_name)

    if "generator_params" in config:
        params = config["generator_params"]
        if "size" in params:
            terrain_generator.size = tuple(params["size"])
        if "border_width" in params:
            terrain_generator.border_width = params["border_width"]
        if "num_rows" in params:
            terrain_generator.num_rows = params["num_rows"]
        if "num_cols" in params:
            terrain_generator.num_cols = params["num_cols"]
        if "horizontal_scale" in params:
            terrain_generator.horizontal_scale = params["horizontal_scale"]
        if "vertical_scale" in params:
            terrain_generator.vertical_scale = params["vertical_scale"]
        if "slope_threshold" in params:
            terrain_generator.slope_threshold = params["slope_threshold"]
        if "difficulty_range" in params:
            terrain_generator.difficulty_range = tuple(
                params["difficulty_range"]
            )

    terrain_cfg = TerrainImporterCfg(
        prim_path=prim_path,
        terrain_type="generator",
        terrain_generator=terrain_generator,
        max_init_terrain_level=config.get(
            "max_init_terrain_level",
            terrain_generator.num_rows - 1,
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode=friction_combine_mode,
            restitution_combine_mode=restitution_combine_mode,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        ),
        debug_vis=config.get("debug_vis", False),
    )

    return terrain_cfg
