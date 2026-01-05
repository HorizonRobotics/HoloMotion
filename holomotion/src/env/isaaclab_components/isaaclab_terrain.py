import os

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.height_field import (
    HfDiscreteObstaclesTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)
from loguru import logger


def _convert_range_like_params(params: dict) -> dict:
    """Convert list values for common *range and size keys to tuples for IsaacLab config classes."""
    converted = {}
    for key, value in params.items():
        if isinstance(value, list) and (
            key.endswith("_range") or key in ("size", "difficulty_range")
        ):
            converted[key] = tuple(value)
        else:
            converted[key] = value
    return converted


class RandomSpawnTerrainImporter(TerrainImporter):
    """Terrain importer that spawns robots randomly within each sub-terrain."""

    def _compute_env_origins_curriculum(
        self, num_envs: int, origins: torch.Tensor
    ) -> torch.Tensor:
        """Compute environment origins with random (x, y) positions within each sub-terrain.

        This overrides the default curriculum-based distribution to add random offsets
        within each sub-terrain's bounds.

        Args:
            num_envs: Number of environments.
            origins: Terrain origins tensor of shape (num_rows, num_cols, 3).

        Returns:
            Environment origins tensor of shape (num_envs, 3).
        """
        num_rows, num_cols = origins.shape[:2]

        # Get sub-terrain size from terrain generator config
        if self.cfg.terrain_generator is None:
            raise ValueError(
                "terrain_generator config is required for random spawning"
            )
        sub_terrain_size = self.cfg.terrain_generator.size
        terrain_width, terrain_length = (
            sub_terrain_size[0],
            sub_terrain_size[1],
        )

        # Maximum initial level possible for the terrains
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)

        # Store maximum terrain level possible
        self.max_terrain_level = num_rows

        # Use default curriculum-based assignment
        self.terrain_levels = torch.randint(
            0, max_init_level + 1, (num_envs,), device=self.device
        )
        self.terrain_types = torch.div(
            torch.arange(num_envs, device=self.device),
            (num_envs / num_cols),
            rounding_mode="floor",
        ).to(torch.long)

        # Create environment origins tensor starting from terrain origins
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]

        # Add random (x, y) offsets within each sub-terrain's bounds
        # Offset range: [-size/2, size/2] for both x and y
        x_offsets = torch.empty(num_envs, device=self.device).uniform_(
            -terrain_width / 2, terrain_width / 2
        )
        y_offsets = torch.empty(num_envs, device=self.device).uniform_(
            -terrain_length / 2, terrain_length / 2
        )

        env_origins[:, 0] += x_offsets
        env_origins[:, 1] += y_offsets

        # Store terrain size for use in update_env_origins
        self._terrain_width = terrain_width
        self._terrain_length = terrain_length

        return env_origins

    def update_env_origins(
        self,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ):
        """Update environment origins with random positions when terrain levels change."""
        # Check if grid-like spawning
        if self.terrain_origins is None:
            return

        # Update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        # The minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(
                self.terrain_levels[env_ids], self.max_terrain_level
            ),
            torch.clip(self.terrain_levels[env_ids], 0),
        )

        # Update the env origins with terrain origins
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

        # Add random (x, y) offsets within each sub-terrain's bounds
        if hasattr(self, "_terrain_width") and hasattr(
            self, "_terrain_length"
        ):
            num_updated = len(env_ids)
            x_offsets = torch.empty(num_updated, device=self.device).uniform_(
                -self._terrain_width / 2, self._terrain_width / 2
            )
            y_offsets = torch.empty(num_updated, device=self.device).uniform_(
                -self._terrain_length / 2, self._terrain_length / 2
            )

            self.env_origins[env_ids, 0] += x_offsets
            self.env_origins[env_ids, 1] += y_offsets


def build_terrain_config(
    config: dict, scene_env_spacing: float = None
) -> TerrainImporterCfg:
    """Build terrain configuration.

    Preferred usage in Holomotion is via the IsaacLab terrain generator API with
    height-field sub-terrains fully specified from Hydra configs.

    For backward compatibility only, two legacy modes are still supported:

    * ``terrain_type=\"plane\"``: simple infinite plane using Isaac Sim's grid.
    * ``terrain_type=\"usd\"``: load terrain from a local USD file.

    All paths are offline by construction. Visual materials must use local data:

    * ``visual_material.type=\"color\"`` maps to :class:`PreviewSurfaceCfg`
      with ``diffuse_color``, ``metallic`` and ``roughness``.
    * ``visual_material.type=\"mdl\"`` is accepted only for local MDL files and
      never uses NVIDIA Nucleus. When paths are invalid, a neutral color
      material is used instead.

    Args:
        config: Terrain configuration dictionary with fields:

            * ``terrain_type``: ``\"generator\"`` (preferred), ``\"plane\"`` or
              ``\"usd\"`` (legacy).
            * ``generator`` (required when ``terrain_type=\"generator\"``):
              high-level :class:`TerrainGeneratorCfg` parameters such as
              ``num_rows``, ``num_cols``, ``size``, ``border_width``,
              ``horizontal_scale``, ``vertical_scale``, ``slope_threshold``,
              ``difficulty_range``, ``color_scheme``.
            * ``height_field`` (required when ``terrain_type=\"generator\"``):
              height-field sub-terrain configuration with:

              - ``type``: ``\"random_uniform\"`` or ``\"discrete_obstacles\"``.
              - Remaining keys are forwarded to the corresponding
                :class:`HfRandomUniformTerrainCfg` or
                :class:`HfDiscreteObstaclesTerrainCfg`.
            * ``random_spawn`` (optional): if True, spawns robots at random
              (x, y) positions within each sub-terrain's bounds.
            * ``visual_material`` (optional): offline visual material config.
            * ``static_friction``, ``dynamic_friction``, ``restitution``, etc.

        scene_env_spacing: Environment spacing from scene config (used only when
            ``terrain_type=\"plane\"`` is selected).

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

    terrain_type = config.get("terrain_type", "generator")

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

    if terrain_type != "generator":
        raise ValueError(
            f"Unsupported terrain_type '{terrain_type}'. Expected 'generator', 'plane', or 'usd'."
        )

    generator_cfg_dict = config.get("generator")
    if generator_cfg_dict is None:
        raise ValueError(
            "When 'terrain_type' is 'generator', a 'generator' dict must be provided in terrain config."
        )

    # Optional new path: multiple sub-terrains defined under generator.sub_terrains.
    sub_terrains_cfg_dict = generator_cfg_dict.get("sub_terrains")
    sub_terrains_cfg = None

    if sub_terrains_cfg_dict is not None:
        if not isinstance(sub_terrains_cfg_dict, dict):
            raise ValueError(
                "Expected 'generator.sub_terrains' to be a mapping from names to sub-terrain configs."
            )
        sub_terrains_cfg = {}
        for sub_name, sub_cfg_dict in sub_terrains_cfg_dict.items():
            if not isinstance(sub_cfg_dict, dict):
                raise ValueError(
                    f"Sub-terrain '{sub_name}' must be a dictionary with at least a 'type' field."
                )
            sub_type = sub_cfg_dict.get("type", "random_uniform")
            sub_proportion = sub_cfg_dict.get("proportion", 1.0)
            sub_params_raw = {
                key: value
                for key, value in sub_cfg_dict.items()
                if key not in ("type", "proportion")
            }
            sub_params = _convert_range_like_params(sub_params_raw)

            if sub_type == "random_uniform":
                hf_cfg = HfRandomUniformTerrainCfg(
                    proportion=sub_proportion, **sub_params
                )
            elif sub_type == "discrete_obstacles":
                hf_cfg = HfDiscreteObstaclesTerrainCfg(
                    proportion=sub_proportion, **sub_params
                )
            elif sub_type == "pyramid_sloped":
                hf_cfg = HfPyramidSlopedTerrainCfg(
                    proportion=sub_proportion, **sub_params
                )
            else:
                raise ValueError(
                    f"Unknown sub_terrains['{sub_name}'].type '{sub_type}'. "
                    "Expected 'random_uniform' or 'discrete_obstacles'."
                )
            sub_terrains_cfg[sub_name] = hf_cfg

    # Deprecated path: single height_field block at top-level.
    if sub_terrains_cfg is None:
        height_field_cfg_dict = config.get("height_field")
        if height_field_cfg_dict is None:
            raise ValueError(
                "When 'terrain_type' is 'generator', either 'generator.sub_terrains' "
                "or a 'height_field' dict must be provided in terrain config."
            )

        logger.warning(
            "Terrain config is using deprecated 'height_field' key. "
            "Please migrate to 'generator.sub_terrains' for multi-sub-terrain support."
        )

        hf_type = height_field_cfg_dict.get("type", "random_uniform")
        hf_params_raw = {
            key: value
            for key, value in height_field_cfg_dict.items()
            if key != "type"
        }
        hf_params = _convert_range_like_params(hf_params_raw)

        if hf_type == "random_uniform":
            height_field_cfg = HfRandomUniformTerrainCfg(**hf_params)
        elif hf_type == "discrete_obstacles":
            height_field_cfg = HfDiscreteObstaclesTerrainCfg(**hf_params)
        else:
            raise ValueError(
                f"Unknown height_field.type '{hf_type}'. "
                "Expected 'random_uniform' or 'discrete_obstacles'."
            )
        sub_terrains_cfg = {"main": height_field_cfg}

    # Build TerrainGeneratorCfg from Hydra config.
    generator_params = _convert_range_like_params(
        {
            key: value
            for key, value in generator_cfg_dict.items()
            if key != "sub_terrains"
        }
    )
    terrain_generator = terrain_gen.TerrainGeneratorCfg(
        **{
            key: value
            for key, value in generator_params.items()
            if key
            in (
                "size",
                "border_width",
                "border_height",
                "num_rows",
                "num_cols",
                "horizontal_scale",
                "vertical_scale",
                "slope_threshold",
                "difficulty_range",
                "color_scheme",
                "curriculum",
                "seed",
                "use_cache",
                "cache_dir",
            )
        },
        sub_terrains=sub_terrains_cfg,
    )

    # Configure visual material for offline use
    visual_material = None
    if "visual_material" in config:
        visual_material_dict = config["visual_material"]
        material_type = visual_material_dict.get("type", "color")

        if material_type == "color":
            # Use PreviewSurfaceCfg with diffuse_color (no internet needed)
            diffuse_color_raw = visual_material_dict.get(
                "diffuse_color", (0.8, 0.8, 0.8)
            )
            # Convert list to tuple if needed (YAML loads lists as Python lists)
            # Ensure it's a tuple of floats as required by PreviewSurfaceCfg
            if isinstance(diffuse_color_raw, list):
                diffuse_color = tuple(float(x) for x in diffuse_color_raw)
            elif isinstance(diffuse_color_raw, tuple):
                diffuse_color = tuple(float(x) for x in diffuse_color_raw)
            else:
                diffuse_color = diffuse_color_raw
            metallic = float(visual_material_dict.get("metallic", 0.0))
            roughness = float(visual_material_dict.get("roughness", 0.5))
            visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=diffuse_color,
                metallic=metallic,
                roughness=roughness,
            )
        elif material_type == "none":
            # No visual material, rely on vertex colors (e.g. from height map)
            visual_material = None
        elif material_type == "mdl":
            # Use MdlFileCfg with local mdl_path
            mdl_path = visual_material_dict.get("mdl_path")
            if mdl_path is None:
                logger.warning(
                    "visual_material type is 'mdl' but no mdl_path specified. "
                    "Falling back to color material to avoid internet requirements."
                )
                visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.5, 0.5)
                )
            else:
                # Resolve relative paths
                if not os.path.isabs(mdl_path):
                    if os.path.exists(mdl_path):
                        resolved_mdl_path = os.path.abspath(mdl_path)
                    else:
                        workspace_root = os.path.abspath(
                            os.path.join(
                                os.path.dirname(__file__), "../../../.."
                            )
                        )
                        resolved_mdl_path = os.path.join(
                            workspace_root, mdl_path
                        )
                else:
                    resolved_mdl_path = mdl_path

                if os.path.exists(resolved_mdl_path):
                    visual_material = sim_utils.MdlFileCfg(
                        mdl_path=resolved_mdl_path
                    )
                else:
                    logger.warning(
                        f"MDL file not found at {resolved_mdl_path}. "
                        "Falling back to color material to avoid internet requirements."
                    )
                    visual_material = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.5, 0.5, 0.5)
                    )
        else:
            logger.warning(
                f"Unknown visual_material type: {material_type}. "
                "Using default color material."
            )
            visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5)
            )

    # Configure random spawning within sub-terrains if requested
    random_spawn = config.get("random_spawn", False)
    terrain_importer_class = (
        RandomSpawnTerrainImporter if random_spawn else TerrainImporter
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
        visual_material=visual_material,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode=friction_combine_mode,
            restitution_combine_mode=restitution_combine_mode,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        ),
        debug_vis=config.get("debug_vis", False),
        class_type=terrain_importer_class,
    )

    return terrain_cfg
