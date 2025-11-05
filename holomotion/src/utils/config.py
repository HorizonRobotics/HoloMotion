# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from loguru import logger
from omegaconf import OmegaConf


def setup_hydra_resolvers():
    """Set up custom resolvers for OmegaConf.

    This function registers a set of custom resolvers with OmegaConf to allow
    for more dynamic and flexible configurations within Hydra. These resolvers
    enable performing calculations, conditional logic, and other operations
    directly in the YAML configuration files. For example,
    you can use `${sqrt:4}` to get `2.0`.

    The registered resolvers include:
    - `eval`: Evaluates a Python expression.
    - `if`: Conditional logic (if-else).
    - `eq`: Case-insensitive string comparison.
    - `sqrt`: Calculates the square root.
    - `sum`: Sums a list of numbers.
    - `ceil`: Computes the ceiling of a number.
    - `int`: Casts a value to an integer.
    - `len`: Returns the length of a list or string.
    - `sum_list`: Sums a list of numbers.
    """
    try:
        OmegaConf.register_new_resolver("eval", eval)
        OmegaConf.register_new_resolver(
            "if", lambda pred, a, b: a if pred else b
        )
        OmegaConf.register_new_resolver(
            "eq", lambda x, y: x.lower() == y.lower()
        )
        OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
        OmegaConf.register_new_resolver("sum", lambda x: sum(x))
        OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
        OmegaConf.register_new_resolver("int", lambda x: int(x))
        OmegaConf.register_new_resolver("len", lambda x: len(x))
        OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
    except Exception as e:
        logger.warning(f"Warning: Some resolvers already registered: {e}")


def compile_config(
    config: OmegaConf,
    accelerator: Accelerator = None,
    eval: bool = False,
) -> None:
    """Compile the configuration.

    Args:
        config: Unresolved configuration.
        accelerator: Accelerator instance.

    Returns:
        Compiled configuration.

    """
    setup_hydra_resolvers()
    config = copy.deepcopy(config)
    config = compile_config_hf_accelerate(config, accelerator)
    config = compile_config_directories(config, eval)
    config = compile_config_obs(config)
    config = compile_config_devices(config)
    return config


def compile_config_hf_accelerate(
    config,
    accelerator: Accelerator = None,
) -> None:
    """Compile the configuration for HF Accelerate.

    Args:
        config: Configuration.
        accelerator: Accelerator instance.

    Returns:
        Compiled configuration.

    """
    if accelerator is not None:
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        process_idx = accelerator.process_index
        total_processes = accelerator.num_processes
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        process_idx = 0
        total_processes = 1

    config.process_id = process_idx
    config.num_processes = total_processes
    config.main_process = is_main_process

    if hasattr(config, "device"):
        config.device = str(device)

    logger.info(f"Using device: {device}")
    if is_main_process:
        logger.info(f"Process {process_idx} on device: {device}")

    return config


def compile_config_devices(config):
    """Propagate device and process metadata into the environment configuration."""
    config = copy.deepcopy(config)
    if hasattr(config, "device"):
        device_str = str(config.device)
    else:
        device_str = str(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    world_size = getattr(config, "num_processes", 1)
    process_rank = getattr(config, "process_id", 0)
    is_main_process = getattr(config, "main_process", True)

    if hasattr(config, "env") and hasattr(config.env, "config"):
        env_cfg = config.env.config
        env_cfg_struct = OmegaConf.is_struct(env_cfg)
        OmegaConf.set_struct(env_cfg, False)
        env_cfg.num_processes = world_size
        env_cfg.process_id = process_rank
        env_cfg.main_process = is_main_process
        env_cfg.simulation_device = device_str
        for key in [
            "sim_device",
            "rl_device",
            "compute_device",
            "physx_device",
        ]:
            setattr(env_cfg, key, device_str)
        if hasattr(env_cfg, "simulation"):
            for sim_key in ["device", "compute_device", "rl_device"]:
                sim_cfg = env_cfg.simulation
                sim_struct = OmegaConf.is_struct(sim_cfg)
                OmegaConf.set_struct(sim_cfg, False)
                setattr(sim_cfg, sim_key, device_str)
                OmegaConf.set_struct(sim_cfg, sim_struct)
        OmegaConf.set_struct(env_cfg, env_cfg_struct)

    return config


def compile_config_directories(config, eval: bool = False) -> None:
    """Compile the configuration for folders.

    Args:
        config: Configuration.

    Returns:
        Compiled configuration.

    """
    if eval:
        return config
    config = copy.deepcopy(config)
    experiment_save_dir = Path(config.experiment_dir)
    experiment_save_dir.mkdir(exist_ok=True, parents=True)
    config.experiment_save_dir = str(experiment_save_dir)
    config.env.config.save_rendering_dir = str(
        Path(config.experiment_dir) / "renderings_training"
    )
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    if config.main_process:
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
    return config


def compile_config_obs(config) -> None:
    """Build the final observation dimension dictionary for actor critic.

    Args:
        config: Configuration.

    Returns:
        Compiled configuration.

    """
    config = copy.deepcopy(config)

    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary

    each_dict_obs_dims = {
        k: v for d in config.env.config.obs.obs_dims for k, v in d.items()
    }
    config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {each_dict_obs_dims}")
    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        for _key, _num in aux_config.items():
            if _key not in config.env.config.obs.obs_dims.keys():
                logger.warning(f"{_key} not in obs_dims")
            assert _key in config.env.config.obs.obs_dims.keys()
            auxiliary_obs_dims[aux_obs_key] += (
                config.env.config.obs.obs_dims[_key] * _num
            )
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"):
                key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys():
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(
                    f"{obs_key}: {key} has dim: "
                    f"{config.env.config.obs.obs_dims[key]}"
                )
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(
                    f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}"
                )
    config.robot.algo_obs_dim_dict = obs_dim_dict
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")

    return config
