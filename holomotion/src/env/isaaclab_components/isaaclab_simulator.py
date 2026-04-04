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


from isaaclab.sim import SimulationCfg, PhysxCfg


def build_simulator_config(sim_config_dict: dict) -> SimulationCfg:
    """Build simulation configuration from config dictionary."""
    policy_freq = sim_config_dict.get("policy_freq", 50)
    sim_freq = sim_config_dict.get("sim_freq", 200)
    decimation = int(sim_freq / policy_freq)
    dt = 1.0 / sim_freq
    device = sim_config_dict.get("device", "cuda")

    # PhysX configuration
    physx_config = sim_config_dict.get("physx", {})
    physx = PhysxCfg(
        bounce_threshold_velocity=physx_config.get(
            "bounce_threshold_velocity", 0.2
        ),
        gpu_max_rigid_patch_count=physx_config.get(
            "gpu_max_rigid_patch_count", int(10 * 2**15)
        ),
    )

    return SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physx=physx,
        device=device,
    )
