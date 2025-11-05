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
