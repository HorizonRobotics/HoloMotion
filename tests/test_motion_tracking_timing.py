import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_motion_tracking_command.py"
)


class _DummyConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _install_fake_motion_command_deps(monkeypatch):
    isaaclab_mdp = ModuleType("isaaclab.envs.mdp")
    isaaclab_sim = ModuleType("isaaclab.sim")
    isaaclab_sim.PreviewSurfaceCfg = _DummyConfig
    isaaclab_sim.PhysxCfg = _DummyConfig
    isaaclab_sim.SimulationCfg = _DummyConfig
    isaaclab_math = ModuleType("isaaclab.utils.math")
    isaaclab_math.quat_apply_inverse = lambda quat, vec: vec
    isaaclab_math.quat_apply = lambda quat, vec: vec
    isaaclab_math.yaw_quat = lambda quat: quat
    isaaclab_math.quat_inv = lambda quat: quat
    isaaclab_math.quat_mul = lambda lhs, rhs: lhs
    isaaclab_math.sample_uniform = (
        lambda low, high, shape, device=None: torch.zeros(
            *shape, device=device
        )
    )

    isaaclab_actuators = ModuleType("isaaclab.actuators")
    isaaclab_actuators.ImplicitActuatorCfg = _DummyConfig

    isaaclab_assets = ModuleType("isaaclab.assets")
    isaaclab_assets.Articulation = object
    isaaclab_assets.ArticulationCfg = _DummyConfig
    isaaclab_assets.AssetBaseCfg = _DummyConfig

    isaaclab_envs = ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedRLEnv = object
    isaaclab_envs.ManagerBasedRLEnvCfg = _DummyConfig
    isaaclab_envs.ViewerCfg = _DummyConfig

    isaaclab_envs_mdp_actions = ModuleType("isaaclab.envs.mdp.actions")
    isaaclab_envs_mdp_actions.JointEffortActionCfg = _DummyConfig

    isaaclab_managers = ModuleType("isaaclab.managers")
    isaaclab_managers.ActionTermCfg = _DummyConfig
    isaaclab_managers.CommandTerm = object
    isaaclab_managers.CommandTermCfg = _DummyConfig
    isaaclab_managers.EventTermCfg = _DummyConfig
    isaaclab_managers.ObservationGroupCfg = _DummyConfig
    isaaclab_managers.ObservationTermCfg = _DummyConfig
    isaaclab_managers.RewardTermCfg = _DummyConfig
    isaaclab_managers.TerminationTermCfg = _DummyConfig

    isaaclab_markers = ModuleType("isaaclab.markers")
    isaaclab_markers.VisualizationMarkers = _DummyConfig
    isaaclab_markers.VisualizationMarkersCfg = _DummyConfig

    isaaclab_markers_config = ModuleType("isaaclab.markers.config")
    isaaclab_markers_config.SPHERE_MARKER_CFG = SimpleNamespace(
        replace=lambda **kwargs: SimpleNamespace(
            markers={"sphere": SimpleNamespace(radius=None)},
            **kwargs,
        )
    )

    isaaclab_scene = ModuleType("isaaclab.scene")
    isaaclab_scene.InteractiveSceneCfg = _DummyConfig

    isaaclab_sensors = ModuleType("isaaclab.sensors")
    isaaclab_sensors.ContactSensorCfg = _DummyConfig
    isaaclab_sensors.RayCasterCfg = _DummyConfig
    isaaclab_sensors.patterns = _DummyConfig

    isaaclab_terrains = ModuleType("isaaclab.terrains")
    isaaclab_terrains.TerrainImporterCfg = _DummyConfig

    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls

    isaaclab_noise = ModuleType("isaaclab.utils.noise")
    isaaclab_noise.AdditiveUniformNoiseCfg = _DummyConfig

    h5_dataloader = ModuleType("holomotion.src.training.h5_dataloader")
    h5_dataloader.Hdf5MotionDataset = object
    h5_dataloader.Hdf5RootDofDataset = object
    h5_dataloader.MotionClipBatchCache = object
    h5_dataloader.build_motion_datasets_from_cfg = lambda *args, **kwargs: None

    rotations = ModuleType("holomotion.src.utils.isaac_utils.rotations")
    rotations.calc_heading_quat_inv = lambda *args, **kwargs: None
    rotations.get_euler_xyz = lambda *args, **kwargs: None
    rotations.my_quat_rotate = lambda *args, **kwargs: None
    rotations.quat_inverse = lambda *args, **kwargs: None
    rotations.quat_mul = lambda *args, **kwargs: None
    rotations.quat_rotate = lambda *args, **kwargs: None
    rotations.quat_rotate_inverse = lambda *args, **kwargs: None
    rotations.quaternion_to_matrix = lambda *args, **kwargs: None
    rotations.wrap_to_pi = lambda *args, **kwargs: None
    rotations.wxyz_to_xyzw = lambda x: x
    rotations.xyzw_to_wxyz = lambda x: x

    reference_prefix = ModuleType("holomotion.src.utils.reference_prefix")
    reference_prefix.resolve_reference_tensor_key = (
        lambda batch_tensors, base_key, prefix="ref_": f"{prefix}{base_key}"
    )

    omegaconf = ModuleType("omegaconf")
    omegaconf.OmegaConf = SimpleNamespace(
        to_container=lambda value, resolve=True: value
    )

    loguru = ModuleType("loguru")
    loguru.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    tqdm = ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, *args, **kwargs: iterable

    scipy = ModuleType("scipy")
    scipy_spatial = ModuleType("scipy.spatial")
    scipy_transform = ModuleType("scipy.spatial.transform")
    scipy_transform.Rotation = object

    for name, module in {
        "isaaclab.envs.mdp": isaaclab_mdp,
        "isaaclab.sim": isaaclab_sim,
        "isaaclab.utils.math": isaaclab_math,
        "isaaclab.actuators": isaaclab_actuators,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.envs.mdp.actions": isaaclab_envs_mdp_actions,
        "isaaclab.managers": isaaclab_managers,
        "isaaclab.markers": isaaclab_markers,
        "isaaclab.markers.config": isaaclab_markers_config,
        "isaaclab.scene": isaaclab_scene,
        "isaaclab.sensors": isaaclab_sensors,
        "isaaclab.terrains": isaaclab_terrains,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.noise": isaaclab_noise,
        "holomotion.src.training.h5_dataloader": h5_dataloader,
        "holomotion.src.utils.isaac_utils.rotations": rotations,
        "holomotion.src.utils.reference_prefix": reference_prefix,
        "omegaconf": omegaconf,
        "loguru": loguru,
        "tqdm": tqdm,
        "scipy": scipy,
        "scipy.spatial": scipy_spatial,
        "scipy.spatial.transform": scipy_transform,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_motion_command_module(monkeypatch):
    _install_fake_motion_command_deps(monkeypatch)
    module_name = "_test_motion_tracking_timing"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_immediate_next_reference_getters_use_slot_one(monkeypatch):
    module = _load_motion_command_module(monkeypatch)
    command = module.RefMotionCommand.__new__(module.RefMotionCommand)
    command.urdf2sim_dof_idx = torch.tensor([1, 0], dtype=torch.long)
    command.urdf2sim_body_idx = torch.tensor([1, 0], dtype=torch.long)
    command._env_origins = torch.tensor(
        [[10.0, 20.0, 30.0]], dtype=torch.float32
    )
    base_tensors = {
        "ref_dof_pos": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float32
        ),
        "ref_root_pos": torch.tensor(
            [[[0.0, 1.0, 2.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]],
            dtype=torch.float32,
        ),
        "ref_body_vel": torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
                    [[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
                ]
            ],
            dtype=torch.float32,
        ),
    }
    command._get_ref_state_array = (
        lambda base_key, prefix="ref_": base_tensors[f"{prefix}{base_key}"]
    )

    dof_pos = command.get_ref_motion_dof_pos_immediate_next()
    root_pos = command.get_ref_motion_root_global_pos_immediate_next()
    body_lin_vel = (
        command.get_ref_motion_bodylink_global_lin_vel_immediate_next()
    )

    assert torch.allclose(dof_pos, torch.tensor([[4.0, 3.0]]))
    assert torch.allclose(root_pos, torch.tensor([[17.0, 28.0, 39.0]]))
    assert torch.allclose(
        body_lin_vel,
        torch.tensor([[[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]]),
    )


def test_update_command_skips_just_reset_envs(monkeypatch):
    module = _load_motion_command_module(monkeypatch)
    command = module.RefMotionCommand.__new__(module.RefMotionCommand)
    command.device = torch.device("cpu")
    command.num_envs = 3
    command._frame_indices = torch.tensor([10, 20, 30], dtype=torch.long)
    command._swap_step_counter = 0
    command._swap_pending = False
    command._motion_cache = SimpleNamespace(swap_interval_steps=100)
    command._env = SimpleNamespace(
        episode_length_buf=torch.tensor([5, 0, 2], dtype=torch.long)
    )
    command._filter_env_ids_for_motion_task = lambda env_ids: env_ids
    command._resample_when_motion_end_cache = lambda: None
    command._update_ref_motion_state_from_cache = lambda env_ids=None: None

    command._update_command()

    assert torch.equal(command._frame_indices, torch.tensor([11, 20, 31]))
    assert command._swap_step_counter == 1


def test_update_command_resumes_advancing_after_reset_step(monkeypatch):
    module = _load_motion_command_module(monkeypatch)
    command = module.RefMotionCommand.__new__(module.RefMotionCommand)
    command.device = torch.device("cpu")
    command.num_envs = 1
    command._frame_indices = torch.tensor([20], dtype=torch.long)
    command._swap_step_counter = 0
    command._swap_pending = False
    command._motion_cache = SimpleNamespace(swap_interval_steps=100)
    command._env = SimpleNamespace(episode_length_buf=torch.tensor([0]))
    command._filter_env_ids_for_motion_task = lambda env_ids: env_ids
    command._resample_when_motion_end_cache = lambda: None
    command._update_ref_motion_state_from_cache = lambda env_ids=None: None

    command._update_command()
    assert torch.equal(command._frame_indices, torch.tensor([20]))

    command._env.episode_length_buf = torch.tensor([1])
    command._update_command()
    assert torch.equal(command._frame_indices, torch.tensor([21]))


def test_mpjpe_metrics_use_immediate_next_reference(monkeypatch):
    module = _load_motion_command_module(monkeypatch)
    command = module.RefMotionCommand.__new__(module.RefMotionCommand)
    command.device = torch.device("cpu")
    command.num_envs = 1
    command.metrics = {}
    command.arm_dof_indices = [0]
    command.torso_dof_indices = [1]
    command.leg_dof_indices = [2]
    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        )
    )
    command.get_ref_motion_dof_pos_cur = lambda prefix="ref_": (
        _ for _ in ()
    ).throw(AssertionError("current reference should not be used"))
    command.get_ref_motion_dof_pos_immediate_next = (
        lambda prefix="ref_": torch.tensor(
            [[0.1, 0.2, 0.3]], dtype=torch.float32
        )
    )

    command._update_mpjpe_metrics()

    assert torch.allclose(
        command.metrics["Task/MPJPE_WholeBody"], torch.zeros(1)
    )


def test_mpkpe_metrics_use_immediate_next_reference(monkeypatch):
    module = _load_motion_command_module(monkeypatch)
    command = module.RefMotionCommand.__new__(module.RefMotionCommand)
    command.device = torch.device("cpu")
    command.num_envs = 1
    command.metrics = {}
    command.arm_body_indices = [0]
    command.torso_body_indices = [1]
    command.leg_body_indices = [2]
    command.robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=torch.tensor(
                [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
                dtype=torch.float32,
            )
        )
    )
    command.get_ref_motion_bodylink_global_pos_cur = lambda prefix="ref_": (
        _ for _ in ()
    ).throw(AssertionError("current reference should not be used"))
    command.get_ref_motion_bodylink_global_pos_immediate_next = (
        lambda prefix="ref_": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]],
            dtype=torch.float32,
        )
    )

    command._update_mpkpe_metrics()

    assert torch.allclose(
        command.metrics["Task/MPKPE_WholeBody"], torch.zeros(1)
    )
