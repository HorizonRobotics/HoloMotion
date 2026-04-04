import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

OBSERVATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_observation.py"
)


class _DummyConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Scene(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4, dtype=torch.float32)
    quat[..., 0] = 1.0
    return quat


def _load_observation_module(monkeypatch):
    isaaclab = ModuleType("isaaclab")
    isaaclab_mdp = ModuleType("isaaclab.envs.mdp")
    isaaclab_math = ModuleType("isaaclab.utils.math")
    isaaclab_math.quat_apply = lambda quat, vec: vec
    isaaclab_math.quat_apply_inverse = lambda quat, vec: vec
    isaaclab_math.quat_inv = lambda quat: quat
    isaaclab_math.matrix_from_quat = lambda quat: torch.zeros(
        *quat.shape[:-1], 3, 3, dtype=quat.dtype, device=quat.device
    )
    isaaclab_math.subtract_frame_transforms = lambda t01, q01, t02, q02: (
        t02 - t01,
        q02,
    )
    isaaclab_math.__getattr__ = lambda name: (lambda *args, **kwargs: None)
    isaaclab_noise = ModuleType("isaaclab.utils.noise")
    isaaclab_noise.__getattr__ = lambda name: _DummyConfig

    isaaclab_envs = ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedRLEnv = object
    isaaclab_envs.ManagerBasedRLEnvCfg = _DummyConfig
    isaaclab_envs.ViewerCfg = _DummyConfig
    isaaclab_sim = ModuleType("isaaclab.sim")
    isaaclab_sim.__getattr__ = lambda name: _DummyConfig
    isaaclab_actuators = ModuleType("isaaclab.actuators")
    isaaclab_actuators.ImplicitActuatorCfg = _DummyConfig
    isaaclab_assets = ModuleType("isaaclab.assets")
    isaaclab_assets.Articulation = object
    isaaclab_assets.ArticulationCfg = _DummyConfig
    isaaclab_assets.AssetBaseCfg = _DummyConfig
    isaaclab_managers = ModuleType("isaaclab.managers")
    isaaclab_managers.__getattr__ = lambda name: _DummyConfig
    isaaclab_markers = ModuleType("isaaclab.markers")
    isaaclab_markers.VisualizationMarkers = _DummyConfig
    isaaclab_markers.VisualizationMarkersCfg = _DummyConfig
    isaaclab_markers_config = ModuleType("isaaclab.markers.config")
    isaaclab_markers_config.FRAME_MARKER_CFG = _DummyConfig
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

    omegaconf = ModuleType("omegaconf")
    omegaconf.DictConfig = dict
    omegaconf.ListConfig = list
    omegaconf.OmegaConf = SimpleNamespace(
        to_container=lambda value, resolve=True: value
    )

    fake_utils_module = ModuleType(
        "holomotion.src.env.isaaclab_components.isaaclab_utils"
    )
    fake_utils_module.resolve_holo_config = lambda value: value

    isaaclab.envs = isaaclab_envs
    isaaclab.sim = isaaclab_sim
    isaaclab.actuators = isaaclab_actuators
    isaaclab.assets = isaaclab_assets
    isaaclab.managers = isaaclab_managers
    isaaclab.markers = isaaclab_markers
    isaaclab.scene = isaaclab_scene
    isaaclab.sensors = isaaclab_sensors
    isaaclab.terrains = isaaclab_terrains
    isaaclab.utils = isaaclab_utils
    isaaclab_envs.mdp = isaaclab_mdp
    isaaclab_utils.math = isaaclab_math
    isaaclab_utils.noise = isaaclab_noise

    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.envs.mdp": isaaclab_mdp,
        "isaaclab.utils.math": isaaclab_math,
        "isaaclab.utils.noise": isaaclab_noise,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.sim": isaaclab_sim,
        "isaaclab.actuators": isaaclab_actuators,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.managers": isaaclab_managers,
        "isaaclab.markers": isaaclab_markers,
        "isaaclab.markers.config": isaaclab_markers_config,
        "isaaclab.scene": isaaclab_scene,
        "isaaclab.sensors": isaaclab_sensors,
        "isaaclab.terrains": isaaclab_terrains,
        "isaaclab.utils": isaaclab_utils,
        "omegaconf": omegaconf,
        (
            "holomotion.src.env.isaaclab_components.isaaclab_utils"
        ): fake_utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_observation_frames"
    spec = importlib.util.spec_from_file_location(
        module_name, OBSERVATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_ref_future_observations_can_limit_num_frames(monkeypatch):
    observation = _load_observation_module(monkeypatch)

    class _Command:
        def get_ref_motion_dof_pos_fut(self, prefix="ref_"):
            return torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

        def get_ref_motion_dof_vel_fut(self, prefix="ref_"):
            return torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

        def get_ref_motion_gravity_projection_fut(self, prefix="ref_"):
            return torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

        def get_ref_motion_base_linvel_fut(self, prefix="ref_"):
            return torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

        def get_ref_motion_base_angvel_fut(self, prefix="ref_"):
            return torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

        def get_ref_motion_root_global_pos_fut(self, prefix="ref_"):
            pos = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
            pos[..., 2] = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
                dtype=torch.float32,
            )
            return pos

    class _CommandManager:
        def get_term(self, name):
            return _Command()

    env = SimpleNamespace(
        command_manager=_CommandManager(),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
    )

    dof_pos = observation.ObservationFunctions._get_obs_ref_dof_pos_fut(
        env, num_frames=2
    )
    dof_vel = observation.ObservationFunctions._get_obs_ref_dof_vel_fut(
        env, num_frames=2
    )
    gravity = (
        observation.ObservationFunctions._get_obs_ref_gravity_projection_fut(
            env, num_frames=2
        )
    )
    base_linvel = (
        observation.ObservationFunctions._get_obs_ref_base_linvel_fut(
            env, num_frames=2
        )
    )
    base_angvel = (
        observation.ObservationFunctions._get_obs_ref_base_angvel_fut(
            env, num_frames=2
        )
    )
    root_height = (
        observation.ObservationFunctions._get_obs_ref_root_height_fut(
            env, num_frames=2
        )
    )

    assert dof_pos.shape == (2, 2, 3)
    assert dof_vel.shape == (2, 6)
    assert gravity.shape == (2, 2, 3)
    assert base_linvel.shape == (2, 2, 3)
    assert base_angvel.shape == (2, 2, 3)
    assert root_height.shape == (2, 2, 1)
    torch.testing.assert_close(
        dof_pos,
        torch.tensor(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        dof_vel,
        torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        root_height[..., 0], torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    )


def _make_env():
    env_origins = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    robot_data = SimpleNamespace(
        body_pos_w=torch.tensor(
            [[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]], dtype=torch.float32
        ),
        body_quat_w=_identity_quat(1, 2),
    )
    robot = SimpleNamespace(body_names=["anchor", "target"], data=robot_data)
    command = SimpleNamespace(
        anchor_bodylink_name="anchor",
        get_ref_motion_anchor_bodylink_global_pos_cur=(
            lambda prefix="ref_": torch.tensor([[10.0, 0.0, 0.0]])
        ),
        get_ref_motion_anchor_bodylink_global_rot_wxyz_cur=(
            lambda prefix="ref_": _identity_quat(1)
        ),
    )
    return SimpleNamespace(
        num_envs=1,
        scene=_Scene(env_origins=env_origins, robot=robot),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )


def test_global_robot_bodylink_pos_is_in_environment_frame(monkeypatch):
    observation = _load_observation_module(monkeypatch)
    env = _make_env()

    pos = observation.ObservationFunctions._get_obs_global_robot_bodylink_pos(
        env,
        keybody_names=["target"],
    )

    assert torch.allclose(pos, torch.tensor([[[1.0, 0.0, 0.0]]]))


def test_root_rel_robot_bodylink_pos_uses_consistent_env_frame(monkeypatch):
    observation = _load_observation_module(monkeypatch)
    env = _make_env()
    observation.isaaclab_mdp.root_pos_w = lambda _env: torch.zeros(
        1, 3, dtype=torch.float32
    )
    observation.isaaclab_mdp.root_quat_w = lambda _env: _identity_quat(1)

    pos = (
        observation.ObservationFunctions._get_obs_root_rel_robot_bodylink_pos(
            env,
            keybody_names=["target"],
        )
    )

    assert torch.allclose(pos, torch.tensor([[[1.0, 0.0, 0.0]]]))


def test_global_anchor_pos_diff_uses_environment_frame_consistently(
    monkeypatch,
):
    observation = _load_observation_module(monkeypatch)
    env = _make_env()

    pos_diff = (
        observation.ObservationFunctions._get_obs_global_anchor_pos_diff(env)
    )

    assert torch.allclose(pos_diff, torch.zeros(1, 3))


def test_build_additive_uniform_noise_cfg_supports_optional_z_override(
    monkeypatch,
):
    observation = _load_observation_module(monkeypatch)

    noise = observation._build_noise_cfg(
        {
            "type": "AdditiveUniformNoiseCfg",
            "params": {
                "n_min": -0.1,
                "n_max": 0.1,
                "n_min_z": -0.02,
                "n_max_z": 0.03,
            },
        }
    )

    assert torch.equal(
        noise.kwargs["n_min"], torch.tensor([-0.1, -0.1, -0.02])
    )
    assert torch.equal(noise.kwargs["n_max"], torch.tensor([0.1, 0.1, 0.03]))


def test_build_additive_uniform_noise_cfg_keeps_scalar_bounds_without_z_override(
    monkeypatch,
):
    observation = _load_observation_module(monkeypatch)

    noise = observation._build_noise_cfg(
        {
            "type": "AdditiveUniformNoiseCfg",
            "params": {
                "n_min": -0.1,
                "n_max": 0.1,
            },
        }
    )

    assert noise.kwargs["n_min"] == pytest.approx(-0.1)
    assert noise.kwargs["n_max"] == pytest.approx(0.1)
