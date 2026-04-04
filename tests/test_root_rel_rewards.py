import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

REWARDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_rewards.py"
)
MOTION_TRACKING_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "motion_tracking.py"
)


class _DummyConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if args:
            self.name = args[0]
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not hasattr(self, "params"):
            self.params = {}


class _DummyManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    def reset(self, env_ids=None):
        pass


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4, dtype=torch.float32)
    quat[..., 0] = 1.0
    return quat


def _load_rewards_module(monkeypatch):
    isaaclab_root = ModuleType("isaaclab")
    isaaclab_assets = ModuleType("isaaclab.assets")
    isaaclab_assets.Articulation = object
    isaaclab_envs = ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedRLEnv = object
    isaaclab_mdp = ModuleType("isaaclab.envs.mdp")
    isaaclab_mdp.__getattr__ = lambda name: (lambda *args, **kwargs: None)
    isaaclab_managers = ModuleType("isaaclab.managers")
    isaaclab_managers.ManagerTermBase = _DummyManagerTermBase
    isaaclab_managers.RewardTermCfg = _DummyConfig
    isaaclab_managers.SceneEntityCfg = _DummyConfig
    isaaclab_sensors = ModuleType("isaaclab.sensors")
    isaaclab_sensors.ContactSensor = object
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls
    isaaclab_math = ModuleType("isaaclab.utils.math")
    isaaclab_math.quat_apply = lambda quat, vec: vec
    isaaclab_math.quat_apply_inverse = lambda quat, vec: vec
    isaaclab_math.quat_inv = lambda quat: quat
    isaaclab_math.quat_mul = lambda lhs, rhs: lhs
    isaaclab_math.yaw_quat = lambda quat: quat
    isaaclab_math.quat_error_magnitude = lambda lhs, rhs: torch.linalg.norm(
        lhs - rhs, dim=-1
    )
    isaaclab_math.__getattr__ = lambda name: (lambda *args, **kwargs: None)

    hydra_utils = ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda value, *args, **kwargs: value

    omegaconf = ModuleType("omegaconf")
    omegaconf.DictConfig = dict
    omegaconf.ListConfig = list
    omegaconf.OmegaConf = SimpleNamespace(
        to_container=lambda value, resolve=True: value
    )

    loguru = ModuleType("loguru")
    loguru.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )

    fake_command_module = ModuleType(
        "holomotion.src.env.isaaclab_components."
        "isaaclab_motion_tracking_command"
    )
    fake_command_module.RefMotionCommand = object

    fake_utils_module = ModuleType(
        "holomotion.src.env.isaaclab_components.isaaclab_utils"
    )
    fake_utils_module._get_body_indices = lambda robot, keybody_names: [
        robot.body_names.index(name) for name in keybody_names
    ]
    fake_utils_module._get_dof_indices = lambda robot, key_dofs: []
    fake_utils_module.resolve_holo_config = lambda value: value

    for name, module in {
        "isaaclab": isaaclab_root,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.envs.mdp": isaaclab_mdp,
        "isaaclab.managers": isaaclab_managers,
        "isaaclab.sensors": isaaclab_sensors,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.math": isaaclab_math,
        "hydra.utils": hydra_utils,
        "omegaconf": omegaconf,
        "loguru": loguru,
        (
            "holomotion.src.env.isaaclab_components."
            "isaaclab_motion_tracking_command"
        ): fake_command_module,
        (
            "holomotion.src.env.isaaclab_components.isaaclab_utils"
        ): fake_utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    isaaclab_root.assets = isaaclab_assets
    isaaclab_root.envs = isaaclab_envs
    isaaclab_root.managers = isaaclab_managers
    isaaclab_root.sensors = isaaclab_sensors
    isaaclab_root.utils = isaaclab_utils
    isaaclab_envs.mdp = isaaclab_mdp
    isaaclab_utils.math = isaaclab_math

    module_name = "_test_root_rel_rewards"
    spec = importlib.util.spec_from_file_location(module_name, REWARDS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_motion_tracking_module(monkeypatch):
    class _DummyConfigClass:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    isaaclab_root = ModuleType("isaaclab")
    isaaclab_actuators = ModuleType("isaaclab.actuators")
    isaaclab_actuators.ImplicitActuatorCfg = _DummyConfigClass

    isaaclab_assets = ModuleType("isaaclab.assets")
    isaaclab_assets.Articulation = object

    isaaclab_envs = ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedEnv = object
    isaaclab_envs.ManagerBasedRLEnv = object
    isaaclab_envs.ManagerBasedRLEnvCfg = object
    isaaclab_envs.ViewerCfg = _DummyConfigClass

    isaaclab_envs_mdp = ModuleType("isaaclab.envs.mdp")
    isaaclab_envs_mdp.__getattr__ = lambda name: (lambda *args, **kwargs: None)

    isaaclab_envs_mdp_events = ModuleType("isaaclab.envs.mdp.events")
    isaaclab_envs_mdp_events._randomize_prop_by_op = (
        lambda *args, **kwargs: None
    )

    isaaclab_managers = ModuleType("isaaclab.managers")
    isaaclab_managers.EventTermCfg = _DummyConfigClass
    isaaclab_managers.SceneEntityCfg = _DummyConfig

    isaaclab_sim = ModuleType("isaaclab.sim")
    isaaclab_sim.PhysxCfg = _DummyConfigClass
    isaaclab_sim.SimulationCfg = _DummyConfigClass

    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls

    isaaclab_utils_io = ModuleType("isaaclab.utils.io")
    isaaclab_utils_io.dump_yaml = lambda *args, **kwargs: None

    isaaclab_utils_math = ModuleType("isaaclab.utils.math")
    isaaclab_utils_math.__getattr__ = lambda name: (
        lambda *args, **kwargs: None
    )

    easydict = ModuleType("easydict")
    easydict.EasyDict = lambda value=None: value if value is not None else {}

    omegaconf = ModuleType("omegaconf")
    omegaconf.OmegaConf = SimpleNamespace(
        to_container=lambda value, resolve=True: value
    )

    loguru = ModuleType("loguru")
    loguru.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )

    isaaclab_components = ModuleType("holomotion.src.env.isaaclab_components")
    for name in [
        "ActionsCfg",
        "VelTrack_CommandsCfg",
        "MoTrack_CommandsCfg",
        "EventsCfg",
        "MotionTrackingSceneCfg",
        "ObservationsCfg",
        "RewardsCfg",
        "TerminationsCfg",
        "CurriculumCfg",
    ]:
        setattr(isaaclab_components, name, _DummyConfigClass)
    for name in [
        "build_actions_config",
        "build_motion_tracking_commands_config",
        "build_velocity_commands_config",
        "build_domain_rand_config",
        "build_curriculum_config",
        "build_observations_config",
        "build_rewards_config",
        "build_scene_config",
        "build_terminations_config",
    ]:
        setattr(isaaclab_components, name, lambda *args, **kwargs: None)

    fake_observation_module = ModuleType(
        "holomotion.src.env.isaaclab_components.isaaclab_observation"
    )
    fake_observation_module.ObservationFunctions = object

    fake_utils_module = ModuleType(
        "holomotion.src.env.isaaclab_components.isaaclab_utils"
    )
    fake_utils_module.resolve_holo_config = lambda value: value

    for name, module in {
        "isaaclab": isaaclab_root,
        "isaaclab.actuators": isaaclab_actuators,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.envs.mdp": isaaclab_envs_mdp,
        "isaaclab.envs.mdp.events": isaaclab_envs_mdp_events,
        "isaaclab.managers": isaaclab_managers,
        "isaaclab.sim": isaaclab_sim,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.io": isaaclab_utils_io,
        "isaaclab.utils.math": isaaclab_utils_math,
        "easydict": easydict,
        "omegaconf": omegaconf,
        "loguru": loguru,
        "holomotion.src.env.isaaclab_components": isaaclab_components,
        (
            "holomotion.src.env.isaaclab_components.isaaclab_observation"
        ): fake_observation_module,
        (
            "holomotion.src.env.isaaclab_components.isaaclab_utils"
        ): fake_utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    isaaclab_root.actuators = isaaclab_actuators
    isaaclab_root.assets = isaaclab_assets
    isaaclab_root.envs = isaaclab_envs
    isaaclab_root.managers = isaaclab_managers
    isaaclab_root.sim = isaaclab_sim
    isaaclab_root.utils = isaaclab_utils
    isaaclab_envs.mdp = isaaclab_envs_mdp
    isaaclab_utils.io = isaaclab_utils_io
    isaaclab_utils.math = isaaclab_utils_math

    module_name = "_test_motion_tracking"
    spec = importlib.util.spec_from_file_location(
        module_name, MOTION_TRACKING_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_env():
    env_origins = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    robot_data = SimpleNamespace(
        body_pos_w=torch.tensor(
            [[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]], dtype=torch.float32
        ),
        body_quat_w=_identity_quat(1, 2),
        body_lin_vel_w=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
        ),
        body_ang_vel_w=torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
        ),
    )
    robot = SimpleNamespace(body_names=["anchor", "target"], data=robot_data)
    command = SimpleNamespace(
        robot=robot,
        anchor_bodylink_idx=0,
        get_ref_motion_root_global_pos_cur=lambda prefix="ref_": torch.tensor(
            [[10.0, 0.0, 0.0]], dtype=torch.float32
        ),
        get_ref_motion_root_global_pos_immediate_next=(
            lambda prefix="ref_": torch.tensor(
                [[10.0, 0.0, 0.0]], dtype=torch.float32
            )
        ),
        get_ref_motion_root_global_rot_quat_wxyz_cur=(
            lambda prefix="ref_": _identity_quat(1)
        ),
        get_ref_motion_root_global_rot_quat_wxyz_immediate_next=(
            lambda prefix="ref_": _identity_quat(1)
        ),
        get_ref_motion_root_global_lin_vel_cur=(
            lambda prefix="ref_": torch.zeros(1, 3, dtype=torch.float32)
        ),
        get_ref_motion_root_global_lin_vel_immediate_next=(
            lambda prefix="ref_": torch.zeros(1, 3, dtype=torch.float32)
        ),
        get_ref_motion_root_global_ang_vel_cur=(
            lambda prefix="ref_": torch.tensor([[0.0, 0.0, 1.0]])
        ),
        get_ref_motion_root_global_ang_vel_immediate_next=(
            lambda prefix="ref_": torch.tensor([[0.0, 0.0, 1.0]])
        ),
        get_ref_motion_bodylink_global_pos_cur=(
            lambda prefix="ref_": torch.tensor(
                [[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]], dtype=torch.float32
            )
        ),
        get_ref_motion_bodylink_global_pos_immediate_next=(
            lambda prefix="ref_": torch.tensor(
                [[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]], dtype=torch.float32
            )
        ),
        get_ref_motion_bodylink_global_lin_vel_cur=(
            lambda prefix="ref_": torch.tensor(
                [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
            )
        ),
        get_ref_motion_bodylink_global_lin_vel_immediate_next=(
            lambda prefix="ref_": torch.tensor(
                [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32
            )
        ),
        get_ref_motion_bodylink_global_rot_wxyz_immediate_next=(
            lambda prefix="ref_": _identity_quat(1, 2)
        ),
        get_ref_motion_bodylink_global_ang_vel_immediate_next=(
            lambda prefix="ref_": torch.tensor(
                [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
            )
        ),
    )
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene=SimpleNamespace(env_origins=env_origins),
    )


def _make_torque_rate_env(
    applied_torque: torch.Tensor,
    actuators: dict,
    joint_vel: torch.Tensor | None = None,
    joint_vel_limits: torch.Tensor | None = None,
):
    class _Scene(dict):
        pass

    if joint_vel is None:
        joint_vel = torch.zeros_like(applied_torque)
    if joint_vel_limits is None:
        joint_vel_limits = torch.ones_like(applied_torque)

    asset = SimpleNamespace(
        data=SimpleNamespace(
            applied_torque=applied_torque.clone(),
            joint_vel=joint_vel.clone(),
            joint_vel_limits=joint_vel_limits.clone(),
        ),
        actuators=actuators,
    )
    scene = _Scene(robot=asset)
    return SimpleNamespace(
        scene=scene,
        num_envs=applied_torque.shape[0],
        device=applied_torque.device,
        episode_length_buf=torch.zeros(
            applied_torque.shape[0],
            dtype=torch.long,
            device=applied_torque.device,
        ),
    )


def _make_action_acc_env(action: torch.Tensor):
    return SimpleNamespace(
        action_manager=SimpleNamespace(action=action.clone()),
        num_envs=action.shape[0],
        device=action.device,
        episode_length_buf=torch.zeros(
            action.shape[0], dtype=torch.long, device=action.device
        ),
    )


def test_root_rel_keybody_pos_reward_uses_true_root_frame(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_env()
    rewards.isaaclab_mdp.root_pos_w = lambda _env: torch.zeros(
        1, 3, dtype=torch.float32
    )
    rewards.isaaclab_mdp.root_quat_w = lambda _env: _identity_quat(1)

    reward = rewards.root_rel_keybodylink_pos_tracking_l2_exp(
        env,
        std=1.0,
        keybody_names=["target"],
    )

    assert torch.allclose(reward, torch.ones(1))


def test_root_rel_keybody_pos_bydmmc_reward_uses_true_root_frame(
    monkeypatch,
):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_env()
    rewards.isaaclab_mdp.root_pos_w = lambda _env: torch.zeros(
        1, 3, dtype=torch.float32
    )
    rewards.isaaclab_mdp.root_quat_w = lambda _env: _identity_quat(1)

    reward = rewards.root_rel_keybodylink_pos_tracking_l2_exp_bydmmc_style(
        env,
        std=1.0,
        keybody_names=["target"],
    )

    assert torch.allclose(reward, torch.ones(1))


def test_root_rel_keybody_lin_vel_reward_uses_true_root_frame(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_env()
    rewards.isaaclab_mdp.root_pos_w = lambda _env: torch.zeros(
        1, 3, dtype=torch.float32
    )
    rewards.isaaclab_mdp.root_quat_w = lambda _env: _identity_quat(1)
    rewards.isaaclab_mdp.root_lin_vel_w = lambda _env: torch.zeros(
        1, 3, dtype=torch.float32
    )
    rewards.isaaclab_mdp.root_ang_vel_w = lambda _env: torch.tensor(
        [[0.0, 0.0, 1.0]], dtype=torch.float32
    )

    reward = rewards.root_rel_keybodylink_lin_vel_tracking_l2_exp(
        env,
        std=1.0,
        keybody_names=["target"],
    )

    assert torch.allclose(reward, torch.ones(1))


def test_root_pos_xy_tracking_uses_immediate_next_reference(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot_data = SimpleNamespace(
        root_pos_w=torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
    )
    robot = SimpleNamespace(data=robot_data)
    command = SimpleNamespace(
        robot=robot,
        get_ref_motion_root_global_pos_cur=(
            lambda prefix="ref_": (_ for _ in ()).throw(
                AssertionError("current reference should not be used")
            )
        ),
        get_ref_motion_root_global_pos_immediate_next=(
            lambda prefix="ref_": torch.tensor(
                [[1.0, 2.0, 3.0]], dtype=torch.float32
            )
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command)
    )

    reward = rewards.root_pos_xy_tracking_exp(env, std=1.0)

    assert torch.allclose(reward, torch.ones(1))


def test_global_keybody_lin_vel_tracking_uses_immediate_next_reference(
    monkeypatch,
):
    rewards = _load_rewards_module(monkeypatch)
    robot_data = SimpleNamespace(
        body_lin_vel_w=torch.tensor(
            [[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]], dtype=torch.float32
        )
    )
    robot = SimpleNamespace(
        body_names=["anchor", "target"],
        data=robot_data,
    )
    command = SimpleNamespace(
        robot=robot,
        get_ref_motion_bodylink_global_lin_vel_cur=(
            lambda prefix="ref_": (_ for _ in ()).throw(
                AssertionError("current reference should not be used")
            )
        ),
        get_ref_motion_bodylink_global_lin_vel_immediate_next=(
            lambda prefix="ref_": torch.tensor(
                [[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]], dtype=torch.float32
            )
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command)
    )

    reward = rewards.global_keybodylink_lin_vel_tracking_l2_exp(
        env,
        std=1.0,
        keybody_names=["target"],
    )

    assert torch.allclose(reward, torch.ones(1))


def test_normed_torque_rate_matches_selected_joint_math(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_torque_rate_env(
        applied_torque=torch.zeros(2, 3, dtype=torch.float32),
        actuators={
            "all_joints": SimpleNamespace(
                joint_indices=slice(None),
                effort_limit=torch.tensor(
                    [[10.0, 20.0, 40.0], [10.0, 20.0, 40.0]],
                    dtype=torch.float32,
                ),
            )
        },
    )
    term = rewards.normed_torque_rate(_DummyConfig(params={}), env)
    asset_cfg = SimpleNamespace(
        name="robot", joint_ids=torch.tensor([0, 2], dtype=torch.long)
    )

    first = term(env, asset_cfg=asset_cfg)
    assert torch.allclose(first, torch.zeros(2))

    env.episode_length_buf[:] = 1
    env.scene["robot"].data.applied_torque = torch.tensor(
        [[1.0, 9.0, 4.0], [2.0, 7.0, 8.0]],
        dtype=torch.float32,
    )
    reward = term(env, asset_cfg=asset_cfg)

    expected = torch.tensor(
        [
            (1.0 / 10.0) ** 2 + (4.0 / 40.0) ** 2,
            (2.0 / 10.0) ** 2 + (8.0 / 40.0) ** 2,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(reward, expected)


def test_normed_torque_rate_assembles_limits_across_actuator_groups(
    monkeypatch,
):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_torque_rate_env(
        applied_torque=torch.zeros(1, 3, dtype=torch.float32),
        actuators={
            "implicit_group": SimpleNamespace(
                joint_indices=torch.tensor([0, 2], dtype=torch.long),
                effort_limit=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
            ),
            "unitree_group": SimpleNamespace(
                joint_indices=torch.tensor([1], dtype=torch.long),
                effort_limit=torch.tensor([[5.0]], dtype=torch.float32),
            ),
        },
    )
    term = rewards.normed_torque_rate(_DummyConfig(params={}), env)
    asset_cfg = SimpleNamespace(
        name="robot", joint_ids=torch.tensor([0, 1, 2], dtype=torch.long)
    )

    _ = term(env, asset_cfg=asset_cfg)
    env.episode_length_buf[:] = 1
    env.scene["robot"].data.applied_torque = torch.tensor(
        [[1.0, 1.0, 2.0]], dtype=torch.float32
    )
    reward = term(env, asset_cfg=asset_cfg)

    expected = torch.tensor(
        [(1.0 / 10.0) ** 2 + (1.0 / 5.0) ** 2 + (2.0 / 20.0) ** 2],
        dtype=torch.float32,
    )
    assert torch.allclose(reward, expected)


def test_normed_torque_rate_resets_first_step_history(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_torque_rate_env(
        applied_torque=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        ),
        actuators={
            "all_joints": SimpleNamespace(
                joint_indices=slice(None),
                effort_limit=torch.tensor(
                    [[10.0, 10.0], [10.0, 10.0]], dtype=torch.float32
                ),
            )
        },
    )
    term = rewards.normed_torque_rate(_DummyConfig(params={}), env)
    asset_cfg = SimpleNamespace(
        name="robot", joint_ids=torch.tensor([0, 1], dtype=torch.long)
    )

    first = term(env, asset_cfg=asset_cfg)
    assert torch.allclose(first, torch.zeros(2))

    env.episode_length_buf[:] = 1
    env.scene["robot"].data.applied_torque = torch.tensor(
        [[2.0, 4.0], [5.0, 8.0]], dtype=torch.float32
    )
    second = term(env, asset_cfg=asset_cfg)
    assert torch.all(second > 0.0)

    term.reset(env_ids=[0])
    env.scene["robot"].data.applied_torque = torch.tensor(
        [[7.0, 9.0], [6.0, 10.0]], dtype=torch.float32
    )
    after_reset = term(env, asset_cfg=asset_cfg)

    assert torch.isclose(after_reset[0], torch.tensor(0.0))
    assert after_reset[1] > 0.0


def test_normed_torque_rate_reuses_cached_normalization(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    actuator = SimpleNamespace(
        joint_indices=slice(None),
        effort_limit=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
    )
    env = _make_torque_rate_env(
        applied_torque=torch.zeros(1, 2, dtype=torch.float32),
        actuators={"all_joints": actuator},
    )
    term = rewards.normed_torque_rate(_DummyConfig(params={}), env)
    asset_cfg = SimpleNamespace(
        name="robot", joint_ids=torch.tensor([0, 1], dtype=torch.long)
    )

    _ = term(env, asset_cfg=asset_cfg)

    actuator.effort_limit = torch.tensor(
        [[1000.0, 1000.0]], dtype=torch.float32
    )
    env.episode_length_buf[:] = 1
    env.scene["robot"].data.applied_torque = torch.tensor(
        [[2.0, 4.0]], dtype=torch.float32
    )
    reward = term(env, asset_cfg=asset_cfg)

    expected = torch.tensor(
        [(2.0 / 10.0) ** 2 + (4.0 / 20.0) ** 2], dtype=torch.float32
    )
    assert torch.allclose(reward, expected)


def test_normed_positive_work_matches_selected_joint_math(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_torque_rate_env(
        applied_torque=torch.tensor(
            [[2.0, 5.0, 8.0], [3.0, -4.0, 6.0]], dtype=torch.float32
        ),
        joint_vel=torch.tensor(
            [[1.0, -5.0, 2.0], [2.0, 3.0, -2.0]], dtype=torch.float32
        ),
        joint_vel_limits=torch.tensor(
            [[4.0, 10.0, 8.0], [4.0, 10.0, 8.0]], dtype=torch.float32
        ),
        actuators={
            "all_joints": SimpleNamespace(
                joint_indices=slice(None),
                effort_limit=torch.tensor(
                    [[4.0, 10.0, 16.0], [4.0, 10.0, 16.0]],
                    dtype=torch.float32,
                ),
            )
        },
    )
    term = rewards.normed_positive_work(_DummyConfig(params={}), env)
    asset_cfg = SimpleNamespace(
        name="robot", joint_ids=torch.tensor([0, 2], dtype=torch.long)
    )

    reward = term(env, asset_cfg=asset_cfg)

    expected = torch.tensor(
        [
            (2.0 / 4.0) * (1.0 / 4.0) + (8.0 / 16.0) * (2.0 / 8.0),
            (3.0 / 4.0) * (2.0 / 4.0),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(reward, expected)


def test_normed_positive_work_assembles_effort_limits_across_actuators(
    monkeypatch,
):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_torque_rate_env(
        applied_torque=torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32),
        joint_vel=torch.tensor([[5.0, 2.0, -1.0]], dtype=torch.float32),
        joint_vel_limits=torch.tensor([[10.0, 4.0, 8.0]], dtype=torch.float32),
        actuators={
            "implicit_group": SimpleNamespace(
                joint_indices=torch.tensor([0, 2], dtype=torch.long),
                effort_limit=torch.tensor([[4.0, 20.0]], dtype=torch.float32),
            ),
            "unitree_group": SimpleNamespace(
                joint_indices=torch.tensor([1], dtype=torch.long),
                effort_limit=torch.tensor([[6.0]], dtype=torch.float32),
            ),
        },
    )
    term = rewards.normed_positive_work(_DummyConfig(params={}), env)
    asset_cfg = SimpleNamespace(
        name="robot", joint_ids=torch.tensor([0, 1, 2], dtype=torch.long)
    )

    reward = term(env, asset_cfg=asset_cfg)

    expected = torch.tensor(
        [
            (2.0 / 4.0) * (5.0 / 10.0) + (3.0 / 6.0) * (2.0 / 4.0),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(reward, expected)


def test_normed_positive_work_reuses_cached_effort_limits(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    actuator = SimpleNamespace(
        joint_indices=slice(None),
        effort_limit=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
    )
    env = _make_torque_rate_env(
        applied_torque=torch.tensor([[2.0, 4.0]], dtype=torch.float32),
        joint_vel=torch.tensor([[5.0, 10.0]], dtype=torch.float32),
        joint_vel_limits=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        actuators={"all_joints": actuator},
    )
    term = rewards.normed_positive_work(_DummyConfig(params={}), env)
    asset_cfg = SimpleNamespace(
        name="robot", joint_ids=torch.tensor([0, 1], dtype=torch.long)
    )

    first = term(env, asset_cfg=asset_cfg)
    assert torch.allclose(
        first,
        torch.tensor(
            [(2.0 / 10.0) * (5.0 / 10.0) + (4.0 / 20.0) * (10.0 / 20.0)],
            dtype=torch.float32,
        ),
    )

    actuator.effort_limit = torch.tensor(
        [[1000.0, 1000.0]], dtype=torch.float32
    )
    reward = term(env, asset_cfg=asset_cfg)

    expected = torch.tensor(
        [(2.0 / 10.0) * (5.0 / 10.0) + (4.0 / 20.0) * (10.0 / 20.0)],
        dtype=torch.float32,
    )
    assert torch.allclose(reward, expected)


def test_normed_positive_work_requires_positive_finite_velocity_limits(
    monkeypatch,
):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_torque_rate_env(
        applied_torque=torch.tensor([[2.0, 4.0]], dtype=torch.float32),
        joint_vel=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        joint_vel_limits=torch.tensor([[0.0, torch.inf]], dtype=torch.float32),
        actuators={
            "all_joints": SimpleNamespace(
                joint_indices=slice(None),
                effort_limit=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
            )
        },
    )
    term = rewards.normed_positive_work(_DummyConfig(params={}), env)

    try:
        term(
            env,
            asset_cfg=SimpleNamespace(
                name="robot", joint_ids=torch.tensor([0, 1], dtype=torch.long)
            ),
        )
    except ValueError as exc:
        assert (
            "normed_positive_work requires finite, strictly positive"
            in str(exc)
        )
    else:
        raise AssertionError(
            "expected normed_positive_work to reject invalid limits"
        )


def test_action_acc_matches_second_order_action_change(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_action_acc_env(torch.zeros(2, 2, dtype=torch.float32))
    term = rewards.action_acc(_DummyConfig(params={}), env)

    first = term(env)
    assert torch.allclose(first, torch.zeros(2))

    env.episode_length_buf[:] = 1
    env.action_manager.action = torch.tensor(
        [[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32
    )
    second = term(env)
    assert torch.allclose(second, torch.zeros(2))

    env.episode_length_buf[:] = 2
    env.action_manager.action = torch.tensor(
        [[3.0, 1.0], [5.0, 1.0]], dtype=torch.float32
    )
    third = term(env)

    expected = torch.tensor([10.0, 2.0], dtype=torch.float32)
    assert torch.allclose(third, expected)


def test_action_acc_reset_clears_history(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    env = _make_action_acc_env(torch.zeros(1, 2, dtype=torch.float32))
    term = rewards.action_acc(_DummyConfig(params={}), env)

    assert torch.allclose(term(env), torch.zeros(1))

    env.episode_length_buf[:] = 1
    env.action_manager.action = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(term(env), torch.zeros(1))

    env.episode_length_buf[:] = 2
    env.action_manager.action = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(term(env), torch.tensor([4.0]))

    term.reset(env_ids=[0])
    env.action_manager.action = torch.tensor([[7.0, 7.0]], dtype=torch.float32)

    assert torch.allclose(term(env), torch.zeros(1))


def test_build_rewards_config_exposes_action_acc_term(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)

    rewards_cfg = rewards.build_rewards_config(
        {
            "action_acc": {
                "weight": -2.5,
                "params": {},
            }
        }
    )

    assert rewards_cfg.action_acc.func is rewards.action_acc
    assert rewards_cfg.action_acc.weight == -2.5
    assert rewards_cfg.action_acc.params == {}


def test_motion_tracking_logs_normed_torque_rate_metric(monkeypatch):
    motion_tracking = _load_motion_tracking_module(monkeypatch)
    env = motion_tracking.MotionTrackingEnv.__new__(
        motion_tracking.MotionTrackingEnv
    )
    env.metrics = {}
    env._robot_prev_joint_vel = None
    env._robot_prev_applied_torque = None
    env._robot_torque_rate_inv_effort_limit = None
    env._robot_torque_rate_needs_reseed = None

    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_vel=torch.zeros(2, 2, dtype=torch.float32),
            applied_torque=torch.zeros(2, 2, dtype=torch.float32),
        ),
        actuators={
            "all_joints": SimpleNamespace(
                joint_indices=slice(None),
                effort_limit=torch.tensor(
                    [[10.0, 20.0], [10.0, 20.0]], dtype=torch.float32
                ),
            )
        },
    )
    env._env = SimpleNamespace(
        step_dt=0.5,
        action_manager=SimpleNamespace(
            action=torch.zeros(2, 2, dtype=torch.float32),
            prev_action=torch.zeros(2, 2, dtype=torch.float32),
        ),
        scene={"robot": robot},
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        num_envs=2,
        device=torch.device("cpu"),
    )

    infos = {"log": {}}
    env._update_robot_metrics(infos)
    assert torch.isclose(
        infos["log"]["Metrics/Robot/Normed_Torque_Rate"],
        torch.tensor(0.0),
    )

    env._env.episode_length_buf[:] = 1
    robot.data.applied_torque = torch.tensor(
        [[1.0, 4.0], [2.0, 8.0]], dtype=torch.float32
    )
    env._update_robot_metrics(infos)

    expected = torch.tensor(
        [
            (1.0 / 10.0) ** 2 + (4.0 / 20.0) ** 2,
            (2.0 / 10.0) ** 2 + (8.0 / 20.0) ** 2,
        ],
        dtype=torch.float32,
    ).mean()
    assert torch.allclose(
        infos["log"]["Metrics/Robot/Normed_Torque_Rate"], expected
    )
