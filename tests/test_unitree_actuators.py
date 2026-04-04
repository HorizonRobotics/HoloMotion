import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ACTUATOR_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "unitree_actuators.py"
)
SCENE_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_scene.py"
)


class _DummyArticulationActions:
    def __init__(
        self,
        joint_positions=None,
        joint_velocities=None,
        joint_efforts=None,
        joint_indices=None,
    ):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_efforts = joint_efforts
        self.joint_indices = joint_indices


class _DummyDelayedPDActuatorCfg:
    min_delay = 0
    max_delay = 0

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyDelayedPDActuator:
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self._num_envs = kwargs.get("num_envs", 4)
        self._device = kwargs.get("device", "cpu")
        self.num_joints = len(
            kwargs.get("joint_names", ["joint_a", "joint_b"])
        )
        self.computed_effort = torch.zeros(
            self._num_envs, self.num_joints, device=self._device
        )
        self.applied_effort = torch.zeros_like(self.computed_effort)
        effort_limit = kwargs.get("effort_limit", 100.0)
        if isinstance(effort_limit, torch.Tensor):
            self.effort_limit = effort_limit.clone().to(device=self._device)
        else:
            self.effort_limit = torch.full_like(
                self.computed_effort, float(effort_limit)
            )
        self.super_compute_inputs = []
        self.super_compute_joint_positions = []
        self.reset_calls = []

    def _parse_joint_parameter(self, value, default):
        if value is None:
            value = default
        if isinstance(value, torch.Tensor):
            return value.clone().to(device=self._device)
        if isinstance(value, dict):
            values = list(value.values())
            tensor = torch.tensor(
                values, dtype=torch.float32, device=self._device
            )
            return tensor.unsqueeze(0).repeat(self._num_envs, 1)
        if isinstance(value, (float, int)):
            return torch.full_like(self.computed_effort, float(value))
        raise TypeError(f"Unsupported parameter type: {type(value)}")

    def reset(self, env_ids):
        self.reset_calls.append(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        if control_action.joint_efforts is None:
            self.super_compute_inputs.append(None)
        else:
            self.super_compute_inputs.append(
                control_action.joint_efforts.clone()
            )
        if control_action.joint_positions is None:
            self.super_compute_joint_positions.append(None)
        else:
            self.super_compute_joint_positions.append(
                control_action.joint_positions.clone()
            )
        self.computed_effort = control_action.joint_efforts.clone()
        self.applied_effort = control_action.joint_efforts.clone()
        return control_action


def _configclass(cls):
    annotations = getattr(cls, "__annotations__", {})
    defaults = {
        name: getattr(cls, name) for name in annotations if hasattr(cls, name)
    }

    def __init__(self, **kwargs):
        for name, value in defaults.items():
            setattr(self, name, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    cls.__init__ = __init__
    return cls


def _load_unitree_actuator_module(monkeypatch):
    isaaclab_root = ModuleType("isaaclab")
    isaaclab_actuators = ModuleType("isaaclab.actuators")
    isaaclab_actuators.DelayedPDActuator = _DummyDelayedPDActuator
    isaaclab_actuators.DelayedPDActuatorCfg = _DummyDelayedPDActuatorCfg
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = _configclass
    isaaclab_utils_types = ModuleType("isaaclab.utils.types")
    isaaclab_utils_types.ArticulationActions = _DummyArticulationActions

    for name, module in {
        "isaaclab": isaaclab_root,
        "isaaclab.actuators": isaaclab_actuators,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.types": isaaclab_utils_types,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    isaaclab_root.actuators = isaaclab_actuators
    isaaclab_root.utils = isaaclab_utils
    isaaclab_utils.types = isaaclab_utils_types

    module_name = "_test_unitree_actuators"
    spec = importlib.util.spec_from_file_location(
        module_name, ACTUATOR_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_erfi_actuator(module, *, cfg_kwargs=None, num_envs=4, num_joints=3):
    if cfg_kwargs is None:
        cfg_kwargs = {}
    cfg_defaults = {
        "Y1": 100.0,
        "Y2": 120.0,
        "erfi_enabled": True,
        "ema_filter_enabled": False,
        "ema_filter_alpha": 1.0,
        "ema_filter_debug_dump_path": None,
        "ema_filter_debug_stop_after_dump": False,
        "rfi_probability": 0.5,
        "rfi_lim": 0.1,
        "randomize_rfi_lim": True,
        "rfi_lim_range": (0.5, 1.5),
        "rao_lim": 0.1,
    }
    cfg_defaults.update(cfg_kwargs)
    cfg = module.UnitreeErfiActuatorCfg(**cfg_defaults)
    actuator = module.UnitreeErfiActuator(
        cfg,
        joint_names=[f"joint_{idx}" for idx in range(num_joints)],
        joint_ids=torch.arange(num_joints),
        num_envs=num_envs,
        device="cpu",
        stiffness=0.0,
        damping=0.0,
        armature=0.0,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
        effort_limit=100.0,
        velocity_limit=100.0,
    )
    return actuator


def _make_action(actuator):
    return _DummyArticulationActions(
        joint_positions=torch.zeros_like(actuator.computed_effort),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )


def test_unitree_erfi_reset_samples_all_rfi(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={"rfi_probability": 1.0},
    )

    actuator.reset(torch.tensor([0, 1, 2, 3], dtype=torch.long))

    assert torch.all(actuator._mode_is_rfi)
    assert torch.allclose(
        actuator._rao_scale, torch.zeros_like(actuator._rao_scale)
    )


def test_unitree_erfi_reset_samples_all_rao(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={"rfi_probability": 0.0},
    )

    actuator.reset(torch.tensor([0, 1, 2, 3], dtype=torch.long))

    assert not torch.any(actuator._mode_is_rfi)
    assert torch.any(actuator._rao_scale != 0.0)


def test_unitree_erfi_rfi_without_randomized_limit_uses_effort_limit_ratio(
    monkeypatch,
):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={
            "rfi_probability": 1.0,
            "randomize_rfi_lim": False,
            "rfi_lim": 0.1,
        },
        num_envs=2,
        num_joints=2,
    )
    actuator.reset(torch.tensor([0, 1], dtype=torch.long))

    torch.manual_seed(0)
    actuator.compute(
        _make_action(actuator),
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )

    injected = actuator.super_compute_inputs[-1]
    assert torch.all(injected.abs() <= 10.0 + 1.0e-6)


def test_unitree_erfi_reset_randomizes_rfi_scale_within_range(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={
            "rfi_probability": 1.0,
            "rfi_lim_range": (0.5, 1.5),
        },
        num_envs=2,
        num_joints=2,
    )

    actuator.reset(torch.tensor([0, 1], dtype=torch.long))

    assert torch.all(actuator._rfi_lim_scale >= 0.5)
    assert torch.all(actuator._rfi_lim_scale <= 1.5)


def test_unitree_erfi_rao_bias_stays_constant_between_resets(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={"rfi_probability": 0.0, "rao_lim": 0.1},
        num_envs=2,
        num_joints=2,
    )
    actuator.reset(torch.tensor([0, 1], dtype=torch.long))
    action = _make_action(actuator)

    actuator.compute(
        action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    first = actuator.super_compute_inputs[-1].clone()
    actuator.compute(
        action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    second = actuator.super_compute_inputs[-1].clone()

    assert torch.allclose(first, second)


def test_unitree_erfi_rfi_changes_each_compute(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={"rfi_probability": 1.0, "randomize_rfi_lim": False},
        num_envs=2,
        num_joints=2,
    )
    actuator.reset(torch.tensor([0, 1], dtype=torch.long))
    action = _make_action(actuator)

    torch.manual_seed(0)
    actuator.compute(
        action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    first = actuator.super_compute_inputs[-1].clone()
    torch.manual_seed(1)
    actuator.compute(
        action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    second = actuator.super_compute_inputs[-1].clone()

    assert not torch.allclose(first, second)


def test_unitree_erfi_disabled_matches_plain_unitree(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={"erfi_enabled": False},
        num_envs=2,
        num_joints=2,
    )
    action = _make_action(actuator)

    actuator.reset(torch.tensor([0, 1], dtype=torch.long))
    actuator.compute(
        action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )

    assert torch.allclose(
        actuator.super_compute_inputs[-1],
        torch.zeros_like(actuator.super_compute_inputs[-1]),
    )


def test_unitree_erfi_ema_filters_joint_positions(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={
            "erfi_enabled": False,
            "ema_filter_enabled": True,
            "ema_filter_alpha": 0.25,
        },
        num_envs=2,
        num_joints=2,
    )
    first_action = _DummyArticulationActions(
        joint_positions=torch.tensor(
            [[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32
        ),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )
    second_action = _DummyArticulationActions(
        joint_positions=torch.tensor(
            [[3.0, 1.0], [1.5, 0.5]], dtype=torch.float32
        ),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )

    actuator.compute(
        first_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    actuator.compute(
        second_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )

    assert torch.allclose(
        actuator.super_compute_joint_positions[0],
        first_action.joint_positions,
    )
    expected_second = (
        0.25 * second_action.joint_positions
        + 0.75 * first_action.joint_positions
    )
    assert torch.allclose(
        actuator.super_compute_joint_positions[1], expected_second
    )


def test_unitree_erfi_ema_reset_clears_only_selected_envs(monkeypatch):
    module = _load_unitree_actuator_module(monkeypatch)
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={
            "erfi_enabled": False,
            "ema_filter_enabled": True,
            "ema_filter_alpha": 0.5,
        },
        num_envs=2,
        num_joints=1,
    )
    one_action = _DummyArticulationActions(
        joint_positions=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )
    two_action = _DummyArticulationActions(
        joint_positions=torch.tensor([[2.0], [2.0]], dtype=torch.float32),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )
    zero_action = _DummyArticulationActions(
        joint_positions=torch.zeros_like(actuator.computed_effort),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )

    actuator.compute(
        one_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    actuator.compute(
        two_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    actuator.reset(torch.tensor([1], dtype=torch.long))
    actuator.compute(
        zero_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )

    assert torch.allclose(
        actuator.super_compute_joint_positions[1],
        torch.tensor([[1.5], [1.5]], dtype=torch.float32),
    )
    assert torch.allclose(
        actuator.super_compute_joint_positions[2],
        torch.tensor([[0.75], [0.0]], dtype=torch.float32),
    )


def test_unitree_erfi_ema_debug_dump_records_formula(monkeypatch, tmp_path):
    module = _load_unitree_actuator_module(monkeypatch)
    dump_path = tmp_path / "ema_verify.json"
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={
            "erfi_enabled": False,
            "ema_filter_enabled": True,
            "ema_filter_alpha": 0.25,
            "ema_filter_debug_dump_path": str(dump_path),
        },
        num_envs=2,
        num_joints=2,
    )
    first_action = _DummyArticulationActions(
        joint_positions=torch.tensor(
            [[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32
        ),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )
    second_action = _DummyArticulationActions(
        joint_positions=torch.tensor(
            [[3.0, 1.0], [1.5, 0.5]], dtype=torch.float32
        ),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )

    actuator.compute(
        first_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    actuator.compute(
        second_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )

    assert dump_path.is_file()
    payload = json.loads(dump_path.read_text())
    expected_second = (
        0.25 * second_action.joint_positions[0]
        + 0.75 * first_action.joint_positions[0]
    )
    assert payload["alpha"] == 0.25
    assert payload["matched"] is True
    assert payload["env_index"] == 0
    assert payload["raw_joint_positions"] == [3.0, 1.0]
    assert payload["previous_filtered_joint_positions"] == [1.0, -1.0]
    assert payload["expected_filtered_joint_positions"] == pytest.approx(
        expected_second.tolist()
    )
    assert payload["actual_filtered_joint_positions"] == pytest.approx(
        expected_second.tolist()
    )


def test_unitree_erfi_ema_debug_stop_after_dump(monkeypatch, tmp_path):
    module = _load_unitree_actuator_module(monkeypatch)
    dump_path = tmp_path / "ema_verify.json"
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={
            "erfi_enabled": False,
            "ema_filter_enabled": True,
            "ema_filter_alpha": 0.5,
            "ema_filter_debug_dump_path": str(dump_path),
            "ema_filter_debug_stop_after_dump": True,
        },
        num_envs=1,
        num_joints=1,
    )
    first_action = _DummyArticulationActions(
        joint_positions=torch.tensor([[1.0]], dtype=torch.float32),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )
    second_action = _DummyArticulationActions(
        joint_positions=torch.tensor([[3.0]], dtype=torch.float32),
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )

    actuator.compute(
        first_action,
        joint_pos=torch.zeros_like(actuator.computed_effort),
        joint_vel=torch.zeros_like(actuator.computed_effort),
    )
    with pytest.raises(RuntimeError, match="EMA verification dump written"):
        actuator.compute(
            second_action,
            joint_pos=torch.zeros_like(actuator.computed_effort),
            joint_vel=torch.zeros_like(actuator.computed_effort),
        )

    assert dump_path.is_file()


def test_unitree_erfi_ema_debug_dump_records_skip_reason(
    monkeypatch, tmp_path
):
    module = _load_unitree_actuator_module(monkeypatch)
    dump_path = tmp_path / "ema_verify_skip.json"
    actuator = _make_erfi_actuator(
        module,
        cfg_kwargs={
            "erfi_enabled": False,
            "ema_filter_enabled": True,
            "ema_filter_debug_dump_path": str(dump_path),
            "ema_filter_debug_stop_after_dump": True,
        },
        num_envs=1,
        num_joints=1,
    )
    action = _DummyArticulationActions(
        joint_positions=None,
        joint_velocities=torch.zeros_like(actuator.computed_effort),
        joint_efforts=torch.zeros_like(actuator.computed_effort),
    )

    with pytest.raises(RuntimeError, match="EMA verification dump written"):
        actuator.compute(
            action,
            joint_pos=torch.zeros_like(actuator.computed_effort),
            joint_vel=torch.zeros_like(actuator.computed_effort),
        )

    payload = json.loads(dump_path.read_text())
    assert payload["applied"] is False
    assert payload["reason"] == "joint_positions_none"


def _load_scene_module(monkeypatch):
    actuator_module = _load_unitree_actuator_module(monkeypatch)

    isaaclab_root = ModuleType("isaaclab")
    isaaclab_sim = ModuleType("isaaclab.sim")
    isaaclab_sim.UrdfFileCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    isaaclab_sim.RigidBodyPropertiesCfg = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    isaaclab_sim.ArticulationRootPropertiesCfg = (
        lambda **kwargs: SimpleNamespace(**kwargs)
    )
    isaaclab_sim.UrdfConverterCfg = SimpleNamespace(
        JointDriveCfg=SimpleNamespace(
            PDGainsCfg=lambda **kwargs: SimpleNamespace(**kwargs)
        )
    )
    isaaclab_actuators = ModuleType("isaaclab.actuators")
    isaaclab_actuators.ImplicitActuatorCfg = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    isaaclab_assets = ModuleType("isaaclab.assets")
    isaaclab_assets.ArticulationCfg = SimpleNamespace(
        InitialStateCfg=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    isaaclab_assets.ArticulationCfg = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    isaaclab_assets.AssetBaseCfg = lambda **kwargs: SimpleNamespace(**kwargs)
    isaaclab_scene = ModuleType("isaaclab.scene")
    isaaclab_scene.InteractiveSceneCfg = object
    isaaclab_sensors = ModuleType("isaaclab.sensors")
    isaaclab_sensors.ContactSensorCfg = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    isaaclab_sensors.RayCasterCfg = SimpleNamespace(
        OffsetCfg=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    isaaclab_sensors.patterns = SimpleNamespace(
        GridPatternCfg=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    isaaclab_terrains = ModuleType("isaaclab.terrains")
    isaaclab_terrains.TerrainImporterCfg = object
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = _configclass
    loguru = ModuleType("loguru")
    loguru.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    fake_terrain = ModuleType(
        "holomotion.src.env.isaaclab_components.isaaclab_terrain"
    )
    fake_terrain.build_terrain_config = lambda *args, **kwargs: None

    fake_unitree = ModuleType(
        "holomotion.src.env.isaaclab_components.unitree_actuators"
    )
    fake_unitree.UnitreeActuator = actuator_module.UnitreeActuator
    fake_unitree.UnitreeActuatorCfg = actuator_module.UnitreeActuatorCfg
    fake_unitree.UnitreeErfiActuator = actuator_module.UnitreeErfiActuator
    fake_unitree.UnitreeErfiActuatorCfg = (
        actuator_module.UnitreeErfiActuatorCfg
    )

    for name, module in {
        "isaaclab": isaaclab_root,
        "isaaclab.sim": isaaclab_sim,
        "isaaclab.actuators": isaaclab_actuators,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.scene": isaaclab_scene,
        "isaaclab.sensors": isaaclab_sensors,
        "isaaclab.terrains": isaaclab_terrains,
        "isaaclab.utils": isaaclab_utils,
        "loguru": loguru,
        (
            "holomotion.src.env.isaaclab_components.isaaclab_terrain"
        ): fake_terrain,
        (
            "holomotion.src.env.isaaclab_components.unitree_actuators"
        ): fake_unitree,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_isaaclab_scene"
    spec = importlib.util.spec_from_file_location(
        module_name, SCENE_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_scene_builder_selects_unitree_erfi_cfg(monkeypatch):
    module = _load_scene_module(monkeypatch)

    actuators = module._build_unitree_actuator_cfg(
        {"actuator_type": "unitree_erfi"},
        {"erfi": {"enabled": True, "rfi_lim": 0.2}},
    )

    assert isinstance(actuators["all_joints"], module.UnitreeErfiActuatorCfg)
    assert actuators["all_joints"].erfi_enabled is True
    assert actuators["all_joints"].rfi_lim == 0.2


def test_scene_builder_keeps_plain_unitree_cfg(monkeypatch):
    module = _load_scene_module(monkeypatch)

    actuators = module._build_unitree_actuator_cfg(
        {"actuator_type": "unitree"}, {}
    )

    assert isinstance(actuators["all_joints"], module.UnitreeActuatorCfg)
    assert not hasattr(actuators["all_joints"], "rfi_lim")


def test_scene_builder_disables_erfi_when_domain_rand_missing(monkeypatch):
    module = _load_scene_module(monkeypatch)

    actuators = module._build_unitree_actuator_cfg(
        {"actuator_type": "unitree_erfi"}, {}
    )

    assert isinstance(actuators["all_joints"], module.UnitreeErfiActuatorCfg)
    assert actuators["all_joints"].erfi_enabled is False


def test_scene_builder_applies_domain_rand_action_delay_to_unitree(
    monkeypatch,
):
    module = _load_scene_module(monkeypatch)

    actuators = module._build_unitree_actuator_cfg(
        {"actuator_type": "unitree"},
        {"action_delay": {"enabled": True, "min_delay": 1, "max_delay": 3}},
    )

    assert isinstance(actuators["all_joints"], module.UnitreeActuatorCfg)
    assert actuators["all_joints"].min_delay == 1
    assert actuators["all_joints"].max_delay == 3


def test_scene_builder_applies_domain_rand_action_delay_to_unitree_erfi(
    monkeypatch,
):
    module = _load_scene_module(monkeypatch)

    actuators = module._build_unitree_actuator_cfg(
        {"actuator_type": "unitree_erfi"},
        {
            "erfi": {"enabled": True},
            "action_delay": {
                "enabled": True,
                "min_delay": 2,
                "max_delay": 4,
            },
        },
    )

    assert isinstance(actuators["all_joints"], module.UnitreeErfiActuatorCfg)
    assert actuators["all_joints"].min_delay == 2
    assert actuators["all_joints"].max_delay == 4


def test_scene_builder_applies_erfi_ema_filter_config(monkeypatch):
    module = _load_scene_module(monkeypatch)

    actuators = module._build_unitree_actuator_cfg(
        {
            "actuator_type": "unitree_erfi",
            "ema_filter_enabled": True,
            "ema_filter_alpha": 0.37,
        },
        {"erfi": {"enabled": True}},
    )

    assert isinstance(actuators["all_joints"], module.UnitreeErfiActuatorCfg)
    assert actuators["all_joints"].class_type.__name__ == "UnitreeErfiActuator"
    assert actuators["all_joints"].ema_filter_enabled is True
    assert actuators["all_joints"].ema_filter_alpha == 0.37


def test_scene_builder_disables_action_delay_when_domain_rand_missing(
    monkeypatch,
):
    module = _load_scene_module(monkeypatch)

    actuators = module._build_unitree_actuator_cfg(
        {"actuator_type": "unitree"}, {}
    )

    assert isinstance(actuators["all_joints"], module.UnitreeActuatorCfg)
    assert actuators["all_joints"].min_delay == 0
    assert actuators["all_joints"].max_delay == 0
