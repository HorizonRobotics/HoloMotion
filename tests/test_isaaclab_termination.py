import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_termination.py"
)

MOTION_COMMAND_MODULE_NAME = (
    "holomotion.src.env.isaaclab_components.isaaclab_motion_tracking_command"
)
ISAACLAB_UTILS_MODULE_NAME = (
    "holomotion.src.env.isaaclab_components.isaaclab_utils"
)


class _Scene(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


def _load_isaaclab_termination_module(module_name: str):
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_envs = types.ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedRLEnv = object

    isaaclab_terminations = types.SimpleNamespace(
        time_out=lambda env: torch.zeros(1, dtype=torch.bool),
        bad_orientation=lambda env, limit_angle: torch.zeros(
            1, dtype=torch.bool
        ),
        root_height_below_minimum=lambda env, minimum_height: torch.zeros(
            1, dtype=torch.bool
        ),
        native_only_term=lambda env, margin: torch.zeros(1, dtype=torch.bool),
    )
    isaaclab_envs_mdp = types.ModuleType("isaaclab.envs.mdp")
    isaaclab_envs_mdp.terminations = isaaclab_terminations

    isaaclab_managers = types.ModuleType("isaaclab.managers")

    class _TerminationTermCfg:
        def __init__(self, func, params=None, time_out=False):
            self.func = func
            self.params = {} if params is None else params
            self.time_out = time_out

    isaaclab_managers.TerminationTermCfg = _TerminationTermCfg
    isaaclab_managers.SceneEntityCfg = object

    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls
    isaaclab_utils_math = types.ModuleType("isaaclab.utils.math")
    isaaclab_utils_math.quat_apply_inverse = (
        lambda quat, vec: torch.zeros_like(vec)
    )
    isaaclab_utils.math = isaaclab_utils_math

    isaaclab_assets = types.ModuleType("isaaclab.assets")
    isaaclab_assets.Articulation = object

    isaaclab_components_package = types.ModuleType(
        "holomotion.src.env.isaaclab_components"
    )
    motion_command_module = types.ModuleType(MOTION_COMMAND_MODULE_NAME)
    motion_command_module.RefMotionCommand = object

    isaaclab_utils_module = types.ModuleType(ISAACLAB_UTILS_MODULE_NAME)
    isaaclab_utils_module._get_body_indices = lambda robot, keybody_names: None
    isaaclab_utils_module.resolve_holo_config = lambda cfg: cfg
    isaaclab_components_package.isaaclab_motion_tracking_command = (
        motion_command_module
    )
    isaaclab_components_package.isaaclab_utils = isaaclab_utils_module

    fake_modules = {
        "isaaclab": isaaclab_module,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.envs.mdp": isaaclab_envs_mdp,
        "isaaclab.managers": isaaclab_managers,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.math": isaaclab_utils_math,
        "isaaclab.assets": isaaclab_assets,
        "holomotion.src.env.isaaclab_components": isaaclab_components_package,
        MOTION_COMMAND_MODULE_NAME: motion_command_module,
        ISAACLAB_UTILS_MODULE_NAME: isaaclab_utils_module,
    }
    original_modules = {name: sys.modules.get(name) for name in fake_modules}

    sys.modules.update(fake_modules)
    try:
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_wholebody_mpjpe_far_flags_envs_above_mean_error_threshold():
    termination_module = _load_isaaclab_termination_module(
        "isaaclab_termination_under_test"
    )

    current_dof_pos = torch.tensor(
        [
            [0.0, 0.2, 0.6],
            [0.0, 0.1, 0.2],
        ]
    )
    ref_dof_pos = torch.zeros_like(current_dof_pos)
    command = SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=current_dof_pos)),
        get_ref_motion_dof_pos_cur=lambda prefix="ref_": ref_dof_pos,
        get_ref_motion_dof_pos_immediate_next=lambda prefix="ref_": ref_dof_pos,
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command)
    )

    result = termination_module.wholebody_mpjpe_far(env, threshold=0.2)

    assert result.dtype == torch.bool
    assert torch.equal(result, torch.tensor([True, False]))


def test_wholebody_mpjpe_far_uses_immediate_next_reference():
    termination_module = _load_isaaclab_termination_module(
        "isaaclab_termination_under_test_next_dof"
    )

    current_dof_pos = torch.tensor([[0.0, 0.1, 0.2]])
    command = SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=current_dof_pos)),
        get_ref_motion_dof_pos_cur=lambda prefix="ref_": (_ for _ in ()).throw(
            AssertionError("current reference should not be used")
        ),
        get_ref_motion_dof_pos_immediate_next=lambda prefix="ref_": current_dof_pos,
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command)
    )

    result = termination_module.wholebody_mpjpe_far(env, threshold=0.05)

    assert torch.equal(result, torch.tensor([False]))


def test_keybody_ref_pos_far_uses_immediate_next_reference():
    termination_module = _load_isaaclab_termination_module(
        "isaaclab_termination_under_test_next_keybody"
    )

    body_pos = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]], dtype=torch.float32
    )
    robot = SimpleNamespace(
        body_names=["anchor", "target"],
        data=SimpleNamespace(body_pos_w=body_pos),
    )
    command = SimpleNamespace(
        robot=robot,
        get_ref_motion_bodylink_global_pos_cur=(
            lambda prefix="ref_": (_ for _ in ()).throw(
                AssertionError("current reference should not be used")
            )
        ),
        get_ref_motion_bodylink_global_pos_immediate_next=(
            lambda prefix="ref_": body_pos
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command)
    )

    result = termination_module.keybody_ref_pos_far(
        env,
        threshold=0.1,
        keybody_names=["target"],
    )

    assert torch.equal(result, torch.tensor([False]))


def test_ref_gravity_projection_far_uses_immediate_next_reference():
    termination_module = _load_isaaclab_termination_module(
        "isaaclab_termination_under_test_next_gravity"
    )
    gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    anchor_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            GRAVITY_VEC_W=gravity,
            body_quat_w=anchor_quat[:, None, :],
        )
    )
    command = SimpleNamespace(
        robot=robot,
        anchor_bodylink_idx=0,
        get_ref_motion_anchor_bodylink_global_rot_wxyz_cur=(
            lambda prefix="ref_": (_ for _ in ()).throw(
                AssertionError("current reference should not be used")
            )
        ),
        get_ref_motion_anchor_bodylink_global_rot_wxyz_immediate_next=(
            lambda prefix="ref_": anchor_quat
        ),
    )
    env = SimpleNamespace(
        scene=_Scene(robot=robot),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )

    result = termination_module.ref_gravity_projection_far(
        env,
        threshold=0.1,
    )

    assert torch.equal(result, torch.tensor([False]))


def test_build_terminations_config_registers_wholebody_mpjpe_far():
    termination_module = _load_isaaclab_termination_module(
        "isaaclab_termination_under_test_for_cfg"
    )

    config = termination_module.build_terminations_config(
        {
            "wholebody_mpjpe_far": {
                "params": {"threshold": 0.3},
            }
        }
    )

    assert (
        config.wholebody_mpjpe_far.func
        is termination_module.wholebody_mpjpe_far
    )
    assert config.wholebody_mpjpe_far.params == {"threshold": 0.3}
    assert config.wholebody_mpjpe_far.time_out is False


def test_build_terminations_config_resolves_native_isaaclab_termination():
    termination_module = _load_isaaclab_termination_module(
        "isaaclab_termination_under_test_for_native_cfg"
    )

    config = termination_module.build_terminations_config(
        {
            "native_only_term": {
                "params": {"margin": 0.3},
            }
        }
    )

    assert (
        config.native_only_term.func
        is termination_module.isaaclab_mdp.terminations.native_only_term
    )
    assert config.native_only_term.params == {"margin": 0.3}
    assert config.native_only_term.time_out is False


def test_build_terminations_config_raises_on_unknown_termination():
    termination_module = _load_isaaclab_termination_module(
        "isaaclab_termination_under_test_for_unknown_cfg"
    )

    with pytest.raises(ValueError, match="Unknown termination function"):
        termination_module.build_terminations_config({"missing_term": {}})
