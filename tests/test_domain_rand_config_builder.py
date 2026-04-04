import importlib.util
import sys
from pathlib import Path
from types import ModuleType


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_domain_rand.py"
)


class _DummyEventTermCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _DummySceneEntityCfg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def resolve(self, _scene):
        return None


def _load_domain_rand_module(monkeypatch):
    isaaclab = ModuleType("isaaclab")
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls
    isaaclab_utils_math = ModuleType("isaaclab.utils.math")

    isaaclab_assets = ModuleType("isaaclab.assets")
    isaaclab_assets.Articulation = object

    isaaclab_envs = ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedEnv = object
    isaaclab_envs_mdp = ModuleType("isaaclab.envs.mdp")
    isaaclab_envs_mdp.events = ModuleType("isaaclab.envs.mdp.events")
    isaaclab_envs_mdp.events._randomize_prop_by_op = (
        lambda *args, **kwargs: None
    )

    isaaclab_managers = ModuleType("isaaclab.managers")
    isaaclab_managers.SceneEntityCfg = _DummySceneEntityCfg
    isaaclab_managers.EventTermCfg = _DummyEventTermCfg

    isaaclab.envs = isaaclab_envs
    isaaclab.assets = isaaclab_assets
    isaaclab.utils = isaaclab_utils
    isaaclab_envs.mdp = isaaclab_envs_mdp
    isaaclab_utils.math = isaaclab_utils_math

    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.utils": isaaclab_utils,
        "isaaclab.utils.math": isaaclab_utils_math,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.envs.mdp": isaaclab_envs_mdp,
        "isaaclab.envs.mdp.events": isaaclab_envs_mdp.events,
        "isaaclab.managers": isaaclab_managers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_test_domain_rand_builder"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_domain_rand_config_skips_non_event_metadata(monkeypatch):
    module = _load_domain_rand_module(monkeypatch)

    events_cfg = module.build_domain_rand_config(
        {
            "erfi": {
                "enabled": True,
                "rfi_probability": 0.5,
                "rfi_lim": 0.1,
                "randomize_rfi_lim": True,
                "rfi_lim_range": [0.5, 1.5],
                "rao_lim": 0.1,
            },
            "action_delay": {
                "enabled": True,
                "min_delay": 1,
                "max_delay": 3,
            },
            "motion_init_perturb": {
                "root_pose_perturb_range": {"x": [-0.1, 0.1]}
            },
            "obs_noise": {"actor_dof_pos": {"n_min": -0.01, "n_max": 0.01}},
            "default_dof_pos_bias": {
                "mode": "startup",
                "params": {
                    "joint_names": [".*"],
                    "pos_distribution_params": [-0.01, 0.01],
                    "operation": "add",
                    "distribution": "uniform",
                },
            },
        }
    )

    assert hasattr(events_cfg, "default_dof_pos_bias")
    assert events_cfg.default_dof_pos_bias.mode == "startup"
    assert not hasattr(events_cfg, "erfi")
    assert not hasattr(events_cfg, "action_delay")
    assert not hasattr(events_cfg, "motion_init_perturb")
    assert not hasattr(events_cfg, "obs_noise")
