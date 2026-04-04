import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from holomotion.src.evaluation.eval_mujoco_sim2sim import MujocoEvaluator
from holomotion.src.evaluation.obs.obs_builder import PolicyObsBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBS_CONFIG_PATH = (
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motrack_tf_ref_v3_with_freq.yaml"
)
MODULE_CONFIG_PATH = (
    PROJECT_ROOT
    / "holomotion/config/modules/motion_tracking/tf_motrack_v3_with_ft.yaml"
)
OBS_CONFIG_PATH_V2 = (
    PROJECT_ROOT
    / "holomotion/config/env/observations/motion_tracking/obs_motrack_tf_ref_v3_sonic_router_v2.yaml"
)
MODULE_CONFIG_PATH_V2 = (
    PROJECT_ROOT
    / "holomotion/config/modules/motion_tracking/tf_motrack_v3_wo_eepos_ref_route_v2.yaml"
)
DOMAIN_RAND_CONFIG_PATH = (
    PROJECT_ROOT
    / "holomotion/config/env/domain_randomization/domain_rand_strong.yaml"
)


def _make_minimal_motion_npz(path: Path, *, include_cutoff: bool) -> None:
    payload = {
        "ref_global_translation": np.zeros((2, 1, 3), dtype=np.float32),
        "ref_global_rotation_quat": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (2, 1, 1),
        ),
        "ref_global_velocity": np.zeros((2, 1, 3), dtype=np.float32),
        "ref_global_angular_velocity": np.zeros((2, 1, 3), dtype=np.float32),
        "ref_dof_pos": np.zeros((2, 2), dtype=np.float32),
        "ref_dof_vel": np.zeros((2, 2), dtype=np.float32),
    }
    if include_cutoff:
        payload["filter_cutoff_hz"] = np.array(
            [[2.0], [3.0]], dtype=np.float32
        )
    np.savez(path, **payload)


def test_policy_obs_list_accepts_shared_cutoff_term():
    config = OmegaConf.merge(
        OmegaConf.load(OBS_CONFIG_PATH),
        OmegaConf.load(MODULE_CONFIG_PATH),
    )
    evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
    evaluator.config = config

    atomic_obs_list = evaluator._get_policy_atomic_obs_list()

    term_names = [str(list(item.keys())[0]) for item in atomic_obs_list]
    assert term_names[0] == "ref_motion_filter_cutoff_hz"
    assert "actor_ref_gravity_projection_cur" in term_names


def test_cutoff_obs_getters_use_current_frame_and_default_zero():
    evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
    evaluator.motion_frame_idx = 1
    evaluator.filter_cutoff_hz = np.array([[2.0], [3.0]], dtype=np.float32)

    assert evaluator._get_obs_ref_motion_filter_cutoff_hz() == np.float32(3.0)
    assert (
        evaluator._get_obs_actor_ref_motion_filter_cutoff_hz()
        == np.float32(3.0)
    )

    missing = MujocoEvaluator.__new__(MujocoEvaluator)
    missing.motion_frame_idx = 0
    assert missing._get_obs_ref_motion_filter_cutoff_hz() == 0.0


def test_policy_obs_list_v2_uses_only_actor_schema_terms():
    config = OmegaConf.merge(
        OmegaConf.load(OBS_CONFIG_PATH_V2),
        OmegaConf.load(MODULE_CONFIG_PATH_V2),
        OmegaConf.load(DOMAIN_RAND_CONFIG_PATH),
    )
    evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
    evaluator.config = config

    atomic_obs_list = evaluator._get_policy_atomic_obs_list()
    term_names = [str(list(item.keys())[0]) for item in atomic_obs_list]

    assert not any(name.startswith("actor_moe_router_") for name in term_names)
    assert "actor_ref_gravity_projection_cur" in term_names
    assert "actor_ref_base_linvel_fut" in term_names


def test_load_specific_motion_loads_cutoff_metadata_with_zero_fallback():
    with tempfile.TemporaryDirectory() as tmp_dir:
        with_cutoff = Path(tmp_dir) / "with_cutoff.npz"
        without_cutoff = Path(tmp_dir) / "without_cutoff.npz"
        _make_minimal_motion_npz(with_cutoff, include_cutoff=True)
        _make_minimal_motion_npz(without_cutoff, include_cutoff=False)

        evaluator = MujocoEvaluator.__new__(MujocoEvaluator)
        evaluator.load_specific_motion(with_cutoff)
        np.testing.assert_allclose(
            evaluator.filter_cutoff_hz,
            np.array([[2.0], [3.0]], dtype=np.float32),
        )

        evaluator.load_specific_motion(without_cutoff)
        np.testing.assert_allclose(
            evaluator.filter_cutoff_hz,
            np.zeros((2, 1), dtype=np.float32),
        )
