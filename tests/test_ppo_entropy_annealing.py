from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.algo.algo_base import BaseOnpolicyRL
from holomotion.src.algo.ppo import PPO


def _build_entropy_algo(
    *,
    initial_entropy_coef: float,
    anneal_entropy: bool,
    zero_entropy_point: float,
    current_learning_iteration: int,
    total_learning_iterations: int,
    num_learning_iterations: int = 0,
):
    algo = PPO.__new__(PPO)
    algo.initial_entropy_coef = float(initial_entropy_coef)
    algo.anneal_entropy = bool(anneal_entropy)
    algo.zero_entropy_point = float(zero_entropy_point)
    algo.current_learning_iteration = int(current_learning_iteration)
    algo.total_learning_iterations = int(total_learning_iterations)
    algo.num_learning_iterations = int(num_learning_iterations)
    return algo


def test_entropy_coef_is_constant_when_annealing_disabled():
    algo = _build_entropy_algo(
        initial_entropy_coef=5.0e-3,
        anneal_entropy=False,
        zero_entropy_point=1.0,
        current_learning_iteration=50,
        total_learning_iterations=100,
    )

    assert algo._get_effective_entropy_coef() == pytest.approx(5.0e-3)


def test_entropy_coef_decays_and_respects_resumed_total_iterations():
    algo = _build_entropy_algo(
        initial_entropy_coef=5.0e-3,
        anneal_entropy=True,
        zero_entropy_point=1.0,
        current_learning_iteration=123,
        total_learning_iterations=133,
        num_learning_iterations=10,
    )

    expected = 5.0e-3 * max(0.0, 1.0 - 123.0 / 133.0)
    assert algo._get_effective_entropy_coef() == pytest.approx(expected)


def test_entropy_coef_clamps_to_zero_at_and_after_zero_point():
    algo = _build_entropy_algo(
        initial_entropy_coef=5.0e-3,
        anneal_entropy=True,
        zero_entropy_point=0.75,
        current_learning_iteration=75,
        total_learning_iterations=100,
    )

    assert algo._get_effective_entropy_coef() == pytest.approx(0.0)

    algo.current_learning_iteration = 90
    assert algo._get_effective_entropy_coef() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("initial_entropy_coef", "anneal_entropy", "zero_entropy_point"),
    [
        (-1.0, False, 1.0),
        (1.0, True, 0.0),
        (1.0, True, -0.1),
        (1.0, True, 1.1),
    ],
)
def test_validate_entropy_schedule_config_rejects_invalid_values(
    initial_entropy_coef: float,
    anneal_entropy: bool,
    zero_entropy_point: float,
):
    with pytest.raises(ValueError):
        PPO._validate_entropy_schedule_config(
            initial_entropy_coef=initial_entropy_coef,
            anneal_entropy=anneal_entropy,
            zero_entropy_point=zero_entropy_point,
        )


def test_learn_sets_current_iteration_before_each_update():
    algo = BaseOnpolicyRL.__new__(BaseOnpolicyRL)
    algo.env = SimpleNamespace(reset_all=lambda: ({},))
    algo._wrap_obs_dict = lambda obs_dict: obs_dict
    algo._ensure_storage = lambda obs_td: None
    algo.train_mode = lambda: None
    algo.rollout_policy = lambda obs_td: obs_td
    algo.log_dir = "/tmp/holomotion-test"
    algo.num_learning_iterations = 3
    algo.current_learning_iteration = 5
    algo.total_learning_iterations = 0
    algo.log_interval = 100
    algo.save_interval = 100
    algo.is_main_process = False
    algo.ep_infos = []
    algo._post_iteration_hook = lambda it: None
    algo._post_training_hook = lambda: None
    algo._release_cuda_cache = lambda: None
    algo.save = lambda *args, **kwargs: None
    algo.accelerator = SimpleNamespace(
        wait_for_everyone=lambda: None,
        end_training=lambda: None,
    )

    observed_iterations = []
    observed_totals = []

    def _update():
        observed_iterations.append(algo.current_learning_iteration)
        observed_totals.append(algo.total_learning_iterations)
        return {}

    algo.update = _update

    BaseOnpolicyRL.learn(algo)

    assert observed_iterations == [5, 6, 7]
    assert observed_totals == [8, 8, 8]
