from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.algo.algo_base import BaseOnpolicyRL


def test_log_iteration_uses_checkpoint_start_for_total_iterations():
    algo = BaseOnpolicyRL.__new__(BaseOnpolicyRL)
    algo.log_dir = "/tmp/holomotion-test"
    algo.gpu_world_size = 1
    algo.num_steps_per_env = 8
    algo.num_envs = 16
    algo.num_learning_iterations = 10
    algo.current_learning_iteration = 123
    algo.rewbuffer = []
    algo.lenbuffer = []
    algo._aggregate_episode_log_metrics = lambda: {}
    algo._get_additional_log_metrics = lambda: {}
    algo.algo_logger = mock.Mock()

    BaseOnpolicyRL._log_iteration(
        algo,
        it=123,
        loss_dict={"policy": 1.5},
        collection_time=2.0,
        learn_time=2.0,
    )

    algo.algo_logger.log_iteration.assert_called_once()
    _, kwargs = algo.algo_logger.log_iteration.call_args
    assert kwargs["step"] == 123
    assert kwargs["total_learning_iterations"] == 133
    assert kwargs["metrics"]["0-Train/iteration"] == 123
    assert kwargs["metrics"]["0-Train/iterations_total"] == 133
