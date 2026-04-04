import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "not_for_commit"
    / "plot_state_series.py"
)


def _load_plot_state_series_module():
    spec = importlib.util.spec_from_file_location(
        "plot_state_series", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_plot_dump_exports_time_matched_scalar_series(tmp_path):
    module = _load_plot_state_series_module()

    robot_config_path = tmp_path / "robot.yaml"
    robot_config_path.write_text(
        "robot:\n  dof_names:\n    - joint_a\n    - joint_b\n",
        encoding="utf-8",
    )

    npz_path = tmp_path / "demo_eval.npz"
    np.savez(
        npz_path,
        robot_dof_torque=np.arange(10, dtype=np.float32).reshape(5, 2),
        robot_dof_acc=np.arange(10, 20, dtype=np.float32).reshape(5, 2),
        robot_action_rate=np.linspace(0.0, 1.0, 5, dtype=np.float32),
        reward=np.linspace(1.0, 2.0, 5, dtype=np.float32),
        bad_scalar=np.array([1.0, 2.0], dtype=np.float32),
        metadata=np.array("demo", dtype="<U4"),
    )

    module.plot_dump(npz_path, robot_config_path)

    output_dir = tmp_path / "demo_eval"
    assert (output_dir / "torque.pdf").is_file()
    assert (output_dir / "dof_acc.pdf").is_file()
    assert (output_dir / "robot_action_rate.pdf").is_file()
    assert (output_dir / "reward.pdf").is_file()
    assert not (output_dir / "bad_scalar.pdf").exists()
    assert not (output_dir / "metadata.pdf").exists()
