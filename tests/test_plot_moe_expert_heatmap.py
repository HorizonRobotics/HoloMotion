import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "not_for_commit"
    / "plot_moe_expert_heatmap.py"
)


def _load_plot_moe_expert_heatmap_module():
    spec = importlib.util.spec_from_file_location(
        "plot_moe_expert_heatmap", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_eval_npz(path: Path) -> None:
    np.savez(
        path,
        robot_moe_expert_logits=np.array(
            [
                [[0.0, 1.0, 2.0, 3.0], [1.5, 0.5, -0.5, -1.5]],
                [[0.1, 1.1, 2.1, 3.1], [1.0, 0.0, -1.0, -2.0]],
                [[0.2, 1.2, 2.2, 3.2], [0.5, -0.5, -1.5, -2.5]],
                [[0.3, 1.3, 2.3, 3.3], [0.0, -1.0, -2.0, -3.0]],
                [[0.4, 1.4, 2.4, 3.4], [-0.5, -1.5, -2.5, -3.5]],
            ],
            dtype=np.float32,
        ),
        robot_moe_expert_indices=np.array(
            [
                [[3, 2], [0, 1]],
                [[3, 2], [0, 1]],
                [[3, 2], [0, 1]],
                [[3, 2], [0, 1]],
                [[3, 2], [0, 1]],
            ],
            dtype=np.int64,
        ),
        robot_dof_torque=np.linspace(-1.0, 1.0, 15, dtype=np.float32).reshape(
            5, 3
        ),
        robot_actions=np.linspace(-0.5, 0.5, 15, dtype=np.float32).reshape(
            5, 3
        ),
        robot_low_level_dof_torque=np.zeros((20, 3), dtype=np.float32),
        robot_low_level_torque_dt=np.array(0.01, dtype=np.float32),
    )


def test_plot_dump_exports_moe_heatmap_pdf(tmp_path):
    module = _load_plot_moe_expert_heatmap_module()

    npz_path = tmp_path / "demo_eval.npz"
    _write_eval_npz(npz_path)

    output_path = module.plot_dump(npz_path)

    assert output_path == (
        tmp_path / "demo_eval_moe_expert_probability_heatmap.pdf"
    )
    assert output_path.is_file()
    assert (tmp_path / "demo_eval_robot_dof_torque_line_plot.pdf").is_file()
    assert (tmp_path / "demo_eval_robot_actions_line_plot.pdf").is_file()


def test_selected_expert_weights_are_renormalized_within_selected_ids():
    module = _load_plot_moe_expert_heatmap_module()

    probabilities = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]],
            [[0.25, 0.25, 0.25, 0.25], [0.05, 0.15, 0.3, 0.5]],
        ],
        dtype=np.float32,
    )
    expert_indices = np.array(
        [
            [[1, 3], [0, 2]],
            [[0, 2], [1, 3]],
        ],
        dtype=np.int64,
    )

    selected_weights = module.compute_selected_expert_weights(
        probabilities, expert_indices
    )

    np.testing.assert_allclose(
        selected_weights,
        np.array(
            [
                [[1.0 / 3.0, 2.0 / 3.0], [0.875, 0.125]],
                [[0.5, 0.5], [0.23076923, 0.7692308]],
            ],
            dtype=np.float32,
        ),
    )


def test_selected_expert_heatmap_only_colors_activated_experts():
    module = _load_plot_moe_expert_heatmap_module()

    probabilities = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]],
            [[0.25, 0.25, 0.25, 0.25], [0.05, 0.15, 0.3, 0.5]],
        ],
        dtype=np.float32,
    )
    expert_indices = np.array(
        [
            [[1, 3], [0, 2]],
            [[0, 2], [1, 3]],
        ],
        dtype=np.int64,
    )

    selected_heatmap = module.build_selected_expert_heatmap(
        probabilities, expert_indices
    )

    np.testing.assert_allclose(
        selected_heatmap,
        np.array(
            [
                [
                    [0.0, 1.0 / 3.0, 0.0, 2.0 / 3.0],
                    [0.875, 0.0, 0.125, 0.0],
                ],
                [
                    [0.5, 0.0, 0.5, 0.0],
                    [0.0, 0.23076923, 0.0, 0.7692308],
                ],
            ],
            dtype=np.float32,
        ),
    )


def test_collect_npz_paths_recursively_sorts_directory_entries(tmp_path):
    module = _load_plot_moe_expert_heatmap_module()

    input_dir = tmp_path / "evals"
    first_npz = input_dir / "z_branch" / "clip_z.npz"
    second_npz = input_dir / "a_branch" / "nested" / "clip_a.npz"
    second_npz.parent.mkdir(parents=True)
    first_npz.parent.mkdir(parents=True)
    _write_eval_npz(first_npz)
    _write_eval_npz(second_npz)
    (input_dir / "ignore.txt").write_text("ignore", encoding="utf-8")

    assert module.collect_npz_paths(input_dir) == [second_npz, first_npz]


def test_plot_input_path_directory_generates_all_heatmaps_with_tqdm(
    tmp_path,
):
    module = _load_plot_moe_expert_heatmap_module()

    input_dir = tmp_path / "evals"
    npz_paths = [
        input_dir / "z_branch" / "clip_z.npz",
        input_dir / "a_branch" / "nested" / "clip_a.npz",
    ]
    for npz_path in npz_paths:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        _write_eval_npz(npz_path)

    expected_output_paths = [
        input_dir
        / "a_branch"
        / "nested"
        / "clip_a_moe_expert_probability_heatmap.pdf",
        input_dir / "z_branch" / "clip_z_moe_expert_probability_heatmap.pdf",
    ]

    fake_tqdm = MagicMock(side_effect=lambda iterable, **_: iterable)
    original_tqdm = module.tqdm
    module.tqdm = fake_tqdm
    try:
        output_paths = module.plot_input_path(input_dir)
    finally:
        module.tqdm = original_tqdm

    assert output_paths == expected_output_paths
    assert all(path.is_file() for path in expected_output_paths)
    expected_torque_paths = [
        input_dir
        / "a_branch"
        / "nested"
        / "clip_a_robot_dof_torque_line_plot.pdf",
        input_dir / "z_branch" / "clip_z_robot_dof_torque_line_plot.pdf",
    ]
    assert all(path.is_file() for path in expected_torque_paths)
    expected_action_paths = [
        input_dir
        / "a_branch"
        / "nested"
        / "clip_a_robot_actions_line_plot.pdf",
        input_dir / "z_branch" / "clip_z_robot_actions_line_plot.pdf",
    ]
    assert all(path.is_file() for path in expected_action_paths)
    assert list(fake_tqdm.call_args.args[0]) == sorted(npz_paths)
    assert fake_tqdm.call_args.kwargs == {
        "desc": "Generating plot PDFs",
        "unit": "file",
        "dynamic_ncols": True,
    }


def test_plot_dump_requires_2d_robot_dof_torque(tmp_path):
    module = _load_plot_moe_expert_heatmap_module()

    npz_path = tmp_path / "bad_eval.npz"
    np.savez(
        npz_path,
        robot_moe_expert_logits=np.zeros((2, 1, 3), dtype=np.float32),
        robot_dof_torque=np.zeros((2,), dtype=np.float32),
    )

    try:
        module.plot_dump(npz_path)
    except ValueError as exc:
        assert "robot_dof_torque must have shape [frames, dofs]" in str(exc)
    else:
        raise AssertionError(
            "Expected plot_dump to reject 1-D robot_dof_torque"
        )
