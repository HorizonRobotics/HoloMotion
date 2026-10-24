"""Shared task ownership must not depend on a particular learning method."""

import ast
from pathlib import Path


def test_shared_task_has_no_algorithm_or_script_imports():
    root = Path(__file__).resolve().parents[1]
    for directory in (
        "env/reference_tracking",
        "evaluation/reference_tracking",
    ):
        for path in (root / "holomotion/src" / directory).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert not module.startswith(
                        ("holomotion.src.algo", "holomotion.scripts")
                    ), path
                elif isinstance(node, ast.Import):
                    assert not any(
                        alias.name.startswith(
                            ("holomotion.src.algo", "holomotion.scripts")
                        )
                        for alias in node.names
                    ), path


def test_isaac_factory_keeps_each_run_paths_in_its_own_constructor():
    from types import SimpleNamespace

    from holomotion.src.algo.supermimic.backends import (
        backend_class,
        evaluator_class,
    )

    first = SimpleNamespace(
        physics_backend="isaac_physx",
        robot_urdf=Path("/one/robot.urdf"),
        output=Path("/one/run"),
    )
    second = SimpleNamespace(
        physics_backend="isaac_physx",
        robot_urdf=Path("/two/robot.urdf"),
        output=Path("/two/run"),
    )
    first_backend, first_evaluator = (
        backend_class(first),
        evaluator_class(first),
    )
    second_backend, second_evaluator = (
        backend_class(second),
        evaluator_class(second),
    )
    for constructor in (first_backend, first_evaluator):
        assert constructor.keywords == dict(
            robot_urdf=first.robot_urdf, cache_dir=first.output / "isaac_usd"
        )
    for constructor in (second_backend, second_evaluator):
        assert constructor.keywords == dict(
            robot_urdf=second.robot_urdf, cache_dir=second.output / "isaac_usd"
        )
    assert first_backend.keywords["robot_urdf"] == first.robot_urdf
