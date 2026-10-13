"""Exercise the image CLI with inert policy checks and launcher stubs."""

from __future__ import annotations
import os
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "holomotion_v2_entrypoint.sh"
)
MODES = ("Pelvis-1", "Keypoint-5")


@pytest.fixture
def isolated_cli(tmp_path):
    root = tmp_path / "app"
    deploy = root / "deployment/unitree_g1_ros2_29dof"
    for name in (
        "install/setup.bash",
        "src/config/g1_29dof_holomotion_v2.yaml",
        "src/models/velocity_tracking_model/config.yaml",
        "src/models/velocity_tracking_model/exported/model.onnx",
        "src/models/v2_motion_model/exported/model.onnx",
    ):
        path = deploy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    checker = tmp_path / "inert-python"
    checker.write_text('#!/bin/sh\nprintf "checked\\n" >> "$CHECK_LOG"\n')
    checker.chmod(0o755)
    launcher = deploy / "launch_holomotion_29dof_docker.sh"
    launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    script = tmp_path / "holomotion"
    script.write_text(
        ENTRYPOINT.read_text()
        .replace("ROOT=/opt/holomotion", f'ROOT="{root}"')
        .replace(
            "PYTHON=/root/miniconda3/envs/holomotion_deploy/bin/python",
            f'PYTHON="{checker}"',
        )
    )
    check_log = tmp_path / "checks.log"
    return script, check_log


@pytest.mark.parametrize("mode", MODES)
def test_teleop_passes_canonical_mode_after_checks(isolated_cli, mode):
    script, check_log = isolated_cli
    result = subprocess.run(
        ["bash", str(script), "teleop", "--mode", mode],
        env={**os.environ, "CHECK_LOG": str(check_log)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert check_log.read_text().splitlines() == [
        "checked",
        "checked",
        "checked",
    ]
    assert result.stdout.splitlines() == [
        "--set",
        "runtime.build_before_launch=false",
        "--set",
        "robot.config_file=config/g1_29dof_holomotion_v2.yaml",
        "--set",
        f"policy.link_control_mode={mode}",
    ]


@pytest.mark.parametrize("mode", ["invalid", "Keypoint-3", "WholeBody-14"])
def test_invalid_mode_exits_before_policy_checks(isolated_cli, mode):
    script, check_log = isolated_cli
    result = subprocess.run(
        ["bash", str(script), "teleop", "--mode", mode],
        env={**os.environ, "CHECK_LOG": str(check_log)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Unsupported v2 control mode" in result.stderr
    assert not check_log.exists()


def test_help_lists_canonical_modes():
    result = subprocess.run(
        ["bash", str(ENTRYPOINT), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "|".join(MODES) in result.stdout


def test_default_teleop_uses_five_point_mode(isolated_cli):
    script, check_log = isolated_cli
    result = subprocess.run(
        ["bash", str(script), "teleop"],
        env={**os.environ, "CHECK_LOG": str(check_log)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "policy.link_control_mode=Keypoint-5" in result.stdout
    assert "WholeBody-14" not in result.stdout


def test_failed_ownership_check_blocks_launch(isolated_cli):
    script, check_log = isolated_cli
    checker = script.parent / "inert-python"
    checker.write_text(
        '#!/bin/sh\ncase "$1" in *control_ownership.py) exit 9 ;; esac\nprintf "checked\\n" >> "$CHECK_LOG"\n'
    )
    result = subprocess.run(
        ["bash", str(script), "teleop"],
        env={**os.environ, "CHECK_LOG": str(check_log)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 9
    assert "policy.link_control_mode=" not in result.stdout
