#!/usr/bin/env python3
"""Validate a v2 link policy bundle without starting ROS or robot control."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEPLOY_ROOT / "src"))


def inspect_velocity_bundle(
    config_path: Path, onnx_path: Path, robot_config_path: Path
) -> dict:
    """Check the retained velocity policy before any live-control launch."""
    import math

    import onnx
    import yaml

    velocity_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot_config = yaml.safe_load(
        robot_config_path.read_text(encoding="utf-8")
    )
    if not isinstance(velocity_config, dict) or not isinstance(
        robot_config, dict
    ):
        raise ValueError("Velocity and robot configs must be YAML mappings")
    expected = robot_config.get("policy_dof_order")
    if (
        not isinstance(expected, list)
        or len(expected) != 29
        or len(set(expected)) != 29
    ):
        raise ValueError("Robot config must define 29 unique policy DOF names")

    model = onnx.load_model(str(onnx_path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    names = [
        name.strip()
        for name in metadata.get("joint_names", "").split(",")
        if name.strip()
    ]
    if (
        len(names) != 29
        or len(set(names)) != 29
        or set(names) != set(expected)
    ):
        raise ValueError(
            "Velocity ONNX joint_names disagree with robot 29-DOF config"
        )
    for key in (
        "action_scale",
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
    ):
        values = [
            float(value)
            for value in metadata.get(key, "").split(",")
            if value.strip()
        ]
        if len(values) != 29 or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError(
                f"Velocity ONNX {key} must contain 29 finite values"
            )
    if not any(
        output.type.tensor_type.shape.dim
        and output.type.tensor_type.shape.dim[-1].dim_value == 29
        for output in model.graph.output
    ):
        raise ValueError("Velocity ONNX has no 29-DOF action output")
    return {"velocity_joint_count": len(names), "velocity_action_width": 29}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Saved training config.yaml"
    )
    parser.add_argument(
        "--onnx", required=True, type=Path, help="Exported motion ONNX"
    )
    parser.add_argument(
        "--mode-table",
        required=True,
        type=Path,
        help="Saved link_command_modes.json",
    )
    parser.add_argument(
        "--asset", required=True, type=Path, help="G1 URDF used for FK"
    )
    parser.add_argument("--velocity-config", type=Path)
    parser.add_argument("--velocity-onnx", type=Path)
    parser.add_argument("--robot-config", type=Path)
    args = parser.parse_args()
    velocity_args = (
        args.velocity_config,
        args.velocity_onnx,
        args.robot_config,
    )
    if any(value is not None for value in velocity_args) and not all(
        value is not None for value in velocity_args
    ):
        parser.error(
            "velocity-config, velocity-onnx and robot-config "
            "must be supplied together"
        )
    policy_freq = None
    if args.robot_config is not None:
        import yaml

        robot_config = yaml.safe_load(
            args.robot_config.read_text(encoding="utf-8")
        )
        if (
            not isinstance(robot_config, dict)
            or "policy_freq" not in robot_config
        ):
            raise ValueError(
                "Robot config must define policy_freq for V2 preflight"
            )
        policy_freq = robot_config["policy_freq"]

    from humanoid_policy.link_policy_contract import inspect_v2_link_bundle

    result = inspect_v2_link_bundle(
        config_path=args.config,
        onnx_path=args.onnx,
        mode_table_path=args.mode_table,
        asset_path=args.asset,
        policy_freq=policy_freq,
    )
    if all(value is not None for value in velocity_args):
        result["velocity"] = inspect_velocity_bundle(
            args.velocity_config, args.velocity_onnx, args.robot_config
        )
    print(json.dumps(result, indent=2))
    print("HoloMotion v2 link preflight PASSED. No robot action was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
