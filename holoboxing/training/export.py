"""Export the trained attacker as a self-contained competition ONNX policy."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from holoboxing.evaluation.runtime.model_contract import (
    CONTRACT,
    check_contract,
    check_model,
)
from holoboxing.evaluation.runtime.policy_output import sanitize_target_pose
from holoboxing.evaluation.runtime.upper import neutral_observation
from holoboxing.training.cli import existing_path
from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
from holoboxing.training.rsl_rl.modules.actor_critic import JointActor
from holoboxing.training.rsl_rl.modules.competition import CompetitionPolicy
from holoboxing.training.rsl_rl.modules.normalization import (
    ActorNormalizer,
    validate_statistics,
)
from holoboxing.training.rsl_rl.runners.on_policy_runner import (
    CHECKPOINT_FORMAT,
)
from holoboxing.training.spherical_actor import (
    FORMAT as SPHERICAL_FORMAT,
    actor_from_state,
)
from holoboxing.training.spherical_rl import FORMAT as SPHERICAL_RL_FORMAT
from holoboxing.training.supervised.training import FORMAT, load_supervised
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig


def check_export_parity(actual, expected):
    """Bound translation in meters and orientation in radians separately.

    FP32 matrix multiplication errors can grow during rotation normalization.
    A componentwise relative tolerance has no physical meaning for this mix
    of positions and rotation columns. Keep translation within 10 micrometers
    and rotations within 0.0001 radians, including near-zero components.
    """
    actual, expected = (
        np.asarray(v, dtype=np.float64) for v in (actual, expected)
    )
    if (
        actual.shape != expected.shape
        or actual.ndim != 2
        or not actual.shape[0]
        or actual.shape[-1] != 225
    ):
        raise ValueError("Export parity requires matching C225 targets")
    if not all(np.isfinite(v).all() for v in (actual, expected)):
        raise ValueError("Export parity requires finite targets")
    a, b = (v.reshape(-1, 5, 5, 9) for v in (actual, expected))
    position_error = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)

    def rotation(points):
        x, z = points[..., 3:6], points[..., 6:9]
        return np.stack((x, np.cross(z, x), z), axis=-1)

    relative = rotation(a) @ rotation(b).swapaxes(-1, -2)
    skew = np.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    )
    angle = np.arctan2(
        np.linalg.norm(skew, axis=-1),
        np.trace(relative, axis1=-2, axis2=-1) - 1,
    )
    # Bound columns too; a scaled matrix must not evade the angle gate.
    column_error = np.max(np.abs(a[..., 3:] - b[..., 3:]))
    result = dict(
        max_position_error_m=float(position_error.max()),
        max_rotation_error_rad=float(angle.max()),
        max_rotation_component_error=float(column_error),
        max_abs_error=float(np.abs(actual - expected).max()),
    )
    if (
        result["max_position_error_m"] > 1e-5
        or result["max_rotation_error_rad"] > 1e-4
        or column_error > 1e-4
    ):
        raise ValueError(
            f"Export parity exceeds physical tolerances: {result}"
        )
    return result


def load_policy(checkpoint):
    """Load the attacker and saved statistics without the lower controller."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("format") not in (
        FORMAT,
        SPHERICAL_FORMAT,
        CHECKPOINT_FORMAT,
        SPHERICAL_RL_FORMAT,
    ):
        raise ValueError("Expected a current supervised or RL checkpoint")
    config = TrainConfig(**saved["config"])
    if saved["format"] in (FORMAT, SPHERICAL_FORMAT):
        actor, normalizer, _ = load_supervised(
            checkpoint, config, saved["actor"]["neutral_reference"]
        )
    else:
        actor_state = {
            name.removeprefix("taskpolicy_ego."): value
            for name, value in saved["policy"].items()
            if name.startswith("taskpolicy_ego.")
        }
        norm_state = {
            name.removeprefix("actor_ego."): value
            for name, value in saved["normalizers"].items()
            if name.startswith("actor_ego.")
        }
        require_finite((actor_state, norm_state), "export checkpoint")
        with torch.random.fork_rng(devices=[]):
            actor = (
                actor_from_state(actor_state, saved["policy_spec"], config)
                if saved["format"] == SPHERICAL_RL_FORMAT
                else JointActor(actor_state["neutral_reference"], config)
            )
            normalizer = ActorNormalizer(until=int(norm_state["until"]))
        if not torch.equal(
            actor.physical_scale, actor_state["physical_scale"]
        ):
            raise ValueError(
                "Saved action scale differs from checkpoint config"
            )
        actor.load_state_dict(actor_state, strict=True)
        normalizer.load_state_dict(norm_state, strict=True)
    require_finite(
        (actor.state_dict(), normalizer.state_dict()), "export policy"
    )
    validate_statistics(normalizer.state_dict())
    if int(normalizer.count) <= 0:
        raise ValueError("Cannot export unfitted observation statistics")
    return CompetitionPolicy(actor, normalizer).eval()


@torch.no_grad()
def export_policy(checkpoint, output):
    output = Path(output).expanduser().absolute()
    if output.suffix.lower() != ".onnx":
        raise ValueError("--output must be an .onnx file")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    policy = load_policy(checkpoint)
    sample = neutral_observation()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        candidate = Path(directory) / "policy.onnx"
        torch.onnx.export(
            policy,
            (torch.from_numpy(sample),),
            candidate,
            input_names=["obs"],
            output_names=["target_pose"],
            opset_version=17,
            dynamo=False,
            external_data=False,
        )
        graph = onnx.load(candidate)
        onnx.helper.set_model_props(graph, {"policy_contract": CONTRACT})
        onnx.save(graph, candidate)
        graph, size = check_model(candidate)
        check_contract(graph)

        # Check CPU export parity, not GPU competition qualification.
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(candidate),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        probes = [sample, np.zeros_like(sample)]
        rng = np.random.default_rng(42)
        probes.extend(
            rng.normal(0, scale, sample.shape).astype(np.float32)
            for scale in (0.1, 1.0, 10.0, 1000.0)
        )
        parity = dict.fromkeys(
            (
                "max_abs_error",
                "max_position_error_m",
                "max_rotation_error_rad",
                "max_rotation_component_error",
            ),
            0.0,
        )
        for observations in probes:
            expected = policy(torch.from_numpy(observations)).numpy()
            actual = session.run(["target_pose"], {"obs": observations})[0]
            if not np.isfinite(actual).all():
                raise ValueError("Exported policy produced nonfinite targets")
            _, repairs = sanitize_target_pose(actual)
            if any(repairs.values()):
                raise ValueError("Exported targets must not require repair")
            errors = check_export_parity(actual, expected)
            parity = {k: max(v, errors[k]) for k, v in parity.items()}
        del session
        # Publish only after validation, without overwriting an existing file.
        output.hardlink_to(candidate)
    return dict(model=str(output), bytes=size, **parity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=existing_path,
        required=True,
        help="trusted epoch_*.pt or model_*.pt; exports the attacker",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new .onnx file"
    )
    args = parser.parse_args(argv)
    print(json.dumps(export_policy(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
