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
from holoboxing.training.supervised.training import FORMAT, load_supervised
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig


def load_policy(checkpoint):
    """Load the attacker and saved statistics without the lower controller."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("format") not in (
        FORMAT,
        CHECKPOINT_FORMAT,
    ):
        raise ValueError("Expected a current supervised or RL checkpoint")
    config = TrainConfig(**saved["config"])
    if saved["format"] == FORMAT:
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
            actor = JointActor(actor_state["neutral_reference"], config)
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
        max_error = 0.0
        for observations in probes:
            expected = policy(torch.from_numpy(observations)).numpy()
            actual = session.run(["target_pose"], {"obs": observations})[0]
            if not np.isfinite(actual).all():
                raise ValueError("Exported policy produced nonfinite targets")
            _, repairs = sanitize_target_pose(actual)
            if any(repairs.values()):
                raise ValueError("Exported targets must not require repair")
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            max_error = max(
                max_error, float(np.max(np.abs(actual - expected)))
            )
        del session
        # Publish only after validation, without overwriting an existing file.
        output.hardlink_to(candidate)
    return dict(model=str(output), bytes=size, max_abs_error=max_error)


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
