"""Choose a frozen candidate using completed validation results only."""

import argparse
import math
from pathlib import Path

from holoboxing.data.common import sha256, write_json
from holoboxing.training.motion_prior import load_prior

FORMAT = "holoboxing_prior_validation_selection_v1"


def select(candidates, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, population = [], None
    for path in candidates:
        path = Path(path).resolve(strict=True)
        _, saved = load_prior(path)
        validation = saved.get("best_validation") or {}
        cursor = saved.get("cursor", {})
        metric = validation.get("reconstruction", math.nan)
        if (
            validation.get("status") != "PASS"
            or validation.get("samples", 0) <= 0
            or validation.get("samples") != validation.get("expected_samples")
            or cursor.get("epoch") != validation.get("epoch")
            or cursor.get("batch") != 0
            or not math.isfinite(metric)
            or metric < 0
        ):
            raise ValueError(
                "Candidate must contain its best completed validation weights"
            )
        identity = (
            saved.get("source_selection"),
            saved.get("validation_selection"),
        )
        if (
            identity[0] is None
            or identity[1] is None
            or identity[0].get("split") != "train"
            or identity[1].get("split") != "validation"
        ):
            raise ValueError(
                "Candidate lacks separate train/validation provenance"
            )
        if population is not None and identity != population:
            raise ValueError(
                "Candidate training or validation populations differ"
            )
        population = identity
        rows.append(
            dict(
                checkpoint=str(path),
                checkpoint_sha256=sha256(path),
                validation=validation,
                config=saved["config"],
                recipe=saved["recipe"],
            )
        )
    if not rows or len({row["checkpoint_sha256"] for row in rows}) != len(
        rows
    ):
        raise ValueError("Supply nonempty, distinct candidates")
    selected = min(
        rows,
        key=lambda row: (
            row["validation"]["reconstruction"],
            row["checkpoint_sha256"],
        ),
    )
    report = dict(
        format=FORMAT,
        criterion="lowest_validation_window_mean_normalized_reconstruction",
        source_selection=population[0],
        validation_selection=population[1],
        candidates=rows,
        selected_checkpoint_sha256=selected["checkpoint_sha256"],
        test_evaluated=False,
    )
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    select(args.candidate, args.output)


if __name__ == "__main__":
    main()
