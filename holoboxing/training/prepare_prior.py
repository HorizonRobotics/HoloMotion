"""Prepare source-only reference windows without synthetic opponent samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from holoboxing.data.common import sha256, write_json
from holoboxing.data.contract import (
    CompetitionAssets,
    same_data_assets,
    target_from_world,
)
from holoboxing.data.prepare import assign_splits, source_records
from holoboxing.data.records import motion_kinematics

FORMAT = "holoboxing_source_reference_windows_v2"


def prepare(
    sources,
    assets,
    output,
    *,
    seed=42,
    validation_fraction=0.1,
    test_fraction=0.1,
    exclude_groups=(),
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    tree = CompetitionAssets.load(assets)
    records = []
    for root in sources:
        paired = Path(root) / "motions.json"
        if paired.is_file() and not same_data_assets(
            json.loads(paired.read_text()).get("asset_identity"),
            tree.identity(),
        ):
            raise ValueError("Paired motions use different robot assets")
        records.extend(source_records(root))
    splits, used = assign_splits(
        records,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    excluded = set(exclude_groups)
    unknown = excluded - {row["group"] for row in records}
    if unknown:
        raise ValueError(f"Unknown excluded source groups: {sorted(unknown)}")
    # Test shards are retained only for an independent final evaluator.
    used = [row for row in records if row["group"] not in excluded]
    identities = [
        (row["dataset"], row["group"], row["sha256"]) for row in used
    ]
    if len(set(identities)) != len(identities) or not used:
        raise ValueError(
            "Prior sources must be nonempty without duplicate clips"
        )
    output.mkdir(parents=True)
    rows = []
    for index, source in enumerate(used):
        arrays, trajectory = motion_kinematics(tree, source["file"])
        n = len(arrays["position"]) - 5
        future = torch.arange(n)[:, None] + torch.arange(1, 6)[None]
        targets = []
        for role in range(trajectory["roles"]):
            targets.append(
                target_from_world(
                    arrays["position"][:n, role],
                    arrays["quaternion"][:n, role],
                    arrays["position"][future, role],
                    arrays["quaternion"][future, role],
                )
            )
        values = torch.cat(targets).numpy()
        path = output / f"reference-{index:05d}.npz"
        np.savez_compressed(path, target_pose=values)
        row = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "dataset": source["dataset"],
            "group": source["group"],
            "split": splits[source["group"]],
            "samples": len(values),
            "frames": n + 5,
            "roles": trajectory["roles"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    manifest = {
        "format": FORMAT,
        "asset_identity": tree.identity(),
        "control_hz": 50,
        "future_offsets": [1, 2, 3, 4, 5],
        "coordinate_frame": "source_current_pelvis_full_3d",
        "input": "reference_only_225",
        "synthetic_opponent_samples": 0,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "excluded_groups": sorted(excluded),
        "held_out_test_groups": sorted(
            {row["group"] for row in records if splits[row["group"]] == "test"}
        ),
        "records": rows,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--exclude-group", action="append", default=[])
    args = parser.parse_args(argv)
    prepare(
        args.source,
        args.assets,
        args.output,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        exclude_groups=args.exclude_group,
    )


if __name__ == "__main__":
    main()
