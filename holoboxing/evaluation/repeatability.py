"""Compare complete match results, excluding identities and wall-clock timing.

Run with ``python -m holoboxing.evaluation.repeatability FIRST_DIR SECOND_DIR``.
Both directories must contain a match JSON and replay.hdf5. A mismatch or
incomplete artifact exits nonzero. No GPU, model execution or physics is used.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

FRAME_TIMING = frozenset(
    {
        "policy_start_ns",
        "policy_end_ns",
        "upper_policy_start_ns",
        "upper_policy_end_ns",
        "competitor_sanitization_ns",
        "control_step_start_ns",
        "control_step_end_ns",
    }
)
SUMMARY_TIMING = frozenset({"avg_fps", "avg_latency_ms"})
METADATA_IDENTITY = frozenset(
    {
        "match_id",
        "control_loop_benchmark",
        "policy_path",
        "retargeted_path",
        "release_config",
    }
)


def _without(values, excluded):
    return {key: value for key, value in values.items() if key not in excluded}


def _payload(directory):
    for name in ("match.json", "evaluation.json"):
        path = directory / name
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(
        f"Missing match.json or evaluation.json in {directory}"
    )


def compare(first, second):
    first, second = Path(first), Path(second)
    a, b = _payload(first), _payload(second)
    # A zero-frame or truncated file cannot prove a completed match repeats.
    for payload in (a, b):
        if (
            not payload["frames"]
            or payload["metadata"].get("match_completed") is not True
            or payload["metadata"].get("frames_recorded")
            != len(payload["frames"])
        ):
            raise ValueError("Complete nonempty match results are required")
    summary_equal = _without(a["summary"], SUMMARY_TIMING) == _without(
        b["summary"], SUMMARY_TIMING
    )
    metadata_equal = _without(a["metadata"], METADATA_IDENTITY) == _without(
        b["metadata"], METADATA_IDENTITY
    )
    frame_mismatches = []
    frames_equal = len(a["frames"]) == len(b["frames"])
    for index, (left, right) in enumerate(zip(a["frames"], b["frames"])):
        if _without(left, FRAME_TIMING) != _without(right, FRAME_TIMING):
            frames_equal = False
            if len(frame_mismatches) < 20:
                frame_mismatches.append(
                    {
                        "frame": index,
                        "fields": [
                            key
                            for key in sorted(set(left) | set(right))
                            if key not in FRAME_TIMING
                            and left.get(key) != right.get(key)
                        ],
                    }
                )
    datasets = {}
    with (
        h5py.File(first / "replay.hdf5") as f,
        h5py.File(second / "replay.hdf5") as g,
    ):

        def names(file):
            result = set()
            file.visititems(
                lambda name, item: result.add(name)
                if isinstance(item, h5py.Dataset)
                else None
            )
            return {
                name
                for name in result
                if name != "meta/json"
                and name.rsplit("/", 1)[-1] not in FRAME_TIMING
            }

        left_names, right_names = names(f), names(g)
        if not left_names or not right_names:
            raise ValueError("Replay datasets are required")
        for name in sorted(left_names | right_names):
            if name not in left_names or name not in right_names:
                datasets[name] = {
                    "exact": False,
                    "missing_in": "first"
                    if name not in left_names
                    else "second",
                }
                continue
            x, y = f[name][...], g[name][...]
            item = {
                "exact": x.dtype == y.dtype and np.array_equal(x, y),
                "first_shape": list(x.shape),
                "second_shape": list(y.shape),
            }
            if (
                x.shape == y.shape
                and np.issubdtype(x.dtype, np.number)
                and np.issubdtype(y.dtype, np.number)
            ):
                delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                item["max_absolute_difference"] = (
                    float(delta.max()) if delta.size else 0.0
                )
                item["unequal_elements"] = int(np.count_nonzero(x != y))
            datasets[name] = item
    replay_equal = all(item["exact"] for item in datasets.values())
    return {
        "status": "PASS"
        if summary_equal and metadata_equal and frames_equal and replay_equal
        else "FAIL",
        "frames": [len(a["frames"]), len(b["frames"])],
        "summary_exact": summary_equal,
        "configuration_exact": metadata_equal,
        "frames_exact": frames_equal,
        "replay_exact": replay_equal,
        "first_frame_mismatches": frame_mismatches,
        "datasets": datasets,
        "excluded_frame_fields": sorted(FRAME_TIMING),
        "excluded_summary_fields": sorted(SUMMARY_TIMING),
        "excluded_metadata_fields": sorted(METADATA_IDENTITY),
        "excluded_replay_metadata": "meta/json; source match metadata is checked above",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args(argv)
    result = compare(args.first, args.second)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
