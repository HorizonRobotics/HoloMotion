"""Strict HDF5 replay I/O with immutable, atomically finalized artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .schema import STATE_SCHEMA


class ReplayValidationError(ValueError):
    pass


@dataclass
class ReplayData:
    meta: dict
    time: np.ndarray
    state: np.ndarray
    score: np.ndarray
    rounds: np.ndarray
    events: list[dict]


def state_dataset(meta: dict) -> str:
    schema = meta.get("state_schema")
    if not isinstance(schema, dict):
        raise ReplayValidationError("Missing state schema")
    if (
        not isinstance(schema.get("id"), str)
        or not schema["id"]
        or not isinstance(schema.get("version"), (int, str))
    ):
        raise ReplayValidationError("State schema requires an id and version")
    width = schema.get("width")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not 0 < width <= 10_000
    ):
        raise ReplayValidationError("Invalid state schema width")
    dataset = schema.get("dataset", "frames/state")
    if dataset not in {"frames/state", "frames/mujoco_state"}:
        raise ReplayValidationError("Unsupported state dataset path")
    return dataset


def _validate_mock_state(data: ReplayData) -> None:
    """Layout-specific checks are an extension, not the replay container contract."""
    schema = {
        key: value
        for key, value in data.meta["state_schema"].items()
        if key != "dataset"
    }
    if schema != STATE_SCHEMA:
        raise ReplayValidationError(
            "Unsupported mock-boxing-v1 layout revision"
        )
    for robot in schema["robots"].values():
        q = robot["fields"]["root_quaternion"]["offset"]
        if not np.allclose(
            np.linalg.norm(data.state[:, q : q + 4], axis=1),
            1.0,
            atol=1e-8,
            rtol=0,
        ):
            raise ReplayValidationError("Root quaternions must be normalized")


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def _vector(value: object, size: int, label: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != size
        or not all(_finite_number(v) for v in value)
    ):
        raise ReplayValidationError(
            f"{label} must contain {size} finite numbers"
        )
    return np.asarray(value, dtype=np.float64)


def _unit_quaternions(values: np.ndarray, label: str) -> None:
    # Runtime state is commonly exported from float32 engines; do not require
    # float64 machine precision, and never silently normalize damaged records.
    if not np.allclose(
        np.linalg.norm(values, axis=-1), 1.0, rtol=0, atol=1e-4
    ):
        raise ReplayValidationError(
            f"{label} must be normalized wxyz quaternions"
        )


def _validate_articulated_state(data: ReplayData) -> None:
    schema = data.meta["state_schema"]
    if schema.get("version") != 1 or not _integer(schema.get("version")):
        raise ReplayValidationError(
            "Unsupported generic_articulated_replay_v1 layout revision"
        )
    if schema.get("dataset", "frames/state") != "frames/state":
        raise ReplayValidationError(
            "Generic articulated poses must use frames/state"
        )
    if (
        schema.get("coordinate_system") != "right-handed-z-up"
        or schema.get("quaternion_order") != "wxyz"
    ):
        raise ReplayValidationError(
            "Generic articulated state requires right-handed-z-up and wxyz"
        )
    entities = schema.get("entities")
    if not isinstance(entities, dict) or not entities:
        raise ReplayValidationError(
            "Generic articulated state requires entities"
        )
    used_columns = np.zeros(schema["width"], dtype=bool)
    geometry_dimensions = {"sphere": 1, "box": 3, "capsule": 2, "cylinder": 2}
    for entity_id, entity in entities.items():
        if (
            not isinstance(entity_id, str)
            or not entity_id
            or not isinstance(entity, dict)
        ):
            raise ReplayValidationError(
                "Entities need nonempty string IDs and object descriptors"
            )
        names = entity.get("body_names")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ReplayValidationError(
                f"{entity_id}: body_names must be nonempty and unique"
            )
        count = len(names)
        parents = entity.get("parent_indices")
        if (
            not isinstance(parents, list)
            or len(parents) != count
            or any(not _integer(p) or p < -1 or p >= count for p in parents)
        ):
            raise ReplayValidationError(f"{entity_id}: invalid parent_indices")
        # Runtime body order need not be parent-first. Validate a forest without
        # recursion (long articulations must not exhaust the Python call stack).
        visited = [0] * count
        for start in range(count):
            current = start
            chain = []
            while current != -1 and visited[current] == 0:
                visited[current] = 1
                chain.append(current)
                current = parents[current]
            if current != -1 and visited[current] == 1:
                raise ReplayValidationError(
                    f"{entity_id}: parent_indices contain a cycle"
                )
            for index in chain:
                visited[index] = 2
        fields = entity.get("fields")
        if (
            not isinstance(fields, dict)
            or {"body_world_position", "body_world_quaternion"} - fields.keys()
        ):
            raise ReplayValidationError(
                f"{entity_id}: missing world body pose fields"
            )
        for name, field in fields.items():
            if not isinstance(field, dict):
                raise ReplayValidationError(
                    f"{entity_id}.{name}: field must be an object"
                )
            offset, shape = field.get("offset"), field.get("shape")
            if (
                not _integer(offset)
                or offset < 0
                or not isinstance(shape, list)
                or not shape
                or any(not _integer(v) or v <= 0 for v in shape)
            ):
                raise ReplayValidationError(
                    f"{entity_id}.{name}: invalid offset or shape"
                )
            size = 1
            for dimension in shape:
                size *= dimension
                if size > schema["width"]:
                    raise ReplayValidationError(
                        f"{entity_id}.{name}: field exceeds state width"
                    )
            if offset + size > schema["width"] or np.any(
                used_columns[offset : offset + size]
            ):
                raise ReplayValidationError(
                    f"{entity_id}.{name}: field is out of bounds or overlaps another field"
                )
            if "size" in field and (
                not _integer(field["size"]) or field["size"] != size
            ):
                raise ReplayValidationError(
                    f"{entity_id}.{name}: size differs from shape"
                )
            used_columns[offset : offset + size] = True
        position = fields["body_world_position"]
        quaternion = fields["body_world_quaternion"]
        if position["shape"] != [count, 3] or quaternion["shape"] != [
            count,
            4,
        ]:
            raise ReplayValidationError(
                f"{entity_id}: world pose shape differs from body_names"
            )
        q = quaternion["offset"]
        _unit_quaternions(
            data.state[:, q : q + count * 4].reshape(-1, count, 4), entity_id
        )
        geometries = entity.get("geometries")
        if not isinstance(geometries, list):
            raise ReplayValidationError(
                f"{entity_id}: geometries must be an array"
            )
        for geometry in geometries:
            if not isinstance(geometry, dict):
                raise ReplayValidationError(
                    f"{entity_id}: geometry must be an object"
                )
            index, kind = geometry.get("body_index"), geometry.get("type")
            if not _integer(index) or not 0 <= index < count:
                raise ReplayValidationError(
                    f"{entity_id}: geometry has invalid body_index"
                )
            if not isinstance(kind, str) or kind not in geometry_dimensions:
                raise ReplayValidationError(
                    f"{entity_id}: unsupported geometry type"
                )
            dimensions = _vector(
                geometry.get("dimensions"),
                geometry_dimensions[kind],
                f"{entity_id}.{kind} dimensions",
            )
            # Capsule length is its cylindrical segment (hemispheres excluded).
            # A zero-length capsule is a valid sphere; all other lengths > 0.
            if np.any(dimensions <= 0) and not (
                kind == "capsule" and dimensions[0] > 0 and dimensions[1] == 0
            ):
                raise ReplayValidationError(
                    f"{entity_id}.{kind}: geometry dimensions must be positive"
                )
            _vector(
                geometry.get("local_position"),
                3,
                f"{entity_id}.{kind} local_position",
            )
            local_q = _vector(
                geometry.get("local_quaternion"),
                4,
                f"{entity_id}.{kind} local_quaternion",
            )
            _unit_quaternions(local_q, f"{entity_id}.{kind} local_quaternion")
            if "color" in geometry:
                color = _vector(
                    geometry["color"], 4, f"{entity_id}.{kind} color"
                )
                if np.any((color < 0) | (color > 1)):
                    raise ReplayValidationError(
                        f"{entity_id}.{kind}: RGBA color must lie in [0,1]"
                    )


def _validate_explicit_time(data: ReplayData) -> None:
    meta = data.meta
    timebase = meta.get("timebase")
    if (
        not isinstance(timebase, dict)
        or timebase.get("mode") != "explicit_timestamps"
    ):
        raise ReplayValidationError(
            "Replay 1.1 requires an explicit_timestamps timebase"
        )
    nominal = timebase.get("nominal_step_seconds")
    if not _finite_number(nominal) or nominal <= 0:
        raise ReplayValidationError(
            "nominal_step_seconds must be finite and positive"
        )
    if abs(data.time[0]) > 1e-10 or np.any(np.diff(data.time) <= 0):
        raise ReplayValidationError(
            "Explicit timestamps must start at zero and strictly increase"
        )
    if "control_fps" in meta:
        fps = meta["control_fps"]
        if (
            not _finite_number(fps)
            or fps <= 0
            or not np.isclose(fps * nominal, 1.0, rtol=0, atol=1e-8)
        ):
            raise ReplayValidationError(
                "Optional control_fps must equal reciprocal nominal_step_seconds"
            )
    if "physics_dt" in timebase and (
        not _finite_number(timebase["physics_dt"])
        or timebase["physics_dt"] <= 0
    ):
        raise ReplayValidationError("physics_dt must be finite and positive")
    if "decimation" in timebase and (
        not _integer(timebase["decimation"]) or timebase["decimation"] <= 0
    ):
        raise ReplayValidationError("decimation must be a positive integer")
    if (
        "physics_dt" in timebase
        and "decimation" in timebase
        and not np.isclose(
            timebase["physics_dt"] * timebase["decimation"],
            nominal,
            rtol=0,
            atol=1e-10,
        )
    ):
        raise ReplayValidationError(
            "Physics timing differs from nominal control step"
        )


STATE_VALIDATORS = {
    "mock-boxing-v1": _validate_mock_state,
    "generic_articulated_replay_v1": _validate_articulated_state,
}


def validate_replay(data: ReplayData) -> None:
    m = data.meta
    required = {
        "schema_version",
        "match_id",
        "seed",
        "duration_seconds",
        "frame_count",
        "state_schema",
        "red",
        "blue",
        "environment",
        "result",
        "development_only",
    }
    if not isinstance(m, dict) or required - m.keys():
        raise ReplayValidationError("Missing replay metadata fields")
    if m["schema_version"] not in {"1.0", "1.1"}:
        raise ReplayValidationError("Unsupported replay schema")
    if m["schema_version"] == "1.0" and "control_fps" not in m:
        raise ReplayValidationError("Missing replay metadata fields")
    state_dataset(m)
    if m["development_only"] is not True:
        raise ReplayValidationError(
            "Mock replay must be labeled development_only"
        )
    for color in ("red", "blue"):
        if (
            not isinstance(m[color], dict)
            or {"team_id", "model_id", "model_sha256", "controller_kind"}
            - m[color].keys()
        ):
            raise ReplayValidationError(f"Missing {color} model provenance")
    if (
        not isinstance(m["environment"], dict)
        or {
            "environment_version",
            "simulator_version",
            "asset_version",
            "asset_sha256",
            "controller_version",
            "scoring_version",
        }
        - m["environment"].keys()
    ):
        raise ReplayValidationError(
            "Missing evaluation environment provenance"
        )
    t = len(data.time)
    if t < 2 or m["frame_count"] != t:
        raise ReplayValidationError("Invalid frame count")
    expected = [
        (data.time, (t,)),
        (data.state, (t, m["state_schema"]["width"])),
        (data.score, (t, 2)),
        (data.rounds, (t,)),
    ]
    if any(a.shape != shape for a, shape in expected):
        raise ReplayValidationError("Invalid frame dataset shape")
    if any(not np.isfinite(a).all() for a, _ in expected):
        raise ReplayValidationError("Nonfinite frame values")
    if not np.issubdtype(data.rounds.dtype, np.integer) or np.any(
        data.rounds < 1
    ):
        raise ReplayValidationError("Round indices must be positive integers")
    if m["schema_version"] == "1.0":
        fps = m["control_fps"]
        if (
            not isinstance(fps, int)
            or fps <= 0
            or not np.allclose(
                data.time, np.arange(t) / fps, rtol=0, atol=1e-10
            )
        ):
            raise ReplayValidationError(
                "Time must start at zero and advance at control_fps"
            )
    else:
        _validate_explicit_time(data)
    if (
        not np.isfinite(m["duration_seconds"])
        or abs(data.time[-1] - m["duration_seconds"]) > 1e-10
    ):
        raise ReplayValidationError("Duration does not match final frame")
    validator = STATE_VALIDATORS.get(m["state_schema"]["id"])
    if validator:
        validator(data)
    last_frame = -1
    for event in data.events:
        if {"frame", "type", "actor", "target", "value"} - event.keys():
            raise ReplayValidationError("Missing event fields")
        frame = event["frame"]
        if (
            not isinstance(frame, (int, np.integer))
            or not last_frame <= frame < t
            or frame < 0
        ):
            raise ReplayValidationError(
                "Event frame out of range or out of order"
            )
        if (
            any(
                not isinstance(event[field], str)
                for field in ("type", "actor", "target")
            )
            or not event["type"]
            or not np.isfinite(event["value"])
        ):
            raise ReplayValidationError("Invalid event payload")
        last_frame = frame
    result = m["result"]
    if (
        not isinstance(result, dict)
        or {"score_red", "score_blue", "winner"} - result.keys()
    ):
        raise ReplayValidationError("Missing result metadata")
    if not np.array_equal(
        data.score[-1], [result["score_red"], result["score_blue"]]
    ):
        raise ReplayValidationError("Result score differs from final frame")
    if result["winner"] not in {"red", "blue", "draw"}:
        raise ReplayValidationError("Invalid result winner")
    if m["environment"]["scoring_version"] == "development-only-mock-v1":
        winner = (
            "red"
            if data.score[-1, 0] > data.score[-1, 1]
            else "blue"
            if data.score[-1, 1] > data.score[-1, 0]
            else "draw"
        )
        if result["winner"] != winner:
            raise ReplayValidationError(
                "Mock scoring winner differs from final score"
            )


def write_replay(path: Path, data: ReplayData) -> Path:
    """Publish exactly once. Existing target files are never replaced."""
    validate_replay(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        strings = h5py.string_dtype(encoding="utf-8")
        with h5py.File(temp, "w") as file:
            file.create_dataset(
                "meta/json",
                data=json.dumps(
                    data.meta, ensure_ascii=False, allow_nan=False
                ),
                dtype=strings,
            )
            for name, values, dtype in (
                ("frames/time", data.time, "<f8"),
                (state_dataset(data.meta), data.state, "<f8"),
                ("frames/score", data.score, "<f8"),
                ("frames/round", data.rounds, "<i4"),
            ):
                file.create_dataset(
                    name,
                    data=values,
                    dtype=dtype,
                    compression="gzip",
                    compression_opts=4,
                )
            for key, dtype in (
                ("frame", "<i8"),
                ("type", strings),
                ("actor", strings),
                ("target", strings),
                ("value", "<f8"),
            ):
                file.create_dataset(
                    f"events/{key}",
                    data=[e[key] for e in data.events],
                    dtype=dtype,
                )
            file.flush()
        with temp.open("rb") as source:
            os.fsync(source.fileno())
        # link is an atomic no-clobber publish on the same filesystem.
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return path
    finally:
        temp.unlink(missing_ok=True)


def read_replay(path: Path) -> ReplayData:
    try:
        with h5py.File(path, "r") as file:
            if (
                "meta/json" not in file
                or not isinstance(file["meta/json"], h5py.Dataset)
                or file["meta/json"].shape != ()
            ):
                raise ReplayValidationError(
                    "meta/json must be a scalar UTF-8 JSON string"
                )
            raw_meta = file["meta/json"].asstr()[()]
            if len(raw_meta.encode("utf-8")) > 1_000_000:
                raise ReplayValidationError("Metadata too large")
            meta = json.loads(raw_meta)
            if not isinstance(meta, dict):
                raise ReplayValidationError("Metadata must be a JSON object")
            state_path = state_dataset(meta)
            datasets = {
                "meta/json",
                "frames/time",
                state_path,
                "frames/score",
                "frames/round",
                "events/frame",
                "events/type",
                "events/actor",
                "events/target",
                "events/value",
            }
            if any(
                name not in file or not isinstance(file[name], h5py.Dataset)
                for name in datasets
            ):
                raise ReplayValidationError("Missing required HDF5 datasets")
            if file["meta/json"].shape != ():
                raise ReplayValidationError(
                    "meta/json must be a scalar UTF-8 JSON string"
                )
            if (
                sum(
                    file[name].size * file[name].dtype.itemsize
                    for name in datasets
                )
                > 128_000_000
            ):
                raise ReplayValidationError(
                    "Replay exceeds development reader allocation limit"
                )
            for name, kind, size in (
                ("frames/time", "f", 8),
                (state_path, "f", 8),
                ("frames/score", "f", 8),
                ("frames/round", "i", 4),
                ("events/frame", "i", 8),
                ("events/value", "f", 8),
            ):
                if (
                    file[name].dtype.kind != kind
                    or file[name].dtype.itemsize != size
                ):
                    raise ReplayValidationError(f"Invalid dtype for {name}")
            event_datasets = [
                file[f"events/{name}"]
                for name in ("frame", "type", "actor", "target", "value")
            ]
            if (
                any(d.ndim != 1 for d in event_datasets)
                or len({d.shape for d in event_datasets}) != 1
            ):
                raise ReplayValidationError(
                    "Event datasets must have equal one-dimensional lengths"
                )
            event_columns = [
                d.asstr()[:] if index in (1, 2, 3) else d[:]
                for index, d in enumerate(event_datasets)
            ]
            events = [
                {
                    "frame": int(f),
                    "type": kind,
                    "actor": actor,
                    "target": target,
                    "value": float(value),
                }
                for f, kind, actor, target, value in zip(*event_columns)
            ]
            data = ReplayData(
                meta,
                file["frames/time"][:],
                file[state_path][:],
                file["frames/score"][:],
                file["frames/round"][:],
                events,
            )
        validate_replay(data)
        return data
    except ReplayValidationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ReplayValidationError(f"Invalid HDF5 replay: {exc}") from exc
