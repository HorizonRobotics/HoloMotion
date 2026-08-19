"""Simulator-independent helpers for G1 morphology and payload variation."""

from __future__ import annotations
import fcntl
import glob
import hashlib
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import torch

_VARIANT_CACHE_SCHEMA = "holomotion-g1-variant-cache-v1"
_VARIANT_MANIFEST_SCHEMA = "holomotion-g1-variant-manifest-v1"


@dataclass(frozen=True)
class ResolvedRobotVariant:
    """One immutable URDF selection shared by scene and reference FK."""

    urdf_path: str
    urdf_sha256: str
    variant_index: int | None
    process_id: int


def _urdf_sha256(urdf_path: str) -> str:
    digest = hashlib.sha256()
    with open(urdf_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_source_signature(
    urdf_paths: Sequence[str], *, conversion_options: dict[str, Any]
) -> str:
    """Hash variant sources and conversion options for a shared USD cache.

    URDF contents are hashed exactly. Referenced mesh metadata is included once
    per resolved file so mesh updates invalidate the ready marker without every
    distributed rank repeatedly hashing large binary assets.
    """

    digest = hashlib.sha256()
    digest.update(_VARIANT_CACHE_SCHEMA.encode())
    digest.update(
        json.dumps(
            conversion_options,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    mesh_paths: set[Path] = set()
    for raw_path in urdf_paths:
        urdf_path = Path(raw_path).expanduser().resolve()
        digest.update(str(urdf_path).encode())
        digest.update(urdf_path.read_bytes())
        root = ElementTree.parse(urdf_path).getroot()
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename or "://" in filename:
                continue
            mesh_paths.add((urdf_path.parent / filename).resolve())

    for mesh_path in sorted(mesh_paths):
        stat = mesh_path.stat()
        digest.update(str(mesh_path).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def prepare_shared_file_cache(
    cache_root: str,
    *,
    cache_key: str,
    expected_paths: Sequence[str],
    populate: Callable[[], Sequence[str]],
    force: bool = False,
) -> tuple[list[str], bool]:
    """Populate a cache once across ranks sharing the same filesystem.

    The filesystem lock elects one writer. On a shared filesystem this is one
    writer for the whole job; on node-local storage it naturally becomes one
    writer per node. A ready marker is published atomically only after all
    expected outputs exist.

    Returns:
        Absolute expected paths and whether this caller ran ``populate``.
    """

    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected = [
        str(Path(path).expanduser().resolve()) for path in expected_paths
    ]
    lock_path = root / ".holomotion-variant-cache.lock"
    marker_path = root / f".holomotion-variant-cache-{cache_key}.ready"

    prepared = False
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cache_ready = marker_path.is_file() and all(
            Path(path).is_file() for path in expected
        )
        if force or not cache_ready:
            produced = [
                str(Path(path).expanduser().resolve()) for path in populate()
            ]
            missing = [path for path in expected if not Path(path).is_file()]
            if missing:
                raise RuntimeError(
                    "Variant cache population did not create expected files: "
                    + ", ".join(missing)
                )
            if set(produced) != set(expected):
                raise RuntimeError(
                    "Variant cache population returned unexpected paths. "
                    f"expected={expected}, produced={produced}"
                )
            temporary_marker = marker_path.with_name(
                f"{marker_path.name}.tmp.{os.getpid()}"
            )
            temporary_marker.write_text(
                json.dumps({"cache_key": cache_key, "paths": expected}) + "\n"
            )
            os.replace(temporary_marker, marker_path)
            prepared = True
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return expected, prepared


def resolve_urdf_variants(asset_config: Any) -> list[str]:
    """Resolve an explicit list or deterministic glob of URDF variants."""

    variants = list(asset_config.get("urdf_variants", []) or [])
    variant_glob = asset_config.get("urdf_variant_glob")
    variant_manifest = asset_config.get("urdf_variant_manifest")
    configured_sources = sum(
        bool(source) for source in (variants, variant_glob, variant_manifest)
    )
    if configured_sources > 1:
        raise ValueError(
            "Configure only one of urdf_variants, urdf_variant_glob, or "
            "urdf_variant_manifest."
        )
    if variant_manifest:
        manifest_path = Path(str(variant_manifest)).expanduser().resolve()
        payload = json.loads(manifest_path.read_text())
        if payload.get("schema") != _VARIANT_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported G1 variant manifest schema in {manifest_path}: "
                f"{payload.get('schema')!r}"
            )
        entries = payload.get("variants")
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"G1 variant manifest has no variants: {manifest_path}"
            )
        variant_ids: set[str] = set()
        resolved_paths: set[str] = set()
        variants = []
        for entry in entries:
            variant_id = str(entry["variant_id"])
            if variant_id in variant_ids:
                raise ValueError(
                    f"Duplicate variant_id {variant_id!r} in {manifest_path}"
                )
            urdf_path = Path(str(entry["urdf_path"]))
            if not urdf_path.is_absolute():
                urdf_path = manifest_path.parent / urdf_path
            resolved_path = str(urdf_path.resolve())
            if resolved_path in resolved_paths:
                raise ValueError(
                    f"Duplicate URDF path {resolved_path!r} in {manifest_path}"
                )
            if not Path(resolved_path).is_file():
                raise FileNotFoundError(resolved_path)
            expected_sha256 = str(entry["urdf_sha256"])
            actual_sha256 = _urdf_sha256(resolved_path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"URDF SHA-256 mismatch for {variant_id}: "
                    f"expected={expected_sha256}, actual={actual_sha256}, "
                    f"path={resolved_path}"
                )
            variant_ids.add(variant_id)
            resolved_paths.add(resolved_path)
            variants.append(resolved_path)
    elif variant_glob:
        variants = sorted(glob.glob(str(variant_glob)))
    expected_count = asset_config.get("urdf_variant_expected_count")
    if expected_count is not None and len(variants) != int(expected_count):
        raise ValueError(
            f"Expected {int(expected_count)} URDF variants, found "
            f"{len(variants)}."
        )
    return [str(path) for path in variants]


def balanced_variant_schedule_period(
    num_variants: int, world_size: int
) -> int:
    """Return the shortest segment cycle with exact variant exposure."""

    if num_variants <= 0:
        raise ValueError("num_variants must be positive")
    if world_size <= 0 or world_size > num_variants:
        raise ValueError(
            "world_size must be in [1, num_variants] for unique rank variants"
        )
    return num_variants // math.gcd(num_variants, world_size)


def build_balanced_variant_rank_map(
    num_variants: int,
    world_size: int,
    segment_index: int,
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Build one deterministic, unique, exposure-balanced rank mapping.

    Consecutive segments advance by ``world_size`` in the fixed manifest. Over
    one schedule period, every variant receives exactly the same number of
    training segments. A SHA-based ordering redistributes active variants
    across ranks without relying on Python's random implementation.
    """

    balanced_variant_schedule_period(num_variants, world_size)
    segment_index = int(segment_index)
    start = (segment_index * world_size) % num_variants
    active = [(start + index) % num_variants for index in range(world_size)]

    def rank_key(variant_index: int) -> bytes:
        return hashlib.sha256(
            f"{int(seed)}:{segment_index}:{variant_index}".encode()
        ).digest()

    return tuple(sorted(active, key=rank_key))


def resolve_rank_urdf_path(
    asset_config: Any,
    process_id: int,
    num_processes: int | None = None,
) -> tuple[str, int | None]:
    """Resolve one homogeneous URDF per distributed rank."""

    variants = resolve_urdf_variants(asset_config)
    if not variants:
        urdf_file = asset_config.get("urdf_file")
        if urdf_file is None:
            urdf_file = asset_config.urdf_file
        return str(urdf_file), None

    mode = str(asset_config.get("urdf_variant_mode", "per_rank")).lower()
    if mode not in {"per_rank", "per_env"}:
        raise ValueError(
            "urdf_variant_mode must be either 'per_rank' or 'per_env'."
        )
    if mode == "per_env":
        return str(variants[0]), None

    process_id = int(process_id)
    explicit_rank_map = list(
        asset_config.get("urdf_variant_rank_map", []) or []
    )
    schedule = str(asset_config.get("urdf_variant_schedule", "") or "").lower()
    if explicit_rank_map and schedule:
        raise ValueError(
            "Configure only one of urdf_variant_rank_map or "
            "urdf_variant_schedule."
        )
    if explicit_rank_map:
        if num_processes is not None and len(explicit_rank_map) != int(
            num_processes
        ):
            raise ValueError(
                "urdf_variant_rank_map length must equal num_processes: "
                f"map={len(explicit_rank_map)}, num_processes={num_processes}"
            )
        if len(set(map(int, explicit_rank_map))) != len(explicit_rank_map):
            raise ValueError(
                "urdf_variant_rank_map must contain unique indices"
            )
        if process_id < 0 or process_id >= len(explicit_rank_map):
            raise ValueError(
                f"process_id {process_id} is outside urdf_variant_rank_map"
            )
        variant_index = int(explicit_rank_map[process_id])
        if variant_index < 0 or variant_index >= len(variants):
            raise ValueError(
                f"Variant index {variant_index} is outside "
                f"[0, {len(variants)})"
            )
    elif schedule:
        if schedule != "balanced_cycle":
            raise ValueError(
                "urdf_variant_schedule must be 'balanced_cycle' when enabled"
            )
        if num_processes is None:
            raise ValueError(
                "num_processes is required for balanced variant scheduling"
            )
        rank_map = build_balanced_variant_rank_map(
            len(variants),
            int(num_processes),
            int(asset_config.get("urdf_variant_schedule_segment", 0)),
            seed=int(asset_config.get("urdf_variant_schedule_seed", 0)),
        )
        if process_id < 0 or process_id >= len(rank_map):
            raise ValueError(
                f"process_id {process_id} is outside world size "
                f"{len(rank_map)}"
            )
        variant_index = rank_map[process_id]
    else:
        rank_offset = int(asset_config.get("urdf_variant_rank_offset", 0))
        variant_index = (process_id + rank_offset) % len(variants)
    return str(variants[variant_index]), variant_index


def resolve_rank_robot_variant(
    asset_config: Any,
    process_id: int,
    num_processes: int | None = None,
) -> ResolvedRobotVariant:
    """Resolve and fingerprint the homogeneous robot used by one rank."""

    urdf_path, variant_index = resolve_rank_urdf_path(
        asset_config, process_id, num_processes
    )
    resolved_path = str(Path(urdf_path).expanduser().resolve())
    if not Path(resolved_path).is_file():
        raise FileNotFoundError(f"URDF file not found: {resolved_path}")
    return ResolvedRobotVariant(
        urdf_path=resolved_path,
        urdf_sha256=_urdf_sha256(resolved_path),
        variant_index=variant_index,
        process_id=int(process_id),
    )


def bind_rank_robot_variant(
    robot_config: Any,
    command_config: Any,
    resolved_variant: ResolvedRobotVariant,
) -> None:
    """Bind one resolved URDF to both the scene and motion-FK configs."""

    robot_config.asset["urdf_file"] = resolved_variant.urdf_path
    robot_config.motion["fk_robot_file_path"] = resolved_variant.urdf_path

    ref_motion = command_config.get("ref_motion")
    if ref_motion is None:
        return
    motion_lib_cfg = ref_motion["params"]["motion_lib_cfg"]
    motion_lib_cfg["fk_robot_file_path"] = resolved_variant.urdf_path
    if robot_config.asset["urdf_file"] != motion_lib_cfg["fk_robot_file_path"]:
        raise RuntimeError(
            "Scene and reference FK resolved different rank URDF paths."
        )


def sample_correlated_hand_payloads(
    num_envs: int,
    *,
    object_mass_range: tuple[float, float],
    hardware_mass_range: tuple[float, float] = (0.0, 0.0),
    mode_probabilities: tuple[float, float, float, float] = (
        0.25,
        0.25,
        0.25,
        0.25,
    ),
    bimanual_split_range: tuple[float, float] = (0.3, 0.7),
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample left/right added mass while bounding held-object total mass.

    Modes are ordered as none, left, right, and bimanual. Hardware mass is
    sampled independently per wrist; object mass is sampled once per robot and
    never exceeds ``object_mass_range[1]`` across both hands.
    """

    if num_envs < 0:
        raise ValueError("num_envs must be non-negative.")
    object_low, object_high = map(float, object_mass_range)
    hardware_low, hardware_high = map(float, hardware_mass_range)
    split_low, split_high = map(float, bimanual_split_range)
    if not 0.0 <= object_low <= object_high:
        raise ValueError("object_mass_range must satisfy 0 <= low <= high.")
    if not 0.0 <= hardware_low <= hardware_high:
        raise ValueError("hardware_mass_range must satisfy 0 <= low <= high.")
    if not 0.0 <= split_low <= split_high <= 1.0:
        raise ValueError(
            "bimanual_split_range must satisfy 0 <= low <= high <= 1."
        )

    probabilities = torch.as_tensor(
        mode_probabilities, dtype=torch.float32, device=device
    )
    if probabilities.shape != (4,) or bool((probabilities < 0).any()):
        raise ValueError(
            "mode_probabilities must contain four non-negative values."
        )
    probability_sum = float(probabilities.sum().item())
    if probability_sum <= 0.0:
        raise ValueError("mode_probabilities must have a positive sum.")
    probabilities = probabilities / probability_sum

    hardware = torch.empty((num_envs, 2), device=device).uniform_(
        hardware_low, hardware_high
    )
    object_by_hand = torch.zeros((num_envs, 2), device=device)
    if num_envs == 0:
        return hardware, object_by_hand

    modes = torch.multinomial(probabilities, num_envs, replacement=True)
    total_object_mass = torch.empty(num_envs, device=device).uniform_(
        object_low, object_high
    )
    left_mask = modes == 1
    right_mask = modes == 2
    bimanual_mask = modes == 3
    object_by_hand[left_mask, 0] = total_object_mass[left_mask]
    object_by_hand[right_mask, 1] = total_object_mass[right_mask]
    if bool(bimanual_mask.any()):
        split = torch.empty(
            int(bimanual_mask.sum().item()), device=device
        ).uniform_(split_low, split_high)
        bimanual_total = total_object_mass[bimanual_mask]
        object_by_hand[bimanual_mask, 0] = bimanual_total * split
        object_by_hand[bimanual_mask, 1] = bimanual_total * (1.0 - split)

    return hardware + object_by_hand, object_by_hand
