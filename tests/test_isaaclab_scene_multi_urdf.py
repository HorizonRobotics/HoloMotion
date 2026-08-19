import hashlib
import json
from types import SimpleNamespace

import pytest

from holomotion.src.utils.g1_variants import (
    balanced_variant_schedule_period,
    bind_rank_robot_variant,
    build_balanced_variant_rank_map,
    prepare_shared_file_cache,
    resolve_rank_robot_variant,
    resolve_rank_urdf_path,
    resolve_urdf_variants,
    variant_source_signature,
)


def test_balanced_rank_schedule_covers_66_variants_fairly_on_64_ranks():
    num_variants = 66
    world_size = 64
    exposure = [0] * num_variants

    assert balanced_variant_schedule_period(num_variants, world_size) == 33
    for segment_index in range(6):
        rank_map = build_balanced_variant_rank_map(
            num_variants, world_size, segment_index, seed=17
        )
        assert len(rank_map) == world_size
        assert len(set(rank_map)) == world_size
        for variant_index in rank_map:
            exposure[variant_index] += 1

    assert min(exposure) == 5
    assert max(exposure) == 6
    assert sum(exposure) == 6 * world_size


def test_balanced_rank_schedule_scales_to_hundreds_without_long_cycle():
    num_variants = 300
    world_size = 64
    exposure = [0] * num_variants

    for segment_index in range(6):
        rank_map = build_balanced_variant_rank_map(
            num_variants, world_size, segment_index, seed=29
        )
        assert len(set(rank_map)) == world_size
        for variant_index in rank_map:
            exposure[variant_index] += 1

    assert min(exposure) == 1
    assert max(exposure) == 2
    assert sum(exposure) == 6 * world_size


def test_balanced_rank_schedule_is_deterministic_and_seeded():
    first = build_balanced_variant_rank_map(66, 64, 4, seed=11)
    repeated = build_balanced_variant_rank_map(66, 64, 4, seed=11)
    reseeded = build_balanced_variant_rank_map(66, 64, 4, seed=12)

    assert first == repeated
    assert set(first) == set(reseeded)
    assert first != reseeded


def test_manifest_preserves_variant_order_and_rejects_tampering(tmp_path):
    first = tmp_path / "first.urdf"
    second = tmp_path / "second.urdf"
    first.write_text('<robot name="first"><link name="base"/></robot>')
    second.write_text('<robot name="second"><link name="base"/></robot>')
    manifest = tmp_path / "variants.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "holomotion-g1-variant-manifest-v1",
                "variants": [
                    {
                        "variant_id": "second",
                        "urdf_path": second.name,
                        "urdf_sha256": hashlib.sha256(
                            second.read_bytes()
                        ).hexdigest(),
                    },
                    {
                        "variant_id": "first",
                        "urdf_path": first.name,
                        "urdf_sha256": hashlib.sha256(
                            first.read_bytes()
                        ).hexdigest(),
                    },
                ],
            }
        )
    )

    asset = {"urdf_variant_manifest": str(manifest)}
    assert resolve_urdf_variants(asset) == [
        str(second.resolve()),
        str(first.resolve()),
    ]

    first.write_text('<robot name="tampered"><link name="base"/></robot>')
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_urdf_variants(asset)


def test_resolved_rank_variant_binds_scene_and_reference_fk(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="test"><link name="base"/></robot>')
    robot = SimpleNamespace(
        asset={"urdf_file": "fallback.urdf", "urdf_variants": [str(urdf)]},
        motion={"fk_robot_file_path": "old.urdf"},
    )
    commands = {
        "ref_motion": {
            "params": {"motion_lib_cfg": {"fk_robot_file_path": "old.urdf"}}
        }
    }

    resolved = resolve_rank_robot_variant(robot.asset, process_id=3)
    bind_rank_robot_variant(robot, commands, resolved)

    assert resolved.urdf_path == str(urdf.resolve())
    assert (
        resolved.urdf_sha256 == hashlib.sha256(urdf.read_bytes()).hexdigest()
    )
    assert robot.asset["urdf_file"] == resolved.urdf_path
    assert robot.motion["fk_robot_file_path"] == resolved.urdf_path
    assert (
        commands["ref_motion"]["params"]["motion_lib_cfg"][
            "fk_robot_file_path"
        ]
        == resolved.urdf_path
    )


def test_rank_urdf_variants_cycle_by_global_process_index():
    asset = {
        "urdf_file": "fallback.urdf",
        "urdf_variants": ["a.urdf", "b.urdf", "c.urdf"],
        "urdf_variant_mode": "per_rank",
    }

    assert resolve_rank_urdf_path(asset, 0) == ("a.urdf", 0)
    assert resolve_rank_urdf_path(asset, 2) == ("c.urdf", 2)
    assert resolve_rank_urdf_path(asset, 4) == ("b.urdf", 1)


def test_rank_urdf_variant_offset_supports_repeatable_coverage_shift():
    asset = {
        "urdf_file": "fallback.urdf",
        "urdf_variants": ["a.urdf", "b.urdf", "c.urdf"],
        "urdf_variant_rank_offset": 2,
    }

    assert resolve_rank_urdf_path(asset, 0) == ("c.urdf", 2)
    assert resolve_rank_urdf_path(asset, 2) == ("b.urdf", 1)


def test_rank_urdf_falls_back_to_single_asset():
    asset = SimpleNamespace(
        urdf_file="fallback.urdf",
        get=lambda key, default=None: default,
    )

    assert resolve_rank_urdf_path(asset, 7) == ("fallback.urdf", None)


def test_rank_urdf_per_environment_mode_uses_first_asset_as_template():
    asset = {
        "urdf_file": "fallback.urdf",
        "urdf_variants": ["a.urdf", "b.urdf"],
        "urdf_variant_mode": "per_env",
    }

    assert resolve_rank_urdf_path(asset, 0) == ("a.urdf", None)


def test_urdf_variant_glob_is_sorted_and_counted(tmp_path):
    (tmp_path / "b.urdf").touch()
    (tmp_path / "a.urdf").touch()
    asset = {
        "urdf_variant_glob": str(tmp_path / "*.urdf"),
        "urdf_variant_expected_count": 2,
    }

    assert resolve_urdf_variants(asset) == [
        str(tmp_path / "a.urdf"),
        str(tmp_path / "b.urdf"),
    ]


def test_urdf_variant_glob_rejects_incomplete_bundle(tmp_path):
    (tmp_path / "only.urdf").touch()
    asset = {
        "urdf_variant_glob": str(tmp_path / "*.urdf"),
        "urdf_variant_expected_count": 66,
    }

    with pytest.raises(ValueError, match="Expected 66"):
        resolve_urdf_variants(asset)


def test_shared_variant_cache_populates_once_per_source_signature(tmp_path):
    urdf = tmp_path / "robot.urdf"
    mesh = tmp_path / "body.stl"
    mesh.write_text("solid body\nendsolid body\n")
    urdf.write_text(
        '<robot name="test"><link name="base"><visual><geometry>'
        '<mesh filename="body.stl"/></geometry></visual></link></robot>'
    )
    cache_root = tmp_path / "cache"
    output = cache_root / "robot" / "robot.usd"
    calls = []

    def populate():
        calls.append(True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("usd")
        return [str(output)]

    cache_key = variant_source_signature(
        [str(urdf)], conversion_options={"merge_fixed_joints": True}
    )
    _, first_prepared = prepare_shared_file_cache(
        str(cache_root),
        cache_key=cache_key,
        expected_paths=[str(output)],
        populate=populate,
    )
    _, second_prepared = prepare_shared_file_cache(
        str(cache_root),
        cache_key=cache_key,
        expected_paths=[str(output)],
        populate=populate,
    )

    assert first_prepared is True
    assert second_prepared is False
    assert len(calls) == 1


def test_shared_variant_cache_repopulates_after_source_change(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="first"><link name="base"/></robot>')
    cache_root = tmp_path / "cache"
    output = cache_root / "robot.usd"
    calls = []

    def populate():
        calls.append(True)
        output.write_text("usd")
        return [str(output)]

    first_key = variant_source_signature([str(urdf)], conversion_options={})
    prepare_shared_file_cache(
        str(cache_root),
        cache_key=first_key,
        expected_paths=[str(output)],
        populate=populate,
    )
    urdf.write_text('<robot name="second"><link name="base"/></robot>')
    second_key = variant_source_signature([str(urdf)], conversion_options={})
    _, prepared = prepare_shared_file_cache(
        str(cache_root),
        cache_key=second_key,
        expected_paths=[str(output)],
        populate=populate,
    )

    assert second_key != first_key
    assert prepared is True
    assert len(calls) == 2
