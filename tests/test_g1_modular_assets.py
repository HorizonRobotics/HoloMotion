import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from assets.robots.unitree.G1.modular.g1_urdf_composer import (
    CANONICAL_DOF_NAMES,
    compose_all_g1_urdfs,
    runtime_component_dependencies,
    unique_g1_combinations,
    validate_generated_urdf_bundle,
)

ASSET_ROOT = Path("assets/robots/unitree/G1/modular")
LEGACY_URDF_SET_SHA256 = (
    "a1a6766a82ffb923c297c31f7b2a92740f8cd8ce7d43956c97989fb92650b9ec"
)


def test_all_generated_variants_share_one_articulation_topology(tmp_path):
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    assert manifest["generated_unique_models"] == 66

    results = compose_all_g1_urdfs(tmp_path, validate=False)
    assert {result.output_path.stem for result in results} == {
        item["name"] for item in manifest["generated"]
    }
    for result in results:
        path = result.output_path
        root = ET.parse(path).getroot()
        links = [link.get("name") for link in root.findall("link")]
        joints = root.findall("joint")
        assert len(links) == len(set(links)) == 30, path
        assert len(joints) == 29, path
        assert all(joint.get("type") != "fixed" for joint in joints), path
        assert (
            tuple(joint.get("name") for joint in joints) == CANONICAL_DOF_NAMES
        )


def test_each_rank_balances_8192_environments_across_all_variants():
    counts = Counter(env_id % 66 for env_id in range(8192))
    assert len(counts) == 66
    assert sorted(counts.values()) == [124] * 58 + [125] * 8


def test_online_composition_dependency_closure_is_exact():
    dependencies = runtime_component_dependencies()

    assert len(unique_g1_combinations()) == 66
    assert len(dependencies.component_descriptions) == 13
    assert len(dependencies.meshes) == 183
    assert all(path.is_file() for path in dependencies.component_descriptions)
    assert all(path.is_file() for path in dependencies.meshes)


def test_online_composition_reproduces_legacy_donor_urdfs(tmp_path):
    results = compose_all_g1_urdfs(tmp_path, validate=False)
    urdfs, meshes = validate_generated_urdf_bundle(tmp_path)
    runtime_manifest = json.loads(
        (tmp_path / "variant_manifest.json").read_text()
    )

    assert len(results) == len(urdfs) == 66
    assert len(meshes) == 183
    assert runtime_manifest["schema"] == "holomotion-g1-variant-manifest-v1"
    assert len(runtime_manifest["variants"]) == 66
    assert [item["variant_id"] for item in runtime_manifest["variants"]] == [
        result.output_path.stem for result in results
    ]
    digest = hashlib.sha256()
    for generated in sorted(urdfs):
        digest.update(generated.name.encode())
        digest.update(b"\0")
        digest.update(generated.read_bytes())
    assert digest.hexdigest() == LEGACY_URDF_SET_SHA256
