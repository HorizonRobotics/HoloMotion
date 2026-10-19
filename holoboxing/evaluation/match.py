"""Two competitor policies through the same public GPU evaluation core."""

from pathlib import Path
import json

from .export_hdf5 import export
from .resources import ResourceTree
from .runtime.model_contract import check_model, check_contract, sha256


def evaluate_match(
    red_model,
    blue_model,
    assets,
    lower,
    output_dir,
    *,
    match_id=None,
    context=None,
):
    resources = ResourceTree(assets)
    paths = [
        Path(path).expanduser().resolve(strict=True)
        for path in (red_model, blue_model)
    ]
    for path in set(paths):
        graph, _ = check_model(path)
        check_contract(graph)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("match.json", "replay.hdf5")):
        raise FileExistsError(
            "Use an output directory without match.json or replay.hdf5"
        )
    from .runtime.holomotion_v2_runtime import (
        install_runtime,
        verify_result_metadata,
    )
    from .runtime.upper import FivePointReferencePair

    runtime = install_runtime(lower)
    # Read after configure_bundle; importing the old None value above is unsafe.
    from .runtime import lower as lower_runtime

    checkpoint = lower_runtime.MODEL_ROOT / "exported/model_10500.onnx"
    config = runtime.MatchConfig(
        retargeted=resources.initial_state,
        policy_0=checkpoint,
        policy_1=checkpoint,
        mujoco_model=resources.scene,
        mesh_root=resources.root / "meshes",
        output=output / "match.json",
        model_id_0=sha256(paths[0]),
        model_id_1=sha256(paths[1]),
    )
    if match_id is not None:
        config.match_id = str(match_id)
    runtime.run_match(
        config,
        reference_factory=lambda scene, policies: FivePointReferencePair(
            paths,
            scene,
            policies,
            device=config.device,
        ),
    )
    payload = json.loads(config.output.read_text())
    verify_result_metadata(payload["metadata"])
    replay = export(
        config.output,
        output / "replay.hdf5",
        context=context,
        asset_manifest=resources.path("manifest.json"),
    )
    return {
        "status": "PASS",
        "match_id": config.match_id,
        "summary": payload["summary"],
        "match": str(config.output),
        "replay": str(output / "replay.hdf5"),
        "export": replay,
    }
