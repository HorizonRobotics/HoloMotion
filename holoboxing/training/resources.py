"""Resolve the same public asset and controller bundles used by evaluation."""

from holoboxing.evaluation.models import frozen_manifest, verify_bundle
from holoboxing.evaluation.resources import ResourceTree
from holoboxing.evaluation.runtime.model_contract import sha256


def resolve_resources(assets, lower):
    tree = ResourceTree(assets)
    bundle = verify_bundle(lower)
    paths = {
        "lower_config": bundle / "config.yaml",
        "lower_onnx": bundle / frozen_manifest()["model"],
        "assets": tree.root / "resources.json",
        "arena_scene": tree.scene,
        "initial_state": tree.initial_state,
    }
    return (
        tree,
        paths,
        {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
    )


def neutral_reference(robot_file, joint_names, joint_position):
    """Use identical CPU kinematics for the fixed SL and RL action origin."""
    import torch

    from holoboxing.training.tasks.combat.config.sandbag import POINTS
    from holoboxing.training.tasks.combat.mdp.geometry import encode_output
    from holomotion.src.motion_tracking.link_fk import SharedLinkFK
    from holoboxing.data.robot import robot_file as extracted_robot

    with extracted_robot(robot_file) as xml:
        fk = SharedLinkFK(str(xml), "cpu", joint_names=joint_names)
    with torch.no_grad():
        position, quaternion = fk.forward(joint_position.detach().cpu()[None])
        indices = [fk.body_names.index(name) for name in POINTS]
        position, quaternion = position[:, indices], quaternion[:, indices]
        return encode_output(
            position,
            quaternion,
            position[:, None].expand(-1, 5, -1, -1),
            quaternion[:, None].expand(-1, 5, -1, -1),
        )[0]


def load_neutral_reference(model, robot_file):
    """Read ONNX metadata without constructing an inference session."""
    import onnx

    from holoboxing.training.envs.control import controller_parameters

    graph = onnx.load(model, load_external_data=False)
    metadata = {item.key: item.value for item in graph.metadata_props}
    del graph
    names, joint_position, _ = controller_parameters(metadata)
    return neutral_reference(robot_file, names, joint_position)
