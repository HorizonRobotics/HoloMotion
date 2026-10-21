"""Mixed-controller evaluation preserves the candidate's public observation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from holoboxing.data.contract import CompetitionHistory
from holoboxing.evaluation.runtime.reference import ReferenceSchema
from holoboxing.evaluation.runtime.rules import MatchRules, MatchStateMachine
from holoboxing.evaluation.runtime.upper import FivePointReferencePair
from holoboxing.training.evaluate_native import (
    initial_pair,
    legal_observation,
    world_targets,
)
from holoboxing.training.tasks.combat.config.sandbag import POINTS


def test_direct_target_decode_matches_public_reference_schema():
    rng = np.random.default_rng(53)
    schema = ReferenceSchema()
    provider = FivePointReferencePair.__new__(FivePointReferencePair)
    provider.schemas = [schema, schema]
    for _ in range(8):
        position = rng.normal(size=(5, 5, 3))
        rotations = (
            Rotation.random(25, random_state=rng)
            .as_matrix()
            .reshape(5, 5, 3, 3)
        )
        axes = np.concatenate((rotations[..., :, 0], rotations[..., :, 2]), -1)
        target = np.concatenate((position, axes), -1).astype(np.float32)
        root = rng.normal(size=3).astype(np.float32)
        quaternion = (
            Rotation.random(random_state=rng)
            .as_quat()[[3, 0, 1, 2]]
            .astype(np.float32)
        )
        reference = (
            provider.reference(
                torch.from_numpy(target.flatten()), 0, float(root[2])
            )
            .numpy()
            .copy()
        )
        expected = schema.to_targets(reference, root, quaternion)
        actual = world_targets(target, root, quaternion)
        np.testing.assert_allclose(
            actual.future_positions, expected.future_positions, atol=1e-6
        )
        np.testing.assert_allclose(
            actual.future_rotations_wxyz,
            expected.future_rotations_wxyz,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            actual.frame_offsets, expected.frame_offsets
        )


def test_candidate_observation_matches_public_history_across_round_reset():
    rng = np.random.default_rng(64)
    names = [prefix + name for prefix in ("p1_", "p2_") for name in POINTS]
    model = SimpleNamespace(
        body=lambda name: SimpleNamespace(id=names.index(name)),
        body_rootid=np.repeat([0, 5], 5),
    )
    scene = SimpleNamespace(model=model, _root_dof=[0, 6])
    data = SimpleNamespace(
        xpos=rng.normal(size=(10, 3)),
        xquat=np.tile([1.0, 0, 0, 0], (10, 1)),
        cvel=rng.normal(size=(10, 6)),
        subtree_com=rng.normal(size=(10, 3)),
        qvel=rng.normal(size=12),
    )
    match = MatchStateMachine(MatchRules())
    reference = FivePointReferencePair.__new__(FivePointReferencePair)
    reference.match = match
    reference.reset()
    history = CompetitionHistory(1)
    ids = np.arange(10).reshape(2, 5)
    for frame in range(4):
        data.xpos += rng.normal(0, 0.05, (10, 3))
        if frame == 3:
            reference.reset()
        match.score = [frame + 1, frame]
        actual = legal_observation(
            history, scene, data, match, frame, frame in (0, 3)
        )
        p, q, cvel, com = (
            torch.tensor(values, dtype=torch.float32)
            for values in (
                data.xpos[ids],
                data.xquat[ids],
                data.cvel[ids],
                data.subtree_com[model.body_rootid[ids]],
            )
        )
        velocity = cvel[..., 3:] + torch.cross(cvel[..., :3], p - com, dim=-1)
        omega = torch.tensor(
            np.stack((data.qvel[3:6], data.qvel[9:12])), dtype=torch.float32
        )
        expected = reference.observations(p, q, velocity, omega).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_initial_pose_preserves_competition_roots_and_joint_order():
    values = np.zeros((1, 2, 36), np.float32)
    values[0, :, :3] = [[-1, 0, 0.8], [1, 0, 0.8]]
    values[..., 3] = 1
    values[..., 7:] = np.arange(29)
    names = [f"joint-{i}" for i in range(29)]
    motion = SimpleNamespace(qpos=values, dof_names=tuple(names))
    reordered = initial_pair(motion, names[::-1])
    np.testing.assert_array_equal(reordered[:, :7], values[0, :, :7])
    np.testing.assert_array_equal(
        reordered[:, 7:], np.tile(np.arange(28, -1, -1), (2, 1))
    )
    exact = initial_pair(motion, names)
    np.testing.assert_array_equal(exact, values[0])
    exact[0, 0] = 5
    assert values[0, 0, 0] == -1


def test_native_match_uses_official_substep_impulses(monkeypatch):
    from holoboxing.training import evaluate_native as native

    calls = []
    monkeypatch.setattr(
        native, "install_hand_rules", lambda: calls.append("installed")
    )
    thresholds = SimpleNamespace(assign=lambda values: None)
    scene = SimpleNamespace(
        _gpu_physics=SimpleNamespace(
            slot_map=object(), hand_thresholds=thresholds
        )
    )

    def create_scene(**kwargs):
        assert calls == ["installed"]
        return scene

    monkeypatch.setattr(native, "BoxingMatchScene", create_scene)
    actual_scene, referee = native.scene_and_referee(
        SimpleNamespace(scene="official.xml"), "cuda:0", MatchRules()
    )
    assert actual_scene is scene and referee.scene is scene
    # The legacy 50 Hz aggregate has no impulse. The four measured substeps
    # cross the official 5 N.s threshold only with all four samples.
    contact = dict(
        attacker=0,
        defender=1,
        attacking_body="left_hand",
        target_body="torso",
        physical_contact=True,
        peak_force=300.0,
        closing_speed=0.0,
        position=[0.0, 0.0, 0.0],
        normal_impulse_step_ns=1.5,
        sample_dt_s=0.005,
        pair_id=0,
    )
    scene._hand_contact_samples = [(i * 0.005, [contact]) for i in range(1, 5)]
    result = referee.observe_frame(
        sim_time_s=0.02,
        contacts=[{"legacy": True}],
        root_pos=np.zeros((2, 3)),
        fallen=[False, False],
    )
    assert result.score == (1, 0)
    assert (
        len([event for event in result.events if event["type"] == "HIT"]) == 1
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_native_entrypoint_publishes_numpy_referee_scores(
    tmp_path, monkeypatch, compiled
):
    """Publish real referee scores through a CPU scene double."""
    import json

    from holoboxing.training import evaluate_native as native
    from holoboxing.training.tasks.combat.config.sandbag import JOINT_NAMES

    candidate, opponent = tmp_path / "policy.onnx", tmp_path / "opponent.pt"
    candidate.write_bytes(b"checked-by-session-double")
    torch.save({}, opponent)
    motion = SimpleNamespace(qpos=np.zeros((1, 2, 36)), dof_names=JOINT_NAMES)
    motion.qpos[..., 2:4] = [1, 1]
    data = SimpleNamespace(qpos=motion.qpos[0].reshape(-1).copy())
    kinematics = SimpleNamespace(
        root_position=np.zeros((1, 2, 3)),
        root_quaternion_wxyz=np.tile([1.0, 0, 0, 0], (1, 2, 1)),
        joint_position=np.zeros((1, 2, 29)),
    )
    scene = SimpleNamespace(
        articulation_joint_names=JOINT_NAMES,
        _root_qpos=[0, 36],
        write_pair_state=lambda pose: None,
        drain_reference_frame_qpos=lambda: None,
        read_kinematics=lambda: kinematics,
        stage_pair_joint_targets=lambda targets: None,
        step_reference_frame=lambda: None,
        contact_samples=lambda **kw: [[]],
        close=lambda: None,
    )
    resources = SimpleNamespace(
        initial_state="reset.npz",
        scene="scene.xml",
        manifest_sha256="resources",
    )

    class NativeController:
        kp = kd = torch.ones(29)

        def __init__(self, policy):
            pass

        def act(self, *args):
            return torch.zeros(1, 29), torch.zeros(1, 29)

        def reset(self):
            pass

    class LowerController:
        kp = kd = np.ones(29)
        model_to_articulation = np.arange(29)
        path = candidate

        def __init__(self, *args, **kwargs):
            pass

        def attach_scene(self, scene):
            pass

        def act(self, **kwargs):
            if compiled:
                scene._targets_on_gpu = True
            return np.zeros(29), np.zeros(29), 1, 2

        def reset(self):
            pass

    monkeypatch.setattr(native, "check_model", lambda p: (None, 100))
    monkeypatch.setattr(native, "compiled_available", lambda: compiled)
    from holoboxing.evaluation.runtime import holomotion_v2_runtime

    monkeypatch.setattr(
        holomotion_v2_runtime,
        "install_runtime",
        lambda lower: SimpleNamespace(),
    )

    class Pair:
        sessions = [None, None]
        metadata = {"compiled_observation_manifest_sha256": "verified-double"}
        frame_data = {"competitor_sanitization": [{"nonfinite_components": 0}]}

        def __init__(self, *args, **kwargs):
            pass

        def window(self, *args, **kwargs):
            return [
                np.tile(
                    np.array([0, 0, 0, 1, 0, 0, 0, 0, 1], np.float32), (1, 25)
                )
            ]

        def reset(self):
            pass

    monkeypatch.setattr(native, "FivePointReferencePair", Pair)
    monkeypatch.setattr(native, "check_contract", lambda g: None)
    monkeypatch.setattr(
        native, "require_gpu", lambda device: torch.device("cpu")
    )
    monkeypatch.setattr(native, "NativeRoboController", NativeController)
    monkeypatch.setattr(
        native, "NativeRoboStriker", lambda saved: torch.nn.Identity()
    )
    monkeypatch.setattr(native, "ResourceTree", lambda path: resources)
    monkeypatch.setattr(native.ResetPose, "load", lambda path: motion)
    monkeypatch.setattr(
        native.lower_runtime, "configure_bundle", lambda path: None
    )
    monkeypatch.setattr(native.lower_runtime, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(native, "create_device_session", lambda *a, **kw: None)
    monkeypatch.setattr(
        native,
        "scene_and_referee",
        lambda *a, **kw: (
            scene,
            MatchStateMachine(MatchRules(round_duration_s=0.02)),
        ),
    )
    monkeypatch.setattr(native, "CurrentStateLinkPolicy", LowerController)
    monkeypatch.setattr(
        native,
        "CurrentObservationFK",
        lambda s: SimpleNamespace(data=data, refresh=lambda: None),
    )
    monkeypatch.setattr(
        native, "legal_observation", lambda *a: np.zeros((2, 1097), np.float32)
    )
    monkeypatch.setattr(native, "world_targets", lambda *a: None)
    monkeypatch.setattr(
        native,
        "run_competitor",
        lambda *a: np.tile(
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 1], np.float32), (1, 25)
        ),
    )
    monkeypatch.setattr(native, "read_native_state", lambda *a: ())
    monkeypatch.setattr(native, "fallen_from_scene", lambda s: [False, False])
    monkeypatch.setattr(native, "verify_cuda_execution", lambda session: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    report = native.evaluate(
        candidate,
        opponent,
        native.sha256(opponent),
        "assets",
        "lower",
        tmp_path / "match",
    )
    assert report["status"] == "COMPLETE" and report["frames"] == 1
    assert report["initial_pose_mode"] == "official"
    assert report["candidate_runtime_backend"] == (
        "published_compiled" if compiled else "source_torch"
    )
    assert scene._targets_on_gpu is False
    assert json.loads((tmp_path / "match/result.json").read_text())[
        "score"
    ] == [0, 0]
    assert json.loads((tmp_path / "match/match.json").read_text())["metadata"][
        "match_completed"
    ]
