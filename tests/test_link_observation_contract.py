import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holomotion.src.motion_tracking.link_observation import (
    DEFAULT_COMMAND_LINK_NAMES,
    DEFAULT_FUTURE_FRAME_OFFSETS,
    LINK_FEATURE_DIM,
    LinkObservation,
    _safe_quaternion,
    _safe_vector,
    quat_apply_wxyz,
    quat_multiply_wxyz,
    quat_to_rot6d_xz_wxyz,
)


def make_observation(*, with_errors=True):
    return LinkObservation(
        target_pos=torch.full((1, 2, 2, 3), 1.0),
        position_error=(
            torch.full((1, 2, 2, 3), 2.0) if with_errors else None
        ),
        target_rot6d=torch.full((1, 2, 2, 6), 3.0),
        rotation_error6d=(
            torch.full((1, 2, 2, 6), 4.0) if with_errors else None
        ),
        time_offsets=torch.tensor([[[5.0], [6.0]]]),
        link_mask=torch.tensor([[1.0, 0.0]]),
        time_valid=torch.tensor([[1.0, 1.0]]),
        reference_height=torch.tensor([[0.7]]),
        effective_mask=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )


def test_link_contract_dimensions_and_term_major_order():
    obs = make_observation()

    assert len(DEFAULT_COMMAND_LINK_NAMES) == 14
    assert DEFAULT_FUTURE_FRAME_OFFSETS == (1, 2)
    assert LINK_FEATURE_DIM == 18
    assert obs.task.shape == (1, 2, 37)
    assert obs.reference_task.shape == (1, 2, 19)
    assert obs.flat_command.shape == (1, 79)
    assert obs.flat_reference_command.shape == (1, 43)
    torch.testing.assert_close(obs.task[0, 0, :6], torch.ones(6))
    torch.testing.assert_close(obs.task[0, 0, 6:12], torch.full((6,), 2.0))
    torch.testing.assert_close(obs.task[0, 0, 12:24], torch.full((12,), 3.0))
    torch.testing.assert_close(obs.task[0, 0, 24:36], torch.full((12,), 4.0))
    assert obs.task[0, :, -1].tolist() == [5.0, 6.0]


def test_reference_only_contract_and_partial_errors():
    obs = make_observation(with_errors=False)

    torch.testing.assert_close(obs.task, obs.reference_task)
    assert obs.task.shape == (1, 2, 19)
    with pytest.raises(ValueError, match="enabled together"):
        _ = replace(make_observation(), position_error=None).task


def test_wxyz_geometry_and_invalid_quaternion_fallback():
    yaw_90 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    vector = torch.tensor([[1.0, 0.0, 0.0]])

    torch.testing.assert_close(
        quat_apply_wxyz(yaw_90, vector),
        torch.tensor([[0.0, 1.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(quat_multiply_wxyz(yaw_90, identity), yaw_90)
    torch.testing.assert_close(
        quat_to_rot6d_xz_wxyz(identity),
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )

    torch.testing.assert_close(
        quat_to_rot6d_xz_wxyz(yaw_90),
        torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        quat_to_rot6d_xz_wxyz(torch.zeros_like(identity)),
        quat_to_rot6d_xz_wxyz(identity),
    )


def test_nonfinite_inputs_are_zeroed_with_invalidity_signal():
    vector, vector_valid = _safe_vector(
        torch.tensor([[float("nan"), 1.0, 2.0]])
    )
    quaternion, quaternion_valid = _safe_quaternion(
        torch.tensor([[float("inf"), 0.0, 0.0, 0.0]])
    )

    assert vector.tolist() == [[0.0, 0.0, 0.0]]
    assert vector_valid.tolist() == [False]
    assert quaternion.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert quaternion_valid.tolist() == [False]


@pytest.mark.parametrize("with_errors", [False, True])
@pytest.mark.parametrize("future_mask", [False, True])
def test_link_adapter_matches_training_packing(
    monkeypatch, with_errors, future_mask
):
    import torch
    from omegaconf import OmegaConf

    from holomotion.src.evaluation.obs import mujoco_link_adapter
    from holomotion.src.evaluation.obs.obs_builder import PolicyObsBuilder
    from holomotion.src.motion_tracking.link_actor_packing import (
        LINK_CURRENT_TERMS,
        link_actor_layout,
        pack_link_actor_observation,
    )
    from holomotion.src.motion_tracking.link_observation import (
        LinkObservationInput,
        build_link_observation,
    )

    names = ["pelvis"] + [f"link{i}" for i in range(13)]
    link_cfg = dict(
        enabled=True,
        link_names=names,
        frame_offsets=[1, 3],
        include_error_terms=with_errors,
        include_future_link_mask=future_mask,
    )
    cfg = OmegaConf.create(
        {
            "link_observation": link_cfg,
            "robot": {
                "body_names": names,
                "motion": {"fk_robot_file_path": "fixture"},
            },
            "link_eval": {
                "mask_mode": "custom",
                "active_link_names": names[:3],
            },
        }
    )
    pos = np.arange(5 * 14 * 3, dtype=np.float32).reshape(5, 14, 3) / 100
    xyzw = np.zeros((5, 14, 4), dtype=np.float32)
    xyzw[..., 2:] = np.sqrt(0.5)  # Non-identity reference orientation.
    proprio = [
        np.arange(size, dtype=np.float32) / 100 for size in (3, 3, 29, 29, 29)
    ]
    local_pos = torch.arange(42, dtype=torch.float32).reshape(1, 14, 3) / 100
    local_quat = torch.zeros((1, 14, 4))
    local_quat[..., 0] = 1

    class FakeFK:
        body_names = names

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, q):
            np.testing.assert_allclose(q.numpy()[0], proprio[2] + 0.1)
            return local_pos, local_quat

    monkeypatch.setattr(mujoco_link_adapter, "SharedLinkFK", FakeFK)
    ev = SimpleNamespace(
        config=cfg,
        num_actions=29,
        policy_dt=0.02,
        dof_names_onnx=[f"joint{i}" for i in range(29)],
        default_angles_onnx=np.full(29, 0.1, dtype=np.float32),
        n_motion_frames=5,
        motion_frame_idx=0,
        ref_global_translation=pos,
        ref_global_rotation_quat_xyzw=xyzw,
        _ref_to_sim_q_wxyz=np.array(
            [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], np.float32
        ),
        _ref_to_sim_t=np.array([2.0, 3.0, 0.0], np.float32),
        _ensure_ref_to_sim_transform_rigid=lambda: None,
        robot_global_bodylink_rot=local_quat.numpy()[0],
        root_body_idx=0,
    )
    for name, value in zip(LINK_CURRENT_TERMS[:5], proprio, strict=True):
        setattr(ev, f"_get_obs_{name}", lambda value=value: value.copy())
    layout = link_actor_layout(link_cfg)
    builder = PolicyObsBuilder(
        ev.dof_names_onnx,
        ev.default_angles_onnx,
        ev,
        {
            "atomic_obs_list": [
                {name: {"history_length": 1}}
                for name in layout["atomic_terms"]
            ]
        },
    )
    mask = torch.zeros((1, 14))
    mask[:, :3] = 1
    for current in (0, 3):
        ev.motion_frame_idx = current
        packed = builder.build_policy_obs()
        indices = np.minimum(current + np.array([1, 3]), 4)
        # Apply +90 degree alignment independently of adapter state.
        aligned = pos[..., [1, 0, 2]].copy()
        aligned[..., 0] *= -1
        aligned += ev._ref_to_sim_t
        target_quat = torch.zeros((1, 2, 14, 4))
        target_quat[..., 3] = 1
        command = build_link_observation(
            LinkObservationInput(
                reference_link_pos=torch.from_numpy(aligned[indices][None]),
                reference_link_quat_wxyz=target_quat,
                reference_root_pos=torch.from_numpy(aligned[current, 0][None]),
                robot_root_quat_wxyz=local_quat[:, 0],
                robot_link_pos_local=local_pos if with_errors else None,
                robot_link_quat_local_wxyz=local_quat if with_errors else None,
                frame_offsets=torch.tensor([1.0, 3.0]),
                link_mask=mask,
                time_valid=torch.tensor(
                    [[current + 1 < 5, current + 3 < 5]], dtype=torch.float32
                ),
                reference_root_height=torch.tensor([[pos[current, 0, 2]]]),
            ),
            root_link_index=0,
            include_error_terms=with_errors,
        )
        expected = pack_link_actor_observation(
            torch.from_numpy(np.concatenate(proprio)[None]),
            command,
            with_errors,
            future_mask,
        )[0].numpy()
        np.testing.assert_allclose(packed, expected, atol=2e-6, rtol=0)
        assert packed.shape == (layout["actor_dim"],)
    builder.reset()
    assert builder.link_adapter.atomic == {}
    ev.motion_frame_idx = 0
    again = builder.build_policy_obs()
    builder.link_adapter.validate_packed(again)
    with pytest.raises(ValueError, match="expected finite"):
        builder.link_adapter.validate_packed(again[:-1])
    again[0] = np.nan
    with pytest.raises(ValueError, match="expected finite"):
        builder.link_adapter.validate_packed(again)
