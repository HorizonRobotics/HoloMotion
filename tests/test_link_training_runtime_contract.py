"""CPU contracts for the opt-in link training command path."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.motion_tracking.link_command_state import (
    LinkCommandObservationAdapter,
)
from holomotion.src.training.h5_dataloader import (
    ClipBatch,
    MotionClipBatchCache,
)


def _cache(tensors, lengths):
    cache = MotionClipBatchCache.__new__(MotionClipBatchCache)
    clip_count = len(lengths)
    cache._current_batch = ClipBatch(
        tensors=tensors,
        lengths=torch.tensor(lengths, dtype=torch.long),
        motion_keys=[f"clip-{index}" for index in range(clip_count)],
        raw_motion_keys=[f"clip-{index}" for index in range(clip_count)],
        window_indices=torch.arange(clip_count),
        max_frame_length=max(lengths),
    )
    return cache


def test_gather_offsets_preserves_order_and_marks_invalid_frames():
    field = torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)
    cache = _cache({"ref_root_pos": field}, lengths=[5, 3])

    values, valid = cache.gather_tensor_offsets(
        "ref_root_pos",
        clip_indices=torch.tensor([0, 1, 1]),
        frame_indices=torch.tensor([2, 1, 2]),
        frame_offsets=(-2, 0, 1, 1, 3),
    )

    assert values.shape == (3, 5, 1)
    assert values[..., 0].tolist() == [
        [0, 2, 3, 3, 4],
        [5, 6, 7, 7, 7],
        [5, 7, 7, 7, 7],
    ]
    assert valid.dtype == torch.bool
    assert valid.tolist() == [
        [True, True, True, True, False],
        [False, True, True, True, False],
        [True, True, False, False, False],
    ]
    # The legacy continuous gather remains independent of arbitrary offsets.
    continuous = cache.gather_tensor(
        "ref_root_pos",
        clip_indices=torch.tensor([1]),
        frame_indices=torch.tensor([1]),
        n_future_frames=2,
    )
    assert continuous[0, :, 0].tolist() == [6, 7, 7]


def test_gather_tensor_offsets_rejects_bad_requests():
    cache = _cache({"ref_root_pos": torch.zeros(1, 3, 3)}, lengths=[3])
    ids = torch.tensor([0])
    with pytest.raises(KeyError, match="not present"):
        cache.gather_tensor_offsets(
            "missing", clip_indices=ids, frame_indices=ids, frame_offsets=[0]
        )
    with pytest.raises(TypeError, match="integer"):
        cache.gather_tensor_offsets(
            "ref_root_pos",
            clip_indices=ids,
            frame_indices=ids,
            frame_offsets=[0.5],
        )
    with pytest.raises(ValueError, match="nonempty"):
        cache.gather_tensor_offsets(
            "ref_root_pos",
            clip_indices=ids,
            frame_indices=ids,
            frame_offsets=[],
        )


class _MeasuredFK:
    dof_names = ("joint",)
    body_names = ("pelvis", "arm")

    def __init__(self):
        self.inputs = []

    def __call__(self, joint_angles):
        self.inputs.append(joint_angles.clone())
        batch = joint_angles.shape[0]
        positions = torch.zeros(batch, 2, 3)
        positions[:, 1, 0] = joint_angles[:, 0]
        rotations = torch.zeros(batch, 2, 4)
        rotations[..., 0] = 1.0
        return positions, rotations


def _command_for_adapter():
    batch, frames, links = 2, 4, 2
    root = torch.zeros(1, frames, 3)
    root[..., 2] = 1.0
    reference_pos = torch.zeros(1, frames, links, 3)
    reference_pos[..., 0, 2] = 1.0
    reference_pos[..., 1, 0] = 1.0
    reference_pos[..., 1, 2] = 1.0
    reference_rot = torch.zeros(1, frames, links, 4)
    reference_rot[..., 3] = 1.0  # Stored reference quaternions are XYZW.
    cache = _cache(
        {
            "ref_root_pos": root,
            "ref_rg_pos": reference_pos,
            "ref_rb_rot": reference_rot,
        },
        lengths=[frames],
    )
    identity = torch.zeros(batch, links, 4)
    identity[..., 0] = 1.0
    robot_data = SimpleNamespace(
        root_quat_w=torch.tensor([[1.0, 0, 0, 0]]).expand(batch, -1).clone(),
        root_ang_vel_b=torch.zeros(batch, 3),
        joint_pos=torch.tensor([[0.1], [0.2]]),
        default_joint_pos=torch.zeros(batch, 1),
        joint_vel=torch.zeros(batch, 1),
        root_pos_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(batch, -1).clone(),
        body_pos_w=reference_pos[0, :batch].clone(),
        body_quat_w=identity,
    )
    robot_data.body_pos_w[:, 0, :] = robot_data.root_pos_w
    robot_data.body_pos_w[:, 1, :] = torch.tensor([1.0, 0.0, 1.0])
    environment = SimpleNamespace(
        action_manager=SimpleNamespace(action=torch.zeros(batch, 1)),
        scene=SimpleNamespace(env_origins=torch.zeros(batch, 3)),
        cfg=SimpleNamespace(
            observations=SimpleNamespace(
                unified=SimpleNamespace(enable_corruption=True)
            )
        ),
    )
    config = {
        "enabled": True,
        "link_names": ["pelvis", "arm"],
        "frame_offsets": [1, 2],
        "include_error_terms": True,
        "mask": {"min_active_links": 2, "max_active_links": 2},
        "proprio_noise": {"dof_pos": {"n_min": 0.25, "n_max": 0.25}},
    }
    return SimpleNamespace(
        cfg=SimpleNamespace(
            link_observation=config, n_fut_frames=2, target_fps=50
        ),
        mcfg=SimpleNamespace(fk_robot_file_path="unused.xml"),
        urdf_body_names=["pelvis", "arm"],
        simulator_body_names=["pelvis", "arm"],
        simulator_dof_names=["joint"],
        robot=SimpleNamespace(data=robot_data),
        num_envs=batch,
        device=torch.device("cpu"),
        _is_evaluating=False,
        _motion_cache=cache,
        _clip_indices=torch.zeros(batch, dtype=torch.long),
        _frame_indices=torch.tensor([0, 1]),
        _env=environment,
    )


def test_link_adapter_reuses_feedback_and_invalidates_mask_change():
    command = _command_for_adapter()
    fk = _MeasuredFK()
    with mock.patch(
        "holomotion.src.motion_tracking.link_command_state.SharedLinkFK",
        return_value=fk,
    ):
        adapter = LinkCommandObservationAdapter(command)

    first = adapter.snapshot()
    assert adapter.snapshot() is first
    measured = adapter.get_state("dof_pos", actor=True)
    torch.testing.assert_close(measured, torch.tensor([[0.35], [0.45]]))
    torch.testing.assert_close(fk.inputs[0], measured)
    assert len(fk.inputs) == 1
    torch.testing.assert_close(
        adapter.get_term("position_error")[:, :, 3],
        torch.tensor([[0.65, 0.65], [0.55, 0.55]]),
    )
    assert len(fk.inputs) == 1
    torch.testing.assert_close(
        adapter.get_state("dof_pos", actor=False),
        torch.tensor([[0.1], [0.2]]),
    )

    adapter.set_link_mask([1, 0], env_ids=torch.tensor([0]))
    assert "link_snapshot" not in command._reference_observation_cache
    masked = adapter.snapshot()
    assert masked is not first
    assert len(fk.inputs) == 2
    assert masked["actor"].link_mask[0].tolist() == [1.0, 0.0]
    assert masked["actor"].target_pos[0, :, 1].count_nonzero().item() == 0
    assert masked["critic"].target_pos[0, :, 1, 0].tolist() == [1.0, 1.0]

    adapter.resample_mask(torch.tensor([0]))
    assert "link_snapshot" not in command._reference_observation_cache
    restored = adapter.snapshot()
    assert len(fk.inputs) == 3
    assert restored["actor"].link_mask.tolist() == [[1.0, 1.0]] * 2
    assert restored["actor"].target_pos[0, :, 1, 0].tolist() == [1.0, 1.0]
