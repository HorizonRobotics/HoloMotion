from dataclasses import replace

import pytest
import torch

from holomotion.src.motion_tracking.link_actor_packing import (
    LINK_CURRENT_TERMS,
    link_actor_layout,
    link_future_terms,
    pack_link_actor_current,
    pack_link_actor_future,
    pack_link_actor_observation,
)
from holomotion.src.motion_tracking.link_observation import (
    DEFAULT_COMMAND_LINK_NAMES,
    LinkObservation,
)


def make_command() -> LinkObservation:
    target_pos = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
    target_rot6d = torch.arange(48, dtype=torch.float32).reshape(2, 2, 2, 6)
    link_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    time_valid = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    return LinkObservation(
        target_pos=target_pos + 10.0,
        position_error=target_pos + 100.0,
        target_rot6d=target_rot6d + 200.0,
        rotation_error6d=target_rot6d + 300.0,
        time_offsets=torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]]]),
        link_mask=link_mask,
        time_valid=time_valid,
        reference_height=torch.tensor([[0.8], [0.9]]),
        effective_mask=time_valid[:, :, None] * link_mask[:, None, :],
    )


def test_actor_packing_is_term_major_and_preserves_proprioception():
    command = make_command()
    proprioception = torch.arange(186, dtype=torch.float32).reshape(2, 93)
    current = pack_link_actor_current(proprioception, command)
    future = pack_link_actor_future(command)
    packed = pack_link_actor_observation(proprioception, command)

    expected_current = torch.cat(
        (
            proprioception,
            command.reference_height,
            command.link_mask,
            command.time_valid,
        ),
        dim=-1,
    )
    expected_future = torch.cat(
        (
            command.target_pos.flatten(start_dim=1),
            command.position_error.flatten(start_dim=1),
            command.target_rot6d.flatten(start_dim=1),
            command.rotation_error6d.flatten(start_dim=1),
            command.time_offsets.flatten(start_dim=1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(current, expected_current)
    torch.testing.assert_close(future, expected_future)
    torch.testing.assert_close(
        packed, torch.cat((expected_current, expected_future), dim=-1)
    )
    assert current.shape == (2, 98)
    assert future.shape == (2, 74)
    assert packed.shape == (2, 172)
    assert future[0, 0] == command.target_pos[0, 0, 0, 0]
    assert future[0, 12] == command.position_error[0, 0, 0, 0]


def test_target_only_and_optional_future_mask_keep_explicit_schema():
    command = make_command()
    target_only = replace(command, position_error=None, rotation_error6d=None)
    target_future = pack_link_actor_future(
        target_only, include_error_terms=False
    )
    expected_target = torch.cat(
        (
            command.target_pos.flatten(start_dim=1),
            command.target_rot6d.flatten(start_dim=1),
            command.time_offsets.flatten(start_dim=1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(target_future, expected_target)
    assert target_future.shape == (2, 38)

    with_mask = pack_link_actor_future(command, include_future_link_mask=True)
    expected_with_mask = torch.cat(
        (
            command.target_pos.flatten(start_dim=1),
            command.position_error.flatten(start_dim=1),
            command.target_rot6d.flatten(start_dim=1),
            command.rotation_error6d.flatten(start_dim=1),
            command.effective_mask.flatten(start_dim=1),
            command.time_offsets.flatten(start_dim=1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(with_mask, expected_with_mask)
    assert with_mask.shape == (2, 78)
    assert link_future_terms(True, True)[-2:] == (
        "actor_link_effective_mask",
        "actor_link_time_offsets",
    )
    with pytest.raises(ValueError, match="Error-enabled"):
        pack_link_actor_future(target_only)


def test_layout_validates_saved_order_and_frame_offsets():
    config = {
        "link_names": ["pelvis", "left_wrist_yaw_link"],
        "frame_offsets": [1, 2],
        "include_error_terms": True,
    }
    layout = link_actor_layout(config)
    assert layout["current_dim"] == 98
    assert layout["future_dim"] == 74
    assert layout["actor_dim"] == 172
    schema = {
        "current": {
            "terms": [f"unified/{name}" for name in LINK_CURRENT_TERMS],
            "seq_len": 1,
        },
        "future": {
            "terms": [f"unified/{name}" for name in layout["future_terms"]],
            "seq_len": 2,
        },
    }
    assert link_actor_layout(config, schema)["actor_dim"] == 172
    wrong_schema = {
        **schema,
        "future": {
            **schema["future"],
            "terms": schema["future"]["terms"][::-1],
        },
    }
    with pytest.raises(ValueError, match="Saved actor schema order"):
        link_actor_layout(config, wrong_schema)
    with pytest.raises(ValueError, match="sequence lengths"):
        link_actor_layout(
            config, {**schema, "future": {**schema["future"], "seq_len": 3}}
        )
    with pytest.raises(ValueError, match="distinct positive"):
        link_actor_layout({**config, "frame_offsets": [1, 1]})

    full_layout = link_actor_layout(
        {**config, "link_names": DEFAULT_COMMAND_LINK_NAMES}
    )
    assert (
        full_layout["current_dim"],
        full_layout["future_dim"],
        full_layout["actor_dim"],
    ) == (110, 506, 616)


def test_packing_rejects_incompatible_proprioception():
    with pytest.raises(ValueError, match="proprioception"):
        pack_link_actor_current(torch.zeros(2, 92), make_command())
