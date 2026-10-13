"""Source-independent, device-resident link reference for motion policies."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class DeviceLinkReference:
    """Canonical reference shared by 1/3/5/14-link control modes.

    Positions are world-frame metres; rotations are world-frame WXYZ
    quaternions. The root position anchors the current reference frame, not
    the measured robot position. Future offsets use the saved model's 50 Hz
    frame grid. Unselected links and invalid future frames are masked by the
    observation builder. All arrays stay on the same CUDA device.

    Producers own the buffers: consume this reference before reusing them.
    No Pico, retarget, joint-sequence or ROS schema belongs to this interface.
    """

    link_names: tuple[str, ...]
    frame_offsets: tuple[int, ...]
    positions: wp.array
    rotations_wxyz: wp.array
    root_position: wp.array
    time_valid: wp.array
    source_valid: wp.array | None = None

    def validate_layout(self, *, link_names, frame_offsets, device) -> None:
        """Check metadata without staging reference tensors on the host."""
        device = wp.get_device(device)
        if not device.is_cuda:
            raise ValueError("Link deployment references require CUDA")
        if self.link_names != tuple(link_names):
            raise ValueError(
                "Reference link order differs from the saved model"
            )
        if self.frame_offsets != tuple(frame_offsets):
            raise ValueError(
                "Reference future offsets differ from the saved model"
            )
        frames, links = len(frame_offsets), len(link_names)
        arrays = (
            (self.positions, (frames, links, 3), wp.float32),
            (self.rotations_wxyz, (frames, links, 4), wp.float32),
            (self.root_position, (3,), wp.float32),
            (self.time_valid, (frames,), wp.float32),
        )
        if self.source_valid is not None:
            arrays += ((self.source_valid, (1,), wp.int32),)
        for array, shape, dtype in arrays:
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(
                    f"Invalid link reference layout; expected {shape}"
                )
            if array.device != device:
                raise ValueError(
                    "Reference must stay on the observation CUDA device"
                )
