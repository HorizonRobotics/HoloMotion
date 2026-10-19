# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Circular buffer for storing experience replay."""

import torch


class CircularBuffer:
    """A simple circular buffer for storing tensors.

    This buffer stores data in a circular fashion, overwriting old data
    when the buffer is full. Useful for experience replay in RL.

    Args:
        max_length: Maximum number of samples to store
        device: Device to store tensors on
    """

    def __init__(self, max_length: int, device: str = "cpu"):
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("Buffer capacity must be positive")
        self.max_length = max_length
        self.device = device
        self.buffer = None
        self.current_size = 0
        self.current_idx = 0

    def store(self, data: torch.Tensor):
        """Store data in the buffer.

        Args:
            data: Tensor of shape [batch_size, ...] to store
        """
        batch_size = data.shape[0]

        # Initialize buffer on first store
        if self.buffer is None:
            self.buffer = torch.zeros(
                (self.max_length, *data.shape[1:]),
                dtype=data.dtype,
                device=self.device,
            )

        # Same ring order as the source loop, without one GPU launch per row.
        if batch_size >= self.max_length:
            start = (self.current_idx + batch_size - self.max_length) % self.max_length
            kept = data[-self.max_length :]
        else:
            start, kept = self.current_idx, data
        first = min(len(kept), self.max_length - start)
        self.buffer[start : start + first].copy_(kept[:first])
        self.buffer[: len(kept) - first].copy_(kept[first:])
        self.current_idx = (self.current_idx + batch_size) % self.max_length
        self.current_size = min(self.current_size + batch_size, self.max_length)

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample a batch of data from the buffer.

        Args:
            batch_size: Number of samples to return

        Returns:
            Tensor of shape [batch_size, ...] containing sampled data
        """
        if self.buffer is None or self.current_size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Sample random indices from filled portion of buffer
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        return self.buffer[indices]

    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return self.current_size

    def clear(self):
        """Clear the buffer."""
        self.buffer = None
        self.current_size = 0
        self.current_idx = 0
