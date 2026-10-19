# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

from holomotion.src.modules.network_modules import (
    EmpiricalNormalization as HoloMotionNormalization,
)


class EmpiricalNormalization(HoloMotionNormalization):
    """Use HoloMotion's cumulative normalizer with legacy boxing weights."""

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy = ("_mean", "_var", "_std", "count")
        added = (
            "_ex2", "_last_sync_mean", "_last_sync_var", "_last_sync_count"
        )
        if all(prefix + key in state_dict for key in legacy) and not any(
            prefix + key in state_dict for key in added
        ):
            # Earlier boxing checkpoints stored only cumulative statistics.
            # Preserve them exactly; new EMA/sync buffers start unsynchronized.
            for key in added:
                state_dict[prefix + key] = getattr(self, key).clone()
            state_dict[prefix + "_ex2"] = (
                state_dict[prefix + "_var"]
                + state_dict[prefix + "_mean"].square()
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
