"""Evaluation-only backend; native training class remains byte-for-byte intact."""

import warp as wp

from holomotion.src.env.reference_tracking.warp_backend import (
    WarpTrackingBackend,
)

from .driver import CanonicalDriver


class CanonicalEvaluationBackend(WarpTrackingBackend):
    def __init__(self, model, *args, evaluation=False, **kwargs):
        self.canonical = CanonicalDriver(model, evaluation=evaluation)
        super().__init__(model, *args, evaluation=evaluation, **kwargs)

    def _advance(self):
        if not self.model.is_sparse:
            raise ValueError("canonical_gpu_v1 requires sparse constraints")
        for _ in range(4):
            self.canonical.step(self.model, self.data)
            self._overflow_seen.bitwise_or_(wp.to_torch(self.data.overflow))
