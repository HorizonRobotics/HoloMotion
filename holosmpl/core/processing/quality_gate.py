from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityGateConfig:
    enabled: bool = False
    mode: str = "skip"
    root_height_mean_min_m: float | None = None
    root_height_min_min_m: float | None = None
    upright_z_p05_min: float | None = None
