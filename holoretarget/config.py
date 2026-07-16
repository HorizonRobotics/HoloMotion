"""Configuration for the production HoloRetarget path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HOLOMOTION_ROOT = Path(__file__).resolve().parents[1]


def default_asset_root() -> Path:
    return HOLOMOTION_ROOT / "holoretarget" / "assets"


@dataclass(frozen=True)
class HoloRetargetConfig:
    """Fixed online configuration selected by the HoloRetarget evaluation."""

    asset_root: str = ""
    robot: str = "unitree_g1"
    target_table: str = "ik_match_table2"
    robot_asset: str = "holoretarget_mjcf"
    newton_iterations: int = 1
    root_seed_mode: str = "pelvis"
    max_joint_step: float = 0.5
    joint_limit_weight: float = 8.0
    use_cuda_graph: bool = True
    ground_calibration_frames: int = 50
    ground_calibration_mode: str = "sliding_min"
    ground_target_scope: str = "all"
    ground_lift_only: bool = False
    ground_height: float = 0.0
    profile_timing: bool = False

    @property
    def resolved_asset_root(self) -> Path:
        return Path(self.asset_root).expanduser() if self.asset_root else default_asset_root()


DEFAULT_CONFIG = HoloRetargetConfig()
