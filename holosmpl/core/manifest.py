from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    dataset: str
    split: str
    raw_key: str
    raw_path: Path
