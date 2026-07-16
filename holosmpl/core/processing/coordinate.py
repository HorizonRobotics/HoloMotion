from __future__ import annotations

import numpy as np


def yup_trans_to_canonical_zup(trans: np.ndarray) -> np.ndarray:
    """Convert y-up source translation to project canonical z-up coordinates."""

    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must be [T,3], got {trans.shape}")
    out = np.empty_like(trans)
    out[:, 0] = trans[:, 0]
    out[:, 1] = -trans[:, 2]
    out[:, 2] = trans[:, 1]
    return out
