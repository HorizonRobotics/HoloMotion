"""Detect and discard nonfinite rows belonging to individual paired worlds."""

import torch

from holoboxing.training.rsl_rl.algorithms.ppo import require_finite


def nonfinite_rows(value, num_envs, device):
    checks = []

    def visit(item):
        if torch.is_tensor(item):
            if item.ndim and item.shape[0] == num_envs:
                checks.append(
                    ~torch.isfinite(item).reshape(num_envs, -1).all(-1)
                )
            else:
                # Non-world data cannot safely be attributed to a local failure.
                require_finite(item, "shared environment state")
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            require_finite(item, "shared environment state")

    visit(value)
    return (
        torch.stack(checks).any(0)
        if checks
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )


def clear_invalid_rows(value, invalid):
    """Finite placeholders only; storage excludes these rows from every loss."""
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == len(invalid):
            value[invalid] = 0
    elif isinstance(value, dict):
        for child in value.values():
            clear_invalid_rows(child, invalid)
    elif isinstance(value, (tuple, list)):
        for child in value:
            clear_invalid_rows(child, invalid)
