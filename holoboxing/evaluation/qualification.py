"""Separate local compatibility from the official hardware latency gate."""

import re


def hardware_matches(actual, required="RTX 5090"):
    tokenize = lambda name: re.findall(r"[a-z]+|\d+", name.lower())
    actual_tokens, required_tokens = tokenize(actual), tokenize(required)
    return any(
        actual_tokens[index : index + len(required_tokens)] == required_tokens
        for index in range(len(actual_tokens) - len(required_tokens) + 1)
    )


def qualification(gpu, mean_ms, *, required_gpu="RTX 5090"):
    import math

    if not math.isfinite(mean_ms) or mean_ms <= 0:
        raise ValueError(
            "Completed chain mean latency must be finite and positive"
        )
    match = hardware_matches(gpu, required_gpu)
    return {
        "measured_gpu": gpu,
        "required_gpu": required_gpu,
        "target_hardware_match": match,
        "threshold_ms": 5.0,
        "local_mean_within_budget": mean_ms <= 5.0,
        "official_status": ("PASS" if mean_ms <= 5.0 else "FAIL")
        if match
        else "NOT_EVALUATED",
        "scope_notice": (
            "Measured on the official hardware class; qualification applies to this recorded environment."
            if match
            else "These measurements describe this GPU and environment only. Official hardware qualification "
            "is not established. Latency cannot be translated by a fixed multiplier."
        ),
    }
