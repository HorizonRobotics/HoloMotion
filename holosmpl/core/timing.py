from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterable, Mapping


class StageTimer:
    """Small JSON-friendly stage timer for data production workers."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._stage_ms: defaultdict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stage_ms[stage] += (time.perf_counter() - start) * 1000.0

    def add(self, stage: str, elapsed_ms: float) -> None:
        self._stage_ms[stage] += float(elapsed_ms)

    def update(self, timing_ms: Mapping[str, Any]) -> None:
        for key, value in timing_ms.items():
            if key == "total_ms":
                continue
            try:
                self._stage_ms[str(key)] += float(value)
            except (TypeError, ValueError):
                continue

    def finish(self) -> dict[str, float]:
        total_ms = (time.perf_counter() - self._start) * 1000.0
        out = {key: round(value, 3) for key, value in sorted(self._stage_ms.items())}
        out["total_ms"] = round(total_ms, 3)
        return out


def summarize_timing(
    items: Iterable[Mapping[str, Any]],
    *,
    timing_key: str = "timing_ms",
    frame_key: str | None = None,
) -> dict[str, Any]:
    timings: list[Mapping[str, Any]] = []
    total_frames = 0
    for item in items:
        timing = item.get(timing_key)
        if isinstance(timing, Mapping):
            timings.append(timing)
        if frame_key is not None:
            try:
                total_frames += int(item.get(frame_key, 0))
            except (TypeError, ValueError):
                pass

    stage_values: defaultdict[str, list[float]] = defaultdict(list)
    for timing in timings:
        for key, value in timing.items():
            try:
                stage_values[str(key)].append(float(value))
            except (TypeError, ValueError):
                continue

    stage_summary = {
        stage: _summarize_values(values)
        for stage, values in sorted(stage_values.items())
    }
    total_ms = stage_summary.get("total_ms", {}).get("sum_ms", 0.0)
    summary: dict[str, Any] = {
        "count": len(timings),
        "total_ms": total_ms,
        "stages": stage_summary,
    }
    if frame_key is not None:
        summary["frame_count"] = total_frames
        summary["frames_per_total_stage_second"] = (
            round(total_frames / (total_ms / 1000.0), 3) if total_ms > 0 else None
        )
    return summary


def format_timing_summary(summary: Mapping[str, Any], *, stages: tuple[str, ...]) -> str:
    stage_summary = summary.get("stages", {})
    parts = []
    for stage in stages:
        if not isinstance(stage_summary, Mapping) or stage not in stage_summary:
            continue
        stage_data = stage_summary[stage]
        if isinstance(stage_data, Mapping):
            parts.append(f"{stage}={float(stage_data.get('sum_ms', 0.0)) / 1000.0:.2f}s")
    total_s = float(summary.get("total_ms", 0.0)) / 1000.0
    if total_s > 0:
        parts.insert(0, f"total={total_s:.2f}s")
    fps = summary.get("frames_per_total_stage_second")
    if fps is not None:
        parts.append(f"stage_fps={float(fps):.1f}")
    return " | ".join(parts)


def _summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "sum_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    ordered = sorted(values)
    total = sum(ordered)
    return {
        "count": len(ordered),
        "sum_ms": round(total, 3),
        "mean_ms": round(total / len(ordered), 3),
        "p50_ms": round(_percentile(ordered, 50.0), 3),
        "p95_ms": round(_percentile(ordered, 95.0), 3),
        "max_ms": round(ordered[-1], 3),
    }


def _percentile(ordered_values: list[float], percentile: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    rank = (percentile / 100.0) * (len(ordered_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered_values) - 1)
    frac = rank - lo
    return ordered_values[lo] * (1.0 - frac) + ordered_values[hi] * frac
