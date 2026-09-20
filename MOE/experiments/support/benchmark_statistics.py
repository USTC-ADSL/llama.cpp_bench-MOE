"""Shared percentile convention for retained benchmark/profile samples."""
import math
from statistics import fmean
from typing import Sequence


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile requires values and a fraction in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("summary requires at least one value")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise ValueError("summary values must be finite and non-negative")
    return {"count": len(numeric), "min": min(numeric), "mean": fmean(numeric),
            "p50": percentile(numeric, 0.50), "p95": percentile(numeric, 0.95), "p99": percentile(numeric, 0.99)}
