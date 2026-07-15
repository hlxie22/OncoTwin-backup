"""Simple enhancement summary features for cached per-mask samples."""

from __future__ import annotations

import math
from typing import Sequence


DEFAULT_LOW_ENHANCEMENT_THRESHOLD = 1.0


def enhancement_summary(
    enhancement_values: Sequence[object],
    *,
    mask_voxels: object | None = None,
    voxel_volume_ml: object | None = None,
    low_enhancement_threshold: float = DEFAULT_LOW_ENHANCEMENT_THRESHOLD,
) -> dict[str, object]:
    """Summarize enhancement values sampled inside a tumor mask."""

    values = [_finite_float(value, "enhancement_values") for value in enhancement_values]
    if not values:
        raise ValueError("enhancement_values must contain at least one finite value")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    low_count = sum(value < low_enhancement_threshold for value in values)
    low_fraction = low_count / len(values)

    result: dict[str, object] = {
        "enhancement_mean": mean,
        "enhancement_std": math.sqrt(variance),
        "low_enhancement_fraction": low_fraction,
    }
    if mask_voxels not in (None, "") and voxel_volume_ml not in (None, ""):
        tumor_volume_ml = _positive_float(mask_voxels, "mask_voxels") * _positive_float(
            voxel_volume_ml,
            "voxel_volume_ml",
        )
        result["functional_tumor_volume_ml"] = tumor_volume_ml * (1.0 - low_fraction)
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or value in (None, ""):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number
