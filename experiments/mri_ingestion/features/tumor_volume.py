"""Tumor-volume feature helpers for cached mask metadata."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def tumor_volume_features(
    *,
    mask_voxels: object,
    voxel_volume_ml: object | None = None,
    voxel_spacing_mm: Sequence[object] | None = None,
) -> dict[str, object]:
    """Compute tumor volume in mL from voxel count and voxel geometry."""

    voxels = _positive_int(mask_voxels, "mask_voxels")
    resolved_voxel_volume_ml = resolve_voxel_volume_ml(
        voxel_volume_ml=voxel_volume_ml,
        voxel_spacing_mm=voxel_spacing_mm,
    )
    return {
        "mask_voxels": voxels,
        "voxel_volume_ml": resolved_voxel_volume_ml,
        "tumor_volume_ml": voxels * resolved_voxel_volume_ml,
    }


def resolve_voxel_volume_ml(
    *,
    voxel_volume_ml: object | None = None,
    voxel_spacing_mm: Sequence[object] | None = None,
) -> float:
    if voxel_volume_ml not in (None, ""):
        return _positive_float(voxel_volume_ml, "voxel_volume_ml")
    if voxel_spacing_mm is None or len(voxel_spacing_mm) != 3:
        raise ValueError("voxel_spacing_mm must contain three values when voxel_volume_ml is absent")
    spacing = [_positive_float(value, "voxel_spacing_mm") for value in voxel_spacing_mm]
    return math.prod(spacing) / 1000.0


def merge_volume_metadata(record: Mapping[str, object]) -> dict[str, object]:
    """Convenience wrapper for rows that already use the feature table field names."""

    spacing = record.get("voxel_spacing_mm")
    if isinstance(spacing, str):
        spacing = [part.strip() for part in spacing.split(",")]
    return tumor_volume_features(
        mask_voxels=record.get("mask_voxels"),
        voxel_volume_ml=record.get("voxel_volume_ml"),
        voxel_spacing_mm=spacing if isinstance(spacing, Sequence) and not isinstance(spacing, (str, bytes)) else None,
    )


def _positive_int(value: object, name: str) -> int:
    number = int(_positive_float(value, name))
    if float(number) != float(value):
        raise ValueError(f"{name} must be an integer count")
    return number


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or value in (None, ""):
        raise ValueError(f"{name} must be positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive")
    return number
