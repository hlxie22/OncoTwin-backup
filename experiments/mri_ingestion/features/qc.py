"""Fail-closed QC gates for cached MRI feature rows."""

from __future__ import annotations

import math
from typing import Mapping

from experiments.mri_ingestion.schemas import (
    as_text_list,
    normalize_qc_label,
    stable_unique,
)


LAYER4_POLICY_BY_QC = {
    "high": "conservative_numeric",
    "medium": "conservative_numeric",
    "low": "uncertainty_only",
    "unknown": "report_only",
    "failed": "report_only",
}


def qc_mri_feature_record(
    record: Mapping[str, object],
    *,
    min_tumor_volume_ml: float = 0.01,
    max_tumor_volume_ml: float = 1000.0,
    max_mask_fraction: float = 0.80,
) -> dict[str, object]:
    """Annotate one MRI feature row with fail-closed status and Layer 4 policy."""

    output = dict(record)
    warnings = list(as_text_list(output.get("warnings")))
    segmentation_qc = normalize_qc_label(output.get("segmentation_qc"))
    output["segmentation_qc"] = segmentation_qc
    output["registration_qc"] = normalize_qc_label(output.get("registration_qc"))

    failed = segmentation_qc == "failed"
    mask_voxels = _optional_float(output.get("mask_voxels"))
    tumor_volume_ml = _optional_float(output.get("tumor_volume_ml"))
    voxel_volume_ml = _optional_float(output.get("voxel_volume_ml"))
    image_voxels = _optional_float(output.get("image_voxels"))
    connected_components = _optional_float(output.get("connected_component_count"))

    if mask_voxels is None:
        failed = True
        warnings.append("missing_mask_voxels")
    elif mask_voxels <= 0:
        failed = True
        warnings.append("empty_or_nonpositive_mask")

    if tumor_volume_ml is None:
        failed = True
        warnings.append("missing_tumor_volume_ml")
    elif tumor_volume_ml < min_tumor_volume_ml or tumor_volume_ml > max_tumor_volume_ml:
        failed = True
        warnings.append("implausible_tumor_volume_ml")

    if voxel_volume_ml is None:
        failed = True
        warnings.append("missing_voxel_volume_ml")
    elif voxel_volume_ml <= 0:
        failed = True
        warnings.append("invalid_voxel_volume_ml")

    if image_voxels is not None and mask_voxels is not None and image_voxels > 0:
        if mask_voxels / image_voxels > max_mask_fraction:
            failed = True
            warnings.append("mask_fraction_too_large")

    if connected_components is None:
        warnings.append("missing_connected_component_count")
    elif connected_components < 1:
        failed = True
        warnings.append("invalid_connected_component_count")

    if failed:
        output["mri_feature_status"] = "failed"
        output["segmentation_qc"] = "failed"
        output["layer4_feature_policy"] = LAYER4_POLICY_BY_QC["failed"]
    else:
        output["mri_feature_status"] = "available"
        output["layer4_feature_policy"] = LAYER4_POLICY_BY_QC[segmentation_qc]
    output["warnings"] = list(stable_unique(warnings))
    return output


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
