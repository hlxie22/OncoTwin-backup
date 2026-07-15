"""Build V1 MRI feature rows from cached per-case metadata."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from experiments.mri_ingestion.features.enhancement_features import enhancement_summary
from experiments.mri_ingestion.features.qc import qc_mri_feature_record
from experiments.mri_ingestion.features.tumor_volume import tumor_volume_features
from experiments.mri_ingestion.schemas import as_text_list, present, require_case_id


PASSTHROUGH_FIELDS = (
    "source_dataset",
    "source_series",
    "feature_extractor",
    "feature_extractor_version",
    "segmentation_model",
    "segmentation_model_version",
    "image_voxels",
    "connected_component_count",
)


def extract_case_feature_record(row: Mapping[str, object]) -> dict[str, object]:
    """Convert one cached mask/intensity metadata row into the V1 feature contract."""

    case_id = require_case_id(row)
    output: dict[str, object] = {"case_id": case_id}
    _copy_first(output, row, "source_image", ("source_image", "image_path", "input_image"))
    _copy_first(output, row, "source_mask", ("source_mask", "mask_path", "segmentation_path"))
    for field in PASSTHROUGH_FIELDS:
        if present(row.get(field)):
            output[field] = row[field]

    spacing = _spacing(row.get("voxel_spacing_mm"))
    if present(row.get("mask_voxels")):
        try:
            output.update(
                tumor_volume_features(
                    mask_voxels=row.get("mask_voxels"),
                    voxel_volume_ml=row.get("voxel_volume_ml"),
                    voxel_spacing_mm=spacing,
                )
            )
        except ValueError as error:
            output["mask_voxels"] = row.get("mask_voxels")
            output["warnings"] = [*as_text_list(output.get("warnings")), str(error)]
    for field in ("tumor_volume_ml", "functional_tumor_volume_ml", "voxel_volume_ml"):
        if present(row.get(field)) and field not in output:
            output[field] = row[field]

    enhancement_values = _number_sequence(
        row.get("enhancement_values", row.get("enhancement_samples"))
    )
    if enhancement_values:
        output.update(
            enhancement_summary(
                enhancement_values,
                mask_voxels=output.get("mask_voxels"),
                voxel_volume_ml=output.get("voxel_volume_ml"),
                low_enhancement_threshold=float(
                    row.get("low_enhancement_threshold", 1.0)
                ),
            )
        )
    else:
        for field in ("enhancement_mean", "enhancement_std", "low_enhancement_fraction"):
            if present(row.get(field)):
                output[field] = row[field]

    output["segmentation_qc"] = row.get("segmentation_qc", "unknown")
    output["registration_qc"] = row.get("registration_qc", "unknown")
    output["warnings"] = [*as_text_list(output.get("warnings")), *as_text_list(row.get("warnings"))]
    return qc_mri_feature_record(output)


def extract_feature_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [extract_case_feature_record(row) for row in rows]


def _copy_first(
    output: dict[str, object],
    row: Mapping[str, object],
    target: str,
    source_fields: Sequence[str],
) -> None:
    for field in source_fields:
        if present(row.get(field)):
            output[target] = row[field]
            return


def _spacing(value: object) -> list[object] | None:
    if not present(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            parsed = json.loads(text)
            return list(parsed) if isinstance(parsed, Sequence) else None
        return [part.strip() for part in text.split(",")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return None


def _number_sequence(value: object) -> list[object]:
    if not present(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            parsed = json.loads(text)
            return list(parsed) if isinstance(parsed, Sequence) else []
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []
