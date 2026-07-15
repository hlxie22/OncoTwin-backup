"""Merge cached MRI features into V1 prior-stack eval cohorts.

The V1-D2 lane treats MRI feature extraction as a cached, auditable step. This
module joins a case-level feature table onto the normalized V1-D1 cohort without
dropping rows by default. Failed masks propagate QC and provenance but do not
copy numeric MRI features into the Layer 4 context.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_OUTPUT_PATH = Path(
    "data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort_with_mri.jsonl"
)
SUMMARY_SUFFIX = ".summary.json"

CASE_ID_FIELDS = ("case_id", "patient_id", "subject_id", "participant_id", "ptid")
VALID_QC_LABELS = ("high", "medium", "low", "failed", "unknown")
FAILED_QC_TOKENS = ("failed", "fail", "empty", "no mask", "all zero")
LOW_QC_TOKENS = ("low", "poor", "motion", "artifact", "manual review", "uncertain")
HIGH_QC_TOKENS = ("high", "good", "excellent", "pass", "passed")
UNKNOWN_TOKENS = (
    "",
    "unknown",
    "unspecified",
    "not specified",
    "not available",
    "not assessed",
    "n/a",
    "na",
    "none",
    "null",
)

PROVENANCE_FIELDS = (
    "source_image",
    "source_mask",
    "source_dataset",
    "source_series",
    "feature_extractor",
    "feature_extractor_version",
    "segmentation_model",
    "segmentation_model_version",
)
QC_FIELDS = (
    "segmentation_qc",
    "registration_qc",
    "image_qc",
    "dce_qc",
    "mri_qc",
)
NUMERIC_FEATURE_FIELDS = (
    "tumor_volume_ml",
    "functional_tumor_volume_ml",
    "enhancement_mean",
    "enhancement_std",
    "low_enhancement_fraction",
    "mask_voxels",
    "voxel_volume_ml",
    "connected_component_count",
)
LAYER4_NUMERIC_FIELDS = (
    "tumor_volume_ml",
    "functional_tumor_volume_ml",
    "enhancement_mean",
    "enhancement_std",
    "low_enhancement_fraction",
)
FRACTION_FIELDS = ("low_enhancement_fraction",)
POSITIVE_IF_PRESENT_FIELDS = (
    "tumor_volume_ml",
    "functional_tumor_volume_ml",
    "mask_voxels",
    "voxel_volume_ml",
)
NONNEGATIVE_IF_PRESENT_FIELDS = (
    "enhancement_mean",
    "enhancement_std",
    "connected_component_count",
)


@dataclass(frozen=True)
class MRIFeatureMergeResult:
    """Paths and summary metadata produced by an MRI feature merge."""

    output_path: Path
    summary_path: Path
    summary: Mapping[str, object]


@dataclass(frozen=True)
class _NormalizedFeature:
    case_id: str
    payload: Mapping[str, object]
    status: str
    warnings: tuple[str, ...]


def merge_mri_features_into_v1_cohort(
    cohort_path: Path,
    mri_features_path: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    drop_missing_features: bool = False,
) -> MRIFeatureMergeResult:
    """Join a case-level MRI feature table onto a normalized V1 cohort.

    Cohort rows are preserved by default. Feature rows marked failed, or made
    failed by empty-mask/non-positive-volume checks, retain provenance and QC
    metadata but withhold numeric Layer 4 fields.
    """

    cohort_rows = _read_rows(cohort_path)
    feature_rows = _read_rows(mri_features_path)
    features_by_case = _features_by_case(feature_rows)

    output_rows: list[dict[str, object]] = []
    missing_case_ids: list[str] = []
    matched_case_ids: list[str] = []
    dropped_missing_case_ids: list[str] = []
    feature_status_counts = Counter()
    qc_counts = Counter()
    warning_counts = Counter()
    numeric_feature_copy_counts = Counter()

    for row in cohort_rows:
        case_id = _case_id(row)
        if not case_id:
            raise ValueError("cohort rows require case_id or patient_id")

        feature = features_by_case.get(case_id)
        if feature is None:
            missing_case_ids.append(case_id)
            if drop_missing_features:
                dropped_missing_case_ids.append(case_id)
                continue
            merged = _merge_feature_payload(
                row,
                {
                    "mri_feature_status": "missing",
                    "segmentation_qc": "unknown",
                },
            )
            output_rows.append(merged)
            feature_status_counts["missing"] += 1
            qc_counts["unknown"] += 1
            continue

        matched_case_ids.append(case_id)
        merged = _merge_feature_payload(row, feature.payload)
        output_rows.append(merged)
        feature_status_counts[feature.status] += 1
        qc_counts[str(feature.payload.get("segmentation_qc", "unknown"))] += 1
        for warning in feature.warnings:
            warning_counts[warning] += 1
        for field in LAYER4_NUMERIC_FIELDS:
            if field in feature.payload:
                numeric_feature_copy_counts[field] += 1

    unused_feature_case_ids = sorted(set(features_by_case) - set(matched_case_ids))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, output_rows)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path),
        "mri_features_path": str(mri_features_path),
        "output_path": str(output_path),
        "input_cohort_rows": len(cohort_rows),
        "input_feature_rows": len(feature_rows),
        "output_rows": len(output_rows),
        "matched_feature_rows": len(matched_case_ids),
        "missing_feature_rows": len(missing_case_ids),
        "dropped_missing_feature_rows": len(dropped_missing_case_ids),
        "unused_feature_rows": len(unused_feature_case_ids),
        "feature_status_counts": dict(sorted(feature_status_counts.items())),
        "segmentation_qc_counts": dict(sorted(qc_counts.items())),
        "feature_warning_counts": dict(sorted(warning_counts.items())),
        "numeric_feature_copy_counts": dict(sorted(numeric_feature_copy_counts.items())),
        "mri_feature_completeness": _feature_completeness(output_rows),
        "missing_case_ids_sample": missing_case_ids[:25],
        "unused_feature_case_ids_sample": unused_feature_case_ids[:25],
    }
    summary_path = output_path.with_name(f"{output_path.stem}{SUMMARY_SUFFIX}")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MRIFeatureMergeResult(output_path, summary_path, summary)


def _features_by_case(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, _NormalizedFeature]:
    features: dict[str, _NormalizedFeature] = {}
    for index, row in enumerate(rows, start=1):
        feature = _normalize_feature_row(row)
        if feature.case_id in features:
            raise ValueError(
                f"duplicate MRI feature row for case_id {feature.case_id} "
                f"at input row {index}"
            )
        features[feature.case_id] = feature
    return features


def _normalize_feature_row(row: Mapping[str, object]) -> _NormalizedFeature:
    case_id = _case_id(row)
    if not case_id:
        raise ValueError("MRI feature rows require case_id or patient_id")

    warnings = list(_as_text_list(row.get("warnings")))
    qc = _segmentation_qc(row)
    registration_qc = _normalize_qc(row.get("registration_qc"))
    status = "available"

    if qc == "failed":
        status = "failed"
    if _empty_or_nonpositive_mask(row):
        status = "failed"
        qc = "failed"
        warnings.append("empty_or_nonpositive_mask")
    if _nonpositive_tumor_volume(row):
        status = "failed"
        qc = "failed"
        warnings.append("non_positive_tumor_volume_ml")

    payload: dict[str, object] = {
        "mri_feature_status": status,
        "segmentation_qc": qc,
        "registration_qc": registration_qc,
    }

    for field in PROVENANCE_FIELDS:
        if _present(row.get(field)):
            payload[field] = row[field]

    if status == "failed":
        payload["warnings"] = _stable_unique(warnings)
        return _NormalizedFeature(
            case_id=case_id,
            payload=payload,
            status=status,
            warnings=tuple(payload["warnings"]),
        )

    for field in NUMERIC_FEATURE_FIELDS:
        if not _present(row.get(field)):
            continue
        value = _validated_numeric_feature(row[field], field)
        payload[field] = value
        if field == "tumor_volume_ml":
            payload["volume_ml"] = value

    for field in QC_FIELDS:
        if field in {"segmentation_qc", "registration_qc"}:
            continue
        if _present(row.get(field)):
            payload[field] = _normalize_qc(row[field])

    payload["warnings"] = _stable_unique(warnings)
    return _NormalizedFeature(
        case_id=case_id,
        payload=payload,
        status=status,
        warnings=tuple(payload["warnings"]),
    )


def _merge_feature_payload(
    cohort_row: Mapping[str, object],
    feature_payload: Mapping[str, object],
) -> dict[str, object]:
    merged = _json_ready_mapping(cohort_row)
    existing = merged.get("mri_features")
    if isinstance(existing, Mapping):
        mri_features = dict(existing)
    else:
        mri_features = {}
    mri_features.update(_json_ready_mapping(feature_payload))
    merged["mri_features"] = mri_features
    return merged


def _segmentation_qc(row: Mapping[str, object]) -> str:
    for field in ("segmentation_qc", "mri_qc", "qc_status", "qc"):
        if _present(row.get(field)):
            return _normalize_qc(row[field])
    return "unknown"


def _normalize_qc(value: object) -> str:
    if not _present(value):
        return "unknown"
    text = _normalize(value)
    if any(token in text for token in FAILED_QC_TOKENS):
        return "failed"
    if any(token in text for token in LOW_QC_TOKENS):
        return "low"
    if any(token in text for token in HIGH_QC_TOKENS):
        return "high"
    if text in VALID_QC_LABELS:
        return text
    return "unknown"


def _empty_or_nonpositive_mask(row: Mapping[str, object]) -> bool:
    mask_voxels = _optional_number(row.get("mask_voxels"), "mask_voxels")
    return mask_voxels is not None and mask_voxels <= 0


def _nonpositive_tumor_volume(row: Mapping[str, object]) -> bool:
    volume = _optional_number(row.get("tumor_volume_ml"), "tumor_volume_ml")
    return volume is not None and volume <= 0


def _validated_numeric_feature(value: object, field: str) -> float:
    numeric = _require_finite(value, field)
    if field in FRACTION_FIELDS and not 0 <= numeric <= 1:
        raise ValueError(f"{field} must be in [0, 1]")
    if field in POSITIVE_IF_PRESENT_FIELDS and numeric <= 0:
        raise ValueError(f"{field} must be positive")
    if field in NONNEGATIVE_IF_PRESENT_FIELDS and numeric < 0:
        raise ValueError(f"{field} must be non-negative")
    return numeric


def _optional_number(value: object, field: str) -> float | None:
    if not _present(value):
        return None
    return _require_finite(value, field)


def _require_finite(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _feature_completeness(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        features = row.get("mri_features")
        if not isinstance(features, Mapping):
            continue

        for field in LAYER4_NUMERIC_FIELDS:
            if _present(features.get(field)):
                counts[field] += 1

        for field in ("segmentation_qc", "registration_qc"):
            if field in features and features[field] is not None:
                counts[field] += 1

    return dict(sorted(counts.items()))

def _read_rows(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            _require_mapping(json.loads(line), path)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [_require_mapping(row, path) for row in payload]
        if isinstance(payload, Mapping):
            for key in ("cases", "rows", "features", "mri_features", "data"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [_require_mapping(row, path) for row in rows]
            return [_require_mapping(payload, path)]
        raise ValueError(f"JSON source must contain an object or list: {path}")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"source table must be .csv, .jsonl, or .json: {path}")


def _require_mapping(value: object, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"source rows must be JSON objects: {path}")
    return value


def _case_id(row: Mapping[str, object]) -> str:
    for field in CASE_ID_FIELDS:
        if _present(row.get(field)):
            return str(row[field]).strip()
    return ""


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return _normalize(value) not in UNKNOWN_TOKENS
    return True


def _normalize(value: object) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def _as_text_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if _present(item))
    if not _present(value):
        return ()
    return (str(value),)


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_ready_mapping(row: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_ready(value) for key, value in row.items()}


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value

