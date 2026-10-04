#!/usr/bin/env python3
"""Audit whether a new longitudinal observation can improve frozen Layer 6.

This audit separates:

1. Raw imaging evidence for a T2/inter-regimen landmark.
2. Processed positive-volume evidence at a day strictly after the Layer 6
   early-MRI landmark and strictly before the held-out final landmark.
3. Existing normalized-cohort measurements from information that may still
   need to be extracted from the raw imaging inventory.

It does not train, select, or evaluate a new predictive model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence


ANALYSIS_VERSION = "oncotwin_layer8_temporal_landmark_audit_v0_1"

CASE_ID_NAMES = {
    "caseid",
    "patientid",
    "subjectid",
    "participantid",
    "ptid",
}

DAY_NAMES = {
    "day",
    "relativeday",
    "daysfrombaseline",
    "studyday",
    "timepointday",
    "mriday",
}

VOLUME_NAMES = {
    "tumorvolumeml",
    "volumeml",
    "functionaltumorvolumeml",
    "ftvml",
    "ftv",
}

BASELINE_DAY_FIELDS = (
    "baseline_day",
    "baseline_relative_day",
    "initial_day",
)
EARLY_DAY_FIELDS = (
    "early_day",
    "followup_day",
    "first_followup_day",
)
FINAL_DAY_FIELDS = (
    "final_day",
    "heldout_day",
    "outcome_day",
    "last_day",
)

BASELINE_VOLUME_FIELDS = (
    "baseline_volume_ml",
    "baseline_tumor_volume_ml",
    "initial_volume_ml",
    "baseline_ftv_ml",
)
EARLY_VOLUME_FIELDS = (
    "early_volume_ml",
    "followup_volume_ml",
    "early_tumor_volume_ml",
    "early_ftv_ml",
)
FINAL_VOLUME_FIELDS = (
    "final_volume_ml",
    "heldout_volume_ml",
    "outcome_volume_ml",
    "final_tumor_volume_ml",
    "final_ftv_ml",
)

NOMINAL_DAYS = {
    "t0": 0.0,
    "t1": 21.0,
    "t2": 84.0,
    "t3": 140.0,
}

SEMANTIC_PATTERNS = {
    "t0": (
        r"(^|[^a-z0-9])baseline([^a-z0-9]|$)",
        r"(^|[^a-z0-9])pre[\s_-]*treatment([^a-z0-9]|$)",
        r"(^|[^a-z0-9])pretreatment([^a-z0-9]|$)",
    ),
    "t1": (
        r"(^|[^a-z0-9])early([^a-z0-9]|$)",
        r"(^|[^a-z0-9])week[\s_-]*3([^a-z0-9]|$)",
    ),
    "t2": (
        r"(^|[^a-z0-9])inter[\s_-]*regimen([^a-z0-9]|$)",
        r"(^|[^a-z0-9])mid[\s_-]*treatment([^a-z0-9]|$)",
        r"(^|[^a-z0-9])midpoint([^a-z0-9]|$)",
        r"(^|[^a-z0-9])week[\s_-]*12([^a-z0-9]|$)",
    ),
    "t3": (
        r"(^|[^a-z0-9])pre[\s_-]*surgery([^a-z0-9]|$)",
        r"(^|[^a-z0-9])presurgery([^a-z0-9]|$)",
        r"(^|[^a-z0-9])final([^a-z0-9]|$)",
    ),
}


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _nonempty_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_number(value: object) -> float | None:
    number = _finite_number(value)
    if number is None or number <= 0:
        return None
    return number


def _first_number(
    row: Mapping[str, object],
    names: Sequence[str],
) -> float | None:
    for name in names:
        number = _finite_number(row.get(name))
        if number is not None:
            return number
    return None


def _first_positive(
    row: Mapping[str, object],
    names: Sequence[str],
) -> float | None:
    for name in names:
        number = _positive_number(row.get(name))
        if number is not None:
            return number
    return None


def _fingerprint(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
        }

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)

    return {
        "path": str(path),
        "exists": True,
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def _delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _candidate_columns(
    fieldnames: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    case_columns: list[str] = []
    explicit_columns: list[str] = []
    weak_columns: list[str] = []

    for field in fieldnames:
        normalized = _normalized_name(field)

        if normalized in CASE_ID_NAMES:
            case_columns.append(field)

        if any(
            token in normalized
            for token in (
                "timepoint",
                "visitlabel",
                "visitname",
                "visitnumber",
                "mritime",
                "treatmentphase",
                "examlabel",
            )
        ):
            explicit_columns.append(field)
        elif any(
            token in normalized
            for token in (
                "visit",
                "study",
                "series",
                "description",
                "folder",
                "path",
                "protocol",
                "collection",
            )
        ):
            weak_columns.append(field)

    return case_columns, explicit_columns, weak_columns


def _classify_landmarks(
    value: object,
    *,
    allow_short_codes: bool,
) -> set[str]:
    text = _nonempty_text(value)
    if text is None:
        return set()

    lowered = text.lower()
    labels: set[str] = set()

    for label, patterns in SEMANTIC_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            labels.add(label)

    if not allow_short_codes:
        return labels

    stripped = lowered.strip()

    if re.fullmatch(r"[0-3]", stripped):
        labels.add(f"t{stripped}")

    for index in range(4):
        short_patterns = (
            rf"(^|[^a-z0-9])t[\s_-]*{index}([^a-z0-9]|$)",
            rf"(^|[^a-z0-9])timepoint[\s_-]*{index}([^a-z0-9]|$)",
            rf"(^|[^a-z0-9])visit[\s_-]*{index}([^a-z0-9]|$)",
        )
        if any(re.search(pattern, lowered) for pattern in short_patterns):
            labels.add(f"t{index}")

    return labels


def _row_case_id(
    row: Mapping[str, object],
    case_columns: Sequence[str],
) -> str | None:
    for field in case_columns:
        value = _nonempty_text(row.get(field))
        if value is not None:
            return value
    return None


def _profile_delimited_landmarks(
    path: Path,
    *,
    eligible_case_ids: set[str],
) -> tuple[dict[str, object], dict[str, set[str]]]:
    if not path.is_file():
        return (
            {
                **_fingerprint(path),
                "row_count": 0,
                "case_count": 0,
                "case_id_columns": [],
                "explicit_landmark_columns": [],
                "weak_context_columns": [],
                "explicit_landmark_case_counts": {},
                "weak_landmark_case_counts": {},
                "layer6_eligible_explicit_t2_case_count": 0,
                "layer6_eligible_weak_t2_only_case_count": 0,
                "complete_explicit_t0_t1_t2_t3_case_count": 0,
            },
            {label: set() for label in NOMINAL_DAYS},
        )

    explicit_case_ids = {
        label: set()
        for label in NOMINAL_DAYS
    }
    weak_case_ids = {
        label: set()
        for label in NOMINAL_DAYS
    }
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_case_ids: set[str] = set()
    row_count = 0

    with path.open(
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter=_delimiter(path),
        )
        fieldnames = list(reader.fieldnames or [])
        (
            case_columns,
            explicit_columns,
            weak_columns,
        ) = _candidate_columns(fieldnames)

        for row in reader:
            row_count += 1
            case_id = _row_case_id(row, case_columns)
            if case_id is None:
                continue

            all_case_ids.add(case_id)

            explicit_labels: set[str] = set()
            for field in explicit_columns:
                labels = _classify_landmarks(
                    row.get(field),
                    allow_short_codes=True,
                )
                explicit_labels.update(labels)
                field_counts[field].update(labels)

            weak_labels: set[str] = set()
            for field in weak_columns:
                labels = _classify_landmarks(
                    row.get(field),
                    allow_short_codes=False,
                )
                weak_labels.update(labels)
                field_counts[field].update(labels)

            for label in explicit_labels:
                explicit_case_ids[label].add(case_id)

            for label in weak_labels:
                weak_case_ids[label].add(case_id)

    explicit_t2_overlap = (
        explicit_case_ids["t2"]
        & eligible_case_ids
    )
    weak_t2_only_overlap = (
        weak_case_ids["t2"]
        - explicit_case_ids["t2"]
    ) & eligible_case_ids

    complete_explicit = set(all_case_ids)
    for label in NOMINAL_DAYS:
        complete_explicit &= explicit_case_ids[label]

    summary = {
        **_fingerprint(path),
        "row_count": row_count,
        "case_count": len(all_case_ids),
        "case_id_columns": case_columns,
        "explicit_landmark_columns": explicit_columns,
        "weak_context_columns": weak_columns,
        "explicit_landmark_case_counts": {
            label: len(explicit_case_ids[label])
            for label in NOMINAL_DAYS
        },
        "weak_landmark_case_counts": {
            label: len(weak_case_ids[label])
            for label in NOMINAL_DAYS
        },
        "field_landmark_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in sorted(field_counts.items())
            if counts
        },
        "layer6_eligible_explicit_t2_case_count": (
            len(explicit_t2_overlap)
        ),
        "layer6_eligible_explicit_t2_case_sample": (
            sorted(explicit_t2_overlap)[:20]
        ),
        "layer6_eligible_weak_t2_only_case_count": (
            len(weak_t2_only_overlap)
        ),
        "layer6_eligible_weak_t2_only_case_sample": (
            sorted(weak_t2_only_overlap)[:20]
        ),
        "complete_explicit_t0_t1_t2_t3_case_count": (
            len(complete_explicit)
        ),
        "complete_explicit_t0_t1_t2_t3_case_sample": (
            sorted(complete_explicit)[:20]
        ),
    }

    return summary, explicit_case_ids


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            yield payload


def _measurement_points(
    row: Mapping[str, object],
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    measurements = row.get("measurements")
    if isinstance(measurements, list):
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                continue

            day = _first_number(
                measurement,
                (
                    "day",
                    "relative_day",
                    "days_from_baseline",
                    "study_day",
                    "timepoint_day",
                    "mri_day",
                ),
            )
            volume = _first_positive(
                measurement,
                (
                    "tumor_volume_ml",
                    "volume_ml",
                    "functional_tumor_volume_ml",
                    "ftv_ml",
                    "ftv",
                ),
            )
            if day is not None and volume is not None:
                points.append((day, volume))

    wide_points = (
        (
            _first_number(row, BASELINE_DAY_FIELDS),
            _first_positive(row, BASELINE_VOLUME_FIELDS),
        ),
        (
            _first_number(row, EARLY_DAY_FIELDS),
            _first_positive(row, EARLY_VOLUME_FIELDS),
        ),
        (
            _first_number(row, FINAL_DAY_FIELDS),
            _first_positive(row, FINAL_VOLUME_FIELDS),
        ),
    )

    for day, volume in wide_points:
        if day is not None and volume is not None:
            points.append((day, volume))

    return sorted(set(points))


def _profile_cohort(
    path: Path,
) -> tuple[
    dict[str, object],
    dict[str, tuple[float, float]],
    set[str],
]:
    if not path.is_file():
        raise FileNotFoundError(path)

    case_ids: set[str] = set()
    eligible: dict[str, tuple[float, float]] = {}
    strict_intermediate_ids: set[str] = set()
    measurement_counts: Counter[int] = Counter()
    duplicate_case_ids: set[str] = set()

    for row in _iter_jsonl(path):
        case_id = None
        for field in (
            "case_id",
            "patient_id",
            "subject_id",
            "participant_id",
            "ptid",
        ):
            value = _nonempty_text(row.get(field))
            if value is not None:
                case_id = value
                break

        if case_id is None:
            continue

        if case_id in case_ids:
            duplicate_case_ids.add(case_id)
        case_ids.add(case_id)

        baseline_volume = _first_positive(
            row,
            BASELINE_VOLUME_FIELDS,
        )
        early_volume = _first_positive(
            row,
            EARLY_VOLUME_FIELDS,
        )
        early_day = _first_number(
            row,
            EARLY_DAY_FIELDS,
        )
        final_day = _first_number(
            row,
            FINAL_DAY_FIELDS,
        )

        points = _measurement_points(row)
        measurement_counts[len(points)] += 1

        if (
            baseline_volume is None
            or early_volume is None
            or early_day is None
            or early_day <= 0
            or final_day is None
            or final_day <= early_day
        ):
            continue

        eligible[case_id] = (
            early_day,
            final_day,
        )

        if any(
            early_day < day < final_day
            for day, _volume in points
        ):
            strict_intermediate_ids.add(case_id)

    summary = {
        **_fingerprint(path),
        "case_count": len(case_ids),
        "duplicate_case_id_count": len(duplicate_case_ids),
        "layer6_eligible_case_count": len(eligible),
        "measurement_count_distribution": {
            str(count): cases
            for count, cases in sorted(measurement_counts.items())
        },
        "strict_post_early_pre_final_volume_case_count": (
            len(strict_intermediate_ids)
        ),
        "strict_post_early_pre_final_volume_case_sample": (
            sorted(strict_intermediate_ids)[:20]
        ),
    }

    return summary, eligible, strict_intermediate_ids


def _feature_columns(
    fieldnames: Sequence[str],
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    (
        case_columns,
        explicit_columns,
        weak_columns,
    ) = _candidate_columns(fieldnames)

    day_columns: list[str] = []
    volume_columns: list[str] = []

    for field in fieldnames:
        normalized = _normalized_name(field)

        if (
            normalized in DAY_NAMES
            or normalized.endswith("day")
        ):
            day_columns.append(field)

        if (
            normalized in VOLUME_NAMES
            or (
                "volume" in normalized
                and (
                    normalized.endswith("ml")
                    or "tumorvolume" in normalized
                    or "functional" in normalized
                )
            )
        ):
            volume_columns.append(field)

    return (
        case_columns,
        explicit_columns,
        weak_columns,
        day_columns,
        volume_columns,
    )


def _profile_longitudinal_features(
    path: Path,
    *,
    eligible: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, object], set[str]]:
    if not path.is_file():
        return (
            {
                **_fingerprint(path),
                "row_count": 0,
                "case_count": 0,
                "positive_volume_row_count": 0,
                "usable_post_early_pre_final_volume_case_count": 0,
                "usable_post_early_pre_final_volume_case_sample": [],
                "positive_t2_volume_case_count": 0,
                "positive_t2_volume_case_sample": [],
            },
            set(),
        )

    all_case_ids: set[str] = set()
    usable_case_ids: set[str] = set()
    positive_t2_case_ids: set[str] = set()
    row_count = 0
    positive_volume_rows = 0

    with path.open(
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter=_delimiter(path),
        )
        fieldnames = list(reader.fieldnames or [])
        (
            case_columns,
            explicit_columns,
            weak_columns,
            day_columns,
            volume_columns,
        ) = _feature_columns(fieldnames)

        for row in reader:
            row_count += 1
            case_id = _row_case_id(row, case_columns)
            if case_id is None:
                continue

            all_case_ids.add(case_id)

            volume = None
            for field in volume_columns:
                volume = _positive_number(row.get(field))
                if volume is not None:
                    break

            if volume is None:
                continue

            positive_volume_rows += 1

            explicit_labels: set[str] = set()
            for field in explicit_columns:
                explicit_labels.update(
                    _classify_landmarks(
                        row.get(field),
                        allow_short_codes=True,
                    )
                )

            weak_labels: set[str] = set()
            for field in weak_columns:
                weak_labels.update(
                    _classify_landmarks(
                        row.get(field),
                        allow_short_codes=False,
                    )
                )

            labels = explicit_labels or weak_labels

            day = None
            for field in day_columns:
                day = _finite_number(row.get(field))
                if day is not None:
                    break

            if day is None:
                for label in ("t0", "t1", "t2", "t3"):
                    if label in labels:
                        day = NOMINAL_DAYS[label]
                        break

            if "t2" in labels:
                positive_t2_case_ids.add(case_id)

            bounds = eligible.get(case_id)
            if bounds is None or day is None:
                continue

            early_day, final_day = bounds
            if early_day < day < final_day:
                usable_case_ids.add(case_id)

    summary = {
        **_fingerprint(path),
        "row_count": row_count,
        "case_count": len(all_case_ids),
        "positive_volume_row_count": positive_volume_rows,
        "usable_post_early_pre_final_volume_case_count": (
            len(usable_case_ids)
        ),
        "usable_post_early_pre_final_volume_case_sample": (
            sorted(usable_case_ids)[:20]
        ),
        "positive_t2_volume_case_count": (
            len(positive_t2_case_ids)
        ),
        "positive_t2_volume_case_sample": (
            sorted(positive_t2_case_ids)[:20]
        ),
    }

    return summary, usable_case_ids


def run_audit(
    *,
    cohort_path: Path,
    manifest_path: Path,
    clinical_path: Path | None,
    longitudinal_features_path: Path,
) -> dict[str, object]:
    (
        cohort,
        eligible,
        cohort_intermediate_ids,
    ) = _profile_cohort(cohort_path)

    manifest, explicit_manifest_ids = (
        _profile_delimited_landmarks(
            manifest_path,
            eligible_case_ids=set(eligible),
        )
    )

    if clinical_path is None:
        clinical = {
            "path": None,
            "exists": False,
        }
    else:
        clinical, _clinical_ids = (
            _profile_delimited_landmarks(
                clinical_path,
                eligible_case_ids=set(eligible),
            )
        )

    longitudinal_features, feature_intermediate_ids = (
        _profile_longitudinal_features(
            longitudinal_features_path,
            eligible=eligible,
        )
    )

    explicit_t2_ids = (
        explicit_manifest_ids["t2"]
        & set(eligible)
    )
    usable_ids = (
        cohort_intermediate_ids
        | feature_intermediate_ids
    )

    if usable_ids:
        usable_status = "established"
        decision = (
            "proceed_to_frozen_layer6_t1_to_t3_baseline_"
            "and_t2_incremental_value_screen"
        )
        conclusion = (
            "At least one positive-volume observation exists strictly "
            "after the Layer 6 early-MRI landmark and before the final "
            "target. The next step is a leakage-controlled incremental-"
            "value screen with Layer 6 held frozen."
        )
    elif explicit_t2_ids:
        usable_status = (
            "not_established_missing_processed_t2_volume"
        )
        decision = (
            "build_t2_volume_extraction_and_case_crosswalk"
        )
        conclusion = (
            "The raw inventory contains explicit T2/inter-regimen "
            "candidates overlapping Layer 6 cases, but no usable "
            "post-early, pre-final positive-volume observation has yet "
            "been established. Extract and QC T2 volumes before model work."
        )
    elif int(
        manifest.get(
            "layer6_eligible_weak_t2_only_case_count",
            0,
        )
    ) > 0:
        usable_status = (
            "not_established_ambiguous_textual_t2_evidence"
        )
        decision = (
            "manually_validate_ambiguous_t2_inventory_labels"
        )
        conclusion = (
            "Only weak textual evidence suggests a T2 landmark. Validate "
            "the source labels before extraction or predictive modeling."
        )
    else:
        usable_status = "not_found"
        decision = (
            "stop_layer6_longitudinal_updater_and_reassess_data_source"
        )
        conclusion = (
            "No explicit new T2/inter-regimen landmark was found for the "
            "Layer 6-eligible cohort. A longitudinal updater is not yet "
            "supported by this data inventory."
        )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete",
        "selection_performed": False,
        "cohort": cohort,
        "derived_manifest": manifest,
        "clinical_context": clinical,
        "longitudinal_features": longitudinal_features,
        "landmark_assessment": {
            "frozen_layer6_information_landmark": (
                "baseline plus T1/early MRI"
            ),
            "heldout_prediction_target": (
                "T3/final tumor volume"
            ),
            "candidate_new_observation": (
                "T2/inter-regimen positive tumor-volume measurement"
            ),
            "layer6_eligible_explicit_raw_t2_case_count": (
                len(explicit_t2_ids)
            ),
            "usable_post_early_pre_final_volume_case_count": (
                len(usable_ids)
            ),
            "usable_post_early_pre_final_volume_case_sample": (
                sorted(usable_ids)[:20]
            ),
            "usable_new_observation_status": usable_status,
            "decision": decision,
            "conclusion": conclusion,
        },
        "requirements": {
            "frozen_layer6_model_modified": False,
            "frozen_layer7_model_modified": False,
            "full_updater_built": False,
            "outcome_based_model_selection_performed": False,
            "raw_imaging_evidence_separated_from_volume_evidence": True,
            "t2_must_be_strictly_after_early_and_before_final": True,
        },
        "limitations": [
            (
                "A raw T2 label establishes candidate imaging availability, "
                "not segmentation quality or a usable tumor-volume value."
            ),
            (
                "Weak context fields do not treat a bare T2 token as a "
                "landmark because T2 may describe an MRI sequence."
            ),
            (
                "This audit does not estimate predictive improvement."
            ),
        ],
    }


def _atomic_write_text(
    path: Path,
    text: str,
) -> None:
    if not text:
        raise ValueError("refusing to write empty output")

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        temporary_path = Path(temporary_name)
        if temporary_path.stat().st_size <= 0:
            raise ValueError(
                f"temporary output is empty: {temporary_path}"
            )

        os.replace(temporary_path, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_outputs(
    result: Mapping[str, object],
    *,
    output_json: Path,
    output_md: Path,
) -> None:
    json_text = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    json.loads(json_text)
    _atomic_write_text(output_json, json_text)

    assessment = result["landmark_assessment"]
    cohort = result["cohort"]
    manifest = result["derived_manifest"]
    features = result["longitudinal_features"]

    lines = [
        "# Layer 8 temporal landmark audit",
        "",
        "**Status: temporal information audit complete.**",
        "",
        "This audit does not train or modify Layer 6 or Layer 7.",
        "",
        "## Prediction landmarks",
        "",
        (
            "- Frozen Layer 6 information landmark: "
            f"{assessment['frozen_layer6_information_landmark']}"
        ),
        (
            "- Held-out target: "
            f"{assessment['heldout_prediction_target']}"
        ),
        (
            "- Candidate new observation: "
            f"{assessment['candidate_new_observation']}"
        ),
        "",
        "## Evidence",
        "",
        "| Quantity | Count |",
        "| --- | ---: |",
        (
            "| Layer 6-eligible cohort cases | "
            f"{cohort['layer6_eligible_case_count']} |"
        ),
        (
            "| Cohort cases already containing a strict intermediate "
            f"positive-volume point | "
            f"{cohort['strict_post_early_pre_final_volume_case_count']} |"
        ),
        (
            "| Layer 6 cases with explicit raw T2 evidence | "
            f"{manifest['layer6_eligible_explicit_t2_case_count']} |"
        ),
        (
            "| Processed-feature cases with a usable strict intermediate "
            f"positive-volume point | "
            f"{features['usable_post_early_pre_final_volume_case_count']} |"
        ),
        (
            "| Total usable strict intermediate cases | "
            f"{assessment['usable_post_early_pre_final_volume_case_count']} |"
        ),
        "",
        "## Decision",
        "",
        f"- Status: `{assessment['usable_new_observation_status']}`",
        f"- Next checkpoint: `{assessment['decision']}`",
        f"- Interpretation: {assessment['conclusion']}",
        "",
        "## Guardrails",
        "",
        "- Raw T2 labels are not treated as usable volume measurements.",
        "- Bare T2 text in weak series/path fields is not accepted as a landmark.",
        "- No predictive model was trained or selected.",
        "- Frozen Layer 6 and Layer 7 artifacts were not modified.",
    ]

    _atomic_write_text(
        output_md,
        "\n".join(lines) + "\n",
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--cohort",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--clinical",
        type=Path,
    )
    parser.add_argument(
        "--longitudinal-features",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-md",
        required=True,
        type=Path,
    )
    args = parser.parse_args(argv)

    result = run_audit(
        cohort_path=args.cohort,
        manifest_path=args.manifest,
        clinical_path=args.clinical,
        longitudinal_features_path=(
            args.longitudinal_features
        ),
    )
    write_outputs(
        result,
        output_json=args.output_json,
        output_md=args.output_md,
    )

    assessment = result["landmark_assessment"]
    print(
        "LAYER8_TEMPORAL_AUDIT_COMPLETE "
        f"status={assessment['usable_new_observation_status']} "
        f"decision={assessment['decision']} "
        "raw_t2_cases="
        f"{assessment['layer6_eligible_explicit_raw_t2_case_count']} "
        "usable_cases="
        f"{assessment['usable_post_early_pre_final_volume_case_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
