"""Build normalized real-data cohorts for the V1 prior-stack evals.

The V1-D1 eval consumes a small patient-level JSONL table with baseline and
held-out final tumor volumes. This module keeps the curation rules explicit so
that generated cohorts are accompanied by stable exclusion reasons and a summary
sidecar before they are used for performance claims.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .parameter_contract import TNBC_CHEMO_CONTRACT_ID, resolve_parameter_contract


DEFAULT_OUTPUT_PATH = Path(
    "data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.jsonl"
)
SUMMARY_SUFFIX = ".summary.json"
EXCLUSIONS_SUFFIX = ".exclusions.jsonl"

EXCLUSION_REASONS = (
    "missing_case_id",
    "missing_baseline_volume",
    "missing_final_volume",
    "non_positive_baseline_volume",
    "non_positive_final_volume",
    "invalid_time_order",
    "unresolved_subtype",
    "out_of_v1a_scope",
    "synthetic_or_demo_data",
    "duplicate_case_id",
    "unsupported_treatment_context",
)

CASE_ID_FIELDS = ("case_id", "patient_id", "subject_id", "participant_id", "ptid")
BASELINE_DAY_FIELDS = ("baseline_day", "baseline_relative_day", "initial_day")
BASELINE_VOLUME_FIELDS = (
    "baseline_volume_ml",
    "baseline_tumor_volume_ml",
    "initial_volume_ml",
    "baseline_ftv_ml",
    "baseline_functional_tumor_volume_ml",
)
FINAL_DAY_FIELDS = ("final_day", "heldout_day", "outcome_day", "last_day")
FINAL_VOLUME_FIELDS = (
    "final_volume_ml",
    "heldout_volume_ml",
    "outcome_volume_ml",
    "final_tumor_volume_ml",
    "final_ftv_ml",
    "final_functional_tumor_volume_ml",
)
EARLY_DAY_FIELDS = ("early_day", "followup_day", "first_followup_day")
EARLY_VOLUME_FIELDS = (
    "early_volume_ml",
    "followup_volume_ml",
    "early_tumor_volume_ml",
    "early_ftv_ml",
)
DAY_FIELDS = (
    "day",
    "relative_day",
    "days_from_baseline",
    "study_day",
    "timepoint_day",
    "mri_day",
)
TIMEPOINT_FIELDS = (
    "timepoint",
    "time_point",
    "visit",
    "visit_label",
    "measurement_timepoint",
    "mri_timepoint",
)
LONG_VOLUME_FIELDS = (
    "tumor_volume_ml",
    "volume_ml",
    "functional_tumor_volume_ml",
    "ftv_ml",
    "ftv",
)
SUBTYPE_FIELDS = ("subtype", "disease_context", "cancer_subtype", "tumor_subtype")
TREATMENT_FIELDS = (
    "treatment_context",
    "treatment_regimen",
    "regimen_name",
    "schedule_type",
)
DATA_ORIGIN_FIELDS = ("data_origin", "source_dataset", "dataset", "data_source")
BIOMARKER_FIELDS = (
    "er_status",
    "pr_status",
    "her2_status",
    "hr_status",
    "grade",
    "ki67_percent",
    "ki67",
    "brca_status",
    "brca1_status",
    "brca2_status",
    "hrd_status",
)
MRI_FEATURE_FIELDS = (
    "volume_ml",
    "functional_tumor_volume_ml",
    "ftv_ml",
    "longest_diameter_cm",
    "enhancement_std",
    "segmentation_qc",
    "registration_qc",
)

FAKE_TOKENS = ("demo", "synthetic", "simulated", "toy", "fixture")
NEGATIVE_TOKENS = ("negative", "neg", "-", "0", "false", "no")
POSITIVE_TOKENS = ("positive", "pos", "+", "1", "true", "yes", "amplified")
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
NOMINAL_ISPY2_DAYS = {
    "t0": 0.0,
    "baseline": 0.0,
    "pre": 0.0,
    "pretreatment": 0.0,
    "pre treatment": 0.0,
    "t1": 21.0,
    "early": 21.0,
    "week3": 21.0,
    "week 3": 21.0,
    "t2": 84.0,
    "mid": 84.0,
    "interregimen": 84.0,
    "inter regimen": 84.0,
    "t3": 140.0,
    "final": 140.0,
    "presurgery": 140.0,
    "pre surgery": 140.0,
}


@dataclass(frozen=True)
class CohortBuildResult:
    """Paths and summary metadata produced by the V1-D1 cohort builder."""

    cohort_path: Path
    exclusions_path: Path
    summary_path: Path
    summary: Mapping[str, object]


@dataclass(frozen=True)
class ColumnInspection:
    """Column-level profile for a source table."""

    path: Path
    row_count: int
    columns: tuple[str, ...]
    non_empty_counts: Mapping[str, int]
    suspected_roles: Mapping[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "row_count": self.row_count,
            "columns": list(self.columns),
            "non_empty_counts": dict(self.non_empty_counts),
            "suspected_roles": {
                role: list(columns) for role, columns in self.suspected_roles.items()
            },
        }


def build_v1_prior_eval_cohort(
    measurements_path: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    clinical_path: Path | None = None,
    default_treatment_context: str | None = None,
    data_origin: str | None = None,
    use_nominal_ispy2_days: bool = False,
    allow_demo_data: bool = False,
) -> CohortBuildResult:
    """Normalize a wide or long longitudinal table into the V1-D1 JSONL cohort.

    Rows that cannot support the real-data prior-layer eval are excluded rather
    than coerced. Every exclusion receives one or more stable reason labels in a
    sidecar JSONL file so suite reports can distinguish data readiness from model
    runtime failures.
    """

    measurement_rows = _read_rows(measurements_path)
    clinical_rows = _read_rows(clinical_path) if clinical_path is not None else []
    clinical_by_case = _clinical_by_case(clinical_rows)

    candidates = _candidate_cases(
        measurement_rows,
        use_nominal_ispy2_days=use_nominal_ispy2_days,
    )
    if clinical_by_case:
        candidates = [
            _merge_context(candidate, clinical_by_case.get(candidate.case_id, {}))
            for candidate in candidates
        ]
    if default_treatment_context:
        candidates = [
            _with_default_treatment(candidate, default_treatment_context)
            for candidate in candidates
        ]
    resolved_data_origin = data_origin or _infer_data_origin(measurements_path)
    candidates = [
        _with_default_data_origin(candidate, resolved_data_origin)
        for candidate in candidates
    ]

    included: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    subtype_counts = Counter()

    for candidate in candidates:
        candidate = _with_resolved_subtype(candidate)
        subtype_state = _tnbc_state(candidate.context)
        if subtype_state == "tnbc":
            subtype_counts["tnbc_count"] += 1
        elif subtype_state == "non_tnbc":
            subtype_counts["non_tnbc_count"] += 1

        reasons = list(candidate.initial_exclusion_reasons)
        if not candidate.case_id:
            reasons.append("missing_case_id")
        elif candidate.case_id in seen_case_ids:
            reasons.append("duplicate_case_id")
        if not allow_demo_data and _looks_fake(measurements_path, candidate.context):
            reasons.append("synthetic_or_demo_data")

        reasons.extend(_measurement_exclusion_reasons(candidate))
        if not reasons:
            if subtype_state == "unknown":
                reasons.append("unresolved_subtype")
            elif subtype_state == "non_tnbc":
                reasons.append("out_of_v1a_scope")
            elif _treatment_state(candidate.context) != "chemo":
                reasons.append("unsupported_treatment_context")
            elif resolve_parameter_contract(candidate.context).contract_id != TNBC_CHEMO_CONTRACT_ID:
                reasons.append("out_of_v1a_scope")

        if reasons:
            exclusions.append(candidate.exclusion_record(_stable_unique(reasons)))
            if candidate.case_id:
                seen_case_ids.add(candidate.case_id)
            continue

        row = candidate.output_row()
        included.append(row)
        seen_case_ids.add(candidate.case_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path = output_path.with_name(f"{output_path.stem}{EXCLUSIONS_SUFFIX}")
    summary_path = output_path.with_name(f"{output_path.stem}{SUMMARY_SUFFIX}")
    _write_jsonl(output_path, included)
    _write_jsonl(exclusions_path, exclusions)

    summary = _summary(
        input_row_count=len(measurement_rows),
        included=included,
        exclusions=exclusions,
        subtype_counts=subtype_counts,
        source_files=[measurements_path] + ([clinical_path] if clinical_path else []),
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CohortBuildResult(output_path, exclusions_path, summary_path, summary)


def inspect_v1_data_columns(path: Path) -> ColumnInspection:
    """Return a lightweight column profile for planning a V1-D1 cohort build."""

    rows = _read_rows(path)
    columns = tuple(sorted({key for row in rows for key in row}))
    non_empty = {
        column: sum(_present(row.get(column)) for row in rows)
        for column in columns
    }
    roles = {
        "case_id": _matching_columns(columns, CASE_ID_FIELDS),
        "wide_baseline_volume": _matching_columns(columns, BASELINE_VOLUME_FIELDS),
        "wide_final_volume": _matching_columns(columns, FINAL_VOLUME_FIELDS),
        "wide_early_volume": _matching_columns(columns, EARLY_VOLUME_FIELDS),
        "long_day": _matching_columns(columns, DAY_FIELDS),
        "long_timepoint": _matching_columns(columns, TIMEPOINT_FIELDS),
        "long_volume": _matching_columns(columns, LONG_VOLUME_FIELDS),
        "subtype": _matching_columns(columns, SUBTYPE_FIELDS),
        "treatment": _matching_columns(columns, TREATMENT_FIELDS),
        "biomarkers": _matching_columns(columns, BIOMARKER_FIELDS),
        "mri_features": _matching_columns(columns, MRI_FEATURE_FIELDS),
    }
    return ColumnInspection(
        path=path,
        row_count=len(rows),
        columns=columns,
        non_empty_counts=non_empty,
        suspected_roles={role: cols for role, cols in roles.items() if cols},
    )


@dataclass(frozen=True)
class _CandidateCase:
    case_id: str
    context: Mapping[str, object]
    baseline_day: float | None
    baseline_volume_ml: float | None
    final_day: float | None
    final_volume_ml: float | None
    early_day: float | None = None
    early_volume_ml: float | None = None
    initial_exclusion_reasons: tuple[str, ...] = ()
    source_row_numbers: tuple[int, ...] = ()

    def output_row(self) -> dict[str, object]:
        row = _json_ready_mapping(self.context)
        row.update(
            {
                "case_id": self.case_id,
                "baseline_day": self.baseline_day,
                "baseline_volume_ml": self.baseline_volume_ml,
                "final_day": self.final_day,
                "final_volume_ml": self.final_volume_ml,
            }
        )
        if self.early_day is not None and self.early_volume_ml is not None:
            row["early_day"] = self.early_day
            row["early_volume_ml"] = self.early_volume_ml
        row.setdefault("volume_ml", self.baseline_volume_ml)
        return row

    def exclusion_record(self, reasons: Sequence[str]) -> dict[str, object]:
        return {
            "case_id": self.case_id or None,
            "excluded_reason": reasons[0],
            "excluded_reasons": list(reasons),
            "source_row_numbers": list(self.source_row_numbers),
        }


def _candidate_cases(
    rows: Sequence[Mapping[str, object]],
    *,
    use_nominal_ispy2_days: bool,
) -> list[_CandidateCase]:
    if not rows:
        return []
    if _looks_wide(rows):
        return [_wide_candidate(row, index) for index, row in enumerate(rows, start=1)]
    return _long_candidates(rows, use_nominal_ispy2_days=use_nominal_ispy2_days)


def _looks_wide(rows: Sequence[Mapping[str, object]]) -> bool:
    for row in rows:
        keys = set(row)
        if keys & set(BASELINE_VOLUME_FIELDS) and keys & set(FINAL_VOLUME_FIELDS):
            return True
    return False


def _wide_candidate(row: Mapping[str, object], row_number: int) -> _CandidateCase:
    context = _flatten_context(row)
    baseline_volume = _optional_number_any(row, BASELINE_VOLUME_FIELDS)
    final_volume = _optional_number_any(row, FINAL_VOLUME_FIELDS)
    baseline_day = _optional_number_any(row, BASELINE_DAY_FIELDS)
    final_day = _optional_number_any(row, FINAL_DAY_FIELDS)
    return _CandidateCase(
        case_id=_case_id(row),
        context=context,
        baseline_day=0.0 if baseline_day is None else baseline_day,
        baseline_volume_ml=baseline_volume,
        final_day=final_day,
        final_volume_ml=final_volume,
        early_day=_optional_number_any(row, EARLY_DAY_FIELDS),
        early_volume_ml=_optional_number_any(row, EARLY_VOLUME_FIELDS),
        source_row_numbers=(row_number,),
    )


def _long_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    use_nominal_ispy2_days: bool,
) -> list[_CandidateCase]:
    grouped: dict[str, list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    missing_case_rows: list[_CandidateCase] = []
    for row_number, row in enumerate(rows, start=1):
        case_id = _case_id(row)
        if not case_id:
            missing_case_rows.append(
                _CandidateCase(
                    case_id="",
                    context=_flatten_context(row),
                    baseline_day=None,
                    baseline_volume_ml=None,
                    final_day=None,
                    final_volume_ml=None,
                    initial_exclusion_reasons=("missing_case_id",),
                    source_row_numbers=(row_number,),
                )
            )
            continue
        grouped[case_id].append((row_number, row))

    candidates = list(missing_case_rows)
    for case_id, case_rows in grouped.items():
        candidates.append(
            _long_candidate(
                case_id,
                case_rows,
                use_nominal_ispy2_days=use_nominal_ispy2_days,
            )
        )
    return candidates


def _long_candidate(
    case_id: str,
    case_rows: Sequence[tuple[int, Mapping[str, object]]],
    *,
    use_nominal_ispy2_days: bool,
) -> _CandidateCase:
    context: dict[str, object] = {}
    points: list[tuple[float, float]] = []
    reasons: list[str] = []
    source_row_numbers = []

    for row_number, row in case_rows:
        source_row_numbers.append(row_number)
        context.update(_non_measurement_context(row))
        day = _day(row, use_nominal_ispy2_days=use_nominal_ispy2_days)
        volume = _optional_number_any(row, LONG_VOLUME_FIELDS)
        if day is None or volume is None:
            reasons.append("missing_baseline_volume" if not points else "missing_final_volume")
            continue
        points.append((day, volume))

    points = sorted(points)
    baseline_day = baseline_volume = final_day = final_volume = None
    early_day = early_volume = None
    if len(points) >= 2:
        baseline_day, baseline_volume = points[0]
        final_day, final_volume = points[-1]
        if len(points) > 2:
            early_day, early_volume = points[1]
    else:
        reasons.append("missing_final_volume")
        if len(points) == 0:
            reasons.append("missing_baseline_volume")

    return _CandidateCase(
        case_id=case_id,
        context=context,
        baseline_day=baseline_day,
        baseline_volume_ml=baseline_volume,
        final_day=final_day,
        final_volume_ml=final_volume,
        early_day=early_day,
        early_volume_ml=early_volume,
        initial_exclusion_reasons=tuple(_stable_unique(reasons)),
        source_row_numbers=tuple(source_row_numbers),
    )


def _measurement_exclusion_reasons(candidate: _CandidateCase) -> list[str]:
    reasons: list[str] = []
    if candidate.baseline_volume_ml is None:
        reasons.append("missing_baseline_volume")
    elif candidate.baseline_volume_ml <= 0:
        reasons.append("non_positive_baseline_volume")
    if candidate.final_volume_ml is None:
        reasons.append("missing_final_volume")
    elif candidate.final_volume_ml <= 0:
        reasons.append("non_positive_final_volume")
    if (
        candidate.baseline_day is None
        or candidate.final_day is None
        or candidate.final_day <= candidate.baseline_day
    ):
        reasons.append("invalid_time_order")
    if (candidate.early_day is None) != (candidate.early_volume_ml is None):
        reasons.append("invalid_time_order")
    if (
        candidate.early_day is not None
        and candidate.early_volume_ml is not None
        and (
            candidate.early_volume_ml <= 0
            or candidate.baseline_day is None
            or candidate.final_day is None
            or not (candidate.baseline_day < candidate.early_day < candidate.final_day)
        )
    ):
        reasons.append("invalid_time_order")
    return reasons


def _summary(
    *,
    input_row_count: int,
    included: Sequence[Mapping[str, object]],
    exclusions: Sequence[Mapping[str, object]],
    subtype_counts: Counter[str],
    source_files: Sequence[Path | None],
) -> dict[str, object]:
    excluded_reason_counts = Counter()
    for exclusion in exclusions:
        for reason in exclusion.get("excluded_reasons", []):
            excluded_reason_counts[str(reason)] += 1

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_input_rows": input_row_count,
        "included_rows": len(included),
        "excluded_rows": len(exclusions),
        "tnbc_count": subtype_counts.get("tnbc_count", 0),
        "non_tnbc_count": subtype_counts.get("non_tnbc_count", 0),
        "v1a_in_scope_count": len(included),
        "baseline_volume_available_count": sum(
            _present(row.get("baseline_volume_ml")) for row in included
        ),
        "final_volume_available_count": sum(
            _present(row.get("final_volume_ml")) for row in included
        ),
        "early_followup_available_count": sum(
            _present(row.get("early_volume_ml")) for row in included
        ),
        "biomarker_completeness": _completeness(included, BIOMARKER_FIELDS),
        "mri_feature_completeness": _completeness(included, MRI_FEATURE_FIELDS),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "source_files": [str(path) for path in source_files if path is not None],
    }


def _read_rows(path: Path | None) -> list[Mapping[str, object]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        return [
            _require_mapping(json.loads(line), path)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [_require_mapping(row, path) for row in payload]
        if isinstance(payload, Mapping):
            for key in ("cases", "rows", "measurements", "data"):
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [_require_mapping(row, path) for row in rows]
            return [_require_mapping(payload, path)]
        raise ValueError(f"JSON source must contain an object or list: {path}")
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"source table must be .csv, .jsonl, or .json: {path}")


def _require_mapping(value: object, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"source rows must be JSON objects: {path}")
    return value


def _clinical_by_case(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output = {}
    for row in rows:
        case_id = _case_id(row)
        if case_id and case_id not in output:
            output[case_id] = _flatten_context(row)
    return output


def _merge_context(candidate: _CandidateCase, extra: Mapping[str, object]) -> _CandidateCase:
    if not extra:
        return candidate
    context = dict(candidate.context)
    for key, value in extra.items():
        if key not in context or not _present(context[key]):
            context[key] = value
    return _replace_candidate_context(candidate, context)


def _with_default_treatment(candidate: _CandidateCase, treatment: str) -> _CandidateCase:
    context = dict(candidate.context)
    if not any(_present(context.get(field)) for field in TREATMENT_FIELDS):
        context["treatment_context"] = treatment
    return _replace_candidate_context(candidate, context)


def _with_default_data_origin(
    candidate: _CandidateCase, data_origin: str
) -> _CandidateCase:
    context = dict(candidate.context)
    if not _present(context.get("data_origin")):
        context["data_origin"] = _resolved_data_origin_value(context, data_origin)
    return _replace_candidate_context(candidate, context)


def _resolved_data_origin_value(
    context: Mapping[str, object],
    fallback: str,
) -> object:
    for field in DATA_ORIGIN_FIELDS:
        if field == "data_origin":
            continue
        value = context.get(field)
        if _present(value):
            return value
    return fallback


def _with_resolved_subtype(candidate: _CandidateCase) -> _CandidateCase:
    if any(_present(candidate.context.get(field)) for field in SUBTYPE_FIELDS):
        return candidate
    if _tnbc_state(candidate.context) != "tnbc":
        return candidate
    context = dict(candidate.context)
    context["subtype"] = "TNBC"
    return _replace_candidate_context(candidate, context)


def _replace_candidate_context(
    candidate: _CandidateCase, context: Mapping[str, object]
) -> _CandidateCase:
    return _CandidateCase(
        case_id=candidate.case_id,
        context=context,
        baseline_day=candidate.baseline_day,
        baseline_volume_ml=candidate.baseline_volume_ml,
        final_day=candidate.final_day,
        final_volume_ml=candidate.final_volume_ml,
        early_day=candidate.early_day,
        early_volume_ml=candidate.early_volume_ml,
        initial_exclusion_reasons=candidate.initial_exclusion_reasons,
        source_row_numbers=candidate.source_row_numbers,
    )


def _flatten_context(row: Mapping[str, object]) -> dict[str, object]:
    context = dict(row)
    for nested in ("pathology", "mri_features", "biomarkers", "clinical"):
        nested_value = row.get(nested)
        if isinstance(nested_value, Mapping):
            context.update(nested_value)
    return context


def _non_measurement_context(row: Mapping[str, object]) -> dict[str, object]:
    excluded = set(DAY_FIELDS + TIMEPOINT_FIELDS + LONG_VOLUME_FIELDS)
    return {key: value for key, value in _flatten_context(row).items() if key not in excluded}


def _case_id(row: Mapping[str, object]) -> str:
    for field in CASE_ID_FIELDS:
        if _present(row.get(field)):
            return str(row[field]).strip()
    return ""


def _infer_data_origin(path: Path) -> str:
    normalized_parts = [path.stem.lower()] + [part.lower() for part in path.parts]
    if any("ispy2" in part or "i-spy2" in part for part in normalized_parts):
        return "ISPY2"
    if any("breastdcedl" in part for part in normalized_parts):
        return "BreastDCEDL_ISPY2"
    return path.stem


def _day(
    row: Mapping[str, object],
    *,
    use_nominal_ispy2_days: bool,
) -> float | None:
    numeric = _optional_number_any(row, DAY_FIELDS)
    if numeric is not None:
        return numeric
    if not use_nominal_ispy2_days:
        return None
    for field in TIMEPOINT_FIELDS:
        value = row.get(field)
        if not _present(value):
            continue
        normalized = _normalize(value)
        if normalized in NOMINAL_ISPY2_DAYS:
            return NOMINAL_ISPY2_DAYS[normalized]
    return None


def _optional_number_any(
    row: Mapping[str, object], fields: Sequence[str]
) -> float | None:
    for field in fields:
        if field in row and _present(row[field]):
            return _number(row[field])
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _tnbc_state(context: Mapping[str, object]) -> str:
    subtype_text = " ".join(str(context.get(field, "")) for field in SUBTYPE_FIELDS)
    normalized = _normalize(subtype_text)
    if "tnbc" in normalized or "triple negative" in normalized:
        return "tnbc"
    if any(token in normalized for token in ("her2 positive", "hr positive", "luminal", "er positive")):
        return "non_tnbc"

    er = _status(context.get("er_status") or context.get("estrogen_receptor_status"))
    pr = _status(context.get("pr_status") or context.get("progesterone_receptor_status"))
    her2 = _status(context.get("her2_status") or context.get("erbb2_status"))
    statuses = (er, pr, her2)
    if all(status == "negative" for status in statuses):
        return "tnbc"
    if any(status == "positive" for status in statuses):
        return "non_tnbc"
    return "unknown"


def _treatment_state(context: Mapping[str, object]) -> str:
    text = _normalize(" ".join(str(context.get(field, "")) for field in TREATMENT_FIELDS))
    if not text or text in UNKNOWN_TOKENS:
        return "unknown"
    chemo_tokens = (
        "a/c-t",
        "ac-t",
        "ac t",
        "anthracycline",
        "adriamycin",
        "cyclophosphamide",
        "taxane",
        "taxol",
        "paclitaxel",
        "chemo",
        "chemotherapy",
    )
    return "chemo" if any(token in text for token in chemo_tokens) else "unsupported"


def _status(value: object) -> str:
    if not _present(value):
        return "unknown"
    text = _normalize(value)
    if text in UNKNOWN_TOKENS:
        return "unknown"
    if any(token == text or token in text for token in POSITIVE_TOKENS):
        return "positive"
    if any(token == text or token in text for token in NEGATIVE_TOKENS):
        return "negative"
    return "unknown"


def _looks_fake(path: Path, row: Mapping[str, object]) -> bool:
    haystack = str(path).lower() + json.dumps(row, default=str).lower()
    return any(token in haystack for token in FAKE_TOKENS)


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return _normalize(value) not in UNKNOWN_TOKENS
    return True


def _normalize(value: object) -> str:
    return " ".join(str(value).lower().replace("_", " ").split())


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    output = []
    for value in values:
        if value in EXCLUSION_REASONS and value not in seen:
            output.append(value)
            seen.add(value)
    return tuple(output)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_ready_mapping(row: Mapping[str, object]) -> dict[str, object]:
    return {key: _json_ready(value) for key, value in row.items()}


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _completeness(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> dict[str, int]:
    return {
        field: sum(_present(row.get(field)) for row in rows)
        for field in fields
        if any(field in row for row in rows)
    }


def _matching_columns(columns: Sequence[str], names: Sequence[str]) -> tuple[str, ...]:
    normalized_to_original = {_normalize(column): column for column in columns}
    matches = []
    for name in names:
        normalized = _normalize(name)
        if normalized in normalized_to_original:
            matches.append(normalized_to_original[normalized])
    return tuple(matches)
