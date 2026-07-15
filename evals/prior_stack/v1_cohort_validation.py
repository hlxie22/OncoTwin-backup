"""Validation helpers for V1-D1 real cohort artifacts.

The cohort builder writes normalized JSONL plus summary/exclusion sidecars.
This module gives users a fast post-build check before running expensive prior
evaluations: it verifies the eval loader can read the cohort, source provenance
is present, case IDs are unique, and optionally that V1-A in-scope cases and
builder sidecars exist.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.prior_builder.parameter_contract import (
    TNBC_CHEMO_CONTRACT_ID,
    resolve_parameter_contract,
)

from .v1_real_data_eval import load_real_cohort


SUMMARY_SUFFIX = ".summary.json"
EXCLUSIONS_SUFFIX = ".exclusions.jsonl"
_UNKNOWN_TOKENS = {
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
}


@dataclass(frozen=True)
class CohortValidationResult:
    """Machine-readable V1-D1 cohort validation summary."""

    cohort_path: Path
    case_count: int
    in_scope_case_count: int
    out_of_scope_case_count: int
    data_origin_counts: Mapping[str, int]
    summary_path: Path | None
    exclusions_path: Path | None
    cohort_summary: Mapping[str, object]
    exclusion_report: Mapping[str, object]
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "cohort_path": str(self.cohort_path),
            "case_count": self.case_count,
            "in_scope_case_count": self.in_scope_case_count,
            "out_of_scope_case_count": self.out_of_scope_case_count,
            "data_origin_counts": dict(self.data_origin_counts),
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "exclusions_path": str(self.exclusions_path) if self.exclusions_path else None,
            "cohort_summary": dict(self.cohort_summary),
            "exclusion_report": dict(self.exclusion_report),
            "warnings": list(self.warnings),
        }


def validate_v1_prior_eval_cohort(
    cohort_path: Path,
    *,
    allow_demo_data: bool = False,
    require_in_scope: bool = False,
    min_in_scope_cases: int = 1,
    require_sidecars: bool = False,
    cohort_summary_path: Path | None = None,
    exclusions_path: Path | None = None,
) -> CohortValidationResult:
    """Validate a normalized V1-D1 cohort before running prior-stack evals."""

    if min_in_scope_cases < 1:
        raise ValueError("min_in_scope_cases must be at least 1")

    cases = load_real_cohort(cohort_path, allow_demo_data=allow_demo_data)
    duplicate_case_ids = _duplicate_case_ids(cases)
    if duplicate_case_ids:
        raise ValueError(
            "cohort contains duplicate case_id values: "
            + ", ".join(duplicate_case_ids)
        )

    missing_origin = [
        str(case["case_id"])
        for case in cases
        if not _present(case["context"].get("data_origin"))
    ]
    if missing_origin:
        raise ValueError(
            "normalized V1-D1 cohort rows require data_origin; missing for case_id: "
            + ", ".join(missing_origin)
        )

    in_scope_case_ids = [
        str(case["case_id"])
        for case in cases
        if resolve_parameter_contract(case["context"]).contract_id
        == TNBC_CHEMO_CONTRACT_ID
    ]
    if require_in_scope and len(in_scope_case_ids) < min_in_scope_cases:
        raise ValueError(
            "cohort has "
            f"{len(in_scope_case_ids)} V1-A in-scope cases; expected at least "
            f"{min_in_scope_cases}"
        )

    inferred_summary, inferred_exclusions = _infer_sidecars(cohort_path)
    resolved_summary = cohort_summary_path or inferred_summary
    resolved_exclusions = exclusions_path or inferred_exclusions
    cohort_summary = _read_summary(resolved_summary, required=require_sidecars)
    exclusion_report = _read_exclusions(resolved_exclusions, required=require_sidecars)

    warnings = _summary_warnings(
        case_count=len(cases),
        in_scope_case_count=len(in_scope_case_ids),
        cohort_summary=cohort_summary,
    )

    return CohortValidationResult(
        cohort_path=cohort_path,
        case_count=len(cases),
        in_scope_case_count=len(in_scope_case_ids),
        out_of_scope_case_count=len(cases) - len(in_scope_case_ids),
        data_origin_counts=_data_origin_counts(cases),
        summary_path=resolved_summary if resolved_summary.exists() else None,
        exclusions_path=resolved_exclusions if resolved_exclusions.exists() else None,
        cohort_summary=cohort_summary,
        exclusion_report=exclusion_report,
        warnings=tuple(warnings),
    )


def _duplicate_case_ids(cases: Sequence[Mapping[str, object]]) -> list[str]:
    counts = Counter(str(case["case_id"]) for case in cases)
    return sorted(case_id for case_id, count in counts.items() if count > 1)


def _data_origin_counts(cases: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(case["context"]["data_origin"]) for case in cases)
    return dict(sorted(counts.items()))


def _infer_sidecars(cohort_path: Path) -> tuple[Path, Path]:
    return (
        cohort_path.with_name(f"{cohort_path.stem}{SUMMARY_SUFFIX}"),
        cohort_path.with_name(f"{cohort_path.stem}{EXCLUSIONS_SUFFIX}"),
    )


def _read_summary(path: Path, *, required: bool) -> dict[str, object]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"cohort summary not found: {path}")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"cohort summary must be a JSON object: {path}")
    return dict(payload)


def _read_exclusions(path: Path, *, required: bool) -> dict[str, object]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"exclusions report not found: {path}")
        return {}

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    reasons: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, Mapping):
            reasons["malformed_exclusion_row"] += 1
            continue
        reason_value = row.get("excluded_reasons") or row.get("excluded_reason")
        if isinstance(reason_value, str):
            reasons[reason_value] += 1
        elif isinstance(reason_value, Sequence) and not isinstance(
            reason_value,
            (str, bytes),
        ):
            for reason in reason_value:
                reasons[str(reason)] += 1
        else:
            reasons["unknown"] += 1

    return {
        "excluded_rows": len(rows),
        "excluded_reason_counts": dict(sorted(reasons.items())),
    }


def _summary_warnings(
    *,
    case_count: int,
    in_scope_case_count: int,
    cohort_summary: Mapping[str, object],
) -> list[str]:
    warnings = []
    included_rows = cohort_summary.get("included_rows")
    if included_rows is not None and int(included_rows) != case_count:
        warnings.append(
            "Cohort summary included_rows does not match loadable cohort row count."
        )

    summary_in_scope = cohort_summary.get("v1a_in_scope_count")
    if summary_in_scope is not None and int(summary_in_scope) != in_scope_case_count:
        warnings.append(
            "Cohort summary v1a_in_scope_count does not match contract-resolved count."
        )

    if case_count < 20:
        warnings.append("Cohort has fewer than 20 cases; use only as a smoke check.")
    return warnings


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return " ".join(value.lower().replace("_", " ").split()) not in _UNKNOWN_TOKENS
    return True
