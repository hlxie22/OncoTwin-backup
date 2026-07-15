#!/usr/bin/env python3
"""Apply the validated V1 Layer 5 global growth-down calibrator.

This script materializes the first real-data Layer 5 residual input:

    ai_residual = {
        "validated": True,
        "model_version": "layer5_global_growth_down_calibrator_v0_1",
        "log_growth_rate_shift": -log(1.15),
    }

It does not inspect or modify held-out outcome fields. It only adds a validated
bounded residual signal consumed by experiments.prior_builder.ai_residual_policy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CALIBRATOR_VERSION = "layer5_global_growth_down_calibrator_v0_1"
DEFAULT_LOG_GROWTH_RATE_SHIFT = -math.log(1.15)
DEFAULT_DATA_ORIGIN = "ISPY2"
SUMMARY_SUFFIX = ".summary.json"
EXCLUSIONS_SUFFIX = ".exclusions.jsonl"
RESIDUAL_FIELD = "ai_residual"


def apply_layer5_global_calibrator(
    cohort_path: Path,
    output_path: Path,
    *,
    calibrator_version: str = DEFAULT_CALIBRATOR_VERSION,
    log_growth_rate_shift: float = DEFAULT_LOG_GROWTH_RATE_SHIFT,
    data_origin: str = DEFAULT_DATA_ORIGIN,
    overwrite_existing_residual: bool = False,
    write_sidecars: bool = True,
) -> dict[str, object]:
    """Write a JSONL cohort with a validated Layer 5 global calibrator signal."""

    _validate_shift(log_growth_rate_shift)
    rows = _read_jsonl(cohort_path)
    output_rows = [
        _calibrated_row(
            row,
            row_index=index,
            calibrator_version=calibrator_version,
            log_growth_rate_shift=log_growth_rate_shift,
            data_origin=data_origin,
            overwrite_existing_residual=overwrite_existing_residual,
        )
        for index, row in enumerate(rows, start=1)
    ]

    _write_jsonl(output_path, output_rows)
    if write_sidecars:
        _write_summary_sidecar(
            source_cohort=cohort_path,
            output_cohort=output_path,
            included_rows=len(output_rows),
            calibrator_version=calibrator_version,
            log_growth_rate_shift=log_growth_rate_shift,
            data_origin=data_origin,
            overwrite_existing_residual=overwrite_existing_residual,
        )
        _copy_or_create_exclusions_sidecar(cohort_path, output_path)

    return {
        "cohort_path": str(cohort_path),
        "output_path": str(output_path),
        "rows": len(output_rows),
        "calibrator_version": calibrator_version,
        "log_growth_rate_shift": log_growth_rate_shift,
        "data_origin": data_origin,
        "summary_path": str(_summary_path(output_path)) if write_sidecars else None,
        "exclusions_path": str(_exclusions_path(output_path)) if write_sidecars else None,
    }


def _calibrated_row(
    row: Mapping[str, object],
    *,
    row_index: int,
    calibrator_version: str,
    log_growth_rate_shift: float,
    data_origin: str,
    overwrite_existing_residual: bool,
) -> dict[str, object]:
    output = dict(row)
    existing = output.get(RESIDUAL_FIELD)
    if _present(existing) and not overwrite_existing_residual:
        case_id = output.get("case_id") or output.get("patient_id") or f"row {row_index}"
        raise ValueError(
            f"{case_id} already has {RESIDUAL_FIELD}; pass "
            "--overwrite-existing-residual to replace it"
        )

    output.setdefault("data_origin", data_origin)
    context = output.get("context")
    if isinstance(context, Mapping):
        nested_context = dict(context)
        nested_context.setdefault("data_origin", data_origin)
        output["context"] = nested_context

    output[RESIDUAL_FIELD] = {
        "validated": True,
        "model_version": calibrator_version,
        "log_growth_rate_shift": log_growth_rate_shift,
    }
    return output


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(dict(payload))
    if not rows:
        raise ValueError(f"cohort contains no rows: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_summary_sidecar(
    *,
    source_cohort: Path,
    output_cohort: Path,
    included_rows: int,
    calibrator_version: str,
    log_growth_rate_shift: float,
    data_origin: str,
    overwrite_existing_residual: bool,
) -> None:
    source_summary_path = _summary_path(source_cohort)
    summary: dict[str, object] = {}
    if source_summary_path.exists():
        payload = json.loads(source_summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"source summary must be a JSON object: {source_summary_path}")
        summary.update(payload)

    summary["included_rows"] = included_rows
    summary["source_cohort_path"] = str(source_cohort)
    summary["layer5_calibrator"] = {
        "calibrator_version": calibrator_version,
        "residual_field": RESIDUAL_FIELD,
        "validated": True,
        "policy": "global_growth_down",
        "log_growth_rate_shift": log_growth_rate_shift,
        "data_origin_default": data_origin,
        "overwrite_existing_residual": overwrite_existing_residual,
        "not_patient_specific": True,
        "uses_heldout_outcomes": False,
    }

    path = _summary_path(output_cohort)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_or_create_exclusions_sidecar(source_cohort: Path, output_cohort: Path) -> None:
    source = _exclusions_path(source_cohort)
    target = _exclusions_path(output_cohort)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, target)
    else:
        target.write_text("", encoding="utf-8")


def _summary_path(cohort_path: Path) -> Path:
    return cohort_path.with_name(f"{cohort_path.stem}{SUMMARY_SUFFIX}")


def _exclusions_path(cohort_path: Path) -> Path:
    return cohort_path.with_name(f"{cohort_path.stem}{EXCLUSIONS_SUFFIX}")


def _validate_shift(value: float) -> None:
    if not math.isfinite(value):
        raise ValueError("log_growth_rate_shift must be finite")
    max_abs_shift = math.log(1.15)
    if abs(value) > max_abs_shift:
        raise ValueError("log_growth_rate_shift exceeds Layer 5 bound")


def _present(value: object) -> bool:
    return value not in (None, "", {}, [])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the V1 Layer 5 global growth-down calibrator to a JSONL cohort."
    )
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--calibrator-version",
        default=DEFAULT_CALIBRATOR_VERSION,
        help="Validated residual model_version to write into ai_residual.",
    )
    parser.add_argument(
        "--log-growth-rate-shift",
        type=float,
        default=DEFAULT_LOG_GROWTH_RATE_SHIFT,
        help="Bounded transformed-space growth-rate shift. Default is -log(1.15).",
    )
    parser.add_argument(
        "--data-origin",
        default=DEFAULT_DATA_ORIGIN,
        help="Default data_origin to add when missing.",
    )
    parser.add_argument(
        "--overwrite-existing-residual",
        action="store_true",
        help="Replace existing top-level ai_residual values instead of failing.",
    )
    parser.add_argument(
        "--no-sidecars",
        action="store_true",
        help="Do not write summary/exclusions sidecars for the calibrated cohort.",
    )
    args = parser.parse_args()

    try:
        result = apply_layer5_global_calibrator(
            args.cohort,
            args.output,
            calibrator_version=args.calibrator_version,
            log_growth_rate_shift=args.log_growth_rate_shift,
            data_origin=args.data_origin,
            overwrite_existing_residual=args.overwrite_existing_residual,
            write_sidecars=not args.no_sidecars,
        )
    except Exception as exc:
        print(f"Layer 5 calibration failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
