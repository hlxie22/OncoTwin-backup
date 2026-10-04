#!/usr/bin/env python3
"""Audit executable Layer 6 readiness and freeze a redacted holdout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ANALYSIS_VERSION = (
    "oncotwin_layer8_executable_holdout_readiness_v0_1"
)

PROTOCOL_VERSION = (
    "oncotwin_layer8_same_source_holdout_protocol_v0_1"
)

MIN_VALIDATION_CASES = 100
ALIGNMENT_TOLERANCE = 1e-6
FEATURE_TOLERANCE = 1e-11
REDACTED_FINAL_VOLUME_ML = 1.0

EXPECTED_HASHES = {
    "current_cohort": (
        "8cd51c7f00c073a3d7894692a25bcd8c04384c201c8c3989e91cf5b832c664c0"
    ),
    "full_cohort": (
        "ef7533e1e3af12dc54da284f7df4bce13cd9dd7e895efe291f69aca575ca5fda"
    ),
    "derived_features": (
        "3a6ee0641485d0424925d7482a6930f1d070f1ad90c90e7c9aee4cacbdc5319a"
    ),
    "locked_evaluation": (
        "c910c99eb8b129794faa0db7215cebf669ad922bf6dfe5dd2df09e70146eec48"
    ),
    "strict_nested": (
        "46b6417fd849f7c88bcacc7349f8b4fe8f302b6af85ce5dc6a62d79bdff7a5ba"
    ),
    "ablation": (
        "1331d9b5b30e8950690bfaa270bd63efc9c3d4635a91b256224471415a56bb26"
    ),
    "trainer": (
        "a25016f347a92b3a1bd8280bf80fca2cb65befe5c4877857d6a16a67d8f269b0"
    ),
}

CASE_PATTERN = re.compile(
    r"ISPY2[-_ ]?(\d+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--current-cohort",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--full-cohort",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--derived-features",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--locked-evaluation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--strict-nested",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--audit-markdown",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--protocol-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--temporary-directory",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_hash(
    value: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def load_module(
    name: str,
    path: Path,
) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Unable to import module: {path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )
    sys.modules[name] = module
    specification.loader.exec_module(module)

    return module


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, Mapping):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return dict(value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            value = json.loads(line)

            if not isinstance(value, Mapping):
                raise TypeError(
                    f"{path}:{line_number} is not a JSON object"
                )

            row = dict(value)
            row["_source_line"] = line_number
            rows.append(row)

    return rows


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    text = "".join(
        json.dumps(
            dict(row),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )

    if not text:
        raise RuntimeError(
            f"Refusing to write empty JSONL: {path}"
        )

    atomic_write_text(path, text)


def normalize_case_id(
    value: object,
) -> str | None:
    text = str(value or "").strip().upper()

    if not text:
        return None

    match = CASE_PATTERN.search(text)

    if match:
        return f"ISPY2-{match.group(1)}"

    return text


def case_id_from_row(
    row: Mapping[str, Any],
) -> str | None:
    return normalize_case_id(
        row.get("case_id")
        or row.get("patient_id")
        or row.get("subject_id")
    )


def index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for row in rows:
        case_id = case_id_from_row(row)

        if case_id is None:
            raise RuntimeError(
                f"{name} row is missing case ID"
            )

        if case_id in indexed:
            raise RuntimeError(
                f"{name} contains duplicate case ID: "
                f"{case_id}"
            )

        indexed[case_id] = dict(row)

    return indexed


def finite_float(
    value: object,
) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def positive_float(
    value: object,
) -> float | None:
    number = finite_float(value)

    if number is None or number <= 0:
        return None

    return number


def relative_error(
    left: float,
    right: float,
) -> float:
    return abs(left - right) / max(
        abs(right),
        1e-12,
    )


def normalize_landmark(
    row: Mapping[str, Any],
) -> str | None:
    for field in (
        "timepoint_label_if_available",
        "timepoint_label",
        "timepoint",
        "visit",
        "phase",
    ):
        text = re.sub(
            r"[^A-Z0-9]",
            "",
            str(row.get(field) or "").upper(),
        )

        for label in (
            "T0",
            "T1",
            "T2",
            "T3",
        ):
            if (
                text == label
                or text.startswith(label)
            ):
                return label

    raw_index = row.get("timepoint_index")

    try:
        index = int(
            float(str(raw_index).strip())
        )
    except (TypeError, ValueError):
        return None

    if 0 <= index <= 3:
        return f"T{index}"

    return None


def parse_actual_date(
    row: Mapping[str, Any],
) -> date | None:
    candidates = []

    for key, value in row.items():
        normalized = re.sub(
            r"[^a-z0-9]",
            "",
            str(key).lower(),
        )

        if (
            "date" in normalized
            and "relative" not in normalized
        ):
            candidates.append(value)

    for raw in candidates:
        text = str(raw or "").strip()

        if not text:
            continue

        digits = re.sub(
            r"[^0-9]",
            "",
            text,
        )

        if len(digits) == 8:
            try:
                parsed = datetime.strptime(
                    digits,
                    "%Y%m%d",
                ).date()

                if 1990 <= parsed.year <= 2035:
                    return parsed
            except ValueError:
                pass

        for format_string in (
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%Y/%m/%d",
        ):
            try:
                parsed = datetime.strptime(
                    text[:10],
                    format_string,
                ).date()

                if 1990 <= parsed.year <= 2035:
                    return parsed
            except ValueError:
                continue

    return None


def write_redacted_cohort(
    source_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    output_rows = []

    for source in source_rows:
        row = {
            key: value
            for key, value in source.items()
            if not str(key).startswith("_")
        }

        row["final_volume_ml"] = (
            REDACTED_FINAL_VOLUME_ML
        )

        for alias in (
            "heldout_volume_ml",
            "outcome_volume_ml",
        ):
            if alias in row:
                row[alias] = (
                    REDACTED_FINAL_VOLUME_ML
                )

        output_rows.append(row)

    atomic_write_jsonl(
        path,
        output_rows,
    )


def run_prior_pipeline(
    *,
    trainer: ModuleType,
    cohort_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    locked = trainer.LOCKED
    evalmod = trainer.evalmod

    evaluation = evalmod.run_real_data_eval(
        cohort_path,
        n_samples=2000,
        seed=locked.BASE_SEED,
        interval_calibrator_path=(
            locked.STATIC_CALIBRATOR
        ),
        allow_candidate_interval_calibrator=True,
        early_response_updater_path=(
            locked.EARLY_UPDATER
        ),
        early_response_interval_calibrator_path=(
            locked.EARLY_CALIBRATOR
        ),
        allow_candidate_early_response_updater=True,
    )

    cases = evalmod.load_real_cohort(
        cohort_path,
        allow_demo_data=False,
    )

    evaluated = {
        str(row["case_id"]): dict(row)
        for row in evaluation["case_predictions"]
    }

    return cases, evaluated


def feature_records(
    *,
    trainer: ModuleType,
    cases: Sequence[Mapping[str, Any]],
    evaluated: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    evalmod = trainer.evalmod

    result: dict[str, dict[str, Any]] = {}

    for case in cases:
        case_id = str(case["case_id"])
        evaluation_row = evaluated[case_id]
        predictions = evaluation_row["predictions"]
        activation = trainer.layer6_activation(
            case
        )

        row: dict[str, Any] = {
            "case_id": case_id,
            "active": bool(
                activation["active"]
            ),
            "activation_status": str(
                activation["status"]
            ),
            "feature_vector": None,
            "feature_error": None,
            "static_prediction_available": False,
            "early_prediction_available": False,
        }

        static_prediction = predictions.get(
            evalmod.CALIBRATED_LAYER4_CANDIDATE
        )
        early_prediction = predictions.get(
            evalmod.EARLY_RESPONSE_CANDIDATE
        )

        row["static_prediction_available"] = (
            static_prediction is not None
        )
        row["early_prediction_available"] = (
            early_prediction is not None
        )

        if row["active"]:
            if static_prediction is None:
                row["feature_error"] = (
                    "missing_calibrated_layer4_prediction"
                )
            else:
                try:
                    feature_vector = tuple(
                        float(value)
                        for value
                        in trainer.layer6_feature_vector(
                            case,
                            static_prediction,
                        )
                    )

                    if not all(
                        math.isfinite(value)
                        for value in feature_vector
                    ):
                        raise ValueError(
                            "non-finite feature"
                        )

                    row["feature_vector"] = (
                        feature_vector
                    )

                except Exception as error:
                    row["feature_error"] = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

        result[case_id] = row

    return result


def date_summary(
    values: Sequence[date],
) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
        }

    return {
        "count": len(values),
        "minimum": min(values).isoformat(),
        "maximum": max(values).isoformat(),
    }


def main() -> None:
    args = parse_args()

    actual_hashes = {
        "current_cohort": sha256_file(
            args.current_cohort
        ),
        "full_cohort": sha256_file(
            args.full_cohort
        ),
        "derived_features": sha256_file(
            args.derived_features
        ),
        "locked_evaluation": sha256_file(
            args.locked_evaluation
        ),
        "strict_nested": sha256_file(
            args.strict_nested
        ),
        "ablation": sha256_file(
            args.ablation
        ),
        "trainer": sha256_file(
            args.trainer
        ),
    }

    for name, expected in (
        EXPECTED_HASHES.items()
    ):
        if actual_hashes[name] != expected:
            raise RuntimeError(
                f"{name} hash mismatch: "
                f"expected={expected} "
                f"actual={actual_hashes[name]}"
            )

    strict_nested = load_json(
        args.strict_nested
    )
    ablation = load_json(
        args.ablation
    )

    if strict_nested["decision"]["status"] != (
        "strict_nested_internal_gain_supported_"
        "external_or_temporal_validation_required"
    ):
        raise RuntimeError(
            "Strict nested artifact does not route "
            "to holdout validation"
        )

    if ablation["decision"][
        "recommended_candidate"
    ] != "ridge_t2_prior_only":
        raise RuntimeError(
            "Unexpected selected candidate"
        )

    trainer = load_module(
        "_oncotwin_executable_holdout_trainer",
        args.trainer,
    )

    trainer._verify_locked_artifact(
        args.locked_evaluation
    )

    current_rows = load_jsonl(
        args.current_cohort
    )
    full_rows = load_jsonl(
        args.full_cohort
    )
    derived_rows = load_jsonl(
        args.derived_features
    )

    current_by_id = index_unique(
        current_rows,
        name="current cohort",
    )
    full_by_id = index_unique(
        full_rows,
        name="full cohort",
    )

    if len(current_by_id) != 270:
        raise RuntimeError(
            f"Expected 270 current cases, "
            f"got {len(current_by_id)}"
        )

    if len(full_by_id) != 705:
        raise RuntimeError(
            f"Expected 705 full-cohort cases, "
            f"got {len(full_by_id)}"
        )

    current_ids = set(current_by_id)
    full_ids = set(full_by_id)

    if not current_ids.issubset(
        full_ids
    ):
        raise RuntimeError(
            "Current cohort is not a subset "
            "of the 705-case source"
        )

    outside_ids = full_ids - current_ids

    current_redacted = (
        args.temporary_directory
        / "current_redacted.jsonl"
    )
    full_redacted = (
        args.temporary_directory
        / "full_redacted.jsonl"
    )

    write_redacted_cohort(
        current_rows,
        current_redacted,
    )
    write_redacted_cohort(
        full_rows,
        full_redacted,
    )

    print(
        "REDACTED_PRIOR_PIPELINE_BEGIN "
        "current_cases=270 full_cases=705 "
        "real_t3_values_used=false",
        flush=True,
    )

    reference = trainer._build_training_data()

    reference_records = list(
        reference["records"]
    )
    reference_features = np.asarray(
        reference["feature_matrix"],
        dtype=float,
    )

    reference_index = {
        str(record["case_id"]): index
        for index, record
        in enumerate(reference_records)
    }

    current_cases, current_evaluated = (
        run_prior_pipeline(
            trainer=trainer,
            cohort_path=current_redacted,
        )
    )

    current_features = feature_records(
        trainer=trainer,
        cases=current_cases,
        evaluated=current_evaluated,
    )

    activation_mismatch_count = 0
    feature_mismatch_count = 0
    maximum_feature_difference = 0.0

    for case_id in sorted(current_ids):
        reference_position = reference_index[
            case_id
        ]
        reference_record = reference_records[
            reference_position
        ]
        audited = current_features[case_id]

        if bool(reference_record["active"]) != bool(
            audited["active"]
        ):
            activation_mismatch_count += 1
            continue

        if not bool(reference_record["active"]):
            continue

        feature_vector = audited[
            "feature_vector"
        ]

        if feature_vector is None:
            feature_mismatch_count += 1
            continue

        reference_vector = reference_features[
            reference_position
        ]

        difference = float(
            np.max(
                np.abs(
                    np.asarray(
                        feature_vector,
                        dtype=float,
                    )
                    - reference_vector
                )
            )
        )

        maximum_feature_difference = max(
            maximum_feature_difference,
            difference,
        )

        tolerance = FEATURE_TOLERANCE * max(
            1.0,
            float(
                np.max(
                    np.abs(reference_vector)
                )
            ),
        )

        if difference > tolerance:
            feature_mismatch_count += 1

    if activation_mismatch_count:
        raise RuntimeError(
            "T3-redacted current cohort changed "
            f"activation for {activation_mismatch_count} cases"
        )

    if feature_mismatch_count:
        raise RuntimeError(
            "T3-redacted current cohort changed "
            f"Layer 6 features for {feature_mismatch_count} cases; "
            f"max difference={maximum_feature_difference}"
        )

    print(
        "TARGET_INDEPENDENCE_REPRODUCTION "
        "cases=270 "
        "activation_mismatches=0 "
        "feature_mismatches=0 "
        f"max_feature_difference="
        f"{maximum_feature_difference:.12g} "
        "status=pass",
        flush=True,
    )

    full_cases, full_evaluated = (
        run_prior_pipeline(
            trainer=trainer,
            cohort_path=full_redacted,
        )
    )

    full_features = feature_records(
        trainer=trainer,
        cases=full_cases,
        evaluated=full_evaluated,
    )

    groups: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in derived_rows:
        case_id = case_id_from_row(row)
        landmark = normalize_landmark(row)

        if (
            case_id is not None
            and landmark is not None
        ):
            groups[
                (case_id, landmark)
            ].append(row)

    resolved: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    conflict_count = 0

    for key, rows in groups.items():
        values = sorted(
            value
            for value in (
                positive_float(
                    row.get(
                        "enhancing_volume_ml"
                    )
                )
                for row in rows
            )
            if value is not None
        )

        distinct: list[float] = []

        for value in values:
            if (
                not distinct
                or relative_error(
                    value,
                    distinct[-1],
                )
                > 1e-12
            ):
                distinct.append(value)

        if len(distinct) == 1:
            chosen = dict(rows[0])
            chosen[
                "_resolved_raw_volume"
            ] = distinct[0]
            resolved[key] = chosen

        elif len(distinct) > 1:
            conflict_count += 1

    exclusion_counts: Counter[str] = (
        Counter()
    )
    eligible_rows = []
    eligible_dates = []

    for case_id in sorted(outside_ids):
        feature_audit = full_features.get(
            case_id
        )

        if feature_audit is None:
            exclusion_counts[
                "missing_prior_pipeline_record"
            ] += 1
            continue

        if not feature_audit["active"]:
            exclusion_counts[
                "layer6_inactive"
            ] += 1
            continue

        if feature_audit[
            "feature_vector"
        ] is None:
            exclusion_counts[
                "layer6_feature_generation_failed"
            ] += 1
            continue

        source_rows = {
            label: resolved.get(
                (case_id, label)
            )
            for label in (
                "T0",
                "T1",
                "T2",
                "T3",
            )
        }

        if any(
            row is None
            for row in source_rows.values()
        ):
            exclusion_counts[
                "missing_unique_t0_t1_t2_or_t3"
            ] += 1
            continue

        converted = {}

        for label, source_row in (
            source_rows.items()
        ):
            raw = positive_float(
                source_row.get(
                    "_resolved_raw_volume"
                )
            )

            converted[label] = (
                raw / 1000.0
                if raw is not None
                else None
            )

        if any(
            value is None or value <= 0
            for value in converted.values()
        ):
            exclusion_counts[
                "missing_positive_longitudinal_volume"
            ] += 1
            continue

        canonical = full_by_id[case_id]

        canonical_values = {
            "T0": positive_float(
                canonical.get(
                    "baseline_volume_ml"
                )
            ),
            "T1": positive_float(
                canonical.get(
                    "early_volume_ml"
                )
            ),
            "T3": positive_float(
                canonical.get(
                    "final_volume_ml"
                )
            ),
        }

        if any(
            value is None
            for value in canonical_values.values()
        ):
            exclusion_counts[
                "missing_positive_canonical_volume"
            ] += 1
            continue

        alignment = {
            label: relative_error(
                converted[label],
                canonical_values[label],
            )
            for label in (
                "T0",
                "T1",
                "T3",
            )
        }

        if alignment["T0"] > ALIGNMENT_TOLERANCE:
            exclusion_counts[
                "t0_alignment_failure"
            ] += 1
            continue

        if alignment["T1"] > ALIGNMENT_TOLERANCE:
            exclusion_counts[
                "t1_alignment_failure"
            ] += 1
            continue

        if alignment["T3"] > ALIGNMENT_TOLERANCE:
            exclusion_counts[
                "t3_alignment_failure"
            ] += 1
            continue

        dates = {
            label: parse_actual_date(
                source_rows[label]
            )
            for label in source_rows
        }

        ordered_dates = [
            dates[label]
            for label in (
                "T0",
                "T1",
                "T2",
                "T3",
            )
        ]

        if all(
            value is not None
            for value in ordered_dates
        ):
            if not (
                dates["T0"]
                <= dates["T1"]
                <= dates["T2"]
                <= dates["T3"]
            ):
                exclusion_counts[
                    "actual_date_order_failure"
                ] += 1
                continue

        if dates["T0"] is not None:
            eligible_dates.append(
                dates["T0"]
            )

        feature_vector = tuple(
            float(value)
            for value in feature_audit[
                "feature_vector"
            ]
        )

        eligible_rows.append(
            {
                "case_id": case_id,
                "full_cohort_source_line": int(
                    canonical["_source_line"]
                ),
                "derived_t0_source_line": int(
                    source_rows["T0"][
                        "_source_line"
                    ]
                ),
                "derived_t1_source_line": int(
                    source_rows["T1"][
                        "_source_line"
                    ]
                ),
                "derived_t2_source_line": int(
                    source_rows["T2"][
                        "_source_line"
                    ]
                ),
                "derived_t3_source_line": int(
                    source_rows["T3"][
                        "_source_line"
                    ]
                ),
                "layer6_feature_vector_sha256": (
                    hashlib.sha256(
                        json.dumps(
                            feature_vector,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                ),
                "t0_actual_date": (
                    dates["T0"].isoformat()
                    if dates["T0"] is not None
                    else None
                ),
                "t1_actual_date": (
                    dates["T1"].isoformat()
                    if dates["T1"] is not None
                    else None
                ),
                "t2_actual_date": (
                    dates["T2"].isoformat()
                    if dates["T2"] is not None
                    else None
                ),
                "t3_actual_date": (
                    dates["T3"].isoformat()
                    if dates["T3"] is not None
                    else None
                ),
            }
        )

    eligible_rows.sort(
        key=lambda row: str(
            row["case_id"]
        )
    )

    eligible_ids = {
        str(row["case_id"])
        for row in eligible_rows
    }

    if eligible_ids & current_ids:
        raise RuntimeError(
            "Holdout eligibility overlaps development cohort"
        )

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": (
            "prespecified_before_holdout_performance_evaluation"
        ),
        "candidate": {
            "name": "ridge_t2_prior_only",
            "definition": (
                "Ridge update of log(T3 / Layer6 prior) "
                "using log(T2 / Layer6 prior) as its only feature."
            ),
            "selection_source": (
                "Layer 8 exploratory ablation v0_3"
            ),
            "fit_population": (
                "220 development landmark cases only"
            ),
            "alpha_selection": (
                "Four-fold cross-validation using development "
                "cases only, followed by refitting on all "
                "development cases."
            ),
        },
        "layer6_prior": {
            "runtime": (
                "Frozen Layer 6 deployable ensemble"
            ),
            "information_cutoff": "T1",
            "holdout_targets_used_for_inference": False,
        },
        "new_evidence": "T2",
        "held_out_target": "T3",
        "comparators": {
            "primary": "no_update",
            "secondary": "trend_half",
            "trend_half_definition": (
                "Layer6 prior multiplied by sqrt(T2 / T1)."
            ),
        },
        "primary_endpoints": [
            "paired candidate-minus-no_update delta log RMSE",
            "paired candidate-minus-no_update delta MAE",
        ],
        "success_rule": {
            "delta_log_rmse_bootstrap_95_ci_upper_below_zero": True,
            "delta_mae_bootstrap_95_ci_upper_below_zero": True,
            "coverage80_acceptable_range": [
                0.74,
                0.88,
            ],
            "coverage95_acceptable_range": [
                0.89,
                0.99,
            ],
            "maximum_predictor_defined_subgroup_harm_delta_log_rmse": (
                0.02
            ),
            "minimum_subgroup_size": 20,
        },
        "claim_limits": {
            "same_source_patient_holdout": True,
            "external_validation": False,
            "independent_whole_system_confirmation": False,
        },
        "access_rule": (
            "No holdout T3 values, predictions, errors, outcome "
            "distributions, or candidate rankings may be inspected "
            "until this protocol, the manifest, the development-only "
            "updater, and interval calibration are frozen."
        ),
    }

    protocol_hash = canonical_json_hash(
        protocol
    )

    manifest_rows = []

    for row in eligible_rows:
        manifest_rows.append(
            {
                "manifest_version": (
                    ANALYSIS_VERSION
                ),
                "protocol_version": (
                    PROTOCOL_VERSION
                ),
                "protocol_sha256": (
                    protocol_hash
                ),
                **row,
                "current_development_overlap": False,
                "layer6_executable_feature_vector": True,
                "t0_t1_t2_t3_measurements_present": True,
                "t0_t1_t3_alignment_passed": True,
                "target_value_redacted": True,
                "target_value_in_manifest": False,
                "predictions_generated": False,
                "performance_evaluated": False,
            }
        )

    candidate_count = len(
        manifest_rows
    )

    if candidate_count >= MIN_VALIDATION_CASES:
        decision = (
            "executable_same_source_holdout_manifest_ready"
        )
    elif candidate_count > 0:
        decision = (
            "executable_holdout_exists_but_below_prespecified_size"
        )
    else:
        decision = (
            "no_executable_same_source_holdout"
        )

    atomic_write_json(
        args.protocol_json,
        protocol,
    )

    if manifest_rows:
        atomic_write_jsonl(
            args.manifest_jsonl,
            manifest_rows,
        )
        manifest_sha256 = sha256_file(
            args.manifest_jsonl
        )
    else:
        args.manifest_jsonl.unlink(
            missing_ok=True
        )
        manifest_sha256 = None

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "executable_layer6_holdout_readiness_audit"
        ),
        "inputs": {
            "hashes": actual_hashes,
            "current_case_count": len(
                current_ids
            ),
            "full_case_count": len(
                full_ids
            ),
            "outside_current_count": len(
                outside_ids
            ),
        },
        "target_independence": {
            "redacted_final_volume_ml": (
                REDACTED_FINAL_VOLUME_ML
            ),
            "current_cases_checked": len(
                current_ids
            ),
            "activation_mismatch_count": (
                activation_mismatch_count
            ),
            "feature_mismatch_count": (
                feature_mismatch_count
            ),
            "maximum_absolute_feature_difference": (
                maximum_feature_difference
            ),
            "passed": True,
        },
        "full_executable_pipeline": {
            "full_cases_processed": len(
                full_features
            ),
            "outside_cases": len(
                outside_ids
            ),
            "outside_layer6_active_count": sum(
                bool(
                    full_features[case_id][
                        "active"
                    ]
                )
                for case_id in outside_ids
            ),
            "outside_finite_feature_count": sum(
                full_features[case_id][
                    "feature_vector"
                ]
                is not None
                for case_id in outside_ids
            ),
            "outside_feature_failure_count": sum(
                full_features[case_id][
                    "feature_error"
                ]
                is not None
                for case_id in outside_ids
            ),
            "locked_feature_names": list(
                trainer.LAYER6_FEATURE_NAMES
            ),
        },
        "longitudinal_resolution": {
            "derived_row_count": len(
                derived_rows
            ),
            "resolved_case_landmark_count": (
                len(resolved)
            ),
            "conflicting_group_count": (
                conflict_count
            ),
            "raw_to_ml_divisor": 1000.0,
        },
        "holdout": {
            "minimum_required": (
                MIN_VALIDATION_CASES
            ),
            "eligible_count": candidate_count,
            "development_overlap_count": len(
                eligible_ids & current_ids
            ),
            "exclusion_counts": dict(
                sorted(
                    exclusion_counts.items()
                )
            ),
            "target_values_exported": False,
            "real_holdout_performance_computed": False,
            "predictions_generated_for_manifest_cases": False,
            "t0_date_summary": date_summary(
                eligible_dates
            ),
        },
        "protocol": {
            "path": str(
                args.protocol_json
            ),
            "sha256": protocol_hash,
        },
        "manifest": {
            "path": str(
                args.manifest_jsonl
            ),
            "row_count": candidate_count,
            "sha256": manifest_sha256,
            "target_values_redacted": True,
        },
        "classification": {
            "status": decision,
            "same_source_patient_holdout": (
                candidate_count
                >= MIN_VALIDATION_CASES
            ),
            "external_validation": False,
            "independent_whole_system_confirmation": False,
        },
        "decision": {
            "status": decision,
            "next_step": (
                "Freeze the development-only updater and interval "
                "calibration, then evaluate once on the frozen "
                "same-source holdout manifest."
                if candidate_count >= MIN_VALIDATION_CASES
                else (
                    "Locate an external cohort or reconstruct "
                    "additional complete same-source cases."
                )
            ),
        },
    }

    atomic_write_json(
        args.audit_json,
        payload,
    )

    markdown = "\n".join(
        [
            "# Layer 8 executable holdout readiness",
            "",
            (
                "- Current development cases: "
                f"`{len(current_ids)}`"
            ),
            (
                "- Full same-source cases: "
                f"`{len(full_ids)}`"
            ),
            (
                "- Cases outside development: "
                f"`{len(outside_ids)}`"
            ),
            (
                "- T3-redacted feature mismatches: "
                f"`{feature_mismatch_count}`"
            ),
            (
                "- Executable eligible holdout cases: "
                f"`{candidate_count}`"
            ),
            (
                "- Minimum prespecified size: "
                f"`{MIN_VALIDATION_CASES}`"
            ),
            f"- Decision: `{decision}`",
            "",
            "## Claim boundary",
            "",
            (
                "This is a same-source patient holdout, not "
                "external validation or independent whole-system "
                "confirmation."
            ),
            "",
            (
                "Real holdout T3 values, predictions, errors, "
                "and performance metrics were not exported or "
                "evaluated."
            ),
        ]
    ) + "\n"

    atomic_write_text(
        args.audit_markdown,
        markdown,
    )

    print(
        "REDACTED_PRIOR_PIPELINE_COMPLETE "
        "real_t3_values_used=false "
        "real_holdout_performance_computed=false"
    )

    print(
        "EXECUTABLE_FEATURE_READINESS "
        f"outside_cases={len(outside_ids)} "
        f"active={payload['full_executable_pipeline']['outside_layer6_active_count']} "
        f"finite_features="
        f"{payload['full_executable_pipeline']['outside_finite_feature_count']} "
        f"feature_failures="
        f"{payload['full_executable_pipeline']['outside_feature_failure_count']}"
    )

    print(
        "HOLDOUT_MANIFEST_SUMMARY "
        f"eligible={candidate_count} "
        f"minimum_required={MIN_VALIDATION_CASES} "
        "development_overlap=0 "
        f"exclusions={dict(sorted(exclusion_counts.items()))}"
    )

    print(
        "VALIDATION_CLASSIFICATION "
        f"status={decision} "
        f"same_source_holdout="
        f"{str(candidate_count >= MIN_VALIDATION_CASES).lower()} "
        "external_validation=false "
        "independent_whole_system_confirmation=false"
    )

    print(
        "TARGET_ACCESS_GUARD "
        "real_holdout_targets_used=false "
        "target_values_exported=false "
        "predictions_generated_for_manifest=false "
        "performance_evaluated=false"
    )

    print(
        f"EXECUTABLE_HOLDOUT_AUDIT_JSON "
        f"path={args.audit_json}"
    )
    print(
        f"EXECUTABLE_HOLDOUT_AUDIT_REPORT "
        f"path={args.audit_markdown}"
    )
    print(
        f"HOLDOUT_PROTOCOL path={args.protocol_json}"
    )

    if manifest_rows:
        print(
            f"HOLDOUT_MANIFEST "
            f"path={args.manifest_jsonl}"
        )
    else:
        print(
            "HOLDOUT_MANIFEST not_written=true"
        )


if __name__ == "__main__":
    main()
