#!/usr/bin/env python3
"""Finalize Layer 8 internal evidence and external-validation boundary."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


ANALYSIS_VERSION = (
    "oncotwin_layer8_internal_evidence_closure_v0_1"
)

REQUIREMENTS_VERSION = (
    "oncotwin_layer8_external_validation_requirements_v0_1"
)

INELIGIBILITY_VERSION = (
    "oncotwin_layer8_same_source_ineligibility_v0_1"
)

EXPECTED_HASHES = {
    "current_cohort": (
        "8cd51c7f00c073a3d7894692a25bcd8c04384c201c8c3989e91cf5b832c664c0"
    ),
    "full_cohort": (
        "ef7533e1e3af12dc54da284f7df4bce13cd9dd7e895efe291f69aca575ca5fda"
    ),
    "ablation_json": (
        "1331d9b5b30e8950690bfaa270bd63efc9c3d4635a91b256224471415a56bb26"
    ),
    "strict_json": (
        "46b6417fd849f7c88bcacc7349f8b4fe8f302b6af85ce5dc6a62d79bdff7a5ba"
    ),
    "strict_stdout": (
        "a29bd703d2bcc913ec4c7c5cb3340d8863f60bc85dbc2f6c44b03945307b14a5"
    ),
    "readiness_json": (
        "5e8a497ea7e340c2148d6dfdd46b9f4e880bf5c447d1e651c0c9890dbd5a519e"
    ),
    "fast_context_json": (
        "f58513c2680fb055e75070004e35afe119730ddc1bdedaf2c454dc56deb88a86"
    ),
    "contract_json": (
        "e403901953f2e7a844859b964f2d740a94d582049a9bce6a26666742d672b4bc"
    ),
    "contract_stdout": (
        "f21864e443bf844e8b4bcc403a32b7f7cf5301e8e3458babd97fc66aede3b3e1"
    ),
    "layer6_locked": (
        "c910c99eb8b129794faa0db7215cebf669ad922bf6dfe5dd2df09e70146eec48"
    ),
}

CASE_PATTERN = re.compile(
    r"ISPY2[-_ ]?(\d+)",
    re.IGNORECASE,
)

FORBIDDEN_OUTPUT_FIELDS = {
    "baseline_volume_ml",
    "early_volume_ml",
    "final_volume_ml",
    "t2_intermediate_volume_ml",
    "heldout_volume_ml",
    "outcome_volume_ml",
    "prediction",
    "error",
    "residual",
    "target_value",
}


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
        "--ablation-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--strict-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--strict-stdout",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--readiness-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fast-context-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--contract-stdout",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--layer6-locked",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--closure-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--closure-markdown",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--requirements-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--requirements-markdown",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ineligibility-jsonl",
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
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
    rows = []

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
                    f"{path}:{line_number} is not an object"
                )

            rows.append(dict(value))

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


def normalize_case_id(value: object) -> str | None:
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
    indexed = {}

    for row in rows:
        case_id = case_id_from_row(row)

        if case_id is None:
            raise RuntimeError(
                f"{name} row missing case ID"
            )

        if case_id in indexed:
            raise RuntimeError(
                f"{name} duplicate case ID: {case_id}"
            )

        indexed[case_id] = dict(row)

    return indexed


def normalize_subtype(
    row: Mapping[str, Any],
) -> str:
    raw = (
        row.get("subtype")
        or row.get("cancer_subtype")
        or row.get("disease_context")
        or ""
    )

    text = re.sub(
        r"\s+",
        "",
        str(raw).upper(),
    )

    text = text.replace(
        "POSITIVE",
        "+",
    ).replace(
        "NEGATIVE",
        "-",
    )

    if (
        "TNBC" in text
        or "TRIPLE-" in text
        or "TRIPLENEGATIVE" in text
    ):
        return "TNBC"

    if "HR+/HER2-" in text:
        return "HR+/HER2-"

    if "HR+/HER2+" in text:
        return "HR+/HER2+"

    if "HR-/HER2+" in text:
        return "HR-/HER2+"

    return str(raw).strip()


def normalized_text(value: object) -> str | None:
    text = str(value or "").strip()

    return text if text else None


def extract_float(
    text: str,
    pattern: str,
) -> float:
    match = re.search(
        pattern,
        text,
    )

    if match is None:
        raise RuntimeError(
            f"Could not extract pattern: {pattern}"
        )

    value = float(match.group(1))

    if not math.isfinite(value):
        raise RuntimeError(
            f"Non-finite extracted value: {pattern}"
        )

    return value


def extract_integer(
    text: str,
    pattern: str,
) -> int:
    match = re.search(
        pattern,
        text,
    )

    if match is None:
        raise RuntimeError(
            f"Could not extract pattern: {pattern}"
        )

    return int(match.group(1))


def main() -> None:
    args = parse_args()

    hashes = {
        "current_cohort": sha256_file(
            args.current_cohort
        ),
        "full_cohort": sha256_file(
            args.full_cohort
        ),
        "ablation_json": sha256_file(
            args.ablation_json
        ),
        "strict_json": sha256_file(
            args.strict_json
        ),
        "strict_stdout": sha256_file(
            args.strict_stdout
        ),
        "readiness_json": sha256_file(
            args.readiness_json
        ),
        "fast_context_json": sha256_file(
            args.fast_context_json
        ),
        "contract_json": sha256_file(
            args.contract_json
        ),
        "contract_stdout": sha256_file(
            args.contract_stdout
        ),
        "layer6_locked": sha256_file(
            args.layer6_locked
        ),
    }

    for name, expected in EXPECTED_HASHES.items():
        if hashes[name] != expected:
            raise RuntimeError(
                f"{name} hash mismatch: "
                f"expected={expected} "
                f"actual={hashes[name]}"
            )

    ablation = load_json(
        args.ablation_json
    )
    strict = load_json(
        args.strict_json
    )
    readiness = load_json(
        args.readiness_json
    )
    fast_context = load_json(
        args.fast_context_json
    )
    contract = load_json(
        args.contract_json
    )

    strict_stdout = args.strict_stdout.read_text(
        encoding="utf-8"
    )
    contract_stdout = (
        args.contract_stdout.read_text(
            encoding="utf-8"
        )
    )

    if ablation["decision"][
        "recommended_candidate"
    ] != "ridge_t2_prior_only":
        raise RuntimeError(
            "Unexpected Layer 8 candidate"
        )

    if strict["decision"]["status"] != (
        "strict_nested_internal_gain_supported_"
        "external_or_temporal_validation_required"
    ):
        raise RuntimeError(
            "Unexpected strict nested decision"
        )

    if strict["decision"][
        "model_lock_allowed"
    ]:
        raise RuntimeError(
            "Strict nested artifact permits model locking"
        )

    if readiness["decision"]["status"] != (
        "no_executable_same_source_holdout"
    ):
        raise RuntimeError(
            "Unexpected readiness decision"
        )

    if fast_context["decision"]["status"] != (
        "verified_metadata_sources_found_but_"
        "fewer_than_100_contract_eligible_cases"
    ):
        raise RuntimeError(
            "Unexpected context-bridge decision"
        )

    if contract["decision"]["status"] != (
        "outside_clinical_context_exists_under_"
        "previously_unhandled_fields"
    ):
        raise RuntimeError(
            "Unexpected contract-schema decision"
        )

    current = index_unique(
        load_jsonl(
            args.current_cohort
        ),
        name="current cohort",
    )

    full = index_unique(
        load_jsonl(
            args.full_cohort
        ),
        name="full cohort",
    )

    current_ids = set(current)
    full_ids = set(full)
    outside_ids = full_ids - current_ids

    if (
        len(current_ids) != 270
        or len(full_ids) != 705
        or len(outside_ids) != 435
    ):
        raise RuntimeError(
            "Unexpected cohort counts"
        )

    if not current_ids.issubset(
        full_ids
    ):
        raise RuntimeError(
            "Current cohort is not a full-cohort subset"
        )

    current_subtypes = Counter(
        normalize_subtype(
            full[case_id]
        )
        for case_id in current_ids
    )

    outside_subtypes = Counter(
        normalize_subtype(
            full[case_id]
        )
        for case_id in outside_ids
    )

    expected_current_subtypes = Counter(
        {
            "TNBC": 270,
        }
    )

    expected_outside_subtypes = Counter(
        {
            "HR+/HER2-": 267,
            "HR+/HER2+": 109,
            "HR-/HER2+": 59,
        }
    )

    if current_subtypes != expected_current_subtypes:
        raise RuntimeError(
            "Development subtype distribution changed: "
            f"{dict(current_subtypes)}"
        )

    if outside_subtypes != expected_outside_subtypes:
        raise RuntimeError(
            "Outside subtype distribution changed: "
            f"{dict(outside_subtypes)}"
        )

    if outside_subtypes.get(
        "TNBC",
        0,
    ) != 0:
        raise RuntimeError(
            "TNBC cases unexpectedly exist outside development"
        )

    treatment_counts = Counter(
        normalized_text(
            full[case_id].get(
                "treatment_context"
            )
            or full[case_id].get(
                "schedule_type"
            )
        )
        for case_id in outside_ids
    )

    if treatment_counts != Counter(
        {
            "neoadjuvant chemotherapy": 435,
        }
    ):
        raise RuntimeError(
            "Unexpected outside treatment distribution: "
            f"{dict(treatment_counts)}"
        )

    if (
        "CONTRACT_SCHEMA_DECISION "
        "status=outside_clinical_context_exists_"
        "under_previously_unhandled_fields"
    ) not in contract_stdout:
        raise RuntimeError(
            "Contract stdout lacks expected final decision"
        )

    metrics = {
        "no_update": {
            "mae": extract_float(
                strict_stdout,
                r"STRICT_NESTED_AGGREGATE_METRICS "
                r"model=no_update .*?mae=([-+0-9.eE]+)",
            ),
            "log_rmse": extract_float(
                strict_stdout,
                r"STRICT_NESTED_AGGREGATE_METRICS "
                r"model=no_update .*?log_rmse=([-+0-9.eE]+)",
            ),
        },
        "trend_half": {
            "mae": extract_float(
                strict_stdout,
                r"STRICT_NESTED_AGGREGATE_METRICS "
                r"model=trend_half .*?mae=([-+0-9.eE]+)",
            ),
            "log_rmse": extract_float(
                strict_stdout,
                r"STRICT_NESTED_AGGREGATE_METRICS "
                r"model=trend_half .*?log_rmse=([-+0-9.eE]+)",
            ),
        },
        "ridge_t2_prior_only": {
            "mae": extract_float(
                strict_stdout,
                r"STRICT_NESTED_AGGREGATE_METRICS "
                r"model=ridge_t2_prior_only .*?"
                r"mae=([-+0-9.eE]+)",
            ),
            "log_rmse": extract_float(
                strict_stdout,
                r"STRICT_NESTED_AGGREGATE_METRICS "
                r"model=ridge_t2_prior_only .*?"
                r"log_rmse=([-+0-9.eE]+)",
            ),
        },
    }

    paired = {
        "candidate_vs_no_update": {
            "delta_mae": extract_float(
                strict_stdout,
                r"comparison=candidate_vs_no_update "
                r"delta_mae=([-+0-9.eE]+)",
            ),
            "delta_log_rmse": extract_float(
                strict_stdout,
                r"comparison=candidate_vs_no_update .*?"
                r"delta_log_rmse=([-+0-9.eE]+)",
            ),
        },
        "candidate_vs_trend_half": {
            "delta_mae": extract_float(
                strict_stdout,
                r"comparison=candidate_vs_trend_half "
                r"delta_mae=([-+0-9.eE]+)",
            ),
            "delta_log_rmse": extract_float(
                strict_stdout,
                r"comparison=candidate_vs_trend_half .*?"
                r"delta_log_rmse=([-+0-9.eE]+)",
            ),
        },
    }

    uncertainty = {
        "coverage80": extract_float(
            strict_stdout,
            r"STRICT_NESTED_UNCERTAINTY "
            r"coverage80=([-+0-9.eE]+)",
        ),
        "coverage95": extract_float(
            strict_stdout,
            r"STRICT_NESTED_UNCERTAINTY .*?"
            r"coverage95=([-+0-9.eE]+)",
        ),
    }

    repeat_stability = {
        "candidate_beats_no_update": (
            extract_integer(
                strict_stdout,
                r"STRICT_NESTED_REPEAT_STABILITY "
                r"candidate_beats_no_update=(\d+)/5",
            )
        ),
        "candidate_beats_trend_half": (
            extract_integer(
                strict_stdout,
                r"STRICT_NESTED_REPEAT_STABILITY .*?"
                r"candidate_beats_trend_half=(\d+)/5",
            )
        ),
        "repeat_count": 5,
    }

    ineligibility_rows = []

    for case_id in sorted(outside_ids):
        row = full[case_id]
        subtype = normalize_subtype(row)

        output = {
            "record_version": (
                INELIGIBILITY_VERSION
            ),
            "case_id": case_id,
            "subtype": subtype,
            "hr_status": normalized_text(
                row.get("hr_status")
            ),
            "her2_status": normalized_text(
                row.get("her2_status")
            ),
            "er_status": normalized_text(
                row.get("er_status")
            ),
            "pr_status": normalized_text(
                row.get("pr_status")
            ),
            "treatment_context": (
                normalized_text(
                    row.get(
                        "treatment_context"
                    )
                    or row.get(
                        "schedule_type"
                    )
                )
            ),
            "regimen_name": normalized_text(
                row.get("regimen_name")
                or row.get(
                    "treatment_regimen"
                )
            ),
            "current_tnbc_development_overlap": (
                False
            ),
            "eligible_for_frozen_tnbc_contract": (
                False
            ),
            "ineligibility_reason": (
                "outside_case_is_non_tnbc_and_outside_"
                "the_frozen_tnbc_parameter_contract"
            ),
            "target_value_in_record": False,
            "prediction_generated": False,
            "performance_evaluated": False,
        }

        if {
            key.lower()
            for key in output
        } & FORBIDDEN_OUTPUT_FIELDS:
            raise RuntimeError(
                "Ineligibility record contains a "
                "forbidden target or performance field"
            )

        ineligibility_rows.append(output)

    atomic_write_jsonl(
        args.ineligibility_jsonl,
        ineligibility_rows,
    )

    ineligibility_sha256 = sha256_file(
        args.ineligibility_jsonl
    )

    requirements = {
        "requirements_version": (
            REQUIREMENTS_VERSION
        ),
        "candidate": {
            "name": "ridge_t2_prior_only",
            "status": (
                "frozen_research_candidate_"
                "not_approved_for_deployment"
            ),
            "definition": (
                "Ridge update of log(T3 / Layer6 prior) "
                "using log(T2 / Layer6 prior) as the "
                "only updater feature."
            ),
            "selection_population": (
                "I-SPY2 TNBC patients receiving "
                "neoadjuvant chemotherapy"
            ),
        },
        "required_population": {
            "disease": "breast cancer",
            "subtype": "TNBC",
            "treatment_context": (
                "neoadjuvant chemotherapy"
            ),
            "patient_overlap_with_development": 0,
            "minimum_screening_case_count": 100,
            "sample_size_note": (
                "The threshold of 100 is an intake-screening "
                "minimum. A formal endpoint-specific sample-size "
                "justification must be completed before evaluation."
            ),
        },
        "required_longitudinal_imaging": {
            "T0": (
                "pretreatment baseline enhancing volume"
            ),
            "T1": (
                "early-treatment enhancing volume and "
                "Layer 6 information cutoff"
            ),
            "T2": (
                "later intermediate enhancing volume used "
                "as new Layer 8 evidence"
            ),
            "T3": (
                "final enhancing-volume target withheld "
                "until protocol and models are frozen"
            ),
            "positive_volume_required": True,
            "unit_provenance_required": True,
            "timepoint_provenance_required": True,
            "measurement_method_consistency_required": True,
        },
        "required_clinical_metadata": [
            "case identifier",
            "TNBC or receptor-status evidence",
            "treatment regimen or arm",
            "treatment schedule",
            "acquisition dates or treatment-relative timing",
            "site or institution identifier when available",
        ],
        "development_freeze_requirements": [
            "freeze Layer 6 runtime",
            "freeze Layer 8 candidate definition",
            "select ridge alpha using development cases only",
            "fit updater using development cases only",
            "freeze uncertainty calibration",
            "freeze validation manifest",
            "freeze endpoint and subgroup protocol",
            "do not inspect validation T3 outcomes beforehand",
        ],
        "primary_endpoints": [
            (
                "paired candidate-minus-no_update "
                "delta log RMSE"
            ),
            (
                "paired candidate-minus-no_update "
                "delta MAE"
            ),
        ],
        "secondary_endpoints": [
            "RMSE",
            "bias",
            "80% interval coverage and width",
            "95% interval coverage and width",
            "candidate versus fixed trend_half comparator",
            "predictor-defined subgroup harm",
        ],
        "prespecified_success_rule": {
            "candidate_vs_no_update_delta_log_rmse_"
            "bootstrap_95_ci_upper_below_zero": True,
            "candidate_vs_no_update_delta_mae_"
            "bootstrap_95_ci_upper_below_zero": True,
            "coverage80_acceptable_range": [
                0.74,
                0.88,
            ],
            "coverage95_acceptable_range": [
                0.89,
                0.99,
            ],
            "maximum_predictor_defined_subgroup_"
            "harm_delta_log_rmse": 0.02,
            "minimum_subgroup_size": 20,
        },
        "prohibited_substitutions": [
            (
                "Do not use HR-positive or HER2-positive "
                "patients as TNBC confirmation."
            ),
            (
                "Do not relabel the remaining 435 I-SPY2 "
                "patients as TNBC."
            ),
            (
                "Do not use the 220 candidate-selection "
                "patients as independent validation."
            ),
            (
                "Do not treat additional split repetitions "
                "as new independent patients."
            ),
        ],
        "claim_boundary": {
            "same_source_validation_available": False,
            "external_validation_required": True,
            "deployment_allowed_before_external_validation": False,
            "independent_confirmation_available": False,
        },
    }

    atomic_write_json(
        args.requirements_json,
        requirements,
    )

    requirements_sha256 = sha256_file(
        args.requirements_json
    )

    requirements_markdown = "\n".join(
        [
            "# Layer 8 external validation requirements",
            "",
            "## Required population",
            "",
            (
                "An independent TNBC cohort receiving "
                "neoadjuvant chemotherapy, with no patient "
                "overlap with the 270-case development cohort."
            ),
            "",
            "## Required longitudinal evidence",
            "",
            (
                "Each evaluable patient must have positive, "
                "provenance-traceable enhancing-volume "
                "measurements at T0, T1, T2, and T3."
            ),
            "",
            (
                "Layer 6 is restricted to information available "
                "through T1. T2 is the new Layer 8 evidence, and "
                "T3 remains inaccessible until the protocol, "
                "manifest, updater, and interval calibration are "
                "frozen."
            ),
            "",
            "## Claim boundary",
            "",
            (
                "The remaining 435 same-source I-SPY2 patients "
                "are non-TNBC and cannot provide confirmatory "
                "validation for the frozen TNBC candidate."
            ),
            "",
            (
                "Layer 8 cannot be promoted or deployed before "
                "successful independent TNBC validation."
            ),
        ]
    ) + "\n"

    atomic_write_text(
        args.requirements_markdown,
        requirements_markdown,
    )

    closure = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "layer8_internal_evidence_complete_"
            "external_validation_required"
        ),
        "inputs": {
            "hashes": hashes,
        },
        "candidate": {
            "name": "ridge_t2_prior_only",
            "feature_count": 1,
            "new_evidence": "T2",
            "prior_information_cutoff": "T1",
            "held_out_internal_target": "T3",
            "status": (
                "frozen_research_candidate"
            ),
        },
        "strict_nested_internal_evidence": {
            "case_count": 220,
            "repeat_count": 5,
            "strict_full_stack_nesting": True,
            "metrics": metrics,
            "paired_comparisons": paired,
            "uncertainty": uncertainty,
            "repeat_stability": (
                repeat_stability
            ),
            "subgroup_harm_flags": 0,
            "independent_confirmation": False,
            "model_lock_allowed": False,
        },
        "same_source_population_accounting": {
            "full_case_count": 705,
            "development_tnbc_case_count": 270,
            "outside_development_case_count": 435,
            "development_subtype_counts": dict(
                sorted(
                    current_subtypes.items()
                )
            ),
            "outside_subtype_counts": dict(
                sorted(
                    outside_subtypes.items()
                )
            ),
            "outside_tnbc_case_count": 0,
            "outside_treatment_counts": dict(
                sorted(
                    treatment_counts.items(),
                    key=lambda item: str(
                        item[0]
                    ),
                )
            ),
            "same_source_tnbc_holdout_available": False,
            "reason": (
                "The 270 development cases exhaust the TNBC "
                "patients in the 705-case source. Every remaining "
                "patient belongs to a non-TNBC subtype."
            ),
        },
        "artifacts": {
            "same_source_ineligibility": {
                "path": str(
                    args.ineligibility_jsonl
                ),
                "sha256": (
                    ineligibility_sha256
                ),
                "row_count": len(
                    ineligibility_rows
                ),
                "target_values_exported": False,
            },
            "external_validation_requirements": {
                "path": str(
                    args.requirements_json
                ),
                "sha256": (
                    requirements_sha256
                ),
            },
        },
        "scientific_decision": {
            "layer6_remains_primary_predictor": True,
            "layer8_promoted": False,
            "layer8_deployment_allowed": False,
            "layer8_model_lock_allowed": False,
            "same_source_validation_allowed": False,
            "external_tnbc_validation_required": True,
            "non_tnbc_generalization_claim_allowed": False,
            "decision_status": (
                "retain_layer8_as_internal_research_"
                "candidate_seek_external_tnbc_validation"
            ),
        },
        "claim_boundary": {
            "supported": [
                (
                    "T2 contains incremental response signal "
                    "within the 220-case TNBC development cohort."
                ),
                (
                    "The parsimonious one-feature updater "
                    "survived strict nested internal evaluation."
                ),
                (
                    "Layer 6 features are independent of the "
                    "T3 held-out target."
                ),
            ],
            "not_supported": [
                "independent validation",
                "external validation",
                "same-source TNBC holdout validation",
                "deployment superiority over Layer 6",
                "generalization to non-TNBC disease",
            ],
        },
    }

    atomic_write_json(
        args.closure_json,
        closure,
    )

    closure_markdown = "\n".join(
        [
            "# Layer 8 internal evidence closure",
            "",
            "## Internal result",
            "",
            (
                "The frozen `ridge_t2_prior_only` candidate "
                "improved strict nested internal log RMSE from "
                f"`{metrics['no_update']['log_rmse']:.6f}` to "
                f"`{metrics['ridge_t2_prior_only']['log_rmse']:.6f}`."
            ),
            "",
            (
                "The candidate remains a research candidate, "
                "not a deployable replacement for Layer 6."
            ),
            "",
            "## Same-source validation boundary",
            "",
            (
                "The 705-case source contains 270 TNBC patients, "
                "all of whom are already in the development cohort."
            ),
            "",
            (
                "The remaining 435 patients comprise 267 "
                "HR+/HER2−, 109 HR+/HER2+, and 59 HR−/HER2+ "
                "patients. There are no unused TNBC patients."
            ),
            "",
            "## Decision",
            "",
            (
                "Layer 6 remains the primary deployable predictor. "
                "Layer 8 is not promoted or locked for deployment. "
                "Independent external TNBC validation is required."
            ),
        ]
    ) + "\n"

    atomic_write_text(
        args.closure_markdown,
        closure_markdown,
    )

    print(
        "SAME_SOURCE_POPULATION_ACCOUNTING "
        "full=705 development_tnbc=270 outside=435 "
        "outside_tnbc=0 "
        f"outside_subtypes={dict(sorted(outside_subtypes.items()))}"
    )

    print(
        "LAYER8_INTERNAL_EVIDENCE "
        f"no_update_log_rmse="
        f"{metrics['no_update']['log_rmse']} "
        f"candidate_log_rmse="
        f"{metrics['ridge_t2_prior_only']['log_rmse']} "
        f"delta_log_rmse="
        f"{paired['candidate_vs_no_update']['delta_log_rmse']} "
        f"coverage80={uncertainty['coverage80']} "
        f"coverage95={uncertainty['coverage95']}"
    )

    print(
        "SAME_SOURCE_VALIDATION_DECISION "
        "available=false "
        "reason=no_unused_tnbc_patients"
    )

    print(
        "LAYER8_PROMOTION_DECISION "
        "layer6_remains_primary=true "
        "layer8_promoted=false "
        "layer8_deployment_allowed=false "
        "external_tnbc_validation_required=true"
    )

    print(
        "TARGET_ACCESS_GUARD "
        "ineligibility_targets_exported=false "
        "new_predictions_generated=false "
        "new_performance_computed=false"
    )

    print(
        f"LAYER8_CLOSURE_JSON "
        f"path={args.closure_json}"
    )
    print(
        f"LAYER8_CLOSURE_REPORT "
        f"path={args.closure_markdown}"
    )
    print(
        f"EXTERNAL_VALIDATION_REQUIREMENTS "
        f"path={args.requirements_json}"
    )
    print(
        f"SAME_SOURCE_INELIGIBILITY "
        f"path={args.ineligibility_jsonl}"
    )


if __name__ == "__main__":
    main()
