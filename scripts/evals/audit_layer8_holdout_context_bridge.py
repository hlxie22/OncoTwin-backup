#!/usr/bin/env python3
"""Diagnose and locate a verified clinical-context bridge for Layer 8."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ANALYSIS_VERSION = (
    "oncotwin_layer8_holdout_context_bridge_audit_v0_1"
)

EXPECTED_HASHES = {
    "current_cohort": (
        "8cd51c7f00c073a3d7894692a25bcd8c04384c201c8c3989e91cf5b832c664c0"
    ),
    "full_cohort": (
        "ef7533e1e3af12dc54da284f7df4bce13cd9dd7e895efe291f69aca575ca5fda"
    ),
    "readiness_audit": (
        "5e8a497ea7e340c2148d6dfdd46b9f4e880bf5c447d1e651c0c9890dbd5a519e"
    ),
    "readiness_script": (
        "541f8e0e6fdcb5207f29c89f8d918b971eaa4dac8c4dc79bb6881bccadabab23"
    ),
    "locked_evaluation": (
        "c910c99eb8b129794faa0db7215cebf669ad922bf6dfe5dd2df09e70146eec48"
    ),
    "trainer": (
        "a25016f347a92b3a1bd8280bf80fca2cb65befe5c4877857d6a16a67d8f269b0"
    ),
}

CASE_PATTERN = re.compile(
    r"ISPY2[-_ ]?(\d+)",
    re.IGNORECASE,
)

SUPPORTED_SUFFIXES = {
    ".jsonl",
    ".json",
    ".csv",
    ".tsv",
}

MAX_SOURCE_BYTES = 250_000_000
MIN_SOURCE_OVERLAP = 20
MIN_VERIFIED_AGREEMENT = 0.95
MIN_BRIDGE_CASES = 100

CANONICAL_ALIASES = {
    "data_origin": (
        "data_origin",
        "dataset",
        "dataset_name",
        "source_dataset",
        "study",
        "trial",
    ),
    "subtype": (
        "subtype",
        "cancer_subtype",
        "disease_context",
        "molecular_subtype",
        "tumor_subtype",
        "breast_cancer_subtype",
    ),
    "treatment_context": (
        "treatment_context",
        "schedule_type",
        "treatment_regimen",
        "regimen_name",
        "treatment_arm",
        "regimen",
        "therapy",
    ),
    "er_status": (
        "er_status",
        "er",
        "estrogen_receptor_status",
    ),
    "pr_status": (
        "pr_status",
        "pr",
        "progesterone_receptor_status",
    ),
    "hr_status": (
        "hr_status",
        "hormone_receptor_status",
    ),
    "her2_status": (
        "her2_status",
        "her2",
        "her_2_status",
    ),
    "segmentation_qc": (
        "segmentation_qc",
        "segmentation_quality",
        "mask_qc",
        "mri_qc",
    ),
}

RAW_FILL_FIELDS = {
    "data_origin": (
        "data_origin",
    ),
    "subtype": (
        "subtype",
        "cancer_subtype",
        "disease_context",
    ),
    "treatment_context": (
        "treatment_context",
        "schedule_type",
        "treatment_regimen",
        "regimen_name",
    ),
    "er_status": (
        "er_status",
    ),
    "pr_status": (
        "pr_status",
    ),
    "hr_status": (
        "hr_status",
    ),
    "her2_status": (
        "her2_status",
    ),
    "segmentation_qc": (
        "segmentation_qc",
    ),
}

SOURCE_NAME_TOKENS = (
    "clinical",
    "metadata",
    "manifest",
    "cohort",
    "patient",
    "ispy2",
    "feature",
    "map",
    "stack",
    "curated",
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
        "--readiness-audit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--readiness-script",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--locked-evaluation",
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

            rows.append(dict(value))

    return rows


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = None

        for key in (
            "rows",
            "records",
            "cases",
            "patients",
            "cohort",
            "data",
            "features",
        ):
            candidate = value.get(key)

            if isinstance(candidate, list):
                rows = candidate
                break

        if rows is None:
            return []
    else:
        return []

    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
    ]


def load_delimited(path: Path) -> list[dict[str, Any]]:
    delimiter = (
        "\t"
        if path.suffix.lower() == ".tsv"
        else ","
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return [
            dict(row)
            for row in csv.DictReader(
                handle,
                delimiter=delimiter,
            )
        ]


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return load_jsonl(path)

    if suffix == ".json":
        return load_json_rows(path)

    if suffix in {
        ".csv",
        ".tsv",
    }:
        return load_delimited(path)

    return []


def load_source_rows(
    path: Path,
) -> list[dict[str, Any]]:
    if path.is_file():
        return load_rows(path)

    if not path.is_dir():
        return []

    rows: list[dict[str, Any]] = []

    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if (
            candidate.is_file()
            and candidate.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )

    if len(files) > 3000:
        return []

    total_bytes = 0

    for candidate in files:
        try:
            size = candidate.stat().st_size
        except OSError:
            continue

        total_bytes += size

        if total_bytes > MAX_SOURCE_BYTES:
            return []

        try:
            rows.extend(
                load_rows(candidate)
            )
        except Exception:
            continue

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
        or row.get("patient")
        or row.get("subject")
    )


def nonempty(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False

    text = str(value).strip()

    return text.upper() not in {
        "",
        "NA",
        "N/A",
        "NONE",
        "NULL",
        "NAN",
        "UNKNOWN",
        "NOT AVAILABLE",
    }


def first_alias_value(
    row: Mapping[str, Any],
    canonical: str,
) -> object | None:
    normalized_key_map = {
        re.sub(
            r"[^a-z0-9]",
            "",
            str(key).lower(),
        ): key
        for key in row
    }

    for alias in CANONICAL_ALIASES[canonical]:
        normalized_alias = re.sub(
            r"[^a-z0-9]",
            "",
            alias.lower(),
        )

        source_key = normalized_key_map.get(
            normalized_alias
        )

        if source_key is None:
            continue

        value = row.get(source_key)

        if nonempty(value):
            return value

    return None


def normalize_metadata_value(
    canonical: str,
    value: object,
) -> str | None:
    if not nonempty(value):
        return None

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).strip().lower(),
    ).strip()

    compact = text.replace(" ", "")

    if canonical == "data_origin":
        if (
            "ispy2" in compact
            or "ispy" in compact
        ):
            return "ispy2"

    if canonical == "subtype":
        if (
            "tnbc" in compact
            or "triplenegative" in compact
        ):
            return "tnbc"

    if canonical == "treatment_context":
        if (
            "chemotherap" in text
            or "neoadjuvant" in text
            or compact in {
                "act",
                "acpaclitaxel",
            }
        ):
            return "neoadjuvant_chemotherapy"

    if canonical in {
        "er_status",
        "pr_status",
        "hr_status",
        "her2_status",
    }:
        if any(
            token in text
            for token in (
                "negative",
                "neg",
                "0",
            )
        ):
            return "negative"

        if any(
            token in text
            for token in (
                "positive",
                "pos",
                "1",
            )
        ):
            return "positive"

    return text


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
                f"{name} duplicate case ID: {case_id}"
            )

        indexed[case_id] = dict(row)

    return indexed


def redacted_row(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(source)

    row["final_volume_ml"] = 1.0

    for alias in (
        "heldout_volume_ml",
        "outcome_volume_ml",
    ):
        if alias in row:
            row[alias] = 1.0

    return row


def write_jsonl(
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


def metadata_by_case(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, object]],
    int,
]:
    collected: dict[
        str,
        dict[str, list[object]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        case_id = case_id_from_row(row)

        if case_id is None:
            continue

        for canonical in CANONICAL_ALIASES:
            value = first_alias_value(
                row,
                canonical,
            )

            if value is not None:
                collected[case_id][
                    canonical
                ].append(value)

    result: dict[
        str,
        dict[str, object],
    ] = {}

    conflict_count = 0

    for case_id, field_values in (
        collected.items()
    ):
        result[case_id] = {}

        for canonical, values in (
            field_values.items()
        ):
            normalized_groups: dict[
                str,
                list[object],
            ] = defaultdict(list)

            for value in values:
                normalized = normalize_metadata_value(
                    canonical,
                    value,
                )

                if normalized is not None:
                    normalized_groups[
                        normalized
                    ].append(value)

            if len(normalized_groups) == 1:
                only_values = next(
                    iter(
                        normalized_groups.values()
                    )
                )
                result[case_id][
                    canonical
                ] = only_values[0]

            elif len(normalized_groups) > 1:
                conflict_count += 1

    return result, conflict_count


def discover_sources(
    excluded: set[Path],
) -> list[Path]:
    sources: set[Path] = set()

    roots = (
        Path("data/curated"),
        Path("data/processed"),
        Path("data/ispy2_streaming"),
        Path("data/v1_prior_stack"),
    )

    aggregate_directory_names = {
        "patients",
        "patient_feature_rows",
        "clinical",
        "metadata",
        "manifests",
    }

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            try:
                resolved = path.resolve()

                if resolved in excluded:
                    continue

                if path.is_dir():
                    if (
                        path.name.lower()
                        in aggregate_directory_names
                    ):
                        sources.add(path)
                    continue

                if (
                    path.suffix.lower()
                    not in SUPPORTED_SUFFIXES
                ):
                    continue

                if (
                    path.parent.name.lower()
                    in {
                        "patients",
                        "patient_feature_rows",
                    }
                ):
                    continue

                if path.stat().st_size > MAX_SOURCE_BYTES:
                    continue

                lowered = str(path).lower()

                if any(
                    token in lowered
                    for token in SOURCE_NAME_TOKENS
                ):
                    sources.add(path)

            except OSError:
                continue

    return sorted(
        sources,
        key=lambda path: str(path),
    )


def field_coverage(
    metadata: Mapping[str, Mapping[str, object]],
    case_ids: set[str],
) -> dict[str, int]:
    return {
        canonical: sum(
            canonical in metadata.get(
                case_id,
                {}
            )
            for case_id in case_ids
        )
        for canonical in CANONICAL_ALIASES
    }


def agreement_summary(
    source_metadata: Mapping[
        str,
        Mapping[str, object],
    ],
    current_metadata: Mapping[
        str,
        Mapping[str, object],
    ],
    overlap_ids: set[str],
) -> dict[str, Any]:
    by_field = {}
    total_comparisons = 0
    total_matches = 0

    for canonical in CANONICAL_ALIASES:
        comparisons = 0
        matches = 0

        for case_id in overlap_ids:
            expected = current_metadata.get(
                case_id,
                {},
            ).get(canonical)

            observed = source_metadata.get(
                case_id,
                {},
            ).get(canonical)

            expected_normalized = (
                normalize_metadata_value(
                    canonical,
                    expected,
                )
            )
            observed_normalized = (
                normalize_metadata_value(
                    canonical,
                    observed,
                )
            )

            if (
                expected_normalized is None
                or observed_normalized is None
            ):
                continue

            comparisons += 1

            if (
                expected_normalized
                == observed_normalized
            ):
                matches += 1

        by_field[canonical] = {
            "comparisons": comparisons,
            "matches": matches,
            "agreement": (
                matches / comparisons
                if comparisons
                else None
            ),
        }

        total_comparisons += comparisons
        total_matches += matches

    return {
        "total_comparisons": total_comparisons,
        "total_matches": total_matches,
        "overall_agreement": (
            total_matches / total_comparisons
            if total_comparisons
            else None
        ),
        "by_field": by_field,
    }


def contract_core_complete(
    metadata: Mapping[str, object],
) -> bool:
    return (
        "subtype" in metadata
        and "treatment_context" in metadata
    )


def main() -> None:
    args = parse_args()

    actual_hashes = {
        "current_cohort": sha256_file(
            args.current_cohort
        ),
        "full_cohort": sha256_file(
            args.full_cohort
        ),
        "readiness_audit": sha256_file(
            args.readiness_audit
        ),
        "readiness_script": sha256_file(
            args.readiness_script
        ),
        "locked_evaluation": sha256_file(
            args.locked_evaluation
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

    readiness = load_json(
        args.readiness_audit
    )

    if readiness["decision"]["status"] != (
        "no_executable_same_source_holdout"
    ):
        raise RuntimeError(
            "Unexpected predecessor decision"
        )

    if readiness["target_independence"][
        "feature_mismatch_count"
    ] != 0:
        raise RuntimeError(
            "Predecessor target-independence audit failed"
        )

    if readiness["full_executable_pipeline"][
        "outside_layer6_active_count"
    ] != 417:
        raise RuntimeError(
            "Unexpected predecessor active count"
        )

    if readiness["full_executable_pipeline"][
        "outside_finite_feature_count"
    ] != 0:
        raise RuntimeError(
            "Unexpected predecessor feature count"
        )

    trainer = load_module(
        "_oncotwin_context_bridge_trainer",
        args.trainer,
    )

    trainer._verify_locked_artifact(
        args.locked_evaluation
    )

    evalmod = trainer.evalmod

    resolve_contract = getattr(
        evalmod,
        "resolve_parameter_contract",
        None,
    )
    target_contract_id = getattr(
        evalmod,
        "TNBC_CHEMO_CONTRACT_ID",
        None,
    )

    if (
        resolve_contract is None
        or target_contract_id is None
    ):
        raise RuntimeError(
            "Evaluator does not expose contract resolution"
        )

    current_rows = load_jsonl(
        args.current_cohort
    )
    full_rows = load_jsonl(
        args.full_cohort
    )

    current_raw = index_unique(
        current_rows,
        name="current cohort",
    )
    full_raw = index_unique(
        full_rows,
        name="full cohort",
    )

    current_ids = set(current_raw)
    full_ids = set(full_raw)
    outside_ids = full_ids - current_ids

    if (
        len(current_ids) != 270
        or len(full_ids) != 705
        or len(outside_ids) != 435
    ):
        raise RuntimeError(
            "Unexpected cohort counts"
        )

    current_redacted_path = (
        args.temporary_directory
        / "current_context_redacted.jsonl"
    )
    full_redacted_path = (
        args.temporary_directory
        / "full_context_redacted.jsonl"
    )

    write_jsonl(
        current_redacted_path,
        [
            redacted_row(row)
            for row in current_rows
        ],
    )
    write_jsonl(
        full_redacted_path,
        [
            redacted_row(row)
            for row in full_rows
        ],
    )

    current_cases = evalmod.load_real_cohort(
        current_redacted_path,
        allow_demo_data=False,
    )
    full_cases = evalmod.load_real_cohort(
        full_redacted_path,
        allow_demo_data=False,
    )

    current_case_map = {
        str(case["case_id"]): case
        for case in current_cases
    }
    full_case_map = {
        str(case["case_id"]): case
        for case in full_cases
    }

    current_contract_counts: Counter[str] = (
        Counter()
    )
    outside_contract_counts: Counter[str] = (
        Counter()
    )
    outside_active_contract_counts: Counter[
        str
    ] = Counter()

    outside_active_count = 0
    outside_target_contract_active_count = 0

    for case_id in sorted(current_ids):
        contract = resolve_contract(
            current_case_map[case_id][
                "context"
            ]
        )
        current_contract_counts[
            str(contract.contract_id)
        ] += 1

    for case_id in sorted(outside_ids):
        case = full_case_map[case_id]
        contract = resolve_contract(
            case["context"]
        )
        contract_id = str(
            contract.contract_id
        )

        outside_contract_counts[
            contract_id
        ] += 1

        activation = trainer.layer6_activation(
            case
        )

        if bool(activation["active"]):
            outside_active_count += 1
            outside_active_contract_counts[
                contract_id
            ] += 1

            if contract_id == str(
                target_contract_id
            ):
                outside_target_contract_active_count += 1

    if outside_active_count != 417:
        raise RuntimeError(
            "Direct activation count did not reproduce 417"
        )

    context_failure_explains_all = (
        outside_target_contract_active_count == 0
    )

    current_metadata, current_conflicts = (
        metadata_by_case(current_rows)
    )
    full_metadata, full_conflicts = (
        metadata_by_case(full_rows)
    )

    current_field_coverage = field_coverage(
        current_metadata,
        current_ids,
    )
    outside_field_coverage = field_coverage(
        full_metadata,
        outside_ids,
    )

    print(
        "CONTRACT_FAILURE_DIAGNOSIS "
        f"target_contract={target_contract_id} "
        f"current_contracts="
        f"{dict(sorted(current_contract_counts.items()))} "
        f"outside_contracts="
        f"{dict(sorted(outside_contract_counts.items()))} "
        f"outside_active_contracts="
        f"{dict(sorted(outside_active_contract_counts.items()))} "
        f"outside_active={outside_active_count} "
        f"outside_active_target_contract="
        f"{outside_target_contract_active_count} "
        f"context_failure_explains_all="
        f"{str(context_failure_explains_all).lower()}"
    )

    print(
        "RAW_CONTEXT_COVERAGE "
        f"current={current_field_coverage} "
        f"outside={outside_field_coverage}"
    )

    excluded = {
        args.current_cohort.resolve(),
        args.full_cohort.resolve(),
        args.audit_json.resolve(),
        args.audit_markdown.resolve(),
    }

    source_paths = discover_sources(
        excluded
    )

    print(
        "METADATA_SOURCE_SCAN_BEGIN "
        f"sources={len(source_paths)}"
    )

    source_results = []
    source_metadata_by_path: dict[
        str,
        dict[str, dict[str, object]],
    ] = {}

    for source_path in source_paths:
        result: dict[str, Any] = {
            "path": str(source_path),
            "kind": (
                "directory"
                if source_path.is_dir()
                else "file"
            ),
            "parse_status": "not_started",
        }

        try:
            rows = load_source_rows(
                source_path
            )

            if not rows:
                result["parse_status"] = (
                    "empty_or_unsupported"
                )
                source_results.append(result)
                continue

            metadata, conflict_count = (
                metadata_by_case(rows)
            )

            ids = set(metadata)
            overlap = ids & current_ids

            if len(overlap) < MIN_SOURCE_OVERLAP:
                result.update(
                    {
                        "parse_status": "pass",
                        "row_count": len(rows),
                        "unique_case_count": len(
                            ids
                        ),
                        "current_overlap_count": len(
                            overlap
                        ),
                        "outside_coverage_count": len(
                            ids & outside_ids
                        ),
                        "below_overlap_threshold": (
                            True
                        ),
                    }
                )
                source_results.append(result)
                continue

            agreement = agreement_summary(
                metadata,
                current_metadata,
                overlap,
            )

            outside_source_ids = (
                ids & outside_ids
            )

            outside_coverage = field_coverage(
                metadata,
                outside_ids,
            )

            outside_core_complete = sum(
                contract_core_complete(
                    metadata.get(
                        case_id,
                        {},
                    )
                )
                for case_id in outside_ids
            )

            agreement_fraction = agreement[
                "overall_agreement"
            ]

            verified = bool(
                agreement[
                    "total_comparisons"
                ]
                >= MIN_SOURCE_OVERLAP
                and agreement_fraction
                is not None
                and agreement_fraction
                >= MIN_VERIFIED_AGREEMENT
            )

            result.update(
                {
                    "parse_status": "pass",
                    "row_count": len(rows),
                    "unique_case_count": len(ids),
                    "current_overlap_count": len(
                        overlap
                    ),
                    "outside_coverage_count": len(
                        outside_source_ids
                    ),
                    "metadata_conflict_count": (
                        conflict_count
                    ),
                    "agreement": agreement,
                    "outside_field_coverage": (
                        outside_coverage
                    ),
                    "outside_contract_core_complete_count": (
                        outside_core_complete
                    ),
                    "verified_against_current": (
                        verified
                    ),
                }
            )

            if verified:
                source_metadata_by_path[
                    str(source_path)
                ] = metadata

        except Exception as error:
            result.update(
                {
                    "parse_status": "fail",
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

        source_results.append(result)

    ranked_results = sorted(
        (
            result
            for result in source_results
            if result.get(
                "verified_against_current"
            )
        ),
        key=lambda result: (
            int(
                result.get(
                    "outside_contract_core_complete_count",
                    0,
                )
            ),
            int(
                result.get(
                    "outside_coverage_count",
                    0,
                )
            ),
            float(
                result.get(
                    "agreement",
                    {},
                ).get(
                    "overall_agreement",
                    0.0,
                )
                or 0.0
            ),
            str(result["path"]),
        ),
        reverse=True,
    )

    for result in ranked_results[:20]:
        print(
            "VERIFIED_METADATA_SOURCE "
            f"path={result['path']} "
            f"overlap={result['current_overlap_count']} "
            f"outside={result['outside_coverage_count']} "
            f"core_complete="
            f"{result['outside_contract_core_complete_count']} "
            f"agreement="
            f"{result['agreement']['overall_agreement']} "
            f"conflicts="
            f"{result['metadata_conflict_count']}"
        )

    print(
        "METADATA_SOURCE_SCAN_END "
        f"verified_sources={len(ranked_results)}"
    )

    merged_metadata: dict[
        str,
        dict[str, object],
    ] = defaultdict(dict)

    merge_conflicts: list[
        dict[str, Any]
    ] = []

    for case_id in sorted(outside_ids):
        for canonical in CANONICAL_ALIASES:
            candidates: dict[
                str,
                list[tuple[str, object]],
            ] = defaultdict(list)

            for result in ranked_results:
                source_path = str(
                    result["path"]
                )
                metadata = (
                    source_metadata_by_path[
                        source_path
                    ]
                )

                value = metadata.get(
                    case_id,
                    {},
                ).get(canonical)

                normalized = normalize_metadata_value(
                    canonical,
                    value,
                )

                if normalized is not None:
                    candidates[
                        normalized
                    ].append(
                        (
                            source_path,
                            value,
                        )
                    )

            if len(candidates) == 1:
                only_group = next(
                    iter(candidates.values())
                )
                merged_metadata[
                    case_id
                ][canonical] = only_group[0][1]

            elif len(candidates) > 1:
                merge_conflicts.append(
                    {
                        "case_id": case_id,
                        "field": canonical,
                        "normalized_values": (
                            sorted(candidates)
                        ),
                        "sources": {
                            normalized: [
                                source_path
                                for source_path, _
                                in values
                            ][:20]
                            for normalized, values
                            in candidates.items()
                        },
                    }
                )

    merged_core_complete_ids = {
        case_id
        for case_id in outside_ids
        if contract_core_complete(
            merged_metadata.get(
                case_id,
                {},
            )
        )
    }

    enriched_rows = []

    for case_id in sorted(full_ids):
        row = redacted_row(
            full_raw[case_id]
        )

        if case_id in outside_ids:
            metadata = merged_metadata.get(
                case_id,
                {},
            )

            for canonical, value in (
                metadata.items()
            ):
                for raw_field in RAW_FILL_FIELDS[
                    canonical
                ]:
                    if not nonempty(
                        row.get(raw_field)
                    ):
                        row[raw_field] = value

        enriched_rows.append(row)

    enriched_path = (
        args.temporary_directory
        / "full_context_enriched_redacted.jsonl"
    )

    write_jsonl(
        enriched_path,
        enriched_rows,
    )

    enriched_cases = evalmod.load_real_cohort(
        enriched_path,
        allow_demo_data=False,
    )

    enriched_case_map = {
        str(case["case_id"]): case
        for case in enriched_cases
    }

    enriched_outside_contract_counts: Counter[
        str
    ] = Counter()

    enriched_active_target_contract_count = 0

    for case_id in sorted(outside_ids):
        case = enriched_case_map[case_id]
        contract = resolve_contract(
            case["context"]
        )
        contract_id = str(
            contract.contract_id
        )

        enriched_outside_contract_counts[
            contract_id
        ] += 1

        activation = trainer.layer6_activation(
            case
        )

        if (
            bool(activation["active"])
            and contract_id
            == str(target_contract_id)
        ):
            enriched_active_target_contract_count += 1

    best_source = (
        ranked_results[0]
        if ranked_results
        else None
    )

    if (
        enriched_active_target_contract_count
        >= MIN_BRIDGE_CASES
        and not merge_conflicts
    ):
        decision = (
            "verified_context_bridge_ready_for_"
            "target_redacted_feature_reconstruction"
        )
    elif (
        enriched_active_target_contract_count
        >= MIN_BRIDGE_CASES
    ):
        decision = (
            "context_bridge_reaches_required_count_"
            "but_metadata_conflicts_must_be_resolved"
        )
    elif ranked_results:
        decision = (
            "verified_metadata_sources_found_but_"
            "insufficient_contract_reconstruction"
        )
    else:
        decision = (
            "no_verified_context_bridge_found_"
            "seek_raw_clinical_manifest_or_external_cohort"
        )

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "holdout_context_bridge_diagnostic"
        ),
        "inputs": {
            "hashes": actual_hashes,
            "current_case_count": len(
                current_ids
            ),
            "full_case_count": len(
                full_ids
            ),
            "outside_case_count": len(
                outside_ids
            ),
        },
        "failure_diagnosis": {
            "target_contract_id": str(
                target_contract_id
            ),
            "current_contract_counts": dict(
                sorted(
                    current_contract_counts.items()
                )
            ),
            "outside_contract_counts": dict(
                sorted(
                    outside_contract_counts.items()
                )
            ),
            "outside_active_contract_counts": dict(
                sorted(
                    outside_active_contract_counts.items()
                )
            ),
            "outside_active_count": (
                outside_active_count
            ),
            "outside_active_target_contract_count": (
                outside_target_contract_active_count
            ),
            "context_failure_explains_all": (
                context_failure_explains_all
            ),
            "interpretation": (
                "The calibrated Layer 4 candidate is generated "
                "only after a case passes the frozen TNBC "
                "chemotherapy parameter-contract gate."
            ),
        },
        "raw_context_coverage": {
            "current": current_field_coverage,
            "outside": outside_field_coverage,
            "current_metadata_conflicts": (
                current_conflicts
            ),
            "full_metadata_conflicts": (
                full_conflicts
            ),
        },
        "metadata_source_scan": {
            "source_count": len(
                source_paths
            ),
            "verified_source_count": len(
                ranked_results
            ),
            "minimum_overlap": (
                MIN_SOURCE_OVERLAP
            ),
            "minimum_agreement": (
                MIN_VERIFIED_AGREEMENT
            ),
            "ranked_verified_sources": (
                ranked_results[:100]
            ),
            "all_source_results": (
                source_results
            ),
        },
        "verified_union": {
            "outside_contract_core_complete_count": (
                len(
                    merged_core_complete_ids
                )
            ),
            "metadata_conflict_count": len(
                merge_conflicts
            ),
            "metadata_conflict_sample": (
                merge_conflicts[:100]
            ),
            "enriched_outside_contract_counts": dict(
                sorted(
                    enriched_outside_contract_counts.items()
                )
            ),
            "enriched_active_target_contract_count": (
                enriched_active_target_contract_count
            ),
        },
        "best_source": (
            {
                "path": best_source["path"],
                "agreement": (
                    best_source[
                        "agreement"
                    ]["overall_agreement"]
                ),
                "current_overlap_count": (
                    best_source[
                        "current_overlap_count"
                    ]
                ),
                "outside_coverage_count": (
                    best_source[
                        "outside_coverage_count"
                    ]
                ),
                "outside_contract_core_complete_count": (
                    best_source[
                        "outside_contract_core_complete_count"
                    ]
                ),
            }
            if best_source is not None
            else None
        ),
        "guards": {
            "real_holdout_target_values_logged": False,
            "predictions_generated": False,
            "performance_computed": False,
            "manifest_written": False,
            "temporary_enriched_cohort_persisted": False,
        },
        "decision": {
            "status": decision,
            "next_step": (
                "Build a versioned, provenance-preserving "
                "target-redacted enriched cohort and rerun the "
                "exact Layer 6 feature pipeline."
                if enriched_active_target_contract_count
                >= MIN_BRIDGE_CASES
                else (
                    "Locate a validated I-SPY2 clinical metadata "
                    "manifest or move to an external cohort."
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
            "# Layer 8 holdout context-bridge audit",
            "",
            (
                "- Outside-development cases: "
                f"`{len(outside_ids)}`"
            ),
            (
                "- Layer 6-active outside cases: "
                f"`{outside_active_count}`"
            ),
            (
                "- Active outside cases already in frozen "
                "TNBC-chemo scope: "
                f"`{outside_target_contract_active_count}`"
            ),
            (
                "- Verified metadata sources: "
                f"`{len(ranked_results)}`"
            ),
            (
                "- Verified-union contract-complete cases: "
                f"`{len(merged_core_complete_ids)}`"
            ),
            (
                "- Enriched active TNBC-chemo cases: "
                f"`{enriched_active_target_contract_count}`"
            ),
            (
                "- Metadata merge conflicts: "
                f"`{len(merge_conflicts)}`"
            ),
            (
                "- Best source: "
                f"`{best_source['path'] if best_source else None}`"
            ),
            f"- Decision: `{decision}`",
            "",
            "## Safety boundary",
            "",
            (
                "Only target-redacted cohort copies were passed "
                "through contract resolution. No predictions or "
                "performance metrics were generated."
            ),
        ]
    ) + "\n"

    atomic_write_text(
        args.audit_markdown,
        markdown,
    )

    print(
        "VERIFIED_METADATA_UNION "
        f"core_complete="
        f"{len(merged_core_complete_ids)} "
        f"conflicts={len(merge_conflicts)} "
        f"enriched_active_target_contract="
        f"{enriched_active_target_contract_count} "
        f"contract_counts="
        f"{dict(sorted(enriched_outside_contract_counts.items()))}"
    )

    print(
        "METADATA_BRIDGE_SELECTION "
        f"best_source="
        f"{best_source['path'] if best_source else None} "
        f"best_source_core_complete="
        f"{best_source['outside_contract_core_complete_count'] if best_source else None} "
        f"verified_source_count="
        f"{len(ranked_results)}"
    )

    print(
        "CONTEXT_BRIDGE_DECISION "
        f"status={decision}"
    )

    print(
        "TARGET_ACCESS_GUARD "
        "real_holdout_target_values_logged=false "
        "predictions_generated=false "
        "performance_computed=false "
        "manifest_written=false "
        "temporary_enriched_cohort_persisted=false"
    )

    print(
        "CONTEXT_BRIDGE_AUDIT_JSON "
        f"path={args.audit_json}"
    )
    print(
        "CONTEXT_BRIDGE_AUDIT_REPORT "
        f"path={args.audit_markdown}"
    )


if __name__ == "__main__":
    main()
