#!/usr/bin/env python3
"""Bounded Layer 8 clinical-context bridge audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ANALYSIS_VERSION = (
    "oncotwin_layer8_holdout_context_bridge_fast_v0_1"
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

MAX_INDIVIDUAL_FILE_BYTES = 150_000_000
MAX_DIRECTORY_BYTES = 300_000_000

MIN_VERIFICATION_COMPARISONS = 20
MIN_VERIFICATION_AGREEMENT = 0.95
MIN_HOLDOUT_CASES = 100

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
        "molecular_subtype",
        "tumor_subtype",
        "disease_context",
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

ENRICHMENT_FIELDS = {
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
            f"Expected a JSON object: {path}"
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
                    f"{path}:{line_number} is not an object"
                )

            rows.append(dict(value))

    return rows


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(value, list):
        candidates = value

    elif isinstance(value, Mapping):
        candidates = None

        for key in (
            "rows",
            "records",
            "cases",
            "patients",
            "cohort",
            "data",
            "features",
        ):
            possible = value.get(key)

            if isinstance(possible, list):
                candidates = possible
                break

        if candidates is None:
            return []

    else:
        return []

    return [
        dict(row)
        for row in candidates
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
                f"{name} contains a row without case ID"
            )

        if case_id in indexed:
            raise RuntimeError(
                f"{name} contains duplicate case ID: "
                f"{case_id}"
            )

        indexed[case_id] = dict(row)

    return indexed


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


def normalized_key_map(
    row: Mapping[str, Any],
) -> dict[str, str]:
    return {
        re.sub(
            r"[^a-z0-9]",
            "",
            str(key).lower(),
        ): str(key)
        for key in row
    }


def first_alias_value(
    row: Mapping[str, Any],
    canonical: str,
) -> object | None:
    keys = normalized_key_map(row)

    for alias in CANONICAL_ALIASES[canonical]:
        normalized_alias = re.sub(
            r"[^a-z0-9]",
            "",
            alias.lower(),
        )

        source_key = keys.get(
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
            or compact == "ispy"
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
        negative_tokens = {
            "negative",
            "neg",
            "0",
            "false",
        }
        positive_tokens = {
            "positive",
            "pos",
            "1",
            "true",
        }

        words = set(text.split())

        if words & negative_tokens:
            return "negative"

        if words & positive_tokens:
            return "positive"

    return text


def metadata_by_case(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, object]],
    int,
]:
    values: dict[
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
                values[case_id][
                    canonical
                ].append(value)

    resolved: dict[
        str,
        dict[str, object],
    ] = {}

    conflict_count = 0

    for case_id, fields in values.items():
        resolved[case_id] = {}

        for canonical, candidates in fields.items():
            normalized_groups: dict[
                str,
                list[object],
            ] = defaultdict(list)

            for candidate in candidates:
                normalized = normalize_metadata_value(
                    canonical,
                    candidate,
                )

                if normalized is not None:
                    normalized_groups[
                        normalized
                    ].append(candidate)

            if len(normalized_groups) == 1:
                only_values = next(
                    iter(
                        normalized_groups.values()
                    )
                )

                resolved[case_id][
                    canonical
                ] = only_values[0]

            elif len(normalized_groups) > 1:
                conflict_count += 1

        case_metadata = resolved[case_id]

        er = normalize_metadata_value(
            "er_status",
            case_metadata.get("er_status"),
        )
        pr = normalize_metadata_value(
            "pr_status",
            case_metadata.get("pr_status"),
        )
        her2 = normalize_metadata_value(
            "her2_status",
            case_metadata.get("her2_status"),
        )

        if (
            "hr_status" not in case_metadata
            and er == "negative"
            and pr == "negative"
        ):
            case_metadata["hr_status"] = (
                "negative"
            )

        if (
            "subtype" not in case_metadata
            and er == "negative"
            and pr == "negative"
            and her2 == "negative"
        ):
            case_metadata["subtype"] = (
                "tnbc"
            )

    return resolved, conflict_count


def metadata_agreement(
    source: Mapping[
        str,
        Mapping[str, object],
    ],
    reference: Mapping[
        str,
        Mapping[str, object],
    ],
    overlap_ids: set[str],
) -> dict[str, Any]:
    comparisons = 0
    matches = 0
    by_field = {}

    for canonical in CANONICAL_ALIASES:
        field_comparisons = 0
        field_matches = 0

        for case_id in overlap_ids:
            left = normalize_metadata_value(
                canonical,
                source.get(
                    case_id,
                    {},
                ).get(canonical),
            )
            right = normalize_metadata_value(
                canonical,
                reference.get(
                    case_id,
                    {},
                ).get(canonical),
            )

            if left is None or right is None:
                continue

            field_comparisons += 1

            if left == right:
                field_matches += 1

        comparisons += field_comparisons
        matches += field_matches

        by_field[canonical] = {
            "comparisons": field_comparisons,
            "matches": field_matches,
            "agreement": (
                field_matches
                / field_comparisons
                if field_comparisons
                else None
            ),
        }

    return {
        "comparisons": comparisons,
        "matches": matches,
        "agreement": (
            matches / comparisons
            if comparisons
            else None
        ),
        "by_field": by_field,
    }


def contract_core_complete(
    metadata: Mapping[str, object],
) -> bool:
    subtype = normalize_metadata_value(
        "subtype",
        metadata.get("subtype"),
    )
    treatment = normalize_metadata_value(
        "treatment_context",
        metadata.get(
            "treatment_context"
        ),
    )

    return (
        subtype == "tnbc"
        and treatment
        == "neoadjuvant_chemotherapy"
    )


def redacted_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    output = dict(row)

    output["final_volume_ml"] = 1.0

    for alias in (
        "heldout_volume_ml",
        "outcome_volume_ml",
    ):
        if alias in output:
            output[alias] = 1.0

    return output


def bounded_sources() -> list[Path]:
    sources: set[Path] = set()

    explicit = (
        Path(
            "data/ispy2_streaming/features/"
            "ispy2_derived_features.jsonl"
        ),
        Path(
            "data/ispy2_streaming/features/"
            "patients"
        ),
        Path(
            "data/ispy2_streaming/backups/"
            "before_map_stack_full_reprocess_"
            "20260710_224702/"
            "patient_feature_rows"
        ),
    )

    for path in explicit:
        if path.exists():
            sources.add(path)

    name_tokens = (
        "clinical",
        "metadata",
        "manifest",
        "patient",
        "cohort",
        "feature",
        "map",
        "stack",
    )

    for root in (
        Path("data/curated/v1_prior_stack"),
        Path("data/processed/v1_prior_stack"),
        Path("data/ispy2_streaming/features"),
    ):
        if not root.exists():
            continue

        candidates = list(
            root.glob("*")
        ) + list(
            root.glob("*/*")
        )

        for path in candidates:
            try:
                if not path.is_file():
                    continue

                if (
                    path.suffix.lower()
                    not in SUPPORTED_SUFFIXES
                ):
                    continue

                if (
                    path.stat().st_size
                    > MAX_INDIVIDUAL_FILE_BYTES
                ):
                    continue

                lowered = path.name.lower()

                if any(
                    token in lowered
                    for token in name_tokens
                ):
                    sources.add(path)

            except OSError:
                continue

    return sorted(
        sources,
        key=lambda value: str(value),
    )


def load_source(
    path: Path,
) -> list[dict[str, Any]]:
    if path.is_file():
        print(
            "FAST_SOURCE_READ "
            f"path={path} kind=file",
            flush=True,
        )

        return load_rows(path)

    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if (
            candidate.is_file()
            and candidate.suffix.lower()
            in SUPPORTED_SUFFIXES
        )
    )

    total_bytes = 0

    for candidate in files:
        total_bytes += (
            candidate.stat().st_size
        )

    if total_bytes > MAX_DIRECTORY_BYTES:
        raise RuntimeError(
            f"Bounded directory exceeds byte cap: "
            f"{path} bytes={total_bytes}"
        )

    print(
        "FAST_SOURCE_READ "
        f"path={path} kind=directory "
        f"files={len(files)} bytes={total_bytes}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(
        files,
        start=1,
    ):
        rows.extend(
            load_rows(candidate)
        )

        if (
            index == 1
            or index % 100 == 0
            or index == len(files)
        ):
            print(
                "FAST_SOURCE_PROGRESS "
                f"path={path} "
                f"files={index}/{len(files)} "
                f"rows={len(rows)}",
                flush=True,
            )

    return rows


def main() -> None:
    args = parse_args()

    hashes = {
        "current_cohort": sha256_file(
            args.current_cohort
        ),
        "full_cohort": sha256_file(
            args.full_cohort
        ),
        "readiness_audit": sha256_file(
            args.readiness_audit
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
        if hashes[name] != expected:
            raise RuntimeError(
                f"{name} hash mismatch: "
                f"expected={expected} "
                f"actual={hashes[name]}"
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

    trainer = load_module(
        "_oncotwin_context_bridge_fast_trainer",
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
    target_contract = getattr(
        evalmod,
        "TNBC_CHEMO_CONTRACT_ID",
        None,
    )

    if (
        resolve_contract is None
        or target_contract is None
    ):
        raise RuntimeError(
            "Contract resolver is unavailable"
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
            "Unexpected cohort membership counts"
        )

    redacted_path = (
        args.temporary_directory
        / "full_context_redacted.jsonl"
    )

    write_jsonl(
        redacted_path,
        [
            redacted_row(row)
            for row in full_rows
        ],
    )

    cases = evalmod.load_real_cohort(
        redacted_path,
        allow_demo_data=False,
    )

    case_map = {
        str(case["case_id"]): case
        for case in cases
    }

    contract_counts: Counter[str] = (
        Counter()
    )
    active_contract_counts: Counter[str] = (
        Counter()
    )

    active_count = 0
    active_target_contract_count = 0

    for case_id in sorted(outside_ids):
        case = case_map[case_id]

        contract_id = str(
            resolve_contract(
                case["context"]
            ).contract_id
        )

        contract_counts[
            contract_id
        ] += 1

        activation = trainer.layer6_activation(
            case
        )

        if bool(activation["active"]):
            active_count += 1
            active_contract_counts[
                contract_id
            ] += 1

            if contract_id == str(
                target_contract
            ):
                active_target_contract_count += 1

    print(
        "FAST_CONTEXT_DIAGNOSIS "
        f"outside_cases={len(outside_ids)} "
        f"active={active_count} "
        f"target_contract={target_contract} "
        f"active_target_contract="
        f"{active_target_contract_count} "
        f"contract_counts="
        f"{dict(sorted(contract_counts.items()))} "
        f"active_contract_counts="
        f"{dict(sorted(active_contract_counts.items()))}",
        flush=True,
    )

    current_metadata, current_conflicts = (
        metadata_by_case(
            current_rows
        )
    )

    source_paths = bounded_sources()

    print(
        "FAST_SOURCE_SCAN_BEGIN "
        f"sources={len(source_paths)}",
        flush=True,
    )

    source_results = []
    verified_sources = []

    for source_index, source_path in enumerate(
        source_paths,
        start=1,
    ):
        print(
            "FAST_SOURCE_BEGIN "
            f"index={source_index}/{len(source_paths)} "
            f"path={source_path}",
            flush=True,
        )

        result: dict[str, Any] = {
            "path": str(source_path),
            "kind": (
                "directory"
                if source_path.is_dir()
                else "file"
            ),
        }

        try:
            rows = load_source(
                source_path
            )

            metadata, conflicts = (
                metadata_by_case(rows)
            )

            source_ids = set(metadata)
            overlap_ids = (
                source_ids & current_ids
            )
            covered_outside = (
                source_ids & outside_ids
            )

            agreement = metadata_agreement(
                metadata,
                current_metadata,
                overlap_ids,
            )

            core_complete = {
                case_id
                for case_id in outside_ids
                if contract_core_complete(
                    metadata.get(
                        case_id,
                        {},
                    )
                )
            }

            verified = bool(
                agreement["comparisons"]
                >= MIN_VERIFICATION_COMPARISONS
                and agreement["agreement"]
                is not None
                and agreement["agreement"]
                >= MIN_VERIFICATION_AGREEMENT
            )

            result.update(
                {
                    "status": "pass",
                    "row_count": len(rows),
                    "metadata_case_count": len(
                        source_ids
                    ),
                    "current_overlap_count": len(
                        overlap_ids
                    ),
                    "outside_coverage_count": len(
                        covered_outside
                    ),
                    "outside_contract_core_complete_count": len(
                        core_complete
                    ),
                    "metadata_conflict_count": (
                        conflicts
                    ),
                    "agreement": agreement,
                    "verified": verified,
                }
            )

            if verified:
                verified_sources.append(
                    {
                        "path": str(source_path),
                        "metadata": metadata,
                        "result": result,
                    }
                )

                print(
                    "FAST_VERIFIED_SOURCE "
                    f"path={source_path} "
                    f"comparisons="
                    f"{agreement['comparisons']} "
                    f"agreement="
                    f"{agreement['agreement']} "
                    f"outside_coverage="
                    f"{len(covered_outside)} "
                    f"core_complete="
                    f"{len(core_complete)}",
                    flush=True,
                )

        except Exception as error:
            result.update(
                {
                    "status": "fail",
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

        source_results.append(result)

        print(
            "FAST_SOURCE_COMPLETE "
            f"index={source_index}/{len(source_paths)} "
            f"path={source_path} "
            f"status={result['status']} "
            f"verified={result.get('verified', False)}",
            flush=True,
        )

    verified_sources.sort(
        key=lambda item: (
            int(
                item["result"][
                    "outside_contract_core_complete_count"
                ]
            ),
            int(
                item["result"][
                    "outside_coverage_count"
                ]
            ),
            float(
                item["result"][
                    "agreement"
                ]["agreement"]
                or 0.0
            ),
        ),
        reverse=True,
    )

    merged: dict[
        str,
        dict[str, object],
    ] = defaultdict(dict)

    merge_conflicts = []

    for case_id in sorted(outside_ids):
        for canonical in CANONICAL_ALIASES:
            normalized_values: dict[
                str,
                list[tuple[str, object]],
            ] = defaultdict(list)

            for source in verified_sources:
                value = source["metadata"].get(
                    case_id,
                    {},
                ).get(canonical)

                normalized = normalize_metadata_value(
                    canonical,
                    value,
                )

                if normalized is not None:
                    normalized_values[
                        normalized
                    ].append(
                        (
                            source["path"],
                            value,
                        )
                    )

            if len(normalized_values) == 1:
                only_values = next(
                    iter(
                        normalized_values.values()
                    )
                )

                merged[case_id][
                    canonical
                ] = only_values[0][1]

            elif len(normalized_values) > 1:
                merge_conflicts.append(
                    {
                        "case_id": case_id,
                        "field": canonical,
                        "normalized_values": sorted(
                            normalized_values
                        ),
                    }
                )

    union_core_complete = {
        case_id
        for case_id in outside_ids
        if contract_core_complete(
            merged.get(
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
            metadata = merged.get(
                case_id,
                {},
            )

            for canonical, value in metadata.items():
                for field in ENRICHMENT_FIELDS[
                    canonical
                ]:
                    if not nonempty(
                        row.get(field)
                    ):
                        row[field] = value

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

    enriched_map = {
        str(case["case_id"]): case
        for case in enriched_cases
    }

    enriched_contract_counts: Counter[str] = (
        Counter()
    )
    enriched_active_target_count = 0

    for case_id in sorted(outside_ids):
        case = enriched_map[case_id]

        contract_id = str(
            resolve_contract(
                case["context"]
            ).contract_id
        )

        enriched_contract_counts[
            contract_id
        ] += 1

        if (
            bool(
                trainer.layer6_activation(
                    case
                )["active"]
            )
            and contract_id
            == str(target_contract)
        ):
            enriched_active_target_count += 1

    print(
        "FAST_CONTEXT_UNION "
        f"verified_sources="
        f"{len(verified_sources)} "
        f"core_complete="
        f"{len(union_core_complete)} "
        f"merge_conflicts="
        f"{len(merge_conflicts)} "
        f"enriched_active_target_contract="
        f"{enriched_active_target_count} "
        f"enriched_contract_counts="
        f"{dict(sorted(enriched_contract_counts.items()))}",
        flush=True,
    )

    if (
        enriched_active_target_count
        >= MIN_HOLDOUT_CASES
    ):
        decision = (
            "verified_context_bridge_ready_for_"
            "target_redacted_feature_reconstruction"
        )

    elif verified_sources:
        decision = (
            "verified_metadata_sources_found_but_"
            "fewer_than_100_contract_eligible_cases"
        )

    else:
        decision = (
            "no_verified_context_bridge_in_bounded_sources"
        )

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "bounded_holdout_context_bridge_audit"
        ),
        "inputs": {
            "hashes": hashes,
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
        "initial_contract_diagnosis": {
            "target_contract_id": str(
                target_contract
            ),
            "outside_active_count": (
                active_count
            ),
            "outside_active_target_contract_count": (
                active_target_contract_count
            ),
            "contract_counts": dict(
                sorted(
                    contract_counts.items()
                )
            ),
            "active_contract_counts": dict(
                sorted(
                    active_contract_counts.items()
                )
            ),
        },
        "bounded_source_scan": {
            "source_count": len(
                source_paths
            ),
            "source_paths": [
                str(path)
                for path in source_paths
            ],
            "verified_source_count": len(
                verified_sources
            ),
            "results": source_results,
        },
        "verified_union": {
            "contract_core_complete_count": len(
                union_core_complete
            ),
            "merge_conflict_count": len(
                merge_conflicts
            ),
            "merge_conflict_sample": (
                merge_conflicts[:100]
            ),
            "enriched_active_target_contract_count": (
                enriched_active_target_count
            ),
            "enriched_contract_counts": dict(
                sorted(
                    enriched_contract_counts.items()
                )
            ),
        },
        "current_metadata_conflict_count": (
            current_conflicts
        ),
        "guards": {
            "real_holdout_targets_passed_to_contract_resolution": (
                False
            ),
            "predictions_generated": False,
            "performance_computed": False,
            "manifest_written": False,
            "temporary_enriched_cohort_persisted": False,
        },
        "decision": {
            "status": decision,
            "next_step": (
                "Construct a provenance-preserving enriched "
                "target-redacted cohort and rerun exact Layer 6 "
                "feature generation."
                if enriched_active_target_count
                >= MIN_HOLDOUT_CASES
                else (
                    "Locate the raw clinical manifest or move "
                    "to an external validation cohort."
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
            "# Layer 8 bounded context-bridge audit",
            "",
            (
                "- Outside-development cases: "
                f"`{len(outside_ids)}`"
            ),
            (
                "- Layer 6-active outside cases: "
                f"`{active_count}`"
            ),
            (
                "- Initially in target contract: "
                f"`{active_target_contract_count}`"
            ),
            (
                "- Bounded sources inspected: "
                f"`{len(source_paths)}`"
            ),
            (
                "- Verified metadata sources: "
                f"`{len(verified_sources)}`"
            ),
            (
                "- Verified-union contract-complete cases: "
                f"`{len(union_core_complete)}`"
            ),
            (
                "- Enriched active target-contract cases: "
                f"`{enriched_active_target_count}`"
            ),
            (
                "- Metadata merge conflicts: "
                f"`{len(merge_conflicts)}`"
            ),
            f"- Decision: `{decision}`",
            "",
            "No predictions, target metrics, or holdout "
            "manifest were generated.",
        ]
    ) + "\n"

    atomic_write_text(
        args.audit_markdown,
        markdown,
    )

    print(
        "FAST_CONTEXT_BRIDGE_DECISION "
        f"status={decision}",
        flush=True,
    )

    print(
        "TARGET_ACCESS_GUARD "
        "real_holdout_targets_used=false "
        "predictions_generated=false "
        "performance_computed=false "
        "manifest_written=false",
        flush=True,
    )

    print(
        "FAST_CONTEXT_BRIDGE_JSON "
        f"path={args.audit_json}",
        flush=True,
    )

    print(
        "FAST_CONTEXT_BRIDGE_REPORT "
        f"path={args.audit_markdown}",
        flush=True,
    )


if __name__ == "__main__":
    main()
