#!/usr/bin/env python3
"""Inspect exact Layer 8 contract fields and focused clinical sources."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import inspect
import json
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
    "oncotwin_layer8_contract_schema_audit_v0_1"
)

EXPECTED_HASHES = {
    "current_cohort": (
        "8cd51c7f00c073a3d7894692a25bcd8c04384c201c8c3989e91cf5b832c664c0"
    ),
    "full_cohort": (
        "ef7533e1e3af12dc54da284f7df4bce13cd9dd7e895efe291f69aca575ca5fda"
    ),
    "fast_audit": (
        "f58513c2680fb055e75070004e35afe119730ddc1bdedaf2c454dc56deb88a86"
    ),
    "fast_script": (
        "dad0f8b83f114ad56da8f9377e1437f6f724f87f7eb3566904e239e6d9bad34e"
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

CLINICAL_KEY_PATTERN = re.compile(
    r"(subtype|tnbc|triple|treatment|regimen|therapy|"
    r"chemo|neoadjuvant|arm|estrogen|progesterone|"
    r"receptor|her2|er_status|pr_status|hr_status|"
    r"clinical|disease|indication)",
    re.IGNORECASE,
)

CLINICAL_TEXT_PATTERN = re.compile(
    r"(tnbc|triple[\s_-]*negative|chemotherap|"
    r"neoadjuvant|paclitaxel|doxorubicin|"
    r"cyclophosphamide|her2|estrogen receptor|"
    r"progesterone receptor)",
    re.IGNORECASE,
)

SUPPORTED_SUFFIXES = {
    ".jsonl",
    ".json",
    ".csv",
    ".tsv",
}

MAX_FILE_BYTES = 75_000_000
MAX_FILES = 400
MAX_ROWS_PER_SOURCE = 2000
MAX_TEXT_SAMPLE_BYTES = 2_000_000


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
        "--fast-audit",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fast-script",
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


def load_jsonl(
    path: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            value = json.loads(line)

            if isinstance(value, Mapping):
                rows.append(dict(value))

            if (
                limit is not None
                and len(rows) >= limit
            ):
                break

    return rows


def load_json_rows(
    path: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(value, list):
        candidates = value
    elif isinstance(value, Mapping):
        candidates = []

        for key in (
            "rows",
            "records",
            "cases",
            "patients",
            "cohort",
            "data",
            "features",
        ):
            nested = value.get(key)

            if isinstance(nested, list):
                candidates = nested
                break
    else:
        candidates = []

    output = []

    for candidate in candidates:
        if isinstance(candidate, Mapping):
            output.append(dict(candidate))

        if (
            limit is not None
            and len(output) >= limit
        ):
            break

    return output


def load_delimited(
    path: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    delimiter = (
        "\t"
        if path.suffix.lower() == ".tsv"
        else ","
    )

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter=delimiter,
        )

        for row in reader:
            rows.append(dict(row))

            if (
                limit is not None
                and len(rows) >= limit
            ):
                break

    return rows


def load_rows(
    path: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return load_jsonl(
            path,
            limit=limit,
        )

    if suffix == ".json":
        return load_json_rows(
            path,
            limit=limit,
        )

    if suffix in {
        ".csv",
        ".tsv",
    }:
        return load_delimited(
            path,
            limit=limit,
        )

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
                f"{name} row missing case ID"
            )

        if case_id in indexed:
            raise RuntimeError(
                f"{name} duplicate case ID: {case_id}"
            )

        indexed[case_id] = dict(row)

    return indexed


def flatten_value(
    value: object,
    *,
    prefix: str = "",
    output: dict[str, object] | None = None,
    depth: int = 0,
) -> dict[str, object]:
    if output is None:
        output = {}

    if depth > 8:
        return output

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = (
                f"{prefix}.{key_text}"
                if prefix
                else key_text
            )

            flatten_value(
                nested,
                prefix=path,
                output=output,
                depth=depth + 1,
            )

    elif isinstance(value, list):
        if len(value) <= 20:
            for index, nested in enumerate(value):
                path = (
                    f"{prefix}[{index}]"
                    if prefix
                    else f"[{index}]"
                )

                flatten_value(
                    nested,
                    prefix=path,
                    output=output,
                    depth=depth + 1,
                )

    else:
        output[prefix] = value

    return output


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


def summarize_flat_fields(
    rows_by_id: Mapping[
        str,
        Mapping[str, Any],
    ],
    case_ids: set[str],
) -> dict[str, dict[str, Any]]:
    field_values: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for case_id in case_ids:
        row = rows_by_id[case_id]
        flattened = flatten_value(row)

        for field, value in flattened.items():
            if (
                CLINICAL_KEY_PATTERN.search(field)
                or (
                    nonempty(value)
                    and CLINICAL_TEXT_PATTERN.search(
                        str(value)
                    )
                )
            ):
                if nonempty(value):
                    field_values[field].append(
                        str(value).strip()
                    )

    summary = {}

    for field, values in sorted(
        field_values.items()
    ):
        counts = Counter(values)

        summary[field] = {
            "present_count": len(values),
            "coverage": (
                len(values) / len(case_ids)
                if case_ids
                else 0.0
            ),
            "unique_value_count": len(
                counts
            ),
            "top_values": [
                {
                    "value": value,
                    "count": count,
                }
                for value, count
                in counts.most_common(20)
            ],
        }

    return summary


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


def discover_focused_files() -> list[Path]:
    found = set()

    roots = (
        Path("data/curated"),
        Path("data/processed"),
        Path("data/ispy2_streaming"),
        Path("data/v1_prior_stack"),
    )

    excluded_fragments = (
        "/patients/",
        "/patient_feature_rows/",
        "/backups/",
        "/artifacts/",
    )

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if len(found) >= MAX_FILES:
                break

            try:
                if not path.is_file():
                    continue

                if (
                    path.suffix.lower()
                    not in SUPPORTED_SUFFIXES
                ):
                    continue

                normalized_path = (
                    "/"
                    + str(path).replace(
                        "\\",
                        "/",
                    )
                    + "/"
                )

                if any(
                    fragment
                    in normalized_path
                    for fragment
                    in excluded_fragments
                ):
                    continue

                if path.stat().st_size > MAX_FILE_BYTES:
                    continue

                lowered = str(path).lower()

                if any(
                    token in lowered
                    for token in (
                        "clinical",
                        "metadata",
                        "manifest",
                        "cohort",
                        "patient",
                        "subtype",
                        "response",
                        "outcome",
                        "treatment",
                        "ispy2",
                    )
                ):
                    found.add(path)

            except OSError:
                continue

    return sorted(
        found,
        key=lambda path: str(path),
    )


def focused_source_audit(
    path: Path,
    outside_ids: set[str],
) -> dict[str, Any] | None:
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            text = handle.read(
                MAX_TEXT_SAMPLE_BYTES
            )
    except OSError:
        return None

    keyword_hits = sorted(
        {
            match.group(0).lower()
            for match in CLINICAL_TEXT_PATTERN.finditer(
                text
            )
        }
    )

    key_hits = sorted(
        {
            match.group(0).lower()
            for match in CLINICAL_KEY_PATTERN.finditer(
                text
            )
        }
    )

    if not keyword_hits and not key_hits:
        return None

    try:
        rows = load_rows(
            path,
            limit=MAX_ROWS_PER_SOURCE,
        )
    except Exception as error:
        return {
            "path": str(path),
            "status": "parse_failed",
            "error": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
            "keyword_hits": keyword_hits,
            "key_hits": key_hits,
        }

    case_ids = {
        case_id
        for row in rows
        if (
            case_id := case_id_from_row(row)
        )
        is not None
    }

    flattened_fields = Counter()

    for row in rows:
        flattened = flatten_value(row)

        for field, value in flattened.items():
            if (
                CLINICAL_KEY_PATTERN.search(field)
                or (
                    nonempty(value)
                    and CLINICAL_TEXT_PATTERN.search(
                        str(value)
                    )
                )
            ):
                flattened_fields[field] += 1

    return {
        "path": str(path),
        "status": "pass",
        "file_bytes": path.stat().st_size,
        "sample_row_count": len(rows),
        "sample_case_count": len(
            case_ids
        ),
        "sample_outside_case_count": len(
            case_ids & outside_ids
        ),
        "keyword_hits": keyword_hits,
        "key_hits": key_hits,
        "clinical_field_counts": dict(
            flattened_fields.most_common(100)
        ),
    }


def safe_source(
    function: object,
) -> str | None:
    try:
        return inspect.getsource(
            function
        )
    except (
        OSError,
        TypeError,
    ):
        return None


def main() -> None:
    args = parse_args()

    hashes = {
        "current_cohort": sha256_file(
            args.current_cohort
        ),
        "full_cohort": sha256_file(
            args.full_cohort
        ),
        "fast_audit": sha256_file(
            args.fast_audit
        ),
        "fast_script": sha256_file(
            args.fast_script
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

    predecessor = load_json(
        args.fast_audit
    )

    if predecessor["decision"]["status"] != (
        "verified_metadata_sources_found_but_"
        "fewer_than_100_contract_eligible_cases"
    ):
        raise RuntimeError(
            "Unexpected predecessor decision"
        )

    trainer = load_module(
        "_oncotwin_contract_schema_trainer",
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

    if resolve_contract is None:
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
            "Unexpected cohort counts"
        )

    redacted_path = (
        args.temporary_directory
        / "contract_schema_full_redacted.jsonl"
    )

    write_jsonl(
        redacted_path,
        [
            redacted_row(row)
            for row in full_rows
        ],
    )

    loaded_cases = evalmod.load_real_cohort(
        redacted_path,
        allow_demo_data=False,
    )

    loaded_by_id = {
        str(case["case_id"]): dict(case)
        for case in loaded_cases
    }

    raw_current_fields = summarize_flat_fields(
        full_raw,
        current_ids,
    )
    raw_outside_fields = summarize_flat_fields(
        full_raw,
        outside_ids,
    )

    context_by_id = {
        case_id: dict(
            loaded_by_id[case_id].get(
                "context",
                {},
            )
        )
        for case_id in full_ids
    }

    context_current_fields = summarize_flat_fields(
        context_by_id,
        current_ids,
    )
    context_outside_fields = summarize_flat_fields(
        context_by_id,
        outside_ids,
    )

    all_field_names = sorted(
        set(raw_current_fields)
        | set(raw_outside_fields)
        | set(context_current_fields)
        | set(context_outside_fields)
    )

    field_gaps = []

    for field in all_field_names:
        raw_current = raw_current_fields.get(
            field,
            {
                "present_count": 0,
                "coverage": 0.0,
                "top_values": [],
            },
        )
        raw_outside = raw_outside_fields.get(
            field,
            {
                "present_count": 0,
                "coverage": 0.0,
                "top_values": [],
            },
        )
        context_current = context_current_fields.get(
            field,
            {
                "present_count": 0,
                "coverage": 0.0,
                "top_values": [],
            },
        )
        context_outside = context_outside_fields.get(
            field,
            {
                "present_count": 0,
                "coverage": 0.0,
                "top_values": [],
            },
        )

        if (
            raw_current["coverage"] >= 0.20
            or raw_outside["coverage"] > 0
            or context_current["coverage"] >= 0.20
            or context_outside["coverage"] > 0
        ):
            field_gaps.append(
                {
                    "field": field,
                    "raw_current": raw_current,
                    "raw_outside": raw_outside,
                    "loaded_context_current": (
                        context_current
                    ),
                    "loaded_context_outside": (
                        context_outside
                    ),
                }
            )

    resolver_source = safe_source(
        resolve_contract
    )

    related_functions = {}

    for name in (
        "load_real_cohort",
        "build_case_context",
        "normalize_context",
        "resolve_parameter_contract",
        "parameter_contract_from_context",
    ):
        function = getattr(
            evalmod,
            name,
            None,
        )

        if function is not None:
            related_functions[name] = (
                safe_source(function)
            )

    print(
        "CONTRACT_RESOLVER_SOURCE "
        f"module={resolve_contract.__module__} "
        f"function={resolve_contract.__name__} "
        f"source_available="
        f"{str(resolver_source is not None).lower()}",
        flush=True,
    )

    for gap in field_gaps:
        print(
            "CONTRACT_CONTEXT_FIELD "
            f"field={gap['field']} "
            f"raw_current="
            f"{gap['raw_current']['present_count']}/270 "
            f"raw_outside="
            f"{gap['raw_outside']['present_count']}/435 "
            f"context_current="
            f"{gap['loaded_context_current']['present_count']}/270 "
            f"context_outside="
            f"{gap['loaded_context_outside']['present_count']}/435 "
            f"current_values="
            f"{gap['loaded_context_current']['top_values'][:5]} "
            f"outside_values="
            f"{gap['loaded_context_outside']['top_values'][:5]}",
            flush=True,
        )

    focused_paths = discover_focused_files()

    print(
        "FOCUSED_CLINICAL_SOURCE_SCAN_BEGIN "
        f"files={len(focused_paths)}",
        flush=True,
    )

    focused_results = []

    for index, path in enumerate(
        focused_paths,
        start=1,
    ):
        result = focused_source_audit(
            path,
            outside_ids,
        )

        if result is not None:
            focused_results.append(result)

            print(
                "FOCUSED_CLINICAL_SOURCE_CANDIDATE "
                f"path={path} "
                f"status={result['status']} "
                f"sample_cases="
                f"{result.get('sample_case_count')} "
                f"outside_cases="
                f"{result.get('sample_outside_case_count')} "
                f"fields="
                f"{list(result.get('clinical_field_counts', {}))[:20]} "
                f"keywords={result.get('keyword_hits')}",
                flush=True,
            )

        if (
            index == 1
            or index % 25 == 0
            or index == len(focused_paths)
        ):
            print(
                "FOCUSED_CLINICAL_SOURCE_PROGRESS "
                f"files={index}/{len(focused_paths)} "
                f"candidates={len(focused_results)}",
                flush=True,
            )

    print(
        "FOCUSED_CLINICAL_SOURCE_SCAN_END "
        f"candidates={len(focused_results)}",
        flush=True,
    )

    outside_context_evidence = [
        gap
        for gap in field_gaps
        if (
            gap[
                "loaded_context_outside"
            ]["present_count"]
            > 0
            and (
                CLINICAL_KEY_PATTERN.search(
                    gap["field"]
                )
            )
        )
    ]

    source_candidates_with_outside = [
        result
        for result in focused_results
        if int(
            result.get(
                "sample_outside_case_count",
                0,
            )
            or 0
        )
        > 0
        and bool(
            result.get(
                "clinical_field_counts"
            )
        )
    ]

    if outside_context_evidence:
        decision = (
            "outside_clinical_context_exists_under_"
            "previously_unhandled_fields"
        )
    elif source_candidates_with_outside:
        decision = (
            "focused_clinical_source_candidates_found_"
            "require_case_level_validation"
        )
    else:
        decision = (
            "no_local_case_level_contract_metadata_found_"
            "same_source_holdout_blocked"
        )

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "exact_contract_schema_and_"
            "focused_source_audit"
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
        "resolver": {
            "module": (
                resolve_contract.__module__
            ),
            "function": (
                resolve_contract.__name__
            ),
            "source": resolver_source,
            "related_function_sources": (
                related_functions
            ),
        },
        "clinical_field_inventory": {
            "raw_current": (
                raw_current_fields
            ),
            "raw_outside": (
                raw_outside_fields
            ),
            "loaded_context_current": (
                context_current_fields
            ),
            "loaded_context_outside": (
                context_outside_fields
            ),
            "field_gaps": field_gaps,
        },
        "focused_source_scan": {
            "scanned_file_count": len(
                focused_paths
            ),
            "candidate_count": len(
                focused_results
            ),
            "candidates_with_outside_cases": len(
                source_candidates_with_outside
            ),
            "results": focused_results,
        },
        "guards": {
            "real_holdout_targets_used": False,
            "predictions_generated": False,
            "performance_computed": False,
            "manifest_written": False,
        },
        "decision": {
            "status": decision,
            "next_step": (
                "Validate the identified field or source against "
                "the 270 known cases before constructing a bridge."
                if decision
                != "no_local_case_level_contract_metadata_found_same_source_holdout_blocked"
                else (
                    "Preserve the internal Layer 8 result and "
                    "seek an external cohort with T0/T1/T2/T3 "
                    "imaging plus subtype and treatment metadata."
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
            "# Layer 8 exact contract-schema audit",
            "",
            (
                "- Current cases: "
                f"`{len(current_ids)}`"
            ),
            (
                "- Outside-development cases: "
                f"`{len(outside_ids)}`"
            ),
            (
                "- Clinical context fields inventoried: "
                f"`{len(field_gaps)}`"
            ),
            (
                "- Focused files scanned: "
                f"`{len(focused_paths)}`"
            ),
            (
                "- Focused source candidates: "
                f"`{len(focused_results)}`"
            ),
            (
                "- Candidates containing outside case IDs: "
                f"`{len(source_candidates_with_outside)}`"
            ),
            f"- Decision: `{decision}`",
            "",
            "No predictions, target metrics, or validation "
            "manifest were generated.",
        ]
    ) + "\n"

    atomic_write_text(
        args.audit_markdown,
        markdown,
    )

    print(
        "CONTRACT_SCHEMA_DECISION "
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
        "CONTRACT_SCHEMA_AUDIT_JSON "
        f"path={args.audit_json}",
        flush=True,
    )
    print(
        "CONTRACT_SCHEMA_AUDIT_REPORT "
        f"path={args.audit_markdown}",
        flush=True,
    )


if __name__ == "__main__":
    main()
