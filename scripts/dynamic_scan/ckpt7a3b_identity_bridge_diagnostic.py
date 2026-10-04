#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


QUARANTINED_BPC = {
    "cBioPortal_files/data_clinical_supp_survival.txt",
    "cBioPortal_files/data_clinical_supp_survival_treatment.txt",
    "clinical_data/cancer_level_dataset_index.csv",
    "clinical_data/cancer_panel_test_level_dataset.csv",
    "clinical_data/patient_level_dataset.csv",
    "clinical_data/regimen_cancer_level_dataset.csv",
}


###############################################################################
# Utilities
###############################################################################


def norm(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def atomic_json(path: Path, payload: Any) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value:
                int(value)
                if isinstance(value, np.integer)
                else float(value)
                if isinstance(value, np.floating)
                else bool(value)
                if isinstance(value, np.bool_)
                else str(value),
        ),
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"empty JSON write: {path}"
        )

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"empty text write: {path}"
        )

    tmp.replace(path)


def atomic_parquet(
    path: Path,
    frame: pd.DataFrame,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_parquet(
        tmp,
        index=False,
    )

    check = pd.read_parquet(
        tmp
    )

    if len(check) != len(frame):
        raise RuntimeError(
            f"Parquet verification failed: {path}"
        )

    tmp.replace(path)


def delimiter_for(path: Path) -> str:

    if path.suffix.lower() == ".csv":
        return ","

    return "\t"


def read_header(path: Path) -> list[str]:

    candidates = [
        delimiter_for(path),
        "," if delimiter_for(path) == "\t" else "\t",
    ]

    best = []

    for sep in candidates:

        try:

            frame = pd.read_csv(
                path,
                sep=sep,
                comment="#",
                dtype=str,
                nrows=0,
                low_memory=False,
            )

        except Exception:
            continue

        columns = list(
            frame.columns
        )

        if len(columns) > len(best):
            best = columns

    return [
        str(column)
        for column in best
    ]


def identifier_column(column: str) -> bool:

    value = norm(
        column
    )

    exact = {
        "patient_id",
        "sample_id",
        "subject_id",
        "record_id",
        "specimen_id",
        "tumor_sample_barcode",
        "normal_sample_barcode",
        "genie_patient_id",
        "genie_sample_id",
        "dmp_patient_id",
        "dmp_sample_id",
        "dmp_id",
        "case_id",
    }

    if value in exact:
        return True

    if "barcode" in value:
        return True

    if (
        value.endswith("_id")
        and any(
            token in value
            for token in (
                "patient",
                "sample",
                "subject",
                "record",
                "specimen",
                "tumor",
                "genie",
                "dmp",
                "case",
            )
        )
    ):
        return True

    return False


def patient_identifier_column(
    column: str,
) -> bool:

    value = norm(
        column
    )

    return (
        "patient"
        in value
        or "subject"
        in value
        or value
        == "record_id"
    )


def sample_identifier_column(
    column: str,
) -> bool:

    value = norm(
        column
    )

    return (
        "sample"
        in value
        or "specimen"
        in value
        or "barcode"
        in value
    )


def normalized_identifier(
    value: Any,
) -> str:

    return str(
        value
    ).strip().upper()


def compact_identifier(
    value: Any,
) -> str:

    return re.sub(
        r"[^A-Z0-9]+",
        "",
        normalized_identifier(
            value
        ),
    )


def prefix_stripped_variants(
    value: Any,
) -> set[str]:

    raw = normalized_identifier(
        value
    )

    variants = {
        raw,
    }

    known_prefixes = (
        "GENIE-MSK-",
        "GENIE_MSK_",
        "MSK-",
        "MSK_",
        "GENIE-",
        "GENIE_",
    )

    for prefix in known_prefixes:

        if raw.startswith(prefix):

            stripped = raw[
                len(prefix):
            ]

            if stripped:
                variants.add(
                    stripped
                )

    return variants


def import_ckpt7a2(repo: Path):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a2_external_semantic_audit.py"
    )

    spec = importlib.util.spec_from_file_location(
        "ckpt7a2_identity_bridge",
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to import CKPT7A2."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


###############################################################################
# Safe BPC aliases
###############################################################################


def build_bpc_msk_aliases(
    repo: Path,
    bpc_root: Path,
) -> tuple[
    set[str],
    set[str],
    set[str],
    list[dict[str, Any]],
    dict[str, str],
]:

    inventory = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a1"
        / "schema_inventory.parquet"
    )

    ckpt7a2 = import_ckpt7a2(
        repo
    )

    guard = ckpt7a2.AccessGuard(
        bpc_root,
        inventory,
    )

    clinical_patient_relative = (
        "cBioPortal_files/"
        "data_clinical_patient.txt"
    )

    clinical_sample_relative = (
        "cBioPortal_files/"
        "data_clinical_sample.txt"
    )

    clinical_patient = guard.read_predictor(
        clinical_patient_relative
    )

    (
        site_map,
        site_audit,
    ) = ckpt7a2.build_site_map(
        clinical_patient
    )

    if (
        site_audit["counts"]
        != {
            "MSK": 529,
            "DFCI": 428,
            "VICC": 173,
        }
    ):
        raise RuntimeError(
            "BPC site partition changed."
        )

    clinical_sample = guard.read_predictor(
        clinical_sample_relative
    )

    patient_col = next(
        (
            column
            for column
            in clinical_sample.columns
            if norm(column)
            == "patient_id"
        ),
        None,
    )

    sample_col = next(
        (
            column
            for column
            in clinical_sample.columns
            if norm(column)
            == "sample_id"
        ),
        None,
    )

    if (
        patient_col is None
        or sample_col is None
    ):
        raise RuntimeError(
            "Unable to resolve BPC clinical sample IDs."
        )

    sample_to_patient = {
        normalized_identifier(sample):
            normalized_identifier(patient)
        for patient, sample
        in zip(
            clinical_sample[
                patient_col
            ],
            clinical_sample[
                sample_col
            ],
        )
        if (
            str(patient).strip()
            and str(sample).strip()
        )
    }

    msk_patients = {
        normalized_identifier(
            patient
        )
        for patient, site
        in site_map.items()
        if site == "MSK"
    }

    msk_samples = {
        normalized_identifier(
            sample
        )
        for sample, patient
        in sample_to_patient.items()
        if patient in msk_patients
    }

    aliases = set(
        msk_patients
    ) | set(
        msk_samples
    )

    alias_sources: dict[
        str,
        set[str],
    ] = defaultdict(
        set
    )

    for value in msk_patients:
        alias_sources[
            value
        ].add(
            "clinical_patient:PATIENT_ID"
        )

    for value in msk_samples:
        alias_sources[
            value
        ].add(
            "clinical_sample:SAMPLE_ID"
        )

    ###########################################################################
    # Inspect only non-outcome BPC files likely to contain crosswalk aliases.
    ###########################################################################

    selected_paths = []

    for _, row in inventory.iterrows():

        relative = str(
            row[
                "relative_path"
            ]
        )

        if bool(
            row[
                "hard_outcome_sensitive"
            ]
        ):
            continue

        lower = relative.lower()

        if not any(
            token in lower
            for token in (
                "clinical_sample",
                "clinical_patient",
                "sequencing",
                "mutation",
                "gene_matrix",
                "specimen",
            )
        ):
            continue

        columns_raw = row[
            "columns"
        ]

        if isinstance(
            columns_raw,
            str,
        ):

            try:
                columns = json.loads(
                    columns_raw
                )
            except Exception:
                columns = []

        else:
            columns = list(
                columns_raw
            )

        id_columns = [
            column
            for column in columns
            if identifier_column(
                column
            )
        ]

        if not id_columns:
            continue

        selected_paths.append(
            (
                relative,
                id_columns,
            )
        )

    audit_rows = []

    for relative, id_columns in selected_paths:

        if relative in QUARANTINED_BPC:
            raise RuntimeError(
                "Outcome quarantine violation."
            )

        try:

            frame = guard.read_predictor(
                relative,
                usecols=id_columns,
            )

        except Exception as exc:

            audit_rows.append(
                {
                    "dataset":
                        "BPC",

                    "relative_path":
                        relative,

                    "status":
                        "READ_FAILED",

                    "error":
                        f"{type(exc).__name__}: {exc}",
                }
            )

            continue

        patient_columns = [
            column
            for column
            in id_columns
            if patient_identifier_column(
                column
            )
        ]

        sample_columns = [
            column
            for column
            in id_columns
            if sample_identifier_column(
                column
            )
        ]

        matched_rows = 0
        aliases_added = 0

        for _, source_row in frame.iterrows():

            row_site_msk = False

            for column in patient_columns:

                value = normalized_identifier(
                    source_row[
                        column
                    ]
                )

                if (
                    value
                    and value
                    in msk_patients
                ):

                    row_site_msk = True
                    break

            if not row_site_msk:

                for column in sample_columns:

                    value = normalized_identifier(
                        source_row[
                            column
                        ]
                    )

                    patient = sample_to_patient.get(
                        value
                    )

                    if (
                        patient is not None
                        and patient in msk_patients
                    ):

                        row_site_msk = True
                        break

            if not row_site_msk:
                continue

            matched_rows += 1

            for column in id_columns:

                value = normalized_identifier(
                    source_row[
                        column
                    ]
                )

                if not value:
                    continue

                aliases.add(
                    value
                )

                alias_sources[
                    value
                ].add(
                    f"{relative}:{column}"
                )

                aliases_added += 1

        audit_rows.append(
            {
                "dataset":
                    "BPC",

                "relative_path":
                    relative,

                "status":
                    "OK",

                "identifier_columns":
                    json.dumps(
                        id_columns
                    ),

                "rows":
                    int(
                        len(
                            frame
                        )
                    ),

                "msk_rows":
                    int(
                        matched_rows
                    ),

                "alias_assignments":
                    int(
                        aliases_added
                    ),
            }
        )

    ###########################################################################
    # Guard assertion.
    ###########################################################################

    accessed = {
        item[
            "relative_path"
        ]
        for item in guard.access_log
    }

    if (
        accessed
        & QUARANTINED_BPC
    ):

        raise RuntimeError(
            "Outcome-sensitive BPC row access occurred."
        )

    alias_source_flat = {
        alias:
            ";".join(
                sorted(
                    sources
                )
            )
        for alias, sources
        in alias_sources.items()
    }

    return (
        msk_patients,
        msk_samples,
        aliases,
        audit_rows,
        alias_source_flat,
    )


###############################################################################
# CHORD identifier inventory
###############################################################################


def collect_chord_identifiers(
    chord_root: Path,
) -> tuple[
    set[str],
    list[dict[str, Any]],
    dict[str, set[str]],
]:

    allowed_suffixes = {
        ".txt",
        ".tsv",
        ".csv",
        ".maf",
    }

    identifiers = set()

    identifier_sources: dict[
        str,
        set[str],
    ] = defaultdict(
        set
    )

    inventory_rows = []

    for path in sorted(
        chord_root.rglob(
            "*"
        )
    ):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in allowed_suffixes
        ):
            continue

        columns = read_header(
            path
        )

        id_columns = [
            column
            for column in columns
            if identifier_column(
                column
            )
        ]

        if not id_columns:
            continue

        try:

            frame = pd.read_csv(
                path,
                sep=delimiter_for(
                    path
                ),
                comment="#",
                dtype=str,
                keep_default_na=False,
                usecols=id_columns,
                low_memory=False,
            )

        except Exception as exc:

            inventory_rows.append(
                {
                    "dataset":
                        "CHORD",

                    "relative_path":
                        str(
                            path.relative_to(
                                chord_root
                            )
                        ),

                    "status":
                        "READ_FAILED",

                    "identifier_columns":
                        json.dumps(
                            id_columns
                        ),

                    "error":
                        f"{type(exc).__name__}: {exc}",
                }
            )

            continue

        unique_count = 0

        for column in id_columns:

            values = (
                frame[
                    column
                ]
                .astype(str)
                .str.strip()
            )

            for value in values:

                normalized = normalized_identifier(
                    value
                )

                if not normalized:
                    continue

                identifiers.add(
                    normalized
                )

                identifier_sources[
                    normalized
                ].add(
                    (
                        str(
                            path.relative_to(
                                chord_root
                            )
                        )
                        + ":"
                        + column
                    )
                )

                unique_count += 1

        inventory_rows.append(
            {
                "dataset":
                    "CHORD",

                "relative_path":
                    str(
                        path.relative_to(
                            chord_root
                        )
                    ),

                "status":
                    "OK",

                "identifier_columns":
                    json.dumps(
                        id_columns
                    ),

                "rows":
                    int(
                        len(
                            frame
                        )
                    ),

                "identifier_assignments":
                    int(
                        unique_count
                    ),
            }
        )

    return (
        identifiers,
        inventory_rows,
        identifier_sources,
    )


###############################################################################
# Overlap diagnostics
###############################################################################


def overlap_rows(
    bpc_aliases: set[str],
    bpc_sources: dict[str, str],
    chord_identifiers: set[str],
    chord_sources: dict[str, set[str]],
) -> pd.DataFrame:

    rows = []

    exact = sorted(
        bpc_aliases
        & chord_identifiers
    )

    for identifier in exact:

        rows.append(
            {
                "match_type":
                    "EXACT",

                "bpc_identifier":
                    identifier,

                "chord_identifier":
                    identifier,

                "bpc_source":
                    bpc_sources.get(
                        identifier,
                        "",
                    ),

                "chord_source":
                    ";".join(
                        sorted(
                            chord_sources.get(
                                identifier,
                                set(),
                            )
                        )
                    ),
            }
        )

    ###########################################################################
    # Known-prefix-stripped diagnostic.
    ###########################################################################

    chord_lookup: dict[
        str,
        set[str],
    ] = defaultdict(
        set
    )

    for value in chord_identifiers:

        chord_lookup[
            value
        ].add(
            value
        )

    seen = {
        (
            row[
                "bpc_identifier"
            ],
            row[
                "chord_identifier"
            ],
        )
        for row in rows
    }

    for bpc_value in sorted(
        bpc_aliases
    ):

        variants = (
            prefix_stripped_variants(
                bpc_value
            )
            - {
                bpc_value,
            }
        )

        for variant in variants:

            if variant not in chord_lookup:
                continue

            for chord_value in chord_lookup[
                variant
            ]:

                key = (
                    bpc_value,
                    chord_value,
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                rows.append(
                    {
                        "match_type":
                            "KNOWN_PREFIX_STRIPPED",

                        "bpc_identifier":
                            bpc_value,

                        "chord_identifier":
                            chord_value,

                        "derived_variant":
                            variant,

                        "bpc_source":
                            bpc_sources.get(
                                bpc_value,
                                "",
                            ),

                        "chord_source":
                            ";".join(
                                sorted(
                                    chord_sources.get(
                                        chord_value,
                                        set(),
                                    )
                                )
                            ),
                    }
                )

    ###########################################################################
    # Compact diagnostic — hints only, never automatically accepted.
    ###########################################################################

    compact_chord: dict[
        str,
        set[str],
    ] = defaultdict(
        set
    )

    for value in chord_identifiers:

        compact = compact_identifier(
            value
        )

        if (
            len(
                compact
            )
            >= 8
            and any(
                char.isdigit()
                for char in compact
            )
            and any(
                char.isalpha()
                for char in compact
            )
        ):

            compact_chord[
                compact
            ].add(
                value
            )

    for bpc_value in sorted(
        bpc_aliases
    ):

        compact = compact_identifier(
            bpc_value
        )

        if (
            len(
                compact
            )
            < 8
            or not any(
                char.isdigit()
                for char in compact
            )
            or not any(
                char.isalpha()
                for char in compact
            )
        ):
            continue

        for chord_value in compact_chord.get(
            compact,
            set(),
        ):

            key = (
                bpc_value,
                chord_value,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            rows.append(
                {
                    "match_type":
                        "PUNCTUATION_NORMALIZED_HINT",

                    "bpc_identifier":
                        bpc_value,

                    "chord_identifier":
                        chord_value,

                    "derived_variant":
                        compact,

                    "bpc_source":
                        bpc_sources.get(
                            bpc_value,
                            "",
                        ),

                    "chord_source":
                        ";".join(
                            sorted(
                                chord_sources.get(
                                    chord_value,
                                    set(),
                                )
                            )
                        ),
                }
            )

    columns = [
        "match_type",
        "bpc_identifier",
        "chord_identifier",
        "derived_variant",
        "bpc_source",
        "chord_source",
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


###############################################################################
# Namespace diagnostics
###############################################################################


def namespace_signature(
    value: str,
) -> str:

    text = normalized_identifier(
        value
    )

    text = re.sub(
        r"[0-9]+",
        "#",
        text,
    )

    text = re.sub(
        r"#+",
        "#",
        text,
    )

    return text[
        :80
    ]


def top_namespaces(
    values: set[str],
    n: int = 25,
) -> dict[str, int]:

    counts = Counter(
        namespace_signature(
            value
        )
        for value in values
    )

    return {
        key:
            int(value)
        for key, value
        in counts.most_common(
            n
        )
    }


###############################################################################
# Main
###############################################################################


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--bpc-root",
        default=(
            "data/external_sources/"
            "bpc_brca_1_0_public"
        ),
    )

    parser.add_argument(
        "--chord-root",
        default=(
            "data/external_sources/"
            "msk_chord_2024/raw"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/checkpoint7a3b"
        ),
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    bpc_root = (
        repo
        / args.bpc_root
    ).resolve()

    chord_root = (
        repo
        / args.chord_root
    ).resolve()

    out = (
        repo
        / args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # BPC-MSK aliases.
    ###########################################################################

    (
        bpc_patients,
        bpc_samples,
        bpc_aliases,
        bpc_inventory,
        bpc_sources,
    ) = build_bpc_msk_aliases(
        repo,
        bpc_root,
    )

    ###########################################################################
    # CHORD aliases.
    ###########################################################################

    (
        chord_identifiers,
        chord_inventory,
        chord_sources,
    ) = collect_chord_identifiers(
        chord_root
    )

    ###########################################################################
    # Direct subtype overlaps.
    ###########################################################################

    patient_exact = (
        bpc_patients
        & chord_identifiers
    )

    sample_exact = (
        bpc_samples
        & chord_identifiers
    )

    all_exact = (
        bpc_aliases
        & chord_identifiers
    )

    matches = overlap_rows(
        bpc_aliases,
        bpc_sources,
        chord_identifiers,
        chord_sources,
    )

    exact_matches = matches[
        matches[
            "match_type"
        ]
        == "EXACT"
    ]

    prefix_matches = matches[
        matches[
            "match_type"
        ]
        == "KNOWN_PREFIX_STRIPPED"
    ]

    compact_matches = matches[
        matches[
            "match_type"
        ]
        == "PUNCTUATION_NORMALIZED_HINT"
    ]

    ###########################################################################
    # Determine only whether a real bridge is discoverable.
    #
    # Compact matches are intentionally NOT sufficient to declare a bridge.
    ###########################################################################

    if len(
        patient_exact
    ) > 0:

        status = (
            "EXACT_PATIENT_IDENTITY_BRIDGE_FOUND"
        )

    elif len(
        sample_exact
    ) > 0:

        status = (
            "EXACT_SAMPLE_IDENTITY_BRIDGE_FOUND"
        )

    elif len(
        exact_matches
    ) > 0:

        status = (
            "EXACT_ALIAS_IDENTITY_BRIDGE_FOUND"
        )

    elif len(
        prefix_matches
    ) >= 10:

        status = (
            "KNOWN_PREFIX_BRIDGE_CANDIDATE"
        )

    else:

        status = (
            "NO_PUBLIC_IDENTITY_BRIDGE_FOUND"
        )

    ###########################################################################
    # Inventories.
    ###########################################################################

    inventory = pd.DataFrame(
        bpc_inventory
        + chord_inventory
    )

    atomic_parquet(
        out
        / "identifier_column_inventory.parquet",
        inventory,
    )

    atomic_parquet(
        out
        / "identifier_overlap_candidates.parquet",
        matches,
    )

    ###########################################################################
    # Report.
    ###########################################################################

    chord_genie_sources = sorted(
        {
            source
            for identifier, sources
            in chord_sources.items()
            for source in sources
            if "genie" in source.lower()
        }
    )

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "bpc_msk": {
            "patients":
                int(
                    len(
                        bpc_patients
                    )
                ),

            "clinical_samples":
                int(
                    len(
                        bpc_samples
                    )
                ),

            "all_identifier_aliases":
                int(
                    len(
                        bpc_aliases
                    )
                ),

            "top_namespaces":
                top_namespaces(
                    bpc_aliases
                ),
        },

        "chord": {
            "all_identifier_aliases":
                int(
                    len(
                        chord_identifiers
                    )
                ),

            "top_namespaces":
                top_namespaces(
                    chord_identifiers
                ),

            "identifier_columns_with_genie_in_source":
                chord_genie_sources,
        },

        "overlap": {
            "exact_patient_ids":
                int(
                    len(
                        patient_exact
                    )
                ),

            "exact_bpc_sample_ids":
                int(
                    len(
                        sample_exact
                    )
                ),

            "exact_any_aliases":
                int(
                    len(
                        all_exact
                    )
                ),

            "known_prefix_stripped_candidates":
                int(
                    len(
                        prefix_matches
                    )
                ),

            "punctuation_normalized_hints":
                int(
                    len(
                        compact_matches
                    )
                ),
        },

        "policy": {
            "compact_matches_used_as_identity":
                False,

            "prefix_matches_used_as_identity":
                False,

            "outcome_tables_accessed":
                False,
        },

        "next_action": (
            "Repair CKPT7A3 using the exact discovered identity bridge."
            if status.startswith(
                "EXACT_"
            )
            else (
                "Inspect one-to-one consistency of the known-prefix bridge "
                "before using it."
                if status
                == "KNOWN_PREFIX_BRIDGE_CANDIDATE"
                else (
                    "Abandon patient-level BPC-MSK/CHORD positive-control "
                    "matching. Replace CKPT7A3 with schema/semantic and "
                    "distributional development checks, then proceed to "
                    "outcome-blinded DFCI/VICC tensor generation."
                )
            )
        ),
    }

    atomic_json(
        out
        / "identity_bridge.json",
        report,
    )

    ###########################################################################
    # Audit markdown.
    ###########################################################################

    text = f"""# CKPT7A3B — BPC-MSK / CHORD identity bridge diagnostic

Status: **{status}**

External outcomes opened: **NO**

## BPC-MSK

Patients: {len(bpc_patients)}
Clinical samples: {len(bpc_samples)}
Identifier aliases: {len(bpc_aliases)}

## CHORD

Identifier aliases: {len(chord_identifiers)}

## Exact overlap

Patient IDs: {len(patient_exact)}
BPC sample IDs: {len(sample_exact)}
Any BPC-MSK identifier alias: {len(all_exact)}

## Diagnostic-only transformations

Known-prefix-stripped candidates: {len(prefix_matches)}
Punctuation-normalized hints: {len(compact_matches)}

Neither transformed match class is automatically accepted as patient identity.

## Interpretation

{report['next_action']}

## Outcome quarantine

No BPC survival/PFS/death/censoring table values were accessed.
"""

    atomic_text(
        out
        / "audit.md",
        text,
    )

    ###########################################################################
    # Handoff.
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A3B",

        "name":
            "bpc_msk_chord_identity_bridge_diagnostic",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "overlap":
            report[
                "overlap"
            ],

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A3B.json",
        handoff,
    )

    ###########################################################################
    # Terminal output.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A3B SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "bpc_msk_patients="
        f"{len(bpc_patients)}"
    )

    print(
        "bpc_msk_samples="
        f"{len(bpc_samples)}"
    )

    print(
        "bpc_msk_aliases="
        f"{len(bpc_aliases)}"
    )

    print(
        "chord_aliases="
        f"{len(chord_identifiers)}"
    )

    print(
        "exact_patient_overlap="
        f"{len(patient_exact)}"
    )

    print(
        "exact_sample_overlap="
        f"{len(sample_exact)}"
    )

    print(
        "exact_any_alias_overlap="
        f"{len(all_exact)}"
    )

    print(
        "known_prefix_candidates="
        f"{len(prefix_matches)}"
    )

    print(
        "compact_hints="
        f"{len(compact_matches)}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A3B.json"
    )

    print(
        "========== CKPT7A3B SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A3B DECISION PACKET =========="
    )

    print(
        "bpc_top_namespaces="
        f"{report['bpc_msk']['top_namespaces']}"
    )

    print(
        "chord_top_namespaces="
        f"{report['chord']['top_namespaces']}"
    )

    print(
        "chord_genie_identifier_sources="
        f"{report['chord']['identifier_columns_with_genie_in_source']}"
    )

    print(
        "overlap="
        f"{report['overlap']}"
    )

    if len(
        matches
    ):

        print(
            "candidate_match_examples="
            f"{matches.head(30).to_dict(orient='records')}"
        )

    else:

        print(
            "candidate_match_examples=[]"
        )

    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A3B DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
