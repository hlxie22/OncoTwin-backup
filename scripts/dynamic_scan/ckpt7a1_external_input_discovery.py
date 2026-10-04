#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_SITE_COUNTS = {
    "MSK": 529,
    "DFCI": 428,
    "VICC": 173,
}

LOCK_HASHES = {
    "candidate_manifest":
        "0aeddaff5df42e1637884b637f87090e23457062aa3171f516b1f4eeaa3635eb",

    "external_protocol":
        "6e494357438741bbfad8f68ed7a962891c51aed94240e366a4efed42e9e90cfd",

    "protocol_lock":
        "0a56b6067820226414e5a735a63dce42c1a6d404779b1a44cce74dab51791180",
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


def atomic_json(
    path: Path,
    payload: Any,
) -> None:

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
            default=str,
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


def atomic_text(
    path: Path,
    text: str,
) -> None:

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


def sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


###############################################################################
# Delimited-file inspection
###############################################################################


TEXT_SUFFIXES = {
    ".txt",
    ".tsv",
    ".csv",
    ".maf",
}


def delimiter_for(
    path: Path,
) -> str:

    if path.suffix.lower() == ".csv":
        return ","

    return "\t"


def read_header(
    path: Path,
) -> tuple[
    list[str],
    str,
]:

    candidates = []

    preferred = delimiter_for(
        path
    )

    candidates.append(
        preferred
    )

    candidates.append(
        ","
        if preferred == "\t"
        else "\t"
    )

    best_columns = []
    best_sep = preferred

    for separator in candidates:

        try:

            frame = pd.read_csv(
                path,
                sep=separator,
                comment="#",
                dtype=str,
                keep_default_na=False,
                nrows=0,
                low_memory=False,
            )

        except Exception:
            continue

        columns = [
            str(
                column
            ).strip()
            for column
            in frame.columns
        ]

        if len(
            columns
        ) > len(
            best_columns
        ):

            best_columns = columns
            best_sep = separator

    return (
        best_columns,
        best_sep,
    )


def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:

    lookup = {
        norm(
            column
        ):
            column
        for column in columns
    }

    for candidate in candidates:

        key = norm(
            candidate
        )

        if key in lookup:
            return lookup[
                key
            ]

    return None


###############################################################################
# Outcome quarantine
###############################################################################


HARD_OUTCOME_TOKENS = {
    "pfs",
    "progression_free_survival",
    "overall_survival",
    "os_status",
    "os_months",
    "death",
    "deceased",
    "date_of_death",
    "censor",
    "censoring",
    "survival",
    "time_to_event",
    "event_free_survival",
    "treatment_switch_outcome",
    "switch_outcome",
}

SCAN_STATE_TOKENS = {
    "progression",
    "response",
    "scan_state",
    "imaging_response",
    "radiology_response",
    "cancer_presence",
}


def classify_outcome_sensitivity(
    path: Path,
    columns: list[str],
) -> dict[str, Any]:

    normalized_name = norm(
        path.name
    )

    normalized_columns = {
        norm(
            column
        )
        for column in columns
    }

    hard_hits = sorted(
        {
            token
            for token
            in HARD_OUTCOME_TOKENS
            if (
                token
                in normalized_name
                or any(
                    (
                        token
                        == column
                        or token
                        in column
                    )
                    for column
                    in normalized_columns
                )
            )
        }
    )

    scan_state_hits = sorted(
        {
            token
            for token
            in SCAN_STATE_TOKENS
            if (
                token
                in normalized_name
                or any(
                    (
                        token
                        == column
                        or token
                        in column
                    )
                    for column
                    in normalized_columns
                )
            )
        }
    )

    ###########################################################################
    # A table containing ordinary scan progression/response state is NOT
    # automatically hard-quarantined: those fields can be predictors at the
    # landmark. Their values are simply not summarized during CKPT7A1.
    ###########################################################################

    hard_outcome = bool(
        hard_hits
    )

    return {
        "hard_outcome_sensitive":
            hard_outcome,

        "hard_outcome_hits":
            hard_hits,

        "scan_state_sensitive":
            bool(
                scan_state_hits
            ),

        "scan_state_hits":
            scan_state_hits,
    }


###############################################################################
# Site map
###############################################################################


def normalize_site(
    value: Any,
) -> str | None:

    text = str(
        value
    ).strip().upper()

    if not text:
        return None

    if (
        "DFCI"
        in text
        or "DANA"
        in text
    ):
        return "DFCI"

    if (
        "VICC"
        in text
        or "VANDERBILT"
        in text
    ):
        return "VICC"

    if (
        text == "MSK"
        or "MEMORIAL"
        in text
        or "SLOAN"
        in text
    ):
        return "MSK"

    return None


def find_patient_level_file(
    root: Path,
) -> Path:

    preferred = list(
        root.rglob(
            "patient_level_dataset.csv"
        )
    )

    if len(
        preferred
    ) == 1:
        return preferred[
            0
        ]

    candidates = []

    for path in root.rglob(
        "*"
    ):

        if (
            path.is_file()
            and path.suffix.lower()
            in TEXT_SUFFIXES
        ):

            columns, separator = read_header(
                path
            )

            normalized = {
                norm(
                    column
                )
                for column
                in columns
            }

            if (
                (
                    "patient_id"
                    in normalized
                    or "record_id"
                    in normalized
                )
                and any(
                    token
                    in normalized
                    for token
                    in (
                        "institution",
                        "site",
                        "center",
                        "institution_name",
                    )
                )
            ):

                candidates.append(
                    path
                )

    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            "Unable to uniquely resolve BPC patient-level site table. "
            f"candidates={candidates}"
        )

    return candidates[
        0
    ]


def build_site_map(
    path: Path,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, int],
    dict[str, Any],
]:

    columns, separator = read_header(
        path
    )

    id_candidates = [
        column
        for column
        in columns
        if norm(
            column
        )
        in {
            "patient_id",
            "record_id",
            "genie_patient_id",
            "subject_id",
        }
    ]

    site_candidates = [
        column
        for column
        in columns
        if norm(
            column
        )
        in {
            "institution",
            "institution_name",
            "site",
            "center",
            "center_id",
            "hospital",
            "source",
        }
    ]

    if not id_candidates:
        raise RuntimeError(
            "Patient-level site table has no recognized patient identifier."
        )

    if not site_candidates:
        raise RuntimeError(
            "Patient-level site table has no recognized site field."
        )

    frame = pd.read_csv(
        path,
        sep=separator,
        comment="#",
        dtype=str,
        keep_default_na=False,
        usecols=list(
            dict.fromkeys(
                id_candidates
                + site_candidates
            )
        ),
        low_memory=False,
    )

    best_site_column = None
    best_resolved = -1

    for column in site_candidates:

        resolved = (
            frame[
                column
            ]
            .map(
                normalize_site
            )
            .notna()
            .sum()
        )

        if resolved > best_resolved:

            best_resolved = int(
                resolved
            )

            best_site_column = column

    if best_site_column is None:
        raise RuntimeError(
            "Unable to resolve BPC institution field."
        )

    site = frame[
        best_site_column
    ].map(
        normalize_site
    )

    ###########################################################################
    # Choose canonical ID column by uniqueness/resolution, but every recognized
    # ID becomes a safe alias to that canonical patient.
    ###########################################################################

    canonical_column = max(
        id_candidates,
        key=lambda column:
            frame[
                column
            ]
            .astype(str)
            .str.strip()
            .replace(
                "",
                pd.NA,
            )
            .nunique(
                dropna=True
            ),
    )

    alias_to_canonical: dict[
        str,
        str
    ] = {}

    canonical_to_site: dict[
        str,
        str
    ] = {}

    for row_index in range(
        len(
            frame
        )
    ):

        current_site = site.iloc[
            row_index
        ]

        canonical = (
            str(
                frame.iloc[
                    row_index
                ][
                    canonical_column
                ]
            )
            .strip()
        )

        if (
            not canonical
            or current_site is None
        ):
            continue

        canonical_to_site[
            canonical
        ] = current_site

        for column in id_candidates:

            alias = (
                str(
                    frame.iloc[
                        row_index
                    ][
                        column
                    ]
                )
                .strip()
            )

            if alias:

                previous = alias_to_canonical.get(
                    alias
                )

                if (
                    previous is not None
                    and previous
                    != canonical
                ):

                    raise RuntimeError(
                        "Patient alias maps to multiple canonical IDs: "
                        f"{alias}"
                    )

                alias_to_canonical[
                    alias
                ] = canonical

    counts = Counter(
        canonical_to_site.values()
    )

    site_counts = {
        site_name:
            int(
                counts.get(
                    site_name,
                    0
                )
            )
        for site_name
        in (
            "MSK",
            "DFCI",
            "VICC",
        )
    }

    audit = {
        "path":
            str(
                path
            ),

        "id_columns":
            id_candidates,

        "canonical_id_column":
            canonical_column,

        "site_columns_considered":
            site_candidates,

        "selected_site_column":
            best_site_column,

        "canonical_patients":
            int(
                len(
                    canonical_to_site
                )
            ),

        "identifier_aliases":
            int(
                len(
                    alias_to_canonical
                )
            ),

        "site_counts":
            site_counts,
    }

    return (
        alias_to_canonical,
        canonical_to_site,
        site_counts,
        audit,
    )


###############################################################################
# Input-table classification
###############################################################################


DATE_HINTS = (
    "date",
    "day",
    "time",
    "start_date",
    "stop_date",
    "report_date",
    "scan_date",
    "study_date",
    "assessment_date",
    "specimen_date",
    "seq_date",
)

SCAN_HINTS = (
    "scan",
    "imaging",
    "radiology",
    "radiographic",
    "modality",
    "procedure",
    "response",
    "progression",
    "cancer_presence",
    "tumor_site",
    "body_site",
    "lesion",
)

TEMPORAL_HINTS = (
    "treatment",
    "therapy",
    "drug",
    "regimen",
    "lab",
    "marker",
    "cea",
    "ca15",
    "ca_15",
    "ca27",
    "performance",
    "ecog",
    "receptor",
    "er_status",
    "pr_status",
    "her2",
    "pathology",
    "diagnosis",
    "radiation",
    "surgery",
    "specimen",
)

GENOMIC_HINTS = (
    "mutation",
    "maf",
    "gene",
    "panel",
    "sequencing",
    "seq_assay",
    "cna",
    "copy_number",
    "fusion",
    "structural_variant",
    "genomic",
)


def classify_input_surface(
    path: Path,
    columns: list[str],
) -> dict[str, Any]:

    name = norm(
        path.name
    )

    normalized_columns = [
        norm(
            column
        )
        for column
        in columns
    ]

    combined = (
        " "
        + name
        + " "
        + " ".join(
            normalized_columns
        )
        + " "
    )

    date_columns = [
        column
        for column
        in columns
        if any(
            hint
            in norm(
                column
            )
            for hint in DATE_HINTS
        )
    ]

    scan_columns = [
        column
        for column
        in columns
        if any(
            hint
            in norm(
                column
            )
            for hint in SCAN_HINTS
        )
    ]

    temporal_columns = [
        column
        for column
        in columns
        if any(
            hint
            in norm(
                column
            )
            for hint in TEMPORAL_HINTS
        )
    ]

    genomic_columns = [
        column
        for column
        in columns
        if any(
            hint
            in norm(
                column
            )
            for hint in GENOMIC_HINTS
        )
    ]

    scan_name_hit = any(
        hint
        in name
        for hint in SCAN_HINTS
    )

    temporal_name_hit = any(
        hint
        in name
        for hint in TEMPORAL_HINTS
    )

    genomic_name_hit = any(
        hint
        in name
        for hint in GENOMIC_HINTS
    )

    return {
        "date_columns":
            date_columns,

        "scan_columns":
            scan_columns,

        "temporal_columns":
            temporal_columns,

        "genomic_columns":
            genomic_columns,

        "scan_candidate":
            bool(
                scan_columns
                or scan_name_hit
            ),

        "temporal_candidate":
            bool(
                date_columns
                and (
                    scan_columns
                    or temporal_columns
                    or temporal_name_hit
                )
            ),

        "genomic_candidate":
            bool(
                genomic_columns
                or genomic_name_hit
            ),
    }


###############################################################################
# Safe identifier-only table coverage
###############################################################################


PATIENT_ID_CANDIDATES = [
    "PATIENT_ID",
    "patient_id",
    "record_id",
    "GENIE_PATIENT_ID",
    "SUBJECT_ID",
]

SAMPLE_ID_CANDIDATES = [
    "SAMPLE_ID",
    "sample_id",
    "Tumor_Sample_Barcode",
    "SPECIMEN_ID",
]


def build_sample_alias_map(
    inventory: list[dict[str, Any]],
    alias_to_canonical: dict[str, str],
) -> tuple[
    dict[str, str],
    list[dict[str, Any]],
]:

    sample_to_canonical: dict[
        str,
        str
    ] = {}

    sources = []

    for record in inventory:

        if record[
            "hard_outcome_sensitive"
        ]:
            continue

        columns = record[
            "columns"
        ]

        patient_col = find_column(
            columns,
            PATIENT_ID_CANDIDATES,
        )

        sample_col = find_column(
            columns,
            SAMPLE_ID_CANDIDATES,
        )

        if (
            patient_col is None
            or sample_col is None
        ):
            continue

        path = Path(
            record[
                "path"
            ]
        )

        separator = record[
            "separator"
        ]

        try:

            frame = pd.read_csv(
                path,
                sep=separator,
                comment="#",
                dtype=str,
                keep_default_na=False,
                usecols=[
                    patient_col,
                    sample_col,
                ],
                low_memory=False,
            )

        except Exception:
            continue

        mapped = 0

        for patient, sample in zip(
            frame[
                patient_col
            ],
            frame[
                sample_col
            ],
        ):

            patient = str(
                patient
            ).strip()

            sample = str(
                sample
            ).strip()

            if (
                not patient
                or not sample
            ):
                continue

            canonical = (
                alias_to_canonical.get(
                    patient
                )
            )

            if canonical is None:
                continue

            previous = sample_to_canonical.get(
                sample
            )

            if (
                previous is not None
                and previous
                != canonical
            ):

                continue

            sample_to_canonical[
                sample
            ] = canonical

            mapped += 1

        if mapped:

            sources.append(
                {
                    "path":
                        str(
                            path
                        ),

                    "patient_column":
                        patient_col,

                    "sample_column":
                        sample_col,

                    "mapped_rows":
                        int(
                            mapped
                        ),
                }
            )

    return (
        sample_to_canonical,
        sources,
    )


def safe_external_membership(
    record: dict[str, Any],
    alias_to_canonical: dict[str, str],
    canonical_to_site: dict[str, str],
    sample_to_canonical: dict[str, str],
) -> dict[str, Any]:

    if record[
        "hard_outcome_sensitive"
    ]:

        return {
            "row_access":
                "QUARANTINED_HEADER_ONLY",

            "rows_scanned":
                None,

            "external_patients":
                {
                    "DFCI": None,
                    "VICC": None,
                    "MSK": None,
                },

            "external_rows":
                {
                    "DFCI": None,
                    "VICC": None,
                    "MSK": None,
                },
        }

    columns = record[
        "columns"
    ]

    patient_col = find_column(
        columns,
        PATIENT_ID_CANDIDATES,
    )

    sample_col = find_column(
        columns,
        SAMPLE_ID_CANDIDATES,
    )

    use_col = (
        patient_col
        if patient_col is not None
        else sample_col
    )

    identifier_kind = (
        "patient"
        if patient_col is not None
        else "sample"
        if sample_col is not None
        else None
    )

    if use_col is None:

        return {
            "row_access":
                "NO_IDENTIFIER_COLUMN",

            "identifier_kind":
                None,

            "identifier_column":
                None,

            "rows_scanned":
                0,

            "external_patients":
                {
                    "DFCI": 0,
                    "VICC": 0,
                    "MSK": 0,
                },

            "external_rows":
                {
                    "DFCI": 0,
                    "VICC": 0,
                    "MSK": 0,
                },
        }

    patient_sets = {
        "DFCI": set(),
        "VICC": set(),
        "MSK": set(),
    }

    row_counts = {
        "DFCI": 0,
        "VICC": 0,
        "MSK": 0,
    }

    total_rows = 0

    path = Path(
        record[
            "path"
        ]
    )

    try:

        reader = pd.read_csv(
            path,
            sep=record[
                "separator"
            ],
            comment="#",
            dtype=str,
            keep_default_na=False,
            usecols=[
                use_col,
            ],
            chunksize=100000,
            low_memory=False,
        )

        for chunk in reader:

            values = (
                chunk[
                    use_col
                ]
                .astype(str)
                .str.strip()
            )

            total_rows += int(
                len(
                    values
                )
            )

            for value in values:

                if not value:
                    continue

                if identifier_kind == "patient":

                    canonical = (
                        alias_to_canonical.get(
                            value
                        )
                    )

                else:

                    canonical = (
                        sample_to_canonical.get(
                            value
                        )
                    )

                if canonical is None:
                    continue

                site = canonical_to_site.get(
                    canonical
                )

                if site not in patient_sets:
                    continue

                patient_sets[
                    site
                ].add(
                    canonical
                )

                row_counts[
                    site
                ] += 1

    except Exception as exc:

        return {
            "row_access":
                "IDENTIFIER_READ_FAILED",

            "identifier_kind":
                identifier_kind,

            "identifier_column":
                use_col,

            "error":
                f"{type(exc).__name__}: {exc}",

            "rows_scanned":
                0,

            "external_patients":
                {
                    "DFCI": 0,
                    "VICC": 0,
                    "MSK": 0,
                },

            "external_rows":
                {
                    "DFCI": 0,
                    "VICC": 0,
                    "MSK": 0,
                },
        }

    return {
        "row_access":
            "IDENTIFIER_ONLY",

        "identifier_kind":
            identifier_kind,

        "identifier_column":
            use_col,

        "rows_scanned":
            int(
                total_rows
            ),

        "external_patients": {
            site:
                int(
                    len(
                        patients
                    )
                )
            for site, patients
            in patient_sets.items()
        },

        "external_rows": {
            site:
                int(
                    count
                )
            for site, count
            in row_counts.items()
        },
    }


###############################################################################
# Lock audit
###############################################################################


def audit_lock_hashes(
    repo: Path,
) -> dict[str, Any]:

    roots = [
        repo
        / "artifacts"
        / "checkpoint6e",

        repo
        / "artifacts"
        / "handoff",
    ]

    files = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):

            if (
                path.is_file()
                and path.stat().st_size
                > 0
            ):

                files.append(
                    path
                )

    file_hashes = {
        str(
            path.relative_to(
                repo
            )
        ):
            sha256(
                path
            )
        for path in files
    }

    text_hits: dict[
        str,
        list[str],
    ] = {
        label: []
        for label in LOCK_HASHES
    }

    file_hash_hits: dict[
        str,
        list[str],
    ] = {
        label: []
        for label in LOCK_HASHES
    }

    for label, expected in LOCK_HASHES.items():

        for relative, digest in file_hashes.items():

            if digest == expected:

                file_hash_hits[
                    label
                ].append(
                    relative
                )

        for path in files:

            try:

                text = path.read_text(
                    encoding="utf-8"
                )

            except Exception:
                continue

            if expected in text:

                text_hits[
                    label
                ].append(
                    str(
                        path.relative_to(
                            repo
                        )
                    )
                )

    return {
        "expected_hashes":
            LOCK_HASHES,

        "file_hash_hits":
            file_hash_hits,

        "text_reference_hits":
            text_hits,

        "files_hashed":
            len(
                files
            ),
    }


###############################################################################
# Main discovery
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
        "--output-dir",
        default=(
            "artifacts/checkpoint7a1"
        ),
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    root = (
        repo
        / args.bpc_root
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
    # A. Verify site partition from the development metadata only.
    ###########################################################################

    patient_level = find_patient_level_file(
        root
    )

    (
        alias_to_canonical,
        canonical_to_site,
        site_counts,
        site_audit,
    ) = build_site_map(
        patient_level
    )

    if site_counts != EXPECTED_SITE_COUNTS:

        raise RuntimeError(
            "BPC site partition changed unexpectedly: "
            f"observed={site_counts} "
            f"expected={EXPECTED_SITE_COUNTS}"
        )

    print(
        "[CKPT7A1_SITE_PARTITION_PASS]",
        site_counts,
        flush=True,
    )

    ###########################################################################
    # B. Header-only inventory of every text-like BPC source.
    ###########################################################################

    inventory = []

    for path in sorted(
        root.rglob(
            "*"
        )
    ):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in TEXT_SUFFIXES
        ):
            continue

        try:

            columns, separator = read_header(
                path
            )

        except Exception as exc:

            inventory.append(
                {
                    "path":
                        str(
                            path
                        ),

                    "relative_path":
                        str(
                            path.relative_to(
                                root
                            )
                        ),

                    "bytes":
                        int(
                            path.stat().st_size
                        ),

                    "header_error":
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),

                    "columns":
                        [],

                    "separator":
                        None,

                    "hard_outcome_sensitive":
                        True,

                    "hard_outcome_hits":
                        [
                            "UNREADABLE_HEADER",
                        ],

                    "scan_state_sensitive":
                        False,

                    "scan_state_hits":
                        [],

                    "scan_candidate":
                        False,

                    "temporal_candidate":
                        False,

                    "genomic_candidate":
                        False,

                    "date_columns":
                        [],

                    "scan_columns":
                        [],

                    "temporal_columns":
                        [],

                    "genomic_columns":
                        [],
                }
            )

            continue

        outcome = (
            classify_outcome_sensitivity(
                path,
                columns,
            )
        )

        surface = (
            classify_input_surface(
                path,
                columns,
            )
        )

        inventory.append(
            {
                "path":
                    str(
                        path
                    ),

                "relative_path":
                    str(
                        path.relative_to(
                            root
                        )
                    ),

                "bytes":
                    int(
                        path.stat().st_size
                    ),

                "columns":
                    columns,

                "separator":
                    separator,

                **outcome,
                **surface,
            }
        )

    ###########################################################################
    # C. Build safe SAMPLE_ID -> external patient map using input tables only.
    ###########################################################################

    (
        sample_to_canonical,
        sample_map_sources,
    ) = build_sample_alias_map(
        inventory,
        alias_to_canonical,
    )

    ###########################################################################
    # D. Read identifiers ONLY from non-outcome tables.
    #
    # No progression/response/state values are summarized.
    # No death/PFS/survival/censoring table rows are opened.
    ###########################################################################

    enriched = []

    for index, record in enumerate(
        inventory,
        start=1,
    ):

        membership = (
            safe_external_membership(
                record,
                alias_to_canonical,
                canonical_to_site,
                sample_to_canonical,
            )
        )

        enriched_record = {
            **record,
            **membership,
        }

        enriched.append(
            enriched_record
        )

        if (
            index == 1
            or index
            % 20
            == 0
            or index
            == len(
                inventory
            )
        ):

            print(
                "[CKPT7A1_SCHEMA_SCAN]",
                f"{index}/{len(inventory)}",
                flush=True,
            )

    ###########################################################################
    # E. Candidate tables
    ###########################################################################

    def external_count(
        record,
        site,
    ):

        value = (
            record.get(
                "external_patients",
                {}
            ).get(
                site
            )
        )

        return (
            int(
                value
            )
            if value
            is not None
            else 0
        )

    scan_candidates = [
        record
        for record
        in enriched
        if (
            not record[
                "hard_outcome_sensitive"
            ]
            and record[
                "scan_candidate"
            ]
            and (
                external_count(
                    record,
                    "DFCI",
                )
                > 0
                or external_count(
                    record,
                    "VICC",
                )
                > 0
            )
        )
    ]

    temporal_candidates = [
        record
        for record
        in enriched
        if (
            not record[
                "hard_outcome_sensitive"
            ]
            and record[
                "temporal_candidate"
            ]
            and (
                external_count(
                    record,
                    "DFCI",
                )
                > 0
                or external_count(
                    record,
                    "VICC",
                )
                > 0
            )
        )
    ]

    genomic_candidates = [
        record
        for record
        in enriched
        if (
            not record[
                "hard_outcome_sensitive"
            ]
            and record[
                "genomic_candidate"
            ]
            and (
                external_count(
                    record,
                    "DFCI",
                )
                > 0
                or external_count(
                    record,
                    "VICC",
                )
                > 0
            )
        )
    ]

    scan_candidates.sort(
        key=lambda record: (
            external_count(
                record,
                "DFCI",
            )
            + external_count(
                record,
                "VICC",
            ),
            external_count(
                record,
                "DFCI",
            ),
            external_count(
                record,
                "VICC",
            ),
        ),
        reverse=True,
    )

    temporal_candidates.sort(
        key=lambda record: (
            external_count(
                record,
                "DFCI",
            )
            + external_count(
                record,
                "VICC",
            )
        ),
        reverse=True,
    )

    genomic_candidates.sort(
        key=lambda record: (
            external_count(
                record,
                "DFCI",
            )
            + external_count(
                record,
                "VICC",
            )
        ),
        reverse=True,
    )

    dfci_scan_tables = [
        record
        for record
        in scan_candidates
        if external_count(
            record,
            "DFCI",
        )
        > 0
    ]

    vicc_scan_tables = [
        record
        for record
        in scan_candidates
        if external_count(
            record,
            "VICC",
        )
        > 0
    ]

    ###########################################################################
    # F. Decide next adapter branch.
    ###########################################################################

    if (
        dfci_scan_tables
        and vicc_scan_tables
    ):

        status = (
            "READY_FOR_EXTERNAL_ADAPTER_IMPLEMENTATION"
        )

    elif (
        dfci_scan_tables
        or vicc_scan_tables
    ):

        status = (
            "NEEDS_CENTER_SPECIFIC_SCAN_SOURCE_RESOLUTION"
        )

    else:

        status = (
            "NEEDS_ALTERNATE_EXTERNAL_SCAN_SOURCE"
        )

    ###########################################################################
    # G. Persistent outcome-blinding audit.
    ###########################################################################

    quarantined = [
        record
        for record
        in enriched
        if record[
            "hard_outcome_sensitive"
        ]
    ]

    violations = [
        record[
            "relative_path"
        ]
        for record
        in quarantined
        if record.get(
            "row_access"
        )
        != "QUARANTINED_HEADER_ONLY"
    ]

    if violations:

        raise RuntimeError(
            "Outcome-blinding violation: row access occurred in "
            f"{violations}"
        )

    ###########################################################################
    # H. Save machine-readable inventory.
    ###########################################################################

    flat_rows = []

    for record in enriched:

        flat_rows.append(
            {
                "relative_path":
                    record[
                        "relative_path"
                    ],

                "bytes":
                    record[
                        "bytes"
                    ],

                "hard_outcome_sensitive":
                    record[
                        "hard_outcome_sensitive"
                    ],

                "scan_state_sensitive":
                    record[
                        "scan_state_sensitive"
                    ],

                "scan_candidate":
                    record[
                        "scan_candidate"
                    ],

                "temporal_candidate":
                    record[
                        "temporal_candidate"
                    ],

                "genomic_candidate":
                    record[
                        "genomic_candidate"
                    ],

                "row_access":
                    record.get(
                        "row_access"
                    ),

                "identifier_kind":
                    record.get(
                        "identifier_kind"
                    ),

                "identifier_column":
                    record.get(
                        "identifier_column"
                    ),

                "rows_scanned":
                    record.get(
                        "rows_scanned"
                    ),

                "dfci_patients":
                    (
                        record.get(
                            "external_patients",
                            {}
                        ).get(
                            "DFCI"
                        )
                    ),

                "vicc_patients":
                    (
                        record.get(
                            "external_patients",
                            {}
                        ).get(
                            "VICC"
                        )
                    ),

                "msk_patients":
                    (
                        record.get(
                            "external_patients",
                            {}
                        ).get(
                            "MSK"
                        )
                    ),

                "date_columns":
                    json.dumps(
                        record[
                            "date_columns"
                        ]
                    ),

                "scan_columns":
                    json.dumps(
                        record[
                            "scan_columns"
                        ]
                    ),

                "temporal_columns":
                    json.dumps(
                        record[
                            "temporal_columns"
                        ]
                    ),

                "genomic_columns":
                    json.dumps(
                        record[
                            "genomic_columns"
                        ]
                    ),

                "hard_outcome_hits":
                    json.dumps(
                        record[
                            "hard_outcome_hits"
                        ]
                    ),

                "scan_state_hits":
                    json.dumps(
                        record[
                            "scan_state_hits"
                        ]
                    ),

                "columns":
                    json.dumps(
                        record[
                            "columns"
                        ]
                    ),
            }
        )

    atomic_parquet(
        out
        / "schema_inventory.parquet",
        pd.DataFrame(
            flat_rows
        ),
    )

    lock_audit = audit_lock_hashes(
        repo
    )

    report = {
        "status":
            status,

        "policy": {
            "external_outcomes_opened":
                False,

            "hard_outcome_tables_row_access":
                False,

            "allowed_row_access":
                (
                    "Identifiers only from non-outcome tables; "
                    "patient-level institution metadata only."
                ),

            "scan_state_values_summarized":
                False,

            "dfci_vicc_outcome_distributions_inspected":
                False,
        },

        "site_partition":
            site_audit,

        "sample_alias_map": {
            "samples_mapped":
                int(
                    len(
                        sample_to_canonical
                    )
                ),

            "sources":
                sample_map_sources,
        },

        "inventory_counts": {
            "files":
                int(
                    len(
                        enriched
                    )
                ),

            "hard_outcome_quarantined":
                int(
                    len(
                        quarantined
                    )
                ),

            "scan_candidates_external":
                int(
                    len(
                        scan_candidates
                    )
                ),

            "temporal_candidates_external":
                int(
                    len(
                        temporal_candidates
                    )
                ),

            "genomic_candidates_external":
                int(
                    len(
                        genomic_candidates
                    )
                ),
        },

        "dfci_scan_source_count":
            int(
                len(
                    dfci_scan_tables
                )
            ),

        "vicc_scan_source_count":
            int(
                len(
                    vicc_scan_tables
                )
            ),

        "scan_candidates":
            scan_candidates,

        "temporal_candidates":
            temporal_candidates,

        "genomic_candidates":
            genomic_candidates,

        "quarantined_headers":
            [
                {
                    "relative_path":
                        record[
                            "relative_path"
                        ],

                    "hard_outcome_hits":
                        record[
                            "hard_outcome_hits"
                        ],

                    "columns":
                        record[
                            "columns"
                        ],

                    "row_access":
                        record[
                            "row_access"
                        ],
                }
                for record
                in quarantined
            ],

        "lock_audit":
            lock_audit,
    }

    atomic_json(
        out
        / "discovery.json",
        report,
    )

    ###########################################################################
    # I. Human-readable audit.
    ###########################################################################

    lines = [
        "# CKPT7A1 — Outcome-blinded external-input discovery",
        "",
        f"Status: **{status}**",
        "",
        "## Outcome-blinding",
        "",
        "- DFCI/VICC outcome distributions inspected: **NO**",
        "- Hard outcome-sensitive table rows opened: **NO**",
        "- Progression/response value distributions inspected: **NO**",
        "- Only identifiers from non-outcome tables were read for coverage mapping.",
        "",
        "## BPC center partition",
        "",
        f"- MSK: {site_counts['MSK']}",
        f"- DFCI: {site_counts['DFCI']}",
        f"- VICC: {site_counts['VICC']}",
        "",
        "## External scan-source discovery",
        "",
        f"- DFCI candidate scan tables: {len(dfci_scan_tables)}",
        f"- VICC candidate scan tables: {len(vicc_scan_tables)}",
        "",
    ]

    for record in scan_candidates[:25]:

        lines.append(
            "- "
            + record[
                "relative_path"
            ]
            + " | DFCI patients="
            + str(
                external_count(
                    record,
                    "DFCI",
                )
            )
            + " | VICC patients="
            + str(
                external_count(
                    record,
                    "VICC",
                )
            )
            + " | MSK patients="
            + str(
                external_count(
                    record,
                    "MSK",
                )
            )
            + " | date columns="
            + str(
                record[
                    "date_columns"
                ]
            )
            + " | scan columns="
            + str(
                record[
                    "scan_columns"
                ]
            )
        )

    lines.extend(
        [
            "",
            "## External temporal-input candidates",
            "",
        ]
    )

    for record in temporal_candidates[:30]:

        lines.append(
            "- "
            + record[
                "relative_path"
            ]
            + " | DFCI="
            + str(
                external_count(
                    record,
                    "DFCI",
                )
            )
            + " | VICC="
            + str(
                external_count(
                    record,
                    "VICC",
                )
            )
            + " | dates="
            + str(
                record[
                    "date_columns"
                ]
            )
        )

    lines.extend(
        [
            "",
            "## External genomic-input candidates",
            "",
        ]
    )

    for record in genomic_candidates[:30]:

        lines.append(
            "- "
            + record[
                "relative_path"
            ]
            + " | DFCI="
            + str(
                external_count(
                    record,
                    "DFCI",
                )
            )
            + " | VICC="
            + str(
                external_count(
                    record,
                    "VICC",
                )
            )
        )

    lines.extend(
        [
            "",
            "## Outcome quarantine",
            "",
            f"- header-only quarantined files: {len(quarantined)}",
            "- No rows from those files were read by CKPT7A1.",
            "",
        ]
    )

    atomic_text(
        out
        / "audit.md",
        "\n".join(
            lines
        )
        + "\n",
    )

    ###########################################################################
    # J. Handoff
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A1",

        "name":
            "outcome_blinded_external_input_discovery",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "site_counts":
            site_counts,

        "dfci_scan_source_count":
            len(
                dfci_scan_tables
            ),

        "vicc_scan_source_count":
            len(
                vicc_scan_tables
            ),

        "scan_candidates":
            [
                {
                    "relative_path":
                        record[
                            "relative_path"
                        ],

                    "dfci_patients":
                        external_count(
                            record,
                            "DFCI",
                        ),

                    "vicc_patients":
                        external_count(
                            record,
                            "VICC",
                        ),

                    "date_columns":
                        record[
                            "date_columns"
                        ],

                    "scan_columns":
                        record[
                            "scan_columns"
                        ],
                }
                for record
                in scan_candidates
            ],

        "next_action": (
            (
                "Implement the frozen CKPT7A external adapter using "
                "the discovered center-specific scan, temporal, and "
                "genomic input sources. Continue to keep all external "
                "outcome fields unopened."
            )
            if status
            == "READY_FOR_EXTERNAL_ADAPTER_IMPLEMENTATION"
            else (
                "Resolve the missing center-specific scan source before "
                "constructing the external adapter or opening outcomes."
            )
        ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A1.json",
        handoff,
    )

    ###########################################################################
    # K. Terminal packet
    ###########################################################################

    def compact_candidate(
        record,
    ):

        return {
            "path":
                record[
                    "relative_path"
                ],

            "DFCI":
                external_count(
                    record,
                    "DFCI",
                ),

            "VICC":
                external_count(
                    record,
                    "VICC",
                ),

            "MSK":
                external_count(
                    record,
                    "MSK",
                ),

            "dates":
                record[
                    "date_columns"
                ],

            "scan_fields":
                record[
                    "scan_columns"
                ],

            "scan_state_sensitive":
                record[
                    "scan_state_sensitive"
                ],
        }

    print("")
    print(
        "========== CKPT7A1 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        f"site_counts={site_counts}"
    )

    print(
        "files_inventoried="
        f"{len(enriched)}"
    )

    print(
        "hard_outcome_tables_quarantined="
        f"{len(quarantined)}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "sample_aliases="
        f"{len(sample_to_canonical)}"
    )

    print(
        "dfci_scan_source_count="
        f"{len(dfci_scan_tables)}"
    )

    print(
        "vicc_scan_source_count="
        f"{len(vicc_scan_tables)}"
    )

    print(
        "temporal_candidate_count="
        f"{len(temporal_candidates)}"
    )

    print(
        "genomic_candidate_count="
        f"{len(genomic_candidates)}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A1.json"
    )

    print(
        "========== CKPT7A1 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A1 DECISION PACKET =========="
    )

    print(
        "scan_candidates="
        f"{[compact_candidate(record) for record in scan_candidates[:30]]}"
    )

    print(
        "temporal_candidates="
        f"{[compact_candidate(record) for record in temporal_candidates[:30]]}"
    )

    print(
        "genomic_candidates="
        f"{[compact_candidate(record) for record in genomic_candidates[:30]]}"
    )

    print(
        "quarantined_header_only="
        f"{[record['relative_path'] for record in quarantined]}"
    )

    print(
        "lock_audit="
        f"{lock_audit}"
    )

    print(
        "========== CKPT7A1 DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
