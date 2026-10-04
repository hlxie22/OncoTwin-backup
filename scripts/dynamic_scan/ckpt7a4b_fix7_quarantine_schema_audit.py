#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".").resolve()

BPC = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public"
)

OUT = (
    ROOT
    / "artifacts/checkpoint7a4b_fix7_quarantine_schema"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


###############################################################################
# Frozen quarantine
###############################################################################

QUARANTINED = [
    "cBioPortal_files/data_clinical_supp_survival.txt",
    "cBioPortal_files/data_clinical_supp_survival_treatment.txt",
    "clinical_data/cancer_level_dataset_index.csv",
    "clinical_data/cancer_panel_test_level_dataset.csv",
    "clinical_data/patient_level_dataset.csv",
    "clinical_data/regimen_cancer_level_dataset.csv",
]

###############################################################################
# Column-name classification only.
#
# IMPORTANT:
# This does NOT authorize row access.
###############################################################################

PREDICTOR_HINTS = (
    "regimen",
    "line",
    "lot",
    "therapy",
    "treatment",
    "agent",
    "drug",
    "start",
    "stop",
    "end",
    "adv",
    "advanced",
    "metast",
    "distant",
    "stage",
    "diagnos",
    "index",
    "cancer",
    "disease",
    "setting",
    "sequence",
)

OUTCOME_HINTS = (
    "pfs",
    "os_",
    "_os",
    "overall_survival",
    "survival",
    "death",
    "dead",
    "censor",
    "event",
    "progression",
    "response",
    "outcome",
    "followup",
    "follow_up",
    "last_contact",
    "time_to",
    "tt_",
)

IDENTIFIER_HINTS = (
    "record_id",
    "patient_id",
    "sample_id",
    "cancer_no",
)


def norm(value: Any) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def parse_columns(value: Any) -> list[str]:

    if isinstance(
        value,
        list,
    ):
        return [
            str(item)
            for item in value
        ]

    if value is None:
        return []

    text = str(
        value
    ).strip()

    if not text:
        return []

    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list,
        ):

            return [
                str(item)
                for item in parsed
            ]

    except Exception:

        pass

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


def classify_column(
    column: str,
) -> str:

    name = norm(
        column
    )

    if any(
        token in name
        for token in OUTCOME_HINTS
    ):

        return "OUTCOME_OR_POSTBASELINE_SENSITIVE"

    if any(
        token == name
        or name.endswith(
            "_"
            + token
        )
        for token in IDENTIFIER_HINTS
    ):

        return "IDENTIFIER"

    if any(
        token in name
        for token in PREDICTOR_HINTS
    ):

        return "POTENTIAL_PREDICTOR"

    return "OTHER_UNKNOWN"


###############################################################################
# 1. Existing CKPT7A1 schema inventory.
###############################################################################

inventory = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint7a1/schema_inventory.parquet"
)

if "relative_path" not in inventory.columns:

    raise RuntimeError(
        "CKPT7A1 inventory missing relative_path."
    )

selected = inventory[
    inventory[
        "relative_path"
    ].isin(
        QUARANTINED
    )
].copy()

if set(
    selected[
        "relative_path"
    ]
) != set(
    QUARANTINED
):

    missing = sorted(
        set(
            QUARANTINED
        )
        - set(
            selected[
                "relative_path"
            ]
        )
    )

    raise RuntimeError(
        f"Quarantined files absent from inventory: {missing}"
    )


###############################################################################
# 2. Header-only verification.
#
# nrows=0 is deliberate: no patient/outcome row is read.
###############################################################################

records = []

for relative in QUARANTINED:

    path = (
        BPC
        / relative
    )

    if relative.endswith(
        ".txt"
    ):

        header = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            nrows=0,
        )

    else:

        header = pd.read_csv(
            path,
            nrows=0,
        )

    inventory_row = selected[
        selected[
            "relative_path"
        ]
        == relative
    ].iloc[
        0
    ]

    inventory_columns = parse_columns(
        inventory_row.get(
            "columns"
        )
    )

    header_columns = [
        str(
            column
        )
        for column in header.columns
    ]

    ###########################################################################
    # Header verification only.
    ###########################################################################

    inventory_set = set(
        inventory_columns
    )

    header_set = set(
        header_columns
    )

    for column in header_columns:

        records.append(
            {
                "relative_path":
                    relative,

                "column":
                    column,

                "classification":
                    classify_column(
                        column
                    ),

                "normalized_column":
                    norm(
                        column
                    ),

                "present_in_ckpt7a1_inventory":
                    column
                    in inventory_set,
            }
        )

    print("")
    print(
        f"========== HEADER {relative} =========="
    )

    print(
        "column_count=",
        len(
            header_columns
        ),
    )

    print(
        "inventory_column_count=",
        len(
            inventory_columns
        ),
    )

    print(
        "header_only=True"
    )

    print(
        "columns="
    )

    for column in header_columns:

        print(
            f"  {column} :: "
            f"{classify_column(column)}"
        )

    if inventory_columns:

        only_header = sorted(
            header_set
            - inventory_set
        )

        only_inventory = sorted(
            inventory_set
            - header_set
        )

        print(
            "header_minus_inventory=",
            only_header,
        )

        print(
            "inventory_minus_header=",
            only_inventory,
        )

    print(
        f"========== HEADER {relative} END =========="
    )


column_frame = pd.DataFrame(
    records
)

column_frame.to_parquet(
    OUT
    / "quarantined_column_classification.parquet",
    index=False,
)


###############################################################################
# 3. Candidate predictor columns relevant to the unresolved line-zero problem.
###############################################################################

candidate = column_frame[
    column_frame[
        "classification"
    ]
    == "POTENTIAL_PREDICTOR"
].copy()

candidate[
    "line_zero_relevance"
] = (
    candidate[
        "normalized_column"
    ].map(
        lambda value:
            any(
                token in value
                for token in (
                    "line",
                    "lot",
                    "regimen",
                    "start",
                    "adv",
                    "advanced",
                    "metast",
                    "distant",
                    "stage",
                    "diagnos",
                    "setting",
                )
            )
    )
)

candidate = candidate.sort_values(
    [
        "line_zero_relevance",
        "relative_path",
        "column",
    ],
    ascending=[
        False,
        True,
        True,
    ],
    kind="mergesort",
)

candidate.to_parquet(
    OUT
    / "predictor_column_candidates.parquet",
    index=False,
)


###############################################################################
# 4. Public variable-synopsis metadata.
#
# This is documentation, not patient data. We inspect descriptions only.
###############################################################################

synopsis_candidates = list(
    (
        BPC
        / "Documentation"
    ).glob(
        "*Variable Synopsis*.xlsx"
    )
)

documentation_hits = []

documentation_status = (
    "NOT_FOUND"
)

if len(
    synopsis_candidates
) == 1:

    synopsis = synopsis_candidates[
        0
    ]

    documentation_status = (
        "READ_ATTEMPTED"
    )

    try:

        book = pd.ExcelFile(
            synopsis
        )

        target_names = set(
            candidate[
                "normalized_column"
            ]
        )

        for sheet in book.sheet_names:

            frame = pd.read_excel(
                synopsis,
                sheet_name=sheet,
                dtype=str,
            )

            if frame.empty:
                continue

            for row_number, row in frame.iterrows():

                values = [
                    ""
                    if pd.isna(
                        value
                    )
                    else str(
                        value
                    )
                    for value in row.tolist()
                ]

                joined = " | ".join(
                    values
                )

                normalized = norm(
                    joined
                )

                hits = [
                    name
                    for name in target_names
                    if (
                        name
                        and name
                        in normalized
                    )
                ]

                if not hits:
                    continue

                documentation_hits.append(
                    {
                        "sheet":
                            sheet,

                        "excel_row":
                            int(
                                row_number
                                + 2
                            ),

                        "matched_variables":
                            sorted(
                                hits
                            ),

                        "metadata_text":
                            joined[
                                :4000
                            ],
                    }
                )

        documentation_status = (
            "READ_OK"
        )

    except Exception as exc:

        documentation_status = (
            "READ_FAILED:"
            + type(
                exc
            ).__name__
            + ":"
            + str(
                exc
            )
        )

doc_frame = pd.DataFrame(
    documentation_hits
)

if not doc_frame.empty:

    doc_frame.to_parquet(
        OUT
        / "variable_synopsis_matches.parquet",
        index=False,
    )

else:

    pd.DataFrame(
        columns=[
            "sheet",
            "excel_row",
            "matched_variables",
            "metadata_text",
        ]
    ).to_parquet(
        OUT
        / "variable_synopsis_matches.parquet",
        index=False,
    )


###############################################################################
# 5. Structured decision categories.
###############################################################################

regimen_candidates = candidate[
    candidate[
        "relative_path"
    ]
    == "clinical_data/regimen_cancer_level_dataset.csv"
].copy()

cancer_index_candidates = candidate[
    candidate[
        "relative_path"
    ]
    == "clinical_data/cancer_level_dataset_index.csv"
].copy()

high_interest_tokens = (
    "line",
    "lot",
    "regimen",
    "adv",
    "advanced",
    "metast",
    "distant",
    "stage",
    "setting",
)

high_interest = candidate[
    candidate[
        "normalized_column"
    ].map(
        lambda value:
            any(
                token in value
                for token in high_interest_tokens
            )
    )
].copy()

outcome_columns = column_frame[
    column_frame[
        "classification"
    ]
    == "OUTCOME_OR_POSTBASELINE_SENSITIVE"
].copy()


###############################################################################
# 6. Important scientific guard.
#
# A column NAME alone is insufficient to authorize row access.
###############################################################################

if high_interest.empty:

    status = (
        "NO_PREDICTOR_WHITELIST_CANDIDATE_FOUND"
    )

else:

    status = (
        "PREDICTOR_WHITELIST_CANDIDATES_REQUIRE_SEMANTIC_REVIEW"
    )


report = {
    "status":
        status,

    "external_outcome_rows_opened":
        False,

    "quarantined_files":
        QUARANTINED,

    "header_access_only":
        True,

    "patient_rows_read_from_quarantined_tables":
        0,

    "documentation_status":
        documentation_status,

    "column_counts": {
        relative:
            int(
                (
                    column_frame[
                        "relative_path"
                    ]
                    == relative
                ).sum()
            )
        for relative in QUARANTINED
    },

    "high_interest_predictor_candidates":
        high_interest[
            [
                "relative_path",
                "column",
                "normalized_column",
            ]
        ].to_dict(
            orient="records"
        ),

    "regimen_table_predictor_candidates":
        regimen_candidates[
            [
                "column",
                "normalized_column",
                "line_zero_relevance",
            ]
        ].to_dict(
            orient="records"
        ),

    "cancer_index_predictor_candidates":
        cancer_index_candidates[
            [
                "column",
                "normalized_column",
                "line_zero_relevance",
            ]
        ].to_dict(
            orient="records"
        ),

    "outcome_sensitive_column_count":
        int(
            len(
                outcome_columns
            )
        ),

    "variable_synopsis_match_count":
        int(
            len(
                doc_frame
            )
        ),

    "scientific_guard":
        (
            "No quarantined-table row may be opened based only on this "
            "header audit. A predictor column may be whitelisted only after "
            "its public metadata establishes that it is defined independently "
            "of future progression, censoring, death, PFS, OS, or follow-up."
        ),

    "next_action":
        (
            "Review high-interest column names and variable-synopsis "
            "definitions. If an independently curated metastatic-entry or "
            "metastatic-line predictor exists, freeze a narrow column-level "
            "protocol amendment before reading any rows. Otherwise close the "
            "attempt to reproduce CHORD absolute line numbering from BPC."
        ),
}

atomic_json(
    OUT
    / "audit.json",
    report,
)


###############################################################################
# 7. Output
###############################################################################

print("")
print(
    "========== CKPT7A4B FIX7 SUMMARY =========="
)

print(
    "status="
    + status
)

print(
    "external_outcome_rows_opened=False"
)

print(
    "header_access_only=True"
)

print(
    "patient_rows_read_from_quarantined_tables=0"
)

print(
    "documentation_status="
    + documentation_status
)

print(
    "high_interest_candidate_count="
    + str(
        len(
            high_interest
        )
    )
)

print(
    "========== CKPT7A4B FIX7 SUMMARY END =========="
)

print("")
print(
    "========== CKPT7A4B FIX7 DECISION PACKET =========="
)

print("")
print(
    "----- HIGH-INTEREST PREDICTOR CANDIDATES -----"
)

if high_interest.empty:

    print(
        "NONE"
    )

else:

    print(
        high_interest[
            [
                "relative_path",
                "column",
                "classification",
            ]
        ].to_string(
            index=False
        )
    )

print(
    "----- HIGH-INTEREST PREDICTOR CANDIDATES END -----"
)

print("")
print(
    "----- REGIMEN TABLE CANDIDATES -----"
)

if regimen_candidates.empty:

    print(
        "NONE"
    )

else:

    print(
        regimen_candidates[
            [
                "column",
                "classification",
                "line_zero_relevance",
            ]
        ].to_string(
            index=False
        )
    )

print(
    "----- REGIMEN TABLE CANDIDATES END -----"
)

print("")
print(
    "----- CANCER-INDEX TABLE CANDIDATES -----"
)

if cancer_index_candidates.empty:

    print(
        "NONE"
    )

else:

    print(
        cancer_index_candidates[
            [
                "column",
                "classification",
                "line_zero_relevance",
            ]
        ].to_string(
            index=False
        )
    )

print(
    "----- CANCER-INDEX TABLE CANDIDATES END -----"
)

print("")
print(
    "----- VARIABLE SYNOPSIS MATCHES -----"
)

if doc_frame.empty:

    print(
        "NONE"
    )

else:

    for _, row in doc_frame.iterrows():

        print(
            "sheet="
            + str(
                row[
                    "sheet"
                ]
            )
        )

        print(
            "excel_row="
            + str(
                row[
                    "excel_row"
                ]
            )
        )

        print(
            "matched_variables="
            + str(
                row[
                    "matched_variables"
                ]
            )
        )

        print(
            "metadata_text="
            + str(
                row[
                    "metadata_text"
                ]
            )
        )

        print(
            "---"
        )

print(
    "----- VARIABLE SYNOPSIS MATCHES END -----"
)

print("")
print(
    "scientific_guard="
    + report[
        "scientific_guard"
    ]
)

print(
    "next_action="
    + report[
        "next_action"
    ]
)

print(
    "========== CKPT7A4B FIX7 DECISION PACKET END =========="
)

