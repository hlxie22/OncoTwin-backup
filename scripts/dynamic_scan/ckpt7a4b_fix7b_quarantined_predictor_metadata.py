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
    / "artifacts/checkpoint7a4b_fix7b_predictor_metadata"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


###############################################################################
# Exact variables relevant to the unresolved metastatic line-zero problem.
#
# These names come from HEADER-ONLY inspection. No row access is authorized.
###############################################################################

TARGETS = [
    ###########################################################################
    # Cancer-level metastatic-entry candidates.
    ###########################################################################
    "stage_dx",
    "stage_dx_iv",
    "best_ajcc_stage_cd",
    "ca_dmets_yn",
    "ca_first_dmets1",
    "ca_first_dmets2",
    "ca_first_dmets3",
    "ca_first_dmets4",
    "ca_first_dmets5",
    "ca_first_dmets6",
    "ca_first_dmets7",
    "ca_first_dmets8",
    "ca_first_dmets9",
    "ca_first_dmets10",
    "dmets_post_dx",
    "dx_to_dmets_days",
    "dx_to_dmets_mos",
    "dx_to_dmets_yrs",
    "reg_rcvd_before_distant_mets",
    "ca_n_regimens",

    ###########################################################################
    # Site-specific metastatic timing candidates.
    ###########################################################################
    "dist_mets_bone",
    "dx_to_dist_mets_bone_days",
    "dist_mets_brain_cns",
    "dx_to_dist_mets_brain_cns_days",
    "dist_mets_liver",
    "dx_to_dist_mets_liver_days",
    "dist_mets_pulmonary",
    "dx_to_dist_mets_pulmonary_days",
    "dist_mets_pleura_and_malignant_pleural_effusion",
    "dx_to_dist_mets_pleura_and_malignant_pleural_effusion_days",
    "dist_mets_peritoneum_and_malignant_peritoneal_effusion",
    "dx_to_dist_mets_peritoneum_and_malignant_peritoneal_effusion_days",
    "dist_mets_adrenal",
    "dx_to_dist_mets_adrenal_days",
    "dist_mets_bone_marrow",
    "dx_to_dist_mets_bone_marrow_days",
    "dist_mets_thorax",
    "dx_to_dist_mets_thorax_days",
    "dist_mets_abdomen",
    "dx_to_dist_mets_abdomen_days",
    "dist_mets_pelvis",
    "dx_to_dist_mets_pelvis_days",
    "dist_mets_other",
    "dx_to_dist_mets_other_days",

    ###########################################################################
    # Regimen candidates.
    ###########################################################################
    "regimen_number",
    "regimen_number_within_cancer",
    "drugs_num",
    "drugs_inst",
    "drugs_firstinst",
    "drugs_ct_yn",
    "drugs_dc_ynu",
    "regimen_drugs",
    "dx_reg_start_int",
    "dx_reg_start_int_mos",
    "dx_reg_start_int_yrs",
    "dx_reg_end_any_int",
    "dx_reg_end_all_int",
]


QUARANTINED = {
    "cancer":
        BPC
        / "clinical_data/cancer_level_dataset_index.csv",

    "regimen":
        BPC
        / "clinical_data/regimen_cancer_level_dataset.csv",

    "survival":
        BPC
        / "cBioPortal_files/data_clinical_supp_survival.txt",

    "survival_treatment":
        BPC
        / "cBioPortal_files/data_clinical_supp_survival_treatment.txt",
}


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

    tmp = Path(
        str(path)
        + ".tmp"
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

    ###########################################################################
    # Parse-back validation.
    ###########################################################################

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(
        path
    )


def atomic_parquet(
    path: Path,
    frame: pd.DataFrame,
) -> None:

    tmp = Path(
        str(path)
        + ".tmp"
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
            f"Parquet validation failed: {path}"
        )

    tmp.replace(
        path
    )


###############################################################################
# 1. HEADER-ONLY verification.
#
# Critical: nrows=0 means no patient row from these analytic tables is read.
###############################################################################

headers = {}

for key, path in QUARANTINED.items():

    if path.suffix.lower() == ".txt":

        frame = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            nrows=0,
        )

    else:

        frame = pd.read_csv(
            path,
            nrows=0,
        )

    headers[
        key
    ] = [
        str(column)
        for column in frame.columns
    ]


###############################################################################
# 2. Confirm exact candidate availability.
###############################################################################

availability_rows = []

for target in TARGETS:

    normalized_target = norm(
        target
    )

    found = []

    for source, columns in headers.items():

        normalized_columns = {
            norm(column):
                column
            for column in columns
        }

        if normalized_target in normalized_columns:

            found.append(
                {
                    "source":
                        source,

                    "column":
                        normalized_columns[
                            normalized_target
                        ],
                }
            )

    availability_rows.append(
        {
            "target":
                target,

            "found":
                bool(
                    found
                ),

            "sources":
                json.dumps(
                    found,
                    sort_keys=True,
                ),
        }
    )

availability = pd.DataFrame(
    availability_rows
)

atomic_parquet(
    OUT
    / "target_header_availability.parquet",
    availability,
)


###############################################################################
# 3. Read public Variable Synopsis metadata.
#
# Documentation only. This is NOT a patient-level table.
###############################################################################

workbooks = list(
    (
        BPC
        / "Documentation"
    ).glob(
        "*Variable Synopsis*.xlsx"
    )
)

if len(workbooks) != 1:

    raise RuntimeError(
        f"Expected one Variable Synopsis workbook: {workbooks}"
    )

workbook = workbooks[
    0
]

book = pd.ExcelFile(
    workbook
)

target_set = {
    norm(target):
        target
    for target in TARGETS
}

hits = []

sheet_summaries = []

for sheet in book.sheet_names:

    ###########################################################################
    # header=None preserves whatever layout the public documentation uses.
    ###########################################################################

    frame = pd.read_excel(
        workbook,
        sheet_name=sheet,
        header=None,
        dtype=str,
    )

    frame = frame.fillna(
        ""
    )

    sheet_hit_count = 0

    for row_index, row in frame.iterrows():

        cells = [
            str(value).strip()
            for value in row.tolist()
        ]

        normalized_cells = [
            norm(value)
            for value in cells
        ]

        joined = " | ".join(
            value
            for value in cells
            if value
        )

        for normalized_target, original_target in target_set.items():

            ###################################################################
            # Prefer exact-cell match. Also allow the variable name to be
            # embedded in a metadata phrase or combined variable-list cell.
            ###################################################################

            exact = (
                normalized_target
                in normalized_cells
            )

            substring = any(
                normalized_target
                and normalized_target
                in cell
                for cell in normalized_cells
            )

            if not (
                exact
                or substring
            ):

                continue

            hits.append(
                {
                    "target":
                        original_target,

                    "normalized_target":
                        normalized_target,

                    "sheet":
                        sheet,

                    "excel_row":
                        int(
                            row_index
                            + 1
                        ),

                    "exact_cell_match":
                        bool(
                            exact
                        ),

                    "metadata_text":
                        joined[
                            :12000
                        ],
                }
            )

            sheet_hit_count += 1

    sheet_summaries.append(
        {
            "sheet":
                sheet,

            "rows":
                int(
                    len(
                        frame
                    )
                ),

            "columns":
                int(
                    len(
                        frame.columns
                    )
                ),

            "target_hits":
                int(
                    sheet_hit_count
                ),
        }
    )


metadata = pd.DataFrame(
    hits
)

if metadata.empty:

    metadata = pd.DataFrame(
        columns=[
            "target",
            "normalized_target",
            "sheet",
            "excel_row",
            "exact_cell_match",
            "metadata_text",
        ]
    )

else:

    metadata = metadata.drop_duplicates(
        [
            "target",
            "sheet",
            "excel_row",
            "metadata_text",
        ]
    ).sort_values(
        [
            "target",
            "exact_cell_match",
            "sheet",
            "excel_row",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

atomic_parquet(
    OUT
    / "variable_synopsis_target_matches.parquet",
    metadata,
)


###############################################################################
# 4. Focused semantic groups.
###############################################################################

METASTATIC_ENTRY_TARGETS = {
    "stage_dx",
    "stage_dx_iv",
    "ca_dmets_yn",
    "dmets_post_dx",
    "dx_to_dmets_days",
    "dx_to_dmets_mos",
    "dx_to_dmets_yrs",
    "reg_rcvd_before_distant_mets",
}

REGIMEN_TARGETS = {
    "regimen_number",
    "regimen_number_within_cancer",
    "dx_reg_start_int",
    "dx_reg_start_int_mos",
    "dx_reg_start_int_yrs",
    "regimen_drugs",
}


def metadata_for(
    names: set[str],
) -> pd.DataFrame:

    return metadata[
        metadata[
            "target"
        ].isin(
            names
        )
    ].copy()


metastatic_metadata = metadata_for(
    METASTATIC_ENTRY_TARGETS
)

regimen_metadata = metadata_for(
    REGIMEN_TARGETS
)


###############################################################################
# 5. Explicit outcome-column isolation.
#
# We do not access values. We merely record that these columns exist and are
# prohibited from any future predictor whitelist.
###############################################################################

PROHIBITED_PATTERNS = [
    re.compile(
        r"(^|_)pfs($|_)",
        re.I,
    ),

    re.compile(
        r"(^|_)os($|_)",
        re.I,
    ),

    re.compile(
        r"survival",
        re.I,
    ),

    re.compile(
        r"death",
        re.I,
    ),

    re.compile(
        r"last_alive",
        re.I,
    ),

    re.compile(
        r"last_fu",
        re.I,
    ),

    re.compile(
        r"follow.?up",
        re.I,
    ),

    re.compile(
        r"^ttnt",
        re.I,
    ),
]

prohibited = []

for source, columns in headers.items():

    for column in columns:

        normalized = norm(
            column
        )

        if any(
            pattern.search(
                normalized
            )
            for pattern
            in PROHIBITED_PATTERNS
        ):

            prohibited.append(
                {
                    "source":
                        source,

                    "column":
                        column,
                }
            )

prohibited_frame = pd.DataFrame(
    prohibited
)

if prohibited_frame.empty:

    prohibited_frame = pd.DataFrame(
        columns=[
            "source",
            "column",
        ]
    )

atomic_parquet(
    OUT
    / "explicitly_prohibited_columns.parquet",
    prohibited_frame,
)


###############################################################################
# 6. Decide only whether metadata review is possible.
#
# Do NOT decide that row access is safe automatically.
###############################################################################

metastatic_hits = set(
    metastatic_metadata[
        "target"
    ]
)

regimen_hits = set(
    regimen_metadata[
        "target"
    ]
)

critical_metastatic_metadata_present = (
    "dx_to_dmets_days"
    in metastatic_hits
    or "dmets_post_dx"
    in metastatic_hits
    or "ca_dmets_yn"
    in metastatic_hits
)

critical_regimen_metadata_present = (
    "regimen_number_within_cancer"
    in regimen_hits
    or "dx_reg_start_int"
    in regimen_hits
)

if (
    critical_metastatic_metadata_present
    and critical_regimen_metadata_present
):

    status = (
        "METASTATIC_AND_REGIMEN_METADATA_READY_FOR_REVIEW"
    )

elif critical_metastatic_metadata_present:

    status = (
        "METASTATIC_ENTRY_METADATA_READY_FOR_REVIEW"
    )

else:

    status = (
        "PUBLIC_METADATA_INSUFFICIENT_FOR_WHITELIST"
    )


###############################################################################
# 7. Machine-readable audit
###############################################################################

report = {
    "status":
        status,

    "external_outcome_rows_opened":
        False,

    "quarantined_patient_rows_read":
        0,

    "quarantined_access_mode":
        "HEADER_ONLY_NROWS_0",

    "documentation_only_metadata_read":
        True,

    "documentation_workbook":
        str(
            workbook.relative_to(
                ROOT
            )
        ),

    "sheet_summaries":
        sheet_summaries,

    "target_count":
        int(
            len(
                TARGETS
            )
        ),

    "targets_found_in_headers":
        availability[
            availability[
                "found"
            ]
        ][
            "target"
        ].tolist(),

    "metadata_match_count":
        int(
            len(
                metadata
            )
        ),

    "metastatic_metadata_targets_found":
        sorted(
            metastatic_hits
        ),

    "regimen_metadata_targets_found":
        sorted(
            regimen_hits
        ),

    "explicitly_prohibited_column_count":
        int(
            len(
                prohibited_frame
            )
        ),

    "candidate_protocol_amendment":
        (
            "NONE_YET. Review public definitions first. "
            "If definitions establish that metastatic timing and regimen "
            "fields are contemporaneous predictor variables independent of "
            "future progression/PFS/death/censoring, then freeze an exact "
            "column-level whitelist before any patient row is read."
        ),

    "scientific_guard":
        (
            "Presence of a promising column name does not authorize row "
            "access. No quarantined patient-level value may be inspected "
            "until the predictor whitelist and prohibited-column list are "
            "frozen from public metadata."
        ),
}

atomic_json(
    OUT
    / "audit.json",
    report,
)


###############################################################################
# 8. Output
###############################################################################

print("")
print(
    "========== CKPT7A4B FIX7B SUMMARY =========="
)

print(
    "status="
    + status
)

print(
    "external_outcome_rows_opened=False"
)

print(
    "quarantined_patient_rows_read=0"
)

print(
    "quarantined_access_mode=HEADER_ONLY_NROWS_0"
)

print(
    "documentation_only_metadata_read=True"
)

print(
    "metadata_match_count="
    + str(
        len(
            metadata
        )
    )
)

print(
    "========== CKPT7A4B FIX7B SUMMARY END =========="
)

print("")
print(
    "========== CKPT7A4B FIX7B DECISION PACKET =========="
)

print("")
print(
    "----- METASTATIC ENTRY METADATA -----"
)

if metastatic_metadata.empty:

    print(
        "NONE"
    )

else:

    for _, row in metastatic_metadata.iterrows():

        print(
            "target="
            + str(
                row[
                    "target"
                ]
            )
        )

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
            "exact_cell_match="
            + str(
                row[
                    "exact_cell_match"
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
    "----- METASTATIC ENTRY METADATA END -----"
)

print("")
print(
    "----- REGIMEN METADATA -----"
)

if regimen_metadata.empty:

    print(
        "NONE"
    )

else:

    for _, row in regimen_metadata.iterrows():

        print(
            "target="
            + str(
                row[
                    "target"
                ]
            )
        )

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
            "exact_cell_match="
            + str(
                row[
                    "exact_cell_match"
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
    "----- REGIMEN METADATA END -----"
)

print("")
print(
    "----- TARGET HEADER AVAILABILITY -----"
)

print(
    availability[
        availability[
            "found"
        ]
    ][
        [
            "target",
            "sources",
        ]
    ].to_string(
        index=False
    )
)

print(
    "----- TARGET HEADER AVAILABILITY END -----"
)

print("")
print(
    "----- PROHIBITED COLUMN SUMMARY -----"
)

if prohibited_frame.empty:

    print(
        "NONE"
    )

else:

    print(
        prohibited_frame.to_string(
            index=False
        )
    )

print(
    "----- PROHIBITED COLUMN SUMMARY END -----"
)

print("")
print(
    "scientific_guard="
    + report[
        "scientific_guard"
    ]
)

print(
    "candidate_protocol_amendment="
    + report[
        "candidate_protocol_amendment"
    ]
)

print(
    "========== CKPT7A4B FIX7B DECISION PACKET END =========="
)

