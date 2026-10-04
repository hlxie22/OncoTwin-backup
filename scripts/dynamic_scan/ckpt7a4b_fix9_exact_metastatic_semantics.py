#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

OUT = (
    ROOT
    / "artifacts/checkpoint7a4b_fix9_exact_metastatic_semantics"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BPC = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public"
)

CANCER_PATH = (
    BPC
    / "clinical_data/cancer_level_dataset_index.csv"
)

DIAG_PATH = (
    BPC
    / "cBioPortal_files/data_timeline_cancer_diagnosis.txt"
)

TX_PATH = (
    BPC
    / "cBioPortal_files/data_timeline_treatment.txt"
)

IMG_PATH = (
    BPC
    / "cBioPortal_files/data_timeline_imaging.txt"
)


###############################################################################
# Helpers
###############################################################################


def clean(value: Any) -> str:

    if value is None:
        return ""

    text = str(
        value
    ).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }:
        return ""

    return text


def key(value: Any) -> str:

    text = clean(
        value
    )

    if not text:
        return ""

    try:

        number = float(
            text
        )

        if (
            np.isfinite(
                number
            )
            and number.is_integer()
        ):

            return str(
                int(
                    number
                )
            )

    except Exception:

        pass

    return text.upper()


def norm(value: Any) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_col(
    columns,
    candidates,
    required=True,
):

    lookup = {
        norm(column):
            column
        for column in columns
    }

    for candidate in candidates:

        candidate = norm(
            candidate
        )

        if candidate in lookup:
            return lookup[
                candidate
            ]

    if required:

        raise KeyError(
            f"Missing {candidates}; "
            f"available={list(columns)}"
        )

    return None


def atomic_json(
    path: Path,
    payload: Any,
):

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
):

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

    if len(
        check
    ) != len(
        frame
    ):

        raise RuntimeError(
            f"Parquet validation failed: {path}"
        )

    tmp.replace(
        path
    )


def sha256_file(
    path: Path,
):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for block in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                block
            )

    return digest.hexdigest()


def numeric_stats(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:

        return {
            "n": 0,
        }

    return {
        "n":
            int(
                len(values)
            ),

        "min":
            float(
                values.min()
            ),

        "p25":
            float(
                np.quantile(
                    values,
                    0.25,
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "p75":
            float(
                np.quantile(
                    values,
                    0.75,
                )
            ),

        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),

        "p95":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),

        "max":
            float(
                values.max()
            ),

        "mean":
            float(
                values.mean()
            ),
    }


def import_module(
    name,
    path,
):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"Unable to import {path}"
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
# 2. Confirm all earlier outcome-free gates
###############################################################################

fix7b = json.load(
    open(
        ROOT
        / "artifacts/checkpoint7a4b_fix7b_predictor_metadata/"
        "audit.json",
        encoding="utf-8",
    )
)

fix8b = json.load(
    open(
        ROOT
        / "artifacts/checkpoint7a4b_fix8b_key_bridge/"
        "mapping_audit.json",
        encoding="utf-8",
    )
)

assert (
    fix8b[
        "status"
    ]
    == "PASS_TWO_KEY_CANCER_BRIDGE"
)

assert fix8b[
    "external_outcome_rows_opened"
] is False

assert fix8b[
    "external_outcome_distributions_inspected"
] is False


###############################################################################
# 3. Site-specific metastasis variables.
#
# CKPT1 first_metastasis_dates:
#   earliest tumor-site date
#   excluding OTHER and LYMPH NODES.
#
# BPC has no corresponding distant-lymph-node date field in this table.
# We therefore include documented anatomic distant-metastasis timing fields
# and deliberately exclude dx_to_dist_mets_other_days.
###############################################################################

SITE_DAY_COLUMNS = [
    "dx_to_dist_mets_bone_days",
    "dx_to_dist_mets_brain_cns_days",
    "dx_to_dist_mets_liver_days",
    "dx_to_dist_mets_pulmonary_days",
    "dx_to_dist_mets_pleura_and_malignant_pleural_effusion_days",
    "dx_to_dist_mets_peritoneum_and_malignant_peritoneal_effusion_days",
    "dx_to_dist_mets_adrenal_days",
    "dx_to_dist_mets_bone_marrow_days",
    "dx_to_dist_mets_thorax_days",
    "dx_to_dist_mets_abdomen_days",
    "dx_to_dist_mets_pelvis_days",
]

EXCLUDED_SITE_DAY_COLUMNS = [
    "dx_to_dist_mets_other_days",
]

BASE_COLUMNS = [
    "record_id",
    "institution",
    "ca_seq",
    "redcap_ca_index",
    "stage_dx",
    "stage_dx_iv",
    "ca_dmets_yn",
    "dmets_post_dx",
    "dx_to_dmets_days",
]

A2_COLUMNS = (
    BASE_COLUMNS
    + SITE_DAY_COLUMNS
)


###############################################################################
# 4. HEADER ONLY first.
###############################################################################

header = pd.read_csv(
    CANCER_PATH,
    nrows=0,
)

missing = [
    column
    for column in (
        A2_COLUMNS
        + EXCLUDED_SITE_DAY_COLUMNS
    )
    if column
    not in header.columns
]

if missing:

    raise RuntimeError(
        f"Required site-specific columns absent: {missing}"
    )


###############################################################################
# 5. Documentation-only semantic verification.
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

documentation = []

for sheet in book.sheet_names:

    frame = pd.read_excel(
        workbook,
        sheet_name=sheet,
        header=None,
        dtype=str,
    ).fillna(
        ""
    )

    for row_number, row in frame.iterrows():

        cells = [
            str(value).strip()
            for value
            in row.tolist()
        ]

        normalized = [
            norm(value)
            for value
            in cells
        ]

        for target in (
            SITE_DAY_COLUMNS
            + EXCLUDED_SITE_DAY_COLUMNS
        ):

            if norm(
                target
            ) not in normalized:

                continue

            documentation.append(
                {
                    "target":
                        target,

                    "sheet":
                        sheet,

                    "excel_row":
                        int(
                            row_number
                            + 1
                        ),

                    "text":
                        " | ".join(
                            value
                            for value
                            in cells
                            if value
                        ),
                }
            )

doc = pd.DataFrame(
    documentation
)

documented = set(
    doc[
        "target"
    ]
) if not doc.empty else set()

required_documented = set(
    SITE_DAY_COLUMNS
    + EXCLUDED_SITE_DAY_COLUMNS
)

if documented != required_documented:

    raise RuntimeError(
        "Variable Synopsis does not exactly document all site timing fields: "
        f"missing={sorted(required_documented - documented)}"
    )

for target in required_documented:

    texts = (
        doc.loc[
            doc[
                "target"
            ]
            == target,
            "text",
        ]
        .astype(str)
        .str.upper()
        .tolist()
    )

    if not any(
        (
            "DAY"
            in text
            and (
                "METAST"
                in text
                or "METS"
                in text
            )
        )
        for text in texts
    ):

        raise RuntimeError(
            f"Documentation semantics not sufficient for {target}: {texts}"
        )


###############################################################################
# 6. Freeze protocol amendment A2 BEFORE row access.
###############################################################################

PROHIBITED = sorted(
    set(
        header.columns
    )
    - set(
        A2_COLUMNS
    )
)

protocol = {
    "protocol":
        "CKPT6E_EXTERNAL_VALIDATION_PROTOCOL_AMENDMENT_A2",

    "supersedes_for_metastatic_entry":
        "CKPT6E_EXTERNAL_VALIDATION_PROTOCOL_AMENDMENT_A1",

    "purpose":
        (
            "Transport frozen CKPT1 metastatic-entry semantics using "
            "site-specific BPC distant-metastasis timing."
        ),

    "training_semantics":
        {
            "stage_iv":
                (
                    "No first-metastasis date filter. CKPT1 retains treatment "
                    "and progression rows for Stage-IV-at-diagnosis patients."
                ),

            "stage_i_iii":
                (
                    "Retain treatment/progression only on or after first "
                    "qualifying distant-metastasis site date."
                ),

            "first_metastasis":
                (
                    "Minimum documented site-specific distant-metastasis day "
                    "across the whitelisted anatomic fields."
                ),

            "excluded_site_semantics":
                [
                    "OTHER",
                    "LYMPH NODES",
                ],
        },

    "bpc_site_day_columns":
        SITE_DAY_COLUMNS,

    "explicitly_excluded_bpc_site_day_columns":
        EXCLUDED_SITE_DAY_COLUMNS,

    "allowed_columns":
        A2_COLUMNS,

    "prohibited_columns":
        PROHIBITED,

    "cancer_identity":
        {
            "diagnosis":
                "ca_seq -> CANCER_NO",

            "treatment":
                "redcap_ca_index -> REDCAP_CA_INDEX",
        },

    "treatment_exclusions":
        {
            "bpc_investigational_agent":
                "Investigational Drug",

            "reason":
                (
                    "Semantic counterpart of frozen CHORD AGENT="
                    "INVESTIGATIVE exclusion."
                ),
        },

    "development_population":
        "MSK_ONLY",

    "external_outcomes_opened":
        False,

    "external_outcome_distributions_inspected":
        False,

    "model_selection":
        False,

    "alpha_tuning":
        False,
}

protocol_path = (
    OUT
    / "protocol_amendment_A2.json"
)

atomic_json(
    protocol_path,
    protocol,
)

protocol_sha = sha256_file(
    protocol_path
)

print(
    "[CKPT7A4B_FIX9_PROTOCOL_A2_FROZEN]",
    protocol_sha,
    flush=True,
)


###############################################################################
# 7. FIRST A2-authorized row access.
#
# Exact usecols only. No outcome column enters memory.
###############################################################################

cancer = pd.read_csv(
    CANCER_PATH,
    usecols=A2_COLUMNS,
    low_memory=False,
)

if set(
    cancer.columns
) != set(
    A2_COLUMNS
):

    raise RuntimeError(
        "A2 materialized columns differ from frozen whitelist."
    )

source_columns_materialized = sorted(
    cancer.columns
)

print(
    "[CKPT7A4B_FIX9_A2_ROW_ACCESS_PASS]",
    {
        "source_columns":
            source_columns_materialized,

        "outcome_columns_materialized":
            0,
    },
    flush=True,
)


###############################################################################
# 8. Frozen MSK positive-control identity.
###############################################################################

offsets = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint7a3g/"
    "msk_patient_clock_offsets.parquet"
).copy()

offsets[
    "bpc_patient_id"
] = (
    offsets[
        "bpc_patient_id"
    ]
    .astype(str)
    .str.strip()
)

offsets[
    "chord_patient_id"
] = (
    offsets[
        "chord_patient_id"
    ]
    .astype(str)
    .str.strip()
)

msk = set(
    offsets[
        "bpc_patient_id"
    ]
)

assert len(msk) == 416

cancer[
    "_patient"
] = (
    cancer[
        "record_id"
    ]
    .astype(str)
    .str.strip()
)

cancer = cancer[
    cancer[
        "_patient"
    ].isin(
        msk
    )
].copy()

cancer[
    "_diagnosis_key"
] = cancer[
    "ca_seq"
].map(
    key
)

cancer[
    "_treatment_key"
] = cancer[
    "redcap_ca_index"
].map(
    key
)


###############################################################################
# 9. Unique index diagnosis.
###############################################################################

diagnosis = pd.read_csv(
    DIAG_PATH,
    sep="\t",
    comment="#",
    low_memory=False,
)

diagnosis[
    "_patient"
] = (
    diagnosis[
        "PATIENT_ID"
    ]
    .astype(str)
    .str.strip()
)

diagnosis = diagnosis[
    diagnosis[
        "_patient"
    ].isin(
        msk
    )
].copy()

diagnosis[
    "_dx_key"
] = diagnosis[
    "CANCER_NO"
].map(
    key
)

diagnosis[
    "_day"
] = pd.to_numeric(
    diagnosis[
        "START_DATE"
    ],
    errors="coerce",
)

index_text = (
    diagnosis[
        "INDEX_CANCER"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

diagnosis[
    "_is_index"
] = index_text.isin(
    {
        "YES",
        "Y",
        "1",
        "TRUE",
        "T",
        "INDEX",
        "INDEX CANCER",
    }
)

index_dx = diagnosis[
    diagnosis[
        "_is_index"
    ]
    & diagnosis[
        "_day"
    ].notna()
].copy()

index_dx = index_dx.drop_duplicates(
    [
        "_patient",
        "_dx_key",
        "_day",
        "STAGE_DX",
    ]
)

counts = (
    index_dx.groupby(
        "_patient",
        observed=True,
    )
    .size()
)

unique_patients = set(
    counts[
        counts
        == 1
    ].index
)

index_dx = index_dx[
    index_dx[
        "_patient"
    ].isin(
        unique_patients
    )
].copy()

assert index_dx[
    "_patient"
].is_unique


###############################################################################
# 10. Select exact analytic index-cancer row via patient + ca_seq/CANCER_NO.
###############################################################################

bridge = index_dx[
    [
        "_patient",
        "_dx_key",
        "_day",
        "STAGE_DX",
    ]
].merge(
    cancer,
    left_on=[
        "_patient",
        "_dx_key",
    ],
    right_on=[
        "_patient",
        "_diagnosis_key",
    ],
    how="inner",
)

bridge_counts = (
    bridge.groupby(
        "_patient",
        observed=True,
    )
    .size()
)

unique_bridge_patients = set(
    bridge_counts[
        bridge_counts
        == 1
    ].index
)

bridge = bridge[
    bridge[
        "_patient"
    ].isin(
        unique_bridge_patients
    )
].copy()

assert bridge[
    "_patient"
].is_unique

bridge = bridge.rename(
    columns={
        "_day":
            "diagnosis_day",
    }
)


###############################################################################
# 11. Exact Stage-IV semantics.
###############################################################################

stage4 = (
    bridge[
        "stage_dx_iv"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
    .eq(
        "STAGE IV"
    )
)

bridge[
    "is_stage4"
] = stage4


###############################################################################
# 12. Site-specific recurrent metastatic entry.
#
# Only Stage I-III uses this for the CKPT1 filter.
###############################################################################

site_values = []

for column in SITE_DAY_COLUMNS:

    values = pd.to_numeric(
        bridge[
            column
        ],
        errors="coerce",
    ).astype(float)

    values.loc[
        values
        < 0
    ] = np.nan

    site_values.append(
        values.to_numpy(
            dtype=float
        )
    )

site_matrix = np.column_stack(
    site_values
)

with np.errstate(
    all="ignore"
):

    site_min = np.nanmin(
        site_matrix,
        axis=1,
    )

all_missing = np.isnan(
    site_matrix
).all(
    axis=1
)

site_min[
    all_missing
] = np.nan

bridge[
    "site_specific_first_meta_offset"
] = site_min

bridge[
    "site_specific_first_meta_day"
] = (
    bridge[
        "diagnosis_day"
    ].astype(float)
    + bridge[
        "site_specific_first_meta_offset"
    ]
)

overall_offset = pd.to_numeric(
    bridge[
        "dx_to_dmets_days"
    ],
    errors="coerce",
)

overall_offset.loc[
    overall_offset
    < 0
] = np.nan

bridge[
    "overall_first_meta_day"
] = (
    bridge[
        "diagnosis_day"
    ].astype(float)
    + overall_offset
)

recurrent = (
    ~bridge[
        "is_stage4"
    ]
    & bridge[
        "site_specific_first_meta_day"
    ].notna()
)

bridge[
    "has_recurrent_site_anchor"
] = recurrent

stage4_patients = set(
    bridge.loc[
        bridge[
            "is_stage4"
        ],
        "_patient",
    ]
)

recurrent_patients = set(
    bridge.loc[
        recurrent,
        "_patient",
    ]
)

eligible_patients = (
    stage4_patients
    | recurrent_patients
)

print(
    "[CKPT7A4B_FIX9_COHORT]",
    {
        "bridge_patients":
            len(
                bridge
            ),

        "stage4_patients":
            len(
                stage4_patients
            ),

        "recurrent_site_anchor_patients":
            len(
                recurrent_patients
            ),

        "eligible_patients":
            len(
                eligible_patients
            ),
    },
    flush=True,
)


###############################################################################
# 13. CHORD first-metastasis positive control.
#
# Compare ONLY recurrent patients. Stage-IV patients are not first-meta
# filtered in frozen CKPT1, so a first-meta-date agreement requirement would
# be the wrong positive control for that stratum.
###############################################################################

tumor_candidates = list(
    (
        ROOT
        / "data/external_sources/msk_chord_2024/raw"
    ).rglob(
        "data_timeline_tumor_sites.txt"
    )
)

if len(tumor_candidates) != 1:

    raise RuntimeError(
        f"Unexpected CHORD tumor-site files: {tumor_candidates}"
    )

tumor = pd.read_csv(
    tumor_candidates[
        0
    ],
    sep="\t",
    comment="#",
    low_memory=False,
)

tumor[
    "_patient"
] = (
    tumor[
        "PATIENT_ID"
    ]
    .astype(str)
    .str.strip()
)

tumor[
    "_day"
] = pd.to_numeric(
    tumor[
        "START_DATE"
    ],
    errors="coerce",
)

tumor[
    "_site"
] = (
    tumor[
        "TUMOR_SITE"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

tumor = tumor[
    tumor[
        "_day"
    ].notna()
    & ~tumor[
        "_site"
    ].isin(
        {
            "OTHER",
            "LYMPH NODES",
        }
    )
].copy()

first_meta = (
    tumor.groupby(
        "_patient",
        observed=True,
    )[
        "_day"
    ]
    .min()
    .rename(
        "chord_first_meta_day"
    )
    .reset_index()
)

anchor_compare = bridge[
    bridge[
        "_patient"
    ].isin(
        recurrent_patients
    )
][
    [
        "_patient",
        "site_specific_first_meta_day",
        "overall_first_meta_day",
    ]
].copy()

anchor_compare = anchor_compare.rename(
    columns={
        "_patient":
            "bpc_patient_id",
    }
)

anchor_compare = anchor_compare.merge(
    offsets[
        [
            "bpc_patient_id",
            "chord_patient_id",
            "clock_offset",
        ]
    ],
    on="bpc_patient_id",
    how="inner",
    validate="one_to_one",
)

anchor_compare = anchor_compare.merge(
    first_meta,
    left_on="chord_patient_id",
    right_on="_patient",
    how="inner",
    validate="one_to_one",
)

anchor_compare[
    "expected_bpc_first_meta_day"
] = (
    anchor_compare[
        "chord_first_meta_day"
    ]
    - anchor_compare[
        "clock_offset"
    ]
)

for prefix in (
    "site_specific",
    "overall",
):

    anchor_compare[
        prefix
        + "_delta"
    ] = (
        anchor_compare[
            prefix
            + "_first_meta_day"
        ]
        - anchor_compare[
            "expected_bpc_first_meta_day"
        ]
    )

    anchor_compare[
        prefix
        + "_abs_delta"
    ] = np.abs(
        anchor_compare[
            prefix
            + "_delta"
        ]
    )


def anchor_metrics(
    frame,
    prefix,
):

    valid = frame[
        frame[
            prefix
            + "_first_meta_day"
        ].notna()
    ].copy()

    if valid.empty:

        return {
            "patients":
                0,
        }

    absolute = valid[
        prefix
        + "_abs_delta"
    ]

    return {
        "patients":
            int(
                valid[
                    "bpc_patient_id"
                ].nunique()
            ),

        "delta":
            numeric_stats(
                valid[
                    prefix
                    + "_delta"
                ]
            ),

        "within0":
            float(
                (
                    absolute
                    == 0
                ).mean()
            ),

        "within3":
            float(
                (
                    absolute
                    <= 3
                ).mean()
            ),

        "within7":
            float(
                (
                    absolute
                    <= 7
                ).mean()
            ),

        "within30":
            float(
                (
                    absolute
                    <= 30
                ).mean()
            ),
    }


site_anchor_metrics = anchor_metrics(
    anchor_compare,
    "site_specific",
)

overall_anchor_metrics = anchor_metrics(
    anchor_compare,
    "overall",
)

print(
    "[CKPT7A4B_FIX9_RECURRENT_ANCHOR_SITE_SPECIFIC]",
    json.dumps(
        site_anchor_metrics,
        sort_keys=True,
    ),
    flush=True,
)

print(
    "[CKPT7A4B_FIX9_RECURRENT_ANCHOR_OVERALL]",
    json.dumps(
        overall_anchor_metrics,
        sort_keys=True,
    ),
    flush=True,
)


###############################################################################
# 14. Exact CKPT1-like predictor streams.
###############################################################################

patient_info = bridge.set_index(
    "_patient"
)[
    [
        "_treatment_key",
        "is_stage4",
        "site_specific_first_meta_day",
    ]
].to_dict(
    orient="index"
)


###############################################################################
# Treatment
###############################################################################

treatment = pd.read_csv(
    TX_PATH,
    sep="\t",
    comment="#",
    low_memory=False,
)

treatment[
    "_patient"
] = (
    treatment[
        "PATIENT_ID"
    ]
    .astype(str)
    .str.strip()
)

treatment[
    "_key"
] = treatment[
    "REDCAP_CA_INDEX"
].map(
    key
)

treatment[
    "_day"
] = pd.to_numeric(
    treatment[
        "START_DATE"
    ],
    errors="coerce",
)

treatment[
    "_anchor_key"
] = treatment[
    "_patient"
].map(
    lambda patient:
        patient_info.get(
            patient,
            {},
        ).get(
            "_treatment_key",
            "",
        )
)

treatment[
    "_stage4"
] = treatment[
    "_patient"
].map(
    lambda patient:
        bool(
            patient_info.get(
                patient,
                {},
            ).get(
                "is_stage4",
                False,
            )
        )
)

treatment[
    "_meta_day"
] = treatment[
    "_patient"
].map(
    lambda patient:
        patient_info.get(
            patient,
            {},
        ).get(
            "site_specific_first_meta_day",
            np.nan,
        )
)

tx_keep = (
    treatment[
        "_patient"
    ].isin(
        eligible_patients
    )
    & treatment[
        "_day"
    ].notna()
    & (
        treatment[
            "_key"
        ]
        == treatment[
            "_anchor_key"
        ]
    )
    & (
        treatment[
            "_stage4"
        ]
        |
        (
            treatment[
                "_meta_day"
            ].notna()
            & (
                treatment[
                    "_day"
                ]
                >= treatment[
                    "_meta_day"
                ]
            )
        )
    )
)

treatment = treatment.loc[
    tx_keep
].copy()

###############################################################################
# Semantic counterpart of CHORD AGENT == INVESTIGATIVE exclusion.
###############################################################################

agent_upper = (
    treatment[
        "AGENT"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

investigational_rows = int(
    agent_upper.isin(
        {
            "INVESTIGATIVE",
            "INVESTIGATIONAL DRUG",
        }
    ).sum()
)

treatment = treatment.loc[
    ~agent_upper.isin(
        {
            "INVESTIGATIVE",
            "INVESTIGATIONAL DRUG",
        }
    )
].copy()

###############################################################################
# BPC treatment file is systemic-only in the audited public export.
###############################################################################

treatment_types = sorted(
    set(
        treatment[
            "TREATMENT_TYPE"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )
)

if any(
    value
    and value
    != "Systemic Therapy"
    for value
    in treatment_types
):

    raise RuntimeError(
        "Unexpected BPC treatment type after exact cancer/metastatic filter: "
        f"{treatment_types}"
    )


###############################################################################
# Progression / imaging
###############################################################################

imaging = pd.read_csv(
    IMG_PATH,
    sep="\t",
    comment="#",
    low_memory=False,
)

imaging[
    "_patient"
] = (
    imaging[
        "PATIENT_ID"
    ]
    .astype(str)
    .str.strip()
)

imaging[
    "_day"
] = pd.to_numeric(
    imaging[
        "START_DATE"
    ],
    errors="coerce",
)

imaging[
    "_stage4"
] = imaging[
    "_patient"
].map(
    lambda patient:
        bool(
            patient_info.get(
                patient,
                {},
            ).get(
                "is_stage4",
                False,
            )
        )
)

imaging[
    "_meta_day"
] = imaging[
    "_patient"
].map(
    lambda patient:
        patient_info.get(
            patient,
            {},
        ).get(
            "site_specific_first_meta_day",
            np.nan,
        )
)

img_keep = (
    imaging[
        "_patient"
    ].isin(
        eligible_patients
    )
    & imaging[
        "_day"
    ].notna()
    & (
        imaging[
            "_stage4"
        ]
        |
        (
            imaging[
                "_meta_day"
            ].notna()
            & (
                imaging[
                    "_day"
                ]
                >= imaging[
                    "_meta_day"
                ]
            )
        )
    )
)

imaging = imaging.loc[
    img_keep
].copy()


###############################################################################
# Strip internal helper columns before frozen line builder.
###############################################################################

treatment_for_builder = treatment.drop(
    columns=[
        column
        for column in treatment.columns
        if column.startswith(
            "_"
        )
    ],
    errors="ignore",
)

imaging_for_builder = imaging.drop(
    columns=[
        column
        for column in imaging.columns
        if column.startswith(
            "_"
        )
    ],
    errors="ignore",
)


###############################################################################
# 15. Frozen line algorithm.
###############################################################################

a4b = import_module(
    "ckpt7a4b_fix9_base",
    ROOT
    / "scripts/dynamic_scan/"
    "ckpt7a4b_external_prediction_freeze.py",
)

(
    candidate_lines,
    candidate_lookup,
    line_audit,
) = a4b.build_line_intervals(
    treatment_for_builder,
    imaging_for_builder,
    eligible_patients,
)

print(
    "[CKPT7A4B_FIX9_STREAM_COUNTS]",
    {
        "treatment_rows":
            len(
                treatment_for_builder
            ),

        "treatment_patients":
            int(
                treatment_for_builder[
                    "PATIENT_ID"
                ].nunique()
            ),

        "investigational_rows_removed":
            investigational_rows,

        "imaging_rows":
            len(
                imaging_for_builder
            ),

        "imaging_patients":
            int(
                imaging_for_builder[
                    "PATIENT_ID"
                ].nunique()
            ),
    },
    flush=True,
)


###############################################################################
# 16. Aligned scan positive control.
###############################################################################

bpc_scans = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint7a3/"
    "bpc_w3_predictor_episodes.parquet"
)

scan_pid = find_col(
    bpc_scans.columns,
    [
        "patient_id",
        "PATIENT_ID",
    ],
)

scan_site = find_col(
    bpc_scans.columns,
    [
        "site",
        "institution",
    ],
)

scan_day = find_col(
    bpc_scans.columns,
    [
        "episode_end_day",
        "landmark_day",
    ],
)

scan_episode = find_col(
    bpc_scans.columns,
    [
        "bpc_episode_id",
        "scan_episode_id",
    ],
)

bpc_scans = bpc_scans[
    bpc_scans[
        scan_site
    ]
    .astype(str)
    .str.upper()
    .eq(
        "MSK"
    )
].copy()

bpc_scans[
    "bpc_patient_id"
] = (
    bpc_scans[
        scan_pid
    ]
    .astype(str)
    .str.strip()
)

bpc_scans[
    "bpc_day"
] = pd.to_numeric(
    bpc_scans[
        scan_day
    ],
    errors="coerce",
)

bpc_scans[
    "bpc_episode_id"
] = (
    bpc_scans[
        scan_episode
    ]
    .astype(str)
)

bpc_scans = bpc_scans[
    bpc_scans[
        "bpc_patient_id"
    ].isin(
        eligible_patients
    )
    & bpc_scans[
        "bpc_day"
    ].notna()
].copy()

assigned = []

for _, row in bpc_scans.iterrows():

    line, line_start, _ = a4b.assign_line(
        row[
            "bpc_patient_id"
        ],
        float(
            row[
                "bpc_day"
            ]
        ),
        candidate_lookup,
    )

    assigned.append(
        (
            int(
                line
            ),
            float(
                line_start
            )
            if np.isfinite(
                line_start
            )
            else np.nan,
        )
    )

bpc_scans[
    "bpc_line"
] = [
    item[
        0
    ]
    for item in assigned
]

bpc_scans[
    "bpc_line_start"
] = [
    item[
        1
    ]
    for item in assigned
]

bpc_scans = bpc_scans[
    bpc_scans[
        "bpc_line"
    ]
    > 0
].copy()

bpc_scans[
    "metastatic_stratum"
] = bpc_scans[
    "bpc_patient_id"
].map(
    lambda patient:
        (
            "STAGE_IV"
            if patient
            in stage4_patients
            else "RECURRENT"
        )
)

bpc_scans = bpc_scans.merge(
    offsets[
        [
            "bpc_patient_id",
            "chord_patient_id",
            "clock_offset",
        ]
    ],
    on="bpc_patient_id",
    how="inner",
    validate="many_to_one",
)

bpc_scans[
    "aligned_day"
] = (
    bpc_scans[
        "bpc_day"
    ]
    + bpc_scans[
        "clock_offset"
    ]
)


###############################################################################
# CHORD scan table.
###############################################################################

chord_scans = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint1/canonical/"
    "chord_mbc_scan_episodes_w3.parquet"
)

cpid = find_col(
    chord_scans.columns,
    [
        "patient_id",
        "PATIENT_ID",
    ],
)

cday = find_col(
    chord_scans.columns,
    [
        "episode_end_day",
        "landmark_day",
    ],
)

cline = find_col(
    chord_scans.columns,
    [
        "treatment_line",
        "line",
    ],
)

chord_scan = pd.DataFrame(
    {
        "chord_patient_id":
            chord_scans[
                cpid
            ]
            .astype(str)
            .str.strip(),

        "chord_day":
            pd.to_numeric(
                chord_scans[
                    cday
                ],
                errors="coerce",
            ),

        "chord_line":
            pd.to_numeric(
                chord_scans[
                    cline
                ],
                errors="coerce",
            ),
    }
).dropna()

chord_groups = {
    patient:
        group.sort_values(
            "chord_day",
            kind="mergesort",
        ).reset_index(
            drop=True
        )
    for patient, group
    in chord_scan.groupby(
        "chord_patient_id",
        observed=True,
    )
}

rows = []

for _, row in bpc_scans.iterrows():

    group = chord_groups.get(
        row[
            "chord_patient_id"
        ]
    )

    if (
        group is None
        or group.empty
    ):

        continue

    days = group[
        "chord_day"
    ].to_numpy(
        dtype=float
    )

    distance = np.abs(
        days
        - float(
            row[
                "aligned_day"
            ]
        )
    )

    index = int(
        np.argmin(
            distance
        )
    )

    delta = float(
        distance[
            index
        ]
    )

    if delta > 3.0:
        continue

    chosen = group.iloc[
        index
    ]

    rows.append(
        {
            "bpc_patient_id":
                row[
                    "bpc_patient_id"
                ],

            "chord_patient_id":
                row[
                    "chord_patient_id"
                ],

            "bpc_episode_id":
                row[
                    "bpc_episode_id"
                ],

            "metastatic_stratum":
                row[
                    "metastatic_stratum"
                ],

            "bpc_day":
                float(
                    row[
                        "bpc_day"
                    ]
                ),

            "chord_day":
                float(
                    chosen[
                        "chord_day"
                    ]
                ),

            "abs_day_delta":
                delta,

            "bpc_line":
                int(
                    row[
                        "bpc_line"
                    ]
                ),

            "bpc_line_start":
                float(
                    row[
                        "bpc_line_start"
                    ]
                ),

            "chord_line":
                int(
                    chosen[
                        "chord_line"
                    ]
                ),
        }
    )

matches = pd.DataFrame(
    rows
)

if matches.empty:

    raise RuntimeError(
        "No aligned scan matches."
    )


###############################################################################
# 17. Frozen CHORD line starts and exact CKPT5 context error.
###############################################################################

chord_lines = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint1/canonical/"
    "chord_mbc_lines.parquet"
)

lpid = find_col(
    chord_lines.columns,
    [
        "patient_id",
        "PATIENT_ID",
    ],
)

lnum = find_col(
    chord_lines.columns,
    [
        "treatment_line",
        "line",
    ],
)

lstart = find_col(
    chord_lines.columns,
    [
        "line_start_day",
        "LINE_START",
    ],
)

lookup = {}

for _, row in chord_lines.iterrows():

    number = pd.to_numeric(
        pd.Series(
            [
                row[
                    lnum
                ]
            ]
        ),
        errors="coerce",
    ).iloc[
        0
    ]

    start = pd.to_numeric(
        pd.Series(
            [
                row[
                    lstart
                ]
            ]
        ),
        errors="coerce",
    ).iloc[
        0
    ]

    if (
        pd.isna(
            number
        )
        or pd.isna(
            start
        )
    ):

        continue

    lookup[
        (
            str(
                row[
                    lpid
                ]
            ),
            int(
                number
            ),
        )
    ] = float(
        start
    )

matches[
    "chord_line_start"
] = [
    lookup.get(
        (
            patient,
            int(
                line
            ),
        ),
        np.nan,
    )
    for patient, line
    in zip(
        matches[
            "chord_patient_id"
        ],
        matches[
            "chord_line"
        ],
    )
]

matches = matches[
    matches[
        "chord_line_start"
    ].notna()
].copy()

matches[
    "line_equal"
] = (
    matches[
        "bpc_line"
    ]
    == matches[
        "chord_line"
    ]
)

matches[
    "bpc_line_scaled"
] = (
    np.clip(
        matches[
            "bpc_line"
        ],
        1,
        10,
    )
    / 10.0
)

matches[
    "chord_line_scaled"
] = (
    np.clip(
        matches[
            "chord_line"
        ],
        1,
        10,
    )
    / 10.0
)

matches[
    "bpc_elapsed_scaled"
] = (
    np.clip(
        matches[
            "bpc_day"
        ]
        - matches[
            "bpc_line_start"
        ],
        0,
        1460,
    )
    / 365.0
)

matches[
    "chord_elapsed_scaled"
] = (
    np.clip(
        matches[
            "chord_day"
        ]
        - matches[
            "chord_line_start"
        ],
        0,
        1460,
    )
    / 365.0
)

matches[
    "line_scaled_error"
] = np.abs(
    matches[
        "bpc_line_scaled"
    ]
    - matches[
        "chord_line_scaled"
    ]
)

matches[
    "elapsed_scaled_error"
] = np.abs(
    matches[
        "bpc_elapsed_scaled"
    ]
    - matches[
        "chord_elapsed_scaled"
    ]
)


###############################################################################
# 18. Transition geometry.
###############################################################################


def transition_metrics(
    frame,
):

    transition_rows = []

    for patient, group in frame.groupby(
        "bpc_patient_id",
        observed=True,
    ):

        group = group.sort_values(
            [
                "bpc_day",
                "bpc_episode_id",
            ],
            kind="mergesort",
        ).reset_index(
            drop=True
        )

        for index in range(
            1,
            len(
                group
            ),
        ):

            previous = group.iloc[
                index
                - 1
            ]

            current = group.iloc[
                index
            ]

            bpc_changed = (
                current[
                    "bpc_line"
                ]
                != previous[
                    "bpc_line"
                ]
            )

            chord_changed = (
                current[
                    "chord_line"
                ]
                != previous[
                    "chord_line"
                ]
            )

            transition_rows.append(
                bool(
                    bpc_changed
                    == chord_changed
                )
            )

    if not transition_rows:

        return None

    return float(
        np.mean(
            transition_rows
        )
    )


def line_metrics(
    frame,
):

    if frame.empty:

        return {
            "rows":
                0,
        }

    return {
        "rows":
            int(
                len(
                    frame
                )
            ),

        "patients":
            int(
                frame[
                    "bpc_patient_id"
                ].nunique()
            ),

        "line_agreement":
            float(
                frame[
                    "line_equal"
                ].mean()
            ),

        "line_number_scaled_mae":
            float(
                frame[
                    "line_scaled_error"
                ].mean()
            ),

        "elapsed_on_line_scaled_mae":
            float(
                frame[
                    "elapsed_scaled_error"
                ].mean()
            ),

        "transition_agreement":
            transition_metrics(
                frame
            ),

        "median_abs_scan_day_delta":
            float(
                frame[
                    "abs_day_delta"
                ].median()
            ),
    }


overall_metrics = line_metrics(
    matches
)

stage4_metrics = line_metrics(
    matches[
        matches[
            "metastatic_stratum"
        ]
        == "STAGE_IV"
    ]
)

recurrent_metrics = line_metrics(
    matches[
        matches[
            "metastatic_stratum"
        ]
        == "RECURRENT"
    ]
)


###############################################################################
# 19. Line-1 start comparison.
###############################################################################

candidate_line1 = candidate_lines[
    candidate_lines[
        "treatment_line"
    ]
    == 1
][
    [
        "patient_id",
        "line_start_day",
    ]
].copy()

candidate_line1 = candidate_line1.rename(
    columns={
        "patient_id":
            "bpc_patient_id",

        "line_start_day":
            "bpc_line1_start",
    }
)

candidate_line1[
    "metastatic_stratum"
] = candidate_line1[
    "bpc_patient_id"
].map(
    lambda patient:
        (
            "STAGE_IV"
            if patient
            in stage4_patients
            else "RECURRENT"
        )
)

candidate_line1 = candidate_line1.merge(
    offsets[
        [
            "bpc_patient_id",
            "chord_patient_id",
            "clock_offset",
        ]
    ],
    on="bpc_patient_id",
    how="inner",
    validate="one_to_one",
)

candidate_line1[
    "aligned_bpc_line1_start"
] = (
    candidate_line1[
        "bpc_line1_start"
    ]
    + candidate_line1[
        "clock_offset"
    ]
)

chord_line1 = chord_lines[
    pd.to_numeric(
        chord_lines[
            lnum
        ],
        errors="coerce",
    )
    == 1
][
    [
        lpid,
        lstart,
    ]
].copy()

chord_line1 = chord_line1.rename(
    columns={
        lpid:
            "chord_patient_id",

        lstart:
            "chord_line1_start",
    }
)

line1 = candidate_line1.merge(
    chord_line1,
    on="chord_patient_id",
    how="inner",
    validate="one_to_one",
)

line1[
    "delta"
] = (
    line1[
        "aligned_bpc_line1_start"
    ]
    - line1[
        "chord_line1_start"
    ]
)

line1[
    "abs_delta"
] = np.abs(
    line1[
        "delta"
    ]
)


def line1_metrics(
    frame,
):

    if frame.empty:

        return {
            "rows":
                0,
        }

    return {
        "rows":
            int(
                len(
                    frame
                )
            ),

        "delta":
            numeric_stats(
                frame[
                    "delta"
                ]
            ),

        "within3":
            float(
                (
                    frame[
                        "abs_delta"
                    ]
                    <= 3
                ).mean()
            ),

        "within7":
            float(
                (
                    frame[
                        "abs_delta"
                    ]
                    <= 7
                ).mean()
            ),

        "within30":
            float(
                (
                    frame[
                        "abs_delta"
                    ]
                    <= 30
                ).mean()
            ),
    }


line1_all = line1_metrics(
    line1
)

line1_stage4 = line1_metrics(
    line1[
        line1[
            "metastatic_stratum"
        ]
        == "STAGE_IV"
    ]
)

line1_recurrent = line1_metrics(
    line1[
        line1[
            "metastatic_stratum"
        ]
        == "RECURRENT"
    ]
)


###############################################################################
# 20. Decision.
#
# Keep the original high semantic-fidelity gate. Do not lower it because of
# observed positive-control performance.
###############################################################################

strong = (
    overall_metrics[
        "line_agreement"
    ]
    >= 0.90

    and overall_metrics[
        "transition_agreement"
    ]
    is not None

    and overall_metrics[
        "transition_agreement"
    ]
    >= 0.90

    and overall_metrics[
        "line_number_scaled_mae"
    ]
    <= 0.02

    and overall_metrics[
        "elapsed_on_line_scaled_mae"
    ]
    <= 0.10
)

if strong:

    status = (
        "PASS_EXACT_CKPT1_EXTERNAL_LINE_CONTEXT_BRIDGE"
    )

elif (
    overall_metrics[
        "transition_agreement"
    ]
    is not None
    and overall_metrics[
        "transition_agreement"
    ]
    >= 0.90
):

    status = (
        "LINE_BOUNDARIES_SIMILAR_ABSOLUTE_CONTEXT_NOT_TRANSPORTABLE"
    )

else:

    status = (
        "LINE_CONTEXT_NOT_TRANSPORTABLE"
    )


###############################################################################
# 21. Persist.
###############################################################################

report = {
    "status":
        status,

    "external_outcome_rows_opened":
        False,

    "external_outcome_distributions_inspected":
        False,

    "development_population":
        "MSK_ONLY",

    "source_columns_materialized":
        source_columns_materialized,

    "derived_helper_columns_are_not_source_columns":
        True,

    "protocol_amendment_A2":
        str(
            protocol_path.relative_to(
                ROOT
            )
        ),

    "protocol_amendment_A2_sha256":
        protocol_sha,

    "semantics": {
        "stage4":
            "UNFILTERED_BY_FIRST_METASTASIS",

        "recurrent":
            "FILTER_AT_SITE_SPECIFIC_FIRST_METASTASIS",

        "excluded_metastatic_sites":
            [
                "OTHER",
                "LYMPH NODES",
            ],

        "investigational_agent_excluded":
            True,
    },

    "recurrent_anchor_site_specific":
        site_anchor_metrics,

    "recurrent_anchor_overall_dx_to_dmets":
        overall_anchor_metrics,

    "line_context": {
        "overall":
            overall_metrics,

        "stage4":
            stage4_metrics,

        "recurrent":
            recurrent_metrics,
    },

    "line1_start": {
        "overall":
            line1_all,

        "stage4":
            line1_stage4,

        "recurrent":
            line1_recurrent,
    },

    "line_builder_audit":
        line_audit,

    "next_action": (
        "Do not patch A4B unless status is "
        "PASS_EXACT_CKPT1_EXTERNAL_LINE_CONTEXT_BRIDGE. "
        "If transition geometry remains high but absolute context fails, "
        "treat frozen line-number/time-on-line features as a documented "
        "cross-source transport limitation rather than continuing to tune "
        "their reconstruction."
    ),
}

atomic_json(
    OUT
    / "audit.json",
    report,
)

atomic_parquet(
    OUT
    / "recurrent_metastatic_anchor_comparison.parquet",
    anchor_compare,
)

atomic_parquet(
    OUT
    / "msk_scan_line_context_comparison.parquet",
    matches,
)

atomic_parquet(
    OUT
    / "msk_line1_start_comparison.parquet",
    line1,
)

atomic_parquet(
    OUT
    / "variable_synopsis_site_metadata.parquet",
    doc,
)


###############################################################################
# 22. Output.
###############################################################################

print("")
print(
    "========== CKPT7A4B FIX9 SUMMARY =========="
)

print(
    "status="
    + status
)

print(
    "external_outcome_rows_opened=False"
)

print(
    "external_outcome_distributions_inspected=False"
)

print(
    "development_population=MSK_ONLY"
)

print(
    "stage4_patients="
    + str(
        len(
            stage4_patients
        )
    )
)

print(
    "recurrent_patients="
    + str(
        len(
            recurrent_patients
        )
    )
)

print(
    "investigational_rows_removed="
    + str(
        investigational_rows
    )
)

print(
    "========== CKPT7A4B FIX9 SUMMARY END =========="
)

print("")
print(
    "========== CKPT7A4B FIX9 DECISION PACKET =========="
)

print(
    "recurrent_anchor_site_specific="
    + json.dumps(
        site_anchor_metrics,
        sort_keys=True,
    )
)

print(
    "recurrent_anchor_overall="
    + json.dumps(
        overall_anchor_metrics,
        sort_keys=True,
    )
)

print(
    "line_context_overall="
    + json.dumps(
        overall_metrics,
        sort_keys=True,
    )
)

print(
    "line_context_stage4="
    + json.dumps(
        stage4_metrics,
        sort_keys=True,
    )
)

print(
    "line_context_recurrent="
    + json.dumps(
        recurrent_metrics,
        sort_keys=True,
    )
)

print(
    "line1_overall="
    + json.dumps(
        line1_all,
        sort_keys=True,
    )
)

print(
    "line1_stage4="
    + json.dumps(
        line1_stage4,
        sort_keys=True,
    )
)

print(
    "line1_recurrent="
    + json.dumps(
        line1_recurrent,
        sort_keys=True,
    )
)

print(
    "line_builder_audit="
    + json.dumps(
        line_audit,
        sort_keys=True,
        default=str,
    )
)

print(
    "status="
    + status
)

print(
    "next_action="
    + report[
        "next_action"
    ]
)

print(
    "========== CKPT7A4B FIX9 DECISION PACKET END =========="
)

