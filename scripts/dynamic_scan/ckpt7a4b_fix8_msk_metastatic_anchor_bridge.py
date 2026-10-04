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
    / "artifacts/checkpoint7a4b_fix8_metastatic_anchor"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CANCER_PATH = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public/"
    "clinical_data/cancer_level_dataset_index.csv"
)

DIAGNOSIS_PATH = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public/"
    "cBioPortal_files/data_timeline_cancer_diagnosis.txt"
)

TREATMENT_PATH = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public/"
    "cBioPortal_files/data_timeline_treatment.txt"
)

IMAGING_PATH = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public/"
    "cBioPortal_files/data_timeline_imaging.txt"
)


###############################################################################
# Utility
###############################################################################


def norm(value: Any) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def clean(value: Any) -> str:

    if value is None:
        return ""

    value = str(
        value
    ).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
    }:
        return ""

    return value


def key(value: Any) -> str:

    value = clean(
        value
    )

    if not value:
        return ""

    try:

        number = float(
            value
        )

        if np.isfinite(
            number
        ) and number.is_integer():

            return str(
                int(
                    number
                )
            )

    except Exception:

        pass

    return value.upper()


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
            f"Missing {candidates}; columns={list(columns)}"
        )

    return None


def atomic_json(
    path: Path,
    payload: Any,
):

    temporary = Path(
        str(path)
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    json.loads(
        temporary.read_text(
            encoding="utf-8"
        )
    )

    temporary.replace(
        path
    )


def atomic_parquet(
    path: Path,
    frame: pd.DataFrame,
):

    temporary = Path(
        str(path)
        + ".tmp"
    )

    frame.to_parquet(
        temporary,
        index=False,
    )

    check = pd.read_parquet(
        temporary
    )

    if len(
        check
    ) != len(
        frame
    ):

        raise RuntimeError(
            f"Parquet verification failed: {path}"
        )

    temporary.replace(
        path
    )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for block in iter(
            lambda:
                handle.read(
                    1024
                    * 1024
                ),
            b"",
        ):

            digest.update(
                block
            )

    return digest.hexdigest()


def stats(
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

    if len(
        values
    ) == 0:

        return {
            "n": 0,
        }

    return {
        "n":
            int(
                len(
                    values
                )
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
    name: str,
    path: Path,
):

    specification = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):

        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    sys.modules[
        specification.name
    ] = module

    specification.loader.exec_module(
        module
    )

    return module


###############################################################################
# 1. Verify FIX7B metadata gate.
###############################################################################

fix7b = json.load(
    open(
        ROOT
        / "artifacts/checkpoint7a4b_fix7b_predictor_metadata/"
        "audit.json",
        encoding="utf-8",
    )
)

if fix7b[
    "status"
] != "METASTATIC_AND_REGIMEN_METADATA_READY_FOR_REVIEW":

    raise RuntimeError(
        "FIX7B metadata gate not satisfied."
    )

if fix7b[
    "external_outcome_rows_opened"
] is not False:

    raise RuntimeError(
        "Unexpected external outcome access before FIX8."
    )


###############################################################################
# 2. HEADER ONLY, then freeze exact protocol amendment.
#
# IMPORTANT:
# No patient row has been read from the quarantined cancer table yet.
###############################################################################

header = pd.read_csv(
    CANCER_PATH,
    nrows=0,
)

WHITELIST = [
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

missing = [
    column
    for column in WHITELIST
    if column not in header.columns
]

if missing:

    raise RuntimeError(
        f"Whitelist columns absent: {missing}"
    )

PROHIBITED = sorted(
    set(
        header.columns
    )
    - set(
        WHITELIST
    )
)

protocol_amendment = {
    "protocol":
        "CKPT6E_EXTERNAL_VALIDATION_PROTOCOL_AMENDMENT_A1",

    "purpose":
        (
            "Recover metastatic-entry timing needed to transport the "
            "already-frozen CKPT1 treatment-line context into BPC."
        ),

    "scientific_basis":
        {
            "stage_dx_iv":
                (
                    "Public Variable Synopsis: Derived Stage IV at Diagnosis."
                ),

            "dx_to_dmets_days":
                (
                    "Public Variable Synopsis: Time (Days) from Diagnosis "
                    "of Stage I-III to Distant Metastasis."
                ),

            "dmets_post_dx":
                (
                    "Public Variable Synopsis: Distant Metastasis Post "
                    "Diagnosis."
                ),
        },

    "source":
        str(
            CANCER_PATH.relative_to(
                ROOT
            )
        ),

    "source_sha256":
        sha256_file(
            CANCER_PATH
        ),

    "allowed_columns":
        WHITELIST,

    "prohibited_columns":
        PROHIBITED,

    "row_access_rule":
        (
            "Only allowed_columns may be materialized. No other column "
            "from this mixed predictor/outcome table may be read into a "
            "dataframe."
        ),

    "development_rule":
        (
            "Only MSK positive-control patients may influence decisions "
            "about metastatic-anchor or line semantics. DFCI/VICC "
            "predictor distributions must not be used for adapter "
            "selection."
        ),

    "availability_rule":
        (
            "Metastatic-entry date may be used only for landmarks on or "
            "after metastatic entry. It is historical state, not a future "
            "PFS/death/censoring endpoint."
        ),

    "model_selection":
        False,

    "alpha_tuning":
        False,

    "outcome_access":
        False,

    "external_outcome_distributions_inspected":
        False,
}

amendment_path = (
    OUT
    / "protocol_amendment_A1.json"
)

atomic_json(
    amendment_path,
    protocol_amendment,
)

amendment_sha = sha256_file(
    amendment_path
)

print(
    "[CKPT7A4B_FIX8_PROTOCOL_AMENDMENT_FROZEN]",
    amendment_sha,
    flush=True,
)

print(
    "[CKPT7A4B_FIX8_WHITELIST]",
    WHITELIST,
    flush=True,
)

print(
    "[CKPT7A4B_FIX8_PROHIBITED_COLUMN_COUNT]",
    len(
        PROHIBITED
    ),
    flush=True,
)


###############################################################################
# 3. FIRST AUTHORIZED PATIENT-ROW ACCESS.
#
# usecols is exact. No OS/PFS/death/censor/TTNT/etc. column is loaded.
###############################################################################

cancer = pd.read_csv(
    CANCER_PATH,
    usecols=WHITELIST,
    low_memory=False,
)

if list(
    cancer.columns
) != WHITELIST:

    ###########################################################################
    # pandas preserves file order, not usecols order. Compare as sets.
    ###########################################################################

    if set(
        cancer.columns
    ) != set(
        WHITELIST
    ):

        raise RuntimeError(
            "Materialized cancer columns exceed frozen whitelist."
        )

if any(
    column
    not in WHITELIST
    for column in cancer.columns
):

    raise RuntimeError(
        "A prohibited column was materialized."
    )

print(
    "[CKPT7A4B_FIX8_PREDICTOR_ONLY_ROW_ACCESS_PASS]",
    {
        "materialized_columns":
            sorted(
                cancer.columns
            ),

        "outcome_columns_materialized":
            0,
    },
    flush=True,
)


###############################################################################
# 4. Frozen MSK identity/clock bridge.
###############################################################################

offsets = pd.read_parquet(
    ROOT
    / "artifacts/checkpoint7a3g/"
    "msk_patient_clock_offsets.parquet"
).copy()

required_offset = {
    "bpc_patient_id",
    "chord_patient_id",
    "clock_offset",
}

if not required_offset.issubset(
    offsets.columns
):

    raise RuntimeError(
        f"Unexpected clock-offset schema: {list(offsets.columns)}"
    )

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

msk_ids = set(
    offsets[
        "bpc_patient_id"
    ]
)

cancer[
    "_record"
] = (
    cancer[
        "record_id"
    ]
    .astype(str)
    .str.strip()
)

cancer[
    "_institution"
] = (
    cancer[
        "institution"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

###############################################################################
# We inspect only MSK for the development positive control.
###############################################################################

cancer_msk = cancer[
    cancer[
        "_record"
    ].isin(
        msk_ids
    )
].copy()

direct_patient_coverage = (
    cancer_msk[
        "_record"
    ].nunique()
    / max(
        len(
            msk_ids
        ),
        1,
    )
)

print(
    "[CKPT7A4B_FIX8_CANCER_PATIENT_IDENTITY]",
    {
        "frozen_msk_patients":
            len(
                msk_ids
            ),

        "exact_record_id_matches":
            int(
                cancer_msk[
                    "_record"
                ].nunique()
            ),

        "coverage":
            float(
                direct_patient_coverage
            ),
    },
    flush=True,
)

if direct_patient_coverage < 0.90:

    raise RuntimeError(
        "Cancer-level record_id does not directly bridge to frozen BPC-MSK "
        f"patient IDs with sufficient coverage: {direct_patient_coverage}"
    )


###############################################################################
# 5. Safe public diagnosis timeline establishes the absolute BPC diagnosis day
#    and exact index-cancer key.
###############################################################################

diagnosis = pd.read_csv(
    DIAGNOSIS_PATH,
    sep="\t",
    comment="#",
    low_memory=False,
)

required_diagnosis = {
    "PATIENT_ID",
    "START_DATE",
    "CANCER_NO",
    "INDEX_CANCER",
    "STAGE_DX",
}

if not required_diagnosis.issubset(
    diagnosis.columns
):

    raise RuntimeError(
        "Diagnosis timeline missing required columns: "
        f"{sorted(required_diagnosis - set(diagnosis.columns))}"
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
        msk_ids
    )
].copy()

diagnosis[
    "_start"
] = pd.to_numeric(
    diagnosis[
        "START_DATE"
    ],
    errors="coerce",
)

diagnosis[
    "_cancer_key"
] = diagnosis[
    "CANCER_NO"
].map(
    key
)

index_values = (
    diagnosis[
        "INDEX_CANCER"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

print(
    "[CKPT7A4B_FIX8_INDEX_CANCER_VALUES]",
    index_values.value_counts().to_dict(),
    flush=True,
)

truthy = {
    "1",
    "Y",
    "YES",
    "TRUE",
    "T",
    "INDEX",
    "INDEX CANCER",
}

diagnosis[
    "_is_index"
] = index_values.isin(
    truthy
)

index_dx = diagnosis[
    diagnosis[
        "_is_index"
    ]
    & diagnosis[
        "_start"
    ].notna()
].copy()

###############################################################################
# Collapse only literal duplicate index rows.
###############################################################################

index_dx = index_dx.drop_duplicates(
    [
        "_patient",
        "_cancer_key",
        "_start",
        "STAGE_DX",
    ]
)

index_counts = (
    index_dx.groupby(
        "_patient",
        observed=True,
    )
    .size()
)

ambiguous_index_patients = set(
    index_counts[
        index_counts
        != 1
    ].index
)

index_dx = index_dx[
    ~index_dx[
        "_patient"
    ].isin(
        ambiguous_index_patients
    )
].copy()

print(
    "[CKPT7A4B_FIX8_INDEX_DIAGNOSIS]",
    {
        "patients":
            int(
                index_dx[
                    "_patient"
                ].nunique()
            ),

        "ambiguous_patients":
            int(
                len(
                    ambiguous_index_patients
                )
            ),
    },
    flush=True,
)

if index_dx[
    "_patient"
].nunique() < 300:

    raise RuntimeError(
        "Insufficient unique index-cancer diagnosis rows in MSK."
    )


###############################################################################
# 6. Exact cancer-key bridge:
#    cancer-level redcap_ca_index <-> cBioPortal CANCER_NO.
###############################################################################

###############################################################################
# Outcome-free FIX8B established the public BPC two-key cancer identity:
#
#   cancer_level.ca_seq
#       -> timeline diagnosis CANCER_NO
#
#   cancer_level.redcap_ca_index
#       -> timeline treatment REDCAP_CA_INDEX
#
# cancer_level_dataset_index may contain multiple cancer rows per patient;
# patient + ca_seq identifies the index cancer.
###############################################################################

cancer_msk[
    "_diagnosis_cancer_key"
] = cancer_msk[
    "ca_seq"
].map(
    key
)

cancer_msk[
    "_treatment_cancer_key"
] = cancer_msk[
    "redcap_ca_index"
].map(
    key
)

bridge = cancer_msk.merge(
    index_dx[
        [
            "_patient",
            "_cancer_key",
            "_start",
            "STAGE_DX",
        ]
    ],
    left_on=[
        "_record",
        "_diagnosis_cancer_key",
    ],
    right_on=[
        "_patient",
        "_cancer_key",
    ],
    how="inner",
)

bridge = bridge.rename(
    columns={
        "_start":
            "diagnosis_day",

        "STAGE_DX":
            "timeline_stage_dx",
    }
)

###############################################################################
# Require one analytic index-cancer row per patient.
###############################################################################

bridge = bridge.drop_duplicates()

bridge_counts = (
    bridge.groupby(
        "_record",
        observed=True,
    )
    .size()
)

ambiguous_bridge = set(
    bridge_counts[
        bridge_counts
        != 1
    ].index
)

bridge = bridge[
    ~bridge[
        "_record"
    ].isin(
        ambiguous_bridge
    )
].copy()

bridge_coverage = (
    bridge[
        "_record"
    ].nunique()
    / max(
        index_dx[
            "_patient"
        ].nunique(),
        1,
    )
)

print(
    "[CKPT7A4B_FIX8_CANCER_KEY_BRIDGE]",
    {
        "patients":
            int(
                bridge[
                    "_record"
                ].nunique()
            ),

        "coverage_of_unique_index_dx":
            float(
                bridge_coverage
            ),

        "ambiguous_patients":
            int(
                len(
                    ambiguous_bridge
                )
            ),
    },
    flush=True,
)

if bridge_coverage < 0.80:

    raise RuntimeError(
        "REDCAP cancer-key bridge insufficient: "
        f"{bridge_coverage}"
    )


###############################################################################
# 7. Derive metastatic entry using the frozen metadata-defined rule.
#
# De novo:
#   stage_dx_iv == Stage IV -> metastatic entry at diagnosis day.
#
# Recurrent:
#   dmets_post_dx == 1 AND dx_to_dmets_days finite/nonnegative
#   -> diagnosis day + dx_to_dmets_days.
###############################################################################

stage_iv = (
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

dmets_post = pd.to_numeric(
    bridge[
        "dmets_post_dx"
    ],
    errors="coerce",
)

dx_to_dmets = pd.to_numeric(
    bridge[
        "dx_to_dmets_days"
    ],
    errors="coerce",
)

bridge[
    "metastatic_entry_source"
] = ""

bridge[
    "metastatic_entry_day"
] = np.nan

bridge.loc[
    stage_iv,
    "metastatic_entry_day",
] = bridge.loc[
    stage_iv,
    "diagnosis_day",
].astype(float)

bridge.loc[
    stage_iv,
    "metastatic_entry_source",
] = "STAGE_IV_AT_DIAGNOSIS"

recurrent = (
    ~stage_iv
    & (
        dmets_post
        == 1
    )
    & dx_to_dmets.notna()
    & (
        dx_to_dmets
        >= 0
    )
)

bridge.loc[
    recurrent,
    "metastatic_entry_day",
] = (
    bridge.loc[
        recurrent,
        "diagnosis_day",
    ].astype(float)
    + dx_to_dmets.loc[
        recurrent
    ].astype(float)
)

bridge.loc[
    recurrent,
    "metastatic_entry_source",
] = "STAGE_I_III_TO_DISTANT_METASTASIS"

anchors = bridge[
    bridge[
        "metastatic_entry_day"
    ].notna()
].copy()

anchors = anchors.rename(
    columns={
        "_record":
            "bpc_patient_id",
    }
)

anchor_patients = set(
    anchors[
        "bpc_patient_id"
    ]
)

print(
    "[CKPT7A4B_FIX8_ANCHOR_COUNTS]",
    {
        "patients":
            int(
                anchors[
                    "bpc_patient_id"
                ].nunique()
            ),

        "sources":
            anchors[
                "metastatic_entry_source"
            ]
            .value_counts()
            .to_dict(),
    },
    flush=True,
)

if anchors[
    "bpc_patient_id"
].nunique() < 200:

    raise RuntimeError(
        "Too few MSK metastatic anchors."
    )


###############################################################################
# 8. Compare anchor directly with the frozen CHORD first-metastasis definition.
#
# This is the strongest test of whether the newly whitelisted variable means
# the same thing as CKPT1's metastatic-entry rule.
###############################################################################

chord_tumor_candidates = list(
    (
        ROOT
        / "data/external_sources/msk_chord_2024/raw"
    ).rglob(
        "data_timeline_tumor_sites.txt"
    )
)

if len(
    chord_tumor_candidates
) != 1:

    raise RuntimeError(
        f"Unexpected CHORD tumor-site files: {chord_tumor_candidates}"
    )

tumor = pd.read_csv(
    chord_tumor_candidates[
        0
    ],
    sep="\t",
    comment="#",
    low_memory=False,
)

tumor_pid = find_col(
    tumor.columns,
    [
        "PATIENT_ID",
    ],
)

tumor_day = find_col(
    tumor.columns,
    [
        "START_DATE",
    ],
)

tumor_site = find_col(
    tumor.columns,
    [
        "TUMOR_SITE",
    ],
)

tumor[
    "_patient"
] = (
    tumor[
        tumor_pid
    ]
    .astype(str)
    .str.strip()
)

tumor[
    "_day"
] = pd.to_numeric(
    tumor[
        tumor_day
    ],
    errors="coerce",
)

tumor[
    "_site"
] = (
    tumor[
        tumor_site
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

anchor_compare = anchors.merge(
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

anchor_compare[
    "anchor_delta_days"
] = (
    anchor_compare[
        "metastatic_entry_day"
    ]
    - anchor_compare[
        "expected_bpc_first_meta_day"
    ]
)

anchor_compare[
    "anchor_abs_delta_days"
] = np.abs(
    anchor_compare[
        "anchor_delta_days"
    ]
)

anchor_metrics = {
    "rows":
        int(
            len(
                anchor_compare
            )
        ),

    "patients":
        int(
            anchor_compare[
                "bpc_patient_id"
            ].nunique()
        ),

    "delta":
        stats(
            anchor_compare[
                "anchor_delta_days"
            ]
        ),

    "within0_fraction":
        float(
            (
                anchor_compare[
                    "anchor_abs_delta_days"
                ]
                == 0
            ).mean()
        ),

    "within3_fraction":
        float(
            (
                anchor_compare[
                    "anchor_abs_delta_days"
                ]
                <= 3
            ).mean()
        ),

    "within7_fraction":
        float(
            (
                anchor_compare[
                    "anchor_abs_delta_days"
                ]
                <= 7
            ).mean()
        ),

    "within30_fraction":
        float(
            (
                anchor_compare[
                    "anchor_abs_delta_days"
                ]
                <= 30
            ).mean()
        ),
}

print(
    "[CKPT7A4B_FIX8_METASTATIC_ANCHOR_POSITIVE_CONTROL]",
    json.dumps(
        anchor_metrics,
        sort_keys=True,
    ),
    flush=True,
)


###############################################################################
# 9. Build metastatic-only BPC treatment/progression inputs.
#
# Treatment rows must:
#   * belong to the exact index cancer key;
#   * occur on/after metastatic entry.
#
# Imaging progression rows:
#   * occur on/after metastatic entry.
###############################################################################

treatment = pd.read_csv(
    TREATMENT_PATH,
    sep="\t",
    comment="#",
    low_memory=False,
)

imaging = pd.read_csv(
    IMAGING_PATH,
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
    "_cancer_key"
] = treatment[
    "REDCAP_CA_INDEX"
].map(
    key
)

treatment[
    "_start"
] = pd.to_numeric(
    treatment[
        "START_DATE"
    ],
    errors="coerce",
)

anchor_lookup = (
    anchors.set_index(
        "bpc_patient_id"
    )[
        [
            "_treatment_cancer_key",
            "metastatic_entry_day",
        ]
    ]
    .to_dict(
        orient="index"
    )
)

treatment[
    "_anchor"
] = treatment[
    "_patient"
].map(
    lambda patient:
        anchor_lookup.get(
            patient,
            {},
        ).get(
            "metastatic_entry_day",
            np.nan,
        )
)

treatment[
    "_anchor_cancer_key"
] = treatment[
    "_patient"
].map(
    lambda patient:
        anchor_lookup.get(
            patient,
            {},
        ).get(
            "_treatment_cancer_key",
            "",
        )
)

tx_keep = (
    treatment[
        "_patient"
    ].isin(
        anchor_patients
    )
    & treatment[
        "_start"
    ].notna()
    & treatment[
        "_anchor"
    ].notna()
    & (
        treatment[
            "_cancer_key"
        ]
        == treatment[
            "_anchor_cancer_key"
        ]
    )
    & (
        treatment[
            "_start"
        ]
        >= treatment[
            "_anchor"
        ]
    )
)

treatment_meta = treatment.loc[
    tx_keep
].drop(
    columns=[
        "_patient",
        "_cancer_key",
        "_start",
        "_anchor",
        "_anchor_cancer_key",
    ],
    errors="ignore",
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
    "_start"
] = pd.to_numeric(
    imaging[
        "START_DATE"
    ],
    errors="coerce",
)

imaging[
    "_anchor"
] = imaging[
    "_patient"
].map(
    lambda patient:
        anchor_lookup.get(
            patient,
            {},
        ).get(
            "metastatic_entry_day",
            np.nan,
        )
)

image_keep = (
    imaging[
        "_patient"
    ].isin(
        anchor_patients
    )
    & imaging[
        "_start"
    ].notna()
    & imaging[
        "_anchor"
    ].notna()
    & (
        imaging[
            "_start"
        ]
        >= imaging[
            "_anchor"
        ]
    )
)

imaging_meta = imaging.loc[
    image_keep
].drop(
    columns=[
        "_patient",
        "_start",
        "_anchor",
    ],
    errors="ignore",
)

print(
    "[CKPT7A4B_FIX8_POSTMETA_STREAM_COUNTS]",
    {
        "patients_with_anchor":
            len(
                anchor_patients
            ),

        "treatment_rows":
            len(
                treatment_meta
            ),

        "treatment_patients":
            int(
                treatment_meta[
                    "PATIENT_ID"
                ].nunique()
            ),

        "imaging_rows":
            len(
                imaging_meta
            ),

        "imaging_patients":
            int(
                imaging_meta[
                    "PATIENT_ID"
                ].nunique()
            ),
    },
    flush=True,
)


###############################################################################
# 10. Frozen CKPT1-style line replay.
###############################################################################

a4b = import_module(
    "ckpt7a4b_fix8_base",
    ROOT
    / "scripts/dynamic_scan/"
    "ckpt7a4b_external_prediction_freeze.py",
)

(
    candidate_lines,
    candidate_lookup,
    line_builder_audit,
) = a4b.build_line_intervals(
    treatment_meta,
    imaging_meta,
    anchor_patients,
)


###############################################################################
# 11. Aligned W3 scan-level line positive control.
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

bpc = bpc_scans[
    bpc_scans[
        scan_site
    ]
    .astype(str)
    .str.upper()
    .eq(
        "MSK"
    )
].copy()

bpc[
    "bpc_patient_id"
] = (
    bpc[
        scan_pid
    ]
    .astype(str)
    .str.strip()
)

bpc[
    "bpc_day"
] = pd.to_numeric(
    bpc[
        scan_day
    ],
    errors="coerce",
)

bpc[
    "bpc_episode_id"
] = (
    bpc[
        scan_episode
    ]
    .astype(str)
)

bpc = bpc[
    bpc[
        "bpc_patient_id"
    ].isin(
        anchor_patients
    )
    & bpc[
        "bpc_day"
    ].notna()
].copy()

assigned_lines = []

assigned_starts = []

for _, row in bpc.iterrows():

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

    assigned_lines.append(
        int(
            line
        )
    )

    assigned_starts.append(
        float(
            line_start
        )
        if np.isfinite(
            line_start
        )
        else np.nan
    )

bpc[
    "bpc_line"
] = assigned_lines

bpc[
    "bpc_line_start"
] = assigned_starts

bpc = bpc[
    bpc[
        "bpc_line"
    ]
    > 0
].copy()

bpc = bpc.merge(
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

bpc[
    "aligned_day"
] = (
    bpc[
        "bpc_day"
    ]
    + bpc[
        "clock_offset"
    ]
)


###############################################################################
# CHORD W3.
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

chord_work = pd.DataFrame(
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
    in chord_work.groupby(
        "chord_patient_id",
        observed=True,
    )
}

matches = []

for _, row in bpc.iterrows():

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

    matches.append(
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

            "bpc_day":
                float(
                    row[
                        "bpc_day"
                    ]
                ),

            "aligned_day":
                float(
                    row[
                        "aligned_day"
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

match_frame = pd.DataFrame(
    matches
)

if match_frame.empty:

    raise RuntimeError(
        "No scan-level MSK matches after metastatic anchoring."
    )

match_frame[
    "line_equal"
] = (
    match_frame[
        "bpc_line"
    ]
    == match_frame[
        "chord_line"
    ]
)


###############################################################################
# 12. CHORD line-start context comparison.
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

line_start_lookup = (
    chord_lines[
        [
            lpid,
            lnum,
            lstart,
        ]
    ]
    .drop_duplicates()
)

line_start_lookup[
    "_key"
] = list(
    zip(
        line_start_lookup[
            lpid
        ].astype(str),
        pd.to_numeric(
            line_start_lookup[
                lnum
            ],
            errors="coerce",
        ).astype(
            "Int64"
        ),
    )
)

chord_start_dict = dict(
    zip(
        line_start_lookup[
            "_key"
        ],
        pd.to_numeric(
            line_start_lookup[
                lstart
            ],
            errors="coerce",
        ),
    )
)

match_frame[
    "chord_line_start"
] = [
    chord_start_dict.get(
        (
            str(
                patient
            ),
            int(
                line
            ),
        ),
        np.nan,
    )
    for patient, line
    in zip(
        match_frame[
            "chord_patient_id"
        ],
        match_frame[
            "chord_line"
        ],
    )
]

match_frame = match_frame[
    match_frame[
        "chord_line_start"
    ].notna()
].copy()

match_frame[
    "bpc_line_scaled"
] = (
    np.clip(
        match_frame[
            "bpc_line"
        ].astype(float),
        1,
        10,
    )
    / 10.0
)

match_frame[
    "chord_line_scaled"
] = (
    np.clip(
        match_frame[
            "chord_line"
        ].astype(float),
        1,
        10,
    )
    / 10.0
)

match_frame[
    "bpc_elapsed_scaled"
] = (
    np.clip(
        match_frame[
            "bpc_day"
        ]
        - match_frame[
            "bpc_line_start"
        ],
        0,
        1460,
    )
    / 365.0
)

match_frame[
    "chord_elapsed_scaled"
] = (
    np.clip(
        match_frame[
            "chord_day"
        ]
        - match_frame[
            "chord_line_start"
        ],
        0,
        1460,
    )
    / 365.0
)

match_frame[
    "line_scaled_abs_error"
] = np.abs(
    match_frame[
        "bpc_line_scaled"
    ]
    - match_frame[
        "chord_line_scaled"
    ]
)

match_frame[
    "elapsed_scaled_abs_error"
] = np.abs(
    match_frame[
        "bpc_elapsed_scaled"
    ]
    - match_frame[
        "chord_elapsed_scaled"
    ]
)


###############################################################################
# 13. Transition geometry.
###############################################################################

transition_rows = []

for patient, group in match_frame.groupby(
    "bpc_patient_id",
    observed=True,
):

    group = group.sort_values(
        [
            "aligned_day",
            "bpc_episode_id",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    for row_number in range(
        1,
        len(
            group
        ),
    ):

        previous = group.iloc[
            row_number
            - 1
        ]

        current = group.iloc[
            row_number
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
            {
                "bpc_patient_id":
                    patient,

                "bpc_changed":
                    bool(
                        bpc_changed
                    ),

                "chord_changed":
                    bool(
                        chord_changed
                    ),

                "equal":
                    bool(
                        bpc_changed
                        == chord_changed
                    ),
            }
        )

transitions = pd.DataFrame(
    transition_rows
)

transition_agreement = (
    float(
        transitions[
            "equal"
        ].mean()
    )
    if not transitions.empty
    else None
)


###############################################################################
# 14. Line-1 start comparison.
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
    "bpc_line1_start_aligned"
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

line1_compare = candidate_line1.merge(
    chord_line1,
    on="chord_patient_id",
    how="inner",
    validate="one_to_one",
)

line1_compare[
    "line1_delta_days"
] = (
    line1_compare[
        "bpc_line1_start_aligned"
    ]
    - line1_compare[
        "chord_line1_start"
    ]
)

line1_compare[
    "line1_abs_delta_days"
] = np.abs(
    line1_compare[
        "line1_delta_days"
    ]
)


###############################################################################
# 15. Metrics and status.
###############################################################################

line_agreement = float(
    match_frame[
        "line_equal"
    ].mean()
)

line_context_mae = float(
    match_frame[
        "line_scaled_abs_error"
    ].mean()
)

elapsed_context_mae = float(
    match_frame[
        "elapsed_scaled_abs_error"
    ].mean()
)

line1_metrics = {
    "rows":
        int(
            len(
                line1_compare
            )
        ),

    "delta":
        stats(
            line1_compare[
                "line1_delta_days"
            ]
        ),

    "within3_fraction":
        float(
            (
                line1_compare[
                    "line1_abs_delta_days"
                ]
                <= 3
            ).mean()
        ),

    "within7_fraction":
        float(
            (
                line1_compare[
                    "line1_abs_delta_days"
                ]
                <= 7
            ).mean()
        ),

    "within30_fraction":
        float(
            (
                line1_compare[
                    "line1_abs_delta_days"
                ]
                <= 30
            ).mean()
        ),
}

anchor_strong = (
    anchor_metrics[
        "patients"
    ]
    >= 100
    and anchor_metrics[
        "within7_fraction"
    ]
    >= 0.90
)

if (
    anchor_strong
    and line_agreement
    >= 0.80
    and transition_agreement is not None
    and transition_agreement
    >= 0.85
):

    status = (
        "STRONG_MSK_METASTATIC_ANCHOR_BRIDGE"
    )

elif anchor_strong:

    status = (
        "METASTATIC_ANCHOR_CONFIRMED_LINE_STREAM_LIMITED"
    )

else:

    status = (
        "METASTATIC_ANCHOR_NOT_CONFIRMED"
    )

report = {
    "status":
        status,

    "protocol_amendment":
        str(
            amendment_path.relative_to(
                ROOT
            )
        ),

    "protocol_amendment_sha256":
        amendment_sha,

    "external_outcome_rows_opened":
        False,

    "external_outcome_distributions_inspected":
        False,

    "quarantined_source":
        str(
            CANCER_PATH.relative_to(
                ROOT
            )
        ),

    "quarantined_allowed_columns":
        WHITELIST,

    "quarantined_materialized_columns":
        sorted(
            cancer.columns
        ),

    "quarantined_prohibited_columns_materialized":
        [],

    "development_population":
        "MSK_ONLY",

    "metastatic_anchor":
        anchor_metrics,

    "line_builder_audit":
        line_builder_audit,

    "scan_line_positive_control": {
        "rows":
            int(
                len(
                    match_frame
                )
            ),

        "patients":
            int(
                match_frame[
                    "bpc_patient_id"
                ].nunique()
            ),

        "line_agreement":
            line_agreement,

        "median_abs_day_delta":
            float(
                match_frame[
                    "abs_day_delta"
                ].median()
            ),

        "transition_agreement":
            transition_agreement,

        "line_number_scaled_mae":
            line_context_mae,

        "elapsed_on_line_scaled_mae":
            elapsed_context_mae,
    },

    "line1_start_positive_control":
        line1_metrics,

    "scientific_interpretation":
        (
            "Metastatic entry is taken from explicit BPC metastatic disease "
            "metadata rather than inferred from treatment history. "
            "Outcome-free FIX8B froze ca_seq as the diagnosis CANCER_NO "
            "bridge and redcap_ca_index as the treatment REDCAP_CA_INDEX "
            "bridge. Treatment line boundaries themselves remain the frozen "
            "CKPT1-style "
            "365-day/new-agent/progression-grace construction."
        ),

    "next_action": (
        "Do not modify A4B automatically. If metastatic-anchor and line-context "
        "positive controls are sufficiently strong, patch A4B to use this "
        "frozen predictor-only metastatic-entry bridge. Otherwise inspect "
        "remaining treatment-line transport mismatch without opening outcomes."
    ),
}

atomic_json(
    OUT
    / "audit.json",
    report,
)


###############################################################################
# 16. Persist MSK-only positive-control artifacts.
#
# Never persist DFCI/VICC predictor rows from the quarantined table here.
###############################################################################

anchor_save_columns = [
    "bpc_patient_id",
    "_diagnosis_cancer_key",
    "_treatment_cancer_key",
    "diagnosis_day",
    "stage_dx",
    "stage_dx_iv",
    "ca_dmets_yn",
    "dmets_post_dx",
    "dx_to_dmets_days",
    "metastatic_entry_day",
    "metastatic_entry_source",
]

atomic_parquet(
    OUT
    / "msk_metastatic_anchors.parquet",
    anchors[
        [
            column
            for column in anchor_save_columns
            if column in anchors.columns
        ]
    ],
)

atomic_parquet(
    OUT
    / "msk_anchor_chord_comparison.parquet",
    anchor_compare[
        [
            "bpc_patient_id",
            "chord_patient_id",
            "metastatic_entry_day",
            "metastatic_entry_source",
            "expected_bpc_first_meta_day",
            "anchor_delta_days",
            "anchor_abs_delta_days",
        ]
    ],
)

atomic_parquet(
    OUT
    / "msk_scan_line_comparison.parquet",
    match_frame,
)

atomic_parquet(
    OUT
    / "msk_line1_start_comparison.parquet",
    line1_compare,
)

if transitions.empty:

    transitions = pd.DataFrame(
        columns=[
            "bpc_patient_id",
            "bpc_changed",
            "chord_changed",
            "equal",
        ]
    )

atomic_parquet(
    OUT
    / "msk_line_transition_comparison.parquet",
    transitions,
)


###############################################################################
# 17. Output
###############################################################################

print("")
print(
    "========== CKPT7A4B FIX8 SUMMARY =========="
)

print(
    "status="
    + status
)

print(
    "protocol_amendment_sha256="
    + amendment_sha
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
    "metastatic_anchor_patients="
    + str(
        anchor_metrics[
            "patients"
        ]
    )
)

print(
    "metastatic_anchor_within7="
    + str(
        anchor_metrics[
            "within7_fraction"
        ]
    )
)

print(
    "scan_line_agreement="
    + str(
        line_agreement
    )
)

print(
    "transition_agreement="
    + str(
        transition_agreement
    )
)

print(
    "line_number_scaled_mae="
    + str(
        line_context_mae
    )
)

print(
    "elapsed_on_line_scaled_mae="
    + str(
        elapsed_context_mae
    )
)

print(
    "========== CKPT7A4B FIX8 SUMMARY END =========="
)

print("")
print(
    "========== CKPT7A4B FIX8 DECISION PACKET =========="
)

print(
    "metastatic_anchor="
    + json.dumps(
        anchor_metrics,
        sort_keys=True,
    )
)

print(
    "line1_start_positive_control="
    + json.dumps(
        line1_metrics,
        sort_keys=True,
    )
)

print(
    "scan_line_positive_control="
    + json.dumps(
        report[
            "scan_line_positive_control"
        ],
        sort_keys=True,
    )
)

print(
    "line_builder_audit="
    + json.dumps(
        line_builder_audit,
        sort_keys=True,
        default=str,
    )
)

print(
    "status="
    + status
)

print(
    "scientific_interpretation="
    + report[
        "scientific_interpretation"
    ]
)

print(
    "next_action="
    + report[
        "next_action"
    ]
)

print(
    "========== CKPT7A4B FIX8 DECISION PACKET END =========="
)


if __name__ == "__main__":
    pass
