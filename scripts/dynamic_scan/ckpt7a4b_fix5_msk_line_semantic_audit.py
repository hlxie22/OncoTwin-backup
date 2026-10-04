#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


###############################################################################
# Utilities
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

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "na",
        "null",
    }:
        return ""

    return text


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

        key = norm(candidate)

        if key in lookup:
            return lookup[key]

    if required:

        raise KeyError(
            f"Missing {candidates}; "
            f"available={list(columns)}"
        )

    return None


def import_module(
    name: str,
    path: Path,
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
            f"Cannot import {path}"
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


def atomic_json(
    path: Path,
    payload,
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


def stats(values):

    array = np.asarray(
        values,
        dtype=float,
    )

    array = array[
        np.isfinite(
            array
        )
    ]

    if len(array) == 0:
        return {
            "n": 0,
        }

    return {
        "n":
            int(
                len(array)
            ),

        "min":
            float(
                array.min()
            ),

        "p25":
            float(
                np.quantile(
                    array,
                    0.25,
                )
            ),

        "median":
            float(
                np.median(
                    array
                )
            ),

        "p75":
            float(
                np.quantile(
                    array,
                    0.75,
                )
            ),

        "p90":
            float(
                np.quantile(
                    array,
                    0.90,
                )
            ),

        "max":
            float(
                array.max()
            ),
    }


###############################################################################
# MSK clock-aligned scan positive-control surface
###############################################################################


def prepare_scan_matches(
    root: Path,
    a4b,
):

    offsets = pd.read_parquet(
        root
        / "artifacts/checkpoint7a3g/"
        "msk_patient_clock_offsets.parquet"
    )

    required = {
        "bpc_patient_id",
        "chord_patient_id",
        "clock_offset",
    }

    if not required.issubset(
        offsets.columns
    ):

        raise RuntimeError(
            f"Clock schema changed: {list(offsets.columns)}"
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

    bpc_scans = pd.read_parquet(
        root
        / "artifacts/checkpoint7a3/"
        "bpc_w3_predictor_episodes.parquet"
    )

    patient_col = find_col(
        bpc_scans.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    site_col = find_col(
        bpc_scans.columns,
        [
            "site",
            "institution",
        ],
    )

    day_col = find_col(
        bpc_scans.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
    )

    episode_col = find_col(
        bpc_scans.columns,
        [
            "bpc_episode_id",
            "scan_episode_id",
        ],
    )

    bpc = bpc_scans[
        bpc_scans[
            site_col
        ]
        .astype(str)
        .str.upper()
        .eq(
            "MSK"
        )
    ].copy()

    bpc[
        "_bpc_patient"
    ] = (
        bpc[
            patient_col
        ]
        .astype(str)
        .str.strip()
    )

    bpc[
        "_bpc_day"
    ] = pd.to_numeric(
        bpc[
            day_col
        ],
        errors="coerce",
    )

    bpc[
        "_episode"
    ] = (
        bpc[
            episode_col
        ]
        .astype(str)
    )

    bpc = bpc.merge(
        offsets[
            [
                "bpc_patient_id",
                "chord_patient_id",
                "clock_offset",
            ]
        ],
        left_on="_bpc_patient",
        right_on="bpc_patient_id",
        how="inner",
        validate="many_to_one",
    )

    bpc = bpc[
        bpc[
            "_bpc_day"
        ].notna()
    ].copy()

    bpc[
        "_aligned_day"
    ] = (
        bpc[
            "_bpc_day"
        ].astype(float)
        + bpc[
            "clock_offset"
        ].astype(float)
    )

    chord = pd.read_parquet(
        root
        / "artifacts/checkpoint1/canonical/"
        "chord_mbc_scan_episodes_w3.parquet"
    )

    cpid = find_col(
        chord.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    cday = find_col(
        chord.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
    )

    cline = find_col(
        chord.columns,
        [
            "treatment_line",
            "line",
        ],
    )

    chord = pd.DataFrame(
        {
            "_chord_patient":
                chord[
                    cpid
                ]
                .astype(str)
                .str.strip(),

            "_chord_day":
                pd.to_numeric(
                    chord[
                        cday
                    ],
                    errors="coerce",
                ),

            "_chord_line":
                pd.to_numeric(
                    chord[
                        cline
                    ],
                    errors="coerce",
                ),
        }
    ).dropna()

    chord_groups = {
        patient:
            group.sort_values(
                "_chord_day",
                kind="mergesort",
            ).reset_index(
                drop=True
            )
        for patient, group
        in chord.groupby(
            "_chord_patient",
            observed=True,
        )
    }

    matches = []

    for _, row in bpc.iterrows():

        group = chord_groups.get(
            str(
                row[
                    "chord_patient_id"
                ]
            )
        )

        if (
            group is None
            or group.empty
        ):
            continue

        days = group[
            "_chord_day"
        ].to_numpy(
            dtype=float
        )

        delta = np.abs(
            days
            - float(
                row[
                    "_aligned_day"
                ]
            )
        )

        idx = int(
            np.argmin(
                delta
            )
        )

        if float(
            delta[
                idx
            ]
        ) > 3.0:

            continue

        chosen = group.iloc[
            idx
        ]

        matches.append(
            {
                "bpc_patient_id":
                    row[
                        "_bpc_patient"
                    ],

                "chord_patient_id":
                    row[
                        "chord_patient_id"
                    ],

                "bpc_episode_id":
                    row[
                        "_episode"
                    ],

                "bpc_day":
                    float(
                        row[
                            "_bpc_day"
                        ]
                    ),

                "clock_offset":
                    float(
                        row[
                            "clock_offset"
                        ]
                    ),

                "aligned_day":
                    float(
                        row[
                            "_aligned_day"
                        ]
                    ),

                "chord_day":
                    float(
                        chosen[
                            "_chord_day"
                        ]
                    ),

                "abs_day_delta":
                    float(
                        delta[
                            idx
                        ]
                    ),

                "chord_line":
                    int(
                        chosen[
                            "_chord_line"
                        ]
                    ),
            }
        )

    result = pd.DataFrame(
        matches
    )

    if result.empty:

        raise RuntimeError(
            "No aligned MSK scan matches."
        )

    return (
        result,
        offsets,
    )


###############################################################################
# Diagnosis semantic inventory
###############################################################################


def diagnosis_inventory(
    diagnosis: pd.DataFrame,
    msk_patients: set[str],
):

    pid = find_col(
        diagnosis.columns,
        [
            "PATIENT_ID",
        ],
    )

    diagnosis = diagnosis.copy()

    diagnosis[
        "_patient"
    ] = (
        diagnosis[
            pid
        ]
        .astype(str)
        .str.strip()
    )

    diagnosis = diagnosis[
        diagnosis[
            "_patient"
        ].isin(
            msk_patients
        )
    ].copy()

    report = {}

    for column in (
        "CA_D_TYPE",
        "STAGE_DX",
        "INDEX_CANCER",
        "CA_TX_PRE_PATH_STAGE",
    ):

        if column not in diagnosis.columns:
            continue

        values = (
            diagnosis[
                column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        counts = (
            values[
                values.ne("")
            ]
            .value_counts()
        )

        report[
            column
        ] = {
            "nonblank_rows":
                int(
                    values.ne("").sum()
                ),

            "unique_values":
                int(
                    counts.size
                ),

            "counts":
                {
                    str(key):
                        int(value)
                    for key, value
                    in counts.head(
                        100
                    ).items()
                },
        }

    return (
        diagnosis,
        report,
    )


###############################################################################
# CHORD first-metastatic date positive control
###############################################################################


def chord_first_meta(
    root: Path,
    offsets: pd.DataFrame,
):

    chord_root = (
        root
        / "data/external_sources/msk_chord_2024/raw"
    )

    candidates = list(
        chord_root.rglob(
            "data_timeline_tumor_sites.txt"
        )
    )

    if len(candidates) != 1:

        raise RuntimeError(
            f"Expected one CHORD tumor-sites file: {candidates}"
        )

    sites = pd.read_csv(
        candidates[
            0
        ],
        sep="\t",
        comment="#",
        low_memory=False,
    )

    pid = find_col(
        sites.columns,
        [
            "PATIENT_ID",
        ],
    )

    day = find_col(
        sites.columns,
        [
            "START_DATE",
        ],
    )

    site = find_col(
        sites.columns,
        [
            "TUMOR_SITE",
        ],
    )

    work = sites.copy()

    work[
        "_patient"
    ] = (
        work[
            pid
        ]
        .astype(str)
        .str.strip()
    )

    work[
        "_day"
    ] = pd.to_numeric(
        work[
            day
        ],
        errors="coerce",
    )

    work[
        "_site"
    ] = (
        work[
            site
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    work = work[
        work[
            "_day"
        ].notna()
        & ~work[
            "_site"
        ].isin(
            {
                "OTHER",
                "LYMPH NODES",
            }
        )
    ]

    first = (
        work.groupby(
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

    bridge = offsets.merge(
        first,
        left_on="chord_patient_id",
        right_on="_patient",
        how="left",
        validate="one_to_one",
    )

    ###########################################################################
    # Convert CHORD metastatic day back to native BPC time coordinate.
    #
    # aligned_day = bpc_day + clock_offset
    # therefore bpc_day = chord_day - clock_offset
    ###########################################################################

    bridge[
        "expected_bpc_first_meta_day"
    ] = (
        bridge[
            "chord_first_meta_day"
        ]
        - bridge[
            "clock_offset"
        ]
    )

    return bridge


###############################################################################
# Candidate metastatic anchors
###############################################################################


def make_anchor_candidates(
    diagnosis: pd.DataFrame,
    meta_bridge: pd.DataFrame,
):

    day_col = find_col(
        diagnosis.columns,
        [
            "START_DATE",
        ],
    )

    work = diagnosis.copy()

    work[
        "_day"
    ] = pd.to_numeric(
        work[
            day_col
        ],
        errors="coerce",
    )

    candidates = {}

    ###########################################################################
    # Stage-IV-at-diagnosis candidates from exact observed STAGE_DX strings.
    ###########################################################################

    stage4_patients = set()

    if "STAGE_DX" in work.columns:

        stage_text = (
            work[
                "STAGE_DX"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        stage4_mask = stage_text.str.contains(
            r"(^|[^A-Z0-9])(IV|4)([^A-Z0-9]|$)",
            regex=True,
        )

        stage4_patients = set(
            work.loc[
                stage4_mask,
                "_patient",
            ]
        )

    ###########################################################################
    # Every exact CA_D_TYPE category with reasonable MSK support becomes a
    # semantic candidate. We do not decide what it "means" from its label.
    ###########################################################################

    if "CA_D_TYPE" in work.columns:

        type_values = (
            work[
                "CA_D_TYPE"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        for value, count in type_values.value_counts().items():

            value = str(
                value
            ).strip()

            if (
                not value
                or int(
                    count
                )
                < 5
            ):
                continue

            mask = (
                type_values
                == value
            )

            first = (
                work.loc[
                    mask
                    & work[
                        "_day"
                    ].notna(),
                    [
                        "_patient",
                        "_day",
                    ],
                ]
                .groupby(
                    "_patient",
                    observed=True,
                )[
                    "_day"
                ]
                .min()
                .to_dict()
            )

            candidates[
                f"CA_D_TYPE::{value}"
            ] = first

    ###########################################################################
    # Prespecified semantic union, only as a diagnostic candidate.
    ###########################################################################

    if "CA_D_TYPE" in work.columns:

        normalized = (
            work[
                "CA_D_TYPE"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        semantic_mask = normalized.str.contains(
            r"METAST|DISTANT|RECURREN|ADVANC",
            regex=True,
        )

        semantic = (
            work.loc[
                semantic_mask
                & work[
                    "_day"
                ].notna(),
                [
                    "_patient",
                    "_day",
                ],
            ]
            .groupby(
                "_patient",
                observed=True,
            )[
                "_day"
            ]
            .min()
            .to_dict()
        )

        candidates[
            "CA_D_TYPE::SEMANTIC_METASTATIC_UNION"
        ] = semantic

    ###########################################################################
    # Compare candidate anchor dates to CHORD first metastatic date on MSK.
    ###########################################################################

    anchor_report = []

    bridge_lookup = (
        meta_bridge.set_index(
            "bpc_patient_id"
        )[
            "expected_bpc_first_meta_day"
        ]
        .to_dict()
    )

    for name, mapping in candidates.items():

        deltas = []

        patients = []

        for patient, candidate_day in mapping.items():

            expected = bridge_lookup.get(
                patient
            )

            if (
                expected is None
                or pd.isna(
                    expected
                )
            ):

                continue

            deltas.append(
                float(
                    candidate_day
                )
                - float(
                    expected
                )
            )

            patients.append(
                patient
            )

        absolute = np.abs(
            np.asarray(
                deltas,
                dtype=float,
            )
        )

        anchor_report.append(
            {
                "candidate":
                    name,

                "candidate_patients":
                    int(
                        len(
                            mapping
                        )
                    ),

                "comparable_patients":
                    int(
                        len(
                            deltas
                        )
                    ),

                "median_signed_delta":
                    (
                        float(
                            np.median(
                                deltas
                            )
                        )
                        if deltas
                        else None
                    ),

                "median_abs_delta":
                    (
                        float(
                            np.median(
                                absolute
                            )
                        )
                        if len(
                            absolute
                        )
                        else None
                    ),

                "within30_fraction":
                    (
                        float(
                            (
                                absolute
                                <= 30
                            ).mean()
                        )
                        if len(
                            absolute
                        )
                        else None
                    ),

                "within90_fraction":
                    (
                        float(
                            (
                                absolute
                                <= 90
                            ).mean()
                        )
                        if len(
                            absolute
                        )
                        else None
                    ),
            }
        )

    anchor_report = pd.DataFrame(
        anchor_report
    )

    if not anchor_report.empty:

        anchor_report = anchor_report.sort_values(
            [
                "median_abs_delta",
                "comparable_patients",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )

    return (
        candidates,
        stage4_patients,
        anchor_report,
    )


###############################################################################
# Treatment filters
###############################################################################


def treatment_value_inventory(
    treatment: pd.DataFrame,
    msk_patients: set[str],
):

    pid = find_col(
        treatment.columns,
        [
            "PATIENT_ID",
        ],
    )

    work = treatment[
        treatment[
            pid
        ]
        .astype(str)
        .str.strip()
        .isin(
            msk_patients
        )
    ].copy()

    result = {}

    for column in (
        "TREATMENT_TYPE",
        "AGENT",
        "REGIMEN_NUMBER",
        "DRUGS_CT_YN",
        "DRUGS_DC_YN",
    ):

        if column not in work.columns:
            continue

        values = (
            work[
                column
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        counts = (
            values[
                values.ne("")
            ]
            .value_counts()
        )

        result[
            column
        ] = {
            "unique":
                int(
                    len(
                        counts
                    )
                ),

            "top_counts":
                {
                    str(key):
                        int(value)
                    for key, value
                    in counts.head(
                        60
                    ).items()
                },
        }

    return result


def basic_ckpt1_filter(
    treatment: pd.DataFrame,
):

    work = treatment.copy()

    if "AGENT" in work.columns:

        work = work[
            work[
                "AGENT"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .ne(
                "INVESTIGATIVE"
            )
        ].copy()

    ###########################################################################
    # BPC has no CHORD SUBTYPE field. Only exclude a Bone Treatment category
    # if the public BPC treatment-type field literally contains it.
    ###########################################################################

    if "TREATMENT_TYPE" in work.columns:

        type_text = (
            work[
                "TREATMENT_TYPE"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        work = work[
            type_text.ne(
                "BONE TREATMENT"
            )
        ].copy()

    return work


###############################################################################
# Line-strategy evaluation
###############################################################################


def evaluate_lookup(
    name: str,
    matches: pd.DataFrame,
    lookup,
):

    rows = []

    for _, row in matches.iterrows():

        line, line_start, _ = (
            lookup(
                row[
                    "bpc_patient_id"
                ],
                float(
                    row[
                        "bpc_day"
                    ]
                ),
            )
        )

        if line <= 0:
            continue

        rows.append(
            {
                "bpc_patient_id":
                    row[
                        "bpc_patient_id"
                    ],

                "bpc_episode_id":
                    row[
                        "bpc_episode_id"
                    ],

                "chord_line":
                    int(
                        row[
                            "chord_line"
                        ]
                    ),

                "candidate_line":
                    int(
                        line
                    ),

                "candidate_line_start":
                    float(
                        line_start
                    ),

                "equal":
                    (
                        int(
                            line
                        )
                        == int(
                            row[
                                "chord_line"
                            ]
                        )
                    ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    if frame.empty:

        return {
            "strategy":
                name,

            "rows":
                0,

            "patients":
                0,

            "agreement":
                None,
        }, frame

    return {
        "strategy":
            name,

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

        "agreement":
            float(
                frame[
                    "equal"
                ].mean()
            ),

        "line_delta":
            stats(
                frame[
                    "candidate_line"
                ]
                - frame[
                    "chord_line"
                ]
            ),
    }, frame


def regimen_lookup_builder(
    treatment: pd.DataFrame,
    msk_patients: set[str],
    *,
    mode: str,
):

    pid = find_col(
        treatment.columns,
        [
            "PATIENT_ID",
        ],
    )

    start = find_col(
        treatment.columns,
        [
            "START_DATE",
        ],
    )

    stop = find_col(
        treatment.columns,
        [
            "STOP_DATE",
        ],
        required=False,
    )

    regimen = find_col(
        treatment.columns,
        [
            "REGIMEN_NUMBER",
        ],
    )

    work = treatment.copy()

    work[
        "_patient"
    ] = (
        work[
            pid
        ]
        .astype(str)
        .str.strip()
    )

    work = work[
        work[
            "_patient"
        ].isin(
            msk_patients
        )
    ].copy()

    work[
        "_start"
    ] = pd.to_numeric(
        work[
            start
        ],
        errors="coerce",
    )

    work[
        "_regimen"
    ] = pd.to_numeric(
        work[
            regimen
        ],
        errors="coerce",
    )

    work[
        "_stop"
    ] = (
        pd.to_numeric(
            work[
                stop
            ],
            errors="coerce",
        )
        if stop
        else np.nan
    )

    work = work[
        work[
            "_start"
        ].notna()
        & work[
            "_regimen"
        ].notna()
        & (
            work[
                "_regimen"
            ]
            > 0
        )
    ].copy()

    groups = {
        patient:
            group.sort_values(
                [
                    "_start",
                    "_regimen",
                ],
                kind="mergesort",
            )
        for patient, group
        in work.groupby(
            "_patient",
            observed=True,
        )
    }

    def lookup(
        patient,
        day,
    ):

        group = groups.get(
            patient
        )

        if (
            group is None
            or group.empty
        ):

            return (
                0,
                np.nan,
                np.nan,
            )

        eligible = group[
            group[
                "_start"
            ]
            <= day
        ].copy()

        if mode == "active":

            eligible = eligible[
                eligible[
                    "_stop"
                ].isna()
                | (
                    eligible[
                        "_stop"
                    ]
                    >= day
                )
            ]

        if eligible.empty:

            return (
                0,
                np.nan,
                np.nan,
            )

        chosen = eligible.sort_values(
            [
                "_start",
                "_regimen",
            ],
            kind="mergesort",
        ).iloc[
            -1
        ]

        return (
            int(
                chosen[
                    "_regimen"
                ]
            ),

            float(
                chosen[
                    "_start"
                ]
            ),

            (
                float(
                    chosen[
                        "_stop"
                    ]
                )
                if pd.notna(
                    chosen[
                        "_stop"
                    ]
                )
                else np.inf
            ),
        )

    return lookup


###############################################################################
# Main
###############################################################################


def main():

    root = Path(".").resolve()

    out = (
        root
        / "artifacts/checkpoint7a4b_fix5_line_semantics"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    a4b = import_module(
        "ckpt7a4b_fix5_base",
        root
        / "scripts/dynamic_scan/"
        "ckpt7a4b_external_prediction_freeze.py",
    )

    bpc_root = (
        root
        / "data/external_sources/"
        "bpc_brca_1_0_public/"
        "cBioPortal_files"
    )

    treatment = pd.read_csv(
        bpc_root
        / "data_timeline_treatment.txt",
        sep="\t",
        comment="#",
        low_memory=False,
    )

    imaging = pd.read_csv(
        bpc_root
        / "data_timeline_imaging.txt",
        sep="\t",
        comment="#",
        low_memory=False,
    )

    diagnosis = pd.read_csv(
        bpc_root
        / "data_timeline_cancer_diagnosis.txt",
        sep="\t",
        comment="#",
        low_memory=False,
    )

    (
        matches,
        offsets,
    ) = prepare_scan_matches(
        root,
        a4b,
    )

    msk_patients = set(
        offsets[
            "bpc_patient_id"
        ]
    )

    (
        diagnosis_msk,
        diagnosis_report,
    ) = diagnosis_inventory(
        diagnosis,
        msk_patients,
    )

    treatment_report = (
        treatment_value_inventory(
            treatment,
            msk_patients,
        )
    )

    meta_bridge = chord_first_meta(
        root,
        offsets,
    )

    (
        anchor_candidates,
        stage4_patients,
        anchor_report,
    ) = make_anchor_candidates(
        diagnosis_msk,
        meta_bridge,
    )

    treatment_filtered = (
        basic_ckpt1_filter(
            treatment
        )
    )

    strategies = []

    strategy_frames = {}

    ###########################################################################
    # Baseline current FIX4 strategy: all treatments.
    ###########################################################################

    (
        lines_all,
        lookup_all,
        audit_all,
    ) = a4b.build_line_intervals(
        treatment_filtered,
        imaging,
        msk_patients,
    )

    result, frame = evaluate_lookup(
        "CKPT1_REPLAY_ALL_TREATMENTS",
        matches,
        lambda patient, day:
            a4b.assign_line(
                patient,
                day,
                lookup_all,
            ),
    )

    result[
        "line_builder_audit"
    ] = audit_all

    strategies.append(
        result
    )

    strategy_frames[
        result[
            "strategy"
        ]
    ] = frame

    ###########################################################################
    # Raw BPC regimen-number strategies AFTER correct clock alignment.
    ###########################################################################

    for mode in (
        "active",
        "latest",
    ):

        lookup = regimen_lookup_builder(
            treatment_filtered,
            msk_patients,
            mode=mode,
        )

        result, frame = evaluate_lookup(
            "REGIMEN_NUMBER_"
            + mode.upper(),
            matches,
            lookup,
        )

        strategies.append(
            result
        )

        strategy_frames[
            result[
                "strategy"
            ]
        ] = frame

    ###########################################################################
    # Post-metastatic variants.
    #
    # Stage-IV-at-diagnosis patients follow the CKPT1 rule and retain their
    # treatment stream. For everyone else, rows before the candidate BPC
    # metastatic anchor are excluded from BOTH treatment and progression.
    ###########################################################################

    pid_tx = find_col(
        treatment_filtered.columns,
        [
            "PATIENT_ID",
        ],
    )

    tx_day = find_col(
        treatment_filtered.columns,
        [
            "START_DATE",
        ],
    )

    pid_img = find_col(
        imaging.columns,
        [
            "PATIENT_ID",
        ],
    )

    img_day = find_col(
        imaging.columns,
        [
            "START_DATE",
        ],
    )

    for anchor_name, anchor_map in anchor_candidates.items():

        tx = treatment_filtered.copy()

        tx[
            "_patient"
        ] = (
            tx[
                pid_tx
            ]
            .astype(str)
            .str.strip()
        )

        tx[
            "_day"
        ] = pd.to_numeric(
            tx[
                tx_day
            ],
            errors="coerce",
        )

        tx[
            "_anchor"
        ] = tx[
            "_patient"
        ].map(
            anchor_map
        )

        tx_keep = (
            tx[
                "_patient"
            ].isin(
                stage4_patients
            )
            | (
                tx[
                    "_anchor"
                ].notna()
                & (
                    tx[
                        "_day"
                    ]
                    >= tx[
                        "_anchor"
                    ]
                )
            )
        )

        tx = tx.loc[
            tx_keep
        ].drop(
            columns=[
                "_patient",
                "_day",
                "_anchor",
            ],
            errors="ignore",
        )

        rad = imaging.copy()

        rad[
            "_patient"
        ] = (
            rad[
                pid_img
            ]
            .astype(str)
            .str.strip()
        )

        rad[
            "_day"
        ] = pd.to_numeric(
            rad[
                img_day
            ],
            errors="coerce",
        )

        rad[
            "_anchor"
        ] = rad[
            "_patient"
        ].map(
            anchor_map
        )

        rad_keep = (
            rad[
                "_patient"
            ].isin(
                stage4_patients
            )
            | (
                rad[
                    "_anchor"
                ].notna()
                & (
                    rad[
                        "_day"
                    ]
                    >= rad[
                        "_anchor"
                    ]
                )
            )
        )

        rad = rad.loc[
            rad_keep
        ].drop(
            columns=[
                "_patient",
                "_day",
                "_anchor",
            ],
            errors="ignore",
        )

        try:

            (
                lines,
                lookup,
                line_audit,
            ) = a4b.build_line_intervals(
                tx,
                rad,
                msk_patients,
            )

        except Exception as exc:

            strategies.append(
                {
                    "strategy":
                        "POSTMETA::"
                        + anchor_name,

                    "rows":
                        0,

                    "patients":
                        0,

                    "agreement":
                        None,

                    "error":
                        (
                            type(
                                exc
                            ).__name__
                            + ": "
                            + str(
                                exc
                            )
                        ),
                }
            )

            continue

        result, frame = evaluate_lookup(
            "POSTMETA::"
            + anchor_name,
            matches,
            lambda patient, day, lookup=lookup:
                a4b.assign_line(
                    patient,
                    day,
                    lookup,
                ),
        )

        result[
            "line_builder_audit"
        ] = line_audit

        result[
            "anchor_coverage_patients"
        ] = int(
            len(
                anchor_map
            )
        )

        strategies.append(
            result
        )

        strategy_frames[
            result[
                "strategy"
            ]
        ] = frame

    strategy_table = pd.DataFrame(
        strategies
    )

    strategy_table[
        "_agreement_sort"
    ] = pd.to_numeric(
        strategy_table[
            "agreement"
        ],
        errors="coerce",
    )

    strategy_table = strategy_table.sort_values(
        [
            "_agreement_sort",
            "rows",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    ).drop(
        columns=[
            "_agreement_sort",
        ]
    ).reset_index(
        drop=True
    )

    ###########################################################################
    # Compare first candidate line start with frozen CHORD line-1 start.
    ###########################################################################

    chord_lines = pd.read_parquet(
        root
        / "artifacts/checkpoint1/canonical/"
        "chord_mbc_lines.parquet"
    )

    cpid = find_col(
        chord_lines.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    cline = find_col(
        chord_lines.columns,
        [
            "line",
            "treatment_line",
        ],
    )

    cstart = find_col(
        chord_lines.columns,
        [
            "line_start_day",
            "LINE_START",
        ],
    )

    chord_l1 = chord_lines[
        pd.to_numeric(
            chord_lines[
                cline
            ],
            errors="coerce",
        )
        == 1
    ].copy()

    chord_l1 = (
        chord_l1.groupby(
            cpid,
            observed=True,
        )[
            cstart
        ]
        .min()
        .rename(
            "chord_line1_start"
        )
        .reset_index()
    )

    line1_bridge = offsets.merge(
        chord_l1,
        left_on="chord_patient_id",
        right_on=cpid,
        how="left",
        validate="one_to_one",
    )

    line1_bridge[
        "expected_bpc_line1_start"
    ] = (
        line1_bridge[
            "chord_line1_start"
        ]
        - line1_bridge[
            "clock_offset"
        ]
    )

    current_line1 = (
        lines_all[
            lines_all[
                "treatment_line"
            ]
            == 1
        ][
            [
                "patient_id",
                "line_start_day",
            ]
        ]
        .rename(
            columns={
                "patient_id":
                    "bpc_patient_id",

                "line_start_day":
                    "all_tx_line1_start",
            }
        )
    )

    line1_bridge = line1_bridge.merge(
        current_line1,
        on="bpc_patient_id",
        how="left",
        validate="one_to_one",
    )

    line1_bridge[
        "all_tx_line1_minus_expected"
    ] = (
        line1_bridge[
            "all_tx_line1_start"
        ]
        - line1_bridge[
            "expected_bpc_line1_start"
        ]
    )

    ###########################################################################
    # Persist.
    ###########################################################################

    matches.to_parquet(
        out
        / "aligned_msk_scan_matches.parquet",
        index=False,
    )

    anchor_report.to_parquet(
        out
        / "metastatic_anchor_candidates.parquet",
        index=False,
    )

    strategy_table.to_parquet(
        out
        / "line_strategy_comparison.parquet",
        index=False,
    )

    line1_bridge.to_parquet(
        out
        / "line1_start_alignment.parquet",
        index=False,
    )

    report = {
        "status":
            "MSK_LINE_SEMANTIC_AUDIT_COMPLETE",

        "external_outcomes_opened":
            False,

        "aligned_scan_matches":
            int(
                len(
                    matches
                )
            ),

        "aligned_scan_patients":
            int(
                matches[
                    "bpc_patient_id"
                ].nunique()
            ),

        "diagnosis_value_inventory":
            diagnosis_report,

        "treatment_value_inventory":
            treatment_report,

        "stage4_patient_count":
            int(
                len(
                    stage4_patients
                )
            ),

        "metastatic_anchor_candidates":
            anchor_report.to_dict(
                orient="records"
            ),

        "line_strategy_comparison":
            strategy_table.to_dict(
                orient="records"
            ),

        "current_all_treatment_line1_offset":
            stats(
                line1_bridge[
                    "all_tx_line1_minus_expected"
                ]
            ),

        "next_action":
            (
                "Choose the BPC metastatic-entry / line strategy only if "
                "MSK positive control shows strong semantic agreement and "
                "adequate support. Do not open DFCI/VICC outcomes."
            ),
    }

    atomic_json(
        out
        / "audit.json",
        report,
    )

    ###########################################################################
    # Compact packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A4B FIX5 SUMMARY =========="
    )

    print(
        "status=MSK_LINE_SEMANTIC_AUDIT_COMPLETE"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "aligned_scan_matches="
        f"{len(matches)}"
    )

    print(
        "aligned_scan_patients="
        f"{matches['bpc_patient_id'].nunique()}"
    )

    print(
        "stage4_patient_count="
        f"{len(stage4_patients)}"
    )

    print(
        "current_all_treatment_line1_offset="
        f"{report['current_all_treatment_line1_offset']}"
    )

    print(
        "========== CKPT7A4B FIX5 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A4B FIX5 DECISION PACKET =========="
    )

    print("")
    print(
        "----- DIAGNOSIS VALUE INVENTORY -----"
    )

    print(
        json.dumps(
            diagnosis_report,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "----- DIAGNOSIS VALUE INVENTORY END -----"
    )

    print("")
    print(
        "----- TREATMENT VALUE INVENTORY -----"
    )

    print(
        json.dumps(
            treatment_report,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "----- TREATMENT VALUE INVENTORY END -----"
    )

    print("")
    print(
        "----- METASTATIC ANCHOR CANDIDATES -----"
    )

    if anchor_report.empty:

        print(
            "NONE"
        )

    else:

        print(
            anchor_report.to_string(
                index=False
            )
        )

    print(
        "----- METASTATIC ANCHOR CANDIDATES END -----"
    )

    print("")
    print(
        "----- LINE STRATEGY COMPARISON -----"
    )

    display_columns = [
        column
        for column in (
            "strategy",
            "rows",
            "patients",
            "agreement",
            "anchor_coverage_patients",
            "error",
        )
        if column
        in strategy_table.columns
    ]

    print(
        strategy_table[
            display_columns
        ]
        .head(
            40
        )
        .to_string(
            index=False
        )
    )

    print(
        "----- LINE STRATEGY COMPARISON END -----"
    )

    print("")
    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A4B FIX5 DECISION PACKET END =========="
    )


if __name__ == "__main__":

    main()
