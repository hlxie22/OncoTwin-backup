#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
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


def normalize_agent(value: Any) -> str:

    value = clean(value).upper()

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


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
            f"Missing {candidates}; columns={list(columns)}"
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

    if len(check) != len(frame):

        raise RuntimeError(
            f"Parquet verification failed: {path}"
        )

    tmp.replace(
        path
    )


def numeric_stats(values):

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


def find_chord_file(
    root: Path,
    basename: str,
) -> Path:

    candidates = list(
        (
            root
            / "data/external_sources/msk_chord_2024/raw"
        ).rglob(
            basename
        )
    )

    if len(candidates) != 1:

        raise RuntimeError(
            f"Expected exactly one {basename}; found={candidates}"
        )

    return candidates[
        0
    ]


###############################################################################
# Frozen MSK identity/clock bridge
###############################################################################


def load_offsets(
    root: Path,
):

    offsets = pd.read_parquet(
        root
        / "artifacts/checkpoint7a3g/"
        "msk_patient_clock_offsets.parquet"
    ).copy()

    required = {
        "bpc_patient_id",
        "chord_patient_id",
        "clock_offset",
    }

    if not required.issubset(
        offsets.columns
    ):

        raise RuntimeError(
            f"Unexpected offset schema: {list(offsets.columns)}"
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

    return offsets


###############################################################################
# Stage-IV predictor-side subset
###############################################################################


def load_stage4_patients(
    root: Path,
):

    path = (
        root
        / "data/external_sources/bpc_brca_1_0_public/"
        "cBioPortal_files/data_timeline_cancer_diagnosis.txt"
    )

    diagnosis = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        low_memory=False,
    )

    pid = find_col(
        diagnosis.columns,
        [
            "PATIENT_ID",
        ],
    )

    stage = find_col(
        diagnosis.columns,
        [
            "STAGE_DX",
        ],
    )

    stage_text = (
        diagnosis[
            stage
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mask = (
        stage_text
        == "STAGE IV"
    )

    return set(
        diagnosis.loc[
            mask,
            pid,
        ]
        .astype(str)
        .str.strip()
    )


###############################################################################
# Aligned scan surface
###############################################################################


def build_aligned_scan_matches(
    root: Path,
    offsets: pd.DataFrame,
):

    bpc_scans = pd.read_parquet(
        root
        / "artifacts/checkpoint7a3/"
        "bpc_w3_predictor_episodes.parquet"
    )

    pid = find_col(
        bpc_scans.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    site = find_col(
        bpc_scans.columns,
        [
            "site",
            "institution",
        ],
    )

    day = find_col(
        bpc_scans.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
    )

    episode = find_col(
        bpc_scans.columns,
        [
            "bpc_episode_id",
            "scan_episode_id",
        ],
    )

    bpc = bpc_scans[
        bpc_scans[
            site
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
            pid
        ]
        .astype(str)
        .str.strip()
    )

    bpc[
        "bpc_day"
    ] = pd.to_numeric(
        bpc[
            day
        ],
        errors="coerce",
    )

    bpc[
        "bpc_episode_id"
    ] = (
        bpc[
            episode
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
        on="bpc_patient_id",
        how="inner",
        validate="many_to_one",
    )

    bpc = bpc[
        bpc[
            "bpc_day"
        ].notna()
    ].copy()

    bpc[
        "aligned_day"
    ] = (
        bpc[
            "bpc_day"
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

    chord_work = pd.DataFrame(
        {
            "chord_patient_id":
                chord[
                    cpid
                ]
                .astype(str)
                .str.strip(),

            "chord_day":
                pd.to_numeric(
                    chord[
                        cday
                    ],
                    errors="coerce",
                ),

            "chord_line":
                pd.to_numeric(
                    chord[
                        cline
                    ],
                    errors="coerce",
                ),
        }
    ).dropna()

    groups = {
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

    rows = []

    for _, row in bpc.iterrows():

        group = groups.get(
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

                "bpc_day":
                    float(
                        row[
                            "bpc_day"
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
            "No aligned MSK scan matches."
        )

    return matches


###############################################################################
# Predictor-side BPC lines
###############################################################################


def load_bpc_predictors(
    root: Path,
):

    base = (
        root
        / "data/external_sources/bpc_brca_1_0_public/"
        "cBioPortal_files"
    )

    treatment = pd.read_csv(
        base
        / "data_timeline_treatment.txt",
        sep="\t",
        comment="#",
        low_memory=False,
    )

    imaging = pd.read_csv(
        base
        / "data_timeline_imaging.txt",
        sep="\t",
        comment="#",
        low_memory=False,
    )

    return (
        treatment,
        imaging,
    )


def treatment_variant(
    treatment: pd.DataFrame,
    name: str,
):

    work = treatment.copy()

    if name == "RAW":

        return work

    if name == "DROP_INVESTIGATIONAL_DRUG":

        agent = find_col(
            work.columns,
            [
                "AGENT",
            ],
        )

        normalized = (
            work[
                agent
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        return work[
            ~normalized.isin(
                {
                    "INVESTIGATIVE",
                    "INVESTIGATIONAL DRUG",
                }
            )
        ].copy()

    raise ValueError(
        name
    )


def attach_candidate_lines(
    matches: pd.DataFrame,
    lookup,
    a4b,
):

    result = matches.copy()

    candidate_line = []

    candidate_start = []

    for _, row in result.iterrows():

        line, start, _ = (
            a4b.assign_line(
                str(
                    row[
                        "bpc_patient_id"
                    ]
                ),
                float(
                    row[
                        "bpc_day"
                    ]
                ),
                lookup,
            )
        )

        candidate_line.append(
            int(
                line
            )
        )

        candidate_start.append(
            float(
                start
            )
            if np.isfinite(
                start
            )
            else np.nan
        )

    result[
        "bpc_line"
    ] = candidate_line

    result[
        "bpc_line_start"
    ] = candidate_start

    result = result[
        result[
            "bpc_line"
        ]
        > 0
    ].copy()

    result[
        "line_equal"
    ] = (
        result[
            "bpc_line"
        ]
        == result[
            "chord_line"
        ]
    )

    result[
        "raw_line_delta"
    ] = (
        result[
            "bpc_line"
        ]
        - result[
            "chord_line"
        ]
    )

    return result


###############################################################################
# Absolute vs patient-offset vs transition geometry
###############################################################################


def line_geometry(
    frame: pd.DataFrame,
):

    if frame.empty:

        return {
            "rows": 0,
            "patients": 0,
        }, frame

    work = frame.copy()

    patient_offsets = {}

    for patient, group in work.groupby(
        "bpc_patient_id",
        observed=True,
    ):

        deltas = (
            group[
                "bpc_line"
            ]
            - group[
                "chord_line"
            ]
        ).astype(
            int
        )

        counts = Counter(
            deltas.tolist()
        )

        mode_offset = sorted(
            counts.items(),
            key=lambda item:
                (
                    -item[
                        1
                    ],
                    abs(
                        item[
                            0
                        ]
                    ),
                    item[
                        0
                    ],
                ),
        )[
            0
        ][
            0
        ]

        patient_offsets[
            patient
        ] = int(
            mode_offset
        )

    work[
        "patient_mode_line_offset"
    ] = work[
        "bpc_patient_id"
    ].map(
        patient_offsets
    )

    work[
        "offset_adjusted_bpc_line"
    ] = (
        work[
            "bpc_line"
        ]
        - work[
            "patient_mode_line_offset"
        ]
    )

    work[
        "offset_adjusted_equal"
    ] = (
        work[
            "offset_adjusted_bpc_line"
        ]
        == work[
            "chord_line"
        ]
    )

    ###########################################################################
    # Boundary/transition agreement on sequential matched scans.
    ###########################################################################

    transition_rows = []

    for patient, group in work.groupby(
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

        if len(group) < 2:
            continue

        for index in range(
            1,
            len(group),
        ):

            previous = group.iloc[
                index
                - 1
            ]

            current = group.iloc[
                index
            ]

            bpc_changed = (
                int(
                    current[
                        "bpc_line"
                    ]
                )
                != int(
                    previous[
                        "bpc_line"
                    ]
                )
            )

            chord_changed = (
                int(
                    current[
                        "chord_line"
                    ]
                )
                != int(
                    previous[
                        "chord_line"
                    ]
                )
            )

            transition_rows.append(
                {
                    "bpc_patient_id":
                        patient,

                    "previous_day":
                        float(
                            previous[
                                "aligned_day"
                            ]
                        ),

                    "current_day":
                        float(
                            current[
                                "aligned_day"
                            ]
                        ),

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

    if transitions.empty:

        transition_metrics = {
            "pairs":
                0,
        }

    else:

        tp = int(
            (
                transitions[
                    "bpc_changed"
                ]
                & transitions[
                    "chord_changed"
                ]
            ).sum()
        )

        fp = int(
            (
                transitions[
                    "bpc_changed"
                ]
                & ~transitions[
                    "chord_changed"
                ]
            ).sum()
        )

        fn = int(
            (
                ~transitions[
                    "bpc_changed"
                ]
                & transitions[
                    "chord_changed"
                ]
            ).sum()
        )

        transition_metrics = {
            "pairs":
                int(
                    len(
                        transitions
                    )
                ),

            "patients":
                int(
                    transitions[
                        "bpc_patient_id"
                    ].nunique()
                ),

            "agreement":
                float(
                    transitions[
                        "equal"
                    ].mean()
                ),

            "bpc_transition_rows":
                int(
                    transitions[
                        "bpc_changed"
                    ].sum()
                ),

            "chord_transition_rows":
                int(
                    transitions[
                        "chord_changed"
                    ].sum()
                ),

            "transition_precision":
                (
                    float(
                        tp
                        / (
                            tp
                            + fp
                        )
                    )
                    if (
                        tp
                        + fp
                    )
                    else None
                ),

            "transition_recall":
                (
                    float(
                        tp
                        / (
                            tp
                            + fn
                        )
                    )
                    if (
                        tp
                        + fn
                    )
                    else None
                ),
        }

    summary = {
        "rows":
            int(
                len(
                    work
                )
            ),

        "patients":
            int(
                work[
                    "bpc_patient_id"
                ].nunique()
            ),

        "absolute_line_agreement":
            float(
                work[
                    "line_equal"
                ].mean()
            ),

        "patient_offset_adjusted_agreement":
            float(
                work[
                    "offset_adjusted_equal"
                ].mean()
            ),

        "raw_line_delta":
            numeric_stats(
                work[
                    "raw_line_delta"
                ]
            ),

        "patient_mode_offset":
            numeric_stats(
                list(
                    patient_offsets.values()
                )
            ),

        "transition_geometry":
            transition_metrics,
    }

    return (
        summary,
        work,
        transitions,
    )


###############################################################################
# Treatment stream alignment
###############################################################################


def prepare_treatment_stream(
    frame: pd.DataFrame,
    *,
    patient_col: str,
    day_col: str,
    agent_col: str | None,
    patient_name: str,
    day_name: str,
):

    out = pd.DataFrame(
        {
            patient_name:
                frame[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            day_name:
                pd.to_numeric(
                    frame[
                        day_col
                    ],
                    errors="coerce",
                ),
        }
    )

    if agent_col is not None:

        out[
            "agent"
        ] = frame[
            agent_col
        ].map(
            normalize_agent
        )

    else:

        out[
            "agent"
        ] = ""

    return out.dropna(
        subset=[
            day_name,
        ]
    )


def treatment_stream_alignment(
    root: Path,
    offsets: pd.DataFrame,
    bpc_treatment: pd.DataFrame,
    stage4_patients: set[str],
):

    bpid = find_col(
        bpc_treatment.columns,
        [
            "PATIENT_ID",
        ],
    )

    bday = find_col(
        bpc_treatment.columns,
        [
            "START_DATE",
        ],
    )

    bagent = find_col(
        bpc_treatment.columns,
        [
            "AGENT",
        ],
        required=False,
    )

    bpc = prepare_treatment_stream(
        bpc_treatment,
        patient_col=bpid,
        day_col=bday,
        agent_col=bagent,
        patient_name="bpc_patient_id",
        day_name="bpc_day",
    )

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

    chord_path = find_chord_file(
        root,
        "data_timeline_treatment.txt",
    )

    chord_raw = pd.read_csv(
        chord_path,
        sep="\t",
        comment="#",
        low_memory=False,
    )

    cpid = find_col(
        chord_raw.columns,
        [
            "PATIENT_ID",
        ],
    )

    cday = find_col(
        chord_raw.columns,
        [
            "START_DATE",
        ],
    )

    cagent = find_col(
        chord_raw.columns,
        [
            "AGENT",
        ],
        required=False,
    )

    ###########################################################################
    # Exact CKPT1 treatment filtering that is possible without the metastatic
    # entry step.
    ###########################################################################

    chord_filtered = chord_raw.copy()

    if "SUBTYPE" in chord_filtered.columns:

        chord_filtered = chord_filtered[
            chord_filtered[
                "SUBTYPE"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne(
                "Bone Treatment"
            )
        ].copy()

    if cagent is not None:

        chord_filtered = chord_filtered[
            chord_filtered[
                cagent
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne(
                "INVESTIGATIVE"
            )
        ].copy()

    chord = prepare_treatment_stream(
        chord_filtered,
        patient_col=cpid,
        day_col=cday,
        agent_col=cagent,
        patient_name="chord_patient_id",
        day_name="chord_day",
    )

    mapped_chord = set(
        offsets[
            "chord_patient_id"
        ]
    )

    chord = chord[
        chord[
            "chord_patient_id"
        ].isin(
            mapped_chord
        )
    ].copy()

    chord_groups = {
        patient:
            group.sort_values(
                "chord_day",
                kind="mergesort",
            )
        for patient, group
        in chord.groupby(
            "chord_patient_id",
            observed=True,
        )
    }

    rows = []

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

        distances = np.abs(
            days
            - float(
                row[
                    "aligned_day"
                ]
            )
        )

        index = int(
            np.argmin(
                distances
            )
        )

        chosen = group.iloc[
            index
        ]

        rows.append(
            {
                "bpc_patient_id":
                    row[
                        "bpc_patient_id"
                    ],

                "stage4":
                    bool(
                        row[
                            "bpc_patient_id"
                        ]
                        in stage4_patients
                    ),

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
                    float(
                        distances[
                            index
                        ]
                    ),

                "bpc_agent":
                    row[
                        "agent"
                    ],

                "chord_agent":
                    chosen[
                        "agent"
                    ],

                "agent_equal":
                    (
                        row[
                            "agent"
                        ]
                        != ""
                        and row[
                            "agent"
                        ]
                        == chosen[
                            "agent"
                        ]
                    ),
            }
        )

    alignment = pd.DataFrame(
        rows
    )

    def summarize(
        frame,
    ):

        if frame.empty:

            return {
                "rows": 0,
            }

        delta = frame[
            "abs_day_delta"
        ].to_numpy(
            dtype=float
        )

        exact = (
            frame[
                "abs_day_delta"
            ]
            == 0
        )

        within3 = (
            frame[
                "abs_day_delta"
            ]
            <= 3
        )

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

            "nearest_day":
                numeric_stats(
                    delta
                ),

            "within0_fraction":
                float(
                    exact.mean()
                ),

            "within3_fraction":
                float(
                    within3.mean()
                ),

            "within7_fraction":
                float(
                    (
                        frame[
                            "abs_day_delta"
                        ]
                        <= 7
                    ).mean()
                ),

            "within30_fraction":
                float(
                    (
                        frame[
                            "abs_day_delta"
                        ]
                        <= 30
                    ).mean()
                ),

            "exact_day_agent_agreement":
                (
                    float(
                        frame.loc[
                            exact,
                            "agent_equal",
                        ].mean()
                    )
                    if exact.any()
                    else None
                ),

            "within3_agent_agreement":
                (
                    float(
                        frame.loc[
                            within3,
                            "agent_equal",
                        ].mean()
                    )
                    if within3.any()
                    else None
                ),
        }

    return (
        alignment,
        {
            "all":
                summarize(
                    alignment
                ),

            "stage4":
                summarize(
                    alignment[
                        alignment[
                            "stage4"
                        ]
                    ]
                ),
        },
    )


###############################################################################
# Progression stream alignment
###############################################################################


def progression_stream_alignment(
    root: Path,
    offsets: pd.DataFrame,
    bpc_imaging: pd.DataFrame,
    stage4_patients: set[str],
):

    pid = find_col(
        bpc_imaging.columns,
        [
            "PATIENT_ID",
        ],
    )

    day = find_col(
        bpc_imaging.columns,
        [
            "START_DATE",
        ],
    )

    status_col = find_col(
        bpc_imaging.columns,
        [
            "CURATED_CANCER_STATUS",
        ],
    )

    bpc = bpc_imaging.copy()

    bpc[
        "bpc_patient_id"
    ] = (
        bpc[
            pid
        ]
        .astype(str)
        .str.strip()
    )

    bpc[
        "bpc_day"
    ] = pd.to_numeric(
        bpc[
            day
        ],
        errors="coerce",
    )

    status = (
        bpc[
            status_col
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    bpc[
        "bpc_progression"
    ] = ""

    bpc.loc[
        status
        == "PROGRESSING/WORSENING/ENLARGING",
        "bpc_progression",
    ] = "Y"

    bpc.loc[
        status.isin(
            {
                "STABLE/NO CHANGE",
                "IMPROVING/RESPONDING",
            }
        ),
        "bpc_progression",
    ] = "N"

    bpc = bpc[
        bpc[
            "bpc_day"
        ].notna()
        & bpc[
            "bpc_progression"
        ].isin(
            {
                "Y",
                "N",
            }
        )
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

    chord_path = find_chord_file(
        root,
        "data_timeline_progression.txt",
    )

    chord = pd.read_csv(
        chord_path,
        sep="\t",
        comment="#",
        low_memory=False,
    )

    cpid = find_col(
        chord.columns,
        [
            "PATIENT_ID",
        ],
    )

    cday = find_col(
        chord.columns,
        [
            "START_DATE",
        ],
    )

    cprog = find_col(
        chord.columns,
        [
            "PROGRESSION",
        ],
    )

    chord[
        "chord_patient_id"
    ] = (
        chord[
            cpid
        ]
        .astype(str)
        .str.strip()
    )

    chord[
        "chord_day"
    ] = pd.to_numeric(
        chord[
            cday
        ],
        errors="coerce",
    )

    chord[
        "chord_progression"
    ] = (
        chord[
            cprog
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    chord = chord[
        chord[
            "chord_day"
        ].notna()
        & chord[
            "chord_progression"
        ].isin(
            {
                "Y",
                "N",
            }
        )
    ].copy()

    groups = {
        patient:
            group.sort_values(
                "chord_day",
                kind="mergesort",
            )
        for patient, group
        in chord.groupby(
            "chord_patient_id",
            observed=True,
        )
    }

    rows = []

    for _, row in bpc.iterrows():

        group = groups.get(
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

        distances = np.abs(
            days
            - float(
                row[
                    "aligned_day"
                ]
            )
        )

        index = int(
            np.argmin(
                distances
            )
        )

        distance = float(
            distances[
                index
            ]
        )

        if distance > 3.0:
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

                "stage4":
                    bool(
                        row[
                            "bpc_patient_id"
                        ]
                        in stage4_patients
                    ),

                "abs_day_delta":
                    distance,

                "bpc_progression":
                    row[
                        "bpc_progression"
                    ],

                "chord_progression":
                    chosen[
                        "chord_progression"
                    ],

                "equal":
                    (
                        row[
                            "bpc_progression"
                        ]
                        == chosen[
                            "chord_progression"
                        ]
                    ),
            }
        )

    aligned = pd.DataFrame(
        rows
    )

    def summarize(
        frame,
    ):

        if frame.empty:

            return {
                "rows": 0,
            }

        confusion = (
            frame.groupby(
                [
                    "bpc_progression",
                    "chord_progression",
                ],
                observed=True,
            )
            .size()
            .rename(
                "n"
            )
            .reset_index()
            .to_dict(
                orient="records"
            )
        )

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

            "agreement":
                float(
                    frame[
                        "equal"
                    ].mean()
                ),

            "median_abs_day_delta":
                float(
                    frame[
                        "abs_day_delta"
                    ].median()
                ),

            "confusion":
                confusion,
        }

    return (
        aligned,
        {
            "all":
                summarize(
                    aligned
                ),

            "stage4":
                summarize(
                    aligned[
                        aligned[
                            "stage4"
                        ]
                    ]
                ),
        },
    )


###############################################################################
# Main
###############################################################################


def main():

    root = Path(
        "."
    ).resolve()

    out = (
        root
        / "artifacts/checkpoint7a4b_fix6_line_boundary_stream"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    a4b = import_module(
        "ckpt7a4b_fix6_base",
        root
        / "scripts/dynamic_scan/"
        "ckpt7a4b_external_prediction_freeze.py",
    )

    offsets = load_offsets(
        root
    )

    stage4_patients = (
        load_stage4_patients(
            root
        )
    )

    matches = (
        build_aligned_scan_matches(
            root,
            offsets,
        )
    )

    treatment, imaging = (
        load_bpc_predictors(
            root
        )
    )

    mapped_patients = set(
        offsets[
            "bpc_patient_id"
        ]
    )

    ###########################################################################
    # Compare current line builder under two treatment filters.
    ###########################################################################

    line_variant_results = {}

    line_variant_frames = {}

    transition_frames = {}

    for treatment_variant_name in (
        "RAW",
        "DROP_INVESTIGATIONAL_DRUG",
    ):

        tx = treatment_variant(
            treatment,
            treatment_variant_name,
        )

        (
            _,
            lookup,
            builder_audit,
        ) = a4b.build_line_intervals(
            tx,
            imaging,
            mapped_patients,
        )

        attached = attach_candidate_lines(
            matches,
            lookup,
            a4b,
        )

        all_summary, all_rows, all_transitions = (
            line_geometry(
                attached
            )
        )

        stage4_rows_input = attached[
            attached[
                "bpc_patient_id"
            ].isin(
                stage4_patients
            )
        ].copy()

        (
            stage4_summary,
            stage4_rows,
            stage4_transitions,
        ) = line_geometry(
            stage4_rows_input
        )

        line_variant_results[
            treatment_variant_name
        ] = {
            "builder_audit":
                builder_audit,

            "all":
                all_summary,

            "stage4":
                stage4_summary,
        }

        line_variant_frames[
            treatment_variant_name
        ] = all_rows

        transition_frames[
            treatment_variant_name
        ] = all_transitions

        if treatment_variant_name == "DROP_INVESTIGATIONAL_DRUG":

            atomic_parquet(
                out
                / "line_matches_drop_investigational.parquet",
                all_rows,
            )

            atomic_parquet(
                out
                / "line_transitions_drop_investigational.parquet",
                all_transitions,
            )

    ###########################################################################
    # Native source alignments.
    ###########################################################################

    (
        treatment_alignment,
        treatment_summary,
    ) = treatment_stream_alignment(
        root,
        offsets,
        treatment,
        stage4_patients,
    )

    (
        progression_alignment,
        progression_summary,
    ) = progression_stream_alignment(
        root,
        offsets,
        imaging,
        stage4_patients,
    )

    atomic_parquet(
        out
        / "treatment_stream_alignment.parquet",
        treatment_alignment,
    )

    atomic_parquet(
        out
        / "progression_stream_alignment.parquet",
        progression_alignment,
    )

    ###########################################################################
    # Decision diagnosis.
    ###########################################################################

    filtered = line_variant_results[
        "DROP_INVESTIGATIONAL_DRUG"
    ]

    adjusted = (
        filtered[
            "all"
        ]
        .get(
            "patient_offset_adjusted_agreement"
        )
    )

    transition = (
        filtered[
            "all"
        ]
        .get(
            "transition_geometry",
            {}
        )
        .get(
            "agreement"
        )
    )

    stage4_absolute = (
        filtered[
            "stage4"
        ]
        .get(
            "absolute_line_agreement"
        )
    )

    stage4_adjusted = (
        filtered[
            "stage4"
        ]
        .get(
            "patient_offset_adjusted_agreement"
        )
    )

    if (
        adjusted is not None
        and adjusted
        >= 0.90
        and transition is not None
        and transition
        >= 0.90
    ):

        diagnosis = (
            "LINE_ZERO_POINT_DOMINANT"
        )

    elif (
        transition is not None
        and transition
        >= 0.85
    ):

        diagnosis = (
            "ZERO_POINT_PLUS_SOME_BOUNDARY_MISMATCH"
        )

    else:

        diagnosis = (
            "BPC_CHORD_LINE_BOUNDARIES_NOT_EQUIVALENT"
        )

    report = {
        "status":
            "MSK_LINE_BOUNDARY_STREAM_AUDIT_COMPLETE",

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

        "stage4_patients_in_bpc":
            int(
                len(
                    stage4_patients
                )
            ),

        "line_variants":
            line_variant_results,

        "treatment_stream":
            treatment_summary,

        "progression_stream":
            progression_summary,

        "diagnosis":
            diagnosis,

        "decision_values": {
            "absolute_line_agreement":
                filtered[
                    "all"
                ]
                .get(
                    "absolute_line_agreement"
                ),

            "patient_offset_adjusted_line_agreement":
                adjusted,

            "transition_agreement":
                transition,

            "stage4_absolute_line_agreement":
                stage4_absolute,

            "stage4_offset_adjusted_line_agreement":
                stage4_adjusted,
        },

        "next_action": (
            "Do not modify A4B or open DFCI/VICC outcomes. "
            "Use stream alignment and offset-adjusted boundary geometry "
            "to determine whether the remaining mismatch is line zero-point "
            "or fundamentally non-equivalent treatment/progression semantics."
        ),
    }

    atomic_json(
        out
        / "audit.json",
        report,
    )

    ###########################################################################
    # Compact output.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A4B FIX6 SUMMARY =========="
    )

    print(
        "status=MSK_LINE_BOUNDARY_STREAM_AUDIT_COMPLETE"
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
        "stage4_patients="
        f"{len(stage4_patients)}"
    )

    print(
        "diagnosis="
        f"{diagnosis}"
    )

    print(
        "========== CKPT7A4B FIX6 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A4B FIX6 DECISION PACKET =========="
    )

    print("")
    print(
        "----- LINE GEOMETRY -----"
    )

    print(
        json.dumps(
            line_variant_results,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print(
        "----- LINE GEOMETRY END -----"
    )

    print("")
    print(
        "----- TREATMENT STREAM ALIGNMENT -----"
    )

    print(
        json.dumps(
            treatment_summary,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print(
        "----- TREATMENT STREAM ALIGNMENT END -----"
    )

    print("")
    print(
        "----- PROGRESSION STREAM ALIGNMENT -----"
    )

    print(
        json.dumps(
            progression_summary,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print(
        "----- PROGRESSION STREAM ALIGNMENT END -----"
    )

    print("")
    print(
        "decision_values="
        + json.dumps(
            report[
                "decision_values"
            ],
            sort_keys=True,
            default=str,
        )
    )

    print(
        "diagnosis="
        f"{diagnosis}"
    )

    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A4B FIX6 DECISION PACKET END =========="
    )


if __name__ == "__main__":

    main()
