#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


PRIMARY_VARIANT = "timeline_sequencing"
SENSITIVITY_VARIANT = "clinical_sample"

EXTERNAL_SITES = (
    "DFCI",
    "VICC",
)

ALPHA = 0.75

EXPECTED_DIMS = {
    "temporal": 192,
    "genomic": 128,
    "scan": 18,
    "context": 6,
    "months": 24,
    "causes": 4,
}

FORBIDDEN_RELATIVE_PATHS = {
    "cBioPortal_files/data_clinical_supp_survival.txt",
    "cBioPortal_files/data_clinical_supp_survival_treatment.txt",
    "clinical_data/cancer_level_dataset_index.csv",
    "clinical_data/cancer_panel_test_level_dataset.csv",
    "clinical_data/patient_level_dataset.csv",
    "clinical_data/regimen_cancer_level_dataset.csv",
}

SAFE_BPC_FILES = {
    "diagnosis":
        "cBioPortal_files/data_timeline_cancer_diagnosis.txt",

    "imaging":
        "cBioPortal_files/data_timeline_imaging.txt",

    "lab":
        "cBioPortal_files/data_timeline_labtest.txt",

    "medonc":
        "cBioPortal_files/data_timeline_medonc.txt",

    "pathology":
        "cBioPortal_files/data_timeline_pathology.txt",

    "sample_acquisition":
        "cBioPortal_files/data_timeline_sample_acquisition.txt",

    "sequencing":
        "cBioPortal_files/data_timeline_sequencing.txt",

    "treatment":
        "cBioPortal_files/data_timeline_treatment.txt",

    "clinical_sample":
        "cBioPortal_files/data_clinical_sample.txt",
}


###############################################################################
# Generic utilities
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


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(block)

    return h.hexdigest()


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
            default=lambda value:
                int(value)
                if isinstance(
                    value,
                    np.integer,
                )
                else float(value)
                if isinstance(
                    value,
                    np.floating,
                )
                else bool(value)
                if isinstance(
                    value,
                    np.bool_,
                )
                else str(value),
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
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def atomic_npy(
    path: Path,
    array: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path)
        + ".tmp"
    )

    with tmp.open(
        "wb"
    ) as f:

        np.save(
            f,
            array,
        )

    check = np.load(
        tmp,
        mmap_mode="r",
    )

    if check.shape != array.shape:

        raise RuntimeError(
            f"NPY verification failed: {path}"
        )

    tmp.replace(
        path
    )


def find_col(
    columns,
    candidates,
    *,
    required=True,
):

    lookup = {
        norm(column):
            column
        for column in columns
    }

    for candidate in candidates:

        key = norm(
            candidate
        )

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


###############################################################################
# Explicit predictor-only BPC access
###############################################################################


class PredictorReader:

    def __init__(
        self,
        root: Path,
    ):

        self.root = root

        self.accessed = []

    def read(
        self,
        key: str,
    ) -> pd.DataFrame:

        relative = SAFE_BPC_FILES[
            key
        ]

        if relative in FORBIDDEN_RELATIVE_PATHS:

            raise RuntimeError(
                "Attempt to access quarantined outcome table."
            )

        lowered = relative.lower()

        if any(
            token in lowered
            for token in (
                "survival",
                "outcome",
                "censor",
                "pfs",
            )
        ):

            raise RuntimeError(
                f"Outcome-like path rejected: {relative}"
            )

        path = (
            self.root
            / relative
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        frame = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
        )

        self.accessed.append(
            relative
        )

        return frame

    def audit(
        self,
    ) -> dict[str, Any]:

        violations = [
            relative
            for relative
            in self.accessed
            if relative
            in FORBIDDEN_RELATIVE_PATHS
            or any(
                token
                in relative.lower()
                for token in (
                    "survival",
                    "outcome",
                    "censor",
                    "pfs",
                )
            )
        ]

        return {
            "accessed":
                list(
                    self.accessed
                ),

            "violations":
                violations,

            "external_outcomes_opened":
                False,
        }


###############################################################################
# Predictor-side treatment-line adapter
###############################################################################


def build_line_intervals(
    treatment: pd.DataFrame,
    imaging: pd.DataFrame,
    external_patients: set[str],
) -> tuple[
    pd.DataFrame,
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """
    Reproduce the frozen CKPT1 treatment-line boundary semantics without
    touching outcome tables.

    Frozen CKPT1 semantics used here:
      * line starts at a treatment START_DATE;
      * nominal split = line_start + 365 days;
      * split treatment must introduce a new AGENT when AGENT is available;
      * observed radiology progression >= 28 days after line start can define
        the next boundary;
      * when progression defines the boundary, the next line may begin at the
        first new treatment in the preceding 28-day grace window.

    BPC imaging is translated to the CHORD Y/N progression stream using only
    observed predictor-side curated scan assessments:
      progressing/worsening/enlarging -> Y
      stable/no change                -> N
      improving/responding            -> N

    Mixed, explicit indeterminate, and blank/unobserved assessments are not
    fabricated into Y or N.
    """

    DAY_SPLIT = 365
    GRACE_DAYS = 28

    patient_col = find_col(
        treatment.columns,
        [
            "PATIENT_ID",
        ],
    )

    start_col = find_col(
        treatment.columns,
        [
            "START_DATE",
        ],
    )

    stop_col = find_col(
        treatment.columns,
        [
            "STOP_DATE",
        ],
        required=False,
    )

    agent_col = find_col(
        treatment.columns,
        [
            "AGENT",
        ],
        required=False,
    )

    regimen_number_col = find_col(
        treatment.columns,
        [
            "REGIMEN_NUMBER",
        ],
        required=False,
    )

    regimen_col = find_col(
        treatment.columns,
        [
            "REGIMEN",
        ],
        required=False,
    )

    tx = treatment.copy()

    tx[
        "_patient"
    ] = (
        tx[
            patient_col
        ]
        .astype(str)
        .str.strip()
    )

    tx = tx[
        tx[
            "_patient"
        ].isin(
            external_patients
        )
    ].copy()

    tx[
        "_start"
    ] = pd.to_numeric(
        tx[
            start_col
        ],
        errors="coerce",
    )

    tx = tx[
        tx[
            "_start"
        ].notna()
    ].copy()

    tx[
        "_start"
    ] = tx[
        "_start"
    ].astype(
        int
    )

    if stop_col is not None:

        tx[
            "_stop"
        ] = pd.to_numeric(
            tx[
                stop_col
            ],
            errors="coerce",
        )

    else:

        tx[
            "_stop"
        ] = np.nan

    if agent_col is not None:

        tx[
            "_agent"
        ] = (
            tx[
                agent_col
            ]
            .fillna("")
            .astype(str)
        )

    else:

        tx[
            "_agent"
        ] = ""

    if regimen_col is not None:

        tx[
            "_regimen"
        ] = (
            tx[
                regimen_col
            ]
            .fillna("")
            .astype(str)
        )

    else:

        tx[
            "_regimen"
        ] = ""

    if regimen_number_col is not None:

        tx[
            "_raw_regimen_number"
        ] = pd.to_numeric(
            tx[
                regimen_number_col
            ],
            errors="coerce",
        )

    else:

        tx[
            "_raw_regimen_number"
        ] = np.nan

    ###########################################################################
    # BPC predictor-side observed Y/N progression stream.
    ###########################################################################

    imaging_patient_col = find_col(
        imaging.columns,
        [
            "PATIENT_ID",
        ],
    )

    imaging_day_col = find_col(
        imaging.columns,
        [
            "START_DATE",
        ],
    )

    curated_col = find_col(
        imaging.columns,
        [
            "CURATED_CANCER_STATUS",
        ],
    )

    rad = imaging.copy()

    rad[
        "_patient"
    ] = (
        rad[
            imaging_patient_col
        ]
        .astype(str)
        .str.strip()
    )

    rad = rad[
        rad[
            "_patient"
        ].isin(
            external_patients
        )
    ].copy()

    rad[
        "_day"
    ] = pd.to_numeric(
        rad[
            imaging_day_col
        ],
        errors="coerce",
    )

    status = (
        rad[
            curated_col
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    progression = pd.Series(
        "",
        index=rad.index,
        dtype=object,
    )

    progression.loc[
        status
        == "PROGRESSING/WORSENING/ENLARGING"
    ] = "Y"

    progression.loc[
        status.isin(
            {
                "STABLE/NO CHANGE",
                "IMPROVING/RESPONDING",
            }
        )
    ] = "N"

    rad[
        "_progression"
    ] = progression

    rad = rad[
        rad[
            "_day"
        ].notna()
        & rad[
            "_progression"
        ].isin(
            {
                "Y",
                "N",
            }
        )
    ].copy()

    rad[
        "_day"
    ] = rad[
        "_day"
    ].astype(
        int
    )

    progression_groups = {
        patient:
            (
                group[
                    [
                        "_day",
                        "_progression",
                    ]
                ]
                .sort_values(
                    "_day",
                    kind="mergesort",
                )
                .reset_index(
                    drop=True
                )
            )
        for patient, group
        in rad.groupby(
            "_patient",
            observed=True,
        )
    }

    ###########################################################################
    # Exact helper reproduced from the frozen CKPT1 line algorithm.
    ###########################################################################

    def next_treatment_index(
        start_dates,
        current_idx,
        target_day,
        *,
        inclusive,
        agent_labels,
        require_new_agent,
    ):

        side = (
            "left"
            if inclusive
            else "right"
        )

        idx = int(
            np.searchsorted(
                start_dates,
                target_day,
                side=side,
            )
        )

        idx = max(
            idx,
            current_idx
            + 1,
        )

        idx = min(
            idx,
            len(
                start_dates
            ),
        )

        if (
            require_new_agent
            and idx
            < len(
                start_dates
            )
            and agent_labels
            is not None
        ):

            last_idx = max(
                min(
                    idx
                    - 1,
                    len(
                        agent_labels
                    )
                    - 1,
                ),
                current_idx,
            )

            last_agent = (
                agent_labels[
                    last_idx
                ]
            )

            while (
                idx
                < len(
                    start_dates
                )
                and (
                    agent_labels[
                        idx
                    ]
                    == last_agent
                )
            ):

                idx += 1

        return idx

    ###########################################################################
    # Construct CKPT1-style line starts.
    ###########################################################################

    all_lines = []

    treatment_row_assignments = []

    for patient, patient_tx in tx.groupby(
        "_patient",
        observed=True,
    ):

        patient_tx = (
            patient_tx
            .sort_values(
                "_start",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
            .copy()
        )

        starts = patient_tx[
            "_start"
        ].to_numpy(
            dtype=int
        )

        n_tx = len(
            starts
        )

        if n_tx == 0:
            continue

        agents = (
            patient_tx[
                "_agent"
            ]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

        events = progression_groups.get(
            patient,
            pd.DataFrame(
                columns=[
                    "_day",
                    "_progression",
                ]
            ),
        )

        event_days = (
            events[
                "_day"
            ].to_numpy(
                dtype=int
            )
            if not events.empty
            else np.array(
                [],
                dtype=int,
            )
        )

        event_types = (
            events[
                "_progression"
            ]
            .astype(str)
            .str.upper()
            .to_numpy()
            if not events.empty
            else np.array(
                [],
                dtype=str,
            )
        )

        treatment_lines = np.empty(
            n_tx,
            dtype=np.int32,
        )

        event_ptr = 0
        idx = 0
        current_line = 1

        patient_line_rows = []

        while idx < n_tx:

            line_start_idx = idx

            line_start = int(
                starts[
                    line_start_idx
                ]
            )

            split_day = (
                line_start
                + DAY_SPLIT
            )

            split_idx = next_treatment_index(
                starts,
                line_start_idx,
                split_day,
                inclusive=True,
                agent_labels=agents,
                require_new_agent=True,
            )

            boundary_day = (
                int(
                    starts[
                        split_idx
                    ]
                )
                if split_idx
                < n_tx
                else None
            )

            search_ptr = event_ptr

            ###################################################################
            # Frozen CKPT1 ignores progression observations at or before the
            # line start.
            ###################################################################

            while (
                search_ptr
                < len(
                    event_days
                )
                and event_days[
                    search_ptr
                ]
                <= line_start
            ):

                search_ptr += 1

            ptr = search_ptr

            progression_event_day = None

            while (
                ptr
                < len(
                    event_days
                )
                and (
                    boundary_day
                    is None
                    or event_days[
                        ptr
                    ]
                    <= boundary_day
                )
            ):

                day = int(
                    event_days[
                        ptr
                    ]
                )

                state = str(
                    event_types[
                        ptr
                    ]
                )

                if state == "N":

                    ptr += 1
                    continue

                if state == "Y":

                    if (
                        day
                        >= line_start
                        + GRACE_DAYS
                    ):

                        progression_event_day = day

                        ptr += 1
                        break

                    ################################################################
                    # Progression during the grace period does not terminate
                    # the line.
                    ################################################################

                    ptr += 1
                    continue

                raise RuntimeError(
                    f"Unexpected BPC progression state: {state}"
                )

            event_ptr = ptr

            boundary_reason = (
                "DAY_SPLIT_OR_NEXT_AGENT"
            )

            if progression_event_day is not None:

                progression_next_idx = next_treatment_index(
                    starts,
                    line_start_idx,
                    int(
                        progression_event_day
                    ),
                    inclusive=True,
                    agent_labels=agents,
                    require_new_agent=True,
                )

                next_idx = progression_next_idx

                if (
                    progression_next_idx
                    > line_start_idx
                ):

                    window_start = (
                        int(
                            progression_event_day
                        )
                        - GRACE_DAYS
                    )

                    window = (
                        starts[
                            line_start_idx:
                            progression_next_idx
                        ]
                        > window_start
                    )

                    if window.any():

                        next_idx = (
                            int(
                                np.argmax(
                                    window
                                )
                            )
                            + line_start_idx
                        )

                boundary_reason = (
                    "OBSERVED_PROGRESSION"
                )

            else:

                next_idx = split_idx

            if next_idx <= line_start_idx:

                raise RuntimeError(
                    "Non-advancing CKPT1-style line boundary: "
                    f"patient={patient} "
                    f"idx={line_start_idx} "
                    f"next_idx={next_idx}"
                )

            treatment_lines[
                line_start_idx:
                next_idx
            ] = current_line

            patient_line_rows.append(
                {
                    "patient_id":
                        patient,

                    "treatment_line":
                        int(
                            current_line
                        ),

                    "line_start_day":
                        float(
                            line_start
                        ),

                    "line_boundary_reason":
                        boundary_reason,

                    "progression_boundary_day":
                        (
                            float(
                                progression_event_day
                            )
                            if progression_event_day
                            is not None
                            else np.nan
                        ),
                }
            )

            if next_idx >= n_tx:
                break

            idx = next_idx

            current_line += 1

        patient_tx[
            "_ckpt1_line"
        ] = treatment_lines

        treatment_row_assignments.append(
            patient_tx
        )

        line_frame = pd.DataFrame(
            patient_line_rows
        )

        if line_frame.empty:
            continue

        next_starts = (
            line_frame[
                "line_start_day"
            ].shift(
                -1
            )
        )

        line_frame[
            "line_stop_day"
        ] = np.where(
            next_starts.notna(),
            next_starts
            - 1.0,
            np.inf,
        )

        all_lines.append(
            line_frame
        )

    if not all_lines:

        raise RuntimeError(
            "No CKPT1-style external treatment lines constructed."
        )

    interval_frame = pd.concat(
        all_lines,
        ignore_index=True,
    )

    interval_frame = interval_frame.sort_values(
        [
            "patient_id",
            "treatment_line",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    if interval_frame[
        [
            "patient_id",
            "treatment_line",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate CKPT1-style line keys."
        )

    if (
        interval_frame[
            "line_start_day"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Missing CKPT1-style line start."
        )

    ###########################################################################
    # Lookup used by external scans and events.
    ###########################################################################

    lookup = {}

    for patient, group in interval_frame.groupby(
        "patient_id",
        observed=True,
    ):

        lookup[
            patient
        ] = (
            group.sort_values(
                [
                    "line_start_day",
                    "treatment_line",
                ],
                kind="mergesort",
            )
            .to_dict(
                "records"
            )
        )

    treatment_assignment_frame = (
        pd.concat(
            treatment_row_assignments,
            ignore_index=True,
        )
        if treatment_row_assignments
        else pd.DataFrame()
    )

    audit = {
        "algorithm":
            "FROZEN_CKPT1_LINE_BOUNDARY_REPLAY",

        "day_split":
            DAY_SPLIT,

        "progression_grace_days":
            GRACE_DAYS,

        "line_patients":
            int(
                interval_frame[
                    "patient_id"
                ].nunique()
            ),

        "lines":
            int(
                len(
                    interval_frame
                )
            ),

        "treatment_rows":
            int(
                len(
                    treatment_assignment_frame
                )
            ),

        "progression_rows_y":
            int(
                (
                    rad[
                        "_progression"
                    ]
                    == "Y"
                ).sum()
            ),

        "progression_rows_n":
            int(
                (
                    rad[
                        "_progression"
                    ]
                    == "N"
                ).sum()
            ),

        "boundary_reasons":
            {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in interval_frame[
                    "line_boundary_reason"
                ]
                .value_counts()
                .to_dict()
                .items()
            },

        "regimen_number_used_for_line_identity":
            False,

        "external_outcomes_used":
            False,
    }

    return (
        interval_frame,
        lookup,
        audit,
    )


def assign_line(
    patient: str,
    day: float,
    lookup: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ],
) -> tuple[
    int,
    float,
    float,
]:

    candidates = []

    for record in lookup.get(
        patient,
        [],
    ):

        start = float(
            record[
                "line_start_day"
            ]
        )

        stop = float(
            record[
                "line_stop_day"
            ]
        )

        if (
            start
            <= day
            <= stop
        ):

            candidates.append(
                record
            )

    if not candidates:

        return (
            0,
            np.nan,
            np.nan,
        )

    chosen = sorted(
        candidates,
        key=lambda record:
            (
                float(
                    record[
                        "line_start_day"
                    ]
                ),
                int(
                    record[
                        "treatment_line"
                    ]
                ),
            ),
    )[
        -1
    ]

    return (
        int(
            chosen[
                "treatment_line"
            ]
        ),
        float(
            chosen[
                "line_start_day"
            ]
        ),
        float(
            chosen[
                "line_stop_day"
            ]
        ),
    )


###############################################################################
# A3 W3 scan surface + active-line assignment
###############################################################################


def prepare_external_scans(
    repo: Path,
    line_lookup,
    feature_names,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    dict[str, Any],
]:

    episode_path = (
        repo
        / "artifacts/checkpoint7a3/"
        "bpc_w3_predictor_episodes.parquet"
    )

    feature_path = (
        repo
        / "artifacts/checkpoint7a3/"
        "bpc_w3_scan_features_f32.npy"
    )

    episodes = pd.read_parquet(
        episode_path
    )

    features = np.load(
        feature_path,
        mmap_mode="r",
    )

    if (
        features.ndim
        != 2
        or features.shape[
            1
        ]
        != 18
        or features.shape[
            0
        ]
        != len(
            episodes
        )
    ):

        raise RuntimeError(
            "A3 scan-feature artifact alignment failure."
        )

    patient_col = find_col(
        episodes.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    site_col = find_col(
        episodes.columns,
        [
            "site",
            "institution",
        ],
    )

    episode_col = find_col(
        episodes.columns,
        [
            "bpc_episode_id",
            "scan_episode_id",
            "episode_id",
        ],
    )

    state_col = find_col(
        episodes.columns,
        [
            "scan_state",
            "progression_state_3",
        ],
    )

    day_col = find_col(
        episodes.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
    )

    episodes = episodes.copy()

    episodes[
        "_feature_row"
    ] = np.arange(
        len(
            episodes
        ),
        dtype=int,
    )

    episodes[
        "patient_id"
    ] = (
        episodes[
            patient_col
        ]
        .astype(str)
        .str.strip()
    )

    episodes[
        "site"
    ] = (
        episodes[
            site_col
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    episodes[
        "bpc_episode_id"
    ] = (
        episodes[
            episode_col
        ]
        .astype(str)
        .str.strip()
    )

    episodes[
        "scan_state"
    ] = (
        episodes[
            state_col
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    episodes[
        "landmark_day"
    ] = pd.to_numeric(
        episodes[
            day_col
        ],
        errors="coerce",
    )

    episodes = episodes[
        episodes[
            "site"
        ].isin(
            EXTERNAL_SITES
        )
        & episodes[
            "landmark_day"
        ].notna()
    ].copy()

    if episodes[
        [
            "patient_id",
            "bpc_episode_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate external W3 episode keys."
        )

    line_values = []

    for _, row in episodes.iterrows():

        line_values.append(
            assign_line(
                str(
                    row[
                        "patient_id"
                    ]
                ),
                float(
                    row[
                        "landmark_day"
                    ]
                ),
                line_lookup,
            )
        )

    episodes[
        "treatment_line"
    ] = [
        value[
            0
        ]
        for value
        in line_values
    ]

    episodes[
        "line_start_day"
    ] = [
        value[
            1
        ]
        for value
        in line_values
    ]

    episodes[
        "line_stop_day"
    ] = [
        value[
            2
        ]
        for value
        in line_values
    ]

    episodes[
        "active_line"
    ] = (
        episodes[
            "treatment_line"
        ]
        > 0
    )

    episodes = episodes.sort_values(
        [
            "patient_id",
            "landmark_day",
            "bpc_episode_id",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    episodes[
        "scan_number_patient"
    ] = (
        episodes.groupby(
            "patient_id",
            observed=True,
        )
        .cumcount()
        + 1
    )

    episodes[
        "scan_number_line"
    ] = 0

    active_mask = episodes[
        "active_line"
    ].astype(
        bool
    )

    active = episodes.loc[
        active_mask
    ].copy()

    line_counts = (
        active.groupby(
            [
                "patient_id",
                "treatment_line",
            ],
            observed=True,
            sort=False,
        )
        .cumcount()
        + 1
    )

    episodes.loc[
        active.index,
        "scan_number_line",
    ] = line_counts.to_numpy(
        dtype=int
    )

    ###########################################################################
    # Feature rows.
    ###########################################################################

    feature_rows = features[
        episodes[
            "_feature_row"
        ].to_numpy(
            dtype=int
        )
    ].astype(
        np.float32
    )

    feature_index = {
        name:
            index
        for index, name
        in enumerate(
            feature_names
        )
    }

    for expected in (
        "raw_coverage_chest",
        "raw_coverage_abdomen",
        "raw_coverage_pelvis",
        "raw_coverage_head",
        "raw_coverage_other",
        "site_bone",
        "site_liver",
        "site_lung",
        "site_brain",
        "site_lymph",
        "site_pleura",
    ):

        if expected not in feature_index:

            raise RuntimeError(
                f"Missing frozen feature: {expected}"
            )

    scan_table = episodes[
        [
            "patient_id",
            "site",
            "bpc_episode_id",
            "landmark_day",
            "scan_state",
            "treatment_line",
        ]
    ].copy()

    scan_table[
        "scan_episode_id"
    ] = scan_table[
        "bpc_episode_id"
    ]

    scan_table[
        "progression_state_3"
    ] = scan_table[
        "scan_state"
    ].where(
        scan_table[
            "scan_state"
        ].isin(
            [
                "NON_PROGRESSIVE",
                "PROGRESSIVE",
                "INDETERMINATE",
            ]
        ),
        "",
    )

    for region in (
        "chest",
        "abdomen",
        "pelvis",
        "head",
        "other",
    ):

        values = feature_rows[
            :,
            feature_index[
                f"raw_coverage_{region}"
            ],
        ]

        scan_table[
            f"coverage_{region}"
        ] = values

        scan_table[
            f"coverage_state_{region}"
        ] = np.where(
            values
            > 0.5,
            "OBSERVED",
            "UNOBSERVED",
        )

    site_feature_to_vocab = {
        "site_bone":
            "BONE",

        "site_liver":
            "LIVER",

        "site_lung":
            "LUNG",

        "site_brain":
            "CNS/BRAIN",

        "site_lymph":
            "LYMPH NODES",

        "site_pleura":
            "PLEURA",
    }

    site_strings = []

    for row_number in range(
        len(
            scan_table
        )
    ):

        labels = []

        for feature_name, vocab_name in site_feature_to_vocab.items():

            if (
                feature_rows[
                    row_number,
                    feature_index[
                        feature_name
                    ],
                ]
                > 0.5
            ):

                labels.append(
                    vocab_name
                )

        site_strings.append(
            " | ".join(
                labels
            )
        )

    scan_table[
        "tumor_sites"
    ] = site_strings

    primary_candidates = episodes[
        episodes[
            "scan_state"
        ]
        == "NON_PROGRESSIVE"
    ].copy()

    primary_active = primary_candidates[
        primary_candidates[
            "active_line"
        ]
    ].copy()

    audit = {
        "all_external_episodes":
            int(
                len(
                    episodes
                )
            ),

        "all_external_patients":
            int(
                episodes[
                    "patient_id"
                ].nunique()
            ),

        "non_progressive_candidates":
            int(
                len(
                    primary_candidates
                )
            ),

        "active_line_non_progressive":
            int(
                len(
                    primary_active
                )
            ),

        "active_fraction_of_nonprogressive":
            float(
                len(
                    primary_active
                )
                / max(
                    len(
                        primary_candidates
                    ),
                    1,
                )
            ),

        "site": {},
    }

    for site in EXTERNAL_SITES:

        site_candidates = primary_candidates[
            primary_candidates[
                "site"
            ]
            == site
        ]

        site_active = primary_active[
            primary_active[
                "site"
            ]
            == site
        ]

        audit[
            "site"
        ][
            site
        ] = {
            "candidate_landmarks":
                int(
                    len(
                        site_candidates
                    )
                ),

            "candidate_patients":
                int(
                    site_candidates[
                        "patient_id"
                    ].nunique()
                ),

            "active_landmarks":
                int(
                    len(
                        site_active
                    )
                ),

            "active_patients":
                int(
                    site_active[
                        "patient_id"
                    ].nunique()
                ),
        }

    return (
        episodes,
        feature_rows,
        scan_table,
        audit,
    )


###############################################################################
# Frozen-source mapping for CKPT4 event source embeddings
###############################################################################


def chord_source_map(
    repo: Path,
) -> dict[str, str]:

    canonical = pd.read_parquet(
        repo
        / "artifacts/checkpoint1/canonical/"
        "chord_canonical_events.parquet"
    )

    event_col = find_col(
        canonical.columns,
        [
            "event_type",
            "EVENT_TYPE",
        ],
    )

    source_col = find_col(
        canonical.columns,
        [
            "source",
            "SOURCE",
            "provenance",
        ],
        required=False,
    )

    if source_col is None:

        return {}

    result = {}

    for event_type, group in canonical.groupby(
        event_col,
        observed=True,
    ):

        values = (
            group[
                source_col
            ]
            .map(
                clean
            )
        )

        values = values[
            values.ne("")
        ]

        if len(
            values
        ):

            result[
                str(
                    event_type
                ).strip().upper()
            ] = (
                values.value_counts()
                .index[
                    0
                ]
            )

    return result


###############################################################################
# External predictor-event corpus
###############################################################################


def safe_payload(
    row: pd.Series,
    fields: list[str],
) -> str:

    values = []

    for field in fields:

        if field not in row.index:
            continue

        value = clean(
            row[
                field
            ]
        )

        if not value:
            continue

        values.append(
            f"{field}={value}"
        )

    return " | ".join(
        values
    )


def build_canonical_events(
    reader: PredictorReader,
    external_patients: set[str],
    line_lookup,
    source_map,
    genomic_variant: str,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    records = []

    def source_for(
        event_type: str,
    ) -> str:

        return source_map.get(
            event_type,
            event_type,
        )

    def add_event(
        patient: str,
        day: float,
        event_type: str,
        *,
        line: int | None = None,
        payload: str = "",
        numeric_value: float | None = None,
    ):

        if (
            patient
            not in external_patients
            or not np.isfinite(
                day
            )
        ):
            return

        if line is None:

            line = assign_line(
                patient,
                float(
                    day
                ),
                line_lookup,
            )[
                0
            ]

        records.append(
            {
                "patient_id":
                    patient,

                "availability_day":
                    float(
                        day
                    ),

                "event_type":
                    event_type,

                "treatment_line":
                    int(
                        line
                    ),

                "source":
                    source_for(
                        event_type
                    ),

                "payload_text":
                    payload,

                "value_numeric":
                    (
                        float(
                            numeric_value
                        )
                        if (
                            numeric_value
                            is not None
                            and np.isfinite(
                                numeric_value
                            )
                        )
                        else np.nan
                    ),
            }
        )

    ###########################################################################
    # Diagnosis
    ###########################################################################

    diagnosis = reader.read(
        "diagnosis"
    )

    pid = find_col(
        diagnosis.columns,
        ["PATIENT_ID"],
    )

    day_col = find_col(
        diagnosis.columns,
        ["START_DATE"],
    )

    diagnosis_fields = [
        "AGE_DX",
        "BCA_SUBTYPE",
        "CA_BCA_ER",
        "CA_BCA_HER_SUMM",
        "CA_BCA_HER2IHC_INTP",
        "CA_BCA_HER2IHC_VAL",
        "CA_BCA_PR",
        "CA_HISTOLOGY",
        "CA_D_TYPE",
        "INDEX_CANCER",
        "STAGE_DX",
    ]

    for _, row in diagnosis.iterrows():

        patient = clean(
            row[
                pid
            ]
        )

        day = pd.to_numeric(
            pd.Series(
                [
                    row[
                        day_col
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        if pd.isna(
            day
        ):
            continue

        add_event(
            patient,
            float(
                day
            ),
            "DIAGNOSIS",
            payload=safe_payload(
                row,
                diagnosis_fields,
            ),
        )

    ###########################################################################
    # Treatments -> START / END.
    ###########################################################################

    treatment = reader.read(
        "treatment"
    )

    pid = find_col(
        treatment.columns,
        ["PATIENT_ID"],
    )

    start_col = find_col(
        treatment.columns,
        ["START_DATE"],
    )

    stop_col = find_col(
        treatment.columns,
        ["STOP_DATE"],
        required=False,
    )

    treatment_fields = [
        "TREATMENT_TYPE",
        "AGENT",
        "REGIMEN",
        "REGIMEN_NUMBER",
        "DRUGS_CT_YN",
        "DRUGS_DC_YN",
    ]

    for _, row in treatment.iterrows():

        patient = clean(
            row[
                pid
            ]
        )

        if patient not in external_patients:
            continue

        start = pd.to_numeric(
            pd.Series(
                [
                    row[
                        start_col
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        if pd.isna(
            start
        ):
            continue

        line = assign_line(
            patient,
            float(
                start
            ),
            line_lookup,
        )[
            0
        ]

        payload = safe_payload(
            row,
            treatment_fields,
        )

        add_event(
            patient,
            float(
                start
            ),
            "TREATMENT_START",
            line=line,
            payload=payload,
        )

        if stop_col:

            stop = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            stop_col
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

            if (
                pd.notna(
                    stop
                )
                and float(
                    stop
                )
                >= float(
                    start
                )
            ):

                add_event(
                    patient,
                    float(
                        stop
                    ),
                    "TREATMENT_END",
                    line=line,
                    payload=payload,
                )

    ###########################################################################
    # Labs -> TUMOR_MARKER.
    ###########################################################################

    lab = reader.read(
        "lab"
    )

    pid = find_col(
        lab.columns,
        ["PATIENT_ID"],
    )

    day_col = find_col(
        lab.columns,
        ["START_DATE"],
    )

    result_col = find_col(
        lab.columns,
        ["RESULT"],
        required=False,
    )

    for _, row in lab.iterrows():

        patient = clean(
            row[
                pid
            ]
        )

        day = pd.to_numeric(
            pd.Series(
                [
                    row[
                        day_col
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        if pd.isna(
            day
        ):
            continue

        numeric = np.nan

        if result_col:

            numeric = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            result_col
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

        add_event(
            patient,
            float(
                day
            ),
            "TUMOR_MARKER",
            payload=safe_payload(
                row,
                [
                    "TEST",
                    "RESULT",
                    "TM_RESULT_UNITS",
                    "TM_NORMAL_RANGE_UPPER",
                ],
            ),
            numeric_value=(
                float(
                    numeric
                )
                if pd.notna(
                    numeric
                )
                else None
            ),
        )

    ###########################################################################
    # Med-onc -> OTHER. No performance-status field is exposed in this table.
    ###########################################################################

    medonc = reader.read(
        "medonc"
    )

    pid = find_col(
        medonc.columns,
        ["PATIENT_ID"],
    )

    day_col = find_col(
        medonc.columns,
        ["START_DATE"],
    )

    for _, row in medonc.iterrows():

        patient = clean(
            row[
                pid
            ]
        )

        day = pd.to_numeric(
            pd.Series(
                [
                    row[
                        day_col
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        if pd.isna(
            day
        ):
            continue

        add_event(
            patient,
            float(
                day
            ),
            "OTHER",
            payload=safe_payload(
                row,
                [
                    "CANCER_STATUS",
                    "CURATED_CANCER_STATUS",
                    "CANCER_SUBTYPE_CURATED",
                    "VISIT_NUMBER",
                ],
            ),
        )

    ###########################################################################
    # Pathology -> SPECIMEN.
    ###########################################################################

    pathology = reader.read(
        "pathology"
    )

    pid = find_col(
        pathology.columns,
        ["PATIENT_ID"],
    )

    day_col = find_col(
        pathology.columns,
        ["START_DATE"],
    )

    for _, row in pathology.iterrows():

        patient = clean(
            row[
                pid
            ]
        )

        day = pd.to_numeric(
            pd.Series(
                [
                    row[
                        day_col
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        if pd.isna(
            day
        ):
            continue

        add_event(
            patient,
            float(
                day
            ),
            "SPECIMEN",
            payload=safe_payload(
                row,
                [
                    "PATH_PROC_TYPE",
                    "PATH_CA_INV_ANY",
                    "PATH_INSITU_ANY",
                    "N_SPECIMEN_INV",
                    "N_SPECIMEN_INSITU",
                    "PDL1_POSITIVE_ANY",
                    "PDL1_TESTING",
                ],
            ),
        )

    ###########################################################################
    # Sample acquisition -> SPECIMEN.
    ###########################################################################

    acquisition = reader.read(
        "sample_acquisition"
    )

    pid = find_col(
        acquisition.columns,
        ["PATIENT_ID"],
    )

    day_col = find_col(
        acquisition.columns,
        ["START_DATE"],
    )

    for _, row in acquisition.iterrows():

        patient = clean(
            row[
                pid
            ]
        )

        day = pd.to_numeric(
            pd.Series(
                [
                    row[
                        day_col
                    ]
                ]
            ),
            errors="coerce",
        ).iloc[
            0
        ]

        if pd.isna(
            day
        ):
            continue

        add_event(
            patient,
            float(
                day
            ),
            "SPECIMEN",
            payload=safe_payload(
                row,
                [
                    "SAMPLE_ID",
                    "CPT_NUMBER",
                    "PATH_PROC_NUMBER",
                    "CA_SEQ",
                ],
            ),
        )

    ###########################################################################
    # Frozen genomic-availability token.
    #
    # Primary:
    #   BPC timeline sequencing START_DATE.
    #
    # Sensitivity:
    #   clinical sample CPT_SEQ_DATE.
    ###########################################################################

    if genomic_variant == "timeline_sequencing":

        sequencing = reader.read(
            "sequencing"
        )

        pid = find_col(
            sequencing.columns,
            ["PATIENT_ID"],
        )

        day_col = find_col(
            sequencing.columns,
            ["START_DATE"],
        )

        for _, row in sequencing.iterrows():

            patient = clean(
                row[
                    pid
                ]
            )

            day = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            day_col
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

            if pd.isna(
                day
            ):
                continue

            add_event(
                patient,
                float(
                    day
                ),
                "GENOMIC_RESULT_AVAILABLE",
                payload=safe_payload(
                    row,
                    [
                        "SAMPLE_ID",
                        "CPT_NUMBER",
                        "CA_SEQ",
                    ],
                ),
            )

    elif genomic_variant == "clinical_sample":

        sample = reader.read(
            "clinical_sample"
        )

        pid = find_col(
            sample.columns,
            ["PATIENT_ID"],
        )

        day_col = find_col(
            sample.columns,
            [
                "CPT_SEQ_DATE",
            ],
        )

        for _, row in sample.iterrows():

            patient = clean(
                row[
                    pid
                ]
            )

            day = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            day_col
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

            if pd.isna(
                day
            ):
                continue

            add_event(
                patient,
                float(
                    day
                ),
                "GENOMIC_RESULT_AVAILABLE",
                payload=safe_payload(
                    row,
                    [
                        "SAMPLE_ID",
                        "GENE_PANEL",
                        "SAMPLE_TYPE",
                    ],
                ),
            )

    else:

        raise ValueError(
            genomic_variant
        )

    canonical = pd.DataFrame(
        records
    )

    if canonical.empty:

        raise RuntimeError(
            "External canonical event corpus empty."
        )

    canonical = canonical.sort_values(
        [
            "patient_id",
            "availability_day",
            "event_type",
            "payload_text",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    counts = (
        canonical[
            "event_type"
        ]
        .value_counts()
        .to_dict()
    )

    audit = {
        "rows":
            int(
                len(
                    canonical
                )
            ),

        "patients":
            int(
                canonical[
                    "patient_id"
                ].nunique()
            ),

        "event_counts":
            {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in counts.items()
            },

        "genomic_variant":
            genomic_variant,
    }

    return (
        canonical,
        audit,
    )


###############################################################################
# Exact frozen CKPT4 external temporal cache
###############################################################################


def build_temporal_cache(
    repo: Path,
    out: Path,
    ckpt4,
    canonical: pd.DataFrame,
    scan_table: pd.DataFrame,
    external_patients: set[str],
    *,
    batch_size: int,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, Any],
]:

    prep = (
        out
        / "prepared"
    )

    prep.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_path = (
        prep
        / "external_canonical_events.parquet"
    )

    scan_path = (
        prep
        / "external_w3_scans.parquet"
    )

    atomic_parquet(
        canonical_path,
        canonical,
    )

    atomic_parquet(
        scan_path,
        scan_table,
    )

    splits = pd.DataFrame(
        {
            "patient_id":
                sorted(
                    external_patients
                ),

            "split":
                "test",
        }
    )

    site_vocab = [
        line.strip()
        for line
        in (
            repo
            / "artifacts/checkpoint4/prepared/"
            "site_vocab_32.txt"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    breast_base = ckpt4.prepare_breast(
        canonical_path,
        scan_path,
        splits,
        site_vocab,
    )

    tokens = ckpt4.add_time_and_sequence_targets(
        breast_base
    )

    atomic_parquet(
        prep
        / "breast_tokens.parquet",
        tokens,
    )

    scan_index = (
        breast_base[
            breast_base[
                "type_id"
            ]
            == ckpt4.EVENT_TO_ID[
                "SCAN_EPISODE"
            ]
        ][
            [
                "patient_id",
                "day",
                "line",
                "scan_state",
                "scan_episode_id",
                "site_mask",
                "site_known",
                "region_imaged_mask",
                "region_known_mask",
                "split",
            ]
        ]
        .copy()
        .sort_values(
            [
                "patient_id",
                "day",
                "scan_episode_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    if scan_index[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate temporal scan-index keys."
        )

    atomic_parquet(
        prep
        / "breast_scan_index.parquet",
        scan_index,
    )

    shutil.copy2(
        repo
        / "artifacts/checkpoint4/temporal_encoder.pt",
        out
        / "temporal_encoder.pt",
    )

    cache_qc = ckpt4.cache_scan_states(
        out,
        batch_size,
    )

    embedding_index = pd.read_parquet(
        out
        / "breast_scan_prepost_index.parquet"
    )

    embeddings = np.load(
        out
        / "breast_scan_prepost_embeddings_f16.npy",
        mmap_mode="r",
    )

    if embeddings.shape != (
        len(
            embedding_index
        ),
        2,
        192,
    ):

        raise RuntimeError(
            "External temporal cache dimension mismatch."
        )

    return (
        embedding_index,
        embeddings,
        cache_qc,
    )


###############################################################################
# Exact CKPT5 context-transform recovery from frozen CHORD tensors
###############################################################################


def derive_chord_genomic_raw(
    repo: Path,
    scan_index: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    samples = pd.read_parquet(
        repo
        / "artifacts/checkpoint5/prepared/"
        "chord_genomic_samples.parquet"
    )

    patient_col = find_col(
        samples.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    availability_col = find_col(
        samples.columns,
        [
            "availability_day",
            "SEQ_DATE",
            "seq_date",
        ],
    )

    sample_map = {}

    for patient, group in samples.groupby(
        patient_col,
        observed=True,
    ):

        days = (
            pd.to_numeric(
                group[
                    availability_col
                ],
                errors="coerce",
            )
            .dropna()
            .sort_values()
            .to_numpy(
                dtype=float
            )
        )

        sample_map[
            str(
                patient
            )
        ] = days

    patient_col_scan = find_col(
        scan_index.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    day_col = find_col(
        scan_index.columns,
        [
            "landmark_day",
            "day",
        ],
    )

    available = np.zeros(
        len(
            scan_index
        ),
        dtype=float,
    )

    age = np.zeros(
        len(
            scan_index
        ),
        dtype=float,
    )

    for row_number, row in scan_index.iterrows():

        patient = str(
            row[
                patient_col_scan
            ]
        )

        landmark = float(
            row[
                day_col
            ]
        )

        days = sample_map.get(
            patient
        )

        if (
            days is None
            or len(
                days
            )
            == 0
        ):

            continue

        selected = (
            np.searchsorted(
                days,
                landmark,
                side="left",
            )
            - 1
        )

        if selected < 0:
            continue

        available[
            row_number
        ] = 1.0

        age[
            row_number
        ] = (
            landmark
            - days[
                selected
            ]
        )

    return (
        available,
        age,
    )


def chord_raw_context_variables(
    repo: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, np.ndarray],
]:

    index = pd.read_parquet(
        repo
        / "artifacts/checkpoint5/prepared/"
        "scan_index.parquet"
    ).reset_index(
        drop=True
    )

    target = np.load(
        repo
        / "artifacts/checkpoint5/prepared/"
        "context_features_f32.npy"
    ).astype(
        np.float64
    )

    if target.shape != (
        len(
            index
        ),
        6,
    ):

        raise RuntimeError(
            "Frozen CHORD context shape mismatch."
        )

    patient_col = find_col(
        index.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    line_col = find_col(
        index.columns,
        [
            "line",
            "treatment_line",
        ],
    )

    day_col = find_col(
        index.columns,
        [
            "landmark_day",
            "day",
        ],
    )

    episode_col = find_col(
        index.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    work = index.copy()

    work[
        "_row"
    ] = np.arange(
        len(
            work
        ),
        dtype=int,
    )

    work[
        "_patient"
    ] = (
        work[
            patient_col
        ]
        .astype(str)
    )

    work[
        "_line"
    ] = pd.to_numeric(
        work[
            line_col
        ],
        errors="coerce",
    ).fillna(
        0
    )

    work[
        "_day"
    ] = pd.to_numeric(
        work[
            day_col
        ],
        errors="coerce",
    )

    if episode_col:

        work[
            "_episode"
        ] = work[
            episode_col
        ].astype(str)

    else:

        work[
            "_episode"
        ] = work[
            "_row"
        ].astype(str)

    ###########################################################################
    # Exact line starts from the frozen CKPT1 semantic line table.
    #
    # Do NOT reconstruct these from minimum TREATMENT_START: CKPT1 line
    # boundaries depend on the 365-day/new-agent rule and observed
    # progression with the 28-day grace semantics.
    ###########################################################################

    line_table = pd.read_parquet(
        repo
        / "artifacts/checkpoint1/canonical/"
        "chord_mbc_lines.parquet"
    )

    line_patient_col = find_col(
        line_table.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    line_number_col = find_col(
        line_table.columns,
        [
            "treatment_line",
            "line",
            "LINE",
        ],
    )

    line_start_col = find_col(
        line_table.columns,
        [
            "line_start_day",
            "LINE_START",
            "line_start",
        ],
    )

    frozen_lines = pd.DataFrame(
        {
            "_patient":
                line_table[
                    line_patient_col
                ]
                .astype(str)
                .str.strip(),

            "_line":
                pd.to_numeric(
                    line_table[
                        line_number_col
                    ],
                    errors="coerce",
                ),

            "_line_start":
                pd.to_numeric(
                    line_table[
                        line_start_col
                    ],
                    errors="coerce",
                ),
        }
    )

    frozen_lines = frozen_lines.dropna(
        subset=[
            "_line",
            "_line_start",
        ]
    )

    duplicate_starts = (
        frozen_lines.groupby(
            [
                "_patient",
                "_line",
            ],
            observed=True,
        )[
            "_line_start"
        ]
        .nunique()
    )

    if (
        duplicate_starts
        > 1
    ).any():

        raise RuntimeError(
            "Frozen CKPT1 line table has inconsistent "
            "line starts for a patient/line key."
        )

    line_start = (
        frozen_lines.groupby(
            [
                "_patient",
                "_line",
            ],
            observed=True,
        )[
            "_line_start"
        ]
        .first()
        .to_dict()
    )

    line_start_values = np.zeros(
        len(
            work
        ),
        dtype=float,
    )

    elapsed = np.zeros(
        len(
            work
        ),
        dtype=float,
    )

    missing_line_keys = []

    for row_number, row in work.iterrows():

        key = (
            row[
                "_patient"
            ],
            float(
                row[
                    "_line"
                ]
            ),
        )

        start = line_start.get(
            key
        )

        if start is None:

            missing_line_keys.append(
                key
            )

            continue

        line_start_values[
            row_number
        ] = float(
            start
        )

        elapsed[
            row_number
        ] = max(
            float(
                row[
                    "_day"
                ]
            )
            - float(
                start
            ),
            0.0,
        )

    if missing_line_keys:

        unique_missing = list(
            dict.fromkeys(
                missing_line_keys
            )
        )

        raise RuntimeError(
            "Frozen CKPT5 scan index contains line keys absent "
            "from frozen CKPT1 line table: "
            f"{unique_missing[:30]}"
        )

    ###########################################################################
    # Frozen scan ordinals.
    ###########################################################################

    ordered = work.sort_values(
        [
            "_patient",
            "_day",
            "_episode",
        ],
        kind="mergesort",
    ).copy()

    ordered[
        "_scan_patient"
    ] = (
        ordered.groupby(
            "_patient",
            observed=True,
        )
        .cumcount()
        + 1
    )

    ordered[
        "_scan_line"
    ] = (
        ordered.groupby(
            [
                "_patient",
                "_line",
            ],
            observed=True,
        )
        .cumcount()
        + 1
    )

    ordered = ordered.sort_values(
        "_row"
    )

    genomic_available, genomic_age = (
        derive_chord_genomic_raw(
            repo,
            index,
        )
    )

    raw = {
        "line_number_scaled":
            work[
                "_line"
            ].to_numpy(
                dtype=float
            ),

        "elapsed_on_line_scaled":
            elapsed,

        "scan_number_patient_log":
            ordered[
                "_scan_patient"
            ].to_numpy(
                dtype=float
            ),

        "scan_number_line_log":
            ordered[
                "_scan_line"
            ].to_numpy(
                dtype=float
            ),

        "genomic_available":
            genomic_available,

        "genomic_age_scaled":
            genomic_age,
    }

    return (
        index,
        target,
        raw,
    )


def numeric_constants_near_feature(
    source: str,
    feature: str,
) -> set[float]:

    lines = source.splitlines()

    hits = [
        index
        for index, line
        in enumerate(
            lines
        )
        if feature
        in line
    ]

    constants = {
        2.0,
        3.0,
        5.0,
        10.0,
        12.0,
        15.0,
        20.0,
        24.0,
        30.0,
        90.0,
        100.0,
        180.0,
        365.0,
        365.25,
        730.0,
        1000.0,
        1825.0,
        3650.0,
    }

    for hit in hits:

        lo = max(
            0,
            hit
            - 160,
        )

        hi = min(
            len(
                lines
            ),
            hit
            + 161,
        )

        block = "\n".join(
            lines[
                lo:
                hi
            ]
        )

        for match in re.findall(
            r"(?<![A-Za-z_])"
            r"([0-9]+(?:\.[0-9]+)?)",
            block,
        ):

            value = float(
                match
            )

            if (
                value
                > 1.0
                and value
                <= 100000.0
            ):

                constants.add(
                    value
                )

    return constants


def candidate_transforms(
    constants: set[float],
) -> list[
    tuple[
        str,
        Callable[
            [
                np.ndarray
            ],
            np.ndarray,
        ],
    ]
]:

    result = []

    def register(
        description,
        fn,
    ):

        result.append(
            (
                description,
                fn,
            )
        )

    for shift in (
        0.0,
        -1.0,
        1.0,
    ):

        def shifted(
            x,
            shift=shift,
        ):

            return np.maximum(
                np.asarray(
                    x,
                    dtype=float,
                )
                + shift,
                0.0,
            )

        prefix = (
            f"max(x{shift:+g},0)"
        )

        register(
            prefix,
            lambda x, f=shifted:
                f(
                    x
                ),
        )

        register(
            f"log1p({prefix})",
            lambda x, f=shifted:
                np.log1p(
                    f(
                        x
                    )
                ),
        )

        register(
            f"sqrt({prefix})",
            lambda x, f=shifted:
                np.sqrt(
                    f(
                        x
                    )
                ),
        )

        for k in sorted(
            constants
        ):

            if k <= 0:
                continue

            register(
                f"{prefix}/{k:g}",
                lambda x, f=shifted, k=k:
                    f(
                        x
                    )
                    / k,
            )

            register(
                f"clip({prefix},0,{k:g})/{k:g}",
                lambda x, f=shifted, k=k:
                    np.clip(
                        f(
                            x
                        ),
                        0.0,
                        k,
                    )
                    / k,
            )

            register(
                f"log1p({prefix})/log1p({k:g})",
                lambda x, f=shifted, k=k:
                    np.log1p(
                        f(
                            x
                        )
                    )
                    / np.log1p(
                        k
                    ),
            )

            if k > 1:

                register(
                    f"log1p({prefix})/log({k:g})",
                    lambda x, f=shifted, k=k:
                        np.log1p(
                            f(
                                x
                            )
                        )
                        / np.log(
                            k
                        ),
                )

            register(
                f"log1p({prefix}/{k:g})",
                lambda x, f=shifted, k=k:
                    np.log1p(
                        f(
                            x
                        )
                        / k
                    ),
            )

    return result


def recover_context_transforms(
    repo: Path,
    feature_names: list[str],
) -> tuple[
    dict[
        str,
        Callable[
            [
                np.ndarray
            ],
            np.ndarray,
        ],
    ],
    dict[str, Any],
]:
    """
    Replay the exact frozen CKPT5 context formulas.

    Frozen CKPT5 prepare() contract:

        line_number_scaled
            = clip(line, 1, 10) / 10

        elapsed_on_line_scaled
            = clip(elapsed_on_line_days, 0, 1460) / 365

        scan_number_patient_log
            = log1p(scan_number_patient) / 5

        scan_number_line_log
            = log1p(scan_number_line) / 5

        genomic_available
            = genomic_available

        genomic_age_scaled
            = clip(genomic_age_days, 0, 3650) / 365

    No transform fitting, parameter search, or external-data tuning is
    permitted here.
    """

    expected = [
        "line_number_scaled",
        "elapsed_on_line_scaled",
        "scan_number_patient_log",
        "scan_number_line_log",
        "genomic_available",
        "genomic_age_scaled",
    ]

    if feature_names != expected:

        raise RuntimeError(
            "Unexpected frozen CKPT5 context feature ordering: "
            f"{feature_names}"
        )

    (
        _,
        target,
        raw,
    ) = chord_raw_context_variables(
        repo
    )

    def line_number_scaled(
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        return (
            np.clip(
                values,
                1.0,
                10.0,
            )
            / 10.0
        ).astype(
            np.float32
        )

    def elapsed_on_line_scaled(
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        return (
            np.clip(
                values,
                0.0,
                1460.0,
            )
            / 365.0
        ).astype(
            np.float32
        )

    def scan_number_patient_log(
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        return (
            np.log1p(
                values
            )
            / 5.0
        ).astype(
            np.float32
        )

    def scan_number_line_log(
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        return (
            np.log1p(
                values
            )
            / 5.0
        ).astype(
            np.float32
        )

    def genomic_available(
        values,
    ):

        return np.asarray(
            values,
            dtype=np.float32,
        )

    def genomic_age_scaled(
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float32,
        )

        return (
            np.clip(
                values,
                0.0,
                3650.0,
            )
            / 365.0
        ).astype(
            np.float32
        )

    selected = {
        "line_number_scaled":
            line_number_scaled,

        "elapsed_on_line_scaled":
            elapsed_on_line_scaled,

        "scan_number_patient_log":
            scan_number_patient_log,

        "scan_number_line_log":
            scan_number_line_log,

        "genomic_available":
            genomic_available,

        "genomic_age_scaled":
            genomic_age_scaled,
    }

    formulas = {
        "line_number_scaled":
            "clip(line,1,10)/10",

        "elapsed_on_line_scaled":
            "clip(elapsed_on_line_days,0,1460)/365",

        "scan_number_patient_log":
            "log1p(scan_number_patient)/5",

        "scan_number_line_log":
            "log1p(scan_number_line)/5",

        "genomic_available":
            "identity",

        "genomic_age_scaled":
            "clip(genomic_age_days,0,3650)/365",
    }

    replay_columns = []

    per_feature = {}

    for column_index, feature in enumerate(
        expected
    ):

        predicted = selected[
            feature
        ](
            raw[
                feature
            ]
        )

        observed = target[
            :,
            column_index
        ].astype(
            np.float32
        )

        if predicted.shape != observed.shape:

            raise RuntimeError(
                f"{feature}: positive-control shape mismatch."
            )

        absolute = np.abs(
            predicted
            - observed
        )

        per_feature[
            feature
        ] = {
            "formula":
                formulas[
                    feature
                ],

            "mean_absolute_error":
                float(
                    absolute.mean()
                ),

            "max_absolute_error":
                float(
                    absolute.max()
                ),

            "exact_fraction":
                float(
                    (
                        absolute
                        <= 1e-7
                    ).mean()
                ),
        }

        replay_columns.append(
            predicted
        )

    replay = np.column_stack(
        replay_columns
    ).astype(
        np.float32
    )

    if replay.shape != target.shape:

        raise RuntimeError(
            "Full CKPT5 context positive-control shape mismatch."
        )

    absolute = np.abs(
        replay
        - target.astype(
            np.float32
        )
    )

    maximum = float(
        absolute.max()
    )

    mean = float(
        absolute.mean()
    )

    exact_fraction = float(
        (
            absolute
            <= 1e-7
        ).mean()
    )

    if maximum > 1e-6:

        raise RuntimeError(
            "Exact frozen CKPT5 context replay failed: "
            f"max_abs_error={maximum}, "
            f"mean_abs_error={mean}, "
            f"per_feature={per_feature}"
        )

    audit = {
        "status":
            "PASS_EXACT_CKPT5_CONTEXT_REPLAY",

        "source":
            (
                "scripts/dynamic_scan/"
                "ckpt5_supervised_dynamic_model.py::prepare"
            ),

        "transform_selection":
            "SOURCE_EXACT_NO_SEARCH",

        "formulas":
            formulas,

        "per_feature":
            per_feature,

        "full_replay_mean_abs_error":
            mean,

        "full_replay_max_abs_error":
            maximum,

        "full_replay_exact_fraction":
            exact_fraction,
    }

    return (
        selected,
        audit,
    )


###############################################################################
# External context tensor
###############################################################################


def build_external_context(
    primary: pd.DataFrame,
    assignments: pd.DataFrame,
    transforms,
    feature_names,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
]:

    assignment = assignments.copy()

    if assignment[
        "external_landmark_row"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate genomic assignment landmark rows."
        )

    frame = primary.merge(
        assignment[
            [
                "external_landmark_row",
                "genomic_available",
                "genomic_age_days",
                "selected_sample_id",
                "selected_availability_day",
            ]
        ],
        on="external_landmark_row",
        how="left",
        validate="one_to_one",
    )

    if frame[
        "genomic_available"
    ].isna().any():

        raise RuntimeError(
            "Missing genomic assignment for primary landmark."
        )

    raw = {
        "line_number_scaled":
            frame[
                "treatment_line"
            ].to_numpy(
                dtype=float
            ),

        "elapsed_on_line_scaled":
            (
                frame[
                    "landmark_day"
                ]
                - frame[
                    "line_start_day"
                ]
            ).to_numpy(
                dtype=float
            ),

        "scan_number_patient_log":
            frame[
                "scan_number_patient"
            ].to_numpy(
                dtype=float
            ),

        "scan_number_line_log":
            frame[
                "scan_number_line"
            ].to_numpy(
                dtype=float
            ),

        "genomic_available":
            frame[
                "genomic_available"
            ].astype(float)
            .to_numpy(),

        "genomic_age_scaled":
            frame[
                "genomic_age_days"
            ]
            .fillna(
                0.0
            )
            .to_numpy(
                dtype=float
            ),
    }

    context = np.column_stack(
        [
            transforms[
                feature
            ](
                raw[
                    feature
                ]
            )
            for feature
            in feature_names
        ]
    ).astype(
        np.float32
    )

    if context.shape != (
        len(
            frame
        ),
        6,
    ):

        raise RuntimeError(
            "External context dimension mismatch."
        )

    if not np.isfinite(
        context
    ).all():

        raise RuntimeError(
            "Nonfinite external context tensor."
        )

    return (
        context,
        frame,
    )


###############################################################################
# Frozen CKPT6B model loading and direct inference
###############################################################################


def instantiate_frozen_model(
    ckpt5,
    checkpoint,
    device,
):

    state = checkpoint[
        "model_state"
    ]

    dimension_values = {
        "temporal_dim":
            int(
                checkpoint[
                    "temporal_dim"
                ]
            ),

        "genomic_dim":
            int(
                checkpoint[
                    "genomic_dim"
                ]
            ),

        "tumor_dim":
            int(
                checkpoint[
                    "genomic_dim"
                ]
            ),

        "scan_dim":
            int(
                checkpoint[
                    "scan_dim"
                ]
            ),

        "context_dim":
            int(
                checkpoint[
                    "context_dim"
                ]
            ),

        "months":
            int(
                checkpoint[
                    "months"
                ]
            ),
    }

    candidates = []

    for name in dir(
        ckpt5
    ):

        obj = getattr(
            ckpt5,
            name
        )

        if not inspect.isclass(
            obj
        ):
            continue

        try:

            source = inspect.getsource(
                obj
            )

        except Exception:

            continue

        if (
            "survival_head"
            not in source
            or "next_scan_head"
            not in source
        ):

            continue

        candidates.append(
            (
                name,
                obj,
            )
        )

    errors = []

    for name, cls in candidates:

        try:

            signature = inspect.signature(
                cls
            )

            kwargs = {}

            unsupported = False

            for parameter_name, parameter in signature.parameters.items():

                if parameter_name in dimension_values:

                    kwargs[
                        parameter_name
                    ] = dimension_values[
                        parameter_name
                    ]

                elif (
                    parameter.default
                    is inspect.Parameter.empty
                ):

                    unsupported = True
                    break

            if unsupported:
                continue

            model = cls(
                **kwargs
            ).to(
                device
            )

            model.load_state_dict(
                state,
                strict=True,
            )

            model.eval()

            return (
                model,
                name,
                kwargs,
            )

        except Exception as exc:

            errors.append(
                f"{name}: "
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Unable to instantiate frozen supervised model. "
        + " | ".join(
            errors[
                :20
            ]
        )
    )


def forward_model(
    model,
    temporal,
    genomic,
    scan,
    context,
):

    signature = inspect.signature(
        model.forward
    )

    required = [
        parameter
        for parameter
        in signature.parameters.values()
        if parameter.name
        != "self"
    ]

    ###########################################################################
    # First try name-driven direct kwargs.
    ###########################################################################

    kwargs = {}

    mapping_ok = True

    for parameter in required:

        name = norm(
            parameter.name
        )

        value = None

        if "temporal" in name:

            value = temporal

        elif (
            "genomic"
            in name
            or "tumor"
            in name
        ):

            value = genomic

        elif "scan" in name:

            value = scan

        elif "context" in name:

            value = context

        elif (
            parameter.default
            is not inspect.Parameter.empty
        ):

            continue

        else:

            mapping_ok = False
            break

        kwargs[
            parameter.name
        ] = value

    if mapping_ok:

        try:

            return model(
                **kwargs
            )

        except TypeError:

            pass

    ###########################################################################
    # Standard positional interface.
    ###########################################################################

    try:

        return model(
            temporal,
            genomic,
            scan,
            context,
        )

    except TypeError:

        pass

    ###########################################################################
    # Single-batch-dict interface.
    ###########################################################################

    batch = {
        "temporal":
            temporal,

        "temporal_state":
            temporal,

        "genomic":
            genomic,

        "genomic_embedding":
            genomic,

        "tumor":
            genomic,

        "tumor_embedding":
            genomic,

        "scan":
            scan,

        "scan_features":
            scan,

        "context":
            context,

        "context_features":
            context,
    }

    return model(
        batch
    )


def extract_survival_logits(
    output,
    batch_size,
):

    preferred = []

    fallback = []

    def visit(
        value,
        path,
    ):

        if torch.is_tensor(
            value
        ):

            if (
                value.shape[
                    0
                ]
                != batch_size
            ):

                return

            per_row = int(
                value.numel()
                / batch_size
            )

            if per_row != 96:
                return

            item = (
                path,
                value,
            )

            if "survival" in path.lower():

                preferred.append(
                    item
                )

            else:

                fallback.append(
                    item
                )

            return

        if isinstance(
            value,
            dict,
        ):

            for key, child in value.items():

                visit(
                    child,
                    (
                        path
                        + "."
                        + str(
                            key
                        )
                    ),
                )

        elif isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):

            for index, child in enumerate(
                value
            ):

                visit(
                    child,
                    (
                        path
                        + f"[{index}]"
                    ),
                )

    visit(
        output,
        "output",
    )

    candidates = (
        preferred
        if preferred
        else fallback
    )

    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            "Unable to uniquely identify 96-logit survival output: "
            f"{[(path, list(t.shape)) for path, t in candidates]}"
        )

    path, tensor = candidates[
        0
    ]

    return (
        tensor.reshape(
            batch_size,
            24,
            4,
        ),
        path,
    )


@torch.no_grad()
def infer_logits(
    model,
    temporal: np.ndarray,
    genomic: np.ndarray,
    scan: np.ndarray,
    context: np.ndarray,
    device,
    *,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
]:

    pre_output = np.zeros(
        (
            len(
                temporal
            ),
            24,
            4,
        ),
        dtype=np.float32,
    )

    post_output = np.zeros_like(
        pre_output
    )

    output_path = None

    for start in range(
        0,
        len(
            temporal
        ),
        batch_size,
    ):

        stop = min(
            start
            + batch_size,
            len(
                temporal
            ),
        )

        temporal_pre = torch.from_numpy(
            temporal[
                start:
                stop,
                0,
            ].astype(
                np.float32
            )
        ).to(
            device
        )

        temporal_post = torch.from_numpy(
            temporal[
                start:
                stop,
                1,
            ].astype(
                np.float32
            )
        ).to(
            device
        )

        genomic_tensor = torch.from_numpy(
            genomic[
                start:
                stop
            ].astype(
                np.float32
            )
        ).to(
            device
        )

        scan_tensor = torch.from_numpy(
            scan[
                start:
                stop
            ].astype(
                np.float32
            )
        ).to(
            device
        )

        zero_scan = torch.zeros_like(
            scan_tensor
        )

        context_tensor = torch.from_numpy(
            context[
                start:
                stop
            ].astype(
                np.float32
            )
        ).to(
            device
        )

        pre_raw = forward_model(
            model,
            temporal_pre,
            genomic_tensor,
            zero_scan,
            context_tensor,
        )

        post_raw = forward_model(
            model,
            temporal_post,
            genomic_tensor,
            scan_tensor,
            context_tensor,
        )

        pre_logits, pre_path = (
            extract_survival_logits(
                pre_raw,
                stop
                - start,
            )
        )

        post_logits, post_path = (
            extract_survival_logits(
                post_raw,
                stop
                - start,
            )
        )

        if pre_path != post_path:

            raise RuntimeError(
                "PRE/POST survival-output paths differ."
            )

        output_path = pre_path

        pre_output[
            start:
            stop
        ] = (
            pre_logits.float()
            .cpu()
            .numpy()
        )

        post_output[
            start:
            stop
        ] = (
            post_logits.float()
            .cpu()
            .numpy()
        )

    return (
        pre_output,
        post_output,
        str(
            output_path
        ),
    )


###############################################################################
# Variant materialization
###############################################################################


def materialize_variant(
    repo: Path,
    out: Path,
    ckpt4,
    model,
    model_device,
    reader: PredictorReader,
    source_map,
    primary: pd.DataFrame,
    scan_features_primary: np.ndarray,
    scan_table: pd.DataFrame,
    external_patients: set[str],
    line_lookup,
    context_transforms,
    context_feature_names,
    variant: str,
    *,
    batch_size: int,
) -> dict[str, Any]:

    variant_out = (
        out
        / variant
    )

    variant_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if variant == "timeline_sequencing":

        assignment_path = (
            repo
            / "artifacts/checkpoint7a4a/"
            "genomic_assignments_timeline.parquet"
        )

        tumor_path = (
            repo
            / "artifacts/checkpoint7a4a/"
            "tumor_embeddings_timeline_f16.npy"
        )

    elif variant == "clinical_sample":

        assignment_path = (
            repo
            / "artifacts/checkpoint7a4a/"
            "genomic_assignments_clinical.parquet"
        )

        tumor_path = (
            repo
            / "artifacts/checkpoint7a4a/"
            "tumor_embeddings_clinical_f16.npy"
        )

    else:

        raise ValueError(
            variant
        )

    assignments = pd.read_parquet(
        assignment_path
    )

    full_tumor = np.load(
        tumor_path,
        mmap_mode="r",
    )

    if full_tumor.shape != (
        len(
            assignments
        ),
        128,
    ):

        raise RuntimeError(
            f"{variant}: A4A tumor tensor mismatch."
        )

    if not np.array_equal(
        assignments[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        ),
        np.arange(
            len(
                assignments
            ),
            dtype=int,
        ),
    ):

        raise RuntimeError(
            f"{variant}: genomic assignment rows not canonical."
        )

    tumor = full_tumor[
        primary[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        )
    ].astype(
        np.float16
    )

    context, context_index = (
        build_external_context(
            primary,
            assignments,
            context_transforms,
            context_feature_names,
        )
    )

    canonical, event_audit = (
        build_canonical_events(
            reader,
            external_patients,
            line_lookup,
            source_map,
            variant,
        )
    )

    temporal_out = (
        variant_out
        / "temporal"
    )

    (
        temporal_index,
        temporal_all,
        temporal_qc,
    ) = build_temporal_cache(
        repo,
        temporal_out,
        ckpt4,
        canonical,
        scan_table,
        external_patients,
        batch_size=batch_size,
    )

    if temporal_index[
        [
            "patient_id",
            "scan_episode_id",
        ]
    ].duplicated().any():

        raise RuntimeError(
            f"{variant}: duplicate temporal cache keys."
        )

    temporal_index = (
        temporal_index
        .reset_index(
            drop=True
        )
    )

    temporal_index[
        "_temporal_row"
    ] = np.arange(
        len(
            temporal_index
        ),
        dtype=int,
    )

    joined = primary[
        [
            "external_landmark_row",
            "patient_id",
            "site",
            "bpc_episode_id",
            "landmark_day",
            "treatment_line",
            "line_start_day",
            "scan_number_patient",
            "scan_number_line",
        ]
    ].merge(
        temporal_index[
            [
                "patient_id",
                "scan_episode_id",
                "_temporal_row",
            ]
        ],
        left_on=[
            "patient_id",
            "bpc_episode_id",
        ],
        right_on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if joined[
        "_temporal_row"
    ].isna().any():

        missing = joined[
            joined[
                "_temporal_row"
            ].isna()
        ][
            [
                "patient_id",
                "bpc_episode_id",
            ]
        ].head(
            20
        )

        raise RuntimeError(
            f"{variant}: missing temporal states: "
            f"{missing.to_dict('records')}"
        )

    temporal = temporal_all[
        joined[
            "_temporal_row"
        ].to_numpy(
            dtype=int
        )
    ].astype(
        np.float16
    )

    if temporal.shape != (
        len(
            primary
        ),
        2,
        192,
    ):

        raise RuntimeError(
            f"{variant}: temporal tensor mismatch."
        )

    if scan_features_primary.shape != (
        len(
            primary
        ),
        18,
    ):

        raise RuntimeError(
            "Primary scan-feature tensor mismatch."
        )

    ###########################################################################
    # Save exact frozen inputs before inference.
    ###########################################################################

    index_out = context_index.copy()

    index_out[
        "variant_row"
    ] = np.arange(
        len(
            index_out
        ),
        dtype=int,
    )

    atomic_parquet(
        variant_out
        / "prediction_index.parquet",
        index_out,
    )

    atomic_npy(
        variant_out
        / "temporal_prepost_f16.npy",
        temporal,
    )

    atomic_npy(
        variant_out
        / "tumor_embeddings_f16.npy",
        tumor,
    )

    atomic_npy(
        variant_out
        / "current_scan_features_f32.npy",
        scan_features_primary.astype(
            np.float32
        ),
    )

    atomic_npy(
        variant_out
        / "context_features_f32.npy",
        context.astype(
            np.float32
        ),
    )

    ###########################################################################
    # Frozen supervised model.
    ###########################################################################

    (
        pre_logits,
        full_post_logits,
        survival_output_path,
    ) = infer_logits(
        model,
        temporal,
        tumor,
        scan_features_primary,
        context,
        model_device,
        batch_size=batch_size,
    )

    bounded_logits = (
        pre_logits
        + ALPHA
        * (
            full_post_logits
            - pre_logits
        )
    ).astype(
        np.float32
    )

    for name, array in (
        (
            "pre_survival_logits_f32.npy",
            pre_logits,
        ),
        (
            "full_post_survival_logits_f32.npy",
            full_post_logits,
        ),
        (
            "bounded_post_survival_logits_f32.npy",
            bounded_logits,
        ),
    ):

        if (
            array.shape
            != (
                len(
                    primary
                ),
                24,
                4,
            )
            or not np.isfinite(
                array
            ).all()
        ):

            raise RuntimeError(
                f"{variant}: invalid prediction tensor {name}"
            )

        atomic_npy(
            variant_out
            / name,
            array,
        )

    ###########################################################################
    # Separate site freezes.
    ###########################################################################

    site_summary = {}

    for site in EXTERNAL_SITES:

        mask = (
            index_out[
                "site"
            ]
            == site
        ).to_numpy()

        site_dir = (
            variant_out
            / site.lower()
        )

        site_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        site_index = (
            index_out.loc[
                mask
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        site_index[
            "site_row"
        ] = np.arange(
            len(
                site_index
            ),
            dtype=int,
        )

        atomic_parquet(
            site_dir
            / "prediction_index.parquet",
            site_index,
        )

        for name, array in (
            (
                "pre_survival_logits_f32.npy",
                pre_logits,
            ),
            (
                "full_post_survival_logits_f32.npy",
                full_post_logits,
            ),
            (
                "bounded_post_survival_logits_f32.npy",
                bounded_logits,
            ),
        ):

            atomic_npy(
                site_dir
                / name,
                array[
                    mask
                ],
            )

        site_summary[
            site
        ] = {
            "rows":
                int(
                    mask.sum()
                ),

            "patients":
                int(
                    site_index[
                        "patient_id"
                    ].nunique()
                ),
        }

    ###########################################################################
    # Hash freeze.
    ###########################################################################

    frozen_files = [
        variant_out
        / "prediction_index.parquet",

        variant_out
        / "temporal_prepost_f16.npy",

        variant_out
        / "tumor_embeddings_f16.npy",

        variant_out
        / "current_scan_features_f32.npy",

        variant_out
        / "context_features_f32.npy",

        variant_out
        / "pre_survival_logits_f32.npy",

        variant_out
        / "full_post_survival_logits_f32.npy",

        variant_out
        / "bounded_post_survival_logits_f32.npy",
    ]

    hashes = {
        str(
            path.relative_to(
                repo
            )
        ):
            sha256_file(
                path
            )
        for path
        in frozen_files
    }

    mean_abs_raw_update = float(
        np.mean(
            np.abs(
                full_post_logits
                - pre_logits
            )
        )
    )

    mean_abs_bounded_update = float(
        np.mean(
            np.abs(
                bounded_logits
                - pre_logits
            )
        )
    )

    result = {
        "variant":
            variant,

        "rows":
            int(
                len(
                    primary
                )
            ),

        "patients":
            int(
                primary[
                    "patient_id"
                ].nunique()
            ),

        "site":
            site_summary,

        "event_audit":
            event_audit,

        "temporal_qc":
            temporal_qc,

        "input_shapes": {
            "temporal":
                list(
                    temporal.shape
                ),

            "genomic":
                list(
                    tumor.shape
                ),

            "scan":
                list(
                    scan_features_primary.shape
                ),

            "context":
                list(
                    context.shape
                ),
        },

        "prediction_shape":
            list(
                bounded_logits.shape
            ),

        "survival_output_path":
            survival_output_path,

        "mean_abs_full_post_minus_pre_logit":
            mean_abs_raw_update,

        "mean_abs_bounded_minus_pre_logit":
            mean_abs_bounded_update,

        "hashes":
            hashes,
    }

    atomic_json(
        variant_out
        / "variant_manifest.json",
        result,
    )

    return result


###############################################################################
# Main
###############################################################################


def main():

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
        default="artifacts/checkpoint7a4b",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    bpc_root = (
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
    # Frozen gates.
    ###########################################################################

    a4a = json.loads(
        (
            repo
            / "artifacts/checkpoint7a4a/"
            "interface_report.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        a4a[
            "status"
        ]
        != "READY_FOR_CKPT7A4B_EXTERNAL_TENSOR_BUILD"
    ):

        raise RuntimeError(
            "A4A not ready."
        )

    if (
        a4a[
            "external_outcomes_opened"
        ]
        or a4a[
            "external_outcome_distributions_inspected"
        ]
    ):

        raise RuntimeError(
            "Outcome-blind gate violated upstream."
        )

    context_feature_names = (
        a4a[
            "frozen_feature_schema"
        ][
            "context_feature_names"
        ]
    )

    scan_feature_names = (
        a4a[
            "frozen_feature_schema"
        ][
            "scan_feature_names"
        ]
    )

    if len(
        scan_feature_names
    ) != 18:

        raise RuntimeError(
            "Frozen scan schema changed."
        )

    if len(
        context_feature_names
    ) != 6:

        raise RuntimeError(
            "Frozen context schema changed."
        )

    ###########################################################################
    # Modules.
    ###########################################################################

    ckpt4 = import_module(
        "ckpt4_external_a4b",
        repo
        / "scripts/dynamic_scan/"
        "ckpt4_temporal_pretrain.py",
    )

    ckpt5 = import_module(
        "ckpt5_external_a4b",
        repo
        / "scripts/dynamic_scan/"
        "ckpt5_supervised_dynamic_model.py",
    )

    ###########################################################################
    # Predictor-only source access.
    ###########################################################################

    reader = PredictorReader(
        bpc_root
    )

    treatment = reader.read(
        "treatment"
    )

    imaging = reader.read(
        "imaging"
    )

    a4a_landmarks = pd.read_parquet(
        repo
        / "artifacts/checkpoint7a4a/"
        "external_primary_landmarks.parquet"
    )

    external_patients = set(
        a4a_landmarks[
            "patient_id"
        ].astype(str)
    )

    (
        line_intervals,
        line_lookup,
        line_audit,
    ) = build_line_intervals(
        treatment,
        imaging,
        external_patients,
    )

    atomic_parquet(
        out
        / "predictor_treatment_lines.parquet",
        line_intervals,
    )

    ###########################################################################
    # Exact scan surface + active-line primary population.
    ###########################################################################

    (
        episodes,
        episode_features,
        scan_table,
        line_scan_audit,
    ) = prepare_external_scans(
        repo,
        line_lookup,
        scan_feature_names,
    )

    a4a_key = (
        a4a_landmarks[
            [
                "external_landmark_row",
                "patient_id",
                "site",
                "bpc_episode_id",
            ]
        ]
        .copy()
    )

    primary = episodes[
        (
            episodes[
                "scan_state"
            ]
            == "NON_PROGRESSIVE"
        )
        & episodes[
            "active_line"
        ]
    ].copy()

    primary = primary.merge(
        a4a_key,
        on=[
            "patient_id",
            "site",
            "bpc_episode_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if primary[
        "external_landmark_row"
    ].isna().any():

        raise RuntimeError(
            "Active-line NP scan missing from A4A candidate set."
        )

    primary[
        "external_landmark_row"
    ] = primary[
        "external_landmark_row"
    ].astype(
        int
    )

    primary = (
        primary.sort_values(
            "external_landmark_row"
        )
        .reset_index(
            drop=True
        )
    )

    primary[
        "prediction_row"
    ] = np.arange(
        len(
            primary
        ),
        dtype=int,
    )

    scan_features_primary = np.load(
        repo
        / "artifacts/checkpoint7a3/"
        "bpc_w3_scan_features_f32.npy",
        mmap_mode="r",
    )[
        primary[
            "_feature_row"
        ].to_numpy(
            dtype=int
        )
    ].astype(
        np.float32
    )

    final_site_counts = {
        site: {
            "rows":
                int(
                    (
                        primary[
                            "site"
                        ]
                        == site
                    ).sum()
                ),

            "patients":
                int(
                    primary.loc[
                        primary[
                            "site"
                        ]
                        == site,
                        "patient_id",
                    ].nunique()
                ),
        }
        for site in EXTERNAL_SITES
    }

    if (
        final_site_counts[
            "DFCI"
        ][
            "rows"
        ]
        < 100
        or final_site_counts[
            "VICC"
        ][
            "rows"
        ]
        < 50
    ):

        raise RuntimeError(
            "Predictor-side active-line filtering leaves "
            "insufficient external landmark support: "
            f"{final_site_counts}"
        )

    atomic_parquet(
        out
        / "frozen_external_prediction_landmarks.parquet",
        primary,
    )

    atomic_parquet(
        out
        / "external_w3_scan_table_ckpt4.parquet",
        scan_table,
    )

    ###########################################################################
    # Recover context scaling exactly; no guessed constants accepted.
    ###########################################################################

    (
        context_transforms,
        context_replay,
    ) = recover_context_transforms(
        repo,
        context_feature_names,
    )

    atomic_json(
        out
        / "context_replay.json",
        context_replay,
    )

    ###########################################################################
    # Load frozen model.
    ###########################################################################

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required for CKPT7A4B."
        )

    device = torch.device(
        "cuda:0"
    )

    candidate = torch.load(
        repo
        / "artifacts/checkpoint6b/"
        "bounded_dynamic_scan_candidate.pt",
        map_location=device,
        weights_only=False,
    )

    if (
        float(
            candidate[
                "ckpt6b_bounded_update"
            ][
                "alpha"
            ]
        )
        != ALPHA
    ):

        raise RuntimeError(
            "Frozen alpha changed."
        )

    for key, expected in (
        (
            "scan_dim",
            18,
        ),
        (
            "context_dim",
            6,
        ),
        (
            "temporal_dim",
            192,
        ),
        (
            "genomic_dim",
            128,
        ),
        (
            "months",
            24,
        ),
    ):

        if int(
            candidate[
                key
            ]
        ) != expected:

            raise RuntimeError(
                f"Candidate dimension drift: {key}"
            )

    (
        model,
        model_class,
        model_init_kwargs,
    ) = instantiate_frozen_model(
        ckpt5,
        candidate,
        device,
    )

    ###########################################################################
    # Source bucket mapping from the training-side canonical stream.
    ###########################################################################

    source_map = chord_source_map(
        repo
    )

    ###########################################################################
    # PRIMARY: BPC timeline-sequencing START_DATE.
    ###########################################################################

    print(
        "[CKPT7A4B] Building PRIMARY timeline-sequencing variant.",
        flush=True,
    )

    primary_result = materialize_variant(
        repo,
        out,
        ckpt4,
        model,
        device,
        reader,
        source_map,
        primary,
        scan_features_primary,
        scan_table,
        external_patients,
        line_lookup,
        context_transforms,
        context_feature_names,
        PRIMARY_VARIANT,
        batch_size=args.batch_size,
    )

    ###########################################################################
    # PREDECLARED SENSITIVITY: clinical-sample CPT_SEQ_DATE.
    ###########################################################################

    print(
        "[CKPT7A4B] Building sequencing-time sensitivity variant.",
        flush=True,
    )

    sensitivity_result = materialize_variant(
        repo,
        out,
        ckpt4,
        model,
        device,
        reader,
        source_map,
        primary,
        scan_features_primary,
        scan_table,
        external_patients,
        line_lookup,
        context_transforms,
        context_feature_names,
        SENSITIVITY_VARIANT,
        batch_size=args.batch_size,
    )

    ###########################################################################
    # Outcome quarantine check.
    ###########################################################################

    access_audit = reader.audit()

    if access_audit[
        "violations"
    ]:

        raise RuntimeError(
            f"Outcome quarantine violation: "
            f"{access_audit['violations']}"
        )

    ###########################################################################
    # Freeze adapter semantics.
    ###########################################################################

    adapter = {
        "status":
            "PASS_OUTCOME_BLINDED_EXTERNAL_PREDICTION_FREEZE",

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "primary_external_sites":
            [
                "DFCI",
                "VICC",
            ],

        "scan_episode_rule":
            "W3",

        "landmark_day":
            "episode_end_day",

        "primary_scan_state":
            "NON_PROGRESSIVE",

        "predictor_eligibility":
            (
                "NON_PROGRESSIVE W3 scan associated with a predictor-only "
                "CKPT1-style treatment line"
            ),

        "treatment_line_adapter": {
            "algorithm":
                "FROZEN_CKPT1_LINE_BOUNDARY_REPLAY",

            "day_split":
                365,

            "progression_grace_days":
                28,

            "treatment_boundary":
                (
                    "next treatment at/after line_start+365, "
                    "requiring a new AGENT when AGENT is available"
                ),

            "progression_boundary":
                (
                    "observed predictor-side progression >=28 days "
                    "after line start; next line may begin at the first "
                    "new treatment in the preceding 28-day window"
                ),

            "bpc_progression_mapping":
                {
                    "PROGRESSING/WORSENING/ENLARGING":
                        "Y",

                    "STABLE/NO CHANGE":
                        "N",

                    "IMPROVING/RESPONDING":
                        "N",

                    "MIXED":
                        "EXCLUDED_FROM_YN_STREAM",

                    "NOT STATED/INDETERMINATE":
                        "EXCLUDED_FROM_YN_STREAM",

                    "UNOBSERVED":
                        "EXCLUDED_FROM_YN_STREAM",
                },

            "regimen_number_used_for_line_identity":
                False,

            "external_outcomes_used":
                False,
        },

        "temporal_rule": {
            "pre":
                "all predictor events with day < landmark_day",

            "post":
                (
                    "PRE plus exactly current W3 scan; "
                    "unrelated same-day events excluded"
                ),

            "persistent_history_across_lines":
                True,

            "msk_clock_offsets_transferred":
                False,
        },

        "genomic_timing": {
            "primary": {
                "variant":
                    PRIMARY_VARIANT,

                "source":
                    (
                        "cBioPortal_files/"
                        "data_timeline_sequencing.txt::START_DATE"
                    ),

                "reason":
                    (
                        "native BPC timeline coordinate shared "
                        "with scan and treatment streams"
                    ),
            },

            "sensitivity": {
                "variant":
                    SENSITIVITY_VARIANT,

                "source":
                    (
                        "cBioPortal_files/"
                        "data_clinical_sample.txt::CPT_SEQ_DATE"
                    ),
            },

            "availability_rule":
                "availability_day < landmark_day",

            "same_day_excluded":
                True,

            "unavailable_tumor_state":
                "exact zero 128-D embedding",
        },

        "scan_features":
            scan_feature_names,

        "context_features":
            context_feature_names,

        "context_scaling":
            context_replay,

        "bounded_update": {
            "space":
                "survival logits",

            "formula":
                (
                    "PRE + 0.75 * "
                    "(FULL_POST - PRE)"
                ),

            "alpha":
                ALPHA,
        },

        "model": {
            "class":
                model_class,

            "init_kwargs":
                model_init_kwargs,

            "temporal_dim":
                192,

            "genomic_dim":
                128,

            "scan_dim":
                18,

            "context_dim":
                6,

            "months":
                24,

            "causes":
                4,
        },

        "primary_population":
            final_site_counts,

        "line_audit":
            line_audit,

        "scan_line_audit":
            line_scan_audit,

        "predictor_access":
            access_audit,
    }

    atomic_json(
        out
        / "frozen_adapter_spec.json",
        adapter,
    )

    ###########################################################################
    # Global immutable manifest.
    ###########################################################################

    freeze_paths = [
        out
        / "frozen_external_prediction_landmarks.parquet",

        out
        / "predictor_treatment_lines.parquet",

        out
        / "frozen_adapter_spec.json",

        out
        / "context_replay.json",

        out
        / PRIMARY_VARIANT
        / "prediction_index.parquet",

        out
        / PRIMARY_VARIANT
        / "temporal_prepost_f16.npy",

        out
        / PRIMARY_VARIANT
        / "tumor_embeddings_f16.npy",

        out
        / PRIMARY_VARIANT
        / "current_scan_features_f32.npy",

        out
        / PRIMARY_VARIANT
        / "context_features_f32.npy",

        out
        / PRIMARY_VARIANT
        / "pre_survival_logits_f32.npy",

        out
        / PRIMARY_VARIANT
        / "full_post_survival_logits_f32.npy",

        out
        / PRIMARY_VARIANT
        / "bounded_post_survival_logits_f32.npy",

        out
        / SENSITIVITY_VARIANT
        / "prediction_index.parquet",

        out
        / SENSITIVITY_VARIANT
        / "temporal_prepost_f16.npy",

        out
        / SENSITIVITY_VARIANT
        / "tumor_embeddings_f16.npy",

        out
        / SENSITIVITY_VARIANT
        / "current_scan_features_f32.npy",

        out
        / SENSITIVITY_VARIANT
        / "context_features_f32.npy",

        out
        / SENSITIVITY_VARIANT
        / "pre_survival_logits_f32.npy",

        out
        / SENSITIVITY_VARIANT
        / "full_post_survival_logits_f32.npy",

        out
        / SENSITIVITY_VARIANT
        / "bounded_post_survival_logits_f32.npy",
    ]

    for path in freeze_paths:

        if (
            not path.exists()
            or path.stat().st_size
            == 0
        ):

            raise RuntimeError(
                f"Missing freeze artifact: {path}"
            )

    upstream = {
        "ckpt4_temporal_encoder":
            sha256_file(
                repo
                / "artifacts/checkpoint4/"
                "temporal_encoder.pt"
            ),

        "ckpt6b_candidate":
            sha256_file(
                repo
                / "artifacts/checkpoint6b/"
                "bounded_dynamic_scan_candidate.pt"
            ),

        "ckpt6e_protocol":
            sha256_file(
                repo
                / "artifacts/checkpoint6e/"
                "external_validation_protocol.json"
            ),

        "ckpt6e_candidate_manifest":
            sha256_file(
                repo
                / "artifacts/checkpoint6e/"
                "frozen_candidate_manifest.json"
            ),

        "ckpt7a4a_interface_report":
            sha256_file(
                repo
                / "artifacts/checkpoint7a4a/"
                "interface_report.json"
            ),

        "ckpt7a3_scan_features":
            sha256_file(
                repo
                / "artifacts/checkpoint7a3/"
                "bpc_w3_scan_features_f32.npy"
            ),
    }

    manifest = {
        "status":
            "PASS_OUTCOME_BLINDED_EXTERNAL_PREDICTION_FREEZE",

        "external_outcomes_opened":
            False,

        "primary_variant":
            PRIMARY_VARIANT,

        "sensitivity_variant":
            SENSITIVITY_VARIANT,

        "rows":
            int(
                len(
                    primary
                )
            ),

        "patients":
            int(
                primary[
                    "patient_id"
                ].nunique()
            ),

        "site":
            final_site_counts,

        "upstream_hashes":
            upstream,

        "frozen_artifact_hashes": {
            str(
                path.relative_to(
                    repo
                )
            ):
                sha256_file(
                    path
                )
            for path
            in freeze_paths
        },

        "primary_result":
            primary_result,

        "sensitivity_result":
            sensitivity_result,

        "next_action":
            (
                "CKPT7B may now open the frozen DFCI/VICC "
                "outcome tables exactly once and evaluate the "
                "already-hashed primary predictions. "
                "No model/input/timing/alpha changes are permitted."
            ),
    }

    atomic_json(
        out
        / "prediction_freeze_manifest.json",
        manifest,
    )

    handoff = {
        "checkpoint":
            "07A4B",

        "status":
            manifest[
                "status"
            ],

        "external_outcomes_opened":
            False,

        "primary_variant":
            PRIMARY_VARIANT,

        "primary_population":
            final_site_counts,

        "prediction_freeze_manifest":
            (
                "artifacts/checkpoint7a4b/"
                "prediction_freeze_manifest.json"
            ),

        "next_action":
            manifest[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts/handoff/"
        "checkpoint_07A4B.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A4B SUMMARY =========="
    )

    print(
        "status=PASS_OUTCOME_BLINDED_EXTERNAL_PREDICTION_FREEZE"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "primary_genomic_timing="
        "data_timeline_sequencing.START_DATE"
    )

    print(
        "sensitivity_genomic_timing="
        "data_clinical_sample.CPT_SEQ_DATE"
    )

    print(
        "primary_population="
        f"{final_site_counts}"
    )

    print(
        "line_audit="
        f"{line_audit}"
    )

    print(
        "scan_line_audit="
        f"{line_scan_audit}"
    )

    print(
        "context_replay_status="
        f"{context_replay['status']}"
    )

    print(
        "context_replay_max_abs_error="
        f"{context_replay['full_replay_max_abs_error']}"
    )

    print(
        "model_class="
        f"{model_class}"
    )

    print(
        "primary_prediction_shape="
        f"{primary_result['prediction_shape']}"
    )

    print(
        "sensitivity_prediction_shape="
        f"{sensitivity_result['prediction_shape']}"
    )

    print(
        "primary_mean_abs_full_post_minus_pre_logit="
        f"{primary_result['mean_abs_full_post_minus_pre_logit']}"
    )

    print(
        "primary_mean_abs_bounded_minus_pre_logit="
        f"{primary_result['mean_abs_bounded_minus_pre_logit']}"
    )

    print(
        "prediction_freeze_manifest="
        "artifacts/checkpoint7a4b/prediction_freeze_manifest.json"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A4B.json"
    )

    print(
        "========== CKPT7A4B SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A4B DECISION PACKET =========="
    )

    print(
        "context_transforms="
    )

    print(
        json.dumps(
            context_replay,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print("")
    print(
        "primary_variant="
    )

    print(
        json.dumps(
            primary_result,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print("")
    print(
        "sensitivity_variant="
    )

    print(
        json.dumps(
            sensitivity_result,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )

    print("")
    print(
        "predictor_access="
        f"{access_audit}"
    )

    print("")
    print(
        "next_action="
        f"{manifest['next_action']}"
    )

    print(
        "========== CKPT7A4B DECISION PACKET END =========="
    )


if __name__ == "__main__":

    main()
