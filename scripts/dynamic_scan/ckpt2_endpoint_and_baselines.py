#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score


RNG_SEED = 20260920

# CKPT1B administratively censored at 730 days.
MAX_MONTHS = 24
MONTH_DAYS = 730.0 / MAX_MONTHS

HORIZONS = (
    3,
    6,
    12,
    18,
)

EPS = 1e-7


###############################################################################
# Basic helpers
###############################################################################


def norm_col(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        type(value).__name__
    )


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
            default=json_default,
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

    # Validate before replacement.
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

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"empty Parquet write: {path}"
        )

    check = pd.read_parquet(
        tmp
    )

    if len(check) != len(frame):
        raise RuntimeError(
            f"Parquet verification failed for {path}: "
            f"{len(check)} != {len(frame)}"
        )

    tmp.replace(path)


def find_col(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:

    lookup = {
        norm_col(column): column
        for column
        in frame.columns
    }

    for candidate in candidates:
        key = norm_col(candidate)

        if key in lookup:
            return lookup[key]

    if required:
        raise KeyError(
            "None of the requested columns were found.\n"
            f"requested={list(candidates)}\n"
            f"available={list(frame.columns)}"
        )

    return None


def coerce_numeric(
    series: pd.Series,
) -> pd.Series:

    text = (
        series
        .astype(str)
        .str.strip()
    )

    lower = text.str.lower()

    mapped = lower.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "y": 1.0,
            "n": 0.0,
            "present": 1.0,
            "absent": 0.0,
        }
    )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.where(
        numeric.notna(),
        mapped,
    )


###############################################################################
# Source loading / normalization
###############################################################################


def load_splits(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        path
    )

    pid = find_col(
        frame,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    split = find_col(
        frame,
        [
            "split",
            "SPLIT",
        ],
    )

    out = frame[
        [
            pid,
            split,
        ]
    ].copy()

    out.columns = [
        "patient_id",
        "split",
    ]

    out[
        "patient_id"
    ] = (
        out[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    out[
        "split"
    ] = (
        out[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "training": "train",
                "validation": "val",
                "valid": "val",
                "dev": "val",
                "testing": "test",
            }
        )
    )

    out = (
        out
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    duplicated = (
        out[
            "patient_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicated.any():
        raise RuntimeError(
            "A patient occurs in multiple split rows:\n"
            + str(
                out.loc[
                    duplicated
                ].head(
                    30
                )
            )
        )

    unknown = (
        set(
            out[
                "split"
            ]
        )
        - {
            "train",
            "val",
            "test",
        }
    )

    if unknown:
        raise RuntimeError(
            f"Unknown split labels: {sorted(unknown)}"
        )

    return out


def load_frozen_pfs(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        path
    )

    pid = find_col(
        frame,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    line = find_col(
        frame,
        [
            "line",
            "LINE",
        ],
    )

    start = find_col(
        frame,
        [
            "line_start_day",
            "LINE_START",
            "line_start",
        ],
    )

    time = find_col(
        frame,
        [
            "pfs_time_days",
            "PFS_TIME_DAYS",
            "PFS_TIME",
        ],
    )

    event = find_col(
        frame,
        [
            "pfs_event",
            "PFS_EVENT",
        ],
    )

    source = find_col(
        frame,
        [
            "line_source",
            "LINE_SOURCE",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "patient_id":
                frame[
                    pid
                ]
                .astype(str)
                .str.strip(),

            "line":
                pd.to_numeric(
                    frame[
                        line
                    ],
                    errors="raise",
                )
                .astype(int),

            "line_start_day":
                pd.to_numeric(
                    frame[
                        start
                    ],
                    errors="raise",
                )
                .astype(float),

            "time_days":
                pd.to_numeric(
                    frame[
                        time
                    ],
                    errors="raise",
                )
                .astype(float),

            "event":
                pd.to_numeric(
                    frame[
                        event
                    ],
                    errors="raise",
                )
                .astype(int),
        }
    )

    if source is not None:
        out[
            "line_source"
        ] = (
            frame[
                source
            ]
            .astype(str)
        )
    else:
        out[
            "line_source"
        ] = ""

    if (
        out[
            [
                "patient_id",
                "line",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Frozen PFS table contains duplicate patient-line keys."
        )

    if not (
        out[
            "event"
        ]
        .isin(
            [
                0,
                1,
            ]
        )
        .all()
    ):
        raise RuntimeError(
            "Frozen PFS event field is not binary."
        )

    if (
        out[
            "time_days"
        ]
        < 0
    ).any():
        raise RuntimeError(
            "Frozen PFS has a negative time."
        )

    return out


def load_design_matrix(
    path: Path,
) -> tuple[
    pd.DataFrame,
    list[str],
]:

    frame = pd.read_parquet(
        path
    )

    pid = find_col(
        frame,
        [
            "PATIENT_ID",
            "patient_id",
        ],
    )

    line = find_col(
        frame,
        [
            "LINE",
            "line",
        ],
    )

    frame = frame.copy()

    frame[
        "patient_id"
    ] = (
        frame[
            pid
        ]
        .astype(str)
        .str.strip()
    )

    frame[
        "line"
    ] = (
        pd.to_numeric(
            frame[
                line
            ],
            errors="raise",
        )
        .astype(int)
    )

    exact_exclusions = {
        norm_col(pid),
        norm_col(line),
        "patient_id",
        "line",
        "line_start",
        "line_start_day",
        "pfs_event",
        "pfs_time",
        "pfs_time_days",
        "line_source",
        "event_day",
    }

    forbidden_tokens = (
        "pfs",
        "target",
        "outcome",
        "qc_",
        "_qc",
        "event_day",
        "survival_time",
        "censor",
        "label",
    )

    feature_columns: list[
        str
    ] = []

    for column in frame.columns:

        normalized = norm_col(
            column
        )

        if (
            normalized
            in exact_exclusions
        ):
            continue

        if any(
            token in normalized
            for token
            in forbidden_tokens
        ):
            continue

        if normalized in {
            "patient_id",
            "line",
        }:
            continue

        feature_columns.append(
            column
        )

    if not feature_columns:
        raise RuntimeError(
            "No usable line-start covariates were found."
        )

    return (
        frame[
            [
                "patient_id",
                "line",
            ]
            + feature_columns
        ].copy(),
        feature_columns,
    )


def load_scan_episodes(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        path
    )

    pid = find_col(
        frame,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    line = find_col(
        frame,
        [
            "treatment_line",
            "line",
            "LINE",
        ],
    )

    landmark = find_col(
        frame,
        [
            "landmark_day",
            "episode_end_day",
            "scan_day",
            "START_DATE",
        ],
    )

    episode_end = find_col(
        frame,
        [
            "episode_end_day",
            "landmark_day",
        ],
        required=False,
    )

    episode_id = find_col(
        frame,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    state = find_col(
        frame,
        [
            "progression_state_3",
            "scan_state",
            "state",
        ],
    )

    out = frame.copy()

    out[
        "patient_id"
    ] = (
        out[
            pid
        ]
        .astype(str)
        .str.strip()
    )

    out[
        "line"
    ] = pd.to_numeric(
        out[
            line
        ],
        errors="coerce",
    )

    out[
        "landmark_day"
    ] = pd.to_numeric(
        out[
            landmark
        ],
        errors="coerce",
    )

    if episode_end is not None:
        out[
            "episode_end_day_norm"
        ] = pd.to_numeric(
            out[
                episode_end
            ],
            errors="coerce",
        )
    else:
        out[
            "episode_end_day_norm"
        ] = out[
            "landmark_day"
        ]

    if episode_id is not None:
        out[
            "scan_episode_id_norm"
        ] = (
            out[
                episode_id
            ]
            .astype(str)
        )
    else:
        out[
            "scan_episode_id_norm"
        ] = (
            out[
                "patient_id"
            ]
            + "::"
            + out[
                "landmark_day"
            ]
            .astype(str)
        )

    out[
        "scan_state_norm"
    ] = (
        out[
            state
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = out[
        out[
            "line"
        ].notna()
        & out[
            "landmark_day"
        ].notna()
    ].copy()

    out[
        "line"
    ] = (
        out[
            "line"
        ]
        .astype(int)
    )

    return out


###############################################################################
# Scan evidence / history engineering
###############################################################################


def add_scan_derived_features(
    scans: pd.DataFrame,
) -> pd.DataFrame:

    out = scans.copy()

    state = (
        out[
            "scan_state_norm"
        ]
        .astype(str)
        .str.upper()
    )

    out[
        "cur_state_non_progressive"
    ] = (
        state
        == "NON_PROGRESSIVE"
    ).astype(float)

    out[
        "cur_state_indeterminate"
    ] = (
        state
        == "INDETERMINATE"
    ).astype(float)

    out[
        "cur_state_progressive"
    ] = (
        state
        == "PROGRESSIVE"
    ).astype(float)

    text_candidates: list[
        str
    ] = []

    useful_tokens = (
        "procedure",
        "modality",
        "scan_type",
        "source_specific",
        "source",
        "tumor_site",
        "tumor_sites",
        "has_cancer",
        "cancer_presence",
    )

    for column in out.columns:

        normalized = norm_col(
            column
        )

        if any(
            token in normalized
            for token
            in useful_tokens
        ):
            text_candidates.append(
                column
            )

    if text_candidates:

        combined = (
            out[
                text_candidates
            ]
            .fillna("")
            .astype(str)
            .agg(
                " | ".join,
                axis=1,
            )
            .str.upper()
        )

    else:

        combined = pd.Series(
            "",
            index=out.index,
        )

    modality_patterns = {
        "ct": r"\bCT\b|COMPUTED TOMOGRAPH",
        "pet": r"\bPET\b|PET-CT",
        "mr": r"\bMR\b|\bMRI\b|MAGNETIC",
        "bone_scan": r"BONE SCAN",
    }

    for name, pattern in modality_patterns.items():

        out[
            f"cur_modality_{name}"
        ] = (
            combined
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
            .astype(float)
        )

    site_patterns = {
        "bone": r"\bBONE\b|VERTEBR|RIB|SKELET",
        "liver": r"\bLIVER\b",
        "lung": r"\bLUNG\b|PULMON",
        "brain": r"\bBRAIN\b|\bCNS\b|CEREBR",
        "lymph": r"LYMPH",
        "pleura": r"PLEURA",
    }

    for name, pattern in site_patterns.items():

        out[
            f"cur_site_{name}"
        ] = (
            combined
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
            .astype(float)
        )

    return out


def choose_current_numeric_features(
    scans: pd.DataFrame,
) -> list[str]:

    allow_tokens = (
        "coverage",
        "chest",
        "abdomen",
        "pelvis",
        "head",
        "other",
        "tumor",
        "site",
        "cancer",
        "procedure",
        "modality",
        "scan_type",
        "cur_",
    )

    deny_tokens = (
        "target",
        "next_",
        "future",
        "pfs",
        "death",
        "switch",
        "censor",
        "outcome",
        "event_day",
        "time_to",
    )

    ignore = {
        "patient_id",
        "line",
        "landmark_day",
        "episode_end_day_norm",
        "scan_episode_id_norm",
        "scan_state_norm",
    }

    chosen: list[
        str
    ] = []

    for column in scans.columns:

        normalized = norm_col(
            column
        )

        if normalized in ignore:
            continue

        if any(
            token in normalized
            for token
            in deny_tokens
        ):
            continue

        if not any(
            token in normalized
            for token
            in allow_tokens
        ):
            continue

        numeric = coerce_numeric(
            scans[
                column
            ]
        )

        nonblank = (
            scans[
                column
            ].notna()
            & (
                scans[
                    column
                ]
                .astype(str)
                .str.strip()
                != ""
            )
        )

        denominator = max(
            int(
                nonblank.sum()
            ),
            1,
        )

        fraction = (
            numeric[
                nonblank
            ]
            .notna()
            .sum()
            / denominator
        )

        if fraction >= 0.80:
            chosen.append(
                column
            )

    # Always retain explicitly derived current-scan features.
    derived = [
        column
        for column
        in scans.columns
        if column.startswith(
            "cur_"
        )
    ]

    chosen = list(
        dict.fromkeys(
            chosen
            + derived
        )
    )

    return chosen[:64]


def build_scan_history(
    scans: pd.DataFrame,
    current_numeric: list[str],
) -> pd.DataFrame:

    work = (
        scans
        .sort_values(
            [
                "patient_id",
                "landmark_day",
                "scan_episode_id_norm",
            ],
            kind="mergesort",
        )
        .copy()
    )

    rows: list[
        dict[str, Any]
    ] = []

    for (
        patient_id,
        group,
    ) in work.groupby(
        "patient_id",
        sort=False,
    ):

        group = group.sort_values(
            [
                "landmark_day",
                "scan_episode_id_norm",
            ],
            kind="mergesort",
        )

        state_counts = Counter()

        same_line_counts = Counter()

        running_sum = {
            column: 0.0
            for column
            in current_numeric
        }

        running_n = {
            column: 0
            for column
            in current_numeric
        }

        last_value = {
            column: np.nan
            for column
            in current_numeric
        }

        previous_day = None

        previous_state = "NONE"

        current_line = None

        same_line_n = 0

        for _, row in group.iterrows():

            line = int(
                row[
                    "line"
                ]
            )

            if (
                current_line
                != line
            ):

                current_line = line

                same_line_n = 0

                same_line_counts = Counter()

            record: dict[
                str,
                Any
            ] = {
                "patient_id":
                    patient_id,

                "line":
                    line,

                "landmark_day":
                    float(
                        row[
                            "landmark_day"
                        ]
                    ),

                "scan_episode_id_norm":
                    str(
                        row[
                            "scan_episode_id_norm"
                        ]
                    ),

                "hist_total_scans":
                    int(
                        sum(
                            state_counts.values()
                        )
                    ),

                "hist_same_line_scans":
                    int(
                        same_line_n
                    ),

                "hist_days_since_prev_scan":
                    (
                        np.nan
                        if previous_day is None
                        else (
                            float(
                                row[
                                    "landmark_day"
                                ]
                            )
                            - float(
                                previous_day
                            )
                        )
                    ),

                "hist_prev_non_progressive":
                    float(
                        previous_state
                        == "NON_PROGRESSIVE"
                    ),

                "hist_prev_indeterminate":
                    float(
                        previous_state
                        == "INDETERMINATE"
                    ),

                "hist_prev_progressive":
                    float(
                        previous_state
                        == "PROGRESSIVE"
                    ),

                "hist_n_non_progressive":
                    int(
                        state_counts[
                            "NON_PROGRESSIVE"
                        ]
                    ),

                "hist_n_indeterminate":
                    int(
                        state_counts[
                            "INDETERMINATE"
                        ]
                    ),

                "hist_n_progressive":
                    int(
                        state_counts[
                            "PROGRESSIVE"
                        ]
                    ),

                "hist_same_line_n_non_progressive":
                    int(
                        same_line_counts[
                            "NON_PROGRESSIVE"
                        ]
                    ),

                "hist_same_line_n_indeterminate":
                    int(
                        same_line_counts[
                            "INDETERMINATE"
                        ]
                    ),

                "hist_same_line_n_progressive":
                    int(
                        same_line_counts[
                            "PROGRESSIVE"
                        ]
                    ),
            }

            for column in current_numeric:

                key = norm_col(
                    column
                )

                record[
                    f"hist_last__{key}"
                ] = last_value[
                    column
                ]

                if (
                    running_n[
                        column
                    ]
                    > 0
                ):
                    record[
                        f"hist_mean__{key}"
                    ] = (
                        running_sum[
                            column
                        ]
                        / running_n[
                            column
                        ]
                    )
                else:
                    record[
                        f"hist_mean__{key}"
                    ] = np.nan

            rows.append(
                record
            )

            state = str(
                row[
                    "scan_state_norm"
                ]
            ).upper()

            state_counts[
                state
            ] += 1

            same_line_counts[
                state
            ] += 1

            same_line_n += 1

            previous_day = float(
                row[
                    "landmark_day"
                ]
            )

            previous_state = state

            for column in current_numeric:

                value = coerce_numeric(
                    pd.Series(
                        [
                            row[
                                column
                            ]
                        ]
                    )
                ).iat[
                    0
                ]

                if pd.notna(
                    value
                ):

                    value = float(
                        value
                    )

                    running_sum[
                        column
                    ] += value

                    running_n[
                        column
                    ] += 1

                    last_value[
                        column
                    ] = value

    history = pd.DataFrame(
        rows
    )

    if (
        history[
            [
                "patient_id",
                "scan_episode_id_norm",
            ]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate scan keys created while constructing scan history."
        )

    return history


###############################################################################
# Numeric feature encoder
###############################################################################


class NumericEncoder:

    def __init__(
        self,
        max_features: int,
        force_columns: Iterable[str] = (),
    ):

        self.max_features = int(
            max_features
        )

        self.force_columns = list(
            dict.fromkeys(
                force_columns
            )
        )

        self.columns: list[
            str
        ] = []

        self.median: dict[
            str,
            float
        ] = {}

        self.mean: dict[
            str,
            float
        ] = {}

        self.std: dict[
            str,
            float
        ] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        candidate_columns: list[str],
    ):

        scored: list[
            tuple[
                float,
                str,
            ]
        ] = []

        eligible: set[
            str
        ] = set()

        for column in candidate_columns:

            if column not in frame.columns:
                continue

            numeric = coerce_numeric(
                frame[
                    column
                ]
            )

            fraction = float(
                numeric
                .notna()
                .mean()
            )

            if fraction < 0.75:
                continue

            variance = (
                float(
                    numeric.var()
                )
                if (
                    numeric
                    .notna()
                    .sum()
                    > 1
                )
                else 0.0
            )

            if (
                not np.isfinite(
                    variance
                )
                or variance
                <= 1e-12
            ):
                continue

            eligible.add(
                column
            )

            scored.append(
                (
                    variance,
                    column,
                )
            )

        scored.sort(
            reverse=True
        )

        forced = [
            column
            for column
            in self.force_columns
            if column
            in eligible
        ]

        remainder = [
            column
            for _, column
            in scored
            if column
            not in forced
        ]

        capacity = max(
            self.max_features
            - len(
                forced
            ),
            0,
        )

        self.columns = (
            forced
            + remainder[
                :capacity
            ]
        )

        if not self.columns:
            raise RuntimeError(
                "NumericEncoder resolved zero usable features."
            )

        for column in self.columns:

            numeric = coerce_numeric(
                frame[
                    column
                ]
            )

            median = (
                float(
                    numeric.median()
                )
                if numeric.notna().any()
                else 0.0
            )

            filled = (
                numeric
                .fillna(
                    median
                )
                .to_numpy(
                    dtype=float
                )
            )

            mean = float(
                filled.mean()
            )

            std = float(
                filled.std()
            )

            if (
                not np.isfinite(
                    std
                )
                or std
                < 1e-8
            ):
                std = 1.0

            self.median[
                column
            ] = median

            self.mean[
                column
            ] = mean

            self.std[
                column
            ] = std

        return self

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:

        columns: list[
            np.ndarray
        ] = []

        for column in self.columns:

            numeric = coerce_numeric(
                frame[
                    column
                ]
            )

            values = (
                numeric
                .fillna(
                    self.median[
                        column
                    ]
                )
                .to_numpy(
                    dtype=float
                )
            )

            values = (
                values
                - self.mean[
                    column
                ]
            ) / self.std[
                column
            ]

            values = np.clip(
                values,
                -12.0,
                12.0,
            )

            columns.append(
                values
            )

        return np.column_stack(
            columns
        ).astype(
            np.float32
        )


###############################################################################
# Person-month discrete survival
###############################################################################


def patient_balance_weights(
    frame: pd.DataFrame,
) -> np.ndarray:

    counts = (
        frame
        .groupby(
            "patient_id"
        )[
            "patient_id"
        ]
        .transform(
            "count"
        )
        .to_numpy(
            dtype=float
        )
    )

    return (
        1.0
        / np.maximum(
            counts,
            1.0,
        )
    )


def build_person_month(
    x: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
    landmark_weights: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    if (
        time_days
        <= 0
    ).any():
        raise RuntimeError(
            "Person-month construction requires positive follow-up."
        )

    last_month = (
        np.ceil(
            time_days
            / MONTH_DAYS
        )
        .astype(int)
    )

    last_month = np.clip(
        last_month,
        1,
        MAX_MONTHS,
    )

    repeated_rows = np.repeat(
        np.arange(
            len(
                time_days
            )
        ),
        last_month,
    )

    month = np.concatenate(
        [
            np.arange(
                1,
                n + 1,
                dtype=int,
            )
            for n
            in last_month
        ]
    )

    y = np.zeros(
        len(
            repeated_rows
        ),
        dtype=np.int8,
    )

    offsets = np.cumsum(
        np.r_[
            0,
            last_month,
        ]
    )

    event_in_horizon = (
        (
            event
            == 1
        )
        & (
            time_days
            <= (
                MAX_MONTHS
                * MONTH_DAYS
                + 1e-8
            )
        )
    )

    event_rows = (
        offsets[
            1:
        ][
            event_in_horizon
        ]
        - 1
    )

    y[
        event_rows
    ] = 1

    # One-hot baseline hazard by month.
    month_one_hot = np.zeros(
        (
            len(
                month
            ),
            MAX_MONTHS,
        ),
        dtype=np.float32,
    )

    month_one_hot[
        np.arange(
            len(
                month
            )
        ),
        month - 1,
    ] = 1.0

    x_period = np.concatenate(
        [
            x[
                repeated_rows
            ],
            month_one_hot,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    # Keep total weight for each landmark constant regardless of follow-up.
    period_weights = np.repeat(
        landmark_weights
        / last_month,
        last_month,
    ).astype(
        np.float64
    )

    return (
        x_period,
        y,
        period_weights,
    )


def discrete_nll_per_row(
    hazards: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:

    hazards = np.clip(
        np.asarray(
            hazards,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    last_month = (
        np.ceil(
            time_days
            / MONTH_DAYS
        )
        .astype(int)
    )

    last_month = np.clip(
        last_month,
        1,
        hazards.shape[
            1
        ],
    )

    log_survive = np.log(
        1.0
        - hazards
    )

    log_event = np.log(
        hazards
    )

    losses = np.zeros(
        len(
            time_days
        ),
        dtype=float,
    )

    for index in range(
        len(
            time_days
        )
    ):

        last = int(
            last_month[
                index
            ]
        )

        if (
            event[
                index
            ]
            == 1
            and time_days[
                index
            ]
            <= (
                hazards.shape[
                    1
                ]
                * MONTH_DAYS
                + 1e-8
            )
        ):

            losses[
                index
            ] = -(
                log_survive[
                    index,
                    :last - 1,
                ].sum()
                + log_event[
                    index,
                    last - 1,
                ]
            )

        else:

            losses[
                index
            ] = -(
                log_survive[
                    index,
                    :last,
                ].sum()
            )

    return losses


def survival_from_hazards(
    hazards: np.ndarray,
) -> np.ndarray:

    hazards = np.clip(
        hazards,
        EPS,
        1.0 - EPS,
    )

    return np.cumprod(
        1.0
        - hazards,
        axis=1,
    )


class DiscreteHazardModel:

    def __init__(
        self,
        alpha: float,
    ):

        self.alpha = float(
            alpha
        )

        self.model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=self.alpha,
            l1_ratio=0.05,
            max_iter=120,
            tol=1e-4,
            average=True,
            random_state=RNG_SEED,
        )

    def fit(
        self,
        x_period: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ):

        if int(
            y.sum()
        ) == 0:
            raise RuntimeError(
                "Discrete-time training set contains no events."
            )

        self.model.fit(
            x_period,
            y,
            sample_weight=weights,
        )

        return self

    def predict_hazards(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        n = len(
            x
        )

        hazards = np.zeros(
            (
                n,
                MAX_MONTHS,
            ),
            dtype=float,
        )

        for month in range(
            1,
            MAX_MONTHS + 1,
        ):

            month_one_hot = np.zeros(
                (
                    n,
                    MAX_MONTHS,
                ),
                dtype=np.float32,
            )

            month_one_hot[
                :,
                month - 1,
            ] = 1.0

            period_x = np.concatenate(
                [
                    x,
                    month_one_hot,
                ],
                axis=1,
            ).astype(
                np.float32
            )

            hazards[
                :,
                month - 1,
            ] = (
                self.model
                .predict_proba(
                    period_x
                )[
                    :,
                    1,
                ]
            )

        return np.clip(
            hazards,
            EPS,
            1.0 - EPS,
        )


def fit_regularized_discrete(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    time_column: str,
    event_column: str,
    force_columns: Iterable[str],
    max_features: int,
) -> tuple[
    NumericEncoder,
    DiscreteHazardModel,
    list[
        dict[
            str,
            float
        ]
    ],
]:

    encoder = NumericEncoder(
        max_features=max_features,
        force_columns=force_columns,
    ).fit(
        train,
        feature_columns,
    )

    x_train = encoder.transform(
        train
    )

    x_validation = encoder.transform(
        validation
    )

    landmark_weights = patient_balance_weights(
        train
    )

    (
        period_x,
        y,
        weights,
    ) = build_person_month(
        x_train,
        train[
            time_column
        ].to_numpy(
            dtype=float
        ),
        train[
            event_column
        ].to_numpy(
            dtype=int
        ),
        landmark_weights,
    )

    tuning: list[
        dict[
            str,
            float
        ]
    ] = []

    best_model = None

    best_nll = None

    for alpha in (
        3e-5,
        1e-4,
        3e-4,
    ):

        model = (
            DiscreteHazardModel(
                alpha
            )
            .fit(
                period_x,
                y,
                weights,
            )
        )

        hazards = model.predict_hazards(
            x_validation
        )

        loss = discrete_nll_per_row(
            hazards,
            validation[
                time_column
            ].to_numpy(
                dtype=float
            ),
            validation[
                event_column
            ].to_numpy(
                dtype=int
            ),
        )

        mean_nll = float(
            np.mean(
                loss
            )
        )

        tuning.append(
            {
                "alpha":
                    float(
                        alpha
                    ),
                "validation_mean_nll":
                    mean_nll,
            }
        )

        if (
            best_nll is None
            or mean_nll
            < best_nll
        ):
            best_nll = mean_nll

            best_model = model

    assert (
        best_model
        is not None
    )

    return (
        encoder,
        best_model,
        tuning,
    )


###############################################################################
# Gradient-boosted discrete hazard
###############################################################################


def sample_person_month_for_boosting(
    x_period: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    max_rows: int = 250000,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    if (
        len(
            y
        )
        <= max_rows
    ):
        return (
            x_period,
            y,
            weights,
        )

    positive = np.where(
        y
        == 1
    )[
        0
    ]

    negative = np.where(
        y
        == 0
    )[
        0
    ]

    rng = np.random.default_rng(
        RNG_SEED
    )

    required_negative = max(
        max_rows
        - len(
            positive
        ),
        0,
    )

    if (
        required_negative
        < len(
            negative
        )
    ):

        negative = rng.choice(
            negative,
            size=required_negative,
            replace=False,
        )

    keep = np.sort(
        np.concatenate(
            [
                positive,
                negative,
            ]
        )
    )

    return (
        x_period[
            keep
        ],
        y[
            keep
        ],
        weights[
            keep
        ],
    )


def fit_gradient_hazard(
    train: pd.DataFrame,
    encoder: NumericEncoder,
    time_column: str,
    event_column: str,
) -> HistGradientBoostingClassifier:

    x = encoder.transform(
        train
    )

    (
        x_period,
        y,
        weights,
    ) = build_person_month(
        x,
        train[
            time_column
        ].to_numpy(
            dtype=float
        ),
        train[
            event_column
        ].to_numpy(
            dtype=int
        ),
        patient_balance_weights(
            train
        ),
    )

    (
        x_period,
        y,
        weights,
    ) = sample_person_month_for_boosting(
        x_period,
        y,
        weights,
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=140,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=RNG_SEED,
    )

    model.fit(
        x_period,
        y,
        sample_weight=weights,
    )

    return model


def predict_gradient_hazards(
    model: HistGradientBoostingClassifier,
    encoder: NumericEncoder,
    frame: pd.DataFrame,
) -> np.ndarray:

    x = encoder.transform(
        frame
    )

    n = len(
        frame
    )

    hazards = np.zeros(
        (
            n,
            MAX_MONTHS,
        ),
        dtype=float,
    )

    for month in range(
        1,
        MAX_MONTHS + 1,
    ):

        month_one_hot = np.zeros(
            (
                n,
                MAX_MONTHS,
            ),
            dtype=np.float32,
        )

        month_one_hot[
            :,
            month - 1,
        ] = 1.0

        period_x = np.concatenate(
            [
                x,
                month_one_hot,
            ],
            axis=1,
        ).astype(
            np.float32
        )

        hazards[
            :,
            month - 1,
        ] = (
            model
            .predict_proba(
                period_x
            )[
                :,
                1,
            ]
        )

    return np.clip(
        hazards,
        EPS,
        1.0 - EPS,
    )


###############################################################################
# Penalized Cox model
###############################################################################


class CoxRidge:

    def __init__(
        self,
        l2: float = 0.10,
    ):

        self.l2 = float(
            l2
        )

        self.beta: np.ndarray | None = None

        self.event_times: np.ndarray | None = None

        self.base_cumhaz: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        time_days: np.ndarray,
        event: np.ndarray,
    ):

        x = np.asarray(
            x,
            dtype=float,
        )

        time_days = np.asarray(
            time_days,
            dtype=float,
        )

        event = np.asarray(
            event,
            dtype=int,
        )

        order = np.argsort(
            -time_days,
            kind="mergesort",
        )

        xs = x[
            order
        ]

        times = time_days[
            order
        ]

        events = event[
            order
        ]

        group_end = np.empty(
            len(
                times
            ),
            dtype=int,
        )

        start = 0

        while (
            start
            < len(
                times
            )
        ):

            end = start

            while (
                end + 1
                < len(
                    times
                )
                and times[
                    end + 1
                ]
                == times[
                    start
                ]
            ):
                end += 1

            group_end[
                start:
                end + 1
            ] = end

            start = end + 1

        event_index = np.where(
            events
            == 1
        )[
            0
        ]

        number_events = max(
            len(
                event_index
            ),
            1,
        )

        def objective(
            beta: np.ndarray,
        ):

            eta = np.clip(
                xs
                @ beta,
                -30.0,
                30.0,
            )

            exp_eta = np.exp(
                eta
            )

            risk_sum = np.cumsum(
                exp_eta
            )

            risk_x_sum = np.cumsum(
                exp_eta[
                    :,
                    None
                ]
                * xs,
                axis=0,
            )

            denominator_index = (
                group_end[
                    event_index
                ]
            )

            log_likelihood = np.sum(
                eta[
                    event_index
                ]
                - np.log(
                    np.clip(
                        risk_sum[
                            denominator_index
                        ],
                        EPS,
                        None,
                    )
                )
            )

            expected_x = (
                risk_x_sum[
                    denominator_index
                ]
                / risk_sum[
                    denominator_index,
                    None,
                ]
            )

            gradient_log_likelihood = np.sum(
                xs[
                    event_index
                ]
                - expected_x,
                axis=0,
            )

            loss = (
                -log_likelihood
                / number_events
                + 0.5
                * self.l2
                * float(
                    beta
                    @ beta
                )
            )

            gradient = (
                -gradient_log_likelihood
                / number_events
                + self.l2
                * beta
            )

            return (
                float(
                    loss
                ),
                gradient,
            )

        initial = np.zeros(
            x.shape[
                1
            ],
            dtype=float,
        )

        result = minimize(
            fun=lambda beta:
                objective(
                    beta
                )[
                    0
                ],
            x0=initial,
            jac=lambda beta:
                objective(
                    beta
                )[
                    1
                ],
            method="L-BFGS-B",
            options={
                "maxiter": 250,
                "ftol": 1e-9,
            },
        )

        if not result.success:

            warnings.warn(
                "Cox optimizer did not fully converge: "
                + str(
                    result.message
                )
            )

        self.beta = result.x

        linear_predictor = (
            x
            @ self.beta
        )

        relative_risk = np.exp(
            np.clip(
                linear_predictor,
                -30.0,
                30.0,
            )
        )

        unique_event_times = np.sort(
            np.unique(
                time_days[
                    event
                    == 1
                ]
            )
        )

        increments = []

        for current_time in unique_event_times:

            deaths = int(
                np.sum(
                    (
                        time_days
                        == current_time
                    )
                    & (
                        event
                        == 1
                    )
                )
            )

            risk = float(
                relative_risk[
                    time_days
                    >= current_time
                ]
                .sum()
            )

            increments.append(
                deaths
                / max(
                    risk,
                    EPS,
                )
            )

        self.event_times = (
            unique_event_times
            .astype(float)
        )

        self.base_cumhaz = np.cumsum(
            np.asarray(
                increments,
                dtype=float,
            )
        )

        return self

    def linear_predictor(
        self,
        x: np.ndarray,
    ) -> np.ndarray:

        if self.beta is None:
            raise RuntimeError(
                "Cox model not fit."
            )

        return (
            np.asarray(
                x,
                dtype=float,
            )
            @ self.beta
        )

    def survival(
        self,
        x: np.ndarray,
        horizons_days: np.ndarray,
    ) -> np.ndarray:

        lp = self.linear_predictor(
            x
        )

        relative_risk = np.exp(
            np.clip(
                lp,
                -30.0,
                30.0,
            )
        )

        cumulative = []

        for horizon in horizons_days:

            index = (
                np.searchsorted(
                    self.event_times,
                    horizon,
                    side="right",
                )
                - 1
            )

            if index < 0:
                cumulative.append(
                    0.0
                )
            else:
                cumulative.append(
                    float(
                        self.base_cumhaz[
                            index
                        ]
                    )
                )

        cumulative = np.asarray(
            cumulative,
            dtype=float,
        )

        return np.exp(
            -relative_risk[
                :,
                None
            ]
            * cumulative[
                None,
                :
            ]
        )


###############################################################################
# Survival metrics
###############################################################################


def fit_censoring_km(
    time_days: np.ndarray,
    event: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    censor = (
        1
        - np.asarray(
            event,
            dtype=int,
        )
    )

    order = np.argsort(
        time_days
    )

    time_days = time_days[
        order
    ]

    censor = censor[
        order
    ]

    unique = np.unique(
        time_days
    )

    at_risk = len(
        time_days
    )

    survival = 1.0

    output_times = []

    output_survival = []

    for current_time in unique:

        mask = (
            time_days
            == current_time
        )

        number_at_time = int(
            mask.sum()
        )

        censorings = int(
            censor[
                mask
            ].sum()
        )

        if (
            at_risk
            > 0
            and censorings
            > 0
        ):

            survival *= (
                1.0
                - censorings
                / at_risk
            )

        output_times.append(
            float(
                current_time
            )
        )

        output_survival.append(
            float(
                max(
                    survival,
                    EPS,
                )
            )
        )

        at_risk -= (
            number_at_time
        )

    return (
        np.asarray(
            output_times
        ),
        np.asarray(
            output_survival
        ),
    )


def censoring_survival_at(
    km: tuple[
        np.ndarray,
        np.ndarray,
    ],
    time: float,
    left_limit: bool,
) -> float:

    times, values = km

    side = (
        "left"
        if left_limit
        else "right"
    )

    index = (
        np.searchsorted(
            times,
            time,
            side=side,
        )
        - 1
    )

    if index < 0:
        return 1.0

    return float(
        max(
            values[
                index
            ],
            EPS,
        )
    )


def ipcw_brier(
    predicted_survival: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
    horizon_days: float,
    censoring_km,
) -> float:

    predicted_survival = np.asarray(
        predicted_survival,
        dtype=float,
    )

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    observed_survival = (
        time_days
        > horizon_days
    ).astype(
        float
    )

    weights = np.zeros(
        len(
            time_days
        ),
        dtype=float,
    )

    for index in range(
        len(
            time_days
        )
    ):

        if (
            time_days[
                index
            ]
            <= horizon_days
            and event[
                index
            ]
            == 1
        ):

            weights[
                index
            ] = (
                1.0
                / censoring_survival_at(
                    censoring_km,
                    time_days[
                        index
                    ],
                    left_limit=True,
                )
            )

        elif (
            time_days[
                index
            ]
            > horizon_days
        ):

            weights[
                index
            ] = (
                1.0
                / censoring_survival_at(
                    censoring_km,
                    horizon_days,
                    left_limit=False,
                )
            )

    denominator = float(
        weights.sum()
    )

    if denominator <= 0:
        return float(
            "nan"
        )

    return float(
        np.sum(
            weights
            * (
                observed_survival
                - predicted_survival
            )
            ** 2
        )
        / denominator
    )


def dynamic_auc(
    risk: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
    horizon_days: float,
) -> float:

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    cases = (
        (
            event
            == 1
        )
        & (
            time_days
            <= horizon_days
        )
    )

    controls = (
        time_days
        > horizon_days
    )

    use = (
        cases
        | controls
    )

    if (
        cases.sum()
        < 5
        or controls.sum()
        < 5
    ):
        return float(
            "nan"
        )

    labels = (
        cases[
            use
        ]
        .astype(int)
    )

    return float(
        roc_auc_score(
            labels,
            np.asarray(
                risk
            )[
                use
            ],
        )
    )


def calibration(
    risk: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
    horizon_days: float,
) -> dict[
    str,
    float | int
]:

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    cases = (
        (
            event
            == 1
        )
        & (
            time_days
            <= horizon_days
        )
    )

    controls = (
        time_days
        > horizon_days
    )

    use = (
        cases
        | controls
    )

    if (
        cases.sum()
        < 10
        or controls.sum()
        < 10
    ):

        return {
            "intercept":
                float(
                    "nan"
                ),
            "slope":
                float(
                    "nan"
                ),
            "n":
                int(
                    use.sum()
                ),
        }

    y = (
        cases[
            use
        ]
        .astype(float)
    )

    p = np.clip(
        np.asarray(
            risk
        )[
            use
        ],
        1e-5,
        1.0 - 1e-5,
    )

    logit = np.log(
        p
        / (
            1.0
            - p
        )
    )

    x = np.column_stack(
        [
            np.ones(
                len(
                    logit
                )
            ),
            logit,
        ]
    )

    def objective(
        beta,
    ):

        eta = np.clip(
            x
            @ beta,
            -30.0,
            30.0,
        )

        probability = (
            1.0
            / (
                1.0
                + np.exp(
                    -eta
                )
            )
        )

        loss = -np.sum(
            y
            * np.log(
                probability
                + EPS
            )
            + (
                1.0
                - y
            )
            * np.log(
                1.0
                - probability
                + EPS
            )
        )

        gradient = (
            x.T
            @ (
                probability
                - y
            )
        )

        return (
            float(
                loss
            ),
            gradient,
        )

    result = minimize(
        fun=lambda beta:
            objective(
                beta
            )[
                0
            ],
        x0=np.asarray(
            [
                0.0,
                1.0,
            ]
        ),
        jac=lambda beta:
            objective(
                beta
            )[
                1
            ],
        method="BFGS",
    )

    return {
        "intercept":
            float(
                result.x[
                    0
                ]
            ),

        "slope":
            float(
                result.x[
                    1
                ]
            ),

        "n":
            int(
                use.sum()
            ),
    }


def c_index(
    risk: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
) -> float:

    risk = np.asarray(
        risk,
        dtype=float,
    )

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    concordant = 0.0

    comparable = 0

    for index in np.where(
        event
        == 1
    )[
        0
    ]:

        later = np.where(
            time_days
            > time_days[
                index
            ]
        )[
            0
        ]

        if (
            len(
                later
            )
            == 0
        ):
            continue

        comparable += len(
            later
        )

        concordant += float(
            np.sum(
                risk[
                    index
                ]
                > risk[
                    later
                ]
            )
        )

        concordant += (
            0.5
            * float(
                np.sum(
                    risk[
                        index
                    ]
                    == risk[
                        later
                    ]
                )
            )
        )

    if comparable == 0:
        return float(
            "nan"
        )

    return float(
        concordant
        / comparable
    )


def evaluate_curves(
    survival: np.ndarray,
    time_days: np.ndarray,
    event: np.ndarray,
    training_time: np.ndarray,
    training_event: np.ndarray,
    elapsed_days: np.ndarray | None = None,
) -> dict[
    str,
    Any
]:

    censoring_km = fit_censoring_km(
        training_time,
        training_event,
    )

    metrics: dict[
        str,
        Any
    ] = {
        "horizons": {},
    }

    for horizon in HORIZONS:

        column = (
            horizon
            - 1
        )

        predicted = survival[
            :,
            column
        ]

        supported = np.isfinite(
            predicted
        )

        if elapsed_days is not None:

            supported &= (
                np.asarray(
                    elapsed_days,
                    dtype=float,
                )
                + horizon
                * MONTH_DAYS
                <= 730.0
                + 1e-8
            )

        if (
            supported.sum()
            < 20
        ):

            metrics[
                "horizons"
            ][
                f"{horizon}m"
            ] = {
                "n":
                    int(
                        supported.sum()
                    )
            }

            continue

        t = np.asarray(
            time_days
        )[
            supported
        ]

        e = np.asarray(
            event
        )[
            supported
        ]

        s = predicted[
            supported
        ]

        risk = (
            1.0
            - s
        )

        metrics[
            "horizons"
        ][
            f"{horizon}m"
        ] = {
            "n":
                int(
                    supported.sum()
                ),

            "ipcw_brier":
                ipcw_brier(
                    s,
                    t,
                    e,
                    horizon
                    * MONTH_DAYS,
                    censoring_km,
                ),

            "dynamic_auc":
                dynamic_auc(
                    risk,
                    t,
                    e,
                    horizon
                    * MONTH_DAYS,
                ),

            "calibration":
                calibration(
                    risk,
                    t,
                    e,
                    horizon
                    * MONTH_DAYS,
                ),
        }

    for horizon in (
        12,
        6,
        3,
    ):

        predicted = survival[
            :,
            horizon - 1
        ]

        supported = np.isfinite(
            predicted
        )

        if elapsed_days is not None:

            supported &= (
                np.asarray(
                    elapsed_days,
                    dtype=float,
                )
                + horizon
                * MONTH_DAYS
                <= 730.0
                + 1e-8
            )

        if (
            supported.sum()
            >= 20
        ):

            metrics[
                "c_index_horizon_months"
            ] = horizon

            metrics[
                "c_index"
            ] = c_index(
                1.0
                - predicted[
                    supported
                ],
                np.asarray(
                    time_days
                )[
                    supported
                ],
                np.asarray(
                    event
                )[
                    supported
                ],
            )

            break

    return metrics


###############################################################################
# Stale line-start conditional forecast
###############################################################################


def stale_conditional_survival(
    line_hazards: np.ndarray,
    elapsed_days: np.ndarray,
) -> np.ndarray:

    elapsed_month = (
        np.floor(
            np.asarray(
                elapsed_days,
                dtype=float,
            )
            / MONTH_DAYS
        )
        .astype(int)
    )

    elapsed_month = np.clip(
        elapsed_month,
        0,
        MAX_MONTHS,
    )

    output = np.full(
        (
            len(
                elapsed_month
            ),
            MAX_MONTHS,
        ),
        np.nan,
        dtype=float,
    )

    for row in range(
        len(
            elapsed_month
        )
    ):

        start = int(
            elapsed_month[
                row
            ]
        )

        running = 1.0

        for horizon in range(
            1,
            MAX_MONTHS + 1,
        ):

            absolute_month = (
                start
                + horizon
            )

            if (
                absolute_month
                > MAX_MONTHS
            ):
                break

            hazard = line_hazards[
                row,
                absolute_month - 1,
            ]

            running *= (
                1.0
                - hazard
            )

            output[
                row,
                horizon - 1,
            ] = running

    return output


def stale_conditional_nll(
    line_hazards: np.ndarray,
    elapsed_days: np.ndarray,
    residual_time_days: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:

    elapsed_days = np.asarray(
        elapsed_days,
        dtype=float,
    )

    residual_time_days = np.asarray(
        residual_time_days,
        dtype=float,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    losses = np.full(
        len(
            elapsed_days
        ),
        np.nan,
        dtype=float,
    )

    elapsed_month = (
        np.floor(
            elapsed_days
            / MONTH_DAYS
        )
        .astype(int)
    )

    absolute_endpoint = (
        elapsed_days
        + residual_time_days
    )

    endpoint_month = (
        np.ceil(
            absolute_endpoint
            / MONTH_DAYS
        )
        .astype(int)
    )

    endpoint_month = np.clip(
        endpoint_month,
        1,
        MAX_MONTHS,
    )

    for row in range(
        len(
            elapsed_days
        )
    ):

        start = int(
            elapsed_month[
                row
            ]
        )

        end = int(
            endpoint_month[
                row
            ]
        )

        if (
            start
            >= MAX_MONTHS
        ):
            continue

        if end <= start:
            end = min(
                start + 1,
                MAX_MONTHS,
            )

        hazards = line_hazards[
            row
        ]

        if (
            event[
                row
            ]
            == 1
            and absolute_endpoint[
                row
            ]
            <= 730.0
            + 1e-8
        ):

            loss = -(
                np.log(
                    np.clip(
                        1.0
                        - hazards[
                            start:
                            end - 1
                        ],
                        EPS,
                        1.0,
                    )
                ).sum()
                + np.log(
                    np.clip(
                        hazards[
                            end - 1
                        ],
                        EPS,
                        1.0,
                    )
                )
            )

        else:

            loss = -np.log(
                np.clip(
                    1.0
                    - hazards[
                        start:
                        end
                    ],
                    EPS,
                    1.0,
                )
            ).sum()

        losses[
            row
        ] = float(
            loss
        )

    return losses


###############################################################################
# Dataset assembly
###############################################################################


def build_line_dataset(
    design: pd.DataFrame,
    design_columns: list[str],
    pfs: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:

    frame = (
        design
        .merge(
            pfs,
            on=[
                "patient_id",
                "line",
            ],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            splits,
            on="patient_id",
            how="inner",
            validate="many_to_one",
        )
    )

    rename = {
        column:
            "line__"
            + norm_col(
                column
            )
        for column
        in design_columns
    }

    frame = frame.rename(
        columns=rename
    )

    return (
        frame,
        list(
            rename.values()
        ),
    )


def build_scan_dataset(
    scans: pd.DataFrame,
    pfs: pd.DataFrame,
    design: pd.DataFrame,
    design_columns: list[str],
    splits: pd.DataFrame,
    primary_only: bool,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:

    scans = add_scan_derived_features(
        scans
    )

    current_numeric = (
        choose_current_numeric_features(
            scans
        )
    )

    history = build_scan_history(
        scans,
        current_numeric,
    )

    if primary_only:
        allowed_states = {
            "NON_PROGRESSIVE",
        }
    else:
        allowed_states = {
            "NON_PROGRESSIVE",
            "INDETERMINATE",
        }

    frame = scans[
        scans[
            "scan_state_norm"
        ].isin(
            allowed_states
        )
    ].copy()

    frame = frame.merge(
        history,
        on=[
            "patient_id",
            "line",
            "landmark_day",
            "scan_episode_id_norm",
        ],
        how="left",
        validate="one_to_one",
    )

    frame = frame.merge(
        pfs,
        on=[
            "patient_id",
            "line",
        ],
        how="inner",
        validate="many_to_one",
    )

    frame = frame.merge(
        design,
        on=[
            "patient_id",
            "line",
        ],
        how="inner",
        validate="many_to_one",
    )

    frame = frame.merge(
        splits,
        on="patient_id",
        how="inner",
        validate="many_to_one",
    )

    frame[
        "elapsed_days"
    ] = (
        frame[
            "landmark_day"
        ]
        - frame[
            "line_start_day"
        ]
    )

    frame[
        "residual_time_days"
    ] = (
        frame[
            "time_days"
        ]
        - frame[
            "elapsed_days"
        ]
    )

    frame[
        "residual_event"
    ] = (
        frame[
            "event"
        ]
        .astype(int)
    )

    frame = frame[
        (
            frame[
                "elapsed_days"
            ]
            >= 0
        )
        & (
            frame[
                "residual_time_days"
            ]
            > 0
        )
        & (
            frame[
                "elapsed_days"
            ]
            < 730.0
            + 1e-8
        )
    ].copy()

    if not np.allclose(
        frame[
            "landmark_day"
        ].to_numpy(
            dtype=float
        ),
        frame[
            "episode_end_day_norm"
        ].to_numpy(
            dtype=float
        ),
    ):
        raise RuntimeError(
            "A scan landmark is not located at episode END day."
        )

    rename = {
        column:
            "line__"
            + norm_col(
                column
            )
        for column
        in design_columns
    }

    frame = frame.rename(
        columns=rename
    )

    line_features = list(
        rename.values()
    )

    history_features = [
        column
        for column
        in frame.columns
        if column.startswith(
            "hist_"
        )
    ]

    current_features = [
        column
        for column
        in current_numeric
        if column
        in frame.columns
    ]

    pre_features = (
        line_features
        + [
            "elapsed_days",
        ]
        + history_features
    )

    post_features = (
        pre_features
        + current_features
    )

    return (
        frame,
        pre_features,
        post_features,
        current_features,
    )


###############################################################################
# Bootstrap information gain
###############################################################################


def patient_bootstrap_gain(
    frame: pd.DataFrame,
    earlier_loss: str,
    later_loss: str,
    repetitions: int = 500,
) -> dict[
    str,
    float | int
]:

    grouped = (
        frame
        .groupby(
            "patient_id"
        )[
            [
                earlier_loss,
                later_loss,
            ]
        ]
        .mean()
        .dropna()
    )

    if grouped.empty:

        return {
            "mean":
                float(
                    "nan"
                ),
            "ci_low":
                float(
                    "nan"
                ),
            "ci_high":
                float(
                    "nan"
                ),
            "patients":
                0,
        }

    delta = (
        grouped[
            earlier_loss
        ]
        - grouped[
            later_loss
        ]
    ).to_numpy(
        dtype=float
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    bootstrap = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(
        repetitions
    ):

        bootstrap[
            index
        ] = (
            rng.choice(
                delta,
                size=len(
                    delta
                ),
                replace=True,
            )
            .mean()
        )

    return {
        "mean":
            float(
                delta.mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    bootstrap,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    bootstrap,
                    0.975,
                )
            ),

        "patients":
            int(
                len(
                    delta
                )
            ),
    }


###############################################################################
# Leakage tests
###############################################################################


def synthetic_future_event_test() -> None:

    scans = pd.DataFrame(
        {
            "patient_id":
                [
                    "synthetic",
                    "synthetic",
                    "synthetic",
                ],

            "line":
                [
                    1,
                    1,
                    2,
                ],

            "landmark_day":
                [
                    10.0,
                    20.0,
                    999.0,
                ],

            "scan_episode_id_norm":
                [
                    "s1",
                    "s2",
                    "future",
                ],

            "scan_state_norm":
                [
                    "NON_PROGRESSIVE",
                    "INDETERMINATE",
                    "PROGRESSIVE",
                ],

            "coverage_chest":
                [
                    1.0,
                    0.0,
                    1.0,
                ],
        }
    )

    history = build_scan_history(
        scans,
        [
            "coverage_chest",
        ],
    )

    second = (
        history[
            history[
                "scan_episode_id_norm"
            ]
            == "s2"
        ]
        .iloc[
            0
        ]
    )

    assert (
        second[
            "hist_total_scans"
        ]
        == 1
    )

    assert (
        second[
            "hist_n_progressive"
        ]
        == 0
    )

    assert (
        second[
            "hist_last__coverage_chest"
        ]
        == 1.0
    )


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
        "--output-dir",
        default=(
            "artifacts/checkpoint2"
        ),
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    out = (
        repo
        / args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "pfs":
            repo
            / "artifacts"
            / "checkpoint1b"
            / "reference_usable_lines.parquet",

        "reconciliation":
            repo
            / "artifacts"
            / "checkpoint1c"
            / "reference_reconciliation.json",

        "design":
            repo
            / "artifacts"
            / "checkpoint1c"
            / "public_main_design_matrix.parquet",

        "splits":
            repo
            / "artifacts"
            / "checkpoint1"
            / "canonical"
            / "chord_patient_splits.parquet",

        "scans":
            repo
            / "artifacts"
            / "checkpoint1"
            / "canonical"
            / "chord_mbc_scan_episodes_w3.parquet",

        "landmarks":
            repo
            / "artifacts"
            / "checkpoint1"
            / "canonical"
            / "chord_landmarks.parquet",

        "events":
            repo
            / "artifacts"
            / "checkpoint1"
            / "canonical"
            / "chord_canonical_events.parquet",

        "bpc":
            repo
            / "artifacts"
            / "checkpoint1"
            / "canonical"
            / "bpc_msk_scan_audit.parquet",
    }

    for name, path in paths.items():

        if not path.exists():

            raise SystemExit(
                "[CKPT2_FATAL] missing "
                f"{name}: {path}"
            )

    ###########################################################################
    # A. Load frozen endpoint / source tables
    ###########################################################################

    print(
        "[CKPT2] loading frozen endpoint and canonical tables",
        flush=True,
    )

    pfs = load_frozen_pfs(
        paths[
            "pfs"
        ]
    )

    splits = load_splits(
        paths[
            "splits"
        ]
    )

    (
        design,
        design_columns,
    ) = load_design_matrix(
        paths[
            "design"
        ]
    )

    scans = load_scan_episodes(
        paths[
            "scans"
        ]
    )

    ###########################################################################
    # B. Persistent leakage / data invariants
    ###########################################################################

    print(
        "[CKPT2] running persistent leakage/data tests",
        flush=True,
    )

    synthetic_future_event_test()

    events = pd.read_parquet(
        paths[
            "events"
        ]
    )

    landmarks = pd.read_parquet(
        paths[
            "landmarks"
        ]
    )

    bpc = pd.read_parquet(
        paths[
            "bpc"
        ]
    )

    split_sets = {
        split:
            set(
                group[
                    "patient_id"
                ]
            )
        for split, group
        in splits.groupby(
            "split"
        )
    }

    checks: dict[
        str,
        Any
    ] = {
        "train_val_patient_disjoint":
            not bool(
                split_sets.get(
                    "train",
                    set(),
                )
                & split_sets.get(
                    "val",
                    set(),
                )
            ),

        "train_test_patient_disjoint":
            not bool(
                split_sets.get(
                    "train",
                    set(),
                )
                & split_sets.get(
                    "test",
                    set(),
                )
            ),

        "val_test_patient_disjoint":
            not bool(
                split_sets.get(
                    "val",
                    set(),
                )
                & split_sets.get(
                    "test",
                    set(),
                )
            ),

        "synthetic_future_event_no_effect":
            True,
    }

    event_day = find_col(
        events,
        [
            "event_day",
            "EVENT_DAY",
        ],
        required=False,
    )

    availability_day = find_col(
        events,
        [
            "availability_day",
            "AVAILABILITY_DAY",
        ],
        required=False,
    )

    if (
        event_day is not None
        and availability_day
        is not None
    ):

        event_numeric = pd.to_numeric(
            events[
                event_day
            ],
            errors="coerce",
        )

        availability_numeric = pd.to_numeric(
            events[
                availability_day
            ],
            errors="coerce",
        )

        observed = (
            event_numeric.notna()
            & availability_numeric.notna()
        )

        checks[
            "availability_not_before_event"
        ] = bool(
            (
                availability_numeric[
                    observed
                ]
                >= event_numeric[
                    observed
                ]
            )
            .all()
        )

        checks[
            "availability_rows_checked"
        ] = int(
            observed.sum()
        )

    else:

        checks[
            "availability_not_before_event"
        ] = False

        checks[
            "availability_rows_checked"
        ] = 0

    landmark_type = find_col(
        landmarks,
        [
            "landmark_type",
        ],
        required=False,
    )

    landmark_state = find_col(
        landmarks,
        [
            "scan_state",
            "progression_state_3",
        ],
        required=False,
    )

    if (
        landmark_type
        is not None
        and landmark_state
        is not None
    ):

        violations = (
            landmarks[
                landmark_type
            ]
            .astype(str)
            .str.upper()
            .eq(
                "SCAN"
            )
            & landmarks[
                landmark_state
            ]
            .astype(str)
            .str.upper()
            .eq(
                "PROGRESSIVE"
            )
        )

        checks[
            "no_progressive_scan_prediction_landmarks"
        ] = bool(
            (
                ~violations
            ).all()
        )

        checks[
            "progressive_landmark_violations"
        ] = int(
            violations.sum()
        )

    else:

        checks[
            "no_progressive_scan_prediction_landmarks"
        ] = False

        checks[
            "progressive_landmark_violations"
        ] = -1

    source = find_col(
        bpc,
        [
            "source",
            "SOURCE",
        ],
        required=False,
    )

    if source is not None:

        checks[
            "bpc_materialized_msk_only"
        ] = (
            set(
                bpc[
                    source
                ]
                .astype(str)
                .unique()
            )
            == {
                "BPC_MSK",
            }
        )

    else:

        checks[
            "bpc_materialized_msk_only"
        ] = False

    failures = [
        key
        for key, value
        in checks.items()
        if (
            isinstance(
                value,
                bool,
            )
            and not value
        )
    ]

    atomic_json(
        out
        / "persistent_leakage_checks.json",
        checks,
    )

    if failures:

        raise RuntimeError(
            "Persistent leakage/data tests failed: "
            + ", ".join(
                failures
            )
        )

    print(
        "[CKPT2_PERSISTENT_LEAKAGE_TESTS_PASS]",
        flush=True,
    )

    ###########################################################################
    # C. Endpoint replication report
    ###########################################################################

    reconciliation = json.loads(
        paths[
            "reconciliation"
        ].read_text(
            encoding="utf-8"
        )
    )

    endpoint_report = {
        "frozen_ckpt1b": {
            "patients":
                int(
                    pfs[
                        "patient_id"
                    ].nunique()
                ),

            "lines":
                int(
                    len(
                        pfs
                    )
                ),

            "line_source_counts": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in pfs[
                    "line_source"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },

        "current_public_main_reconciliation":
            reconciliation,

        "frozen_decision": (
            "CKPT1B/current-public-main PFS construction is frozen. "
            "The current-public-main versus manuscript final-cohort "
            "difference is downstream of PFS construction and must not "
            "be used to change canonical endpoint rules."
        ),
    }

    atomic_json(
        out
        / "endpoint_replication.json",
        endpoint_report,
    )

    ###########################################################################
    # D. Line-start reference dataset
    ###########################################################################

    (
        line_frame,
        line_features,
    ) = build_line_dataset(
        design,
        design_columns,
        pfs,
        splits,
    )

    line_split_counts = {
        split: {
            "patients":
                int(
                    group[
                        "patient_id"
                    ].nunique()
                ),

            "lines":
                int(
                    len(
                        group
                    )
                ),

            "events":
                int(
                    group[
                        "event"
                    ].sum()
                ),
        }
        for split, group
        in line_frame.groupby(
            "split"
        )
    }

    train_line = line_frame[
        line_frame[
            "split"
        ]
        == "train"
    ].copy()

    val_line = line_frame[
        line_frame[
            "split"
        ]
        == "val"
    ].copy()

    test_line = line_frame[
        line_frame[
            "split"
        ]
        == "test"
    ].copy()

    if min(
        len(
            train_line
        ),
        len(
            val_line
        ),
        len(
            test_line
        ),
    ) == 0:

        raise RuntimeError(
            "One line-start split is empty."
        )

    print(
        "[CKPT2] line-start cohort",
        line_split_counts,
        flush=True,
    )

    ###########################################################################
    # E. Penalized Cox reference baseline
    ###########################################################################

    print(
        "[CKPT2] fitting penalized Cox line-start baseline",
        flush=True,
    )

    cox_encoder = NumericEncoder(
        max_features=72,
    ).fit(
        train_line,
        line_features,
    )

    cox_x_train = cox_encoder.transform(
        train_line
    )

    cox_x_test = cox_encoder.transform(
        test_line
    )

    cox_model = CoxRidge(
        l2=0.10
    ).fit(
        cox_x_train,
        train_line[
            "time_days"
        ].to_numpy(
            dtype=float
        ),
        train_line[
            "event"
        ].to_numpy(
            dtype=int
        ),
    )

    cox_lp = cox_model.linear_predictor(
        cox_x_test
    )

    cox_survival = cox_model.survival(
        cox_x_test,
        np.asarray(
            HORIZONS,
            dtype=float,
        )
        * MONTH_DAYS,
    )

    line_censoring_km = fit_censoring_km(
        train_line[
            "time_days"
        ].to_numpy(
            dtype=float
        ),
        train_line[
            "event"
        ].to_numpy(
            dtype=int
        ),
    )

    cox_metrics: dict[
        str,
        Any
    ] = {
        "c_index":
            c_index(
                cox_lp,
                test_line[
                    "time_days"
                ].to_numpy(
                    dtype=float
                ),
                test_line[
                    "event"
                ].to_numpy(
                    dtype=int
                ),
            ),

        "numeric_features":
            cox_encoder.columns,

        "horizons": {},
    }

    for column_index, horizon in enumerate(
        HORIZONS
    ):

        survival = cox_survival[
            :,
            column_index
        ]

        risk = (
            1.0
            - survival
        )

        cox_metrics[
            "horizons"
        ][
            f"{horizon}m"
        ] = {
            "ipcw_brier":
                ipcw_brier(
                    survival,
                    test_line[
                        "time_days"
                    ].to_numpy(
                        dtype=float
                    ),
                    test_line[
                        "event"
                    ].to_numpy(
                        dtype=int
                    ),
                    horizon
                    * MONTH_DAYS,
                    line_censoring_km,
                ),

            "dynamic_auc":
                dynamic_auc(
                    risk,
                    test_line[
                        "time_days"
                    ].to_numpy(
                        dtype=float
                    ),
                    test_line[
                        "event"
                    ].to_numpy(
                        dtype=int
                    ),
                    horizon
                    * MONTH_DAYS,
                ),

            "calibration":
                calibration(
                    risk,
                    test_line[
                        "time_days"
                    ].to_numpy(
                        dtype=float
                    ),
                    test_line[
                        "event"
                    ].to_numpy(
                        dtype=int
                    ),
                    horizon
                    * MONTH_DAYS,
                ),
        }

    ###########################################################################
    # F. Regularized discrete-time line-start baseline
    ###########################################################################

    print(
        "[CKPT2] fitting regularized discrete-time line-start baseline",
        flush=True,
    )

    (
        line_encoder,
        line_discrete,
        line_tuning,
    ) = fit_regularized_discrete(
        train=train_line,
        validation=val_line,
        feature_columns=line_features,
        time_column="time_days",
        event_column="event",
        force_columns=[],
        max_features=128,
    )

    line_hazards_test = (
        line_discrete
        .predict_hazards(
            line_encoder
            .transform(
                test_line
            )
        )
    )

    line_survival_test = (
        survival_from_hazards(
            line_hazards_test
        )
    )

    line_nll_test = (
        discrete_nll_per_row(
            line_hazards_test,
            test_line[
                "time_days"
            ].to_numpy(
                dtype=float
            ),
            test_line[
                "event"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    line_discrete_metrics = (
        evaluate_curves(
            line_survival_test,
            test_line[
                "time_days"
            ].to_numpy(
                dtype=float
            ),
            test_line[
                "event"
            ].to_numpy(
                dtype=int
            ),
            train_line[
                "time_days"
            ].to_numpy(
                dtype=float
            ),
            train_line[
                "event"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    line_discrete_metrics[
        "mean_nll"
    ] = float(
        line_nll_test.mean()
    )

    line_discrete_metrics[
        "tuning"
    ] = line_tuning

    line_discrete_metrics[
        "numeric_features"
    ] = (
        line_encoder.columns
    )

    ###########################################################################
    # G. Gradient-boosted line-start baseline
    ###########################################################################

    print(
        "[CKPT2] fitting gradient-boosted line-start baseline",
        flush=True,
    )

    line_gradient = fit_gradient_hazard(
        train_line,
        line_encoder,
        "time_days",
        "event",
    )

    line_gradient_hazards = (
        predict_gradient_hazards(
            line_gradient,
            line_encoder,
            test_line,
        )
    )

    line_gradient_survival = (
        survival_from_hazards(
            line_gradient_hazards
        )
    )

    line_gradient_nll = (
        discrete_nll_per_row(
            line_gradient_hazards,
            test_line[
                "time_days"
            ].to_numpy(
                dtype=float
            ),
            test_line[
                "event"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    line_gradient_metrics = (
        evaluate_curves(
            line_gradient_survival,
            test_line[
                "time_days"
            ].to_numpy(
                dtype=float
            ),
            test_line[
                "event"
            ].to_numpy(
                dtype=int
            ),
            train_line[
                "time_days"
            ].to_numpy(
                dtype=float
            ),
            train_line[
                "event"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    line_gradient_metrics[
        "mean_nll"
    ] = float(
        line_gradient_nll.mean()
    )

    line_start_report = {
        "split_counts":
            line_split_counts,

        "cox_ridge":
            cox_metrics,

        "regularized_discrete_time":
            line_discrete_metrics,

        "gradient_boosted_discrete_time":
            line_gradient_metrics,
    }

    atomic_json(
        out
        / "line_start_baselines.json",
        line_start_report,
    )

    ###########################################################################
    # H. Primary scan-landmark dataset:
    #    NON_PROGRESSIVE scans only.
    ###########################################################################

    print(
        "[CKPT2] constructing primary NON_PROGRESSIVE scan-landmark cohort",
        flush=True,
    )

    (
        scan_frame,
        pre_features,
        post_features,
        current_features,
    ) = build_scan_dataset(
        scans=scans,
        pfs=pfs,
        design=design,
        design_columns=design_columns,
        splits=splits,
        primary_only=True,
    )

    scan_split_counts = {
        split: {
            "patients":
                int(
                    group[
                        "patient_id"
                    ].nunique()
                ),

            "scans":
                int(
                    len(
                        group
                    )
                ),

            "events":
                int(
                    group[
                        "residual_event"
                    ].sum()
                ),
        }
        for split, group
        in scan_frame.groupby(
            "split"
        )
    }

    train_scan = scan_frame[
        scan_frame[
            "split"
        ]
        == "train"
    ].copy()

    val_scan = scan_frame[
        scan_frame[
            "split"
        ]
        == "val"
    ].copy()

    test_scan = scan_frame[
        scan_frame[
            "split"
        ]
        == "test"
    ].copy()

    if min(
        len(
            train_scan
        ),
        len(
            val_scan
        ),
        len(
            test_scan
        ),
    ) == 0:

        raise RuntimeError(
            "One scan split is empty."
        )

    print(
        "[CKPT2] scan cohort",
        scan_split_counts,
        flush=True,
    )

    ###########################################################################
    # I. Stale line-start forecast carried forward
    ###########################################################################

    print(
        "[CKPT2] evaluating stale carried-forward line-start forecast",
        flush=True,
    )

    stale_line_hazards = (
        line_discrete
        .predict_hazards(
            line_encoder
            .transform(
                test_scan
            )
        )
    )

    stale_survival = (
        stale_conditional_survival(
            stale_line_hazards,
            test_scan[
                "elapsed_days"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    stale_nll = (
        stale_conditional_nll(
            stale_line_hazards,
            test_scan[
                "elapsed_days"
            ].to_numpy(
                dtype=float
            ),
            test_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            test_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    stale_metrics = (
        evaluate_curves(
            stale_survival,
            test_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            test_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
            train_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            train_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
            elapsed_days=test_scan[
                "elapsed_days"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    stale_metrics[
        "mean_nll"
    ] = float(
        np.nanmean(
            stale_nll
        )
    )

    stale_metrics[
        "nll_supported"
    ] = int(
        np.isfinite(
            stale_nll
        ).sum()
    )

    ###########################################################################
    # J. Dynamic PRE-SCAN baseline
    ###########################################################################

    print(
        "[CKPT2] fitting dynamic PRE-SCAN regularized survival model",
        flush=True,
    )

    pre_forced = [
        column
        for column
        in pre_features
        if (
            column
            == "elapsed_days"
            or column.startswith(
                "hist_"
            )
        )
    ]

    (
        pre_encoder,
        pre_discrete,
        pre_tuning,
    ) = fit_regularized_discrete(
        train=train_scan,
        validation=val_scan,
        feature_columns=pre_features,
        time_column="residual_time_days",
        event_column="residual_event",
        force_columns=pre_forced,
        max_features=192,
    )

    pre_hazards = (
        pre_discrete
        .predict_hazards(
            pre_encoder
            .transform(
                test_scan
            )
        )
    )

    pre_survival = survival_from_hazards(
        pre_hazards
    )

    pre_nll = discrete_nll_per_row(
        pre_hazards,
        test_scan[
            "residual_time_days"
        ].to_numpy(
            dtype=float
        ),
        test_scan[
            "residual_event"
        ].to_numpy(
            dtype=int
        ),
    )

    pre_metrics = evaluate_curves(
        pre_survival,
        test_scan[
            "residual_time_days"
        ].to_numpy(
            dtype=float
        ),
        test_scan[
            "residual_event"
        ].to_numpy(
            dtype=int
        ),
        train_scan[
            "residual_time_days"
        ].to_numpy(
            dtype=float
        ),
        train_scan[
            "residual_event"
        ].to_numpy(
            dtype=int
        ),
        elapsed_days=test_scan[
            "elapsed_days"
        ].to_numpy(
            dtype=float
        ),
    )

    pre_metrics[
        "mean_nll"
    ] = float(
        pre_nll.mean()
    )

    ###########################################################################
    # K. Dynamic POST-SCAN baseline
    ###########################################################################

    print(
        "[CKPT2] fitting dynamic POST-SCAN regularized survival model",
        flush=True,
    )

    post_forced = list(
        dict.fromkeys(
            pre_forced
            + current_features
        )
    )

    (
        post_encoder,
        post_discrete,
        post_tuning,
    ) = fit_regularized_discrete(
        train=train_scan,
        validation=val_scan,
        feature_columns=post_features,
        time_column="residual_time_days",
        event_column="residual_event",
        force_columns=post_forced,
        max_features=224,
    )

    post_hazards = (
        post_discrete
        .predict_hazards(
            post_encoder
            .transform(
                test_scan
            )
        )
    )

    post_survival = survival_from_hazards(
        post_hazards
    )

    post_nll = discrete_nll_per_row(
        post_hazards,
        test_scan[
            "residual_time_days"
        ].to_numpy(
            dtype=float
        ),
        test_scan[
            "residual_event"
        ].to_numpy(
            dtype=int
        ),
    )

    post_metrics = evaluate_curves(
        post_survival,
        test_scan[
            "residual_time_days"
        ].to_numpy(
            dtype=float
        ),
        test_scan[
            "residual_event"
        ].to_numpy(
            dtype=int
        ),
        train_scan[
            "residual_time_days"
        ].to_numpy(
            dtype=float
        ),
        train_scan[
            "residual_event"
        ].to_numpy(
            dtype=int
        ),
        elapsed_days=test_scan[
            "elapsed_days"
        ].to_numpy(
            dtype=float
        ),
    )

    post_metrics[
        "mean_nll"
    ] = float(
        post_nll.mean()
    )

    ###########################################################################
    # L. Gradient-boosted PRE / POST scan baselines
    ###########################################################################

    print(
        "[CKPT2] fitting gradient-boosted PRE/POST scan baselines",
        flush=True,
    )

    pre_gradient = fit_gradient_hazard(
        train_scan,
        pre_encoder,
        "residual_time_days",
        "residual_event",
    )

    post_gradient = fit_gradient_hazard(
        train_scan,
        post_encoder,
        "residual_time_days",
        "residual_event",
    )

    pre_gradient_hazards = (
        predict_gradient_hazards(
            pre_gradient,
            pre_encoder,
            test_scan,
        )
    )

    post_gradient_hazards = (
        predict_gradient_hazards(
            post_gradient,
            post_encoder,
            test_scan,
        )
    )

    pre_gradient_survival = (
        survival_from_hazards(
            pre_gradient_hazards
        )
    )

    post_gradient_survival = (
        survival_from_hazards(
            post_gradient_hazards
        )
    )

    pre_gradient_nll = (
        discrete_nll_per_row(
            pre_gradient_hazards,
            test_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            test_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    post_gradient_nll = (
        discrete_nll_per_row(
            post_gradient_hazards,
            test_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            test_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
        )
    )

    pre_gradient_metrics = (
        evaluate_curves(
            pre_gradient_survival,
            test_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            test_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
            train_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            train_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
            elapsed_days=test_scan[
                "elapsed_days"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    post_gradient_metrics = (
        evaluate_curves(
            post_gradient_survival,
            test_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            test_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
            train_scan[
                "residual_time_days"
            ].to_numpy(
                dtype=float
            ),
            train_scan[
                "residual_event"
            ].to_numpy(
                dtype=int
            ),
            elapsed_days=test_scan[
                "elapsed_days"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    pre_gradient_metrics[
        "mean_nll"
    ] = float(
        pre_gradient_nll.mean()
    )

    post_gradient_metrics[
        "mean_nll"
    ] = float(
        post_gradient_nll.mean()
    )

    ###########################################################################
    # M. Paired information-gain proof
    ###########################################################################

    paired = test_scan[
        [
            "patient_id",
            "line",
            "landmark_day",
            "scan_episode_id_norm",
            "elapsed_days",
            "residual_time_days",
            "residual_event",
        ]
    ].copy()

    paired[
        "stale_nll"
    ] = stale_nll

    paired[
        "pre_nll"
    ] = pre_nll

    paired[
        "post_nll"
    ] = post_nll

    paired[
        "pre_gradient_nll"
    ] = pre_gradient_nll

    paired[
        "post_gradient_nll"
    ] = post_gradient_nll

    paired[
        "stale_minus_pre_nll"
    ] = (
        paired[
            "stale_nll"
        ]
        - paired[
            "pre_nll"
        ]
    )

    paired[
        "pre_minus_post_nll"
    ] = (
        paired[
            "pre_nll"
        ]
        - paired[
            "post_nll"
        ]
    )

    paired[
        "pre_gradient_minus_post_gradient_nll"
    ] = (
        paired[
            "pre_gradient_nll"
        ]
        - paired[
            "post_gradient_nll"
        ]
    )

    bootstrap = {
        "pre_vs_stale_nll_gain":
            patient_bootstrap_gain(
                paired,
                "stale_nll",
                "pre_nll",
            ),

        "post_vs_pre_nll_gain":
            patient_bootstrap_gain(
                paired,
                "pre_nll",
                "post_nll",
            ),

        "gradient_post_vs_pre_nll_gain":
            patient_bootstrap_gain(
                paired,
                "pre_gradient_nll",
                "post_gradient_nll",
            ),
    }

    ###########################################################################
    # N. Extended scan-policy sensitivity cohort size
    ###########################################################################

    (
        extended_scan,
        _,
        _,
        _,
    ) = build_scan_dataset(
        scans=scans,
        pfs=pfs,
        design=design,
        design_columns=design_columns,
        splits=splits,
        primary_only=False,
    )

    scan_sensitivity = {
        "primary_non_progressive": {
            "rows":
                int(
                    len(
                        scan_frame
                    )
                ),

            "patients":
                int(
                    scan_frame[
                        "patient_id"
                    ].nunique()
                ),

            "state_counts": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in scan_frame[
                    "scan_state_norm"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },

        "extended_non_progressive_or_indeterminate": {
            "rows":
                int(
                    len(
                        extended_scan
                    )
                ),

            "patients":
                int(
                    extended_scan[
                        "patient_id"
                    ].nunique()
                ),

            "state_counts": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in extended_scan[
                    "scan_state_norm"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },

        "primary_proof_reason": (
            "The primary information-gain comparison uses only "
            "NON_PROGRESSIVE scans. Therefore any post-scan gain "
            "cannot be attributed merely to revealing a progressive "
            "or indeterminate state label."
        ),
    }

    ###########################################################################
    # O. Persist predictions / reports
    ###########################################################################

    prediction = test_scan[
        [
            "patient_id",
            "line",
            "landmark_day",
            "scan_episode_id_norm",
            "elapsed_days",
            "residual_time_days",
            "residual_event",
        ]
    ].copy()

    for name, survival in (
        (
            "stale",
            stale_survival,
        ),
        (
            "pre",
            pre_survival,
        ),
        (
            "post",
            post_survival,
        ),
        (
            "pre_gradient",
            pre_gradient_survival,
        ),
        (
            "post_gradient",
            post_gradient_survival,
        ),
    ):

        for horizon in HORIZONS:

            prediction[
                f"{name}_survival_{horizon}m"
            ] = survival[
                :,
                horizon - 1
            ]

    atomic_parquet(
        out
        / "scan_test_predictions.parquet",
        prediction,
    )

    atomic_parquet(
        out
        / "paired_scan_nll_test.parquet",
        paired,
    )

    proof_report = {
        "primary_scan_policy":
            "NON_PROGRESSIVE only",

        "split_counts":
            scan_split_counts,

        "feature_surface": {
            "line_start_requested":
                int(
                    len(
                        line_features
                    )
                ),

            "line_start_numeric_used":
                int(
                    len(
                        line_encoder.columns
                    )
                ),

            "pre_scan_numeric_used":
                int(
                    len(
                        pre_encoder.columns
                    )
                ),

            "post_scan_numeric_used":
                int(
                    len(
                        post_encoder.columns
                    )
                ),

            "current_scan_features_forced_into_post_model":
                post_forced,

            "current_scan_features_resolved":
                current_features,
        },

        "regularized_tuning": {
            "line_start":
                line_tuning,

            "pre_scan":
                pre_tuning,

            "post_scan":
                post_tuning,
        },

        "methods": {
            "stale_line_start":
                stale_metrics,

            "dynamic_pre_scan":
                pre_metrics,

            "dynamic_post_scan":
                post_metrics,

            "gradient_pre_scan":
                pre_gradient_metrics,

            "gradient_post_scan":
                post_gradient_metrics,
        },

        "paired_patient_bootstrap":
            bootstrap,

        "interpretation": {
            "stale_minus_pre":
                (
                    "Positive means pre-scan history lowers "
                    "held-out NLL relative to the stale "
                    "conditional line-start forecast."
                ),

            "pre_minus_post":
                (
                    "Positive means the current non-progressive "
                    "scan lowers held-out NLL beyond all "
                    "pre-scan information."
                ),
        },
    }

    atomic_json(
        out
        / "scan_updating_proof.json",
        proof_report,
    )

    atomic_json(
        out
        / "scan_policy_sensitivity.json",
        scan_sensitivity,
    )

    ###########################################################################
    # P. Determine checkpoint interpretation
    ###########################################################################

    primary_gain = (
        bootstrap[
            "post_vs_pre_nll_gain"
        ]
    )

    if (
        np.isfinite(
            primary_gain[
                "ci_low"
            ]
        )
        and primary_gain[
            "ci_low"
        ]
        > 0
    ):

        status = (
            "PASS_SCAN_UPDATE_INFORMATION_GAIN"
        )

    elif (
        np.isfinite(
            primary_gain[
                "mean"
            ]
        )
        and primary_gain[
            "mean"
        ]
        > 0
    ):

        status = (
            "PASS_POSITIVE_POST_SCAN_POINT_ESTIMATE_UNCERTAIN"
        )

    else:

        status = (
            "PASS_NO_POSITIVE_POST_SCAN_GAIN_YET"
        )

    qc = {
        "status":
            status,

        "persistent_leakage_checks":
            checks,

        "frozen_pfs_patients":
            int(
                pfs[
                    "patient_id"
                ].nunique()
            ),

        "frozen_pfs_lines":
            int(
                len(
                    pfs
                )
            ),

        "public_design_lines_used":
            int(
                len(
                    line_frame
                )
            ),

        "line_split_counts":
            line_split_counts,

        "scan_split_counts":
            scan_split_counts,

        "primary_scan_rows":
            int(
                len(
                    scan_frame
                )
            ),

        "primary_scan_patients":
            int(
                scan_frame[
                    "patient_id"
                ].nunique()
            ),

        "post_vs_pre_nll_gain":
            primary_gain,

        "errors":
            [],

        "warnings":
            [],
    }

    atomic_json(
        out
        / "qc.json",
        qc,
    )

    ###########################################################################
    # Q. Human-readable audit
    ###########################################################################

    report: list[
        str
    ] = []

    report.append(
        "# Checkpoint 2 - Endpoint Replication and Baselines"
    )

    report.append("")

    report.append(
        f"Status: **{status}**"
    )

    report.append("")

    report.append(
        "## Frozen endpoint"
    )

    report.append("")

    report.append(
        "- CKPT1B/current-public PFS patients: "
        f"`{pfs['patient_id'].nunique()}`"
    )

    report.append(
        "- CKPT1B/current-public PFS lines: "
        f"`{len(pfs)}`"
    )

    report.append(
        "- Endpoint construction remains frozen; "
        "the manuscript/current-public final-design discrepancy "
        "is downstream of PFS construction."
    )

    report.append("")

    report.append(
        "## Patient-disjoint line-start baselines"
    )

    report.append("")

    report.append(
        "- Cox test C-index: "
        f"`{cox_metrics['c_index']:.6f}`"
    )

    report.append(
        "- Regularized discrete-time test mean NLL: "
        f"`{line_discrete_metrics['mean_nll']:.6f}`"
    )

    report.append(
        "- Gradient-boosted discrete-time test mean NLL: "
        f"`{line_gradient_metrics['mean_nll']:.6f}`"
    )

    report.append("")

    report.append(
        "## Primary scan-updating proof"
    )

    report.append("")

    report.append(
        "- Policy: NON_PROGRESSIVE scans only."
    )

    report.append(
        "- Test scans: "
        f"`{len(test_scan)}`"
    )

    report.append(
        "- Test patients: "
        f"`{test_scan['patient_id'].nunique()}`"
    )

    report.append(
        "- Stale line-start mean NLL: "
        f"`{stale_metrics['mean_nll']:.6f}`"
    )

    report.append(
        "- Dynamic pre-scan mean NLL: "
        f"`{pre_metrics['mean_nll']:.6f}`"
    )

    report.append(
        "- Dynamic post-scan mean NLL: "
        f"`{post_metrics['mean_nll']:.6f}`"
    )

    report.append(
        "- Patient-bootstrap pre-vs-stale NLL gain: "
        f"`{bootstrap['pre_vs_stale_nll_gain']}`"
    )

    report.append(
        "- Patient-bootstrap post-vs-pre NLL gain: "
        f"`{bootstrap['post_vs_pre_nll_gain']}`"
    )

    report.append(
        "- Gradient-boosted post-vs-pre NLL gain: "
        f"`{bootstrap['gradient_post_vs_pre_nll_gain']}`"
    )

    report.append("")

    report.append(
        "Positive gain means the later information source "
        "reduced held-out negative log-likelihood."
    )

    report.append("")

    report.append(
        "## Persistent leakage/data checks"
    )

    report.append("")

    for key, value in checks.items():

        report.append(
            f"- {key}: `{value}`"
        )

    report.append("")

    atomic_text(
        out
        / "audit.md",
        "\n".join(
            report
        )
        + "\n",
    )

    ###########################################################################
    # R. Update PROJECT_STATE.md atomically
    ###########################################################################

    state_path = (
        repo
        / "PROJECT_STATE.md"
    )

    if state_path.exists():

        existing = (
            state_path
            .read_text(
                encoding="utf-8"
            )
        )

    else:

        existing = (
            "# PROJECT STATE\n"
        )

    marker = (
        "<!-- CKPT2_ENDPOINT_BASELINES -->"
    )

    if marker in existing:

        existing = (
            existing
            .split(
                marker
            )[
                0
            ]
            .rstrip()
            + "\n"
        )

    state_section = f"""
{marker}
## Checkpoint 2 - Endpoint replication and baseline proof

Status: **{status}**

Frozen endpoint:
- CKPT1B/current-public PFS: {int(pfs['patient_id'].nunique())} patients / {len(pfs)} lines.
- Canonical PFS rules are frozen.
- Do not force the manuscript 2,881 / 8,791 counts.
- Current-public downstream QC discrepancy remains a reproducibility/version finding.

Evaluation design:
- Patient-disjoint CKPT1 split reused before landmark expansion.
- Line-start reference baselines: penalized Cox, regularized discrete-time hazard, gradient-boosted discrete-time hazard.
- Primary scan proof uses NON_PROGRESSIVE scan landmarks only.
- Stale forecast: line-start survival curve conditionally carried forward to scan time.
- Pre-scan model: line-start features + elapsed time + prior scan history.
- Post-scan model: pre-scan features + current scan evidence.
- Paired information gain is bootstrapped by patient.

Primary held-out post-vs-pre NLL gain:
- mean: {primary_gain['mean']}
- patient-bootstrap 95% CI: [{primary_gain['ci_low']}, {primary_gain['ci_high']}]

Artifacts:
- artifacts/checkpoint2/endpoint_replication.json
- artifacts/checkpoint2/persistent_leakage_checks.json
- artifacts/checkpoint2/line_start_baselines.json
- artifacts/checkpoint2/scan_updating_proof.json
- artifacts/checkpoint2/scan_policy_sensitivity.json
- artifacts/checkpoint2/scan_test_predictions.parquet
- artifacts/checkpoint2/paired_scan_nll_test.parquet
- artifacts/checkpoint2/qc.json
- artifacts/checkpoint2/audit.md
- artifacts/handoff/checkpoint_02.json
"""

    atomic_text(
        state_path,
        existing.rstrip()
        + "\n\n"
        + state_section.strip()
        + "\n",
    )

    ###########################################################################
    # S. Handoff
    ###########################################################################

    handoff = {
        "checkpoint":
            "2",

        "name":
            "endpoint_replication_and_baselines",

        "status":
            status,

        "frozen_endpoint": {
            "patients":
                int(
                    pfs[
                        "patient_id"
                    ].nunique()
                ),

            "lines":
                int(
                    len(
                        pfs
                    )
                ),
        },

        "line_start_test": {
            "cox_c_index":
                cox_metrics[
                    "c_index"
                ],

            "regularized_discrete_time_mean_nll":
                line_discrete_metrics[
                    "mean_nll"
                ],

            "gradient_boosted_discrete_time_mean_nll":
                line_gradient_metrics[
                    "mean_nll"
                ],
        },

        "scan_proof": {
            "primary_policy":
                "NON_PROGRESSIVE",

            "test_scans":
                int(
                    len(
                        test_scan
                    )
                ),

            "test_patients":
                int(
                    test_scan[
                        "patient_id"
                    ].nunique()
                ),

            "stale_mean_nll":
                stale_metrics[
                    "mean_nll"
                ],

            "pre_mean_nll":
                pre_metrics[
                    "mean_nll"
                ],

            "post_mean_nll":
                post_metrics[
                    "mean_nll"
                ],

            "pre_vs_stale":
                bootstrap[
                    "pre_vs_stale_nll_gain"
                ],

            "post_vs_pre":
                bootstrap[
                    "post_vs_pre_nll_gain"
                ],

            "gradient_post_vs_pre":
                bootstrap[
                    "gradient_post_vs_pre_nll_gain"
                ],
        },

        "leakage_checks":
            checks,

        "next_action": (
            (
                "Proceed to GENIE genomic encoder pretraining; "
                "retain CKPT2 baselines as the frozen classical/dynamic reference."
            )
            if status
            == "PASS_SCAN_UPDATE_INFORMATION_GAIN"
            else
            (
                "Inspect scan evidence feature availability and the paired "
                "pre/post results before claiming current-scan information gain. "
                "Do not alter frozen endpoint rules."
            )
        ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_02.json",
        handoff,
    )

    ###########################################################################
    # T. Compact terminal output
    ###########################################################################

    def horizon_line(
        metrics,
        horizon,
    ):

        value = (
            metrics
            .get(
                "horizons",
                {},
            )
            .get(
                f"{horizon}m",
                {},
            )
        )

        return {
            "n":
                value.get(
                    "n"
                ),

            "brier":
                value.get(
                    "ipcw_brier"
                ),

            "auc":
                value.get(
                    "dynamic_auc"
                ),

            "calibration":
                value.get(
                    "calibration"
                ),
        }

    print("")
    print(
        "========== CKPT2 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "frozen_pfs_patients="
        f"{int(pfs['patient_id'].nunique())}"
    )

    print(
        "frozen_pfs_lines="
        f"{len(pfs)}"
    )

    print(
        "public_design_lines_used="
        f"{len(line_frame)}"
    )

    print(
        "line_split_counts="
        f"{line_split_counts}"
    )

    print(
        "primary_scan_rows="
        f"{len(scan_frame)}"
    )

    print(
        "primary_scan_patients="
        f"{scan_frame['patient_id'].nunique()}"
    )

    print(
        "scan_split_counts="
        f"{scan_split_counts}"
    )

    print(
        "cox_test_c_index="
        f"{cox_metrics['c_index']:.6f}"
    )

    print(
        "line_discrete_test_mean_nll="
        f"{line_discrete_metrics['mean_nll']:.6f}"
    )

    print(
        "line_gradient_test_mean_nll="
        f"{line_gradient_metrics['mean_nll']:.6f}"
    )

    print(
        "stale_test_mean_nll="
        f"{stale_metrics['mean_nll']:.6f}"
    )

    print(
        "pre_scan_test_mean_nll="
        f"{pre_metrics['mean_nll']:.6f}"
    )

    print(
        "post_scan_test_mean_nll="
        f"{post_metrics['mean_nll']:.6f}"
    )

    print(
        "pre_gradient_test_mean_nll="
        f"{pre_gradient_metrics['mean_nll']:.6f}"
    )

    print(
        "post_gradient_test_mean_nll="
        f"{post_gradient_metrics['mean_nll']:.6f}"
    )

    print(
        "pre_vs_stale_nll_gain="
        f"{bootstrap['pre_vs_stale_nll_gain']}"
    )

    print(
        "post_vs_pre_nll_gain="
        f"{bootstrap['post_vs_pre_nll_gain']}"
    )

    print(
        "gradient_post_vs_pre_nll_gain="
        f"{bootstrap['gradient_post_vs_pre_nll_gain']}"
    )

    print(
        "leakage_checks=PASS"
    )

    print(
        "audit_md="
        "artifacts/checkpoint2/audit.md"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_02.json"
    )

    print(
        "========== CKPT2 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT2 DECISION PACKET =========="
    )

    print(
        "line_tuning="
        f"{line_tuning}"
    )

    print(
        "pre_scan_tuning="
        f"{pre_tuning}"
    )

    print(
        "post_scan_tuning="
        f"{post_tuning}"
    )

    print(
        "resolved_current_scan_features="
        f"{current_features}"
    )

    for horizon in HORIZONS:

        print(
            f"horizon_{horizon}m_stale="
            f"{horizon_line(stale_metrics, horizon)}"
        )

        print(
            f"horizon_{horizon}m_pre="
            f"{horizon_line(pre_metrics, horizon)}"
        )

        print(
            f"horizon_{horizon}m_post="
            f"{horizon_line(post_metrics, horizon)}"
        )

    print(
        "scan_policy_sensitivity="
        f"{scan_sensitivity}"
    )

    print(
        "proof_bootstrap="
        f"{bootstrap}"
    )

    print(
        "========== CKPT2 DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
