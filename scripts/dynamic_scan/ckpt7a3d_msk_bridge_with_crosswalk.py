#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


OBSERVED_STATES = {
    "PROGRESSIVE",
    "NON_PROGRESSIVE",
    "INDETERMINATE",
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


def find_col(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:

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
            return lookup[
                key
            ]

    if required:
        raise KeyError(
            f"Missing {list(candidates)}; "
            f"available={list(columns)}"
        )

    return None


def quantiles(
    values: Iterable[float],
) -> dict[str, Any]:

    array = np.asarray(
        list(values),
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
                np.min(
                    array
                )
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
                np.quantile(
                    array,
                    0.50,
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

        "p95":
            float(
                np.quantile(
                    array,
                    0.95,
                )
            ),

        "max":
            float(
                np.max(
                    array
                )
            ),
    }


def import_ckpt7a3(
    repo: Path,
):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a3_msk_bridge_validation.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "ckpt7a3_reuse",
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to import CKPT7A3."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


###############################################################################
# Canonical CHORD W3 normalization
###############################################################################


def load_chord(
    repo: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    patient_col = find_col(
        frame.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    day_col = find_col(
        frame.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
    )

    state_col = find_col(
        frame.columns,
        [
            "progression_state_3",
            "scan_state",
        ],
    )

    line_col = find_col(
        frame.columns,
        [
            "treatment_line",
            "line",
        ],
        required=False,
    )

    scan_id_col = find_col(
        frame.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "chord_patient_id":
                frame[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "chord_episode_day":
                pd.to_numeric(
                    frame[
                        day_col
                    ],
                    errors="coerce",
                ),

            "chord_scan_state":
                frame[
                    state_col
                ]
                .astype(str)
                .str.strip()
                .str.upper(),
        }
    )

    if line_col is not None:

        out[
            "chord_line"
        ] = pd.to_numeric(
            frame[
                line_col
            ],
            errors="coerce",
        )

    if scan_id_col is not None:

        out[
            "scan_episode_id"
        ] = (
            frame[
                scan_id_col
            ]
            .astype(str)
            .str.strip()
        )

    out[
        "chord_row"
    ] = np.arange(
        len(out),
        dtype=int,
    )

    return out


###############################################################################
# Greedy one-to-one nearest episode matching within patient
###############################################################################


def match_episodes(
    bpc: pd.DataFrame,
    chord: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:

    rows = []

    common = sorted(
        set(
            bpc[
                "chord_patient_id"
            ]
        )
        & set(
            chord[
                "chord_patient_id"
            ]
        )
    )

    for patient in common:

        left = bpc[
            bpc[
                "chord_patient_id"
            ]
            == patient
        ].copy()

        right = chord[
            chord[
                "chord_patient_id"
            ]
            == patient
        ].copy()

        left = left[
            left[
                "episode_end_day"
            ].notna()
        ]

        right = right[
            right[
                "chord_episode_day"
            ].notna()
        ]

        candidates = []

        for _, lrow in left.iterrows():

            for _, rrow in right.iterrows():

                delta = abs(
                    float(
                        lrow[
                            "episode_end_day"
                        ]
                    )
                    - float(
                        rrow[
                            "chord_episode_day"
                        ]
                    )
                )

                if delta <= tolerance:

                    candidates.append(
                        (
                            delta,
                            int(
                                lrow[
                                    "bpc_global_row"
                                ]
                            ),
                            int(
                                rrow[
                                    "chord_row"
                                ]
                            ),
                        )
                    )

        candidates.sort()

        used_bpc = set()
        used_chord = set()

        bpc_lookup = (
            left
            .set_index(
                "bpc_global_row",
                drop=False,
            )
        )

        chord_lookup = (
            right
            .set_index(
                "chord_row",
                drop=False,
            )
        )

        for (
            delta,
            bpc_row,
            chord_row,
        ) in candidates:

            if (
                bpc_row in used_bpc
                or chord_row in used_chord
            ):
                continue

            used_bpc.add(
                bpc_row
            )

            used_chord.add(
                chord_row
            )

            lrow = bpc_lookup.loc[
                bpc_row
            ]

            rrow = chord_lookup.loc[
                chord_row
            ]

            record = {
                "bpc_global_row":
                    bpc_row,

                "chord_row":
                    chord_row,

                "bpc_patient_id":
                    lrow[
                        "patient_id"
                    ],

                "chord_patient_id":
                    patient,

                "bpc_episode_id":
                    lrow[
                        "bpc_episode_id"
                    ],

                "bpc_episode_day":
                    float(
                        lrow[
                            "episode_end_day"
                        ]
                    ),

                "chord_episode_day":
                    float(
                        rrow[
                            "chord_episode_day"
                        ]
                    ),

                "abs_day_delta":
                    float(
                        delta
                    ),

                "bpc_scan_state":
                    lrow[
                        "scan_state"
                    ],

                "chord_scan_state":
                    rrow[
                        "chord_scan_state"
                    ],
            }

            if "chord_line" in rrow.index:

                record[
                    "chord_line"
                ] = rrow[
                    "chord_line"
                ]

            if "scan_episode_id" in rrow.index:

                record[
                    "scan_episode_id"
                ] = rrow[
                    "scan_episode_id"
                ]

            rows.append(
                record
            )

    return pd.DataFrame(
        rows
    )


###############################################################################
# Nearest-day diagnostics unrestricted by tolerance
###############################################################################


def nearest_day_diagnostics(
    bpc: pd.DataFrame,
    chord: pd.DataFrame,
) -> dict[str, Any]:

    nearest = []

    patient_summaries = []

    common = sorted(
        set(
            bpc[
                "chord_patient_id"
            ]
        )
        & set(
            chord[
                "chord_patient_id"
            ]
        )
    )

    for patient in common:

        left_days = (
            bpc.loc[
                bpc[
                    "chord_patient_id"
                ]
                == patient,
                "episode_end_day",
            ]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        right_days = (
            chord.loc[
                chord[
                    "chord_patient_id"
                ]
                == patient,
                "chord_episode_day",
            ]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        if (
            len(left_days) == 0
            or len(right_days) == 0
        ):
            continue

        patient_nearest = []

        for day in left_days:

            distance = float(
                np.min(
                    np.abs(
                        right_days
                        - day
                    )
                )
            )

            nearest.append(
                distance
            )

            patient_nearest.append(
                distance
            )

        patient_summaries.append(
            {
                "chord_patient_id":
                    patient,

                "bpc_episode_count":
                    int(
                        len(
                            left_days
                        )
                    ),

                "chord_episode_count":
                    int(
                        len(
                            right_days
                        )
                    ),

                "median_nearest_day_distance":
                    float(
                        np.median(
                            patient_nearest
                        )
                    ),
            }
        )

    patient_frame = pd.DataFrame(
        patient_summaries
    )

    return {
        "episode_level_nearest_distance":
            quantiles(
                nearest
            ),

        "patients":
            int(
                len(
                    patient_frame
                )
            ),

        "patient_median_distance":
            (
                quantiles(
                    patient_frame[
                        "median_nearest_day_distance"
                    ]
                )
                if len(
                    patient_frame
                )
                else {
                    "n": 0,
                }
            ),

        "patient_frame":
            patient_frame,
    }


###############################################################################
# Treatment-line comparison
###############################################################################


def compare_lines(
    matches: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:

    if (
        matches.empty
        or "chord_line"
        not in matches.columns
    ):
        return {}

    fields = [
        field
        for field in (
            "line_active_interval",
            "line_latest_started",
        )
        if field in episodes.columns
    ]

    lookup_columns = [
        "bpc_global_row",
        *fields,
    ]

    merged = matches.merge(
        episodes[
            lookup_columns
        ],
        on="bpc_global_row",
        how="left",
        validate="one_to_one",
    )

    report = {}

    for field in fields:

        current = merged[
            merged[
                "chord_line"
            ].notna()
            & merged[
                field
            ].notna()
        ].copy()

        if current.empty:

            report[
                field
            ] = {
                "rows": 0,
                "agreement": None,
            }

            continue

        report[
            field
        ] = {
            "rows":
                int(
                    len(
                        current
                    )
                ),

            "agreement":
                float(
                    (
                        current[
                            "chord_line"
                        ].astype(float)
                        == current[
                            field
                        ].astype(float)
                    ).mean()
                ),
        }

    return report


###############################################################################
# CKPT5 feature reference
###############################################################################


def load_ckpt5_reference(
    repo: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[str],
]:

    index = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared"
        / "scan_index.parquet"
    )

    matrix = np.load(
        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared"
        / "current_scan_features_f32.npy",
        mmap_mode="r",
    )

    schema = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint5"
            / "prepared"
            / "feature_schema.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    names = list(
        schema[
            "current_scan"
        ]
    )

    if matrix.shape != (
        len(
            index
        ),
        len(
            names
        ),
    ):
        raise RuntimeError(
            "CKPT5 scan feature shape mismatch."
        )

    patient_col = find_col(
        index.columns,
        [
            "patient_id",
        ],
    )

    day_col = find_col(
        index.columns,
        [
            "landmark_day",
            "episode_end_day",
        ],
    )

    scan_id_col = find_col(
        index.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    out = pd.DataFrame(
        {
            "chord_patient_id":
                index[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "ckpt5_landmark_day":
                pd.to_numeric(
                    index[
                        day_col
                    ],
                    errors="coerce",
                ),

            "ckpt5_feature_row":
                np.arange(
                    len(
                        index
                    ),
                    dtype=int,
                ),
        }
    )

    if scan_id_col is not None:

        out[
            "scan_episode_id"
        ] = (
            index[
                scan_id_col
            ]
            .astype(str)
            .str.strip()
        )

    return (
        out,
        matrix,
        names,
    )


###############################################################################
# Feature comparison via matched CHORD scan
###############################################################################


def compare_scan_features(
    matches: pd.DataFrame,
    bpc_matrix: np.ndarray,
    ckpt5_index: pd.DataFrame,
    ckpt5_matrix: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:

    if matches.empty:

        return {
            "matched_rows":
                0,
        }

    working = matches.copy()

    ###########################################################################
    # Prefer authoritative scan_episode_id if both sides expose it.
    ###########################################################################

    if (
        "scan_episode_id"
        in working.columns
        and "scan_episode_id"
        in ckpt5_index.columns
        and working[
            "scan_episode_id"
        ].astype(str).str.len().gt(0).any()
    ):

        feature_matches = working.merge(
            ckpt5_index[
                [
                    "scan_episode_id",
                    "ckpt5_feature_row",
                ]
            ],
            on="scan_episode_id",
            how="inner",
        )

        join_method = (
            "scan_episode_id"
        )

    else:

        feature_matches = working.merge(
            ckpt5_index,
            left_on=[
                "chord_patient_id",
                "chord_episode_day",
            ],
            right_on=[
                "chord_patient_id",
                "ckpt5_landmark_day",
            ],
            how="inner",
        )

        join_method = (
            "patient_day"
        )

    if feature_matches.empty:

        return {
            "matched_rows":
                0,

            "join_method":
                join_method,
        }

    bpc_rows = feature_matches[
        "bpc_global_row"
    ].to_numpy(
        dtype=int
    )

    ckpt_rows = feature_matches[
        "ckpt5_feature_row"
    ].to_numpy(
        dtype=int
    )

    left = np.asarray(
        bpc_matrix[
            bpc_rows
        ],
        dtype=np.float32,
    )

    right = np.asarray(
        ckpt5_matrix[
            ckpt_rows
        ],
        dtype=np.float32,
    )

    absolute = np.abs(
        left
        - right
    )

    per_feature = {}

    for index, name in enumerate(
        feature_names
    ):

        per_feature[
            name
        ] = {
            "bpc_mean":
                float(
                    left[
                        :,
                        index
                    ].mean()
                ),

            "chord_mean":
                float(
                    right[
                        :,
                        index
                    ].mean()
                ),

            "mean_absolute_difference":
                float(
                    absolute[
                        :,
                        index
                    ].mean()
                ),

            "exact_fraction":
                float(
                    (
                        absolute[
                            :,
                            index
                        ]
                        < 1e-6
                    ).mean()
                ),
        }

    return {
        "matched_rows":
            int(
                len(
                    feature_matches
                )
            ),

        "join_method":
            join_method,

        "mean_absolute_difference":
            float(
                absolute.mean()
            ),

        "median_row_mean_absolute_difference":
            float(
                np.median(
                    absolute.mean(
                        axis=1
                    )
                )
            ),

        "vector_exact_fraction":
            float(
                (
                    absolute.max(
                        axis=1
                    )
                    < 1e-6
                ).mean()
            ),

        "per_feature":
            per_feature,
    }


###############################################################################
# State summary
###############################################################################


def state_summary(
    matches: pd.DataFrame,
) -> dict[str, Any]:

    if matches.empty:

        return {
            "rows": 0,
        }

    observed = matches[
        matches[
            "bpc_scan_state"
        ].isin(
            OBSERVED_STATES
        )
        & matches[
            "chord_scan_state"
        ].isin(
            OBSERVED_STATES
        )
    ].copy()

    if observed.empty:

        return {
            "rows": 0,
        }

    confusion = (
        observed.groupby(
            [
                "bpc_scan_state",
                "chord_scan_state",
            ],
            observed=True,
        )
        .size()
        .rename(
            "n"
        )
        .reset_index()
    )

    return {
        "rows":
            int(
                len(
                    observed
                )
            ),

        "patients":
            int(
                observed[
                    "chord_patient_id"
                ].nunique()
            ),

        "agreement":
            float(
                (
                    observed[
                        "bpc_scan_state"
                    ]
                    == observed[
                        "chord_scan_state"
                    ]
                ).mean()
            ),

        "confusion":
            confusion.to_dict(
                orient="records"
            ),
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
        "--output-dir",
        default=(
            "artifacts/checkpoint7a3d"
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

    out = (
        repo
        / args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Identity crosswalk
    ###########################################################################

    identity = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint7a3c"
            / "identity_validation.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        identity[
            "status"
        ]
        != "PASS_PREFIX_IDENTITY_BRIDGE"
    ):
        raise RuntimeError(
            "CKPT7A3C identity bridge not frozen PASS."
        )

    crosswalk = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3c"
        / "msk_patient_crosswalk.parquet"
    )

    crosswalk = crosswalk[
        [
            "bpc_patient_id",
            "chord_patient_id",
        ]
    ].drop_duplicates()

    if (
        len(
            crosswalk
        )
        != 416
    ):
        raise RuntimeError(
            "Unexpected frozen MSK patient crosswalk size."
        )

    ###########################################################################
    # Load already-generated BPC W3 predictor episodes and scan matrix.
    #
    # These were generated without outcomes in CKPT7A3 and remain valid; only
    # the positive-control identity namespace was wrong.
    ###########################################################################

    episodes = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3"
        / "bpc_w3_predictor_episodes.parquet"
    )

    bpc_matrix = np.load(
        repo
        / "artifacts"
        / "checkpoint7a3"
        / "bpc_w3_scan_features_f32.npy",
        mmap_mode="r",
    )

    semantic_crosswalk = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint7a3"
            / "external_semantic_crosswalk.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    feature_names = list(
        semantic_crosswalk[
            "frozen_current_scan_feature_names"
        ]
    )

    if bpc_matrix.shape != (
        len(
            episodes
        ),
        len(
            feature_names
        ),
    ):
        raise RuntimeError(
            "BPC episode / feature matrix shape mismatch."
        )

    episodes = episodes.copy()

    episodes[
        "bpc_global_row"
    ] = np.arange(
        len(
            episodes
        ),
        dtype=int,
    )

    ###########################################################################
    # Restrict positive control to frozen 416-patient bridge.
    ###########################################################################

    msk = episodes[
        episodes[
            "site"
        ]
        == "MSK"
    ].copy()

    msk = msk.merge(
        crosswalk,
        left_on="patient_id",
        right_on="bpc_patient_id",
        how="inner",
        validate="many_to_one",
    )

    if (
        msk[
            "chord_patient_id"
        ].nunique()
        != 416
    ):
        raise RuntimeError(
            "Not all frozen bridged patients are represented "
            "in BPC W3 episodes."
        )

    observed_msk = msk[
        msk[
            "scan_state"
        ].isin(
            OBSERVED_STATES
        )
    ].copy()

    ###########################################################################
    # CHORD reference.
    ###########################################################################

    chord = load_chord(
        repo
    )

    chord = chord[
        chord[
            "chord_patient_id"
        ].isin(
            set(
                crosswalk[
                    "chord_patient_id"
                ]
            )
        )
    ].copy()

    ###########################################################################
    # Day alignment diagnostics.
    ###########################################################################

    day_diag = nearest_day_diagnostics(
        observed_msk,
        chord,
    )

    atomic_parquet(
        out
        / "patient_day_alignment.parquet",
        day_diag[
            "patient_frame"
        ],
    )

    del day_diag[
        "patient_frame"
    ]

    ###########################################################################
    # Episode matching at exact / 3 / 7 days.
    ###########################################################################

    match0 = match_episodes(
        observed_msk,
        chord,
        tolerance=0,
    )

    match3 = match_episodes(
        observed_msk,
        chord,
        tolerance=3,
    )

    match7 = match_episodes(
        observed_msk,
        chord,
        tolerance=7,
    )

    atomic_parquet(
        out
        / "msk_bridge_matches_exact.parquet",
        match0,
    )

    atomic_parquet(
        out
        / "msk_bridge_matches_within3.parquet",
        match3,
    )

    atomic_parquet(
        out
        / "msk_bridge_matches_within7.parquet",
        match7,
    )

    def matching_summary(
        frame: pd.DataFrame,
    ) -> dict[str, Any]:

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
                        "chord_patient_id"
                    ].nunique()
                )
                if len(
                    frame
                )
                else 0,

            "fraction_of_bpc_observed_episodes":
                float(
                    len(
                        frame
                    )
                    / max(
                        len(
                            observed_msk
                        ),
                        1,
                    )
                ),

            "day_delta":
                (
                    quantiles(
                        frame[
                            "abs_day_delta"
                        ]
                    )
                    if len(
                        frame
                    )
                    else {
                        "n": 0,
                    }
                ),
        }

    matching = {
        "exact":
            matching_summary(
                match0
            ),

        "within3":
            matching_summary(
                match3
            ),

        "within7":
            matching_summary(
                match7
            ),
    }

    ###########################################################################
    # State agreement.
    ###########################################################################

    state0 = state_summary(
        match0
    )

    state3 = state_summary(
        match3
    )

    state7 = state_summary(
        match7
    )

    ###########################################################################
    # Treatment-line agreement.
    #
    # BPC CKPT7A3 already generated two candidate line assignments:
    # active interval and latest started. Compare both without selecting based
    # on external outcomes.
    ###########################################################################

    line0 = compare_lines(
        match0,
        episodes,
    )

    line3 = compare_lines(
        match3,
        episodes,
    )

    line7 = compare_lines(
        match7,
        episodes,
    )

    ###########################################################################
    # Frozen CKPT5 current scan feature bridge.
    ###########################################################################

    (
        ckpt5_index,
        ckpt5_matrix,
        ckpt5_names,
    ) = load_ckpt5_reference(
        repo
    )

    if ckpt5_names != feature_names:

        raise RuntimeError(
            "Frozen CKPT5 feature-name mismatch."
        )

    feature0 = compare_scan_features(
        match0,
        bpc_matrix,
        ckpt5_index,
        ckpt5_matrix,
        feature_names,
    )

    feature3 = compare_scan_features(
        match3,
        bpc_matrix,
        ckpt5_index,
        ckpt5_matrix,
        feature_names,
    )

    ###########################################################################
    # Diagnostic interpretation.
    #
    # This checkpoint intentionally uses broad semantic-fidelity gates.
    # If they fail, we inspect the failed channel rather than modifying the
    # frozen model.
    ###########################################################################

    enough_identity_support = (
        observed_msk[
            "chord_patient_id"
        ].nunique()
        >= 350
    )

    enough_episode_overlap = (
        matching[
            "within3"
        ][
            "patients"
        ]
        >= 300
        and matching[
            "within3"
        ][
            "fraction_of_bpc_observed_episodes"
        ]
        >= 0.50
    )

    state_ok = (
        state3.get(
            "agreement",
            0.0,
        )
        >= 0.80
    )

    line_agreements = [
        value.get(
            "agreement"
        )
        for value in line3.values()
        if (
            value.get(
                "agreement"
            )
            is not None
            and np.isfinite(
                value.get(
                    "agreement"
                )
            )
        )
    ]

    best_line_agreement = (
        max(
            line_agreements
        )
        if line_agreements
        else None
    )

    line_ok = (
        best_line_agreement
        is not None
        and best_line_agreement
        >= 0.85
    )

    feature_rows = int(
        feature3.get(
            "matched_rows",
            0,
        )
    )

    feature_mad = feature3.get(
        "mean_absolute_difference"
    )

    feature_ok = (
        feature_rows
        >= 500
        and feature_mad
        is not None
        and np.isfinite(
            feature_mad
        )
        and feature_mad
        <= 0.15
    )

    gates = {
        "identity_support_ok":
            enough_identity_support,

        "episode_overlap_ok":
            enough_episode_overlap,

        "scan_state_semantics_ok":
            state_ok,

        "treatment_line_semantics_ok":
            line_ok,

        "current_scan_feature_surface_ok":
            feature_ok,
    }

    if all(
        gates.values()
    ):

        status = (
            "PASS_MSK_SEMANTIC_BRIDGE"
        )

    else:

        status = (
            "MSK_SEMANTIC_BRIDGE_REVIEW_REQUIRED"
        )

    ###########################################################################
    # Select a line mapping ONLY for subsequent predictor construction.
    #
    # This selection is based on MSK development positive-control agreement,
    # never on DFCI/VICC outcomes.
    ###########################################################################

    selected_line_strategy = None

    finite_lines = {
        key:
            value[
                "agreement"
            ]
        for key, value in line3.items()
        if (
            value.get(
                "agreement"
            )
            is not None
            and np.isfinite(
                value[
                    "agreement"
                ]
            )
        )
    }

    if finite_lines:

        selected_line_strategy = max(
            finite_lines,
            key=finite_lines.get,
        )

    ###########################################################################
    # Persist final bridge report.
    ###########################################################################

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "identity_bridge": {
            "source":
                "artifacts/checkpoint7a3c/msk_patient_crosswalk.parquet",

            "patients":
                416,

            "mapping_rule":
                (
                    "Persisted one-to-one crosswalk from CKPT7A3C; "
                    "no runtime fuzzy matching."
                ),
        },

        "positive_control_population": {
            "bridged_msk_patients_with_w3":
                int(
                    msk[
                        "chord_patient_id"
                    ].nunique()
                ),

            "all_bpc_w3_episodes":
                int(
                    len(
                        msk
                    )
                ),

            "observed_state_bpc_w3_episodes":
                int(
                    len(
                        observed_msk
                    )
                ),

            "unobserved_state_bpc_w3_episodes":
                int(
                    (
                        msk[
                            "scan_state"
                        ]
                        == "UNOBSERVED"
                    ).sum()
                ),

            "chord_w3_episodes":
                int(
                    len(
                        chord
                    )
                ),
        },

        "day_alignment":
            day_diag,

        "episode_matching":
            matching,

        "scan_state_agreement": {
            "exact":
                state0,

            "within3":
                state3,

            "within7":
                state7,
        },

        "treatment_line_agreement": {
            "exact":
                line0,

            "within3":
                line3,

            "within7":
                line7,

            "best_within3_agreement":
                best_line_agreement,

            "selected_strategy":
                selected_line_strategy,
        },

        "current_scan_feature_bridge": {
            "exact":
                feature0,

            "within3":
                feature3,
        },

        "gates":
            gates,

        "next_action":
            (
                "Freeze the validated external semantic adapter and build "
                "outcome-blinded DFCI/VICC model tensors."
                if status
                == "PASS_MSK_SEMANTIC_BRIDGE"
                else
                "Inspect only the failed MSK positive-control channel(s) "
                "before constructing DFCI/VICC tensors. Do not alter the "
                "frozen model or open external outcomes."
            ),
    }

    atomic_json(
        out
        / "bridge_report.json",
        report,
    )

    ###########################################################################
    # Adapter manifest.
    ###########################################################################

    adapter_manifest = {
        "identity_policy":
            (
                "DFCI/VICC do not use the MSK prefix bridge. "
                "The CKPT7A3C bridge is development positive-control only."
            ),

        "scan_window_days":
            3,

        "scan_state_source":
            "CURATED_CANCER_STATUS",

        "scan_state_crosswalk":
            semantic_crosswalk[
                "row_state_crosswalk"
            ],

        "missing_scan_state_policy":
            semantic_crosswalk[
                "missing_curated_status_policy"
            ],

        "episode_state_precedence":
            semantic_crosswalk[
                "episode_state_precedence"
            ],

        "selected_treatment_line_strategy":
            selected_line_strategy,

        "current_scan_feature_names":
            feature_names,

        "msk_positive_control_status":
            status,

        "external_outcomes_opened":
            False,
    }

    atomic_json(
        out
        / "external_adapter_manifest.json",
        adapter_manifest,
    )

    ###########################################################################
    # Human-readable audit.
    ###########################################################################

    audit = f"""# CKPT7A3D — MSK semantic bridge using frozen identity crosswalk

Status: **{status}**

External outcomes opened: **NO**

## Identity bridge

- Frozen CKPT7A3C crosswalk: 416 patients
- Runtime fuzzy matching: NO
- Bridged MSK patients represented in BPC W3: {msk['chord_patient_id'].nunique()}

## BPC W3 on bridged MSK patients

- all episodes: {len(msk)}
- observed-state episodes: {len(observed_msk)}
- unobserved-state episodes: {(msk['scan_state'] == 'UNOBSERVED').sum()}

## Date alignment

{day_diag}

## Episode matching

{matching}

## Scan-state agreement

Exact:
{state0}

Within 3 days:
{state3}

Within 7 days:
{state7}

## Treatment-line agreement

{report['treatment_line_agreement']}

## Frozen 18-D scan feature bridge

Exact:
{{k: v for k, v in feature0.items() if k != 'per_feature'}}

Within 3 days:
{{k: v for k, v in feature3.items() if k != 'per_feature'}}

## Gates

{gates}

## Policy

No DFCI/VICC survival, death, censoring, PFS, progression-outcome,
or treatment-switch outcome values were accessed.
"""

    atomic_text(
        out
        / "audit.md",
        audit,
    )

    ###########################################################################
    # Handoff.
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A3D",

        "name":
            "msk_semantic_bridge_with_frozen_identity_crosswalk",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "bridged_patients":
            int(
                msk[
                    "chord_patient_id"
                ].nunique()
            ),

        "episode_matching_within3":
            matching[
                "within3"
            ],

        "scan_state_within3":
            state3,

        "best_line_agreement":
            best_line_agreement,

        "selected_line_strategy":
            selected_line_strategy,

        "scan_feature_bridge":
            {
                key:
                    value
                for key, value
                in feature3.items()
                if key != "per_feature"
            },

        "gates":
            gates,

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A3D.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A3D SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "bridged_msk_patients="
        f"{msk['chord_patient_id'].nunique()}"
    )

    print(
        "bpc_w3_all="
        f"{len(msk)}"
    )

    print(
        "bpc_w3_observed_state="
        f"{len(observed_msk)}"
    )

    print(
        "bpc_w3_unobserved_state="
        f"{int((msk['scan_state'] == 'UNOBSERVED').sum())}"
    )

    print(
        "day_alignment="
        f"{day_diag}"
    )

    print(
        "episode_matching="
        f"{matching}"
    )

    print(
        "state_within3="
        f"{state3}"
    )

    print(
        "best_line_agreement="
        f"{best_line_agreement}"
    )

    print(
        "selected_line_strategy="
        f"{selected_line_strategy}"
    )

    print(
        "scan_feature_within3="
        f"{ {k:v for k,v in feature3.items() if k != 'per_feature'} }"
    )

    print(
        "gates="
        f"{gates}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A3D.json"
    )

    print(
        "========== CKPT7A3D SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A3D DECISION PACKET =========="
    )

    print(
        "state_exact="
        f"{state0}"
    )

    print(
        "state_within3="
        f"{state3}"
    )

    print(
        "state_within7="
        f"{state7}"
    )

    print(
        "line_exact="
        f"{line0}"
    )

    print(
        "line_within3="
        f"{line3}"
    )

    print(
        "line_within7="
        f"{line7}"
    )

    print(
        "scan_feature_exact_summary="
        f"{ {k:v for k,v in feature0.items() if k != 'per_feature'} }"
    )

    print(
        "scan_feature_within3_summary="
        f"{ {k:v for k,v in feature3.items() if k != 'per_feature'} }"
    )

    print(
        "scan_feature_per_feature="
        f"{feature3.get('per_feature')}"
    )

    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A3D DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
