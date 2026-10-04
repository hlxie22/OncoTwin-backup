#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
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

TEXT_SUFFIXES = {
    ".txt",
    ".tsv",
    ".csv",
    ".maf",
}

SAMPLE_ID_NAMES = {
    "sample_id",
    "tumor_sample_barcode",
    "specimen_id",
    "genie_sample_id",
    "dmp_sample_id",
}

TIME_EXACT = {
    "start_date",
    "stop_date",
    "seq_date",
    "cpt_seq_date",
    "sequencing_date",
    "specimen_date",
    "sample_collection_date",
    "collection_date",
    "procedure_date",
    "report_date",
    "study_date",
    "assessment_date",
    "event_date",
    "date",
}

FEATURE_TOKENS = {
    "state_non_progressive",
    "state_indeterminate",
    "state_progressive",
    "raw_coverage_chest",
    "raw_coverage_abdomen",
    "raw_coverage_pelvis",
    "raw_coverage_head",
    "raw_coverage_other",
    "modality_ct",
    "modality_pet",
    "modality_mr",
    "modality_bone_scan",
    "site_bone",
    "site_liver",
    "site_lung",
    "site_brain",
    "site_lymph",
    "site_pleura",
}


###############################################################################
# Generic utilities
###############################################################################


def norm(
    value: Any,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def clean(
    value: Any,
) -> str:

    return str(
        value
    ).strip().upper()


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

    tmp.replace(
        path
    )


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

    tmp.replace(
        path
    )


def find_col(
    columns: Iterable[str],
    candidates: Iterable[str],
    *,
    required: bool = False,
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
            f"Unable to resolve {list(candidates)}; "
            f"available={list(columns)}"
        )

    return None


def delimiter_for(
    path: Path,
) -> str:

    return (
        ","
        if path.suffix.lower()
        == ".csv"
        else "\t"
    )


def read_header(
    path: Path,
) -> tuple[
    list[str],
    str,
]:

    preferred = delimiter_for(
        path
    )

    candidates = [
        preferred,
        "," if preferred == "\t" else "\t",
    ]

    best_columns = []
    best_separator = preferred

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
            str(column)
            for column in frame.columns
        ]

        if len(columns) > len(best_columns):

            best_columns = columns
            best_separator = separator

    return (
        best_columns,
        best_separator,
    )


def parse_inventory_columns(
    value: Any,
) -> list[str]:

    if isinstance(
        value,
        list,
    ):

        return [
            str(x)
            for x in value
        ]

    if isinstance(
        value,
        str,
    ):

        try:

            parsed = json.loads(
                value
            )

            if isinstance(
                parsed,
                list,
            ):

                return [
                    str(x)
                    for x in parsed
                ]

        except Exception:

            pass

    return []


def sample_columns(
    columns: Iterable[str],
) -> list[str]:

    result = []

    for column in columns:

        n = norm(
            column
        )

        if (
            n in SAMPLE_ID_NAMES
            or (
                "sample"
                in n
                and n.endswith(
                    "_id"
                )
            )
            or "sample_barcode"
            in n
            or "specimen_id"
            in n
        ):

            result.append(
                column
            )

    return result


def time_columns(
    columns: Iterable[str],
) -> list[str]:

    result = []

    for column in columns:

        n = norm(
            column
        )

        if (
            n in TIME_EXACT
            or n.endswith(
                "_date"
            )
            or n.endswith(
                "_day"
            )
            or n.startswith(
                "date_"
            )
        ):

            result.append(
                column
            )

    return result


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
                np.min(array)
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
                np.max(array)
            ),
    }


###############################################################################
# Import BPC outcome guard
###############################################################################


def import_ckpt7a2(
    repo: Path,
):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a2_external_semantic_audit.py"
    )

    spec = importlib.util.spec_from_file_location(
        "ckpt7a2_anchor_matrix",
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            "Unable to import CKPT7A2."
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
# Collapse one sample/time field to one unique numeric value.
###############################################################################


def unique_sample_time(
    frame: pd.DataFrame,
    sample_col: str,
    time_col: str,
    allowed_samples: set[str],
) -> pd.DataFrame:

    sample = (
        frame[
            sample_col
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    value = pd.to_numeric(
        frame[
            time_col
        ],
        errors="coerce",
    )

    valid = (
        sample.isin(
            allowed_samples
        )
        & value.notna()
    )

    if not valid.any():

        return pd.DataFrame(
            columns=[
                "sample_id",
                "day",
            ]
        )

    work = pd.DataFrame(
        {
            "sample_id":
                sample.loc[
                    valid
                ],

            "day":
                value.loc[
                    valid
                ].astype(float),
        }
    )

    nunique = (
        work.groupby(
            "sample_id",
            observed=True,
        )[
            "day"
        ]
        .nunique()
    )

    unique_samples = set(
        nunique[
            nunique
            == 1
        ].index
    )

    work = work[
        work[
            "sample_id"
        ].isin(
            unique_samples
        )
    ]

    work = (
        work
        .drop_duplicates(
            [
                "sample_id",
                "day",
            ]
        )
        .drop_duplicates(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    return work


###############################################################################
# BPC exhaustive sample-linked clock candidates
###############################################################################


def collect_bpc_sample_times(
    repo: Path,
    bpc_root: Path,
    sample_crosswalk: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    inventory = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a1"
        / "schema_inventory.parquet"
    )

    ckpt7a2 = import_ckpt7a2(
        repo
    )

    guard = ckpt7a2.AccessGuard(
        bpc_root,
        inventory,
    )

    bpc_samples = set(
        sample_crosswalk[
            "bpc_sample_id"
        ].map(
            clean
        )
    )

    sample_map = (
        sample_crosswalk[
            [
                "bpc_patient_id",
                "bpc_sample_id",
                "chord_patient_id",
                "chord_sample_id",
            ]
        ]
        .copy()
    )

    for column in sample_map.columns:

        sample_map[
            column
        ] = sample_map[
            column
        ].map(
            clean
        )

    sources: dict[
        str,
        pd.DataFrame,
    ] = {}

    source_audit = []

    for _, row in inventory.iterrows():

        if bool(
            row[
                "hard_outcome_sensitive"
            ]
        ):

            continue

        relative = str(
            row[
                "relative_path"
            ]
        )

        columns = parse_inventory_columns(
            row[
                "columns"
            ]
        )

        samples = sample_columns(
            columns
        )

        times = time_columns(
            columns
        )

        if (
            not samples
            or not times
        ):

            continue

        #######################################################################
        # Usually one canonical sample ID. If several exist, test each.
        #######################################################################

        for sample_col in samples:

            usecols = list(
                dict.fromkeys(
                    [
                        sample_col,
                        *times,
                    ]
                )
            )

            try:

                frame = guard.read_predictor(
                    relative,
                    usecols=usecols,
                )

            except Exception as exc:

                source_audit.append(
                    {
                        "dataset":
                            "BPC",

                        "relative_path":
                            relative,

                        "sample_column":
                            sample_col,

                        "status":
                            "READ_FAILED",

                        "error":
                            f"{type(exc).__name__}: {exc}",
                    }
                )

                continue

            for time_col in times:

                unique = unique_sample_time(
                    frame,
                    sample_col,
                    time_col,
                    bpc_samples,
                )

                if len(unique) < 5:

                    continue

                unique = unique.rename(
                    columns={
                        "sample_id":
                            "bpc_sample_id",

                        "day":
                            "bpc_day",
                    }
                )

                unique = unique.merge(
                    sample_map,
                    on="bpc_sample_id",
                    how="inner",
                    validate="one_to_one",
                )

                key = (
                    relative
                    + "::"
                    + sample_col
                    + "::"
                    + time_col
                )

                sources[
                    key
                ] = unique

                source_audit.append(
                    {
                        "dataset":
                            "BPC",

                        "relative_path":
                            relative,

                        "sample_column":
                            sample_col,

                        "time_column":
                            time_col,

                        "usable_samples":
                            int(
                                len(unique)
                            ),

                        "usable_patients":
                            int(
                                unique[
                                    "chord_patient_id"
                                ].nunique()
                            ),

                        "status":
                            "OK",
                    }
                )

    return (
        sources,
        source_audit,
        guard.access_log,
    )


###############################################################################
# CHORD exhaustive sample-linked clock candidates
###############################################################################


def collect_chord_sample_times(
    chord_root: Path,
    sample_crosswalk: pd.DataFrame,
) -> tuple[
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
]:

    chord_samples = set(
        sample_crosswalk[
            "chord_sample_id"
        ].map(
            clean
        )
    )

    sources: dict[
        str,
        pd.DataFrame,
    ] = {}

    audit = []

    for path in sorted(
        chord_root.rglob(
            "*"
        )
    ):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in TEXT_SUFFIXES
        ):

            continue

        columns, separator = read_header(
            path
        )

        samples = sample_columns(
            columns
        )

        times = time_columns(
            columns
        )

        if (
            not samples
            or not times
        ):

            continue

        for sample_col in samples:

            usecols = list(
                dict.fromkeys(
                    [
                        sample_col,
                        *times,
                    ]
                )
            )

            try:

                frame = pd.read_csv(
                    path,
                    sep=separator,
                    comment="#",
                    dtype=str,
                    keep_default_na=False,
                    low_memory=False,
                    usecols=usecols,
                )

            except Exception as exc:

                audit.append(
                    {
                        "dataset":
                            "CHORD",

                        "relative_path":
                            str(
                                path.relative_to(
                                    chord_root
                                )
                            ),

                        "sample_column":
                            sample_col,

                        "status":
                            "READ_FAILED",

                        "error":
                            f"{type(exc).__name__}: {exc}",
                    }
                )

                continue

            for time_col in times:

                unique = unique_sample_time(
                    frame,
                    sample_col,
                    time_col,
                    chord_samples,
                )

                if len(unique) < 5:

                    continue

                unique = unique.rename(
                    columns={
                        "sample_id":
                            "chord_sample_id",

                        "day":
                            "chord_day",
                    }
                )

                key = (
                    str(
                        path.relative_to(
                            chord_root
                        )
                    )
                    + "::"
                    + sample_col
                    + "::"
                    + time_col
                )

                sources[
                    key
                ] = unique

                audit.append(
                    {
                        "dataset":
                            "CHORD",

                        "relative_path":
                            str(
                                path.relative_to(
                                    chord_root
                                )
                            ),

                        "sample_column":
                            sample_col,

                        "time_column":
                            time_col,

                        "usable_samples":
                            int(
                                len(unique)
                            ),

                        "status":
                            "OK",
                    }
                )

    return (
        sources,
        audit,
    )


###############################################################################
# Scan references for positive-control alignment
###############################################################################


def load_scan_references(
    repo: Path,
    patient_crosswalk: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    bpc = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3"
        / "bpc_w3_predictor_episodes.parquet"
    )

    bpc = bpc[
        (
            bpc[
                "site"
            ]
            == "MSK"
        )
        & bpc[
            "scan_state"
        ].isin(
            OBSERVED_STATES
        )
    ].copy()

    bpc = bpc.merge(
        patient_crosswalk[
            [
                "bpc_patient_id",
                "chord_patient_id",
            ]
        ],
        left_on="patient_id",
        right_on="bpc_patient_id",
        how="inner",
        validate="many_to_one",
    )

    chord_raw = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    patient_col = find_col(
        chord_raw.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
        required=True,
    )

    day_col = find_col(
        chord_raw.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
        required=True,
    )

    state_col = find_col(
        chord_raw.columns,
        [
            "progression_state_3",
            "scan_state",
        ],
        required=True,
    )

    chord = pd.DataFrame(
        {
            "chord_patient_id":
                chord_raw[
                    patient_col
                ]
                .astype(str)
                .str.strip(),

            "chord_episode_day":
                pd.to_numeric(
                    chord_raw[
                        day_col
                    ],
                    errors="coerce",
                ),

            "chord_state":
                chord_raw[
                    state_col
                ]
                .astype(str)
                .str.strip()
                .str.upper(),
        }
    )

    chord = chord[
        chord[
            "chord_patient_id"
        ].isin(
            set(
                patient_crosswalk[
                    "chord_patient_id"
                ]
            )
        )
    ].copy()

    return (
        bpc,
        chord,
    )


###############################################################################
# Alignment metrics on the same anchored patient subset
###############################################################################


def nearest_metrics(
    bpc: pd.DataFrame,
    chord: pd.DataFrame,
    day_column: str,
) -> dict[str, Any]:

    distances = []

    for patient, group in bpc.groupby(
        "chord_patient_id",
        observed=True,
    ):

        right = chord.loc[
            chord[
                "chord_patient_id"
            ]
            == patient,
            "chord_episode_day",
        ]

        right = pd.to_numeric(
            right,
            errors="coerce",
        ).dropna()

        if right.empty:

            continue

        right_values = right.to_numpy(
            dtype=float
        )

        left = pd.to_numeric(
            group[
                day_column
            ],
            errors="coerce",
        ).dropna()

        for day in left:

            distances.append(
                float(
                    np.min(
                        np.abs(
                            right_values
                            - float(day)
                        )
                    )
                )
            )

    array = np.asarray(
        distances,
        dtype=float,
    )

    if len(array) == 0:

        return {
            "n": 0,
            "within0": 0,
            "within3": 0,
            "within7": 0,
            "within30": 0,
            "distance":
                {
                    "n": 0,
                },
        }

    return {
        "n":
            int(
                len(array)
            ),

        "within0":
            int(
                (
                    array
                    <= 0
                ).sum()
            ),

        "within3":
            int(
                (
                    array
                    <= 3
                ).sum()
            ),

        "within7":
            int(
                (
                    array
                    <= 7
                ).sum()
            ),

        "within30":
            int(
                (
                    array
                    <= 30
                ).sum()
            ),

        "distance":
            quantiles(
                array
            ),
    }


def greedy_state_match(
    bpc: pd.DataFrame,
    chord: pd.DataFrame,
    day_column: str,
    tolerance: float = 3.0,
) -> dict[str, Any]:

    rows = []

    for patient, left in bpc.groupby(
        "chord_patient_id",
        observed=True,
    ):

        right = chord[
            chord[
                "chord_patient_id"
            ]
            == patient
        ].copy()

        if right.empty:
            continue

        candidates = []

        for li, lrow in left.iterrows():

            lday = pd.to_numeric(
                pd.Series(
                    [
                        lrow[
                            day_column
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

            if not np.isfinite(
                lday
            ):
                continue

            for ri, rrow in right.iterrows():

                rday = pd.to_numeric(
                    pd.Series(
                        [
                            rrow[
                                "chord_episode_day"
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[
                    0
                ]

                if not np.isfinite(
                    rday
                ):
                    continue

                delta = abs(
                    float(lday)
                    - float(rday)
                )

                if delta <= tolerance:

                    candidates.append(
                        (
                            delta,
                            li,
                            ri,
                        )
                    )

        candidates.sort()

        used_left = set()
        used_right = set()

        for delta, li, ri in candidates:

            if (
                li in used_left
                or ri in used_right
            ):
                continue

            used_left.add(
                li
            )

            used_right.add(
                ri
            )

            lrow = left.loc[
                li
            ]

            rrow = right.loc[
                ri
            ]

            rows.append(
                {
                    "patient":
                        patient,

                    "bpc_state":
                        lrow[
                            "scan_state"
                        ],

                    "chord_state":
                        rrow[
                            "chord_state"
                        ],

                    "delta":
                        float(
                            delta
                        ),
                }
            )

    if not rows:

        return {
            "rows": 0,
            "patients": 0,
            "agreement": None,
        }

    frame = pd.DataFrame(
        rows
    )

    return {
        "rows":
            int(
                len(frame)
            ),

        "patients":
            int(
                frame[
                    "patient"
                ].nunique()
            ),

        "agreement":
            float(
                (
                    frame[
                        "bpc_state"
                    ]
                    == frame[
                        "chord_state"
                    ]
                ).mean()
            ),
    }


###############################################################################
# Pair every BPC temporal sample field against every CHORD temporal sample field
###############################################################################


def evaluate_anchor_pairs(
    bpc_sources: dict[str, pd.DataFrame],
    chord_sources: dict[str, pd.DataFrame],
    bpc_scans: pd.DataFrame,
    chord_scans: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for bpc_key, bpc_time in bpc_sources.items():

        for chord_key, chord_time in chord_sources.items():

            merged = bpc_time.merge(
                chord_time,
                on="chord_sample_id",
                how="inner",
                validate="one_to_one",
            )

            if len(
                merged
            ) < 20:

                continue

            merged[
                "clock_offset"
            ] = (
                merged[
                    "chord_day"
                ]
                - merged[
                    "bpc_day"
                ]
            )

            offsets = (
                merged.groupby(
                    [
                        "bpc_patient_id",
                        "chord_patient_id",
                    ],
                    observed=True,
                )[
                    "clock_offset"
                ]
                .median()
                .rename(
                    "clock_offset"
                )
                .reset_index()
            )

            if len(
                offsets
            ) < 20:

                continue

            patient_set = set(
                offsets[
                    "chord_patient_id"
                ]
            )

            left = bpc_scans[
                bpc_scans[
                    "chord_patient_id"
                ].isin(
                    patient_set
                )
            ].copy()

            right = chord_scans[
                chord_scans[
                    "chord_patient_id"
                ].isin(
                    patient_set
                )
            ].copy()

            left = left.merge(
                offsets,
                on=[
                    "bpc_patient_id",
                    "chord_patient_id",
                ],
                how="inner",
                validate="many_to_one",
            )

            left[
                "aligned_day"
            ] = (
                left[
                    "episode_end_day"
                ].astype(float)
                + left[
                    "clock_offset"
                ].astype(float)
            )

            baseline = nearest_metrics(
                left,
                right,
                "episode_end_day",
            )

            aligned = nearest_metrics(
                left,
                right,
                "aligned_day",
            )

            state_aligned = greedy_state_match(
                left,
                right,
                "aligned_day",
                tolerance=3.0,
            )

            baseline3 = int(
                baseline[
                    "within3"
                ]
            )

            aligned3 = int(
                aligned[
                    "within3"
                ]
            )

            gain3 = (
                aligned3
                - baseline3
            )

            ratio3 = float(
                aligned3
                / max(
                    baseline3,
                    1,
                )
            )

            rows.append(
                {
                    "bpc_anchor":
                        bpc_key,

                    "chord_anchor":
                        chord_key,

                    "matched_samples":
                        int(
                            merged[
                                "chord_sample_id"
                            ].nunique()
                        ),

                    "matched_patients":
                        int(
                            offsets[
                                "chord_patient_id"
                            ].nunique()
                        ),

                    "offset_median":
                        float(
                            offsets[
                                "clock_offset"
                            ].median()
                        ),

                    "offset_p25":
                        float(
                            offsets[
                                "clock_offset"
                            ].quantile(
                                0.25
                            )
                        ),

                    "offset_p75":
                        float(
                            offsets[
                                "clock_offset"
                            ].quantile(
                                0.75
                            )
                        ),

                    "baseline_episode_n":
                        int(
                            baseline[
                                "n"
                            ]
                        ),

                    "baseline_within3":
                        baseline3,

                    "baseline_within7":
                        int(
                            baseline[
                                "within7"
                            ]
                        ),

                    "baseline_median_nearest":
                        (
                            baseline[
                                "distance"
                            ].get(
                                "median"
                            )
                        ),

                    "aligned_episode_n":
                        int(
                            aligned[
                                "n"
                            ]
                        ),

                    "aligned_within3":
                        aligned3,

                    "aligned_within7":
                        int(
                            aligned[
                                "within7"
                            ]
                        ),

                    "aligned_within30":
                        int(
                            aligned[
                                "within30"
                            ]
                        ),

                    "aligned_median_nearest":
                        (
                            aligned[
                                "distance"
                            ].get(
                                "median"
                            )
                        ),

                    "within3_gain":
                        int(
                            gain3
                        ),

                    "within3_ratio":
                        ratio3,

                    "aligned_state_rows":
                        int(
                            state_aligned[
                                "rows"
                            ]
                        ),

                    "aligned_state_patients":
                        int(
                            state_aligned[
                                "patients"
                            ]
                        ),

                    "aligned_state_agreement":
                        (
                            state_aligned[
                                "agreement"
                            ]
                        ),
                }
            )

    if not rows:

        return pd.DataFrame()

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        [
            "within3_gain",
            "aligned_within3",
            "matched_patients",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return result


###############################################################################
# AST-based exact builder extraction
###############################################################################


def extract_builder_functions(
    path: Path,
) -> list[dict[str, Any]]:

    source = path.read_text(
        encoding="utf-8"
    )

    lines = source.splitlines()

    tree = ast.parse(
        source
    )

    records = []

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            continue

        if not hasattr(
            node,
            "end_lineno",
        ):

            continue

        start = int(
            node.lineno
        )

        end = int(
            node.end_lineno
        )

        segment = "\n".join(
            lines[
                start - 1:
                end
            ]
        )

        hits = sorted(
            token
            for token
            in FEATURE_TOKENS
            if token
            in segment
        )

        related_hits = sorted(
            {
                token
                for token
                in (
                    "current_scan",
                    "scan_features",
                    "progression_state",
                    "coverage",
                    "modality",
                    "tumor_sites",
                    "site_text",
                    "modality_text",
                    "scan_state",
                )
                if token
                in segment
            }
        )

        if (
            not hits
            and not related_hits
        ):

            continue

        relevant_lines = []

        for line_number in range(
            start,
            end + 1,
        ):

            text = lines[
                line_number - 1
            ]

            lowered = text.lower()

            if any(
                token.lower()
                in lowered
                for token
                in (
                    *FEATURE_TOKENS,
                    "coverage",
                    "modality",
                    "progression_state",
                    "tumor_sites",
                    "scan_state",
                    "site_text",
                    "modality_text",
                )
            ):

                relevant_lines.append(
                    (
                        f"{line_number}: "
                        + text
                    )
                )

        records.append(
            {
                "function":
                    node.name,

                "start_line":
                    start,

                "end_line":
                    end,

                "feature_hits":
                    hits,

                "related_hits":
                    related_hits,

                "feature_hit_count":
                    int(
                        len(
                            hits
                        )
                    ),

                "source":
                    segment,

                "relevant_lines":
                    relevant_lines,
            }
        )

    records.sort(
        key=lambda item: (
            item[
                "feature_hit_count"
            ],
            len(
                item[
                    "related_hits"
                ]
            ),
        ),
        reverse=True,
    )

    return records


def save_builder_audit(
    repo: Path,
    out: Path,
) -> dict[str, Any]:

    targets = {
        "ckpt2":
            repo
            / "scripts"
            / "dynamic_scan"
            / "ckpt2_endpoint_and_baselines.py",

        "ckpt5":
            repo
            / "scripts"
            / "dynamic_scan"
            / "ckpt5_supervised_dynamic_model.py",
    }

    report = {}

    text_chunks = []

    for label, path in targets.items():

        records = extract_builder_functions(
            path
        )

        report[
            label
        ] = [
            {
                key:
                    value
                for key, value
                in record.items()
                if key
                != "source"
            }
            for record
            in records[
                :12
            ]
        ]

        text_chunks.append(
            "\n"
            + "=" * 78
        )

        text_chunks.append(
            (
                f"{label.upper()} FEATURE-BUILDER "
                f"CANDIDATES"
            )
        )

        text_chunks.append(
            "=" * 78
        )

        for record in records[
            :5
        ]:

            text_chunks.append(
                (
                    "\n### "
                    + record[
                        "function"
                    ]
                    + " lines "
                    + str(
                        record[
                            "start_line"
                        ]
                    )
                    + "-"
                    + str(
                        record[
                            "end_line"
                        ]
                    )
                    + "\n"
                )
            )

            text_chunks.append(
                "feature_hits="
                + repr(
                    record[
                        "feature_hits"
                    ]
                )
            )

            text_chunks.append(
                "related_hits="
                + repr(
                    record[
                        "related_hits"
                    ]
                )
            )

            text_chunks.append(
                "\nFULL SOURCE:\n"
            )

            text_chunks.append(
                record[
                    "source"
                ]
            )

    atomic_json(
        out
        / "builder_audit.json",
        report,
    )

    atomic_text(
        out
        / "builder_source.txt",
        "\n".join(
            text_chunks
        )
        + "\n",
    )

    return report


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
        "--chord-root",
        default=(
            "data/external_sources/"
            "msk_chord_2024/raw"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/checkpoint7a3f"
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

    chord_root = (
        repo
        / args.chord_root
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
    # Frozen identity bridge.
    ###########################################################################

    patient_crosswalk = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3c"
        / "msk_patient_crosswalk.parquet"
    )

    sample_crosswalk = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3c"
        / "msk_sample_crosswalk.parquet"
    )

    if (
        len(
            patient_crosswalk
        )
        != 416
        or len(
            sample_crosswalk
        )
        != 416
    ):

        raise RuntimeError(
            "Frozen CKPT7A3C crosswalk changed."
        )

    ###########################################################################
    # Enumerate every outcome-free BPC sample/time field.
    ###########################################################################

    (
        bpc_sources,
        bpc_source_audit,
        bpc_access_log,
    ) = collect_bpc_sample_times(
        repo,
        bpc_root,
        sample_crosswalk,
    )

    ###########################################################################
    # Enumerate every CHORD sample/time field.
    ###########################################################################

    (
        chord_sources,
        chord_source_audit,
    ) = collect_chord_sample_times(
        chord_root,
        sample_crosswalk,
    )

    ###########################################################################
    # Load positive-control scans.
    ###########################################################################

    (
        bpc_scans,
        chord_scans,
    ) = load_scan_references(
        repo,
        patient_crosswalk,
    )

    ###########################################################################
    # Exhaustive cross-source anchor matrix.
    ###########################################################################

    anchor_pairs = evaluate_anchor_pairs(
        bpc_sources,
        chord_sources,
        bpc_scans,
        chord_scans,
    )

    if len(
        anchor_pairs
    ):

        atomic_parquet(
            out
            / "anchor_pair_candidates.parquet",
            anchor_pairs,
        )

    else:

        atomic_parquet(
            out
            / "anchor_pair_candidates.parquet",
            pd.DataFrame(
                {
                    "bpc_anchor":
                        pd.Series(
                            dtype=str
                        ),

                    "chord_anchor":
                        pd.Series(
                            dtype=str
                        ),
                }
            ),
        )

    source_audit = pd.DataFrame(
        bpc_source_audit
        + chord_source_audit
    )

    atomic_parquet(
        out
        / "clock_source_inventory.parquet",
        source_audit,
    )

    ###########################################################################
    # Recover exact CKPT2/CKPT5 feature-builder functions.
    ###########################################################################

    builder_report = save_builder_audit(
        repo,
        out,
    )

    ###########################################################################
    # Outcome quarantine assertion.
    ###########################################################################

    for item in bpc_access_log:

        relative = str(
            item[
                "relative_path"
            ]
        ).lower()

        if any(
            token
            in relative
            for token
            in (
                "survival",
                "pfs",
                "censor",
            )
        ):

            raise RuntimeError(
                "Outcome quarantine violation: "
                f"{relative}"
            )

    ###########################################################################
    # Diagnostic status only. No clock mapping is frozen here.
    ###########################################################################

    top = (
        anchor_pairs.iloc[
            0
        ].to_dict()
        if len(
            anchor_pairs
        )
        else None
    )

    if top is None:

        status = (
            "NO_SHARED_SAMPLE_DATE_ANCHOR_FOUND"
        )

    elif (
        int(
            top[
                "matched_patients"
            ]
        )
        >= 50
        and int(
            top[
                "aligned_within3"
            ]
        )
        >= (
            int(
                top[
                    "baseline_within3"
                ]
            )
            + 100
        )
        and float(
            top[
                "within3_ratio"
            ]
        )
        >= 3.0
    ):

        status = (
            "STRONG_CLOCK_BRIDGE_CANDIDATE"
        )

    else:

        status = (
            "CLOCK_ANCHOR_CANDIDATES_READY_FOR_REVIEW"
        )

    ###########################################################################
    # Persistent report.
    ###########################################################################

    top_candidates = (
        anchor_pairs.head(
            25
        ).to_dict(
            orient="records"
        )
        if len(
            anchor_pairs
        )
        else []
    )

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "frozen_identity_patients":
            416,

        "frozen_identity_samples":
            416,

        "bpc_clock_sources":
            int(
                len(
                    bpc_sources
                )
            ),

        "chord_clock_sources":
            int(
                len(
                    chord_sources
                )
            ),

        "anchor_pair_count":
            int(
                len(
                    anchor_pairs
                )
            ),

        "top_anchor":
            top,

        "top_anchor_candidates":
            top_candidates,

        "builder_candidates":
            builder_report,

        "policy": {
            "clock_mapping_frozen":
                False,

            "external_outcomes_accessed":
                False,

            "anchor_search_scope":
                (
                    "all non-outcome BPC sample-linked "
                    "numeric temporal fields crossed against "
                    "all CHORD sample-linked numeric temporal fields"
                ),

            "feature_builder_policy":
                (
                    "recover enclosing CKPT2/CKPT5 source functions "
                    "before implementing any further external "
                    "18-D scan feature transformation"
                ),
        },

        "next_action":
            (
                "Review the leading outcome-free clock-anchor pair and "
                "exact CKPT2/CKPT5 builder source before freezing the "
                "external adapter."
                if len(
                    anchor_pairs
                )
                else
                "Stop trying to establish episode identity from a shared "
                "sample clock. Treat BPC and CHORD imaging as distinct "
                "observational streams and proceed using exact frozen "
                "feature semantics rather than scan-row identity."
            ),
    }

    atomic_json(
        out
        / "diagnostic.json",
        report,
    )

    ###########################################################################
    # Human-readable audit.
    ###########################################################################

    audit_lines = [
        "# CKPT7A3F — exhaustive temporal-anchor and feature-builder audit",
        "",
        f"Status: **{status}**",
        "",
        "External outcomes opened: **NO**",
        "",
        f"BPC sample/time sources: {len(bpc_sources)}",
        f"CHORD sample/time sources: {len(chord_sources)}",
        f"Anchor pairs evaluated: {len(anchor_pairs)}",
        "",
        "## Top clock-anchor candidates",
        "",
    ]

    if len(
        anchor_pairs
    ):

        audit_lines.append(
            anchor_pairs.head(
                20
            ).to_string(
                index=False
            )
        )

    else:

        audit_lines.append(
            "No pair had at least 20 frozen matched samples."
        )

    audit_lines.extend(
        [
            "",
            "## Exact feature-builder source",
            "",
            (
                "Full AST-extracted source is saved in "
                "`artifacts/checkpoint7a3f/builder_source.txt`."
            ),
            "",
        ]
    )

    atomic_text(
        out
        / "audit.md",
        "\n".join(
            audit_lines
        )
        + "\n",
    )

    ###########################################################################
    # Handoff.
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A3F",

        "name":
            "exhaustive_clock_anchor_and_exact_feature_builder_audit",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "anchor_pair_count":
            int(
                len(
                    anchor_pairs
                )
            ),

        "top_anchor":
            top,

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A3F.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A3F SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "bpc_clock_sources="
        f"{len(bpc_sources)}"
    )

    print(
        "chord_clock_sources="
        f"{len(chord_sources)}"
    )

    print(
        "anchor_pair_count="
        f"{len(anchor_pairs)}"
    )

    print(
        "top_anchor="
        f"{top}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A3F.json"
    )

    print(
        "========== CKPT7A3F SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A3F DECISION PACKET =========="
    )

    print("")
    print(
        "----- TOP CLOCK ANCHORS -----"
    )

    if len(
        anchor_pairs
    ):

        print(
            anchor_pairs.head(
                25
            ).to_string(
                index=False
            )
        )

    else:

        print(
            "NO ANCHOR PAIRS"
        )

    print(
        "----- TOP CLOCK ANCHORS END -----"
    )

    print("")
    print(
        "----- EXACT FEATURE BUILDER DIGEST -----"
    )

    for label in (
        "ckpt2",
        "ckpt5",
    ):

        print("")
        print(
            "SOURCE",
            label,
        )

        records = builder_report[
            label
        ]

        for record in records[
            :6
        ]:

            print(
                {
                    "function":
                        record[
                            "function"
                        ],

                    "start_line":
                        record[
                            "start_line"
                        ],

                    "end_line":
                        record[
                            "end_line"
                        ],

                    "feature_hits":
                        record[
                            "feature_hits"
                        ],

                    "related_hits":
                        record[
                            "related_hits"
                        ],

                    "relevant_lines":
                        record[
                            "relevant_lines"
                        ][
                            :60
                        ],
                }
            )

    print(
        "----- EXACT FEATURE BUILDER DIGEST END -----"
    )

    print("")
    print(
        "builder_source="
        "artifacts/checkpoint7a3f/builder_source.txt"
    )

    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A3F DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
