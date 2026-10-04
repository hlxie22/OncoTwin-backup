#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TOP_BPC_ANCHOR = (
    "cBioPortal_files/"
    "data_timeline_sample_acquisition.txt"
    "::SAMPLE_ID::START_DATE"
)

TOP_CHORD_ANCHOR = (
    "msk_chord_2024/"
    "data_timeline_specimen_surgery.txt"
    "::SAMPLE_ID::START_DATE"
)

EXCLUDED_EVENT_BASENAME_TOKENS = {
    "imaging",
    "progression",
    "survival",
    "sample",
    "specimen",
    "sequencing",
    "cancer_presence",
    "tumor_sites",
}


###############################################################################
# Utilities
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

        "p95":
            float(
                np.quantile(
                    array,
                    0.95,
                )
            ),

        "max":
            float(
                array.max()
            ),
    }


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


###############################################################################
# Recover candidate clock using CKPT7A3F's exact discovery machinery.
###############################################################################


def build_frozen_clock_candidate(
    repo: Path,
    bpc_root: Path,
    chord_root: Path,
    sample_crosswalk: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    ckpt7a3f = import_module(
        "ckpt7a3f_reuse",
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a3f_exhaustive_anchor_and_builder_audit.py",
    )

    (
        bpc_sources,
        _,
        _,
    ) = ckpt7a3f.collect_bpc_sample_times(
        repo,
        bpc_root,
        sample_crosswalk,
    )

    (
        chord_sources,
        _,
    ) = ckpt7a3f.collect_chord_sample_times(
        chord_root,
        sample_crosswalk,
    )

    if TOP_BPC_ANCHOR not in bpc_sources:

        raise RuntimeError(
            "Frozen top BPC anchor missing."
        )

    if TOP_CHORD_ANCHOR not in chord_sources:

        raise RuntimeError(
            "Frozen top CHORD anchor missing."
        )

    left = bpc_sources[
        TOP_BPC_ANCHOR
    ]

    right = chord_sources[
        TOP_CHORD_ANCHOR
    ]

    merged = left.merge(
        right,
        on="chord_sample_id",
        how="inner",
        validate="one_to_one",
    )

    if (
        merged[
            "chord_sample_id"
        ].nunique()
        != 416
    ):

        raise RuntimeError(
            "Expected 416 matched anchor samples."
        )

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

    if (
        offsets[
            "chord_patient_id"
        ].nunique()
        != 416
    ):

        raise RuntimeError(
            "Expected clock offset for all 416 patients."
        )

    audit = {
        "bpc_anchor":
            TOP_BPC_ANCHOR,

        "chord_anchor":
            TOP_CHORD_ANCHOR,

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

        "offset_distribution":
            quantiles(
                offsets[
                    "clock_offset"
                ]
            ),
    }

    return (
        offsets,
        audit,
    )


###############################################################################
# Independent non-imaging event sources
###############################################################################


def read_header(
    path: Path,
) -> tuple[
    list[str],
    str,
]:

    preferred = (
        ","
        if path.suffix.lower()
        == ".csv"
        else "\t"
    )

    separators = [
        preferred,
        "," if preferred == "\t" else "\t",
    ]

    best = []
    best_separator = preferred

    for separator in separators:

        try:

            frame = pd.read_csv(
                path,
                sep=separator,
                comment="#",
                dtype=str,
                nrows=0,
                low_memory=False,
            )

        except Exception:

            continue

        columns = list(
            frame.columns
        )

        if len(columns) > len(best):

            best = columns
            best_separator = separator

    return (
        best,
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
            str(item)
            for item in value
        ]

    if isinstance(
        value,
        str,
    ):

        try:

            payload = json.loads(
                value
            )

            if isinstance(
                payload,
                list,
            ):
                return [
                    str(item)
                    for item in payload
                ]

        except Exception:

            pass

    return []


def load_independent_event_pairs(
    repo: Path,
    bpc_root: Path,
    chord_root: Path,
    patient_crosswalk: pd.DataFrame,
) -> tuple[
    dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    list[dict[str, Any]],
]:

    inventory = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a1"
        / "schema_inventory.parquet"
    )

    ckpt7a2 = import_module(
        "ckpt7a2_g",
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a2_external_semantic_audit.py",
    )

    guard = ckpt7a2.AccessGuard(
        bpc_root,
        inventory,
    )

    patient_map = (
        patient_crosswalk[
            [
                "bpc_patient_id",
                "chord_patient_id",
            ]
        ]
        .copy()
    )

    patient_map[
        "bpc_patient_id"
    ] = patient_map[
        "bpc_patient_id"
    ].map(
        clean
    )

    patient_map[
        "chord_patient_id"
    ] = patient_map[
        "chord_patient_id"
    ].map(
        clean
    )

    chord_by_basename = {}

    for path in chord_root.rglob(
        "*"
    ):

        if (
            not path.is_file()
            or path.suffix.lower()
            not in {
                ".txt",
                ".tsv",
                ".csv",
            }
        ):
            continue

        chord_by_basename.setdefault(
            path.name,
            [],
        ).append(
            path
        )

    pairs = {}
    audit = []

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

        basename = Path(
            relative
        ).name

        lower = basename.lower()

        if not lower.startswith(
            "data_timeline_"
        ):
            continue

        if any(
            token in lower
            for token
            in EXCLUDED_EVENT_BASENAME_TOKENS
        ):
            continue

        candidates = chord_by_basename.get(
            basename,
            [],
        )

        if len(
            candidates
        ) != 1:

            continue

        columns = parse_inventory_columns(
            row[
                "columns"
            ]
        )

        bpc_patient_col = find_col(
            columns,
            [
                "PATIENT_ID",
            ],
            required=False,
        )

        bpc_start_col = find_col(
            columns,
            [
                "START_DATE",
            ],
            required=False,
        )

        if (
            bpc_patient_col is None
            or bpc_start_col is None
        ):
            continue

        chord_path = candidates[
            0
        ]

        chord_columns, separator = read_header(
            chord_path
        )

        chord_patient_col = find_col(
            chord_columns,
            [
                "PATIENT_ID",
            ],
            required=False,
        )

        chord_start_col = find_col(
            chord_columns,
            [
                "START_DATE",
            ],
            required=False,
        )

        if (
            chord_patient_col is None
            or chord_start_col is None
        ):
            continue

        try:

            bpc = guard.read_predictor(
                relative,
                usecols=[
                    bpc_patient_col,
                    bpc_start_col,
                ],
            )

            chord = pd.read_csv(
                chord_path,
                sep=separator,
                comment="#",
                dtype=str,
                keep_default_na=False,
                low_memory=False,
                usecols=[
                    chord_patient_col,
                    chord_start_col,
                ],
            )

        except Exception as exc:

            audit.append(
                {
                    "basename":
                        basename,

                    "status":
                        "READ_FAILED",

                    "error":
                        f"{type(exc).__name__}: {exc}",
                }
            )

            continue

        left = pd.DataFrame(
            {
                "bpc_patient_id":
                    bpc[
                        bpc_patient_col
                    ].map(
                        clean
                    ),

                "day":
                    pd.to_numeric(
                        bpc[
                            bpc_start_col
                        ],
                        errors="coerce",
                    ),
            }
        )

        left = left.merge(
            patient_map,
            on="bpc_patient_id",
            how="inner",
            validate="many_to_one",
        )

        left = left[
            left[
                "day"
            ].notna()
        ].copy()

        right = pd.DataFrame(
            {
                "chord_patient_id":
                    chord[
                        chord_patient_col
                    ].map(
                        clean
                    ),

                "day":
                    pd.to_numeric(
                        chord[
                            chord_start_col
                        ],
                        errors="coerce",
                    ),
            }
        )

        right = right[
            right[
                "chord_patient_id"
            ].isin(
                set(
                    patient_map[
                        "chord_patient_id"
                    ]
                )
            )
            & right[
                "day"
            ].notna()
        ].copy()

        common_patients = (
            set(
                left[
                    "chord_patient_id"
                ]
            )
            & set(
                right[
                    "chord_patient_id"
                ]
            )
        )

        if len(
            common_patients
        ) < 20:

            continue

        left = left[
            left[
                "chord_patient_id"
            ].isin(
                common_patients
            )
        ]

        right = right[
            right[
                "chord_patient_id"
            ].isin(
                common_patients
            )
        ]

        pairs[
            basename
        ] = (
            left,
            right,
        )

        audit.append(
            {
                "basename":
                    basename,

                "status":
                    "OK",

                "common_patients":
                    int(
                        len(
                            common_patients
                        )
                    ),

                "bpc_rows":
                    int(
                        len(
                            left
                        )
                    ),

                "chord_rows":
                    int(
                        len(
                            right
                        )
                    ),
            }
        )

    ###########################################################################
    # Outcome quarantine assertion.
    ###########################################################################

    for item in guard.access_log:

        relative = str(
            item[
                "relative_path"
            ]
        ).lower()

        if any(
            token in relative
            for token in (
                "survival",
                "pfs",
                "censor",
            )
        ):

            raise RuntimeError(
                "Outcome quarantine violation: "
                f"{relative}"
            )

    return (
        pairs,
        audit,
    )


###############################################################################
# Event-time alignment
###############################################################################


def nearest_distances(
    left: pd.DataFrame,
    right: pd.DataFrame,
    day_column: str,
) -> np.ndarray:

    distances = []

    common = (
        set(
            left[
                "chord_patient_id"
            ]
        )
        & set(
            right[
                "chord_patient_id"
            ]
        )
    )

    for patient in common:

        a = pd.to_numeric(
            left.loc[
                left[
                    "chord_patient_id"
                ]
                == patient,
                day_column,
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        b = pd.to_numeric(
            right.loc[
                right[
                    "chord_patient_id"
                ]
                == patient,
                "day",
            ],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        if (
            len(a) == 0
            or len(b) == 0
        ):
            continue

        for day in a:

            distances.append(
                float(
                    np.min(
                        np.abs(
                            b
                            - day
                        )
                    )
                )
            )

    return np.asarray(
        distances,
        dtype=float,
    )


def metric_summary(
    values: np.ndarray,
) -> dict[str, Any]:

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

        "within0":
            int(
                (
                    values
                    <= 0
                ).sum()
            ),

        "within3":
            int(
                (
                    values
                    <= 3
                ).sum()
            ),

        "within7":
            int(
                (
                    values
                    <= 7
                ).sum()
            ),

        "within30":
            int(
                (
                    values
                    <= 30
                ).sum()
            ),

        "distance":
            quantiles(
                values
            ),
    }


def evaluate_independent_alignment(
    pairs: dict[
        str,
        tuple[
            pd.DataFrame,
            pd.DataFrame,
        ],
    ],
    offsets: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    rows = []

    all_baseline = []
    all_aligned = []

    for basename, (
        left,
        right,
    ) in pairs.items():

        working = left.merge(
            offsets,
            on=[
                "bpc_patient_id",
                "chord_patient_id",
            ],
            how="inner",
            validate="many_to_one",
        )

        working[
            "aligned_day"
        ] = (
            working[
                "day"
            ].astype(float)
            + working[
                "clock_offset"
            ].astype(float)
        )

        baseline = nearest_distances(
            working,
            right,
            "day",
        )

        aligned = nearest_distances(
            working,
            right,
            "aligned_day",
        )

        if len(
            baseline
        ) == 0:

            continue

        all_baseline.extend(
            baseline.tolist()
        )

        all_aligned.extend(
            aligned.tolist()
        )

        base = metric_summary(
            baseline
        )

        post = metric_summary(
            aligned
        )

        rows.append(
            {
                "event_source":
                    basename,

                "patients":
                    int(
                        working[
                            "chord_patient_id"
                        ].nunique()
                    ),

                "events":
                    int(
                        len(
                            baseline
                        )
                    ),

                "baseline_within3":
                    int(
                        base.get(
                            "within3",
                            0,
                        )
                    ),

                "aligned_within3":
                    int(
                        post.get(
                            "within3",
                            0,
                        )
                    ),

                "within3_gain":
                    int(
                        post.get(
                            "within3",
                            0,
                        )
                        - base.get(
                            "within3",
                            0,
                        )
                    ),

                "within3_ratio":
                    float(
                        post.get(
                            "within3",
                            0,
                        )
                        / max(
                            base.get(
                                "within3",
                                0,
                            ),
                            1,
                        )
                    ),

                "baseline_median_nearest":
                    base.get(
                        "distance",
                        {},
                    ).get(
                        "median"
                    ),

                "aligned_median_nearest":
                    post.get(
                        "distance",
                        {},
                    ).get(
                        "median"
                    ),

                "baseline_within30":
                    int(
                        base.get(
                            "within30",
                            0,
                        )
                    ),

                "aligned_within30":
                    int(
                        post.get(
                            "within30",
                            0,
                        )
                    ),
            }
        )

    table = pd.DataFrame(
        rows
    )

    if len(
        table
    ):

        table = table.sort_values(
            [
                "within3_gain",
                "events",
            ],
            ascending=[
                False,
                False,
            ],
        ).reset_index(
            drop=True
        )

    aggregate = {
        "event_sources":
            int(
                len(
                    table
                )
            ),

        "baseline":
            metric_summary(
                np.asarray(
                    all_baseline,
                    dtype=float,
                )
            ),

        "aligned":
            metric_summary(
                np.asarray(
                    all_aligned,
                    dtype=float,
                )
            ),
    }

    if (
        aggregate[
            "baseline"
        ].get(
            "n",
            0,
        )
        > 0
    ):

        aggregate[
            "within3_gain"
        ] = (
            aggregate[
                "aligned"
            ].get(
                "within3",
                0,
            )
            - aggregate[
                "baseline"
            ].get(
                "within3",
                0,
            )
        )

        aggregate[
            "within3_ratio"
        ] = float(
            aggregate[
                "aligned"
            ].get(
                "within3",
                0,
            )
            / max(
                aggregate[
                    "baseline"
                ].get(
                    "within3",
                    0,
                ),
                1,
            )
        )

    return (
        table,
        aggregate,
    )


###############################################################################
# CKPT5 exact scan_features replay
###############################################################################


def normalize_feature_output(
    output: Any,
    expected_names: list[str],
) -> tuple[
    np.ndarray,
    list[str],
]:

    if isinstance(
        output,
        pd.DataFrame,
    ):

        frame = output.copy()

        if set(
            expected_names
        ).issubset(
            frame.columns
        ):

            return (
                frame[
                    expected_names
                ].to_numpy(
                    dtype=np.float32
                ),
                expected_names,
            )

        if frame.shape[
            1
        ] == len(
            expected_names
        ):

            return (
                frame.to_numpy(
                    dtype=np.float32
                ),
                expected_names,
            )

        raise RuntimeError(
            "DataFrame scan_features output does not "
            "match expected frozen feature schema."
        )

    if isinstance(
        output,
        np.ndarray,
    ):

        if (
            output.ndim == 2
            and output.shape[
                1
            ]
            == len(
                expected_names
            )
        ):

            return (
                np.asarray(
                    output,
                    dtype=np.float32,
                ),
                expected_names,
            )

        raise RuntimeError(
            "ndarray scan_features output has wrong shape."
        )

    if isinstance(
        output,
        tuple,
    ):

        for item in output:

            if isinstance(
                item,
                pd.DataFrame,
            ):

                try:

                    return normalize_feature_output(
                        item,
                        expected_names,
                    )

                except Exception:

                    pass

        arrays = [
            item
            for item in output
            if isinstance(
                item,
                np.ndarray,
            )
        ]

        for array in arrays:

            if (
                array.ndim == 2
                and array.shape[
                    1
                ]
                == len(
                    expected_names
                )
            ):

                names = expected_names

                for item in output:

                    if (
                        isinstance(
                            item,
                            (
                                list,
                                tuple,
                            ),
                        )
                        and len(
                            item
                        )
                        == len(
                            expected_names
                        )
                    ):

                        names = [
                            str(
                                value
                            )
                            for value
                            in item
                        ]

                return (
                    np.asarray(
                        array,
                        dtype=np.float32,
                    ),
                    names,
                )

    raise RuntimeError(
        "Unsupported scan_features return type: "
        f"{type(output)}"
    )


def invoke_load_scans(
    module: Any,
    repo: Path,
) -> tuple[
    pd.DataFrame | None,
    dict[str, Any],
]:

    if not hasattr(
        module,
        "load_scans",
    ):

        return (
            None,
            {
                "status":
                    "NO_LOAD_SCANS_FUNCTION",
            },
        )

    function = module.load_scans

    signature = inspect.signature(
        function
    )

    canonical_file = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    attempts = []

    ###########################################################################
    # First: zero-required-argument function.
    ###########################################################################

    required = [
        parameter
        for parameter in signature.parameters.values()
        if (
            parameter.default
            is inspect._empty
            and parameter.kind
            in (
                parameter.POSITIONAL_ONLY,
                parameter.POSITIONAL_OR_KEYWORD,
                parameter.KEYWORD_ONLY,
            )
        )
    ]

    if not required:

        try:

            result = function()

            if isinstance(
                result,
                pd.DataFrame,
            ):

                return (
                    result,
                    {
                        "status":
                            "PASS",

                        "method":
                            "no_args",

                        "signature":
                            str(
                                signature
                            ),
                    },
                )

        except Exception as exc:

            attempts.append(
                "no_args:"
                f"{type(exc).__name__}:{exc}"
            )

    ###########################################################################
    # Second: infer a single path/root parameter.
    ###########################################################################

    if len(
        required
    ) == 1:

        parameter = required[
            0
        ]

        name = norm(
            parameter.name
        )

        values = []

        if any(
            token in name
            for token in (
                "repo",
                "root",
            )
        ):

            values.extend(
                [
                    repo,
                    str(
                        repo
                    ),
                ]
            )

        if any(
            token in name
            for token in (
                "path",
                "file",
                "scan",
            )
        ):

            values.extend(
                [
                    canonical_file,
                    str(
                        canonical_file
                    ),
                ]
            )

        #######################################################################
        # Also attempt canonical file if name is ambiguous.
        #######################################################################

        values.extend(
            [
                canonical_file,
                str(
                    canonical_file
                ),
            ]
        )

        seen = set()

        for value in values:

            marker = repr(
                value
            )

            if marker in seen:
                continue

            seen.add(
                marker
            )

            try:

                result = function(
                    value
                )

                if isinstance(
                    result,
                    pd.DataFrame,
                ):

                    return (
                        result,
                        {
                            "status":
                                "PASS",

                            "method":
                                (
                                    "single_arg:"
                                    + marker
                                ),

                            "signature":
                                str(
                                    signature
                                ),
                        },
                    )

            except Exception as exc:

                attempts.append(
                    marker
                    + ":"
                    + type(
                        exc
                    ).__name__
                    + ":"
                    + str(
                        exc
                    )
                )

    return (
        None,
        {
            "status":
                "LOAD_SCANS_INVOCATION_FAILED",

            "signature":
                str(
                    signature
                ),

            "attempts":
                attempts[
                    :20
                ],
        },
    )


def feature_replay(
    repo: Path,
) -> dict[str, Any]:

    ckpt5 = import_module(
        "ckpt5_replay",
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt5_supervised_dynamic_model.py",
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

    expected_names = list(
        schema[
            "current_scan"
        ]
    )

    frozen_index = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared"
        / "scan_index.parquet"
    )

    frozen_matrix = np.load(
        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared"
        / "current_scan_features_f32.npy",
        mmap_mode="r",
    )

    scan_signature = str(
        inspect.signature(
            ckpt5.scan_features
        )
    )

    scan_source = inspect.getsource(
        ckpt5.scan_features
    )

    scans, load_report = invoke_load_scans(
        ckpt5,
        repo,
    )

    ###########################################################################
    # If load_scans could not be invoked automatically, use canonical parquet
    # directly as a diagnostic fallback.
    ###########################################################################

    source_method = (
        "ckpt5.load_scans"
    )

    if scans is None:

        scans = pd.read_parquet(
            repo
            / "artifacts"
            / "checkpoint1"
            / "canonical"
            / "chord_mbc_scan_episodes_w3.parquet"
        )

        source_method = (
            "canonical_parquet_fallback"
        )

    ###########################################################################
    # Call the exact function only if all required arguments other than scans
    # are optional.
    ###########################################################################

    signature = inspect.signature(
        ckpt5.scan_features
    )

    required = [
        parameter
        for parameter in signature.parameters.values()
        if (
            parameter.default
            is inspect._empty
            and parameter.kind
            in (
                parameter.POSITIONAL_ONLY,
                parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    ]

    if len(
        required
    ) != 1:

        return {
            "status":
                "NEEDS_SCAN_FEATURE_SIGNATURE_RESOLUTION",

            "scan_features_signature":
                scan_signature,

            "load_scans":
                load_report,

            "scan_source_method":
                source_method,

            "scan_features_source":
                scan_source,
        }

    try:

        output = ckpt5.scan_features(
            scans
        )

        replay_matrix, replay_names = normalize_feature_output(
            output,
            expected_names,
        )

    except Exception as exc:

        return {
            "status":
                "SCAN_FEATURE_REPLAY_CALL_FAILED",

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

            "scan_features_signature":
                scan_signature,

            "load_scans":
                load_report,

            "scan_source_method":
                source_method,

            "input_columns":
                list(
                    scans.columns
                ),

            "scan_features_source":
                scan_source,
        }

    if replay_names != expected_names:

        return {
            "status":
                "FEATURE_NAME_MISMATCH",

            "replay_names":
                replay_names,

            "expected_names":
                expected_names,

            "scan_features_signature":
                scan_signature,

            "load_scans":
                load_report,

            "scan_features_source":
                scan_source,
        }

    ###########################################################################
    # Align replay rows to frozen CKPT5 scan index.
    ###########################################################################

    scan_id_scans = find_col(
        scans.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    scan_id_index = find_col(
        frozen_index.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    replay_rows = None
    join_method = None

    if (
        scan_id_scans is not None
        and scan_id_index is not None
    ):

        lookup = pd.DataFrame(
            {
                "scan_episode_id":
                    scans[
                        scan_id_scans
                    ].astype(str),

                "replay_row":
                    np.arange(
                        len(
                            scans
                        ),
                        dtype=int,
                    ),
            }
        )

        target = pd.DataFrame(
            {
                "scan_episode_id":
                    frozen_index[
                        scan_id_index
                    ].astype(str),

                "frozen_row":
                    np.arange(
                        len(
                            frozen_index
                        ),
                        dtype=int,
                    ),
            }
        )

        merged = target.merge(
            lookup,
            on="scan_episode_id",
            how="inner",
            validate="many_to_one",
        )

        if len(
            merged
        ) == len(
            frozen_index
        ):

            replay_rows = merged[
                "replay_row"
            ].to_numpy(
                dtype=int
            )

            frozen_rows = merged[
                "frozen_row"
            ].to_numpy(
                dtype=int
            )

            join_method = (
                "scan_episode_id"
            )

    if replay_rows is None:

        patient_scans = find_col(
            scans.columns,
            [
                "patient_id",
            ],
            required=False,
        )

        day_scans = find_col(
            scans.columns,
            [
                "episode_end_day",
                "landmark_day",
            ],
            required=False,
        )

        patient_index = find_col(
            frozen_index.columns,
            [
                "patient_id",
            ],
            required=False,
        )

        day_index = find_col(
            frozen_index.columns,
            [
                "landmark_day",
                "episode_end_day",
            ],
            required=False,
        )

        if None not in (
            patient_scans,
            day_scans,
            patient_index,
            day_index,
        ):

            lookup = pd.DataFrame(
                {
                    "patient_id":
                        scans[
                            patient_scans
                        ].astype(str),

                    "day":
                        pd.to_numeric(
                            scans[
                                day_scans
                            ],
                            errors="coerce",
                        ),

                    "replay_row":
                        np.arange(
                            len(
                                scans
                            ),
                            dtype=int,
                        ),
                }
            )

            target = pd.DataFrame(
                {
                    "patient_id":
                        frozen_index[
                            patient_index
                        ].astype(str),

                    "day":
                        pd.to_numeric(
                            frozen_index[
                                day_index
                            ],
                            errors="coerce",
                        ),

                    "frozen_row":
                        np.arange(
                            len(
                                frozen_index
                            ),
                            dtype=int,
                        ),
                }
            )

            duplicates = (
                lookup.duplicated(
                    [
                        "patient_id",
                        "day",
                    ],
                    keep=False,
                )
            )

            lookup = lookup[
                ~duplicates
            ]

            merged = target.merge(
                lookup,
                on=[
                    "patient_id",
                    "day",
                ],
                how="inner",
            )

            if len(
                merged
            ) == len(
                frozen_index
            ):

                replay_rows = merged[
                    "replay_row"
                ].to_numpy(
                    dtype=int
                )

                frozen_rows = merged[
                    "frozen_row"
                ].to_numpy(
                    dtype=int
                )

                join_method = (
                    "patient_day"
                )

    if replay_rows is None:

        return {
            "status":
                "FEATURE_ROWS_COULD_NOT_BE_ALIGNED",

            "replay_rows":
                int(
                    len(
                        replay_matrix
                    )
                ),

            "frozen_rows":
                int(
                    len(
                        frozen_matrix
                    )
                ),

            "scan_features_signature":
                scan_signature,

            "load_scans":
                load_report,

            "scan_source_method":
                source_method,

            "scan_features_source":
                scan_source,
        }

    replay = np.asarray(
        replay_matrix[
            replay_rows
        ],
        dtype=np.float32,
    )

    frozen = np.asarray(
        frozen_matrix[
            frozen_rows
        ],
        dtype=np.float32,
    )

    if replay.shape != frozen.shape:

        raise RuntimeError(
            "Aligned replay/frozen feature shape mismatch."
        )

    absolute = np.abs(
        replay
        - frozen
    )

    per_feature = {}

    for index, name in enumerate(
        expected_names
    ):

        per_feature[
            name
        ] = {
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
                        < 1e-7
                    ).mean()
                ),

            "replay_mean":
                float(
                    replay[
                        :,
                        index
                    ].mean()
                ),

            "frozen_mean":
                float(
                    frozen[
                        :,
                        index
                    ].mean()
                ),
        }

    vector_exact = float(
        (
            absolute.max(
                axis=1
            )
            < 1e-7
        ).mean()
    )

    overall_mad = float(
        absolute.mean()
    )

    replay_pass = bool(
        overall_mad
        <= 1e-7
        and vector_exact
        >= 0.999999
    )

    return {
        "status":
            (
                "PASS_EXACT_CKPT5_SCAN_FEATURE_REPLAY"
                if replay_pass
                else
                "CKPT5_SCAN_FEATURE_REPLAY_MISMATCH"
            ),

        "rows":
            int(
                len(
                    replay
                )
            ),

        "features":
            int(
                replay.shape[
                    1
                ]
            ),

        "join_method":
            join_method,

        "scan_source_method":
            source_method,

        "scan_features_signature":
            scan_signature,

        "load_scans":
            load_report,

        "overall_mean_absolute_difference":
            overall_mad,

        "vector_exact_fraction":
            vector_exact,

        "max_absolute_difference":
            float(
                absolute.max()
            ),

        "per_feature":
            per_feature,

        "scan_features_source":
            scan_source,
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
        "--chord-root",
        default=(
            "data/external_sources/"
            "msk_chord_2024/raw"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/checkpoint7a3g"
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
    # Frozen crosswalks.
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
            "Frozen identity crosswalk changed."
        )

    ###########################################################################
    # Reconstruct leading candidate clock exactly.
    ###########################################################################

    offsets, clock_audit = build_frozen_clock_candidate(
        repo,
        bpc_root,
        chord_root,
        sample_crosswalk,
    )

    atomic_parquet(
        out
        / "msk_patient_clock_offsets.parquet",
        offsets,
    )

    ###########################################################################
    # Independently test temporal alignment on non-imaging streams.
    ###########################################################################

    (
        event_pairs,
        event_audit,
    ) = load_independent_event_pairs(
        repo,
        bpc_root,
        chord_root,
        patient_crosswalk,
    )

    (
        event_table,
        event_aggregate,
    ) = evaluate_independent_alignment(
        event_pairs,
        offsets,
    )

    if len(
        event_table
    ):

        atomic_parquet(
            out
            / "independent_event_alignment.parquet",
            event_table,
        )

    else:

        atomic_parquet(
            out
            / "independent_event_alignment.parquet",
            pd.DataFrame(
                {
                    "event_source":
                        pd.Series(
                            dtype=str
                        ),
                }
            ),
        )

    ###########################################################################
    # Replay exact frozen CKPT5 scan feature builder.
    ###########################################################################

    replay = feature_replay(
        repo
    )

    atomic_text(
        out
        / "ckpt5_scan_features_source.py.txt",
        replay.get(
            "scan_features_source",
            "",
        ),
    )

    replay_public = {
        key:
            value
        for key, value
        in replay.items()
        if key
        not in {
            "scan_features_source",
            "per_feature",
        }
    }

    replay_public[
        "per_feature"
    ] = replay.get(
        "per_feature",
        {}
    )

    atomic_json(
        out
        / "feature_replay.json",
        replay_public,
    )

    ###########################################################################
    # Decide whether the clock has independent confirmation.
    ###########################################################################

    independent_sources = int(
        event_aggregate.get(
            "event_sources",
            0,
        )
    )

    baseline3 = int(
        event_aggregate.get(
            "baseline",
            {},
        ).get(
            "within3",
            0,
        )
    )

    aligned3 = int(
        event_aggregate.get(
            "aligned",
            {},
        ).get(
            "within3",
            0,
        )
    )

    baseline_median = (
        event_aggregate.get(
            "baseline",
            {},
        ).get(
            "distance",
            {},
        ).get(
            "median"
        )
    )

    aligned_median = (
        event_aggregate.get(
            "aligned",
            {},
        ).get(
            "distance",
            {},
        ).get(
            "median"
        )
    )

    clock_confirmed = bool(
        independent_sources
        >= 1
        and aligned3
        >= baseline3
        + 100
        and (
            baseline_median
            is None
            or aligned_median
            is None
            or aligned_median
            < baseline_median
        )
    )

    feature_exact = bool(
        replay.get(
            "status"
        )
        == "PASS_EXACT_CKPT5_SCAN_FEATURE_REPLAY"
    )

    if (
        clock_confirmed
        and feature_exact
    ):

        status = (
            "PASS_CLOCK_AND_EXACT_FEATURE_BRIDGE"
        )

    elif clock_confirmed:

        status = (
            "PASS_CLOCK_BRIDGE_FEATURE_REPLAY_NEEDS_RESOLUTION"
        )

    elif feature_exact:

        status = (
            "PASS_FEATURE_REPLAY_CLOCK_NEEDS_RESOLUTION"
        )

    else:

        status = (
            "BRIDGE_COMPONENTS_NEED_RESOLUTION"
        )

    ###########################################################################
    # Report.
    ###########################################################################

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "clock_candidate":
            clock_audit,

        "clock_offsets_file":
            (
                "artifacts/checkpoint7a3g/"
                "msk_patient_clock_offsets.parquet"
            ),

        "independent_event_source_audit":
            event_audit,

        "independent_event_alignment":
            event_aggregate,

        "independent_event_table":
            (
                event_table.to_dict(
                    orient="records"
                )
                if len(
                    event_table
                )
                else []
            ),

        "clock_independently_confirmed":
            clock_confirmed,

        "feature_replay":
            replay_public,

        "exact_ckpt5_feature_replay":
            feature_exact,

        "policy": {
            "clock_offset_scope":
                (
                    "MSK positive-control bridge only. "
                    "Do not apply this offset mapping "
                    "to DFCI or VICC."
                ),

            "clock_selected_without_external_outcomes":
                True,

            "external_feature_builder_policy":
                (
                    "Reuse the exact frozen CKPT5 "
                    "scan_features implementation rather "
                    "than the manually reconstructed CKPT7A3 "
                    "feature builder."
                ),

            "external_outcomes_accessed":
                False,
        },

        "next_action":
            (
                "Build CKPT7A4 outcome-blinded DFCI/VICC model inputs "
                "using exact frozen feature semantics and native "
                "site timelines; do not transfer MSK clock offsets."
                if status
                == "PASS_CLOCK_AND_EXACT_FEATURE_BRIDGE"
                else
                "Resolve only the failed bridge component before "
                "constructing external model tensors."
            ),
    }

    atomic_json(
        out
        / "bridge_confirmation.json",
        report,
    )

    ###########################################################################
    # Audit.
    ###########################################################################

    audit = f"""# CKPT7A3G — independent clock confirmation and exact CKPT5 feature replay

Status: **{status}**

External outcomes opened: **NO**

## Candidate MSK clock bridge

{clock_audit}

The offset remains scoped to the BPC-MSK / CHORD positive control only.

## Independent non-imaging temporal confirmation

{event_aggregate}

Per-source results:

{event_table.to_string(index=False) if len(event_table) else "NO USABLE SHARED NON-IMAGING EVENT SOURCES"}

Clock independently confirmed:

{clock_confirmed}

## Exact CKPT5 scan-feature replay

{replay_public}

Exact replay pass:

{feature_exact}

## Policy

No DFCI/VICC outcome table was opened.

No MSK clock offset will be transferred to DFCI/VICC.

The eventual external adapter must reuse the exact frozen CKPT5 feature-generation
semantics rather than CKPT7A3's manually reconstructed approximation.
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
            "7A3G",

        "name":
            "confirm_clock_and_exact_ckpt5_feature_replay",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "clock_independently_confirmed":
            clock_confirmed,

        "independent_event_alignment":
            event_aggregate,

        "feature_replay_status":
            replay.get(
                "status"
            ),

        "exact_ckpt5_feature_replay":
            feature_exact,

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A3G.json",
        handoff,
    )

    ###########################################################################
    # Terminal packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT7A3G SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "clock_candidate="
        f"{clock_audit}"
    )

    print(
        "independent_event_sources="
        f"{independent_sources}"
    )

    print(
        "independent_event_alignment="
        f"{event_aggregate}"
    )

    print(
        "clock_independently_confirmed="
        f"{clock_confirmed}"
    )

    print(
        "feature_replay_status="
        f"{replay.get('status')}"
    )

    print(
        "feature_replay_summary="
        f"{replay_public}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A3G.json"
    )

    print(
        "========== CKPT7A3G SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A3G DECISION PACKET =========="
    )

    print("")
    print(
        "----- INDEPENDENT EVENT ALIGNMENT -----"
    )

    if len(
        event_table
    ):

        print(
            event_table.to_string(
                index=False
            )
        )

    else:

        print(
            "NO USABLE SHARED NON-IMAGING EVENT SOURCES"
        )

    print(
        "----- INDEPENDENT EVENT ALIGNMENT END -----"
    )

    print("")
    print(
        "----- FEATURE REPLAY -----"
    )

    print(
        "scan_features_signature="
        f"{replay.get('scan_features_signature')}"
    )

    print(
        "load_scans="
        f"{replay.get('load_scans')}"
    )

    print(
        "scan_source_method="
        f"{replay.get('scan_source_method')}"
    )

    print(
        "rows="
        f"{replay.get('rows')}"
    )

    print(
        "features="
        f"{replay.get('features')}"
    )

    print(
        "join_method="
        f"{replay.get('join_method')}"
    )

    print(
        "overall_mean_absolute_difference="
        f"{replay.get('overall_mean_absolute_difference')}"
    )

    print(
        "max_absolute_difference="
        f"{replay.get('max_absolute_difference')}"
    )

    print(
        "vector_exact_fraction="
        f"{replay.get('vector_exact_fraction')}"
    )

    print(
        "per_feature="
        f"{replay.get('per_feature')}"
    )

    print(
        "----- FEATURE REPLAY END -----"
    )

    print("")
    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A3G DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
