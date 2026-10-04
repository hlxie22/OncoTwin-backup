#!/usr/bin/env python3

from __future__ import annotations

import argparse
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

    return str(
        value
    ).strip().upper()


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

        key = norm(candidate)

        if key in lookup:
            return lookup[key]

    if required:

        raise KeyError(
            f"Unable to resolve {list(candidates)}; "
            f"available={list(columns)}"
        )

    return None


def atomic_json(path: Path, payload: Any) -> None:

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


def atomic_text(path: Path, text: str) -> None:

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

    check = pd.read_parquet(tmp)

    if len(check) != len(frame):

        raise RuntimeError(
            f"Parquet verification failed: {path}"
        )

    tmp.replace(path)


def quantiles(
    values: Iterable[float],
) -> dict[str, Any]:

    arr = np.asarray(
        list(values),
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:

        return {
            "n": 0,
        }

    return {
        "n":
            int(len(arr)),

        "min":
            float(np.min(arr)),

        "p25":
            float(np.quantile(arr, 0.25)),

        "median":
            float(np.quantile(arr, 0.50)),

        "p75":
            float(np.quantile(arr, 0.75)),

        "p90":
            float(np.quantile(arr, 0.90)),

        "p95":
            float(np.quantile(arr, 0.95)),

        "max":
            float(np.max(arr)),
    }


def delimiter_for(path: Path) -> str:

    return (
        ","
        if path.suffix.lower() == ".csv"
        else "\t"
    )


def read_table(path: Path) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep=delimiter_for(path),
        comment="#",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def find_files(
    root: Path,
    basename: str,
) -> list[Path]:

    return [
        path
        for path in root.rglob(basename)
        if path.is_file()
    ]


###############################################################################
# Import BPC outcome guard
###############################################################################


def import_ckpt7a2(repo: Path):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt7a2_external_semantic_audit.py"
    )

    spec = importlib.util.spec_from_file_location(
        "ckpt7a2_clock_diag",
        path,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Unable to import CKPT7A2."
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(module)

    return module


###############################################################################
# Safe BPC predictor loading
###############################################################################


def bpc_guard(
    repo: Path,
    bpc_root: Path,
):

    inventory = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a1"
        / "schema_inventory.parquet"
    )

    ckpt7a2 = import_ckpt7a2(repo)

    return ckpt7a2.AccessGuard(
        bpc_root,
        inventory,
    )


###############################################################################
# Sample-date extraction
###############################################################################


SAMPLE_ID_CANDIDATES = [
    "SAMPLE_ID",
    "sample_id",
    "Tumor_Sample_Barcode",
    "SPECIMEN_ID",
]

DATE_CANDIDATES = [
    "START_DATE",
    "SEQ_DATE",
    "CPT_SEQ_DATE",
    "SEQUENCING_DATE",
    "SPECIMEN_DATE",
    "SAMPLE_COLLECTION_DATE",
]


def extract_sample_dates(
    frame: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:

    sample_col = find_col(
        frame.columns,
        SAMPLE_ID_CANDIDATES,
        required=False,
    )

    if sample_col is None:

        return pd.DataFrame(
            columns=[
                "sample_id",
                "date_field",
                "day",
                "source",
            ]
        )

    rows = []

    for candidate in DATE_CANDIDATES:

        column = find_col(
            frame.columns,
            [
                candidate,
            ],
            required=False,
        )

        if column is None:
            continue

        numeric = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        valid = (
            frame[sample_col]
            .astype(str)
            .str.strip()
            .ne("")
            & numeric.notna()
        )

        current = pd.DataFrame(
            {
                "sample_id":
                    frame.loc[
                        valid,
                        sample_col,
                    ].map(clean),

                "date_field":
                    candidate,

                "day":
                    numeric.loc[
                        valid
                    ].astype(float),

                "source":
                    source_name,
            }
        )

        rows.append(current)

    if not rows:

        return pd.DataFrame(
            columns=[
                "sample_id",
                "date_field",
                "day",
                "source",
            ]
        )

    out = pd.concat(
        rows,
        ignore_index=True,
    )

    ###########################################################################
    # Preserve only unique sample/date-field values.
    ###########################################################################

    counts = (
        out.groupby(
            [
                "sample_id",
                "date_field",
            ],
            observed=True,
        )[
            "day"
        ]
        .nunique()
        .rename(
            "n_unique_days"
        )
        .reset_index()
    )

    unique_keys = counts[
        counts[
            "n_unique_days"
        ]
        == 1
    ][
        [
            "sample_id",
            "date_field",
        ]
    ]

    out = out.merge(
        unique_keys,
        on=[
            "sample_id",
            "date_field",
        ],
        how="inner",
    )

    out = (
        out
        .drop_duplicates(
            [
                "sample_id",
                "date_field",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return out


###############################################################################
# Build date-anchor candidates
###############################################################################


def build_anchor_candidates(
    repo: Path,
    bpc_root: Path,
    chord_root: Path,
    sample_crosswalk: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:

    guard = bpc_guard(
        repo,
        bpc_root,
    )

    bpc_sources = {}

    ###########################################################################
    # BPC timeline sequencing.
    ###########################################################################

    bpc_seq_relative = (
        "cBioPortal_files/"
        "data_timeline_sequencing.txt"
    )

    try:

        bpc_sources[
            "timeline_sequencing"
        ] = guard.read_predictor(
            bpc_seq_relative
        )

    except Exception:

        pass

    ###########################################################################
    # BPC clinical sample.
    ###########################################################################

    try:

        bpc_sources[
            "clinical_sample"
        ] = guard.read_predictor(
            "cBioPortal_files/data_clinical_sample.txt"
        )

    except Exception:

        pass

    ###########################################################################
    # CHORD equivalents.
    ###########################################################################

    chord_sources = {}

    for basename, key in (
        (
            "data_timeline_sequencing.txt",
            "timeline_sequencing",
        ),
        (
            "data_clinical_sample.txt",
            "clinical_sample",
        ),
    ):

        files = find_files(
            chord_root,
            basename,
        )

        if len(files) == 1:

            chord_sources[
                key
            ] = read_table(
                files[0]
            )

    source_audit = {
        "bpc_sources":
            {
                key:
                    {
                        "rows":
                            int(len(frame)),

                        "columns":
                            list(frame.columns),
                    }
                for key, frame
                in bpc_sources.items()
            },

        "chord_sources":
            {
                key:
                    {
                        "rows":
                            int(len(frame)),

                        "columns":
                            list(frame.columns),
                    }
                for key, frame
                in chord_sources.items()
            },

        "bpc_access_log":
            guard.access_log,
    }

    ###########################################################################
    # Extract sample dates from each source.
    ###########################################################################

    bpc_dates = []

    for source, frame in bpc_sources.items():

        current = extract_sample_dates(
            frame,
            source_name=source,
        )

        if len(current):

            current[
                "source_key"
            ] = (
                source
                + ":"
                + current[
                    "date_field"
                ]
            )

            bpc_dates.append(
                current
            )

    chord_dates = []

    for source, frame in chord_sources.items():

        current = extract_sample_dates(
            frame,
            source_name=source,
        )

        if len(current):

            current[
                "source_key"
            ] = (
                source
                + ":"
                + current[
                    "date_field"
                ]
            )

            chord_dates.append(
                current
            )

    if bpc_dates:

        bpc_dates = pd.concat(
            bpc_dates,
            ignore_index=True,
        )

    else:

        bpc_dates = pd.DataFrame()

    if chord_dates:

        chord_dates = pd.concat(
            chord_dates,
            ignore_index=True,
        )

    else:

        chord_dates = pd.DataFrame()

    if (
        bpc_dates.empty
        or chord_dates.empty
    ):

        return (
            pd.DataFrame(),
            source_audit,
        )

    ###########################################################################
    # Translate BPC sample IDs through frozen sample crosswalk.
    ###########################################################################

    sample_map = (
        sample_crosswalk[
            [
                "bpc_patient_id",
                "bpc_sample_id",
                "chord_patient_id",
                "chord_sample_id",
            ]
        ]
        .drop_duplicates()
    )

    bpc_dates = bpc_dates.merge(
        sample_map,
        left_on="sample_id",
        right_on="bpc_sample_id",
        how="inner",
        validate="many_to_one",
    )

    ###########################################################################
    # Compare only like-for-like source/date-field semantics first.
    ###########################################################################

    merged = bpc_dates.merge(
        chord_dates[
            [
                "sample_id",
                "source_key",
                "day",
            ]
        ].rename(
            columns={
                "sample_id":
                    "chord_sample_id",

                "day":
                    "chord_day",
            }
        ),
        on=[
            "chord_sample_id",
            "source_key",
        ],
        how="inner",
    )

    if merged.empty:

        return (
            merged,
            source_audit,
        )

    merged = merged.rename(
        columns={
            "day":
                "bpc_day",
        }
    )

    merged[
        "offset_chord_minus_bpc"
    ] = (
        merged[
            "chord_day"
        ]
        - merged[
            "bpc_day"
        ]
    )

    return (
        merged,
        source_audit,
    )


###############################################################################
# Choose the strongest outcome-free temporal anchor
###############################################################################


def choose_anchor(
    anchors: pd.DataFrame,
) -> tuple[
    str | None,
    pd.DataFrame,
    dict[str, Any],
]:

    if anchors.empty:

        return (
            None,
            pd.DataFrame(),
            {},
        )

    candidates = []

    for source_key, group in anchors.groupby(
        "source_key",
        observed=True,
    ):

        patient_offsets = (
            group.groupby(
                [
                    "bpc_patient_id",
                    "chord_patient_id",
                ],
                observed=True,
            )[
                "offset_chord_minus_bpc"
            ]
            .median()
            .rename(
                "clock_offset"
            )
            .reset_index()
        )

        candidates.append(
            {
                "source_key":
                    source_key,

                "matched_samples":
                    int(
                        group[
                            "chord_sample_id"
                        ].nunique()
                    ),

                "matched_patients":
                    int(
                        patient_offsets[
                            "chord_patient_id"
                        ].nunique()
                    ),

                "offset_distribution":
                    quantiles(
                        patient_offsets[
                            "clock_offset"
                        ]
                    ),
            }
        )

    candidates.sort(
        key=lambda item:
            (
                item[
                    "matched_patients"
                ],
                item[
                    "matched_samples"
                ],
            ),
        reverse=True,
    )

    selected = candidates[0][
        "source_key"
    ]

    selected_rows = anchors[
        anchors[
            "source_key"
        ]
        == selected
    ].copy()

    offsets = (
        selected_rows.groupby(
            [
                "bpc_patient_id",
                "chord_patient_id",
            ],
            observed=True,
        )[
            "offset_chord_minus_bpc"
        ]
        .median()
        .rename(
            "clock_offset"
        )
        .reset_index()
    )

    return (
        selected,
        offsets,
        {
            "candidates":
                candidates,

            "selected_source":
                selected,

            "selected_patients":
                int(
                    offsets[
                        "chord_patient_id"
                    ].nunique()
                ),
        },
    )


###############################################################################
# CHORD canonical scan reference
###############################################################################


def load_chord_w3(
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
        required=True,
    )

    day_col = find_col(
        frame.columns,
        [
            "episode_end_day",
            "landmark_day",
        ],
        required=True,
    )

    state_col = find_col(
        frame.columns,
        [
            "progression_state_3",
            "scan_state",
        ],
        required=True,
    )

    return pd.DataFrame(
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


###############################################################################
# Clock-alignment evaluation
###############################################################################


def nearest_summary(
    bpc: pd.DataFrame,
    chord: pd.DataFrame,
    bpc_day_col: str,
) -> dict[str, Any]:

    distances = []

    patients = 0

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

        left = (
            bpc.loc[
                bpc[
                    "chord_patient_id"
                ]
                == patient,
                bpc_day_col,
            ]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        right = (
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
            len(left) == 0
            or len(right) == 0
        ):

            continue

        patients += 1

        for day in left:

            distances.append(
                float(
                    np.min(
                        np.abs(
                            right
                            - day
                        )
                    )
                )
            )

    arr = np.asarray(
        distances,
        dtype=float,
    )

    return {
        "patients":
            int(patients),

        "distance_distribution":
            quantiles(
                arr
            ),

        "within0":
            int(
                (arr <= 0).sum()
            )
            if len(arr)
            else 0,

        "within3":
            int(
                (arr <= 3).sum()
            )
            if len(arr)
            else 0,

        "within7":
            int(
                (arr <= 7).sum()
            )
            if len(arr)
            else 0,

        "within30":
            int(
                (arr <= 30).sum()
            )
            if len(arr)
            else 0,
    }


def greedy_match(
    bpc: pd.DataFrame,
    chord: pd.DataFrame,
    bpc_day_col: str,
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

        candidates = []

        for li, lrow in left.iterrows():

            if not np.isfinite(
                lrow[
                    bpc_day_col
                ]
            ):
                continue

            for ri, rrow in right.iterrows():

                if not np.isfinite(
                    rrow[
                        "chord_episode_day"
                    ]
                ):
                    continue

                delta = abs(
                    float(
                        lrow[
                            bpc_day_col
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

            used_left.add(li)
            used_right.add(ri)

            lrow = left.loc[li]
            rrow = right.loc[ri]

            rows.append(
                {
                    "bpc_patient_id":
                        lrow[
                            "patient_id"
                        ],

                    "chord_patient_id":
                        patient,

                    "bpc_original_day":
                        float(
                            lrow[
                                "episode_end_day"
                            ]
                        ),

                    "bpc_aligned_day":
                        float(
                            lrow[
                                bpc_day_col
                            ]
                        ),

                    "chord_day":
                        float(
                            rrow[
                                "chord_episode_day"
                            ]
                        ),

                    "delta":
                        float(delta),

                    "bpc_state":
                        lrow[
                            "scan_state"
                        ],

                    "chord_state":
                        rrow[
                            "chord_scan_state"
                        ],
                }
            )

    return pd.DataFrame(rows)


def matched_state_summary(
    matches: pd.DataFrame,
) -> dict[str, Any]:

    if matches.empty:

        return {
            "rows": 0,
        }

    current = matches[
        matches[
            "bpc_state"
        ].isin(
            OBSERVED_STATES
        )
        & matches[
            "chord_state"
        ].isin(
            OBSERVED_STATES
        )
    ]

    if current.empty:

        return {
            "rows": 0,
        }

    return {
        "rows":
            int(len(current)),

        "patients":
            int(
                current[
                    "chord_patient_id"
                ].nunique()
            ),

        "agreement":
            float(
                (
                    current[
                        "bpc_state"
                    ]
                    == current[
                        "chord_state"
                    ]
                ).mean()
            ),

        "confusion":
            (
                current.groupby(
                    [
                        "bpc_state",
                        "chord_state",
                    ],
                    observed=True,
                )
                .size()
                .rename("n")
                .reset_index()
                .to_dict(
                    orient="records"
                )
            ),
    }


###############################################################################
# Exact CKPT5 feature-provenance search
###############################################################################


FEATURE_TOKENS = [
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
]


def provenance_search(
    repo: Path,
) -> dict[str, Any]:

    roots = [
        repo
        / "scripts"
        / "dynamic_scan",

        repo
        / "configs"
        / "dynamic_scan",
    ]

    findings = {}

    for token in FEATURE_TOKENS:

        hits = []

        for root in roots:

            if not root.exists():
                continue

            for path in root.rglob("*"):

                if (
                    not path.is_file()
                    or path.suffix.lower()
                    not in {
                        ".py",
                        ".json",
                    }
                ):
                    continue

                try:

                    lines = path.read_text(
                        encoding="utf-8",
                    ).splitlines()

                except Exception:
                    continue

                for line_number, line in enumerate(
                    lines,
                    start=1,
                ):

                    if token not in line:
                        continue

                    lo = max(
                        0,
                        line_number
                        - 6,
                    )

                    hi = min(
                        len(lines),
                        line_number
                        + 5,
                    )

                    snippet = "\n".join(
                        f"{idx + 1}: {lines[idx]}"
                        for idx in range(
                            lo,
                            hi,
                        )
                    )

                    hits.append(
                        {
                            "path":
                                str(
                                    path.relative_to(
                                        repo
                                    )
                                ),

                            "line":
                                int(
                                    line_number
                                ),

                            "snippet":
                                snippet,
                        }
                    )

        #######################################################################
        # Avoid repeating the same builder block dozens of times.
        #######################################################################

        unique = []

        seen = set()

        for hit in hits:

            key = (
                hit[
                    "path"
                ],
                hit[
                    "snippet"
                ],
            )

            if key in seen:
                continue

            seen.add(key)

            unique.append(hit)

        findings[
            token
        ] = unique[
            :12
        ]

    return findings


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
            "artifacts/checkpoint7a3e"
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
    # Frozen identity maps
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

    ###########################################################################
    # Outcome-free genomic clock anchors
    ###########################################################################

    (
        anchors,
        source_audit,
    ) = build_anchor_candidates(
        repo,
        bpc_root,
        chord_root,
        sample_crosswalk,
    )

    (
        selected_source,
        patient_offsets,
        anchor_report,
    ) = choose_anchor(
        anchors
    )

    if len(anchors):

        atomic_parquet(
            out
            / "sample_clock_anchor_candidates.parquet",
            anchors,
        )

    else:

        atomic_parquet(
            out
            / "sample_clock_anchor_candidates.parquet",
            pd.DataFrame(
                {
                    "empty": pd.Series(
                        dtype=bool
                    )
                }
            ),
        )

    if len(patient_offsets):

        atomic_parquet(
            out
            / "patient_clock_offsets.parquet",
            patient_offsets,
        )

    else:

        atomic_parquet(
            out
            / "patient_clock_offsets.parquet",
            pd.DataFrame(
                {
                    "bpc_patient_id":
                        pd.Series(
                            dtype=str
                        ),

                    "chord_patient_id":
                        pd.Series(
                            dtype=str
                        ),

                    "clock_offset":
                        pd.Series(
                            dtype=float
                        ),
                }
            ),
        )

    ###########################################################################
    # BPC W3 episodes
    ###########################################################################

    bpc = pd.read_parquet(
        repo
        / "artifacts"
        / "checkpoint7a3"
        / "bpc_w3_predictor_episodes.parquet"
    )

    bpc = bpc[
        bpc[
            "site"
        ]
        == "MSK"
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

    bpc = bpc[
        bpc[
            "scan_state"
        ].isin(
            OBSERVED_STATES
        )
    ].copy()

    chord = load_chord_w3(
        repo
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

    ###########################################################################
    # Original-clock reference
    ###########################################################################

    original_clock = nearest_summary(
        bpc,
        chord,
        "episode_end_day",
    )

    ###########################################################################
    # Apply outcome-free genomic clock offset when available
    ###########################################################################

    aligned_clock = None
    aligned_matches = {
        "within0": {
            "rows": 0,
        },
        "within3": {
            "rows": 0,
        },
        "within7": {
            "rows": 0,
        },
    }

    if len(patient_offsets):

        aligned = bpc.merge(
            patient_offsets,
            on=[
                "bpc_patient_id",
                "chord_patient_id",
            ],
            how="inner",
            validate="many_to_one",
        )

        aligned[
            "aligned_episode_day"
        ] = (
            aligned[
                "episode_end_day"
            ].astype(float)
            + aligned[
                "clock_offset"
            ].astype(float)
        )

        aligned_clock = nearest_summary(
            aligned,
            chord,
            "aligned_episode_day",
        )

        m0 = greedy_match(
            aligned,
            chord,
            "aligned_episode_day",
            0,
        )

        m3 = greedy_match(
            aligned,
            chord,
            "aligned_episode_day",
            3,
        )

        m7 = greedy_match(
            aligned,
            chord,
            "aligned_episode_day",
            7,
        )

        atomic_parquet(
            out
            / "clock_aligned_matches_exact.parquet",
            m0,
        )

        atomic_parquet(
            out
            / "clock_aligned_matches_within3.parquet",
            m3,
        )

        atomic_parquet(
            out
            / "clock_aligned_matches_within7.parquet",
            m7,
        )

        aligned_matches = {
            "within0": {
                "rows":
                    int(len(m0)),

                "patients":
                    int(
                        m0[
                            "chord_patient_id"
                        ].nunique()
                    )
                    if len(m0)
                    else 0,

                "state":
                    matched_state_summary(
                        m0
                    ),
            },

            "within3": {
                "rows":
                    int(len(m3)),

                "patients":
                    int(
                        m3[
                            "chord_patient_id"
                        ].nunique()
                    )
                    if len(m3)
                    else 0,

                "state":
                    matched_state_summary(
                        m3
                    ),
            },

            "within7": {
                "rows":
                    int(len(m7)),

                "patients":
                    int(
                        m7[
                            "chord_patient_id"
                        ].nunique()
                    )
                    if len(m7)
                    else 0,

                "state":
                    matched_state_summary(
                        m7
                    ),
            },
        }

    else:

        for name in (
            "clock_aligned_matches_exact.parquet",
            "clock_aligned_matches_within3.parquet",
            "clock_aligned_matches_within7.parquet",
        ):

            atomic_parquet(
                out
                / name,
                pd.DataFrame(
                    {
                        "empty":
                            pd.Series(
                                dtype=bool
                            )
                    }
                ),
            )

    ###########################################################################
    # Exact CKPT5 feature-generation provenance
    ###########################################################################

    feature_provenance = provenance_search(
        repo
    )

    atomic_json(
        out
        / "feature_provenance.json",
        feature_provenance,
    )

    provenance_lines = []

    for token in FEATURE_TOKENS:

        provenance_lines.append(
            (
                "\n============================================================\n"
                + token
                + "\n============================================================"
            )
        )

        hits = feature_provenance[
            token
        ]

        if not hits:

            provenance_lines.append(
                "NO SOURCE HIT"
            )

            continue

        for hit in hits:

            provenance_lines.append(
                (
                    "\n--- "
                    + hit[
                        "path"
                    ]
                    + ":"
                    + str(
                        hit[
                            "line"
                        ]
                    )
                    + " ---\n"
                    + hit[
                        "snippet"
                    ]
                )
            )

    atomic_text(
        out
        / "feature_provenance.txt",
        "\n".join(
            provenance_lines
        )
        + "\n",
    )

    ###########################################################################
    # Diagnose clock result
    ###########################################################################

    original_within3 = int(
        original_clock[
            "within3"
        ]
    )

    aligned_within3 = (
        int(
            aligned_clock[
                "within3"
            ]
        )
        if aligned_clock
        is not None
        else 0
    )

    improvement_ratio = (
        float(
            aligned_within3
            / max(
                original_within3,
                1,
            )
        )
    )

    anchor_patients = int(
        anchor_report.get(
            "selected_patients",
            0,
        )
    )

    if (
        selected_source is None
        or anchor_patients < 50
    ):

        status = (
            "NEEDS_CLOCK_ANCHOR_RESOLUTION"
        )

    elif (
        aligned_within3
        >= 5
        * max(
            original_within3,
            1,
        )
    ):

        status = (
            "PATIENT_SPECIFIC_CLOCK_MISMATCH_CONFIRMED"
        )

    else:

        status = (
            "CLOCK_SHIFT_DOES_NOT_EXPLAIN_SCAN_MISMATCH"
        )

    ###########################################################################
    # Report
    ###########################################################################

    report = {
        "status":
            status,

        "external_outcomes_opened":
            False,

        "external_outcome_distributions_inspected":
            False,

        "anchor_source_audit":
            source_audit,

        "anchor_selection":
            anchor_report,

        "selected_clock_anchor":
            selected_source,

        "patient_offset_distribution":
            (
                quantiles(
                    patient_offsets[
                        "clock_offset"
                    ]
                )
                if len(
                    patient_offsets
                )
                else {
                    "n": 0,
                }
            ),

        "original_clock":
            original_clock,

        "aligned_clock":
            aligned_clock,

        "within3_improvement_ratio":
            improvement_ratio,

        "aligned_matches":
            aligned_matches,

        "feature_provenance_tokens_found":
            {
                token:
                    int(
                        len(
                            hits
                        )
                    )
                for token, hits
                in feature_provenance.items()
            },

        "next_action":
            (
                (
                    "Freeze the outcome-free patient-specific temporal "
                    "crosswalk, then rerun the MSK semantic bridge on the "
                    "aligned clock and reconstruct scan features from the "
                    "exact CKPT5 builder."
                )
                if status
                == "PATIENT_SPECIFIC_CLOCK_MISMATCH_CONFIRMED"
                else (
                    "Do not force episode identity. Inspect whether BPC imaging "
                    "and CHORD radiology are different observational streams, "
                    "and reconstruct the external features from exact CKPT5 "
                    "source semantics."
                    if status
                    == "CLOCK_SHIFT_DOES_NOT_EXPLAIN_SCAN_MISMATCH"
                    else
                    "Resolve an outcome-free shared temporal anchor before "
                    "attempting further episode-level positive control."
                )
            ),
    }

    atomic_json(
        out
        / "diagnostic.json",
        report,
    )

    ###########################################################################
    # Human-readable audit
    ###########################################################################

    audit = f"""# CKPT7A3E — Clock and feature-provenance diagnosis

Status: **{status}**

External outcomes opened: **NO**

## Shared genomic clock anchor

Selected source:
{selected_source}

Anchor report:
{anchor_report}

Patient-specific offset distribution:
{report['patient_offset_distribution']}

## Scan alignment before offset

{original_clock}

## Scan alignment after offset

{aligned_clock}

Within-3-day improvement ratio:
{improvement_ratio}

## Clock-aligned state matching

{aligned_matches}

## Feature provenance

Exact source-code hits were saved to:

`artifacts/checkpoint7a3e/feature_provenance.txt`

No external outcomes were used to select or evaluate the clock transformation.
"""

    atomic_text(
        out
        / "audit.md",
        audit,
    )

    ###########################################################################
    # Handoff
    ###########################################################################

    handoff = {
        "checkpoint":
            "7A3E",

        "name":
            "clock_and_feature_provenance_diagnosis",

        "status":
            status,

        "external_outcomes_opened":
            False,

        "selected_clock_anchor":
            selected_source,

        "anchor_patients":
            anchor_patients,

        "original_clock":
            original_clock,

        "aligned_clock":
            aligned_clock,

        "within3_improvement_ratio":
            improvement_ratio,

        "aligned_matches":
            aligned_matches,

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_07A3E.json",
        handoff,
    )

    ###########################################################################
    # Terminal output
    ###########################################################################

    print("")
    print(
        "========== CKPT7A3E SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "external_outcomes_opened=False"
    )

    print(
        "selected_clock_anchor="
        f"{selected_source}"
    )

    print(
        "anchor_patients="
        f"{anchor_patients}"
    )

    print(
        "patient_offset_distribution="
        f"{report['patient_offset_distribution']}"
    )

    print(
        "original_clock="
        f"{original_clock}"
    )

    print(
        "aligned_clock="
        f"{aligned_clock}"
    )

    print(
        "within3_improvement_ratio="
        f"{improvement_ratio}"
    )

    print(
        "aligned_matches="
        f"{aligned_matches}"
    )

    print(
        "feature_provenance_tokens_found="
        f"{report['feature_provenance_tokens_found']}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_07A3E.json"
    )

    print(
        "========== CKPT7A3E SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT7A3E DECISION PACKET =========="
    )

    print(
        "anchor_candidates="
        f"{anchor_report.get('candidates')}"
    )

    print("")
    print(
        "----- EXACT CKPT5 FEATURE PROVENANCE -----"
    )

    for token in FEATURE_TOKENS:

        hits = feature_provenance[
            token
        ]

        print("")
        print(
            "FEATURE",
            token,
            "HITS",
            len(hits),
        )

        for hit in hits[
            :3
        ]:

            print(
                "---",
                hit[
                    "path"
                ],
                "line",
                hit[
                    "line"
                ],
                "---",
            )

            print(
                hit[
                    "snippet"
                ]
            )

    print(
        "----- EXACT CKPT5 FEATURE PROVENANCE END -----"
    )

    print("")
    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT7A3E DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
