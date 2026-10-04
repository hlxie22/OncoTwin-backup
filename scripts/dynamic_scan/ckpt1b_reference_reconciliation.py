#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAY_SPLIT = 365
GRACE_DAYS = 28
RADIOLOGY_LOOKBACK_DAYS = 90
ADMIN_CENSOR_DAYS = 730

PAPER_PATIENTS = 2881
PAPER_LINES = 8791

DIAGNOSIS_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<histologic>[^|]+?)
    \s*\|\s*
    (?P<site>[^,|()]+?)
    (?:\s*,\s*(?P<subset>.*?))?
    \s*(?=\(\s*M\d{4}/)
    \(\s*
    (?P<icdo_morph>M\d{4}/[012369](?:[0-4]|9)?)
    \s*\|\s*
    (?P<icdo_topo>C\d{3})
    \s*\)\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"empty write: {path}")

    json.loads(
        tmp.read_text(
            encoding="utf-8"
        )
    )

    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"empty write: {path}")

    tmp.replace(path)


def atomic_parquet(
    path: Path,
    df: pd.DataFrame,
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_parquet(
        tmp,
        index=False,
    )

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"empty parquet: {path}")

    # Verify readability before replacing.
    check = pd.read_parquet(tmp)

    if len(check) != len(df):
        raise RuntimeError(
            f"parquet verification mismatch: {path}"
        )

    tmp.replace(path)


def find_one(
    root: Path,
    basename: str,
) -> Path:
    matches = [
        p
        for p in root.rglob(basename)
        if p.is_file()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {basename}; "
            f"found {len(matches)}: {matches[:10]}"
        )

    return matches[0]


def read_timeline(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        comment="#",
        keep_default_na=False,
        low_memory=False,
    )

    df.columns = [
        str(c)
        .strip()
        .upper()
        .replace(" ", "_")
        for c in df.columns
    ]

    if "START_DATE" in df.columns:
        df["START_DATE"] = pd.to_numeric(
            df["START_DATE"],
            errors="coerce",
        )

    if "STOP_DATE" in df.columns:
        df["STOP_DATE"] = pd.to_numeric(
            df["STOP_DATE"].replace("", np.nan),
            errors="coerce",
        )

    return df


def prep_diagnosis(
    chord_root: Path,
) -> tuple[
    pd.DataFrame,
    set[str],
    set[str],
]:
    path = find_one(
        chord_root,
        "data_timeline_diagnosis.txt",
    )

    diag = read_timeline(path)

    if "DX_DESCRIPTION" not in diag.columns:
        raise RuntimeError(
            "diagnosis table lacks DX_DESCRIPTION"
        )

    extracted = diag[
        "DX_DESCRIPTION"
    ].str.extract(
        DIAGNOSIS_PATTERN
    )

    diag["PARSED_SITE"] = (
        extracted["site"]
        .str.strip()
        .str.upper()
    )

    diagnosed_breast = set(
        diag.loc[
            diag["PARSED_SITE"] == "BREAST",
            "PATIENT_ID",
        ]
    )

    # Match the public implementation:
    # patients with any non-breast diagnosis are excluded.
    other_site = set(
        diag.loc[
            diag["PARSED_SITE"] != "BREAST",
            "PATIENT_ID",
        ]
    )

    breast_only = (
        diagnosed_breast
        - (
            diagnosed_breast
            & other_site
        )
    )

    diag = diag[
        diag["PATIENT_ID"].isin(
            breast_only
        )
    ].copy()

    if "STAGE_CDM_DERIVED" not in diag.columns:
        raise RuntimeError(
            "diagnosis table lacks STAGE_CDM_DERIVED"
        )

    stage4 = set(
        diag.loc[
            (
                diag["STAGE_CDM_DERIVED"]
                .astype(str)
                .str.strip()
                .str.upper()
                == "STAGE 4"
            ),
            "PATIENT_ID",
        ]
    )

    return (
        diag,
        breast_only,
        stage4,
    )


def first_metastasis_dates(
    chord_root: Path,
    patients: set[str],
) -> dict[str, int]:
    path = find_one(
        chord_root,
        "data_timeline_tumor_sites.txt",
    )

    df = read_timeline(path)

    df = df[
        df["PATIENT_ID"].isin(
            patients
        )
    ].copy()

    df["TUMOR_SITE"] = (
        df["TUMOR_SITE"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["START_DATE"].notna()
        & ~df["TUMOR_SITE"].isin(
            {
                "OTHER",
                "LYMPH NODES",
            }
        )
    ]

    first = (
        df.groupby(
            "PATIENT_ID",
            observed=True,
        )["START_DATE"]
        .min()
        .astype(int)
    )

    return first.to_dict()


def filter_post_metastatic(
    df: pd.DataFrame,
    stage4: set[str],
    first_meta: dict[str, int],
) -> pd.DataFrame:
    work = df.copy()

    meta_day = (
        work["PATIENT_ID"]
        .map(first_meta)
    )

    keep = (
        work["PATIENT_ID"].isin(
            stage4
        )
        | (
            meta_day.notna()
            & (
                work["START_DATE"]
                >= meta_day
            )
        )
    )

    return (
        work.loc[keep]
        .reset_index(drop=True)
    )


def prep_treatment(
    chord_root: Path,
    breast_only: set[str],
    stage4: set[str],
    first_meta: dict[str, int],
) -> pd.DataFrame:
    path = find_one(
        chord_root,
        "data_timeline_treatment.txt",
    )

    tx = read_timeline(path)

    tx = tx[
        tx["PATIENT_ID"].isin(
            breast_only
        )
    ].copy()

    if "SUBTYPE" in tx.columns:
        tx = tx[
            tx["SUBTYPE"]
            .astype(str)
            .str.strip()
            != "Bone Treatment"
        ]

    if "AGENT" in tx.columns:
        tx = tx[
            tx["AGENT"]
            .astype(str)
            .str.strip()
            != "INVESTIGATIVE"
        ]

    tx = tx[
        tx["START_DATE"].notna()
    ].copy()

    tx["START_DATE"] = (
        tx["START_DATE"]
        .astype(int)
    )

    tx = filter_post_metastatic(
        tx,
        stage4,
        first_meta,
    )

    return (
        tx.sort_values(
            [
                "PATIENT_ID",
                "START_DATE",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def prep_progression(
    chord_root: Path,
    breast_only: set[str],
    stage4: set[str],
    first_meta: dict[str, int],
) -> pd.DataFrame:
    path = find_one(
        chord_root,
        "data_timeline_progression.txt",
    )

    prog = read_timeline(path)

    prog = prog[
        prog["PATIENT_ID"].isin(
            breast_only
        )
    ].copy()

    prog["PROGRESSION"] = (
        prog["PROGRESSION"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Critical public-pipeline restriction.
    prog = prog[
        prog["PROGRESSION"].isin(
            ["Y", "N"]
        )
    ].copy()

    prog = prog[
        prog["START_DATE"].notna()
    ].copy()

    prog["START_DATE"] = (
        prog["START_DATE"]
        .astype(int)
    )

    prog = filter_post_metastatic(
        prog,
        stage4,
        first_meta,
    )

    return (
        prog.sort_values(
            [
                "PATIENT_ID",
                "START_DATE",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def prep_death_times(
    chord_root: Path,
    patients: set[str],
) -> dict[str, int]:
    path = find_one(
        chord_root,
        "data_clinical_patient.txt",
    )

    clinical = read_timeline(path)

    needed = {
        "PATIENT_ID",
        "OS_MONTHS",
        "OS_STATUS",
    }

    missing = (
        needed
        - set(clinical.columns)
    )

    if missing:
        raise RuntimeError(
            f"clinical patient table missing {missing}"
        )

    clinical = clinical[
        clinical["PATIENT_ID"].isin(
            patients
        )
    ].copy()

    clinical["OS_MONTHS_NUM"] = (
        pd.to_numeric(
            clinical["OS_MONTHS"],
            errors="coerce",
        )
    )

    clinical["OS_DAYS"] = (
        clinical["OS_MONTHS_NUM"]
        * 30.437
    )

    clinical = clinical[
        (
            clinical["OS_STATUS"]
            .astype(str)
            .str.strip()
            == "1:DECEASED"
        )
        & clinical["OS_DAYS"].notna()
    ].copy()

    clinical["OS_DAYS"] = (
        clinical["OS_DAYS"]
        .astype(int)
    )

    return (
        clinical
        .set_index("PATIENT_ID")[
            "OS_DAYS"
        ]
        .to_dict()
    )


def measurement_map(
    chord_root: Path,
    patients: set[str],
    progression: pd.DataFrame,
) -> dict[str, np.ndarray]:
    buckets: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for basename in (
        "data_timeline_tumor_sites.txt",
        "data_timeline_cancer_presence.txt",
    ):
        path = find_one(
            chord_root,
            basename,
        )

        df = read_timeline(path)

        df = df[
            df["PATIENT_ID"].isin(
                patients
            )
            & df["START_DATE"].notna()
        ].copy()

        for pid, grp in df.groupby(
            "PATIENT_ID",
            observed=True,
        ):
            buckets[
                str(pid)
            ].extend(
                grp[
                    "START_DATE"
                ]
                .astype(int)
                .tolist()
            )

    # Match public code: supplied progression table
    # is already Y/N and post-metastatic.
    for pid, grp in progression.groupby(
        "PATIENT_ID",
        observed=True,
    ):
        buckets[
            str(pid)
        ].extend(
            grp[
                "START_DATE"
            ]
            .astype(int)
            .tolist()
        )

    return {
        pid: np.sort(
            np.asarray(
                days,
                dtype=np.int64,
            )
        )
        for pid, days in buckets.items()
        if days
    }


def has_prior_measurement(
    measurements: dict[
        str,
        np.ndarray,
    ],
    pid: str,
    line_start: int,
) -> bool:
    days = measurements.get(pid)

    if days is None or days.size == 0:
        return False

    left = int(
        np.searchsorted(
            days,
            line_start
            - RADIOLOGY_LOOKBACK_DAYS,
            side="left",
        )
    )

    right = int(
        np.searchsorted(
            days,
            line_start,
            side="right",
        )
    )

    return right > left


def next_treatment_index(
    start_dates: np.ndarray,
    current_idx: int,
    target_day: int,
    *,
    inclusive: bool,
    agent_labels: np.ndarray | None,
    require_new_agent: bool,
) -> int:
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
        current_idx + 1,
    )

    idx = min(
        idx,
        len(start_dates),
    )

    if (
        require_new_agent
        and idx < len(start_dates)
        and agent_labels is not None
    ):
        last_idx = max(
            min(
                idx - 1,
                len(agent_labels) - 1,
            ),
            current_idx,
        )

        last_agent = (
            agent_labels[
                last_idx
            ]
        )

        while (
            idx < len(start_dates)
            and (
                agent_labels[idx]
                == last_agent
            )
        ):
            idx += 1

    return idx


def assign_patient_lines(
    patient_tx: pd.DataFrame,
    patient_prog: pd.DataFrame,
    prior_measurements: dict[
        str,
        np.ndarray,
    ],
    death_time: int | None,
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    tx = (
        patient_tx
        .sort_values(
            "START_DATE",
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    starts = (
        tx["START_DATE"]
        .to_numpy(
            dtype=int
        )
    )

    n_tx = len(starts)

    if n_tx == 0:
        return tx, []

    if "AGENT" in tx.columns:
        agents = (
            tx["AGENT"]
            .fillna("")
            .astype(str)
            .to_numpy()
        )
    else:
        agents = None

    events = (
        patient_prog
        .sort_values(
            "START_DATE",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    event_days = (
        events["START_DATE"]
        .to_numpy(dtype=int)
        if not events.empty
        else np.array(
            [],
            dtype=int,
        )
    )

    event_types = (
        events["PROGRESSION"]
        .astype(str)
        .str.upper()
        .to_numpy()
        if not events.empty
        else np.array(
            [],
            dtype=str,
        )
    )

    pid = str(
        tx["PATIENT_ID"].iat[0]
    )

    lines = np.empty(
        n_tx,
        dtype=np.int32,
    )

    rows: list[
        dict[str, Any]
    ] = []

    event_ptr = 0
    idx = 0
    current_line = 1

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

        split_idx = (
            next_treatment_index(
                starts,
                line_start_idx,
                split_day,
                inclusive=True,
                agent_labels=agents,
                require_new_agent=True,
            )
        )

        boundary_day = (
            int(starts[split_idx])
            if split_idx < n_tx
            else None
        )

        search_ptr = event_ptr

        # Match public implementation:
        # ignore observations at or before line start.
        while (
            search_ptr
            < len(event_days)
            and event_days[
                search_ptr
            ] <= line_start
        ):
            search_ptr += 1

        ptr = search_ptr

        last_n = None
        pfs_event = None
        event_day = None
        source = "undefined"

        while (
            ptr < len(event_days)
            and (
                boundary_day is None
                or event_days[ptr]
                <= boundary_day
            )
        ):
            day = int(
                event_days[ptr]
            )

            state = str(
                event_types[ptr]
            )

            if state == "N":
                last_n = day
                ptr += 1
                continue

            if state == "Y":
                if (
                    day
                    >= line_start
                    + GRACE_DAYS
                ):
                    event_day = day
                    pfs_event = 1
                    source = (
                        "labelled_by_"
                        "progression_event"
                    )

                    ptr += 1
                    break

                # Y during grace period is
                # not the line's progression endpoint.
                ptr += 1
                continue

            raise RuntimeError(
                f"unexpected progression state: {state}"
            )

        event_ptr = ptr

        if pfs_event == 1:
            event_day_int = int(
                event_day
            )

            progression_next_idx = (
                next_treatment_index(
                    starts,
                    line_start_idx,
                    event_day_int,
                    inclusive=True,
                    agent_labels=agents,
                    require_new_agent=True,
                )
            )

            next_idx = (
                progression_next_idx
            )

            if (
                progression_next_idx
                > line_start_idx
            ):
                window_start = (
                    event_day_int
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

            pfs_time = (
                event_day_int
                - line_start
            )

        elif last_n is not None:
            next_idx = split_idx
            event_day = int(
                last_n
            )

            pfs_event = 0

            pfs_time = (
                event_day
                - line_start
            )

            source = (
                "labelled_by_"
                "non_progression"
            )

        else:
            next_idx = split_idx
            event_day = line_start
            pfs_time = 0
            pfs_event = -1

            source = (
                "no_radiology_follow_up"
            )

        lines[
            line_start_idx:
            next_idx
        ] = current_line

        if (
            next_idx == n_tx
            and pfs_event <= 0
            and death_time is not None
        ):
            if death_time < line_start:
                raise RuntimeError(
                    f"death before line start "
                    f"for {pid}"
                )

            event_day = int(
                death_time
            )

            pfs_event = 1

            pfs_time = (
                event_day
                - line_start
            )

            source = (
                "labelled_by_death_event"
            )

        original_event = None
        original_time = None

        if not has_prior_measurement(
            prior_measurements,
            pid,
            line_start,
        ):
            original_event = (
                int(pfs_event)
            )

            original_time = (
                int(
                    max(
                        0,
                        pfs_time,
                    )
                )
            )

            pfs_event = -1

            source = (
                "no_radiology_within_"
                f"{RADIOLOGY_LOOKBACK_DAYS}"
                "_prior"
            )

        rows.append(
            {
                "PATIENT_ID":
                    pid,

                "LINE":
                    current_line,

                "LINE_START":
                    line_start,

                "EVENT_DAY":
                    int(
                        event_day
                    ),

                "PFS_TIME_DAYS":
                    int(
                        max(
                            0,
                            pfs_time,
                        )
                    ),

                "PFS_EVENT":
                    int(
                        pfs_event
                    ),

                "LINE_SOURCE":
                    source,

                "ORIGINAL_PFS_TIME_DAYS":
                    original_time,

                "ORIGINAL_PFS_EVENT":
                    original_event,
            }
        )

        if next_idx >= n_tx:
            break

        if next_idx <= idx:
            raise RuntimeError(
                f"non-advancing line index "
                f"pid={pid} idx={idx} next={next_idx}"
            )

        idx = next_idx
        current_line += 1

    tx["LINE"] = lines

    ###########################################################################
    # Match public treatment-row inflation:
    # therapies continuing past later line starts are visible in those lines.
    ###########################################################################

    if (
        "STOP_DATE" in tx.columns
        and rows
    ):
        line_starts = (
            pd.DataFrame(
                rows
            )[
                [
                    "LINE",
                    "LINE_START",
                ]
            ]
        )

        inflated = []

        for _, row in tx.iterrows():
            stop_day = row.get(
                "STOP_DATE"
            )

            if pd.isna(
                stop_day
            ):
                continue

            current = int(
                row["LINE"]
            )

            later = line_starts[
                line_starts[
                    "LINE"
                ] > current
            ]

            later = later[
                later[
                    "LINE_START"
                ]
                < float(
                    stop_day
                )
            ]

            for new_line in later[
                "LINE"
            ]:
                dup = row.copy()
                dup["LINE"] = int(
                    new_line
                )

                inflated.append(
                    dup
                )

        if inflated:
            tx = pd.concat(
                [
                    tx,
                    pd.DataFrame(
                        inflated
                    ),
                ],
                ignore_index=True,
            )

    tx = (
        tx.sort_values(
            [
                "START_DATE",
                "LINE",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return tx, rows


def finalize_pfs(
    all_lines: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    pfs = all_lines.copy()

    # PFS_TIME_DAYS must be floating-point because an event exactly at the
    # 730-day administrative horizon is represented as 729.999 so that it
    # remains an event immediately before the censoring boundary.
    pfs["PFS_TIME_DAYS"] = pd.to_numeric(
        pfs["PFS_TIME_DAYS"],
        errors="raise",
    ).astype("float64")

    pfs["PFS_EVENT"] = pd.to_numeric(
        pfs["PFS_EVENT"],
        errors="raise",
    ).astype("int64")

    too_short = (
        (
            pfs[
                "PFS_TIME_DAYS"
            ]
            < GRACE_DAYS
        )
        & (
            pfs[
                "PFS_EVENT"
            ]
            != -1
        )
    )

    n_too_short = int(
        too_short.sum()
    )

    pfs.loc[
        too_short,
        "LINE_SOURCE",
    ] = (
        "line_too_short_"
        "for_effectiveness"
    )

    pfs.loc[
        too_short,
        "PFS_EVENT",
    ] = -1

    # Administrative censoring.
    longer = (
        (
            pfs[
                "PFS_TIME_DAYS"
            ]
            > ADMIN_CENSOR_DAYS
        )
        & (
            pfs[
                "PFS_EVENT"
            ]
            != -1
        )
    )

    n_admin = int(
        longer.sum()
    )

    pfs.loc[
        longer,
        "PFS_TIME_DAYS",
    ] = ADMIN_CENSOR_DAYS

    pfs.loc[
        longer,
        "PFS_EVENT",
    ] = 0

    exact_event = (
        (
            pfs[
                "PFS_TIME_DAYS"
            ]
            == ADMIN_CENSOR_DAYS
        )
        & (
            pfs[
                "PFS_EVENT"
            ]
            == 1
        )
    )

    pfs.loc[
        exact_event,
        "PFS_TIME_DAYS",
    ] = (
        ADMIN_CENSOR_DAYS
        - 0.001
    )

    dropped = (
        pfs[
            pfs[
                "PFS_EVENT"
            ]
            == -1
        ]
        .reset_index(drop=True)
    )

    usable = (
        pfs[
            pfs[
                "PFS_EVENT"
            ]
            != -1
        ]
        .reset_index(drop=True)
    )

    stats = {
        "line_too_short":
            n_too_short,

        "administratively_censored":
            n_admin,

        "dropped_lines":
            len(dropped),

        "usable_lines":
            len(usable),

        "usable_patients":
            int(
                usable[
                    "PATIENT_ID"
                ].nunique()
            ),
    }

    return (
        usable,
        dropped,
        stats,
    )


def normalize_checkpoint1_lines(
    path: Path,
) -> pd.DataFrame:
    df = pd.read_parquet(
        path
    )

    rename = {}

    for col in df.columns:
        low = str(
            col
        ).lower()

        if low == "patient_id":
            rename[col] = (
                "PATIENT_ID"
            )

        elif low == "line":
            rename[col] = "LINE"

        elif low in {
            "line_start_day",
            "line_start",
        }:
            rename[col] = (
                "LINE_START"
            )

        elif low in {
            "line_event_day",
            "event_day",
        }:
            rename[col] = (
                "EVENT_DAY"
            )

        elif low in {
            "line_source",
        }:
            rename[col] = (
                "LINE_SOURCE"
            )

    df = df.rename(
        columns=rename
    )

    required = {
        "PATIENT_ID",
        "LINE",
        "LINE_START",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Checkpoint-1 line table "
            f"missing normalized fields: {missing}"
        )

    return df


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/checkpoint1b"
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

    chord_root = (
        repo
        / "data"
        / "external_sources"
        / "msk_chord_2024"
        / "raw"
    )

    ckpt1_lines_path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_lines.parquet"
    )

    if not chord_root.exists():
        raise SystemExit(
            "[CKPT1B_FATAL] "
            f"CHORD root missing: {chord_root}"
        )

    if not ckpt1_lines_path.exists():
        raise SystemExit(
            "[CKPT1B_FATAL] "
            "Checkpoint-1 line table missing"
        )

    ###########################################################################
    # Stage A: exact public-style cohort filtering.
    ###########################################################################

    print(
        "[CKPT1B] "
        "Reconstructing strict public-style cohort...",
        flush=True,
    )

    (
        diagnosis,
        breast_only,
        stage4,
    ) = prep_diagnosis(
        chord_root
    )

    first_meta = (
        first_metastasis_dates(
            chord_root,
            breast_only,
        )
    )

    treatment = prep_treatment(
        chord_root,
        breast_only,
        stage4,
        first_meta,
    )

    progression = (
        prep_progression(
            chord_root,
            breast_only,
            stage4,
            first_meta,
        )
    )

    tx_patients = set(
        treatment[
            "PATIENT_ID"
        ]
    )

    prog_patients = set(
        progression[
            "PATIENT_ID"
        ]
    )

    strict_patients = (
        tx_patients
        & prog_patients
    )

    # Critical difference from broad CKPT1.
    treatment = (
        treatment[
            treatment[
                "PATIENT_ID"
            ].isin(
                strict_patients
            )
        ]
        .reset_index(drop=True)
    )

    progression = (
        progression[
            progression[
                "PATIENT_ID"
            ].isin(
                strict_patients
            )
        ]
        .reset_index(drop=True)
    )

    print(
        "[CKPT1B] "
        f"breast_only={len(breast_only)} "
        f"stage4={len(stage4)} "
        f"first_metastasis={len(first_meta)} "
        f"postmeta_tx_patients={len(tx_patients)} "
        f"postmeta_YN_progression_patients={len(prog_patients)} "
        f"strict_intersection={len(strict_patients)}",
        flush=True,
    )

    ###########################################################################
    # Stage B: exact public-style line/PFS assignment.
    ###########################################################################

    deaths = prep_death_times(
        chord_root,
        strict_patients,
    )

    measurements = measurement_map(
        chord_root,
        strict_patients,
        progression,
    )

    progression_groups = {
        pid: grp[
            [
                "START_DATE",
                "PROGRESSION",
            ]
        ].reset_index(
            drop=True
        )
        for pid, grp
        in progression.groupby(
            "PATIENT_ID",
            observed=True,
        )
    }

    all_pfs_rows = []
    treatment_rows = []

    total_patients = len(
        strict_patients
    )

    for index, (
        pid,
        patient_tx,
    ) in enumerate(
        treatment.groupby(
            "PATIENT_ID",
            observed=True,
        ),
        start=1,
    ):
        if (
            index == 1
            or index % 500 == 0
            or index == total_patients
        ):
            print(
                "[CKPT1B] "
                f"patient={index}/"
                f"{total_patients}",
                flush=True,
            )

        patient_prog = (
            progression_groups.get(
                pid,
                pd.DataFrame(
                    columns=[
                        "START_DATE",
                        "PROGRESSION",
                    ]
                ),
            )
        )

        tx_labeled, pfs_rows = (
            assign_patient_lines(
                patient_tx,
                patient_prog,
                measurements,
                deaths.get(
                    pid
                ),
            )
        )

        treatment_rows.append(
            tx_labeled
        )

        all_pfs_rows.extend(
            pfs_rows
        )

    all_lines = (
        pd.DataFrame(
            all_pfs_rows
        )
        .sort_values(
            [
                "PATIENT_ID",
                "LINE",
            ]
        )
        .reset_index(drop=True)
    )

    strict_treatment = (
        pd.concat(
            treatment_rows,
            ignore_index=True,
        )
        if treatment_rows
        else pd.DataFrame()
    )

    (
        usable,
        dropped,
        final_stats,
    ) = finalize_pfs(
        all_lines
    )

    ###########################################################################
    # Stage C: normalized baseline-ready table.
    ###########################################################################

    baseline = usable.rename(
        columns={
            "PATIENT_ID":
                "patient_id",

            "LINE":
                "line",

            "LINE_START":
                "line_start_day",

            "EVENT_DAY":
                "event_day",

            "PFS_TIME_DAYS":
                "pfs_time_days",

            "PFS_EVENT":
                "pfs_event",

            "LINE_SOURCE":
                "line_source",
        }
    ).copy()

    baseline[
        "outcome_type"
    ] = np.where(
        baseline[
            "pfs_event"
        ] == 0,
        "CENSOR",
        np.where(
            baseline[
                "line_source"
            ]
            == "labelled_by_death_event",
            "DEATH",
            "PROGRESSION",
        ),
    )

    ###########################################################################
    # Stage D: compare to broad Checkpoint-1 reconstruction.
    ###########################################################################

    broad = (
        normalize_checkpoint1_lines(
            ckpt1_lines_path
        )
    )

    broad_patients = set(
        broad[
            "PATIENT_ID"
        ].astype(str)
    )

    strict_intersection_patients = set(
        strict_patients
    )

    usable_patients = set(
        usable[
            "PATIENT_ID"
        ].astype(str)
    )

    broad_keys = set(
        zip(
            broad[
                "PATIENT_ID"
            ].astype(str),
            broad[
                "LINE_START"
            ].astype(int),
        )
    )

    strict_all_keys = set(
        zip(
            all_lines[
                "PATIENT_ID"
            ].astype(str),
            all_lines[
                "LINE_START"
            ].astype(int),
        )
    )

    strict_usable_keys = set(
        zip(
            usable[
                "PATIENT_ID"
            ].astype(str),
            usable[
                "LINE_START"
            ].astype(int),
        )
    )

    source_counts_all = (
        all_lines[
            "LINE_SOURCE"
        ]
        .value_counts()
        .to_dict()
    )

    source_counts_dropped = (
        dropped[
            "LINE_SOURCE"
        ]
        .value_counts()
        .to_dict()
    )

    source_counts_usable = (
        usable[
            "LINE_SOURCE"
        ]
        .value_counts()
        .to_dict()
    )

    broad_extra_patients = (
        broad_patients
        - strict_intersection_patients
    )

    missing_from_broad = (
        strict_intersection_patients
        - broad_patients
    )

    comparison = {
        "cohort_steps": {
            "breast_only_patients":
                len(
                    breast_only
                ),

            "stage4_patients":
                len(
                    stage4
                ),

            "patients_with_first_metastasis":
                len(
                    first_meta
                ),

            "postmetastatic_treatment_patients":
                len(
                    tx_patients
                ),

            "postmetastatic_YN_progression_patients":
                len(
                    prog_patients
                ),

            "strict_treatment_progression_intersection":
                len(
                    strict_patients
                ),
        },

        "strict_reference": {
            "all_lines_before_evidence_filter":
                len(
                    all_lines
                ),

            "all_line_patients":
                int(
                    all_lines[
                        "PATIENT_ID"
                    ].nunique()
                ),

            "usable_lines":
                len(
                    usable
                ),

            "usable_patients":
                int(
                    usable[
                        "PATIENT_ID"
                    ].nunique()
                ),

            "dropped_lines":
                len(
                    dropped
                ),

            "line_too_short":
                final_stats[
                    "line_too_short"
                ],

            "administratively_censored":
                final_stats[
                    "administratively_censored"
                ],

            "line_sources_all":
                {
                    str(k): int(v)
                    for k, v
                    in source_counts_all.items()
                },

            "line_sources_dropped":
                {
                    str(k): int(v)
                    for k, v
                    in source_counts_dropped.items()
                },

            "line_sources_usable":
                {
                    str(k): int(v)
                    for k, v
                    in source_counts_usable.items()
                },
        },

        "checkpoint1_broad": {
            "lines":
                len(
                    broad
                ),

            "patients":
                int(
                    broad[
                        "PATIENT_ID"
                    ].nunique()
                ),

            "patients_not_in_strict_YN_intersection":
                len(
                    broad_extra_patients
                ),

            "strict_intersection_patients_missing_from_broad":
                len(
                    missing_from_broad
                ),
        },

        "line_start_overlap": {
            "broad_line_starts":
                len(
                    broad_keys
                ),

            "strict_all_line_starts":
                len(
                    strict_all_keys
                ),

            "strict_usable_line_starts":
                len(
                    strict_usable_keys
                ),

            "broad_vs_strict_all_exact_matches":
                len(
                    broad_keys
                    & strict_all_keys
                ),

            "broad_vs_strict_usable_exact_matches":
                len(
                    broad_keys
                    & strict_usable_keys
                ),

            "strict_all_missing_from_broad":
                len(
                    strict_all_keys
                    - broad_keys
                ),

            "broad_not_in_strict_all":
                len(
                    broad_keys
                    - strict_all_keys
                ),
        },

        "paper_analytic_reference": {
            "patients":
                PAPER_PATIENTS,

            "lines":
                PAPER_LINES,

            "strict_usable_patient_delta":
                int(
                    usable[
                        "PATIENT_ID"
                    ].nunique()
                )
                - PAPER_PATIENTS,

            "strict_usable_line_delta":
                len(
                    usable
                )
                - PAPER_LINES,

            "important_note": (
                "The paper's 2,881 / 8,791 numbers "
                "are the final analytic/design-matrix "
                "cohort. Strict usable PFS may still "
                "be larger if downstream feature/QC "
                "filters remove additional lines."
            ),
        },
    }

    ###########################################################################
    # Stage E: decide next scientific step.
    ###########################################################################

    strict_n_patients = int(
        usable[
            "PATIENT_ID"
        ].nunique()
    )

    strict_n_lines = len(
        usable
    )

    if (
        strict_n_patients
        == PAPER_PATIENTS
        and strict_n_lines
        == PAPER_LINES
    ):
        status = (
            "EXACT_REFERENCE_SCALE_MATCH"
        )

    elif (
        strict_n_patients
        >= PAPER_PATIENTS
        and strict_n_lines
        >= PAPER_LINES
    ):
        status = (
            "REFERENCE_PFS_RECONSTRUCTED_"
            "NEEDS_DOWNSTREAM_INCLUSION_AUDIT"
        )

    else:
        status = (
            "REFERENCE_RECONSTRUCTION_"
            "BELOW_PUBLISHED_SCALE"
        )

    comparison[
        "status"
    ] = status

    ###########################################################################
    # Stage F: persist without modifying CKPT1.
    ###########################################################################

    atomic_parquet(
        out
        / "reference_all_lines.parquet",
        all_lines,
    )

    atomic_parquet(
        out
        / "reference_usable_lines.parquet",
        baseline,
    )

    atomic_parquet(
        out
        / "reference_dropped_lines.parquet",
        dropped,
    )

    atomic_parquet(
        out
        / "reference_treatment_rows.parquet",
        strict_treatment,
    )

    atomic_json(
        out
        / "comparison.json",
        comparison,
    )

    atomic_text(
        out
        / "strict_patient_ids.txt",
        "\n".join(
            sorted(
                strict_intersection_patients
            )
        )
        + "\n",
    )

    atomic_text(
        out
        / "usable_patient_ids.txt",
        "\n".join(
            sorted(
                usable_patients
            )
        )
        + "\n",
    )

    ###########################################################################
    # Stage G: human-readable audit.
    ###########################################################################

    md = []

    md.append(
        "# Checkpoint 1B - "
        "Reference LoT/PFS Reconciliation"
    )
    md.append("")

    md.append(
        f"Status: **{status}**"
    )
    md.append("")

    md.append(
        "## Cohort narrowing"
    )
    md.append("")

    for key, value in comparison[
        "cohort_steps"
    ].items():
        md.append(
            f"- {key}: `{value}`"
        )

    md.append("")
    md.append(
        "## Strict reference-style PFS"
    )
    md.append("")

    for key in (
        "all_lines_before_evidence_filter",
        "all_line_patients",
        "usable_lines",
        "usable_patients",
        "dropped_lines",
        "line_too_short",
        "administratively_censored",
    ):
        md.append(
            f"- {key}: "
            f"`{comparison['strict_reference'][key]}`"
        )

    md.append("")
    md.append(
        "## Comparison with CKPT1 broad reconstruction"
    )
    md.append("")

    for key, value in comparison[
        "checkpoint1_broad"
    ].items():
        md.append(
            f"- {key}: `{value}`"
        )

    md.append("")
    md.append(
        "## Exact line-start comparison"
    )
    md.append("")

    for key, value in comparison[
        "line_start_overlap"
    ].items():
        md.append(
            f"- {key}: `{value}`"
        )

    md.append("")
    md.append(
        "## Published final analytic reference"
    )
    md.append("")

    md.append(
        f"- patients: `{PAPER_PATIENTS}`"
    )

    md.append(
        f"- mLoTs: `{PAPER_LINES}`"
    )

    md.append(
        "- The published values describe the "
        "final analytic cohort, so downstream "
        "feature/QC filters may still explain "
        "a remaining positive difference."
    )
    md.append("")

    atomic_text(
        out
        / "audit.md",
        "\n".join(md)
        + "\n",
    )

    ###########################################################################
    # Stage H: handoff.
    ###########################################################################

    handoff = {
        "checkpoint":
            "1B",

        "name":
            "reference_line_pfs_reconciliation",

        "status":
            status,

        "strict_usable_patients":
            strict_n_patients,

        "strict_usable_lines":
            strict_n_lines,

        "paper_reference_patients":
            PAPER_PATIENTS,

        "paper_reference_lines":
            PAPER_LINES,

        "checkpoint1_broad_patients":
            comparison[
                "checkpoint1_broad"
            ][
                "patients"
            ],

        "checkpoint1_broad_lines":
            comparison[
                "checkpoint1_broad"
            ][
                "lines"
            ],

        "artifacts": {
            "comparison":
                (
                    "artifacts/checkpoint1b/"
                    "comparison.json"
                ),

            "strict_lines":
                (
                    "artifacts/checkpoint1b/"
                    "reference_usable_lines.parquet"
                ),

            "all_lines":
                (
                    "artifacts/checkpoint1b/"
                    "reference_all_lines.parquet"
                ),

            "treatment_rows":
                (
                    "artifacts/checkpoint1b/"
                    "reference_treatment_rows.parquet"
                ),
        },

        "next_action": (
            "If strict usable counts remain above "
            "2,881/8,791, reproduce downstream "
            "design-matrix inclusion/QC before "
            "training baselines. If exact, freeze "
            "reference line reconstruction and "
            "begin baseline replication."
        ),
    }

    handoff_path = (
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_01B.json"
    )

    atomic_json(
        handoff_path,
        handoff,
    )

    ###########################################################################
    # Concise decision packet.
    ###########################################################################

    print("")
    print(
        "========== CKPT1B SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        f"breast_only_patients={len(breast_only)}"
    )

    print(
        f"stage4_patients={len(stage4)}"
    )

    print(
        "patients_with_first_metastasis="
        f"{len(first_meta)}"
    )

    print(
        "postmeta_treatment_patients="
        f"{len(tx_patients)}"
    )

    print(
        "postmeta_YN_progression_patients="
        f"{len(prog_patients)}"
    )

    print(
        "strict_treatment_progression_intersection="
        f"{len(strict_patients)}"
    )

    print(
        "strict_all_lines="
        f"{len(all_lines)}"
    )

    print(
        "strict_usable_patients="
        f"{strict_n_patients}"
    )

    print(
        "strict_usable_lines="
        f"{strict_n_lines}"
    )

    print(
        "strict_dropped_lines="
        f"{len(dropped)}"
    )

    print(
        "line_too_short="
        f"{final_stats['line_too_short']}"
    )

    print(
        "administratively_censored="
        f"{final_stats['administratively_censored']}"
    )

    print(
        "paper_reference_patients="
        f"{PAPER_PATIENTS}"
    )

    print(
        "paper_reference_lines="
        f"{PAPER_LINES}"
    )

    print(
        "patient_delta="
        f"{strict_n_patients - PAPER_PATIENTS}"
    )

    print(
        "line_delta="
        f"{strict_n_lines - PAPER_LINES}"
    )

    print(
        "========== CKPT1B SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT1B DECISION PACKET =========="
    )

    print(
        "strict_line_sources_all="
        f"{comparison['strict_reference']['line_sources_all']}"
    )

    print(
        "strict_line_sources_dropped="
        f"{comparison['strict_reference']['line_sources_dropped']}"
    )

    print(
        "strict_line_sources_usable="
        f"{comparison['strict_reference']['line_sources_usable']}"
    )

    print(
        "checkpoint1_broad="
        f"{comparison['checkpoint1_broad']}"
    )

    print(
        "line_start_overlap="
        f"{comparison['line_start_overlap']}"
    )

    print(
        "paper_comparison="
        f"{comparison['paper_analytic_reference']}"
    )

    print(
        "comparison_json="
        "artifacts/checkpoint1b/comparison.json"
    )

    print(
        "strict_lines="
        "artifacts/checkpoint1b/reference_usable_lines.parquet"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_01B.json"
    )

    print(
        "========== CKPT1B DECISION PACKET END =========="
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
