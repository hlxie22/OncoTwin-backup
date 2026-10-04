#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    DataLoader,
    Dataset,
)


###############################################################################
# Frozen CKPT4 configuration.
###############################################################################

SEED = 20260926

MAX_SEQ = 512

TRAIN_STRIDE = 384
EVAL_STRIDE = 512

HIDDEN = 192
LAYERS = 4
HEADS = 6
FF = 768
DROPOUT = 0.15

PAYLOAD_BUCKETS = 8192
SOURCE_BUCKETS = 128
LINE_BUCKETS = 16

SITE_VOCAB_SIZE = 32

# scan:
# 3 state values
# 1 state-observed flag
# report-count
# site-count
# site-observed flag
# 5 region-imaged
# 5 region-flag-observed
SCAN_FEATURE_DIM = 17


###############################################################################
# Event vocabulary.
###############################################################################

PAD_ID = 0
START_ID = 1
DAY_END_ID = 2
MASK_ID = 3

REAL_EVENT_NAMES = [
    "DIAGNOSIS",
    "TREATMENT_START",
    "TREATMENT_END",
    "RADIATION",
    "SURGERY",
    "PRIOR_MEDICATION",
    "TUMOR_MARKER",
    "PERFORMANCE_STATUS",
    "PDL1",
    "MMR",
    "GLEASON",
    "SPECIMEN",
    "SPECIMEN_SURGERY",
    "GENOMIC_RESULT_AVAILABLE",
    "SCAN_EPISODE",
    "OTHER",
]

EVENT_TO_ID = {
    name:
        index + 4
    for index, name
    in enumerate(
        REAL_EVENT_NAMES
    )
}

N_REAL_EVENTS = len(
    REAL_EVENT_NAMES
)

N_TYPES = (
    N_REAL_EVENTS
    + 4
)

SCAN_NP = 0
SCAN_P = 1
SCAN_IND = 2
SCAN_UNKNOWN = -1

REGIONS = (
    "chest",
    "abdomen",
    "pelvis",
    "head",
    "other",
)

RADIOLOGY_FILENAMES = {
    "data_timeline_progression.txt",
    "data_timeline_cancer_presence.txt",
    "data_timeline_tumor_sites.txt",
}


###############################################################################
# Generic helpers.
###############################################################################


def norm(
    value: Any,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(
            value
        )
        .strip()
        .lower(),
    ).strip(
        "_"
    )


def stable_hash(
    value: str,
) -> int:

    return int(
        hashlib.sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()[
            :16
        ],
        16,
    )


def json_default(
    value: Any,
):

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    raise TypeError(
        type(
            value
        ).__name__
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
        path.suffix
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size
        == 0
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
        path.suffix
        + ".tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size
        == 0
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
        path.suffix
        + ".tmp"
    )

    frame.to_parquet(
        tmp,
        index=False,
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size
        == 0
    ):
        raise RuntimeError(
            f"empty parquet write: {path}"
        )

    check = pd.read_parquet(
        tmp
    )

    if len(
        check
    ) != len(
        frame
    ):
        raise RuntimeError(
            f"parquet row mismatch: {path}"
        )

    tmp.replace(
        path
    )


def find_col(
    columns: Iterable[str],
    candidates: Iterable[str],
    required: bool = True,
) -> str | None:

    lookup = {
        norm(
            column
        ):
            column
        for column
        in columns
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
            "missing requested column; "
            f"requested={list(candidates)} "
            f"available={list(columns)}"
        )

    return None


def read_tsv(
    path: Path,
) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def parse_day(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def identifier_or_time_column(
    column: str,
) -> bool:

    nc = norm(
        column
    )

    return (
        nc.endswith(
            "_id"
        )
        or nc
        in {
            "id",
            "patient_id",
            "sample_id",
        }
        or any(
            token
            in nc
            for token
            in (
                "date",
                "day",
                "time",
                "timestamp",
                "availability",
            )
        )
    )


def parse_boolish(
    value: Any,
) -> tuple[
    float,
    float,
]:

    text = (
        str(
            value
        )
        .strip()
        .upper()
    )

    if text in {
        "",
        "NA",
        "NAN",
        "NONE",
        "UNKNOWN",
        "NOT AVAILABLE",
        "NOT_REPORTED",
    }:

        return (
            0.0,
            0.0,
        )

    if text in {
        "1",
        "Y",
        "YES",
        "TRUE",
        "T",
        "PRESENT",
        "IMAGED",
        "COVERED",
    }:

        return (
            1.0,
            1.0,
        )

    if text in {
        "0",
        "N",
        "NO",
        "FALSE",
        "F",
        "ABSENT",
        "NOT IMAGED",
        "UNCOVERED",
    }:

        return (
            0.0,
            1.0,
        )

    try:

        value = float(
            text
        )

        if math.isfinite(
            value
        ):

            return (
                float(
                    value
                    != 0.0
                ),
                1.0,
            )

    except Exception:
        pass

    return (
        0.0,
        0.0,
    )


def numeric_from_row(
    row: pd.Series,
    columns: list[str],
) -> tuple[
    float,
    int,
]:

    columns = [
        column
        for column
        in columns
        if not identifier_or_time_column(
            column
        )
    ]

    preferred = []

    secondary = []

    for column in columns:

        nc = norm(
            column
        )

        if any(
            key in nc
            for key
            in (
                "result",
                "value",
                "score",
                "level",
                "percent",
                "status",
            )
        ):

            preferred.append(
                column
            )

        else:

            secondary.append(
                column
            )

    for column in (
        preferred
        + secondary
    ):

        try:

            text = str(
                row[
                    column
                ]
            ).strip()

            if not text:
                continue

            value = float(
                text
            )

            if math.isfinite(
                value
            ):

                # Signed log transform is intentionally applied
                # before neural normalization.
                return (
                    float(
                        np.sign(
                            value
                        )
                        * np.log1p(
                            abs(
                                value
                            )
                        )
                    ),
                    1,
                )

        except Exception:
            continue

    return (
        0.0,
        0,
    )


def payload_string(
    row: pd.Series,
    columns: list[str],
    exclude: set[str],
) -> str:

    preferred_tokens = (
        "drug",
        "treatment",
        "regimen",
        "therapy",
        "result",
        "status",
        "state",
        "site",
        "marker",
        "test",
        "procedure",
        "modality",
        "diagnosis",
        "class",
        "subtype",
        "description",
        "name",
        "value",
    )

    usable = [
        column
        for column
        in columns
        if (
            column
            not in exclude
            and not identifier_or_time_column(
                column
            )
        )
    ]

    ordered = sorted(
        usable,
        key=lambda column: (
            0
            if any(
                token
                in norm(
                    column
                )
                for token
                in preferred_tokens
            )
            else 1,
            norm(
                column
            ),
        ),
    )

    parts = []

    for column in ordered[
        :12
    ]:

        value = (
            str(
                row[
                    column
                ]
            )
            .strip()
        )

        if (
            value
            and value.upper()
            not in {
                "NA",
                "NAN",
                "NONE",
                "UNKNOWN",
            }
        ):

            parts.append(
                f"{norm(column)}={value[:120]}"
            )

    return "|".join(
        parts
    )[
        :900
    ]


def payload_id(
    text: str,
) -> int:

    if not text:
        return 0

    return (
        stable_hash(
            text
        )
        % (
            PAYLOAD_BUCKETS
            - 1
        )
        + 1
    )


def source_id(
    text: str,
) -> int:

    if not text:
        return 0

    return (
        stable_hash(
            text
        )
        % (
            SOURCE_BUCKETS
            - 1
        )
        + 1
    )


def base_event_row(
    patient_id: str,
    day: int,
    event_name: str,
    payload: str = "",
    source: str = "",
    numeric_value: float = 0.0,
    numeric_observed: int = 0,
    line: int = 0,
    scan_state: int = SCAN_UNKNOWN,
    site_mask: int = 0,
    site_known: int = 0,
    region_imaged_mask: int = 0,
    region_known_mask: int = 0,
    scan_report_count: int = 0,
    scan_site_count: int = 0,
    scan_episode_id: str = "",
) -> dict[
    str,
    Any,
]:

    return {
        "patient_id":
            str(
                patient_id
            ),

        "day":
            int(
                day
            ),

        "type_id":
            int(
                EVENT_TO_ID.get(
                    event_name,
                    EVENT_TO_ID[
                        "OTHER"
                    ],
                )
            ),

        "event_name":
            (
                event_name
                if event_name
                in EVENT_TO_ID
                else "OTHER"
            ),

        "payload_id":
            int(
                payload_id(
                    payload
                )
            ),

        "source_id":
            int(
                source_id(
                    source
                )
            ),

        "numeric_value":
            float(
                numeric_value
            ),

        "numeric_observed":
            int(
                numeric_observed
            ),

        "line":
            int(
                max(
                    0,
                    min(
                        int(
                            line
                        ),
                        LINE_BUCKETS
                        - 1,
                    ),
                )
            ),

        "scan_state":
            int(
                scan_state
            ),

        "site_mask":
            int(
                site_mask
            ),

        "site_known":
            int(
                site_known
            ),

        "region_imaged_mask":
            int(
                region_imaged_mask
            ),

        "region_known_mask":
            int(
                region_known_mask
            ),

        "scan_report_count":
            int(
                scan_report_count
            ),

        "scan_site_count":
            int(
                scan_site_count
            ),

        "scan_episode_id":
            str(
                scan_episode_id
            ),
    }


###############################################################################
# Data discovery / splits.
###############################################################################


def discover_raw_root(
    root: Path,
) -> Path:

    candidates = [
        path.parent
        for path
        in root.rglob(
            "data_clinical_patient.txt"
        )
    ]

    if not candidates:

        raise RuntimeError(
            "could not locate CHORD "
            "data_clinical_patient.txt"
        )

    candidates.sort(
        key=lambda path:
            len(
                str(
                    path
                )
            )
    )

    return candidates[
        0
    ]


def load_mbc_splits(
    path: Path,
) -> pd.DataFrame:

    frame = pd.read_parquet(
        path
    )

    pid = find_col(
        frame.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    split = find_col(
        frame.columns,
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
                "validation":
                    "val",

                "valid":
                    "val",

                "dev":
                    "val",

                "training":
                    "train",

                "testing":
                    "test",
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

    if (
        out[
            "patient_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "patient assigned to more "
            "than one CKPT1 split"
        )

    return out


def make_pan_split(
    all_patients: list[str],
    mbc_splits: pd.DataFrame,
) -> dict[
    str,
    str,
]:

    # Exact CKPT1 breast splits override hashing.
    # Thus no held-out mBC patient can enter pan-cancer pretraining.
    override = dict(
        zip(
            mbc_splits[
                "patient_id"
            ],
            mbc_splits[
                "split"
            ],
        )
    )

    mapping = {}

    for patient in all_patients:

        if patient in override:

            mapping[
                patient
            ] = override[
                patient
            ]

            continue

        bucket = (
            stable_hash(
                patient
            )
            % 100
        )

        if bucket < 80:
            split = "train"

        elif bucket < 90:
            split = "val"

        else:
            split = "test"

        mapping[
            patient
        ] = split

    return mapping


###############################################################################
# Pan-cancer CHORD event construction.
###############################################################################


def raw_event_name(
    filename: str,
    end: bool = False,
) -> str:

    name = (
        filename
        .lower()
    )

    if "diagnosis" in name:
        return "DIAGNOSIS"

    if "treatment" in name:
        return (
            "TREATMENT_END"
            if end
            else "TREATMENT_START"
        )

    if "radiation" in name:
        return "RADIATION"

    if (
        "surgery" in name
        and "specimen"
        not in name
    ):
        return "SURGERY"

    if "prior_meds" in name:
        return "PRIOR_MEDICATION"

    if any(
        token in name
        for token
        in (
            "ca_15",
            "ca_19",
            "cea_labs",
            "psa_labs",
        )
    ):
        return "TUMOR_MARKER"

    if (
        "performance_status"
        in name
    ):
        return "PERFORMANCE_STATUS"

    if "pdl1" in name:
        return "PDL1"

    if "mmr" in name:
        return "MMR"

    if "gleason" in name:
        return "GLEASON"

    if (
        "specimen_surgery"
        in name
    ):
        return "SPECIMEN_SURGERY"

    if "specimen" in name:
        return "SPECIMEN"

    return "OTHER"


def classify_progression_text(
    values: Iterable[Any],
) -> int:

    texts = [
        str(
            value
        )
        .strip()
        .upper()
        for value
        in values
        if str(
            value
        ).strip()
    ]

    positives = {
        "Y",
        "YES",
        "TRUE",
        "1",
        "PD",
        "PROGRESSIVE",
        "PROGRESSION",
    }

    negatives = {
        "N",
        "NO",
        "FALSE",
        "0",
        "SD",
        "STABLE",
        "RESPONDING",
        "RESPONSE",
        "NON_PROGRESSIVE",
    }

    for text in texts:

        if text in positives:
            return SCAN_P

        if (
            "PROGRESS"
            in text
            and "NO PROGRESS"
            not in text
            and "NON-PROGRESS"
            not in text
        ):

            return SCAN_P

    for text in texts:

        if text in negatives:
            return SCAN_NP

        if any(
            token in text
            for token
            in (
                "STABLE",
                "RESPOND",
                "NO PROGRESS",
                "NON-PROGRESS",
            )
        ):

            return SCAN_NP

    return SCAN_UNKNOWN


def extract_site_label(
    row: pd.Series,
    columns: list[str],
    exclude: set[str],
) -> str:

    preferred = [
        column
        for column
        in columns
        if (
            column
            not in exclude
            and "site"
            in norm(
                column
            )
        )
    ]

    other = [
        column
        for column
        in columns
        if (
            column
            not in exclude
            and column
            not in preferred
            and not identifier_or_time_column(
                column
            )
        )
    ]

    for column in (
        preferred
        + other
    ):

        value = (
            str(
                row[
                    column
                ]
            )
            .strip()
        )

        if (
            value
            and value.upper()
            not in {
                "NA",
                "NAN",
                "NONE",
                "UNKNOWN",
                "0",
                "1",
                "Y",
                "N",
            }
        ):

            return value.upper()[
                :120
            ]

    return ""


def scan_episode_group(
    days: list[int],
    window: int = 3,
) -> list[
    list[int]
]:

    if not days:
        return []

    days = sorted(
        set(
            int(
                value
            )
            for value
            in days
        )
    )

    groups = []

    current = [
        days[
            0
        ]
    ]

    start = days[
        0
    ]

    for day in days[
        1:
    ]:

        # Bounded 3-day episode anchored at first report.
        if (
            day
            - start
            <= window
        ):

            current.append(
                day
            )

        else:

            groups.append(
                current
            )

            current = [
                day
            ]

            start = day

    groups.append(
        current
    )

    return groups


def prepare_pan_cancer(
    raw_root: Path,
    mbc_splits: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:

    patient_file = (
        raw_root
        / "data_clinical_patient.txt"
    )

    patients = read_tsv(
        patient_file
    )

    pid_col = find_col(
        patients.columns,
        [
            "PATIENT_ID",
            "patient_id",
        ],
    )

    all_patients = sorted(
        set(
            patients[
                pid_col
            ]
            .astype(str)
            .str.strip()
        )
        - {
            "",
        }
    )

    split_map = make_pan_split(
        all_patients,
        mbc_splits,
    )

    timeline_files = sorted(
        raw_root.glob(
            "data_timeline_*.txt"
        )
    )

    for filename in (
        RADIOLOGY_FILENAMES
    ):

        if not (
            raw_root
            / filename
        ).exists():

            raise RuntimeError(
                "missing required CHORD "
                f"radiology table: {filename}"
            )

    ###########################################################################
    # Build pan-cancer 3-day scan episodes from radiology-derived tables.
    ###########################################################################

    scan_days_by_patient = defaultdict(
        set
    )

    progression_by_patient_day = defaultdict(
        list
    )

    site_labels_by_patient_day = defaultdict(
        list
    )

    report_rows_by_patient_day = Counter()

    for filename in sorted(
        RADIOLOGY_FILENAMES
    ):

        path = (
            raw_root
            / filename
        )

        frame = read_tsv(
            path
        )

        pid = find_col(
            frame.columns,
            [
                "PATIENT_ID",
                "patient_id",
            ],
        )

        start = find_col(
            frame.columns,
            [
                "START_DATE",
                "start_date",
            ],
        )

        stop = find_col(
            frame.columns,
            [
                "STOP_DATE",
                "stop_date",
            ],
            required=False,
        )

        exclude = {
            pid,
            start,
        }

        if stop:
            exclude.add(
                stop
            )

        days = parse_day(
            frame[
                start
            ]
        )

        valid = days.notna()

        frame = (
            frame.loc[
                valid
            ]
            .copy()
        )

        frame[
            "_DAY"
        ] = (
            days[
                valid
            ]
            .astype(int)
            .to_numpy()
        )

        if (
            "progression"
            in filename
        ):

            progression_columns = [
                column
                for column
                in frame.columns
                if (
                    column
                    not in exclude
                    and column
                    != "_DAY"
                    and any(
                        token
                        in norm(
                            column
                        )
                        for token
                        in (
                            "progress",
                            "state",
                            "status",
                            "response",
                        )
                    )
                )
            ]

            if not progression_columns:

                progression_columns = [
                    column
                    for column
                    in frame.columns
                    if (
                        column
                        not in exclude
                        and column
                        != "_DAY"
                    )
                ]

        else:

            progression_columns = []

        for _, row in frame.iterrows():

            patient = (
                str(
                    row[
                        pid
                    ]
                )
                .strip()
            )

            if not patient:
                continue

            day = int(
                row[
                    "_DAY"
                ]
            )

            scan_days_by_patient[
                patient
            ].add(
                day
            )

            report_rows_by_patient_day[
                (
                    patient,
                    day,
                )
            ] += 1

            if (
                "progression"
                in filename
            ):

                progression_by_patient_day[
                    (
                        patient,
                        day,
                    )
                ].append(
                    classify_progression_text(
                        [
                            row[
                                column
                            ]
                            for column
                            in progression_columns
                        ]
                    )
                )

            if (
                "tumor_sites"
                in filename
            ):

                label = extract_site_label(
                    row,
                    list(
                        frame.columns
                    ),
                    exclude
                    | {
                        "_DAY",
                    },
                )

                if label:

                    site_labels_by_patient_day[
                        (
                            patient,
                            day,
                        )
                    ].append(
                        label
                    )

    episode_records = []

    for patient, day_set in (
        scan_days_by_patient.items()
    ):

        for episode_number, days in enumerate(
            scan_episode_group(
                list(
                    day_set
                ),
                3,
            ),
            start=1,
        ):

            end_day = max(
                days
            )

            states = []

            sites = []

            report_count = 0

            for day in days:

                states.extend(
                    progression_by_patient_day.get(
                        (
                            patient,
                            day,
                        ),
                        [],
                    )
                )

                sites.extend(
                    site_labels_by_patient_day.get(
                        (
                            patient,
                            day,
                        ),
                        [],
                    )
                )

                report_count += (
                    report_rows_by_patient_day.get(
                        (
                            patient,
                            day,
                        ),
                        0,
                    )
                )

            explicit_states = [
                state
                for state
                in states
                if state
                != SCAN_UNKNOWN
            ]

            if (
                SCAN_P
                in explicit_states
            ):

                state = SCAN_P

            elif (
                SCAN_NP
                in explicit_states
            ):

                state = SCAN_NP

            else:

                state = SCAN_UNKNOWN

            episode_records.append(
                {
                    "patient_id":
                        patient,

                    "day":
                        int(
                            end_day
                        ),

                    "scan_state":
                        state,

                    "site_labels":
                        sorted(
                            set(
                                sites
                            )
                        ),

                    "site_known":
                        int(
                            bool(
                                sites
                            )
                        ),

                    "report_count":
                        int(
                            report_count
                        ),

                    "scan_episode_id":
                        (
                            f"PAN::{patient}::"
                            f"{end_day}::{episode_number}"
                        ),
                }
            )

    episodes = pd.DataFrame(
        episode_records
    )

    if episodes.empty:

        raise RuntimeError(
            "no pan-cancer scan episodes constructed"
        )

    ###########################################################################
    # Site vocabulary learned ONLY from temporal-training patients.
    ###########################################################################

    site_counter = Counter()

    for _, row in episodes.iterrows():

        if (
            split_map.get(
                str(
                    row[
                        "patient_id"
                    ]
                )
            )
            != "train"
        ):
            continue

        site_counter.update(
            row[
                "site_labels"
            ]
        )

    site_vocab = [
        site
        for site, _
        in site_counter.most_common(
            SITE_VOCAB_SIZE
        )
    ]

    site_to_index = {
        site:
            index
        for index, site
        in enumerate(
            site_vocab
        )
    }

    ###########################################################################
    # Non-radiology structured events.
    ###########################################################################

    events = []

    for path in timeline_files:

        if (
            path.name
            in RADIOLOGY_FILENAMES
        ):
            continue

        frame = read_tsv(
            path
        )

        if frame.empty:
            continue

        pid = find_col(
            frame.columns,
            [
                "PATIENT_ID",
                "patient_id",
            ],
            required=False,
        )

        start = find_col(
            frame.columns,
            [
                "START_DATE",
                "start_date",
            ],
            required=False,
        )

        if (
            pid is None
            or start is None
        ):
            continue

        stop = find_col(
            frame.columns,
            [
                "STOP_DATE",
                "stop_date",
            ],
            required=False,
        )

        seq = find_col(
            frame.columns,
            [
                "SEQ_DATE",
                "seq_date",
            ],
            required=False,
        )

        days = parse_day(
            frame[
                start
            ]
        )

        numeric_columns = [
            column
            for column
            in frame.columns
            if column
            not in {
                pid,
                start,
                stop,
                seq,
            }
        ]

        for row_number, row in frame.iterrows():

            day_value = days.loc[
                row_number
            ]

            if pd.isna(
                day_value
            ):
                continue

            patient = (
                str(
                    row[
                        pid
                    ]
                )
                .strip()
            )

            if not patient:
                continue

            day = int(
                day_value
            )

            exclude = {
                pid,
                start,
            }

            if stop:
                exclude.add(
                    stop
                )

            if seq:
                exclude.add(
                    seq
                )

            payload = payload_string(
                row,
                list(
                    frame.columns
                ),
                exclude,
            )

            (
                numeric_value,
                numeric_observed,
            ) = numeric_from_row(
                row,
                numeric_columns,
            )

            event_name = raw_event_name(
                path.name,
                end=False,
            )

            events.append(
                base_event_row(
                    patient,
                    day,
                    event_name,
                    payload=payload,
                    source=path.name,
                    numeric_value=numeric_value,
                    numeric_observed=numeric_observed,
                )
            )

            ###################################################################
            # Treatment interval end.
            ###################################################################

            if (
                "treatment"
                in path.name.lower()
                and stop
            ):

                try:

                    stop_day = int(
                        float(
                            str(
                                row[
                                    stop
                                ]
                            )
                            .strip()
                        )
                    )

                except Exception:

                    stop_day = day

                if (
                    stop_day
                    > day
                ):

                    events.append(
                        base_event_row(
                            patient,
                            stop_day,
                            "TREATMENT_END",
                            payload=payload,
                            source=path.name,
                        )
                    )

            ###################################################################
            # Sequencing result availability proxy.
            ###################################################################

            if (
                "specimen_surgery"
                in path.name.lower()
                and seq
            ):

                try:

                    seq_day = int(
                        float(
                            str(
                                row[
                                    seq
                                ]
                            )
                            .strip()
                        )
                    )

                except Exception:

                    seq_day = None

                if seq_day is not None:

                    events.append(
                        base_event_row(
                            patient,
                            seq_day,
                            "GENOMIC_RESULT_AVAILABLE",
                            payload=payload,
                            source=path.name,
                        )
                    )

    ###########################################################################
    # Add scan episodes.
    ###########################################################################

    for _, row in episodes.iterrows():

        site_mask = 0

        for site in row[
            "site_labels"
        ]:

            index = (
                site_to_index.get(
                    site
                )
            )

            if index is not None:

                site_mask |= (
                    1
                    << index
                )

        events.append(
            base_event_row(
                str(
                    row[
                        "patient_id"
                    ]
                ),
                int(
                    row[
                        "day"
                    ]
                ),
                "SCAN_EPISODE",
                payload="|".join(
                    row[
                        "site_labels"
                    ][
                        :10
                    ]
                ),
                source="PAN_CHORD_SCAN_EPISODE_W3",
                scan_state=int(
                    row[
                        "scan_state"
                    ]
                ),
                site_mask=int(
                    site_mask
                ),
                site_known=int(
                    row[
                        "site_known"
                    ]
                ),
                scan_report_count=int(
                    row[
                        "report_count"
                    ]
                ),
                scan_site_count=int(
                    len(
                        row[
                            "site_labels"
                        ]
                    )
                ),
                scan_episode_id=str(
                    row[
                        "scan_episode_id"
                    ]
                ),
            )
        )

    base = pd.DataFrame(
        events
    )

    base[
        "split"
    ] = base[
        "patient_id"
    ].map(
        split_map
    )

    base = base[
        base[
            "split"
        ].notna()
    ].copy()

    base[
        "cohort"
    ] = "pan_cancer"

    return (
        base,
        site_vocab,
    )


###############################################################################
# Breast-specific CKPT1 event construction.
###############################################################################


def canonical_type_name(
    value: Any,
) -> str:

    name = (
        str(
            value
        )
        .strip()
        .upper()
    )

    aliases = {
        "PRIOR_MED":
            "PRIOR_MEDICATION",

        "PRIOR_MEDS":
            "PRIOR_MEDICATION",

        "GENOMIC_RESULT":
            "GENOMIC_RESULT_AVAILABLE",

        "SCAN":
            "SCAN_EPISODE",
    }

    name = aliases.get(
        name,
        name,
    )

    if name in EVENT_TO_ID:
        return name

    return "OTHER"


def scan_state_from_text(
    value: Any,
) -> int:

    text = (
        str(
            value
        )
        .strip()
        .upper()
    )

    if text in {
        "NON_PROGRESSIVE",
        "STABLE",
        "RESPONDING",
        "RESPONSE",
    }:

        return SCAN_NP

    if text in {
        "PROGRESSIVE",
        "PROGRESSION",
        "PD",
    }:

        return SCAN_P

    if text in {
        "INDETERMINATE",
        "MIXED",
        "UNKNOWN",
    }:

        return SCAN_IND

    return SCAN_UNKNOWN


def extract_region_masks(
    row: pd.Series,
    columns: list[str],
) -> tuple[
    int,
    int,
]:

    imaged = 0

    known = 0

    for region_index, region in enumerate(
        REGIONS
    ):

        candidates = []

        for column in columns:

            nc = norm(
                column
            )

            if region == "head":

                matches_region = any(
                    token
                    in nc
                    for token
                    in (
                        "head",
                        "brain",
                    )
                )

            else:

                matches_region = (
                    region
                    in nc
                )

            if (
                matches_region
                and any(
                    token
                    in nc
                    for token
                    in (
                        "coverage",
                        "imaged",
                        "scan",
                        "region",
                    )
                )
            ):

                candidates.append(
                    column
                )

        for column in candidates:

            (
                value,
                observed,
            ) = parse_boolish(
                row[
                    column
                ]
            )

            if observed:

                known |= (
                    1
                    << region_index
                )

                if value > 0:

                    imaged |= (
                        1
                        << region_index
                    )

                break

    return (
        imaged,
        known,
    )


def prepare_breast(
    canonical_path: Path,
    scans_path: Path,
    splits: pd.DataFrame,
    site_vocab: list[str],
) -> pd.DataFrame:

    canonical = pd.read_parquet(
        canonical_path
    )

    pid = find_col(
        canonical.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    event_type = find_col(
        canonical.columns,
        [
            "event_type",
            "EVENT_TYPE",
        ],
    )

    day_col = find_col(
        canonical.columns,
        [
            "availability_day",
            "event_day",
            "START_DATE",
        ],
    )

    line_col = find_col(
        canonical.columns,
        [
            "treatment_line",
            "line",
            "LINE",
        ],
        required=False,
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

    split_map = dict(
        zip(
            splits[
                "patient_id"
            ],
            splits[
                "split"
            ],
        )
    )

    events = []

    for _, row in canonical.iterrows():

        patient = (
            str(
                row[
                    pid
                ]
            )
            .strip()
        )

        if patient not in split_map:
            continue

        try:

            day = int(
                float(
                    row[
                        day_col
                    ]
                )
            )

        except Exception:

            continue

        name = canonical_type_name(
            row[
                event_type
            ]
        )

        # Authoritative W3 scans are inserted below.
        if name == "SCAN_EPISODE":
            continue

        line = 0

        if line_col is not None:

            try:

                line = int(
                    float(
                        row[
                            line_col
                        ]
                    )
                )

            except Exception:

                line = 0

        exclude = {
            pid,
            event_type,
            day_col,
        }

        if line_col:
            exclude.add(
                line_col
            )

        if source_col:
            exclude.add(
                source_col
            )

        payload = payload_string(
            row,
            list(
                canonical.columns
            ),
            exclude,
        )

        (
            numeric_value,
            numeric_observed,
        ) = numeric_from_row(
            row,
            [
                column
                for column
                in canonical.columns
                if column
                not in exclude
            ],
        )

        source = (
            str(
                row[
                    source_col
                ]
            )
            if source_col
            else name
        )

        events.append(
            base_event_row(
                patient,
                day,
                name,
                payload=payload,
                source=source,
                numeric_value=numeric_value,
                numeric_observed=numeric_observed,
                line=line,
            )
        )

    ###########################################################################
    # Replace scans with CKPT1 authoritative W3 episodes.
    ###########################################################################

    scans = pd.read_parquet(
        scans_path
    )

    spid = find_col(
        scans.columns,
        [
            "patient_id",
            "PATIENT_ID",
        ],
    )

    sline = find_col(
        scans.columns,
        [
            "treatment_line",
            "line",
            "LINE",
        ],
        required=False,
    )

    sday = find_col(
        scans.columns,
        [
            "landmark_day",
            "episode_end_day",
            "scan_day",
            "START_DATE",
        ],
    )

    sid = find_col(
        scans.columns,
        [
            "scan_episode_id",
            "episode_id",
        ],
        required=False,
    )

    sstate = find_col(
        scans.columns,
        [
            "progression_state_3",
            "scan_state",
            "state",
        ],
    )

    site_to_index = {
        site:
            index
        for index, site
        in enumerate(
            site_vocab
        )
    }

    for row_number, row in scans.iterrows():

        patient = (
            str(
                row[
                    spid
                ]
            )
            .strip()
        )

        if patient not in split_map:
            continue

        try:

            day = int(
                float(
                    row[
                        sday
                    ]
                )
            )

        except Exception:
            continue

        line = 0

        if sline:

            try:

                line = int(
                    float(
                        row[
                            sline
                        ]
                    )
                )

            except Exception:

                line = 0

        state = scan_state_from_text(
            row[
                sstate
            ]
        )

        (
            region_imaged,
            region_known,
        ) = extract_region_masks(
            row,
            list(
                scans.columns
            ),
        )

        #######################################################################
        # Conservative site transfer:
        # site targets are marked known only when a selected CHORD field
        # maps to one of the training-derived pan-cancer site labels.
        #######################################################################

        site_labels = []

        site_columns = [
            column
            for column
            in scans.columns
            if "site"
            in norm(
                column
            )
        ]

        for column in site_columns:

            value = (
                str(
                    row[
                        column
                    ]
                )
                .strip()
                .upper()
            )

            if (
                not value
                or value
                in {
                    "NA",
                    "NAN",
                    "NONE",
                    "UNKNOWN",
                }
            ):

                continue

            for site in site_vocab:

                if (
                    site in value
                    or value in site
                ):

                    site_labels.append(
                        site
                    )

        site_labels = sorted(
            set(
                site_labels
            )
        )

        site_mask = 0

        for site in site_labels:

            site_mask |= (
                1
                << site_to_index[
                    site
                ]
            )

        episode = ""

        if sid is not None:

            candidate = (
                str(
                    row[
                        sid
                    ]
                )
                .strip()
            )

            if (
                candidate
                and candidate.lower()
                not in {
                    "nan",
                    "none",
                }
            ):

                episode = candidate

        if not episode:

            episode = (
                f"BREAST::{patient}::"
                f"{day}::{row_number}"
            )

        payload = (
            "state="
            + str(
                row[
                    sstate
                ]
            )
            .strip()
            .upper()
        )

        events.append(
            base_event_row(
                patient,
                day,
                "SCAN_EPISODE",
                payload=payload,
                source="CKPT1_SCAN_EPISODE_W3",
                line=line,
                scan_state=state,
                site_mask=site_mask,
                site_known=int(
                    bool(
                        site_labels
                    )
                ),
                region_imaged_mask=region_imaged,
                region_known_mask=region_known,
                scan_report_count=1,
                scan_site_count=len(
                    site_labels
                ),
                scan_episode_id=episode,
            )
        )

    base = pd.DataFrame(
        events
    )

    base[
        "split"
    ] = base[
        "patient_id"
    ].map(
        split_map
    )

    base = base[
        base[
            "split"
        ].isin(
            [
                "train",
                "val",
                "test",
            ]
        )
    ].copy()

    base[
        "cohort"
    ] = "breast_mbc"

    return base


###############################################################################
# Sequence targets.
###############################################################################


def gap_bucket(
    days: int,
) -> int:

    value = max(
        int(
            days
        ),
        0,
    )

    edges = [
        0,
        1,
        3,
        7,
        14,
        30,
        60,
        120,
        240,
        480,
    ]

    for index, edge in enumerate(
        edges
    ):

        if value <= edge:
            return index

    return len(
        edges
    )


N_GAP_BINS = 11


def add_time_and_sequence_targets(
    base: pd.DataFrame,
) -> pd.DataFrame:

    if base.empty:

        raise RuntimeError(
            "empty base event table"
        )

    base = base.copy()

    priority = {
        EVENT_TO_ID[
            "DIAGNOSIS"
        ]:
            10,

        EVENT_TO_ID[
            "TREATMENT_END"
        ]:
            20,

        EVENT_TO_ID[
            "TREATMENT_START"
        ]:
            30,

        EVENT_TO_ID[
            "GENOMIC_RESULT_AVAILABLE"
        ]:
            40,

        EVENT_TO_ID[
            "TUMOR_MARKER"
        ]:
            50,

        EVENT_TO_ID[
            "PERFORMANCE_STATUS"
        ]:
            60,

        EVENT_TO_ID[
            "SCAN_EPISODE"
        ]:
            90,
    }

    base[
        "priority"
    ] = (
        base[
            "type_id"
        ]
        .map(
            priority
        )
        .fillna(
            70
        )
        .astype(int)
    )

    output = []

    for patient, group in base.groupby(
        "patient_id",
        sort=False,
    ):

        group = group.sort_values(
            [
                "day",
                "priority",
                "type_id",
                "payload_id",
            ],
            kind="mergesort",
        )

        split = str(
            group[
                "split"
            ]
            .iloc[
                0
            ]
        )

        cohort = str(
            group[
                "cohort"
            ]
            .iloc[
                0
            ]
        )

        days = sorted(
            group[
                "day"
            ]
            .astype(int)
            .unique()
        )

        day_to_events = {
            int(
                day
            ):
                frame.copy()
            for day, frame
            in group.groupby(
                "day",
                sort=True,
            )
        }

        first_day = int(
            days[
                0
            ]
        )

        previous_day = (
            first_day
            - 1
        )

        current_treatment_start = None

        #######################################################################
        # START token.
        #######################################################################

        start_row = base_event_row(
            patient,
            first_day
            - 1,
            "OTHER",
        )

        start_row[
            "type_id"
        ] = START_ID

        start_row[
            "event_name"
        ] = "START"

        start_row.update(
            {
                "split":
                    split,

                "cohort":
                    cohort,

                "gap_days":
                    0,

                "time_on_treatment":
                    0,

                "treatment_active":
                    0,

                "is_day_end":
                    0,

                "target_valid":
                    0,

                "next_type_bits":
                    0,

                "next_gap_bin":
                    -1,

                "next_scan_state":
                    -1,

                "next_site_mask":
                    0,

                "next_site_known":
                    0,
            }
        )

        output.append(
            start_row
        )

        #######################################################################
        # Event days.
        #######################################################################

        for day_index, day in enumerate(
            days
        ):

            day = int(
                day
            )

            day_events = (
                day_to_events[
                    day
                ]
            )

            gap = max(
                day
                - previous_day,
                0,
            )

            for _, row in day_events.iterrows():

                name = str(
                    row[
                        "event_name"
                    ]
                )

                if (
                    name
                    == "TREATMENT_START"
                ):

                    current_treatment_start = (
                        day
                    )

                if (
                    current_treatment_start
                    is None
                ):

                    time_on = 0
                    active = 0

                else:

                    time_on = max(
                        day
                        - int(
                            current_treatment_start
                        ),
                        0,
                    )

                    active = 1

                record = (
                    row
                    .to_dict()
                )

                record.update(
                    {
                        "gap_days":
                            int(
                                gap
                            ),

                        "time_on_treatment":
                            int(
                                time_on
                            ),

                        "treatment_active":
                            int(
                                active
                            ),

                        "is_day_end":
                            0,

                        "target_valid":
                            0,

                        "next_type_bits":
                            0,

                        "next_gap_bin":
                            -1,

                        "next_scan_state":
                            -1,

                        "next_site_mask":
                            0,

                        "next_site_known":
                            0,
                    }
                )

                output.append(
                    record
                )

                # Only the first token on a day receives a nonzero temporal gap.
                gap = 0

                if (
                    name
                    == "TREATMENT_END"
                ):

                    current_treatment_start = (
                        None
                    )

            ###################################################################
            # DAY_END token sees all same-day tokens under ordinary causal
            # attention and therefore serves as the current-day summary state.
            ###################################################################

            day_end = base_event_row(
                patient,
                day,
                "OTHER",
            )

            day_end[
                "type_id"
            ] = DAY_END_ID

            day_end[
                "event_name"
            ] = "DAY_END"

            day_end.update(
                {
                    "split":
                        split,

                    "cohort":
                        cohort,

                    "gap_days":
                        0,

                    "time_on_treatment":
                        (
                            max(
                                day
                                - int(
                                    current_treatment_start
                                ),
                                0,
                            )
                            if current_treatment_start
                            is not None
                            else 0
                        ),

                    "treatment_active":
                        int(
                            current_treatment_start
                            is not None
                        ),

                    "is_day_end":
                        1,

                    "target_valid":
                        0,

                    "next_type_bits":
                        0,

                    "next_gap_bin":
                        -1,

                    "next_scan_state":
                        -1,

                    "next_site_mask":
                        0,

                    "next_site_known":
                        0,
                }
            )

            ###################################################################
            # Next-observed-day self-supervision.
            ###################################################################

            if (
                day_index
                + 1
                < len(
                    days
                )
            ):

                next_day = int(
                    days[
                        day_index
                        + 1
                    ]
                )

                next_events = (
                    day_to_events[
                        next_day
                    ]
                )

                next_type_bits = 0

                for type_id in sorted(
                    set(
                        next_events[
                            "type_id"
                        ]
                        .astype(int)
                    )
                ):

                    real_index = (
                        int(
                            type_id
                        )
                        - 4
                    )

                    if (
                        0
                        <= real_index
                        < N_REAL_EVENTS
                    ):

                        next_type_bits |= (
                            1
                            << real_index
                        )

                scans = next_events[
                    next_events[
                        "type_id"
                    ]
                    == EVENT_TO_ID[
                        "SCAN_EPISODE"
                    ]
                ]

                next_scan_state = -1

                next_site_mask = 0

                next_site_known = 0

                if len(
                    scans
                ):

                    known_states = [
                        int(
                            value
                        )
                        for value
                        in scans[
                            "scan_state"
                        ]
                        if int(
                            value
                        )
                        >= 0
                    ]

                    if (
                        SCAN_P
                        in known_states
                    ):

                        next_scan_state = (
                            SCAN_P
                        )

                    elif (
                        SCAN_NP
                        in known_states
                    ):

                        next_scan_state = (
                            SCAN_NP
                        )

                    elif (
                        SCAN_IND
                        in known_states
                    ):

                        next_scan_state = (
                            SCAN_IND
                        )

                    for value in scans[
                        "site_mask"
                    ]:

                        next_site_mask |= int(
                            value
                        )

                    next_site_known = int(
                        scans[
                            "site_known"
                        ]
                        .astype(int)
                        .max()
                        > 0
                    )

                day_end[
                    "target_valid"
                ] = 1

                day_end[
                    "next_type_bits"
                ] = int(
                    next_type_bits
                )

                day_end[
                    "next_gap_bin"
                ] = int(
                    gap_bucket(
                        next_day
                        - day
                    )
                )

                day_end[
                    "next_scan_state"
                ] = int(
                    next_scan_state
                )

                day_end[
                    "next_site_mask"
                ] = int(
                    next_site_mask
                )

                day_end[
                    "next_site_known"
                ] = int(
                    next_site_known
                )

            output.append(
                day_end
            )

            previous_day = (
                day
            )

    result = pd.DataFrame(
        output
    )

    result = result.drop(
        columns=[
            "priority",
        ],
        errors="ignore",
    )

    return result


###############################################################################
# Preparation command.
###############################################################################


def token_stats(
    frame: pd.DataFrame,
) -> dict[
    str,
    Any,
]:

    patient_lengths = (
        frame
        .groupby(
            "patient_id"
        )
        .size()
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
                    "patient_id"
                ]
                .nunique()
            ),

        "split_patients": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in (
                frame[
                    [
                        "patient_id",
                        "split",
                    ]
                ]
                .drop_duplicates()
                [
                    "split"
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },

        "event_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in frame[
                "event_name"
            ]
            .value_counts()
            .to_dict()
            .items()
        },

        "patient_token_length": {
            "median":
                float(
                    patient_lengths.median()
                ),

            "p95":
                float(
                    patient_lengths.quantile(
                        0.95
                    )
                ),

            "max":
                int(
                    patient_lengths.max()
                ),

            "fraction_gt_512":
                float(
                    (
                        patient_lengths
                        > MAX_SEQ
                    )
                    .mean()
                ),
        },

        "known_scan_state_tokens":
            int(
                (
                    (
                        frame[
                            "type_id"
                        ]
                        == EVENT_TO_ID[
                            "SCAN_EPISODE"
                        ]
                    )
                    & (
                        frame[
                            "scan_state"
                        ]
                        >= 0
                    )
                )
                .sum()
            ),

        "day_end_targets":
            int(
                frame[
                    "target_valid"
                ]
                .sum()
            ),
    }


def prepare_command(
    repo: Path,
    out: Path,
) -> None:

    prep = (
        out
        / "prepared"
    )

    prep.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_root = discover_raw_root(
        repo
        / "data"
        / "external_sources"
        / "msk_chord_2024"
        / "raw"
    )

    split_path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_patient_splits.parquet"
    )

    canonical_path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_canonical_events.parquet"
    )

    scan_path = (
        repo
        / "artifacts"
        / "checkpoint1"
        / "canonical"
        / "chord_mbc_scan_episodes_w3.parquet"
    )

    mbc_splits = load_mbc_splits(
        split_path
    )

    print(
        "[CKPT4_PREP] building pan-cancer raw CHORD events",
        flush=True,
    )

    (
        pan_base,
        site_vocab,
    ) = prepare_pan_cancer(
        raw_root,
        mbc_splits,
    )

    print(
        "[CKPT4_PREP] constructing pan-cancer causal sequences",
        flush=True,
    )

    pan_tokens = add_time_and_sequence_targets(
        pan_base
    )

    print(
        "[CKPT4_PREP] building availability-aware breast corpus",
        flush=True,
    )

    breast_base = prepare_breast(
        canonical_path,
        scan_path,
        mbc_splits,
        site_vocab,
    )

    print(
        "[CKPT4_PREP] constructing breast causal sequences",
        flush=True,
    )

    breast_tokens = add_time_and_sequence_targets(
        breast_base
    )

    atomic_parquet(
        prep
        / "pan_cancer_tokens.parquet",
        pan_tokens,
    )

    atomic_parquet(
        prep
        / "breast_tokens.parquet",
        breast_tokens,
    )

    atomic_text(
        prep
        / "site_vocab_32.txt",
        "\n".join(
            site_vocab
        )
        + "\n",
    )

    ###########################################################################
    # Scan index for exact PRE/POST state caching.
    ###########################################################################

    scan_index = (
        breast_base[
            breast_base[
                "type_id"
            ]
            == EVENT_TO_ID[
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
    )

    scan_index = (
        scan_index
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

    if (
        scan_index[
            [
                "patient_id",
                "scan_episode_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "duplicate breast scan episode keys"
        )

    atomic_parquet(
        prep
        / "breast_scan_index.parquet",
        scan_index,
    )

    report = {
        "status":
            "PREPARED",

        "raw_root":
            str(
                raw_root
            ),

        "configuration": {
            "max_sequence_length":
                MAX_SEQ,

            "event_types":
                REAL_EVENT_NAMES,

            "site_vocab_size":
                len(
                    site_vocab
                ),

            "scan_episode_window_days":
                3,

            "breast_scan_source":
                "CKPT1 authoritative W3 episodes",

            "pan_cancer_time_rule":
                (
                    "CHORD relative START_DATE / SEQ_DATE proxies"
                ),

            "breast_time_rule":
                (
                    "CKPT1 canonical availability_day"
                ),
        },

        "pan_cancer":
            token_stats(
                pan_tokens
            ),

        "breast":
            token_stats(
                breast_tokens
            ),

        "breast_scan_index": {
            "rows":
                int(
                    len(
                        scan_index
                    )
                ),

            "patients":
                int(
                    scan_index[
                        "patient_id"
                    ]
                    .nunique()
                ),

            "states": {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in scan_index[
                    "scan_state"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },

        "site_vocab":
            site_vocab,
    }

    atomic_json(
        prep
        / "prepare_report.json",
        report,
    )

    print("")
    print(
        "========== CKPT4 PREP SUMMARY =========="
    )

    print(
        "pan_cancer="
        f"{report['pan_cancer']}"
    )

    print(
        "breast="
        f"{report['breast']}"
    )

    print(
        "breast_scan_index="
        f"{report['breast_scan_index']}"
    )

    print(
        "site_vocab="
        f"{site_vocab}"
    )

    print(
        "========== CKPT4 PREP SUMMARY END =========="
    )


###############################################################################
# Dataset / batching.
###############################################################################


class TokenWindowDataset(
    Dataset
):

    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        train_mode: bool,
    ):

        frame = frame[
            frame[
                "split"
            ]
            == split
        ].copy()

        if frame.empty:

            raise RuntimeError(
                f"no temporal rows for split={split}"
            )

        frame = (
            frame
            .sort_values(
                [
                    "patient_id",
                    "day",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        self.frame = frame

        self.groups = []

        stride = (
            TRAIN_STRIDE
            if train_mode
            else EVAL_STRIDE
        )

        def add_window(
            index_array: np.ndarray,
        ) -> None:

            if len(
                index_array
            ) < 2:
                return

            window = (
                self.frame
                .iloc[
                    index_array
                ]
            )

            # Training/evaluation losses require at least one
            # next-observed-day target.
            if (
                int(
                    window[
                        "target_valid"
                    ].sum()
                )
                <= 0
            ):
                return

            self.groups.append(
                index_array
            )

        for _, indices in frame.groupby(
            "patient_id",
            sort=False,
        ).indices.items():

            indices = np.asarray(
                indices,
                dtype=np.int64,
            )

            count = len(
                indices
            )

            if count <= MAX_SEQ:

                add_window(
                    indices
                )

                continue

            starts = list(
                range(
                    0,
                    max(
                        count
                        - MAX_SEQ
                        + 1,
                        1,
                    ),
                    stride,
                )
            )

            if (
                not starts
                or starts[
                    -1
                ]
                + MAX_SEQ
                < count
            ):

                starts.append(
                    count
                    - MAX_SEQ
                )

            seen = set()

            for start in starts:

                start = max(
                    int(
                        start
                    ),
                    0,
                )

                end = min(
                    start
                    + MAX_SEQ,
                    count,
                )

                key = (
                    start,
                    end,
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                add_window(
                    indices[
                        start:
                        end
                    ]
                )

        if not self.groups:

            raise RuntimeError(
                f"no target-bearing windows for split={split}"
            )

    def __len__(
        self,
    ) -> int:

        return len(
            self.groups
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[
        str,
        np.ndarray,
    ]:

        rows = self.frame.iloc[
            self.groups[
                index
            ]
        ]

        return {
            column:
                rows[
                    column
                ].to_numpy()
            for column
            in rows.columns
        }


def bits_to_vector(
    bits: np.ndarray,
    width: int,
) -> np.ndarray:

    output = np.zeros(
        (
            len(
                bits
            ),
            width,
        ),
        dtype=np.float32,
    )

    for row_index, value in enumerate(
        bits
    ):

        integer = int(
            value
        )

        for bit in range(
            width
        ):

            if (
                integer
                & (
                    1
                    << bit
                )
            ):

                output[
                    row_index,
                    bit,
                ] = 1.0

    return output


def scan_feature_matrix(
    item: dict[
        str,
        np.ndarray,
    ],
) -> np.ndarray:

    count = len(
        item[
            "type_id"
        ]
    )

    output = np.zeros(
        (
            count,
            SCAN_FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    states = item[
        "scan_state"
    ].astype(
        int
    )

    for state in (
        SCAN_NP,
        SCAN_P,
        SCAN_IND,
    ):

        output[
            :,
            state,
        ] = (
            states
            == state
        ).astype(
            np.float32
        )

    output[
        :,
        3,
    ] = (
        states
        >= 0
    ).astype(
        np.float32
    )

    output[
        :,
        4,
    ] = np.log1p(
        item[
            "scan_report_count"
        ].astype(
            float
        )
    )

    output[
        :,
        5,
    ] = np.log1p(
        item[
            "scan_site_count"
        ].astype(
            float
        )
    )

    output[
        :,
        6,
    ] = item[
        "site_known"
    ].astype(
        np.float32
    )

    imaged = item[
        "region_imaged_mask"
    ].astype(
        np.int64
    )

    known = item[
        "region_known_mask"
    ].astype(
        np.int64
    )

    for index in range(
        5
    ):

        output[
            :,
            7
            + index,
        ] = (
            (
                imaged
                & (
                    1
                    << index
                )
            )
            > 0
        ).astype(
            np.float32
        )

        output[
            :,
            12
            + index,
        ] = (
            (
                known
                & (
                    1
                    << index
                )
            )
            > 0
        ).astype(
            np.float32
        )

    return output


def collate_windows(
    items: list[
        dict[
            str,
            np.ndarray,
        ]
    ],
) -> dict[
    str,
    torch.Tensor,
]:

    max_length = max(
        len(
            item[
                "type_id"
            ]
        )
        for item
        in items
    )

    batch_size = len(
        items
    )

    def padded(
        name: str,
        dtype,
        fill=0,
    ):

        output = np.full(
            (
                batch_size,
                max_length,
            ),
            fill,
            dtype=dtype,
        )

        for index, item in enumerate(
            items
        ):

            values = item[
                name
            ]

            output[
                index,
                :len(
                    values
                ),
            ] = values.astype(
                dtype,
                copy=False,
            )

        return output

    type_id = padded(
        "type_id",
        np.int64,
        PAD_ID,
    )

    payload = padded(
        "payload_id",
        np.int64,
        0,
    )

    source = padded(
        "source_id",
        np.int64,
        0,
    )

    line = padded(
        "line",
        np.int64,
        0,
    )

    day = padded(
        "day",
        np.float32,
        0,
    )

    gap = padded(
        "gap_days",
        np.float32,
        0,
    )

    time_on = padded(
        "time_on_treatment",
        np.float32,
        0,
    )

    active = padded(
        "treatment_active",
        np.float32,
        0,
    )

    numeric = padded(
        "numeric_value",
        np.float32,
        0,
    )

    numeric_observed = padded(
        "numeric_observed",
        np.float32,
        0,
    )

    target_valid = padded(
        "target_valid",
        np.int64,
        0,
    )

    next_gap = padded(
        "next_gap_bin",
        np.int64,
        -1,
    )

    next_scan = padded(
        "next_scan_state",
        np.int64,
        -1,
    )

    next_site_known = padded(
        "next_site_known",
        np.int64,
        0,
    )

    padding = (
        type_id
        == PAD_ID
    )

    next_type = np.zeros(
        (
            batch_size,
            max_length,
            N_REAL_EVENTS,
        ),
        dtype=np.float32,
    )

    next_site = np.zeros(
        (
            batch_size,
            max_length,
            SITE_VOCAB_SIZE,
        ),
        dtype=np.float32,
    )

    scan_features = np.zeros(
        (
            batch_size,
            max_length,
            SCAN_FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    for index, item in enumerate(
        items
    ):

        length = len(
            item[
                "type_id"
            ]
        )

        next_type[
            index,
            :length,
        ] = bits_to_vector(
            item[
                "next_type_bits"
            ],
            N_REAL_EVENTS,
        )

        next_site[
            index,
            :length,
        ] = bits_to_vector(
            item[
                "next_site_mask"
            ],
            SITE_VOCAB_SIZE,
        )

        scan_features[
            index,
            :length,
        ] = scan_feature_matrix(
            item
        )

    return {
        "type_id":
            torch.from_numpy(
                type_id
            ),

        "payload_id":
            torch.from_numpy(
                payload
            ),

        "source_id":
            torch.from_numpy(
                source
            ),

        "line":
            torch.from_numpy(
                line
            ),

        "day":
            torch.from_numpy(
                day
            ),

        "gap_days":
            torch.from_numpy(
                gap
            ),

        "time_on_treatment":
            torch.from_numpy(
                time_on
            ),

        "treatment_active":
            torch.from_numpy(
                active
            ),

        "numeric_value":
            torch.from_numpy(
                numeric
            ),

        "numeric_observed":
            torch.from_numpy(
                numeric_observed
            ),

        "scan_features":
            torch.from_numpy(
                scan_features
            ),

        "padding":
            torch.from_numpy(
                padding
            ),

        "target_valid":
            torch.from_numpy(
                target_valid
            ),

        "next_type":
            torch.from_numpy(
                next_type
            ),

        "next_gap_bin":
            torch.from_numpy(
                next_gap
            ),

        "next_scan_state":
            torch.from_numpy(
                next_scan
            ),

        "next_site":
            torch.from_numpy(
                next_site
            ),

        "next_site_known":
            torch.from_numpy(
                next_site_known
            ),
    }


###############################################################################
# Causal event Transformer.
###############################################################################


class TemporalTransformer(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()

        self.type_embedding = nn.Embedding(
            N_TYPES,
            HIDDEN,
            padding_idx=PAD_ID,
        )

        self.payload_embedding = nn.Embedding(
            PAYLOAD_BUCKETS,
            HIDDEN,
            padding_idx=0,
        )

        self.source_embedding = nn.Embedding(
            SOURCE_BUCKETS,
            HIDDEN,
            padding_idx=0,
        )

        self.line_embedding = nn.Embedding(
            LINE_BUCKETS,
            HIDDEN,
        )

        self.position_embedding = nn.Embedding(
            MAX_SEQ,
            HIDDEN,
        )

        self.continuous_projection = nn.Sequential(
            nn.Linear(
                6,
                HIDDEN,
            ),
            nn.GELU(),
            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),
        )

        self.scan_projection = nn.Sequential(
            nn.Linear(
                SCAN_FEATURE_DIM,
                HIDDEN,
            ),
            nn.GELU(),
            nn.Linear(
                HIDDEN,
                HIDDEN,
            ),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=HEADS,
            dim_feedforward=FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=LAYERS,
            norm=nn.LayerNorm(
                HIDDEN
            ),
        )

        self.masked_type_head = nn.Linear(
            HIDDEN,
            N_REAL_EVENTS,
        )

        self.next_type_head = nn.Linear(
            HIDDEN,
            N_REAL_EVENTS,
        )

        self.gap_head = nn.Linear(
            HIDDEN,
            N_GAP_BINS,
        )

        self.scan_state_head = nn.Linear(
            HIDDEN,
            3,
        )

        self.site_head = nn.Linear(
            HIDDEN,
            SITE_VOCAB_SIZE,
        )

    def forward(
        self,
        batch: dict[
            str,
            torch.Tensor,
        ],
        override_type: torch.Tensor | None = None,
    ) -> dict[
        str,
        torch.Tensor,
    ]:

        type_id = (
            batch[
                "type_id"
            ]
            if override_type
            is None
            else override_type
        )

        batch_size, sequence_length = (
            type_id.shape
        )

        position = (
            torch.arange(
                sequence_length,
                device=type_id.device,
            )[
                None,
                :
            ]
            .expand(
                batch_size,
                -1,
            )
        )

        day = batch[
            "day"
        ].float()

        gap = (
            batch[
                "gap_days"
            ]
            .float()
            .clamp_min(
                0
            )
        )

        time_on = (
            batch[
                "time_on_treatment"
            ]
            .float()
            .clamp_min(
                0
            )
        )

        numeric = (
            batch[
                "numeric_value"
            ]
            .float()
            .clamp(
                -12,
                12,
            )
        )

        continuous = torch.stack(
            [
                (
                    torch.sign(
                        day
                    )
                    * torch.log1p(
                        day.abs()
                    )
                    / 10.0
                ),

                (
                    torch.log1p(
                        gap
                    )
                    / 8.0
                ),

                (
                    torch.log1p(
                        time_on
                    )
                    / 8.0
                ),

                batch[
                    "treatment_active"
                ].float(),

                numeric
                / 10.0,

                batch[
                    "numeric_observed"
                ].float(),
            ],
            dim=-1,
        )

        x = (
            self.type_embedding(
                type_id
            )

            + self.payload_embedding(
                batch[
                    "payload_id"
                ]
            )

            + self.source_embedding(
                batch[
                    "source_id"
                ]
            )

            + self.line_embedding(
                batch[
                    "line"
                ].clamp(
                    0,
                    LINE_BUCKETS
                    - 1,
                )
            )

            + self.position_embedding(
                position.clamp_max(
                    MAX_SEQ
                    - 1
                )
            )

            + self.continuous_projection(
                continuous
            )

            + self.scan_projection(
                batch[
                    "scan_features"
                ].float()
            )
        )

        # Strictly causal across the structured token sequence.
        causal_mask = torch.triu(
            torch.ones(
                (
                    sequence_length,
                    sequence_length,
                ),
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )

        hidden = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=batch[
                "padding"
            ],
        )

        return {
            "hidden":
                hidden,

            "masked_type_logits":
                self.masked_type_head(
                    hidden
                ),

            "next_type_logits":
                self.next_type_head(
                    hidden
                ),

            "gap_logits":
                self.gap_head(
                    hidden
                ),

            "scan_state_logits":
                self.scan_state_head(
                    hidden
                ),

            "site_logits":
                self.site_head(
                    hidden
                ),
        }


###############################################################################
# Objectives.
###############################################################################


def move_batch(
    batch: dict[
        str,
        torch.Tensor,
    ],
    device: torch.device,
) -> dict[
    str,
    torch.Tensor,
]:

    return {
        key:
            value.to(
                device,
                non_blocking=True,
            )
        for key, value
        in batch.items()
    }


def masked_type_inputs(
    batch: dict[
        str,
        torch.Tensor,
    ],
    generator: torch.Generator,
    probability: float = 0.15,
):

    type_id = batch[
        "type_id"
    ]

    real = (
        type_id
        >= 4
    )

    random_values = torch.rand(
        type_id.shape,
        generator=generator,
        device=type_id.device,
    )

    mask = (
        real
        & (
            random_values
            < probability
        )
        & (
            ~batch[
                "padding"
            ]
        )
    )

    if (
        mask.sum()
        == 0
        and real.any()
    ):

        first = torch.nonzero(
            real,
            as_tuple=False,
        )[
            0
        ]

        mask[
            first[
                0
            ],
            first[
                1
            ],
        ] = True

    override = type_id.clone()

    override[
        mask
    ] = MASK_ID

    target = (
        type_id
        - 4
    ).clamp_min(
        0
    )

    return (
        override,
        mask,
        target,
    )


def compute_loss(
    model: TemporalTransformer,
    batch: dict[
        str,
        torch.Tensor,
    ],
    generator: torch.Generator,
    next_type_pos_weight: torch.Tensor,
    site_pos_weight: torch.Tensor,
    consistency: bool,
):

    (
        override,
        reconstruction_mask,
        reconstruction_target,
    ) = masked_type_inputs(
        batch,
        generator,
    )

    output = model(
        batch,
        override_type=override,
    )

    reconstruction_loss = (
        F.cross_entropy(
            output[
                "masked_type_logits"
            ][
                reconstruction_mask
            ].float(),
            reconstruction_target[
                reconstruction_mask
            ],
        )
    )

    summary_mask = (
        (
            batch[
                "target_valid"
            ]
            > 0
        )
        & (
            ~batch[
                "padding"
            ]
        )
    )

    if (
        summary_mask.sum()
        == 0
    ):

        raise RuntimeError(
            "target-bearing temporal batch unexpectedly "
            "contains no DAY_END targets"
        )

    next_type_loss = (
        F.binary_cross_entropy_with_logits(
            output[
                "next_type_logits"
            ][
                summary_mask
            ].float(),
            batch[
                "next_type"
            ][
                summary_mask
            ].float(),
            pos_weight=next_type_pos_weight,
        )
    )

    gap_loss = F.cross_entropy(
        output[
            "gap_logits"
        ][
            summary_mask
        ].float(),
        batch[
            "next_gap_bin"
        ][
            summary_mask
        ].long(),
    )

    scan_mask = (
        summary_mask
        & (
            batch[
                "next_scan_state"
            ]
            >= 0
        )
    )

    if scan_mask.any():

        scan_loss = F.cross_entropy(
            output[
                "scan_state_logits"
            ][
                scan_mask
            ].float(),
            batch[
                "next_scan_state"
            ][
                scan_mask
            ].long(),
        )

    else:

        scan_loss = torch.zeros(
            (),
            device=batch[
                "type_id"
            ].device,
        )

    site_mask = (
        summary_mask
        & (
            batch[
                "next_site_known"
            ]
            > 0
        )
    )

    if site_mask.any():

        site_loss = (
            F.binary_cross_entropy_with_logits(
                output[
                    "site_logits"
                ][
                    site_mask
                ].float(),
                batch[
                    "next_site"
                ][
                    site_mask
                ].float(),
                pos_weight=site_pos_weight,
            )
        )

    else:

        site_loss = torch.zeros(
            (),
            device=batch[
                "type_id"
            ].device,
        )

    consistency_loss = torch.zeros(
        (),
        device=batch[
            "type_id"
        ].device,
    )

    ###########################################################################
    # Event-dropout consistency every few optimizer steps.
    ###########################################################################

    if consistency:

        augmented_type = batch[
            "type_id"
        ].clone()

        real = (
            augmented_type
            >= 4
        )

        drop = (
            (
                torch.rand(
                    augmented_type.shape,
                    generator=generator,
                    device=augmented_type.device,
                )
                < 0.10
            )
            & real
            & (
                ~batch[
                    "padding"
                ]
            )
        )

        augmented_type[
            drop
        ] = MASK_ID

        augmented = model(
            batch,
            override_type=augmented_type,
        )

        consistency_loss = (
            1.0
            - F.cosine_similarity(
                output[
                    "hidden"
                ][
                    summary_mask
                ]
                .float()
                .detach(),

                augmented[
                    "hidden"
                ][
                    summary_mask
                ]
                .float(),

                dim=-1,
            )
        ).mean()

    total = (
        reconstruction_loss

        + next_type_loss

        + 0.30
        * gap_loss

        + 0.50
        * scan_loss

        + 0.25
        * site_loss

        + 0.05
        * consistency_loss
    )

    return (
        total,
        {
            "reconstruction":
                reconstruction_loss.detach(),

            "next_type":
                next_type_loss.detach(),

            "gap":
                gap_loss.detach(),

            "scan":
                scan_loss.detach(),

            "site":
                site_loss.detach(),

            "consistency":
                consistency_loss.detach(),
        },
    )


###############################################################################
# Training utilities.
###############################################################################


def target_weights(
    frame: pd.DataFrame,
):

    day_end = frame[
        (
            frame[
                "target_valid"
            ]
            > 0
        )
        & (
            frame[
                "split"
            ]
            == "train"
        )
    ]

    if day_end.empty:

        raise RuntimeError(
            "no training DAY_END targets"
        )

    type_target = bits_to_vector(
        day_end[
            "next_type_bits"
        ].to_numpy(),
        N_REAL_EVENTS,
    )

    positive = type_target.sum(
        axis=0
    )

    negative = (
        len(
            type_target
        )
        - positive
    )

    type_weight = np.clip(
        negative
        / np.maximum(
            positive,
            1,
        ),
        1.0,
        20.0,
    ).astype(
        np.float32
    )

    site_rows = day_end[
        day_end[
            "next_site_known"
        ]
        > 0
    ]

    if len(
        site_rows
    ):

        site_target = bits_to_vector(
            site_rows[
                "next_site_mask"
            ].to_numpy(),
            SITE_VOCAB_SIZE,
        )

        positive = site_target.sum(
            axis=0
        )

        negative = (
            len(
                site_target
            )
            - positive
        )

        site_weight = np.clip(
            negative
            / np.maximum(
                positive,
                1,
            ),
            1.0,
            20.0,
        ).astype(
            np.float32
        )

    else:

        site_weight = np.ones(
            SITE_VOCAB_SIZE,
            dtype=np.float32,
        )

    priors = {
        "next_type_prevalence":
            type_target.mean(
                axis=0
            ).tolist(),

        "gap_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in day_end[
                "next_gap_bin"
            ]
            .value_counts()
            .to_dict()
            .items()
        },

        "scan_state_counts": {
            str(
                key
            ):
                int(
                    value
                )
            for key, value
            in (
                day_end.loc[
                    day_end[
                        "next_scan_state"
                    ]
                    >= 0,
                    "next_scan_state",
                ]
                .value_counts()
                .to_dict()
                .items()
            )
        },

        "site_known_rows":
            int(
                len(
                    site_rows
                )
            ),
    }

    return (
        type_weight,
        site_weight,
        priors,
    )


def evaluate_loss(
    model: TemporalTransformer,
    loader: DataLoader,
    device: torch.device,
    type_weight: torch.Tensor,
    site_weight: torch.Tensor,
    seed: int,
) -> float:

    model.eval()

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )

    values = []

    with torch.no_grad():

        for batch in loader:

            batch = move_batch(
                batch,
                device,
            )

            loss, _ = compute_loss(
                model,
                batch,
                generator,
                type_weight,
                site_weight,
                consistency=False,
            )

            values.append(
                float(
                    loss
                )
            )

    if not values:
        return float(
            "inf"
        )

    return float(
        np.mean(
            values
        )
    )


def train_stage(
    model: TemporalTransformer,
    frame: pd.DataFrame,
    stage_name: str,
    out: Path,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
):

    train_dataset = TokenWindowDataset(
        frame,
        "train",
        train_mode=True,
    )

    val_dataset = TokenWindowDataset(
        frame,
        "val",
        train_mode=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
        pin_memory=True,
    )

    (
        type_weight_numpy,
        site_weight_numpy,
        priors,
    ) = target_weights(
        frame
    )

    type_weight = (
        torch.from_numpy(
            type_weight_numpy
        )
        .to(
            device
        )
    )

    site_weight = (
        torch.from_numpy(
            site_weight_numpy
        )
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.02,
        betas=(
            0.9,
            0.95,
        ),
    )

    total_steps = max(
        epochs
        * len(
            train_loader
        ),
        1,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=(
                learning_rate
                * 0.10
            ),
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            == "cuda"
        ),
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        SEED
        + stable_hash(
            stage_name
        )
        % 100000
    )

    history = []

    best_val = float(
        "inf"
    )

    best_state = None

    for epoch in range(
        1,
        epochs
        + 1,
    ):

        model.train()

        totals = Counter()

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            batch = move_batch(
                batch,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):

                (
                    loss,
                    pieces,
                ) = compute_loss(
                    model,
                    batch,
                    generator,
                    type_weight,
                    site_weight,
                    consistency=(
                        step
                        % 4
                        == 0
                    ),
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()

            totals[
                "steps"
            ] += 1

            totals[
                "loss"
            ] += float(
                loss.detach()
            )

            for key, value in pieces.items():

                totals[
                    key
                ] += float(
                    value
                )

        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            type_weight,
            site_weight,
            SEED
            + epoch
            + stable_hash(
                stage_name
            )
            % 100000,
        )

        denominator = max(
            totals[
                "steps"
            ],
            1,
        )

        record = {
            "stage":
                stage_name,

            "epoch":
                epoch,

            "train_loss":
                totals[
                    "loss"
                ]
                / denominator,

            "val_loss":
                val_loss,

            "lr":
                float(
                    optimizer
                    .param_groups[
                        0
                    ][
                        "lr"
                    ]
                ),

            "train_reconstruction":
                totals[
                    "reconstruction"
                ]
                / denominator,

            "train_next_type":
                totals[
                    "next_type"
                ]
                / denominator,

            "train_gap":
                totals[
                    "gap"
                ]
                / denominator,

            "train_scan":
                totals[
                    "scan"
                ]
                / denominator,

            "train_site":
                totals[
                    "site"
                ]
                / denominator,

            "train_consistency":
                totals[
                    "consistency"
                ]
                / denominator,
        }

        history.append(
            record
        )

        print(
            "[CKPT4_TRAIN]",
            record,
            flush=True,
        )

        if (
            val_loss
            < best_val
        ):

            best_val = (
                val_loss
            )

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()
                for key, value
                in model.state_dict()
                .items()
            }

    if best_state is None:

        raise RuntimeError(
            f"no best model selected for {stage_name}"
        )

    model.load_state_dict(
        best_state
    )

    atomic_json(
        out
        / (
            f"{stage_name}_"
            "training_history.json"
        ),
        history,
    )

    atomic_json(
        out
        / (
            f"{stage_name}_"
            "target_priors.json"
        ),
        priors,
    )

    return (
        history,
        priors,
    )


###############################################################################
# Evaluation metrics.
###############################################################################


def macro_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[
    float,
    int,
]:

    values = []

    for column in range(
        y_true.shape[
            1
        ]
    ):

        target = y_true[
            :,
            column
        ]

        if (
            target.sum()
            < 2
            or (
                1
                - target
            ).sum()
            < 2
        ):

            continue

        values.append(
            float(
                average_precision_score(
                    target,
                    y_score[
                        :,
                        column
                    ],
                )
            )
        )

    return (
        (
            float(
                np.mean(
                    values
                )
            )
            if values
            else float(
                "nan"
            )
        ),
        len(
            values
        ),
    )


def evaluate_metrics(
    model: TemporalTransformer,
    frame: pd.DataFrame,
    split: str,
    device: torch.device,
    priors: dict[
        str,
        Any,
    ],
    batch_size: int,
) -> dict[
    str,
    Any,
]:

    dataset = TokenWindowDataset(
        frame,
        split,
        train_mode=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )

    next_true = []
    next_score = []

    gap_true = []
    gap_score = []

    scan_true = []
    scan_score = []

    site_true = []
    site_score = []

    reconstruction_true = []
    reconstruction_pred = []

    consistency_values = []

    model.eval()

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        SEED
        + stable_hash(
            split
        )
        % 100000
        + 777
    )

    with torch.no_grad():

        for batch in loader:

            batch = move_batch(
                batch,
                device,
            )

            clean = model(
                batch
            )

            summary_mask = (
                (
                    batch[
                        "target_valid"
                    ]
                    > 0
                )
                & (
                    ~batch[
                        "padding"
                    ]
                )
            )

            if summary_mask.any():

                next_true.append(
                    batch[
                        "next_type"
                    ][
                        summary_mask
                    ]
                    .cpu()
                    .numpy()
                )

                next_score.append(
                    torch.sigmoid(
                        clean[
                            "next_type_logits"
                        ][
                            summary_mask
                        ]
                        .float()
                    )
                    .cpu()
                    .numpy()
                )

                gap_true.append(
                    batch[
                        "next_gap_bin"
                    ][
                        summary_mask
                    ]
                    .cpu()
                    .numpy()
                )

                gap_score.append(
                    torch.softmax(
                        clean[
                            "gap_logits"
                        ][
                            summary_mask
                        ]
                        .float(),
                        dim=-1,
                    )
                    .cpu()
                    .numpy()
                )

            scan_mask = (
                summary_mask
                & (
                    batch[
                        "next_scan_state"
                    ]
                    >= 0
                )
            )

            if scan_mask.any():

                scan_true.append(
                    batch[
                        "next_scan_state"
                    ][
                        scan_mask
                    ]
                    .cpu()
                    .numpy()
                )

                scan_score.append(
                    torch.softmax(
                        clean[
                            "scan_state_logits"
                        ][
                            scan_mask
                        ]
                        .float(),
                        dim=-1,
                    )
                    .cpu()
                    .numpy()
                )

            site_mask = (
                summary_mask
                & (
                    batch[
                        "next_site_known"
                    ]
                    > 0
                )
            )

            if site_mask.any():

                site_true.append(
                    batch[
                        "next_site"
                    ][
                        site_mask
                    ]
                    .cpu()
                    .numpy()
                )

                site_score.append(
                    torch.sigmoid(
                        clean[
                            "site_logits"
                        ][
                            site_mask
                        ]
                        .float()
                    )
                    .cpu()
                    .numpy()
                )

            ###################################################################
            # Deterministic masked-event evaluation.
            ###################################################################

            (
                override,
                reconstruction_mask,
                target,
            ) = masked_type_inputs(
                batch,
                generator,
                probability=0.15,
            )

            masked = model(
                batch,
                override_type=override,
            )

            if reconstruction_mask.any():

                reconstruction_true.append(
                    target[
                        reconstruction_mask
                    ]
                    .cpu()
                    .numpy()
                )

                reconstruction_pred.append(
                    masked[
                        "masked_type_logits"
                    ][
                        reconstruction_mask
                    ]
                    .argmax(
                        dim=-1
                    )
                    .cpu()
                    .numpy()
                )

            ###################################################################
            # Event-dropout consistency.
            ###################################################################

            augmented_type = batch[
                "type_id"
            ].clone()

            real = (
                augmented_type
                >= 4
            )

            drop = (
                (
                    torch.rand(
                        augmented_type.shape,
                        generator=generator,
                        device=device,
                    )
                    < 0.10
                )
                & real
                & (
                    ~batch[
                        "padding"
                    ]
                )
            )

            augmented_type[
                drop
            ] = MASK_ID

            augmented = model(
                batch,
                override_type=augmented_type,
            )

            day_end_mask = (
                (
                    batch[
                        "type_id"
                    ]
                    == DAY_END_ID
                )
                & (
                    ~batch[
                        "padding"
                    ]
                )
            )

            if day_end_mask.any():

                cosine = (
                    F.cosine_similarity(
                        clean[
                            "hidden"
                        ][
                            day_end_mask
                        ]
                        .float(),

                        augmented[
                            "hidden"
                        ][
                            day_end_mask
                        ]
                        .float(),

                        dim=-1,
                    )
                )

                consistency_values.append(
                    cosine
                    .cpu()
                    .numpy()
                )

    result = {
        "split":
            split,

        "windows":
            int(
                len(
                    dataset
                )
            ),
    }

    ###########################################################################
    # Next event.
    ###########################################################################

    if next_true:

        target = np.concatenate(
            next_true
        )

        score = np.concatenate(
            next_score
        )

        result[
            "next_type_micro_auprc"
        ] = float(
            average_precision_score(
                target.ravel(),
                score.ravel(),
            )
        )

        (
            result[
                "next_type_macro_auprc"
            ],
            result[
                "next_type_macro_classes"
            ],
        ) = macro_auprc(
            target,
            score,
        )

        prior = np.asarray(
            priors[
                "next_type_prevalence"
            ],
            dtype=float,
        )

        baseline = np.broadcast_to(
            prior,
            target.shape,
        )

        result[
            "next_type_baseline_micro_auprc"
        ] = float(
            average_precision_score(
                target.ravel(),
                baseline.ravel(),
            )
        )

        result[
            "next_type_rows"
        ] = int(
            len(
                target
            )
        )

    ###########################################################################
    # Gap.
    ###########################################################################

    if gap_true:

        target = np.concatenate(
            gap_true
        ).astype(
            int
        )

        score = np.concatenate(
            gap_score
        )

        result[
            "gap_accuracy"
        ] = float(
            (
                score.argmax(
                    axis=1
                )
                == target
            )
            .mean()
        )

        counts = np.zeros(
            N_GAP_BINS,
            dtype=float,
        )

        for key, value in (
            priors.get(
                "gap_counts",
                {}
            )
            .items()
        ):

            counts[
                int(
                    key
                )
            ] = float(
                value
            )

        mode = int(
            counts.argmax()
        )

        result[
            "gap_baseline_accuracy"
        ] = float(
            (
                target
                == mode
            )
            .mean()
        )

        result[
            "gap_rows"
        ] = int(
            len(
                target
            )
        )

    ###########################################################################
    # Next scan state.
    ###########################################################################

    if scan_true:

        target = np.concatenate(
            scan_true
        ).astype(
            int
        )

        score = np.concatenate(
            scan_score
        )

        one_hot = np.eye(
            3,
            dtype=float,
        )[
            target
        ]

        result[
            "scan_state_accuracy"
        ] = float(
            (
                score.argmax(
                    axis=1
                )
                == target
            )
            .mean()
        )

        (
            result[
                "scan_state_macro_auprc"
            ],
            result[
                "scan_state_macro_classes"
            ],
        ) = macro_auprc(
            one_hot,
            score,
        )

        counts = np.zeros(
            3,
            dtype=float,
        )

        for key, value in (
            priors.get(
                "scan_state_counts",
                {}
            )
            .items()
        ):

            key = int(
                key
            )

            if (
                0
                <= key
                < 3
            ):

                counts[
                    key
                ] = float(
                    value
                )

        prior = (
            counts
            / max(
                counts.sum(),
                1.0,
            )
        )

        prior_score = np.broadcast_to(
            prior,
            score.shape,
        )

        result[
            "scan_state_baseline_accuracy"
        ] = float(
            (
                target
                == int(
                    counts.argmax()
                )
            )
            .mean()
        )

        (
            result[
                "scan_state_baseline_macro_auprc"
            ],
            _,
        ) = macro_auprc(
            one_hot,
            prior_score,
        )

        result[
            "scan_state_rows"
        ] = int(
            len(
                target
            )
        )

    ###########################################################################
    # Next metastatic sites.
    ###########################################################################

    if site_true:

        target = np.concatenate(
            site_true
        )

        score = np.concatenate(
            site_score
        )

        (
            result[
                "site_macro_auprc"
            ],
            result[
                "site_macro_classes"
            ],
        ) = macro_auprc(
            target,
            score,
        )

        result[
            "site_rows"
        ] = int(
            len(
                target
            )
        )

    ###########################################################################
    # Masked reconstruction + dropout consistency.
    ###########################################################################

    if reconstruction_true:

        target = np.concatenate(
            reconstruction_true
        )

        prediction = np.concatenate(
            reconstruction_pred
        )

        result[
            "masked_event_type_accuracy"
        ] = float(
            (
                target
                == prediction
            )
            .mean()
        )

        result[
            "masked_event_rows"
        ] = int(
            len(
                target
            )
        )

    if consistency_values:

        result[
            "event_dropout_embedding_cosine"
        ] = float(
            np.concatenate(
                consistency_values
            )
            .mean()
        )

    return result


###############################################################################
# Training command.
###############################################################################


def train_command(
    repo: Path,
    out: Path,
    pan_epochs: int,
    breast_epochs: int,
    batch_size: int,
) -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required for CKPT4 training"
        )

    device = torch.device(
        "cuda",
        0,
    )

    torch.cuda.set_device(
        0
    )

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    prep = (
        out
        / "prepared"
    )

    pan = pd.read_parquet(
        prep
        / "pan_cancer_tokens.parquet"
    )

    breast = pd.read_parquet(
        prep
        / "breast_tokens.parquet"
    )

    model = TemporalTransformer().to(
        device
    )

    ###########################################################################
    # Stage A: pan-cancer.
    ###########################################################################

    print(
        "[CKPT4] Stage A: pan-cancer CHORD temporal pretraining",
        flush=True,
    )

    (
        _,
        pan_priors,
    ) = train_stage(
        model,
        pan,
        "pan_cancer",
        out,
        pan_epochs,
        2e-4,
        batch_size,
        device,
    )

    pan_test = evaluate_metrics(
        model,
        pan,
        "test",
        device,
        pan_priors,
        batch_size,
    )

    atomic_json(
        out
        / "pan_cancer_test_metrics.json",
        pan_test,
    )

    print(
        "[CKPT4_PAN_TEST]",
        pan_test,
        flush=True,
    )

    ###########################################################################
    # Stage B: breast-specific continued pretraining.
    ###########################################################################

    print(
        "[CKPT4] Stage B: breast-specific continued pretraining",
        flush=True,
    )

    (
        _,
        breast_priors,
    ) = train_stage(
        model,
        breast,
        "breast_continue",
        out,
        breast_epochs,
        1e-4,
        batch_size,
        device,
    )

    breast_test = evaluate_metrics(
        model,
        breast,
        "test",
        device,
        breast_priors,
        batch_size,
    )

    atomic_json(
        out
        / "breast_test_metrics.json",
        breast_test,
    )

    print(
        "[CKPT4_BREAST_TEST]",
        breast_test,
        flush=True,
    )

    ###########################################################################
    # Save frozen encoder.
    ###########################################################################

    checkpoint = {
        "model_state": {
            key:
                value.detach()
                .cpu()
            for key, value
            in model.state_dict()
            .items()
        },

        "config": {
            "max_seq":
                MAX_SEQ,

            "hidden":
                HIDDEN,

            "layers":
                LAYERS,

            "heads":
                HEADS,

            "ff":
                FF,

            "dropout":
                DROPOUT,

            "payload_buckets":
                PAYLOAD_BUCKETS,

            "source_buckets":
                SOURCE_BUCKETS,

            "line_buckets":
                LINE_BUCKETS,

            "scan_feature_dim":
                SCAN_FEATURE_DIM,

            "site_vocab_size":
                SITE_VOCAB_SIZE,

            "event_names":
                REAL_EVENT_NAMES,
        },

        "seed":
            SEED,

        "pan_epochs":
            pan_epochs,

        "breast_epochs":
            breast_epochs,
    }

    temporary = (
        out
        / "temporal_encoder.pt.tmp"
    )

    torch.save(
        checkpoint,
        temporary,
    )

    check = torch.load(
        temporary,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "model_state"
        not in check
    ):

        raise RuntimeError(
            "temporal checkpoint verification failed"
        )

    temporary.replace(
        out
        / "temporal_encoder.pt"
    )


###############################################################################
# Scan PRE/POST state cache.
###############################################################################


def load_temporal_model(
    out: Path,
    device: torch.device,
) -> TemporalTransformer:

    checkpoint = torch.load(
        out
        / "temporal_encoder.pt",
        map_location=device,
        weights_only=False,
    )

    model = TemporalTransformer().to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    return model


def rows_to_item(
    rows: pd.DataFrame,
) -> dict[
    str,
    np.ndarray,
]:

    return {
        column:
            rows[
                column
            ].to_numpy()
        for column
        in rows.columns
    }


@torch.no_grad()
def cache_scan_states(
    out: Path,
    batch_size: int,
) -> dict[
    str,
    Any,
]:

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA required for scan-state cache"
        )

    device = torch.device(
        "cuda",
        0,
    )

    model = load_temporal_model(
        out,
        device,
    )

    prep = (
        out
        / "prepared"
    )

    tokens = pd.read_parquet(
        prep
        / "breast_tokens.parquet"
    )

    scans = pd.read_parquet(
        prep
        / "breast_scan_index.parquet"
    )

    patient_tokens = {
        patient:
            group
            .sort_values(
                [
                    "day",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        for patient, group
        in tokens.groupby(
            "patient_id",
            sort=False,
        )
    }

    scan_token_rows = tokens[
        (
            tokens[
                "type_id"
            ]
            == EVENT_TO_ID[
                "SCAN_EPISODE"
            ]
        )
        & (
            tokens[
                "scan_episode_id"
            ]
            .astype(str)
            != ""
        )
    ].copy()

    scan_lookup = {
        (
            str(
                row[
                    "patient_id"
                ]
            ),
            str(
                row[
                    "scan_episode_id"
                ]
            ),
        ):
            row
        for _, row
        in scan_token_rows.iterrows()
    }

    items = []

    metadata = []

    truncated = 0

    for _, scan in scans.iterrows():

        patient = str(
            scan[
                "patient_id"
            ]
        )

        episode = str(
            scan[
                "scan_episode_id"
            ]
        )

        current = scan_lookup.get(
            (
                patient,
                episode,
            )
        )

        if current is None:

            continue

        history = patient_tokens[
            patient
        ]

        # PRE state:
        # strictly earlier availability days only.
        prior = history[
            history[
                "day"
            ]
            .astype(int)
            < int(
                scan[
                    "day"
                ]
            )
        ].copy()

        if prior.empty:

            prior = (
                history[
                    history[
                        "type_id"
                    ]
                    == START_ID
                ]
                .head(
                    1
                )
                .copy()
            )

        # POST state:
        # prior-day history + ONLY the current scan token.
        #
        # This deliberately excludes all other same-day events, whose
        # within-day ordering is not identifiable from day-level timestamps.
        current_frame = pd.DataFrame(
            [
                current.to_dict()
            ]
        )

        sequence = pd.concat(
            [
                prior,
                current_frame,
            ],
            ignore_index=True,
        )

        if len(
            sequence
        ) > MAX_SEQ:

            truncated += 1

            sequence = (
                sequence
                .iloc[
                    -MAX_SEQ:
                ]
                .copy()
            )

        items.append(
            rows_to_item(
                sequence
            )
        )

        metadata.append(
            {
                "patient_id":
                    patient,

                "line":
                    int(
                        scan[
                            "line"
                        ]
                    ),

                "landmark_day":
                    int(
                        scan[
                            "day"
                        ]
                    ),

                "scan_episode_id":
                    episode,

                "scan_state":
                    int(
                        scan[
                            "scan_state"
                        ]
                    ),

                "split":
                    str(
                        scan[
                            "split"
                        ]
                    ),

                "eligible_residual_pfs":
                    int(
                        int(
                            scan[
                                "scan_state"
                            ]
                        )
                        != SCAN_P
                    ),
            }
        )

    if not items:

        raise RuntimeError(
            "no scan prefixes were created"
        )

    embeddings = np.zeros(
        (
            len(
                items
            ),
            2,
            HIDDEN,
        ),
        dtype=np.float16,
    )

    for start in range(
        0,
        len(
            items
        ),
        batch_size,
    ):

        subset = items[
            start:
            start
            + batch_size
        ]

        batch = move_batch(
            collate_windows(
                subset
            ),
            device,
        )

        output = model(
            batch
        )

        hidden = output[
            "hidden"
        ]

        lengths = (
            (
                ~batch[
                    "padding"
                ]
            )
            .sum(
                dim=1
            )
            .long()
        )

        for local_index in range(
            len(
                subset
            )
        ):

            post_index = (
                int(
                    lengths[
                        local_index
                    ].item()
                )
                - 1
            )

            pre_index = max(
                post_index
                - 1,
                0,
            )

            embeddings[
                start
                + local_index,
                0,
            ] = (
                hidden[
                    local_index,
                    pre_index,
                ]
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float16
                )
            )

            embeddings[
                start
                + local_index,
                1,
            ] = (
                hidden[
                    local_index,
                    post_index,
                ]
                .float()
                .cpu()
                .numpy()
                .astype(
                    np.float16
                )
            )

        if (
            start
            == 0
            or start
            % (
                batch_size
                * 100
            )
            == 0
        ):

            print(
                "[CKPT4_CACHE] "
                f"{start}/{len(items)}",
                flush=True,
            )

    target = (
        out
        / "breast_scan_prepost_embeddings_f16.npy"
    )

    temporary = (
        out
        / "breast_scan_prepost_embeddings_f16.npy.tmp"
    )

    with open(
        temporary,
        "wb",
    ) as handle:

        np.save(
            handle,
            embeddings,
        )

    check = np.load(
        temporary,
        mmap_mode="r",
    )

    if (
        check.shape
        != embeddings.shape
    ):

        raise RuntimeError(
            "scan embedding cache validation failed"
        )

    temporary.replace(
        target
    )

    index = pd.DataFrame(
        metadata
    )

    index[
        "embedding_row"
    ] = np.arange(
        len(
            index
        ),
        dtype=np.int64,
    )

    atomic_parquet(
        out
        / "breast_scan_prepost_index.parquet",
        index,
    )

    pre = embeddings[
        :,
        0,
    ].astype(
        np.float32
    )

    post = embeddings[
        :,
        1,
    ].astype(
        np.float32
    )

    cosine = (
        (
            pre
            * post
        )
        .sum(
            axis=1
        )
        / (
            np.linalg.norm(
                pre,
                axis=1,
            )
            * np.linalg.norm(
                post,
                axis=1,
            )
            + 1e-8
        )
    )

    sample = post[
        :min(
            10000,
            len(
                post
            ),
        )
    ]

    centered = (
        sample
        - sample.mean(
            axis=0,
            keepdims=True,
        )
    )

    singular = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )

    variance = (
        singular
        ** 2
    )

    probability = (
        variance
        / max(
            variance.sum(),
            1e-12,
        )
    )

    effective_rank = float(
        np.exp(
            -np.sum(
                probability
                * np.log(
                    probability
                    + 1e-12
                )
            )
        )
    )

    noncollapsed = int(
        (
            post.std(
                axis=0
            )
            > 1e-4
        )
        .sum()
    )

    result = {
        "rows":
            int(
                len(
                    index
                )
            ),

        "patients":
            int(
                index[
                    "patient_id"
                ]
                .nunique()
            ),

        "eligible_residual_pfs_rows":
            int(
                index[
                    "eligible_residual_pfs"
                ]
                .sum()
            ),

        "truncated_prefixes":
            int(
                truncated
            ),

        "truncated_fraction":
            float(
                truncated
                / len(
                    index
                )
            ),

        "pre_post_cosine_mean":
            float(
                np.mean(
                    cosine
                )
            ),

        "pre_post_cosine_p05":
            float(
                np.quantile(
                    cosine,
                    0.05,
                )
            ),

        "pre_post_cosine_p95":
            float(
                np.quantile(
                    cosine,
                    0.95,
                )
            ),

        "noncollapsed_dimensions":
            noncollapsed,

        "effective_rank":
            effective_rank,

        "embedding_shape":
            list(
                embeddings.shape
            ),
    }

    atomic_json(
        out
        / "scan_embedding_qc.json",
        result,
    )

    print(
        "[CKPT4_SCAN_CACHE]",
        result,
        flush=True,
    )

    return result


###############################################################################
# Finalization.
###############################################################################


def finalize_command(
    repo: Path,
    out: Path,
) -> None:

    prepare = json.loads(
        (
            out
            / "prepared"
            / "prepare_report.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    pan_test = json.loads(
        (
            out
            / "pan_cancer_test_metrics.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    breast_test = json.loads(
        (
            out
            / "breast_test_metrics.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    scan_qc = json.loads(
        (
            out
            / "scan_embedding_qc.json"
        )
        .read_text(
            encoding="utf-8"
        )
    )

    next_event_gain = (
        breast_test.get(
            "next_type_micro_auprc",
            float(
                "nan"
            ),
        )
        - breast_test.get(
            "next_type_baseline_micro_auprc",
            float(
                "nan"
            ),
        )
    )

    next_scan_gain = (
        breast_test.get(
            "scan_state_macro_auprc",
            float(
                "nan"
            ),
        )
        - breast_test.get(
            "scan_state_baseline_macro_auprc",
            float(
                "nan"
            ),
        )
    )

    if (
        np.isfinite(
            next_event_gain
        )
        and np.isfinite(
            next_scan_gain
        )
        and next_event_gain
        > 0
        and next_scan_gain
        > 0
        and scan_qc[
            "noncollapsed_dimensions"
        ]
        >= 96
        and scan_qc[
            "effective_rank"
        ]
        > 2.0
    ):

        status = (
            "PASS_TEMPORAL_PRETRAINING"
        )

    else:

        status = (
            "TEMPORAL_PRETRAINING_NEEDS_TUNING"
        )

    qc = {
        "status":
            status,

        "next_type_auprc_gain_over_prior":
            float(
                next_event_gain
            ),

        "next_scan_macro_auprc_gain_over_prior":
            float(
                next_scan_gain
            ),

        "pan_cancer_test":
            pan_test,

        "breast_test":
            breast_test,

        "scan_embedding_qc":
            scan_qc,

        "errors":
            [],
    }

    atomic_json(
        out
        / "qc.json",
        qc,
    )

    audit = f"""# Checkpoint 4 - CHORD temporal pretraining

Status: **{status}**

## Architecture

- causal event Transformer
- layers: 4
- hidden width: 192
- heads: 6
- feed-forward width: 768
- dropout: 0.15
- maximum structured-event context: 512 tokens

## Stage A

Pan-cancer MSK-CHORD temporal pretraining.

Objectives:
- masked event reconstruction
- next observed-day event type
- time-to-next-event bucket
- next-scan progression state
- next metastatic-site pattern when explicitly observed
- event-dropout consistency

## Stage B

Breast-specific continued pretraining uses CKPT1's availability-aware canonical
timeline and authoritative three-day scan episodes. CKPT1 train/val/test patient
splits are preserved.

## Pan-cancer test

`{pan_test}`

## Breast held-out test

`{breast_test}`

## Scan state cache

`{scan_qc}`

Artifacts:
- `artifacts/checkpoint4/temporal_encoder.pt`
- `artifacts/checkpoint4/pan_cancer_test_metrics.json`
- `artifacts/checkpoint4/breast_test_metrics.json`
- `artifacts/checkpoint4/breast_scan_prepost_embeddings_f16.npy`
- `artifacts/checkpoint4/breast_scan_prepost_index.parquet`
- `artifacts/checkpoint4/qc.json`
- `artifacts/checkpoint4/audit.md`
- `artifacts/handoff/checkpoint_04.json`
"""

    atomic_text(
        out
        / "audit.md",
        audit,
    )

    ###########################################################################
    # PROJECT_STATE.
    ###########################################################################

    state_path = (
        repo
        / "PROJECT_STATE.md"
    )

    existing = (
        state_path.read_text(
            encoding="utf-8"
        )
        if state_path.exists()
        else "# PROJECT STATE\n"
    )

    marker = (
        "<!-- CKPT4_TEMPORAL_PRETRAINING -->"
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

    section = f"""
{marker}
## Checkpoint 4 - CHORD temporal pretraining

Status: **{status}**

Frozen temporal architecture:
- causal event Transformer
- 4 layers / hidden 192 / 6 heads / FF 768 / dropout 0.15
- maximum causal context 512 structured tokens
- pan-cancer CHORD pretraining followed by breast-specific continued pretraining
- objectives: masked event reconstruction, next-event type, next-time bucket,
  next-scan state, next-site pattern, event-dropout consistency
- CKPT1 breast patient splits remain patient-disjoint
- progressive scans remain endpoint/state-update tokens and are not residual-PFS landmarks

Held-out breast temporal evaluation:
- next-event micro-AUPRC: {breast_test.get('next_type_micro_auprc')}
- next-event prior baseline: {breast_test.get('next_type_baseline_micro_auprc')}
- next-scan macro-AUPRC: {breast_test.get('scan_state_macro_auprc')}
- next-scan prior baseline: {breast_test.get('scan_state_baseline_macro_auprc')}

Cached scan states:
- rows: {scan_qc['rows']}
- representation: PRE-scan + POST-scan, each 192-D
- effective rank: {scan_qc['effective_rank']}

Artifacts:
- artifacts/checkpoint4/temporal_encoder.pt
- artifacts/checkpoint4/breast_scan_prepost_embeddings_f16.npy
- artifacts/checkpoint4/breast_scan_prepost_index.parquet
- artifacts/handoff/checkpoint_04.json
"""

    atomic_text(
        state_path,
        existing.rstrip()
        + "\n\n"
        + section.strip()
        + "\n",
    )

    ###########################################################################
    # Handoff.
    ###########################################################################

    handoff = {
        "checkpoint":
            "4",

        "name":
            "chord_temporal_pretraining",

        "status":
            status,

        "architecture": {
            "layers":
                LAYERS,

            "hidden":
                HIDDEN,

            "heads":
                HEADS,

            "ff":
                FF,

            "dropout":
                DROPOUT,

            "max_seq":
                MAX_SEQ,
        },

        "pan_cancer_test":
            pan_test,

        "breast_test":
            breast_test,

        "scan_embedding_qc":
            scan_qc,

        "next_action":
            (
                "Proceed to supervised dynamic breast training: "
                "fuse the temporal state, current scan evidence, "
                "frozen GENIE tumor representation, and structured "
                "landmark features into residual-PFS, next-scan, "
                "and post-progression heads."
                if status
                == "PASS_TEMPORAL_PRETRAINING"
                else (
                    "Inspect temporal objective results and tune "
                    "Checkpoint 4 before supervised dynamic training. "
                    "Do not modify CKPT1 endpoint definitions."
                )
            ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_04.json",
        handoff,
    )

    print("")
    print(
        "========== CKPT4 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "pan_cancer_patients="
        f"{prepare['pan_cancer']['patients']}"
    )

    print(
        "breast_patients="
        f"{prepare['breast']['patients']}"
    )

    print(
        "breast_next_type_micro_auprc="
        f"{breast_test.get('next_type_micro_auprc')}"
    )

    print(
        "breast_next_type_baseline_micro_auprc="
        f"{breast_test.get('next_type_baseline_micro_auprc')}"
    )

    print(
        "breast_scan_state_macro_auprc="
        f"{breast_test.get('scan_state_macro_auprc')}"
    )

    print(
        "breast_scan_state_baseline_macro_auprc="
        f"{breast_test.get('scan_state_baseline_macro_auprc')}"
    )

    print(
        "scan_embedding_rows="
        f"{scan_qc['rows']}"
    )

    print(
        "scan_embedding_effective_rank="
        f"{scan_qc['effective_rank']}"
    )

    print(
        "checkpoint="
        "artifacts/checkpoint4/temporal_encoder.pt"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_04.json"
    )

    print(
        "========== CKPT4 SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT4 DECISION PACKET =========="
    )

    print(
        "pan_cancer_test="
        f"{pan_test}"
    )

    print(
        "breast_test="
        f"{breast_test}"
    )

    print(
        "scan_embedding_qc="
        f"{scan_qc}"
    )

    print(
        "next_type_gain="
        f"{next_event_gain}"
    )

    print(
        "next_scan_gain="
        f"{next_scan_gain}"
    )

    print(
        "========== CKPT4 DECISION PACKET END =========="
    )


###############################################################################
# Synthetic self-test.
###############################################################################


def self_test() -> None:

    rows = [
        base_event_row(
            "p1",
            0,
            "DIAGNOSIS",
            payload="breast",
            source="synthetic",
        ),

        base_event_row(
            "p1",
            5,
            "TREATMENT_START",
            payload="drug=a",
            source="synthetic",
        ),

        base_event_row(
            "p1",
            20,
            "SCAN_EPISODE",
            source="synthetic",
            scan_state=SCAN_NP,
            site_known=1,
            site_mask=1,
        ),

        base_event_row(
            "p1",
            40,
            "SCAN_EPISODE",
            source="synthetic",
            scan_state=SCAN_P,
            site_known=1,
            site_mask=3,
        ),
    ]

    base = pd.DataFrame(
        rows
    )

    base[
        "split"
    ] = "train"

    base[
        "cohort"
    ] = "synthetic"

    tokens = add_time_and_sequence_targets(
        base
    )

    assert (
        tokens[
            "type_id"
        ]
        .eq(
            START_ID
        )
        .sum()
        == 1
    )

    assert (
        tokens[
            "type_id"
        ]
        .eq(
            DAY_END_ID
        )
        .sum()
        == 4
    )

    dataset = TokenWindowDataset(
        tokens,
        "train",
        train_mode=False,
    )

    batch = collate_windows(
        [
            dataset[
                0
            ]
        ]
    )

    model = TemporalTransformer()

    generator = torch.Generator()

    generator.manual_seed(
        1
    )

    loss, _ = compute_loss(
        model,
        batch,
        generator,
        torch.ones(
            N_REAL_EVENTS
        ),
        torch.ones(
            SITE_VOCAB_SIZE
        ),
        consistency=True,
    )

    assert torch.isfinite(
        loss
    )

    loss.backward()

    print(
        "[CKPT4_SELF_TEST_PASS]",
        {
            "loss":
                float(
                    loss
                ),

            "rows":
                len(
                    tokens
                ),
        },
    )


###############################################################################
# CLI.
###############################################################################


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "self-test",
            "prepare",
            "train",
            "cache",
            "finalize",
        ],
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint4",
    )

    parser.add_argument(
        "--pan-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--breast-epochs",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    out_arg = Path(
        args.output_dir
    )

    out = (
        out_arg.resolve()
        if out_arg.is_absolute()
        else (
            repo
            / out_arg
        ).resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.command == "self-test":

        self_test()

    elif args.command == "prepare":

        prepare_command(
            repo,
            out,
        )

    elif args.command == "train":

        train_command(
            repo,
            out,
            args.pan_epochs,
            args.breast_epochs,
            args.batch_size,
        )

    elif args.command == "cache":

        result = cache_scan_states(
            out,
            max(
                args.batch_size,
                32,
            ),
        )

        print(
            "[CKPT4_CACHE_COMPLETE]",
            result,
        )

    elif args.command == "finalize":

        finalize_command(
            repo,
            out,
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
