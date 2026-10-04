#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

CANON = (
    ROOT
    / "artifacts"
    / "checkpoint1"
    / "canonical"
)

OUT = (
    ROOT
    / "artifacts"
    / "checkpoint6d"
)


def norm(
    value,
):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_column(
    columns,
    candidates,
):

    lookup = {
        norm(column):
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

    return None


def atomic_json(
    path,
    payload,
):

    path = Path(path)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
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
    path,
    frame,
):

    path = Path(path)

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
            f"row-count verification failed: {path}"
        )

    tmp.replace(
        path
    )


splits = pd.read_parquet(
    CANON
    / "chord_patient_splits.parquet"
)

lines = pd.read_parquet(
    CANON
    / "chord_mbc_lines.parquet"
)

w3 = pd.read_parquet(
    CANON
    / "chord_mbc_scan_episodes_w3.parquet"
)


patient_col = find_column(
    lines.columns,
    [
        "patient_id",
    ],
)

line_col = find_column(
    lines.columns,
    [
        "line",
        "treatment_line",
        "line_number",
    ],
)

start_col = find_column(
    lines.columns,
    [
        "line_start_day",
        "start_day",
        "treatment_start_day",
        "line_start",
    ],
)

if (
    patient_col is None
    or line_col is None
    or start_col is None
):

    raise RuntimeError(
        "Unable to resolve mBC line columns.\n"
        f"columns={list(lines.columns)}"
    )


lines = lines.copy()

lines[
    patient_col
] = (
    lines[
        patient_col
    ]
    .astype(str)
    .str.strip()
)

lines[
    line_col
] = pd.to_numeric(
    lines[
        line_col
    ],
    errors="raise",
).astype(int)

lines[
    start_col
] = pd.to_numeric(
    lines[
        start_col
    ],
    errors="raise",
).astype(float)


line_groups = {
    patient:
        group
        .sort_values(
            start_col,
            kind="mergesort",
        )[
            [
                start_col,
                line_col,
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )

    for patient, group
    in lines.groupby(
        patient_col,
        sort=False,
    )
}


def assign_line(
    frame,
):

    result = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    patient_values = (
        frame[
            "patient_id"
        ]
        .astype(str)
        .to_numpy()
    )

    day_values = (
        pd.to_numeric(
            frame[
                "landmark_day"
            ],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    by_patient = {}

    for index, patient in enumerate(
        patient_values
    ):

        by_patient.setdefault(
            patient,
            [],
        ).append(
            index
        )

    for patient, rows in by_patient.items():

        line_frame = line_groups.get(
            patient
        )

        if (
            line_frame is None
            or len(line_frame) == 0
        ):
            continue

        starts = (
            line_frame[
                start_col
            ]
            .to_numpy(
                dtype=float
            )
        )

        labels = (
            line_frame[
                line_col
            ]
            .to_numpy(
                dtype=int
            )
        )

        rows_array = np.asarray(
            rows,
            dtype=int,
        )

        days = day_values[
            rows_array
        ]

        position = (
            np.searchsorted(
                starts,
                days,
                side="right",
            )
            - 1
        )

        valid = (
            position
            >= 0
        )

        result[
            rows_array[
                valid
            ]
        ] = (
            labels[
                position[
                    valid
                ]
            ]
        )

    return result


###############################################################################
# Validate the rule against authoritative W3.
###############################################################################

expected_col = find_column(
    w3.columns,
    [
        "treatment_line",
        "line",
    ],
)

if expected_col is None:

    raise RuntimeError(
        "W3 mBC scan table has no line column."
    )


w3_prediction = assign_line(
    w3
)

expected = pd.to_numeric(
    w3[
        expected_col
    ],
    errors="coerce",
).to_numpy(
    dtype=float
)

known = np.isfinite(
    expected
)

agreement = float(
    np.mean(
        w3_prediction[
            known
        ]
        == expected[
            known
        ]
    )
)

null_expected = (
    ~known
)

null_agreement = (
    float(
        np.mean(
            ~np.isfinite(
                w3_prediction[
                    null_expected
                ]
            )
        )
    )
    if null_expected.any()
    else 1.0
)

print(
    "[CKPT6D_LINE_ASSIGNMENT_VALIDATION]",
    {
        "line_col":
            line_col,

        "start_col":
            start_col,

        "known_w3_rows":
            int(
                known.sum()
            ),

        "agreement":
            agreement,

        "null_rows":
            int(
                null_expected.sum()
            ),

        "null_agreement":
            null_agreement,
    },
)

if agreement < 0.999:

    mismatch = w3.loc[
        known
        & (
            w3_prediction
            != expected
        ),
        [
            column
            for column
            in (
                "patient_id",
                "scan_episode_id",
                "landmark_day",
                expected_col,
            )
            if column
            in w3.columns
        ],
    ].head(
        25
    )

    print(
        mismatch.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Derived treatment-line assignment does not "
        "reproduce authoritative W3 with >=99.9% agreement."
    )


###############################################################################
# Filter W0/W7 to frozen mBC patient population and assign line.
###############################################################################

mbc_patients = set(
    splits[
        "patient_id"
    ]
    .astype(str)
    .str.strip()
)

report = {
    "line_assignment": {
        "source":
            "latest treatment-line start <= landmark day",

        "validated_against_authoritative_w3":
            True,

        "agreement":
            agreement,

        "null_agreement":
            null_agreement,

        "line_column":
            line_col,

        "line_start_column":
            start_col,
    },

    "windows":
        {},
}


for window in (
    0,
    7,
):

    source = pd.read_parquet(
        CANON
        / f"chord_breast_scan_episodes_w{window}.parquet"
    )

    source = source.copy()

    source[
        "patient_id"
    ] = (
        source[
            "patient_id"
        ]
        .astype(str)
        .str.strip()
    )

    source = (
        source[
            source[
                "patient_id"
            ].isin(
                mbc_patients
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    assigned = assign_line(
        source
    )

    source[
        "treatment_line"
    ] = pd.Series(
        assigned
    ).astype(
        "Int64"
    )

    target = (
        OUT
        / f"chord_mbc_scan_episodes_w{window}.parquet"
    )

    atomic_parquet(
        target,
        source,
    )

    report[
        "windows"
    ][
        str(window)
    ] = {
        "source_rows":
            int(
                len(
                    pd.read_parquet(
                        CANON
                        / f"chord_breast_scan_episodes_w{window}.parquet",
                        columns=[
                            "patient_id",
                        ],
                    )
                )
            ),

        "mbc_rows":
            int(
                len(source)
            ),

        "patients":
            int(
                source[
                    "patient_id"
                ].nunique()
            ),

        "line_assigned":
            int(
                source[
                    "treatment_line"
                ]
                .notna()
                .sum()
            ),

        "output":
            str(
                target.relative_to(
                    ROOT
                )
            ),
    }


atomic_json(
    OUT
    / "window_input_report.json",
    report,
)

print(
    "[CKPT6D_WINDOW_INPUT_PASS]",
    report,
)
