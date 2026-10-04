#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

G1 = (
    ROOT
    / "artifacts/checkpoint7r3g1_external_prediction_freeze"
)

OUT = (
    ROOT
    / "artifacts/checkpoint7b1_external_endpoint_freeze"
)

REGIMEN = (
    ROOT
    / "data/external_sources/bpc_brca_1_0_public/"
    "clinical_data/regimen_cancer_level_dataset.csv"
)


EXPECTED_MANIFEST_SHA = (
    "88fe02b0d9011828ed62b1e723cb9ae1649d4aa9af91abe5e27c19caef452f2b"
)


REQUIRED_COLUMNS = [
    "record_id",
    "institution",
    "ca_seq",
    "regimen_number",
    "regimen_number_within_cancer",

    "dx_reg_start_int",
    "dx_reg_end_any_int",
    "dx_reg_end_all_int",

    "pfs_i_g_status",
    "tt_pfs_i_g_days",

    "os_g_status",
    "tt_os_g_days",

    "ttnt_any_ca_status",
    "ttnt_any_ca_days",
]


CAUSE_CENSOR = 0
CAUSE_PROGRESSION = 1
CAUSE_DEATH = 2
CAUSE_SWITCH = 3

ADMIN_DAYS = 730.0


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                block
            )

    return h.hexdigest()


def write_json_atomic(
    path: Path,
    payload,
):

    tmp = Path(
        str(
            path
        )
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
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


def write_parquet_atomic(
    path: Path,
    frame: pd.DataFrame,
):

    tmp = Path(
        str(
            path
        )
        + ".tmp"
    )

    frame.to_parquet(
        tmp,
        index=False,
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
            f"Atomic parquet row mismatch: {path}"
        )

    tmp.replace(
        path
    )


def clean_status(
    value,
) -> str | None:

    if pd.isna(
        value
    ):

        return None

    text = str(
        value
    ).strip()

    if not text:

        return None

    return (
        " ".join(
            text.upper().split()
        )
    )


###############################################################################
# Conservative predeclared status parser.
#
# If BPC uses an encoding outside this set, STOP and inspect the already-frozen
# raw extract. Do not reinterpret based on model performance.
###############################################################################

EVENT_EXACT = {
    "1",
    "1.0",
    "EVENT",
    "EVENTED",
    "YES",
    "Y",
    "TRUE",
    "1:EVENT",
    "1: EVENT",
    "1:PROGRESSION",
    "1: PROGRESSION",
    "PROGRESSION",
    "PROGRESSED",
    "DEATH",
    "DEAD",
    "DECEASED",
}

CENSOR_EXACT = {
    "0",
    "0.0",
    "CENSOR",
    "CENSORED",
    "NO",
    "N",
    "FALSE",
    "0:CENSORED",
    "0: CENSORED",
    "ALIVE",
}


def parse_status(
    value,
) -> int | None:

    text = clean_status(
        value
    )

    if text is None:

        return None

    if text in EVENT_EXACT:

        return 1

    if text in CENSOR_EXACT:

        return 0

    ###########################################################################
    # Common cBioPortal style encodings.
    ###########################################################################

    if text.startswith(
        "1:"
    ):

        return 1

    if text.startswith(
        "0:"
    ):

        return 0

    return None


def numeric(
    series,
):

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Prediction manifest re-verification.
    ###########################################################################

    manifest_path = (
        G1
        / "prediction_freeze_manifest.json"
    )

    observed_manifest_sha = sha256_file(
        manifest_path
    )

    if (
        observed_manifest_sha
        != EXPECTED_MANIFEST_SHA
    ):

        raise RuntimeError(
            "Frozen prediction manifest changed before outcome opening."
        )

    predictions = pd.read_parquet(
        G1
        / "timeline_sequencing/"
        "canonical_prediction_index.parquet"
    )

    if len(
        predictions
    ) != 2532:

        raise RuntimeError(
            "Frozen prediction index row count changed."
        )

    center_counts = (
        predictions[
            "site"
        ]
        .value_counts()
        .to_dict()
    )

    if center_counts != {
        "DFCI":
            1728,

        "VICC":
            804,
    }:

        raise RuntimeError(
            f"Frozen center counts changed: {center_counts}"
        )

    frozen_patients = set(
        predictions[
            "patient_id"
        ]
        .astype(str)
    )

    ###########################################################################
    # #########################################################################
    # EXTERNAL OUTCOME EMBARGO IS LIFTED HERE.
    #
    # This is the ONLY original-regimen-table read in this script.
    # #########################################################################
    ###########################################################################

    print(
        "[CKPT7B1_OUTCOME_ACCESS_BEGIN]",
        {
            "source":
                str(
                    REGIMEN.relative_to(
                        ROOT
                    )
                ),

            "prediction_manifest_sha256":
                observed_manifest_sha,
        },
        flush=True,
    )

    raw = pd.read_csv(
        REGIMEN,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
    )

    print(
        "[CKPT7B1_OUTCOME_ACCESS_COMPLETE]",
        {
            "rows_read":
                int(
                    len(
                        raw
                    )
                ),

            "columns":
                REQUIRED_COLUMNS,
        },
        flush=True,
    )

    ###########################################################################
    # From this point forward, original source is never read again.
    ###########################################################################

    raw[
        "record_id"
    ] = raw[
        "record_id"
    ].astype(str)

    relevant = (
        raw[
            raw[
                "record_id"
            ].isin(
                frozen_patients
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    ###########################################################################
    # Freeze exact raw outcome extract FIRST, before interpretation.
    ###########################################################################

    extract_path = (
        OUT
        / "raw_regimen_outcome_extract.parquet"
    )

    write_parquet_atomic(
        extract_path,
        relevant,
    )

    raw_extract_sha = sha256_file(
        extract_path
    )

    ###########################################################################
    # Status-value audit.
    ###########################################################################

    status_columns = [
        "pfs_i_g_status",
        "os_g_status",
        "ttnt_any_ca_status",
    ]

    status_values = {}

    unknown_values = {}

    for column in status_columns:

        counts = (
            relevant[
                column
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts(
                dropna=False
            )
        )

        status_values[
            column
        ] = {
            str(
                value
            ):
                int(
                    count
                )
            for value, count
            in counts.items()
        }

        unknown = []

        for value in counts.index:

            if parse_status(
                value
            ) is None:

                unknown.append(
                    str(
                        value
                    )
                )

        unknown_values[
            column
        ] = sorted(
            set(
                unknown
            )
        )

    ###########################################################################
    # Outcome source audit is ALWAYS written, even if status mapping must stop.
    ###########################################################################

    source_audit = {
        "status":
            (
                "STATUS_MAPPING_REQUIRES_REVIEW"
                if any(
                    unknown_values[
                        column
                    ]
                    for column
                    in status_columns
                )
                else "STATUS_MAPPING_RECOGNIZED"
            ),

        "prediction_manifest_sha256":
            observed_manifest_sha,

        "original_source":
            str(
                REGIMEN.relative_to(
                    ROOT
                )
            ),

        "original_source_sha256":
            sha256_file(
                REGIMEN
            ),

        "original_rows_read_once":
            int(
                len(
                    raw
                )
            ),

        "frozen_prediction_patients":
            int(
                len(
                    frozen_patients
                )
            ),

        "relevant_regimen_rows":
            int(
                len(
                    relevant
                )
            ),

        "raw_extract":
            str(
                extract_path.relative_to(
                    ROOT
                )
            ),

        "raw_extract_sha256":
            raw_extract_sha,

        "status_values":
            status_values,

        "unknown_status_values":
            unknown_values,

        "original_outcome_source_must_not_be_reopened":
            True,

        "model_predictions_changed":
            False,

        "performance_metrics_computed":
            False,
    }

    write_json_atomic(
        OUT
        / "outcome_access_audit.json",
        source_audit,
    )

    ###########################################################################
    # Stop safely if any status encoding is unknown.
    #
    # The raw outcome rows are already frozen, so deterministic mapping can be
    # fixed later WITHOUT reopening the BPC source.
    ###########################################################################

    if any(
        unknown_values[
            column
        ]
        for column
        in status_columns
    ):

        decision = {
            "status":
                "NEEDS_DETERMINISTIC_STATUS_MAPPING",

            "raw_extract_frozen":
                True,

            "raw_extract_sha256":
                raw_extract_sha,

            "unknown_status_values":
                unknown_values,

            "prediction_manifest_sha256":
                observed_manifest_sha,

            "original_outcome_source_reopen":
                "PROHIBITED",

            "external_performance_metrics_computed":
                False,

            "next_action":
                (
                    "Map the reported status strings deterministically from "
                    "the frozen raw_regimen_outcome_extract.parquet, then "
                    "construct endpoints. Do not reopen the original BPC "
                    "regimen table and do not alter predictions."
                ),
        }

        write_json_atomic(
            OUT
            / "decision.json",
            decision,
        )

        print(
            "[CKPT7B1_STATUS_MAPPING_STOP]",
            decision,
            flush=True,
        )

        return

    ###########################################################################
    # Parse numeric fields.
    ###########################################################################

    numeric_columns = [
        "ca_seq",
        "regimen_number",
        "regimen_number_within_cancer",
        "dx_reg_start_int",
        "dx_reg_end_any_int",
        "dx_reg_end_all_int",
        "tt_pfs_i_g_days",
        "tt_os_g_days",
        "ttnt_any_ca_days",
    ]

    for column in numeric_columns:

        relevant[
            column
        ] = numeric(
            relevant[
                column
            ]
        )

    relevant[
        "_pfs_event"
    ] = relevant[
        "pfs_i_g_status"
    ].map(
        parse_status
    )

    relevant[
        "_os_event"
    ] = relevant[
        "os_g_status"
    ].map(
        parse_status
    )

    relevant[
        "_ttnt_event"
    ] = relevant[
        "ttnt_any_ca_status"
    ].map(
        parse_status
    )

    ###########################################################################
    # Basic row validity.
    ###########################################################################

    relevant[
        "_regimen_start"
    ] = relevant[
        "dx_reg_start_int"
    ]

    relevant[
        "_pfs_abs"
    ] = (
        relevant[
            "_regimen_start"
        ]
        + relevant[
            "tt_pfs_i_g_days"
        ]
    )

    relevant[
        "_os_abs"
    ] = (
        relevant[
            "_regimen_start"
        ]
        + relevant[
            "tt_os_g_days"
        ]
    )

    relevant[
        "_ttnt_abs"
    ] = (
        relevant[
            "_regimen_start"
        ]
        + relevant[
            "ttnt_any_ca_days"
        ]
    )

    ###########################################################################
    # Deduplicate exact repeated regimen rows only.
    ###########################################################################

    identity_columns = [
        "record_id",
        "ca_seq",
        "regimen_number",
        "regimen_number_within_cancer",
        "dx_reg_start_int",
    ]

    before = len(
        relevant
    )

    relevant = relevant.drop_duplicates(
        subset=REQUIRED_COLUMNS,
        keep="first",
    ).reset_index(
        drop=True
    )

    exact_duplicates_removed = (
        before
        - len(
            relevant
        )
    )

    ###########################################################################
    # Build patient-specific sorted regimen starts and next-start boundaries.
    ###########################################################################

    relevant[
        "_regimen_row"
    ] = np.arange(
        len(
            relevant
        ),
        dtype=int,
    )

    groups = {}

    for patient_id, frame in relevant.groupby(
        "record_id",
        sort=False,
        observed=True,
    ):

        frame = frame[
            frame[
                "_regimen_start"
            ].notna()
        ].copy()

        frame = frame.sort_values(
            [
                "_regimen_start",
                "regimen_number_within_cancer",
                "regimen_number",
                "_regimen_row",
            ],
            kind="mergesort",
        )

        starts = (
            frame[
                "_regimen_start"
            ]
            .dropna()
            .unique()
        )

        starts = np.sort(
            starts.astype(float)
        )

        next_start_by_start = {}

        for i, current in enumerate(
            starts
        ):

            if (
                i
                + 1
                < len(
                    starts
                )
            ):

                next_start_by_start[
                    float(
                        current
                    )
                ] = float(
                    starts[
                        i
                        + 1
                    ]
                )

            else:

                next_start_by_start[
                    float(
                        current
                    )
                ] = math.inf

        groups[
            str(
                patient_id
            )
        ] = (
            frame,
            next_start_by_start,
        )

    ###########################################################################
    # Landmark -> outcome-only regimen association.
    ###########################################################################

    endpoint_rows = []

    mapping_status_counts = {}

    for _, landmark in predictions.iterrows():

        patient_id = str(
            landmark[
                "patient_id"
            ]
        )

        site = str(
            landmark[
                "site"
            ]
        )

        landmark_day = float(
            landmark[
                "landmark_day"
            ]
        )

        output = {
            "external_landmark_row":
                int(
                    landmark[
                        "external_landmark_row"
                    ]
                ),

            "patient_id":
                patient_id,

            "site":
                site,

            "bpc_episode_id":
                landmark[
                    "bpc_episode_id"
                ],

            "landmark_day":
                landmark_day,

            "outcome_mapping_status":
                None,

            "outcome_regimen_row":
                np.nan,

            "outcome_ca_seq":
                np.nan,

            "outcome_regimen_number":
                np.nan,

            "outcome_regimen_number_within_cancer":
                np.nan,

            "regimen_start_day":
                np.nan,

            "next_regimen_start_day":
                np.nan,

            "pfs_i_status_raw":
                None,

            "os_status_raw":
                None,

            "ttnt_status_raw":
                None,

            "pfs_i_absolute_day":
                np.nan,

            "death_absolute_day":
                np.nan,

            "switch_absolute_day":
                np.nan,

            "survival_time_days":
                np.nan,

            "survival_cause":
                np.nan,

            "survival_cause_name":
                None,

            "administratively_censored":
                False,

            "evaluable":
                False,
        }

        if patient_id not in groups:

            output[
                "outcome_mapping_status"
            ] = "NO_REGIMEN_ROWS"

            endpoint_rows.append(
                output
            )

            continue

        (
            patient_regimens,
            next_start_by_start,
        ) = groups[
            patient_id
        ]

        candidates = patient_regimens[
            patient_regimens[
                "_regimen_start"
            ]
            <= landmark_day
        ].copy()

        if candidates.empty:

            output[
                "outcome_mapping_status"
            ] = "NO_PRIOR_REGIMEN_START"

            endpoint_rows.append(
                output
            )

            continue

        selected_start = float(
            candidates[
                "_regimen_start"
            ].max()
        )

        next_start = float(
            next_start_by_start[
                selected_start
            ]
        )

        #######################################################################
        # Landmark must belong before next distinct regimen begins.
        #######################################################################

        if (
            np.isfinite(
                next_start
            )
            and landmark_day
            >= next_start
        ):

            output[
                "outcome_mapping_status"
            ] = "NOT_WITHIN_SELECTED_REGIMEN_WINDOW"

            endpoint_rows.append(
                output
            )

            continue

        same_start = candidates[
            candidates[
                "_regimen_start"
            ]
            == selected_start
        ].copy()

        #######################################################################
        # Require unique regimen identity at that start.
        #######################################################################

        identity = (
            same_start[
                [
                    "ca_seq",
                    "regimen_number",
                    "regimen_number_within_cancer",
                ]
            ]
            .drop_duplicates()
        )

        if len(
            identity
        ) != 1:

            output[
                "outcome_mapping_status"
            ] = "AMBIGUOUS_REGIMEN_IDENTITY"

            endpoint_rows.append(
                output
            )

            continue

        #######################################################################
        # If exact duplicate rows survived because outcome columns differ,
        # endpoint is ambiguous rather than selected based on values.
        #######################################################################

        if len(
            same_start
        ) != 1:

            output[
                "outcome_mapping_status"
            ] = "MULTIPLE_OUTCOME_ROWS_SAME_REGIMEN"

            endpoint_rows.append(
                output
            )

            continue

        row = same_start.iloc[
            0
        ]

        output[
            "outcome_regimen_row"
        ] = int(
            row[
                "_regimen_row"
            ]
        )

        output[
            "outcome_ca_seq"
        ] = row[
            "ca_seq"
        ]

        output[
            "outcome_regimen_number"
        ] = row[
            "regimen_number"
        ]

        output[
            "outcome_regimen_number_within_cancer"
        ] = row[
            "regimen_number_within_cancer"
        ]

        output[
            "regimen_start_day"
        ] = selected_start

        output[
            "next_regimen_start_day"
        ] = (
            next_start
            if np.isfinite(
                next_start
            )
            else np.nan
        )

        output[
            "pfs_i_status_raw"
        ] = row[
            "pfs_i_g_status"
        ]

        output[
            "os_status_raw"
        ] = row[
            "os_g_status"
        ]

        output[
            "ttnt_status_raw"
        ] = row[
            "ttnt_any_ca_status"
        ]

        #######################################################################
        # Candidate absolute times.
        #######################################################################

        pfs_abs = (
            float(
                row[
                    "_pfs_abs"
                ]
            )
            if pd.notna(
                row[
                    "_pfs_abs"
                ]
            )
            else math.inf
        )

        death_abs = (
            float(
                row[
                    "_os_abs"
                ]
            )
            if (
                row[
                    "_os_event"
                ]
                == 1
                and pd.notna(
                    row[
                        "_os_abs"
                    ]
                )
            )
            else math.inf
        )

        switch_abs = (
            float(
                row[
                    "_ttnt_abs"
                ]
            )
            if (
                row[
                    "_ttnt_event"
                ]
                == 1
                and pd.notna(
                    row[
                        "_ttnt_abs"
                    ]
                )
            )
            else math.inf
        )

        output[
            "pfs_i_absolute_day"
        ] = (
            pfs_abs
            if np.isfinite(
                pfs_abs
            )
            else np.nan
        )

        output[
            "death_absolute_day"
        ] = (
            death_abs
            if np.isfinite(
                death_abs
            )
            else np.nan
        )

        output[
            "switch_absolute_day"
        ] = (
            switch_abs
            if np.isfinite(
                switch_abs
            )
            else np.nan
        )

        #######################################################################
        # Interpret PFS-I.
        #######################################################################

        progression_abs = math.inf
        censor_abs = math.inf

        if (
            row[
                "_pfs_event"
            ]
            == 1
        ):

            if not np.isfinite(
                pfs_abs
            ):

                output[
                    "outcome_mapping_status"
                ] = "PFS_EVENT_WITHOUT_TIME"

                endpoint_rows.append(
                    output
                )

                continue

            ###################################################################
            # PFS event equal to OS event -> death.
            ###################################################################

            if (
                np.isfinite(
                    death_abs
                )
                and abs(
                    death_abs
                    - pfs_abs
                )
                <= 1.0
            ):

                progression_abs = math.inf

            else:

                progression_abs = pfs_abs

        elif (
            row[
                "_pfs_event"
            ]
            == 0
        ):

            if not np.isfinite(
                pfs_abs
            ):

                output[
                    "outcome_mapping_status"
                ] = "PFS_CENSOR_WITHOUT_TIME"

                endpoint_rows.append(
                    output
                )

                continue

            censor_abs = pfs_abs

        else:

            output[
                "outcome_mapping_status"
            ] = "MISSING_PFS_STATUS"

            endpoint_rows.append(
                output
            )

            continue

        #######################################################################
        # Events at/before landmark mean residual PFS is not evaluable here.
        #######################################################################

        candidate_times = [
            (
                death_abs,
                CAUSE_DEATH,
                "DEATH",
                0,
            ),

            (
                progression_abs,
                CAUSE_PROGRESSION,
                "PROGRESSION",
                1,
            ),

            (
                switch_abs,
                CAUSE_SWITCH,
                "SWITCH",
                2,
            ),

            (
                censor_abs,
                CAUSE_CENSOR,
                "CENSOR",
                3,
            ),
        ]

        finite_candidates = [
            item
            for item
            in candidate_times
            if np.isfinite(
                item[
                    0
                ]
            )
        ]

        if not finite_candidates:

            output[
                "outcome_mapping_status"
            ] = "NO_ENDPOINT_TIME"

            endpoint_rows.append(
                output
            )

            continue

        endpoint_abs, cause, cause_name, _ = min(
            finite_candidates,
            key=lambda item:
                (
                    item[
                        0
                    ],
                    item[
                        3
                    ],
                ),
        )

        if endpoint_abs <= landmark_day:

            output[
                "outcome_mapping_status"
            ] = "ENDPOINT_AT_OR_BEFORE_LANDMARK"

            endpoint_rows.append(
                output
            )

            continue

        residual = float(
            endpoint_abs
            - landmark_day
        )

        administratively_censored = False

        if residual > ADMIN_DAYS:

            residual = ADMIN_DAYS
            cause = CAUSE_CENSOR
            cause_name = "CENSOR"
            administratively_censored = True

        output[
            "survival_time_days"
        ] = residual

        output[
            "survival_cause"
        ] = int(
            cause
        )

        output[
            "survival_cause_name"
        ] = cause_name

        output[
            "administratively_censored"
        ] = bool(
            administratively_censored
        )

        output[
            "evaluable"
        ] = True

        output[
            "outcome_mapping_status"
        ] = "EVALUABLE"

        endpoint_rows.append(
            output
        )

    endpoints = pd.DataFrame(
        endpoint_rows
    )

    if len(
        endpoints
    ) != len(
        predictions
    ):

        raise RuntimeError(
            "Endpoint row count is not prediction row count."
        )

    if endpoints[
        "external_landmark_row"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate endpoint landmark rows."
        )

    ###########################################################################
    # Order must match frozen predictions exactly.
    ###########################################################################

    if not np.array_equal(
        endpoints[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        ),
        predictions[
            "external_landmark_row"
        ].to_numpy(
            dtype=int
        ),
    ):

        raise RuntimeError(
            "Endpoint order differs from frozen prediction order."
        )

    write_parquet_atomic(
        OUT
        / "external_landmark_endpoints.parquet",
        endpoints,
    )

    ###########################################################################
    # Descriptive endpoint audit.
    #
    # This is now allowed because the embargo has been lifted, but NO model
    # performance metric is calculated.
    ###########################################################################

    evaluation_summary = {}

    for center in (
        "DFCI",
        "VICC",
    ):

        sub = endpoints[
            endpoints[
                "site"
            ]
            == center
        ]

        evaluable = sub[
            sub[
                "evaluable"
            ].astype(
                bool
            )
        ]

        evaluation_summary[
            center
        ] = {
            "prediction_rows":
                int(
                    len(
                        sub
                    )
                ),

            "prediction_patients":
                int(
                    sub[
                        "patient_id"
                    ].nunique()
                ),

            "evaluable_rows":
                int(
                    len(
                        evaluable
                    )
                ),

            "evaluable_patients":
                int(
                    evaluable[
                        "patient_id"
                    ].nunique()
                ),

            "evaluable_fraction":
                float(
                    len(
                        evaluable
                    )
                    / len(
                        sub
                    )
                )
                if len(
                    sub
                )
                else float(
                    "nan"
                ),

            "cause_counts":
                {
                    str(
                        key
                    ):
                        int(
                            value
                        )
                    for key, value
                    in evaluable[
                        "survival_cause_name"
                    ]
                    .value_counts()
                    .to_dict()
                    .items()
                },

            "mapping_status_counts":
                {
                    str(
                        key
                    ):
                        int(
                            value
                        )
                    for key, value
                    in sub[
                        "outcome_mapping_status"
                    ]
                    .value_counts()
                    .to_dict()
                    .items()
                },
        }

    pooled = endpoints[
        endpoints[
            "evaluable"
        ].astype(
            bool
        )
    ]

    decision = {
        "status":
            "PASS_EXTERNAL_ENDPOINT_FREEZE",

        "prediction_manifest_sha256":
            observed_manifest_sha,

        "raw_outcome_extract_sha256":
            raw_extract_sha,

        "primary_endpoint":
            "REGIMEN_LEVEL_IMAGING_DERIVED_PFS_I",

        "endpoint_file":
            (
                "artifacts/checkpoint7b1_external_endpoint_freeze/"
                "external_landmark_endpoints.parquet"
            ),

        "endpoint_file_sha256":
            sha256_file(
                OUT
                / "external_landmark_endpoints.parquet"
            ),

        "exact_duplicate_regimen_rows_removed":
            int(
                exact_duplicates_removed
            ),

        "centers":
            evaluation_summary,

        "pooled_evaluable_rows":
            int(
                len(
                    pooled
                )
            ),

        "pooled_evaluable_patients":
            int(
                pooled[
                    "patient_id"
                ].nunique()
            ),

        "original_outcome_source_opened":
            True,

        "original_outcome_source_must_not_be_reopened":
            True,

        "model_predictions_changed":
            False,

        "performance_metrics_computed":
            False,

        "next_action":
            (
                "Run CKPT7B2 external evaluation using only the frozen "
                "external_landmark_endpoints.parquet and already-hashed G1 "
                "predictions. Do not reopen the original BPC regimen outcome "
                "table."
            ),
    }

    write_json_atomic(
        OUT
        / "decision.json",
        decision,
    )

    print(
        "[CKPT7B1_ENDPOINT_FREEZE_PASS]",
        decision,
        flush=True,
    )


if __name__ == "__main__":

    main()
