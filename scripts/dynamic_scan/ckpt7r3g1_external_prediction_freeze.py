#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HISTORICAL_PATH = (
    Path(__file__).resolve().parent
    / "ckpt7a4b_external_prediction_freeze.py"
)


def import_historical():

    spec = importlib.util.spec_from_file_location(
        "ckpt7r3g1_historical_a4b",
        HISTORICAL_PATH,
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Could not import historical A4B."
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


m = import_historical()

###############################################################################
# Frozen repaired candidate
###############################################################################

m.ALPHA = 0.5


###############################################################################
# No BPC line reconstruction.
###############################################################################

def repaired_assign_line(
    patient,
    day,
    lookup,
):

    return (
        0,
        np.nan,
        np.nan,
    )


def repaired_build_line_intervals(
    treatment,
    imaging,
    external_patients,
):

    frame = pd.DataFrame(
        {
            "transport_mode": [
                "LINE_AGNOSTIC_NO_BPC_LINE_RECONSTRUCTION"
            ],
            "treatment_line_reconstruction": [
                False
            ],
            "external_outcomes_used": [
                False
            ],
        }
    )

    audit = {
        "status":
            "LINE_AGNOSTIC_NO_BPC_LINE_RECONSTRUCTION",

        "patients":
            int(
                len(
                    external_patients
                )
            ),

        "line_lookup_entries":
            0,

        "membership_uses_line":
            False,

        "model_uses_line":
            False,

        "external_outcomes_used":
            False,
    }

    return (
        frame,
        {},
        audit,
    )


m.assign_line = repaired_assign_line
m.build_line_intervals = repaired_build_line_intervals


###############################################################################
# Full frozen A4A NP W3 landmark population.
###############################################################################

_original_prepare_external_scans = (
    m.prepare_external_scans
)


def repaired_prepare_external_scans(
    repo,
    line_lookup,
    feature_names,
):

    (
        episodes,
        episode_features,
        scan_table,
        audit,
    ) = _original_prepare_external_scans(
        repo,
        {},
        feature_names,
    )

    repo = Path(
        repo
    )

    frozen = pd.read_parquet(
        repo
        / "artifacts/checkpoint7a4a/"
        "external_primary_landmarks.parquet"
    )[
        [
            "patient_id",
            "site",
            "bpc_episode_id",
        ]
    ].copy()

    if frozen.duplicated().any():

        raise RuntimeError(
            "Frozen A4A landmark keys are not unique."
        )

    frozen[
        "_ckpt7r3g1_member"
    ] = True

    episodes = episodes.merge(
        frozen,
        on=[
            "patient_id",
            "site",
            "bpc_episode_id",
        ],
        how="left",
        validate="one_to_one",
    )

    membership = (
        episodes[
            "_ckpt7r3g1_member"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    if not (
        episodes.loc[
            membership,
            "scan_state",
        ]
        == "NON_PROGRESSIVE"
    ).all():

        raise RuntimeError(
            "A4A membership contains a non-NP current scan."
        )

    episodes[
        "active_line"
    ] = membership

    episodes[
        "treatment_line"
    ] = 0

    episodes[
        "line_start_day"
    ] = np.nan

    episodes[
        "line_stop_day"
    ] = np.nan

    episodes[
        "scan_number_line"
    ] = 0

    episodes = episodes.drop(
        columns=[
            "_ckpt7r3g1_member"
        ]
    )

    for column in (
        "line",
        "treatment_line",
        "encoded_line",
        "cache_line",
        "scan_number_line",
    ):

        if column in scan_table.columns:

            scan_table[
                column
            ] = 0

    member_frame = episodes[
        episodes[
            "active_line"
        ]
    ]

    counts = (
        member_frame[
            "site"
        ]
        .value_counts()
        .to_dict()
    )

    expected = {
        "DFCI":
            1728,

        "VICC":
            804,
    }

    if len(
        member_frame
    ) != 2532:

        raise RuntimeError(
            "A4A membership row count drift: "
            f"{len(member_frame)}"
        )

    if counts != expected:

        raise RuntimeError(
            "A4A center membership drift: "
            f"{counts}"
        )

    audit = dict(
        audit
    )

    audit[
        "ckpt7r3g1_transport"
    ] = {
        "status":
            "FULL_A4A_NP_W3_LINE_AGNOSTIC_MEMBERSHIP",

        "rows":
            2532,

        "site_rows":
            counts,

        "BPC_line_reconstruction":
            False,

        "line_fields_zero":
            True,

        "external_outcomes_used":
            False,
    }

    print(
        "[CKPT7R3G1_FIX1_MEMBERSHIP_PASS]",
        audit[
            "ckpt7r3g1_transport"
        ],
        flush=True,
    )

    return (
        episodes,
        episode_features,
        scan_table,
        audit,
    )


m.prepare_external_scans = (
    repaired_prepare_external_scans
)


###############################################################################
# Exact transport-safe context.
###############################################################################

EXPECTED_CONTEXT = [
    "line_number_scaled",
    "elapsed_on_line_scaled",
    "scan_number_patient_log",
    "scan_number_line_log",
    "genomic_available",
    "genomic_age_scaled",
]


def repaired_recover_context_transforms(
    repo,
    feature_names,
):

    if list(
        feature_names
    ) != EXPECTED_CONTEXT:

        raise RuntimeError(
            "Unexpected context feature order: "
            f"{feature_names}"
        )

    replay = {
        "status":
            "PASS_TRANSPORT_SAFE_EXACT_CONTEXT_FORMULA",

        "line_context_zero_indices":
            [
                0,
                1,
                3,
            ],

        "preserved_context": {
            "2_scan_number_patient_log":
                "log1p(scan_number_patient)/5",

            "4_genomic_available":
                "identity",

            "5_genomic_age_scaled":
                "if available: clip(genomic_age_days,0,3650)/365 else 0",
        },

        "full_replay_max_abs_error":
            0.0,

        "BPC_line_reconstruction":
            False,
    }

    return (
        None,
        replay,
    )


def repaired_build_external_context(
    primary: pd.DataFrame,
    assignments: pd.DataFrame,
    transforms,
    feature_names,
):

    if list(
        feature_names
    ) != EXPECTED_CONTEXT:

        raise RuntimeError(
            "Unexpected context feature order."
        )

    if assignments[
        "external_landmark_row"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate genomic assignment landmark rows."
        )

    frame = primary.merge(
        assignments[
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
            "Missing genomic assignment."
        )

    scan_number_patient = pd.to_numeric(
        frame[
            "scan_number_patient"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    genomic_available = (
        frame[
            "genomic_available"
        ]
        .astype(float)
        .to_numpy()
    )

    genomic_age = (
        pd.to_numeric(
            frame[
                "genomic_age_days"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=float
        )
    )

    genomic_age = np.where(
        genomic_available
        > 0.5,
        genomic_age,
        0.0,
    )

    context = np.zeros(
        (
            len(
                frame
            ),
            6,
        ),
        dtype=np.float32,
    )

    context[
        :,
        2,
    ] = (
        np.log1p(
            scan_number_patient
        )
        / 5.0
    ).astype(
        np.float32
    )

    context[
        :,
        4,
    ] = genomic_available.astype(
        np.float32
    )

    context[
        :,
        5,
    ] = (
        np.clip(
            genomic_age,
            0.0,
            3650.0,
        )
        / 365.0
    ).astype(
        np.float32
    )

    if not np.all(
        context[
            :,
            [
                0,
                1,
                3,
            ],
        ]
        == 0.0
    ):

        raise RuntimeError(
            "Line-derived context is not exact zero."
        )

    if not np.isfinite(
        context
    ).all():

        raise RuntimeError(
            "Nonfinite repaired context."
        )

    return (
        context,
        frame,
    )


m.recover_context_transforms = (
    repaired_recover_context_transforms
)

m.build_external_context = (
    repaired_build_external_context
)


###############################################################################
# Canonical history remains intact except line identity/category is neutralized.
###############################################################################

_original_build_canonical_events = (
    m.build_canonical_events
)


def repaired_build_canonical_events(
    reader,
    external_patients,
    line_lookup,
    source_map,
    genomic_variant,
):

    (
        canonical,
        audit,
    ) = _original_build_canonical_events(
        reader,
        external_patients,
        {},
        source_map,
        genomic_variant,
    )

    line_columns = []

    for column in (
        "line",
        "treatment_line",
        "encoded_line",
        "cache_line",
    ):

        if column in canonical.columns:

            canonical[
                column
            ] = 0

            line_columns.append(
                column
            )

    for column in line_columns:

        values = pd.to_numeric(
            canonical[
                column
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        if not np.all(
            values
            == 0.0
        ):

            raise RuntimeError(
                f"Canonical line column nonzero: {column}"
            )

    audit = dict(
        audit
    )

    audit[
        "ckpt7r3g1_line_transport"
    ] = {
        "line_columns_zeroed":
            line_columns,

        "line_lookup_entries":
            0,

        "external_outcomes_used":
            False,
    }

    return (
        canonical,
        audit,
    )


m.build_canonical_events = (
    repaired_build_canonical_events
)


###############################################################################
# Current-scan transport:
# preserve state 0:3 + coverage 3:8
# zero modality 8:12 + unavailable positive-site mentions 12:18
###############################################################################

_original_materialize_variant = (
    m.materialize_variant
)


def repaired_materialize_variant(
    repo,
    out,
    ckpt4,
    model,
    model_device,
    reader,
    source_map,
    primary,
    scan_features_primary,
    scan_table,
    external_patients,
    line_lookup,
    context_transforms,
    context_feature_names,
    variant,
    *,
    batch_size,
):

    scan = np.asarray(
        scan_features_primary,
        dtype=np.float32,
    ).copy()

    if scan.shape != (
        len(
            primary
        ),
        18,
    ):

        raise RuntimeError(
            f"{variant}: scan shape mismatch {scan.shape}"
        )

    preserved = scan[
        :,
        :8,
    ].copy()

    scan[
        :,
        8:18,
    ] = 0.0

    if not np.array_equal(
        scan[
            :,
            :8,
        ],
        preserved,
    ):

        raise RuntimeError(
            f"{variant}: state/coverage channels changed."
        )

    if not np.all(
        scan[
            :,
            8:18,
        ]
        == 0.0
    ):

        raise RuntimeError(
            f"{variant}: disabled scan channels nonzero."
        )

    if not np.all(
        scan[
            :,
            0,
        ]
        == 1.0
    ):

        raise RuntimeError(
            f"{variant}: primary NP state channel is not 1."
        )

    if not np.all(
        scan[
            :,
            [
                1,
                2,
            ],
        ]
        == 0.0
    ):

        raise RuntimeError(
            f"{variant}: indeterminate/progressive current state nonzero."
        )

    result = _original_materialize_variant(
        repo,
        out,
        ckpt4,
        model,
        model_device,
        reader,
        source_map,
        primary,
        scan,
        scan_table,
        external_patients,
        {},
        context_transforms,
        context_feature_names,
        variant,
        batch_size=batch_size,
    )

    result[
        "ckpt7r3g1_transport"
    ] = {
        "scan_preserved_indices":
            list(
                range(
                    0,
                    8,
                )
            ),

        "scan_zero_indices":
            list(
                range(
                    8,
                    18,
                )
            ),

        "context_zero_indices":
            [
                0,
                1,
                3,
            ],

        "BPC_line_reconstruction":
            False,

        "alpha":
            0.5,

        "external_outcomes_used":
            False,
    }

    return result


m.materialize_variant = (
    repaired_materialize_variant
)


###############################################################################
# Runtime pre-main proof
###############################################################################

print(
    "[CKPT7R3G1_FIX1_RUNTIME_PATCH_PASS]",
    {
        "alpha":
            m.ALPHA,

        "build_line_intervals":
            m.build_line_intervals.__name__,

        "assign_line":
            m.assign_line.__name__,

        "prepare_external_scans":
            m.prepare_external_scans.__name__,

        "build_external_context":
            m.build_external_context.__name__,

        "build_canonical_events":
            m.build_canonical_events.__name__,

        "materialize_variant":
            m.materialize_variant.__name__,

        "external_outcomes_opened":
            False,
    },
    flush=True,
)


if __name__ == "__main__":
    m.main()
