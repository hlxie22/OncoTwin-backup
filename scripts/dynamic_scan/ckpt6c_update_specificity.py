#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SEED = 20260927

ALPHA = 0.75

MONTHS = 24
MONTH_DAYS = 730.0 / MONTHS

HORIZONS = (
    3,
    6,
    12,
    18,
)

PERMUTATIONS = 500


###############################################################################
# IO
###############################################################################


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


###############################################################################
# Import frozen implementations
###############################################################################


def import_module(
    path: Path,
    name: str,
):

    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"cannot import {path}"
        )

    module = (
        importlib.util.module_from_spec(
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
# Patient / stratum helpers
###############################################################################


def patient_weights(
    frame: pd.DataFrame,
) -> np.ndarray:

    count = (
        frame.groupby(
            "patient_id",
            observed=True,
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
            count,
            1.0,
        )
    )


def patient_average(
    values: np.ndarray,
    patient_ids: np.ndarray,
) -> float:

    values = np.asarray(
        values,
        dtype=float,
    )

    patient_ids = np.asarray(
        patient_ids,
        dtype=str,
    )

    (
        _,
        inverse,
    ) = np.unique(
        patient_ids,
        return_inverse=True,
    )

    numerator = np.bincount(
        inverse,
        weights=values,
    )

    denominator = np.bincount(
        inverse,
    )

    return float(
        np.mean(
            numerator
            / denominator
        )
    )


def scan_bucket(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=int,
    )

    return np.where(
        values == 1,
        "S1",
        np.where(
            values == 2,
            "S2",
            "S3P",
        ),
    )


def line_bucket(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=int,
    )

    return np.where(
        values == 1,
        "L1",
        np.where(
            values == 2,
            "L2",
            "L3P",
        ),
    )


def make_strata(
    frame: pd.DataFrame,
) -> np.ndarray:

    scan = scan_bucket(
        frame[
            "scan_number_patient"
        ].to_numpy(
            dtype=int
        )
    )

    line = line_bucket(
        frame[
            "line"
        ].to_numpy(
            dtype=int
        )
    )

    genomic = np.where(
        frame[
            "genomic_available"
        ].to_numpy(
            dtype=float
        )
        > 0,
        "G1",
        "G0",
    )

    return np.asarray(
        [
            f"{s}|{l}|{g}"

            for s, l, g
            in zip(
                scan,
                line,
                genomic,
            )
        ],
        dtype=object,
    )


###############################################################################
# Mean-delta controls fit on validation only
###############################################################################


def weighted_mean_delta(
    frame: pd.DataFrame,
    delta: np.ndarray,
) -> np.ndarray:

    weight = patient_weights(
        frame
    )

    denominator = float(
        weight.sum()
    )

    return (
        (
            delta
            * weight[
                :,
                None,
                None,
            ]
        ).sum(
            axis=0
        )
        / denominator
    )


def fit_stratum_mean_delta(
    frame: pd.DataFrame,
    delta: np.ndarray,
) -> tuple[
    dict[
        str,
        np.ndarray,
    ],
    np.ndarray,
]:

    strata = make_strata(
        frame
    )

    global_mean = weighted_mean_delta(
        frame,
        delta,
    )

    fitted = {}

    for label in sorted(
        np.unique(
            strata
        )
    ):

        mask = (
            strata
            == label
        )

        if mask.sum() < 20:
            continue

        fitted[
            str(
                label
            )
        ] = weighted_mean_delta(
            frame.loc[
                mask
            ].reset_index(
                drop=True
            ),
            delta[
                mask
            ],
        )

    return (
        fitted,
        global_mean,
    )


def apply_stratum_mean_delta(
    frame: pd.DataFrame,
    fitted: dict[
        str,
        np.ndarray,
    ],
    global_mean: np.ndarray,
) -> np.ndarray:

    strata = make_strata(
        frame
    )

    output = np.zeros(
        (
            len(
                frame
            ),
            MONTHS,
            4,
        ),
        dtype=np.float32,
    )

    for index, label in enumerate(
        strata
    ):

        output[
            index
        ] = fitted.get(
            str(
                label
            ),
            global_mean,
        )

    return output


###############################################################################
# Fast metrics for permutation null
###############################################################################


def log_softmax_numpy(
    logits: np.ndarray,
) -> np.ndarray:

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    maximum = np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    shifted = (
        logits
        - maximum
    )

    log_denom = np.log(
        np.sum(
            np.exp(
                shifted
            ),
            axis=-1,
            keepdims=True,
        )
    )

    return (
        shifted
        - log_denom
    )


def nll_rows(
    logits: np.ndarray,
    time_days: np.ndarray,
    cause: np.ndarray,
) -> np.ndarray:

    log_probability = (
        log_softmax_numpy(
            logits
        )
    )

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    cause = np.asarray(
        cause,
        dtype=int,
    )

    last_month = np.ceil(
        time_days
        / MONTH_DAYS
    ).astype(
        int
    )

    last_month = np.clip(
        last_month,
        1,
        MONTHS,
    )

    result = np.empty(
        len(
            time_days
        ),
        dtype=np.float64,
    )

    for row in range(
        len(
            time_days
        )
    ):

        month = int(
            last_month[
                row
            ]
        )

        current_cause = int(
            cause[
                row
            ]
        )

        if current_cause == 0:

            result[
                row
            ] = -float(
                log_probability[
                    row,
                    :month,
                    0,
                ].sum()
            )

        else:

            result[
                row
            ] = -float(
                log_probability[
                    row,
                    :month - 1,
                    0,
                ].sum()
                + log_probability[
                    row,
                    month - 1,
                    current_cause,
                ]
            )

    return result


def pfs_curve_numpy(
    logits: np.ndarray,
) -> np.ndarray:

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    maximum = np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    probability = np.exp(
        logits
        - maximum
    )

    probability /= probability.sum(
        axis=-1,
        keepdims=True,
    )

    n = probability.shape[
        0
    ]

    event_free = np.ones(
        n,
        dtype=np.float64,
    )

    progression = np.zeros(
        n,
        dtype=np.float64,
    )

    death = np.zeros(
        n,
        dtype=np.float64,
    )

    pfs = np.zeros(
        (
            n,
            MONTHS,
        ),
        dtype=np.float64,
    )

    for month in range(
        MONTHS
    ):

        current = probability[
            :,
            month,
        ]

        progression += (
            event_free
            * current[
                :,
                1
            ]
        )

        death += (
            event_free
            * current[
                :,
                2
            ]
        )

        event_free *= (
            current[
                :,
                0
            ]
        )

        pfs[
            :,
            month
        ] = (
            1.0
            - progression
            - death
        )

    return pfs


def prepare_brier_targets(
    ckpt6a,
    frame: pd.DataFrame,
    censoring_km,
) -> dict[
    int,
    tuple[
        np.ndarray,
        np.ndarray,
    ],
]:

    output = {}

    for horizon in HORIZONS:

        (
            y,
            weight,
        ) = (
            ckpt6a.horizon_status_and_weight(
                frame,
                horizon
                * MONTH_DAYS,
                censoring_km,
                patient_balanced=True,
            )
        )

        output[
            horizon
        ] = (
            np.asarray(
                y,
                dtype=np.float64,
            ),
            np.asarray(
                weight,
                dtype=np.float64,
            ),
        )

    return output


def patient_ibs_fast(
    pfs: np.ndarray,
    targets: dict[
        int,
        tuple[
            np.ndarray,
            np.ndarray,
        ],
    ],
) -> float:

    values = []

    for horizon in HORIZONS:

        y, weight = (
            targets[
                horizon
            ]
        )

        risk = (
            1.0
            - pfs[
                :,
                horizon - 1,
            ]
        )

        valid = (
            np.isfinite(
                risk
            )
            & np.isfinite(
                y
            )
            & np.isfinite(
                weight
            )
            & (
                weight
                > 0
            )
        )

        numerator = np.sum(
            weight[
                valid
            ]
            * (
                y[
                    valid
                ]
                - risk[
                    valid
                ]
            )
            ** 2
        )

        denominator = np.sum(
            weight[
                valid
            ]
        )

        values.append(
            float(
                numerator
                / denominator
            )
        )

    return float(
        np.mean(
            values
        )
    )


###############################################################################
# Permutation negative control
###############################################################################


def permutation_null(
    ckpt6a,
    frame: pd.DataFrame,
    pre_logits: np.ndarray,
    observed_delta: np.ndarray,
    censoring_km,
    repetitions: int,
) -> dict[
    str,
    Any,
]:

    strata = make_strata(
        frame
    )

    unique_strata = sorted(
        np.unique(
            strata
        )
    )

    patient_ids = (
        frame[
            "patient_id"
        ]
        .astype(str)
        .to_numpy()
    )

    time_days = (
        frame[
            "survival_time_days"
        ]
        .to_numpy(
            dtype=float
        )
    )

    cause = (
        frame[
            "survival_cause"
        ]
        .astype(int)
        .to_numpy()
    )

    targets = prepare_brier_targets(
        ckpt6a,
        frame,
        censoring_km,
    )

    pre_nll = nll_rows(
        pre_logits,
        time_days,
        cause,
    )

    pre_patient_nll = (
        patient_average(
            pre_nll,
            patient_ids,
        )
    )

    pre_ibs = patient_ibs_fast(
        pfs_curve_numpy(
            pre_logits
        ),
        targets,
    )

    observed_logits = (
        pre_logits
        + ALPHA
        * observed_delta
    )

    observed_nll = nll_rows(
        observed_logits,
        time_days,
        cause,
    )

    observed_patient_nll = (
        patient_average(
            observed_nll,
            patient_ids,
        )
    )

    observed_ibs = patient_ibs_fast(
        pfs_curve_numpy(
            observed_logits
        ),
        targets,
    )

    observed_nll_gain = (
        pre_patient_nll
        - observed_patient_nll
    )

    observed_ibs_gain = (
        pre_ibs
        - observed_ibs
    )

    rng = np.random.default_rng(
        SEED
    )

    null_nll_gain = np.empty(
        repetitions,
        dtype=float,
    )

    null_ibs_gain = np.empty(
        repetitions,
        dtype=float,
    )

    for repetition in range(
        repetitions
    ):

        permuted_delta = np.empty_like(
            observed_delta
        )

        for label in unique_strata:

            rows = np.where(
                strata
                == label
            )[
                0
            ]

            if len(
                rows
            ) <= 1:

                permuted_delta[
                    rows
                ] = observed_delta[
                    rows
                ]

                continue

            source = rng.permutation(
                rows
            )

            permuted_delta[
                rows
            ] = observed_delta[
                source
            ]

        logits = (
            pre_logits
            + ALPHA
            * permuted_delta
        )

        row_nll = nll_rows(
            logits,
            time_days,
            cause,
        )

        patient_nll = (
            patient_average(
                row_nll,
                patient_ids,
            )
        )

        ibs = patient_ibs_fast(
            pfs_curve_numpy(
                logits
            ),
            targets,
        )

        null_nll_gain[
            repetition
        ] = (
            pre_patient_nll
            - patient_nll
        )

        null_ibs_gain[
            repetition
        ] = (
            pre_ibs
            - ibs
        )

    nll_p = float(
        (
            1
            + np.sum(
                null_nll_gain
                >= observed_nll_gain
            )
        )
        / (
            repetitions
            + 1
        )
    )

    ibs_p = float(
        (
            1
            + np.sum(
                null_ibs_gain
                >= observed_ibs_gain
            )
        )
        / (
            repetitions
            + 1
        )
    )

    return {
        "repetitions":
            repetitions,

        "stratification":
            (
                "patient_scan_number x treatment_line x genomic_availability"
            ),

        "observed": {
            "patient_nll_gain":
                float(
                    observed_nll_gain
                ),

            "patient_ibs_gain":
                float(
                    observed_ibs_gain
                ),
        },

        "permuted": {
            "nll_gain_mean":
                float(
                    null_nll_gain.mean()
                ),

            "nll_gain_p95":
                float(
                    np.quantile(
                        null_nll_gain,
                        0.95,
                    )
                ),

            "nll_gain_p99":
                float(
                    np.quantile(
                        null_nll_gain,
                        0.99,
                    )
                ),

            "ibs_gain_mean":
                float(
                    null_ibs_gain.mean()
                ),

            "ibs_gain_p95":
                float(
                    np.quantile(
                        null_ibs_gain,
                        0.95,
                    )
                ),

            "ibs_gain_p99":
                float(
                    np.quantile(
                        null_ibs_gain,
                        0.99,
                    )
                ),
        },

        "empirical_one_sided_p": {
            "nll":
                nll_p,

            "ibs":
                ibs_p,
        },
    }


###############################################################################
# Self test
###############################################################################


def self_test() -> None:

    rng = np.random.default_rng(
        123
    )

    logits = rng.normal(
        size=(
            12,
            MONTHS,
            4,
        )
    ).astype(
        np.float32
    )

    pfs = pfs_curve_numpy(
        logits
    )

    assert pfs.shape == (
        12,
        MONTHS,
    )

    assert np.isfinite(
        pfs
    ).all()

    assert (
        pfs
        >= -1e-8
    ).all()

    assert (
        pfs
        <= 1.0 + 1e-8
    ).all()

    assert (
        pfs[
            :,
            1:
        ]
        <= pfs[
            :,
            :-1
        ]
        + 1e-8
    ).all()

    times = np.linspace(
        20,
        700,
        12,
    )

    causes = np.asarray(
        [
            1,
            0,
            2,
            3,
            1,
            0,
            1,
            2,
            3,
            0,
            1,
            2,
        ]
    )

    nll = nll_rows(
        logits,
        times,
        causes,
    )

    assert nll.shape == (
        12,
    )

    assert np.isfinite(
        nll
    ).all()

    patients = np.asarray(
        [
            "a",
            "a",
            "b",
            "b",
            "c",
            "c",
            "d",
            "d",
            "e",
            "e",
            "f",
            "f",
        ]
    )

    value = patient_average(
        nll,
        patients,
    )

    assert np.isfinite(
        value
    )

    print(
        "[CKPT6C_SELF_TEST_PASS]"
    )


###############################################################################
# Main
###############################################################################


def run(
    repo: Path,
    out: Path,
) -> None:

    ckpt6a = import_module(
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt6a_diagnose_scan_update.py",
        "frozen_ckpt6a_for_ckpt6c",
    )

    ckpt6b = import_module(
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt6b_freeze_bounded_update.py",
        "frozen_ckpt6b_for_ckpt6c",
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    data = ckpt6a.load_prepared(
        repo
    )

    (
        ckpt5,
        base_checkpoint,
        model,
    ) = ckpt6a.load_model(
        repo,
        device,
    )

    ###########################################################################
    # Training-only censoring distribution.
    ###########################################################################

    index = data[
        "index"
    ]

    train = index[
        (
            index[
                "split"
            ]
            == "train"
        )
        & index[
            "survival_mask"
        ].astype(
            bool
        )
    ].copy()

    censoring_km = (
        ckpt6a.fit_censoring_km(
            train[
                "survival_time_days"
            ]
            .to_numpy(
                dtype=float
            ),
            train[
                "survival_cause"
            ]
            .astype(int)
            .to_numpy(),
        )
    )

    ###########################################################################
    # Frozen PRE / POST predictions.
    ###########################################################################

    predictions = {}

    for split in (
        "val",
        "test",
    ):

        print(
            "[CKPT6C] infer",
            split,
            flush=True,
        )

        pre = ckpt6a.predict_variant(
            ckpt5,
            model,
            data,
            split,
            "PRE",
            device,
        )

        post = ckpt6a.predict_variant(
            ckpt5,
            model,
            data,
            split,
            "POST",
            device,
        )

        if not pre[
            "frame"
        ][
            [
                "patient_id",
                "scan_episode_id",
            ]
        ].equals(
            post[
                "frame"
            ][
                [
                    "patient_id",
                    "scan_episode_id",
                ]
            ]
        ):

            raise RuntimeError(
                "PRE/POST row alignment mismatch."
            )

        bounded_logits = (
            pre[
                "logits"
            ]
            + ALPHA
            * (
                post[
                    "logits"
                ]
                - pre[
                    "logits"
                ]
            )
        )

        bounded = (
            ckpt6a.prediction_from_logits(
                ckpt5,
                pre[
                    "frame"
                ],
                bounded_logits,
            )
        )

        predictions[
            split
        ] = {
            "pre":
                pre,

            "post":
                post,

            "bounded":
                bounded,

            "delta":
                (
                    post[
                        "logits"
                    ]
                    - pre[
                        "logits"
                    ]
                ),
        }

    ###########################################################################
    # Fit non-individualized controls using VALIDATION ONLY.
    ###########################################################################

    val_frame = predictions[
        "val"
    ][
        "pre"
    ][
        "frame"
    ]

    val_delta = predictions[
        "val"
    ][
        "delta"
    ]

    global_mean_delta = (
        weighted_mean_delta(
            val_frame,
            val_delta,
        )
    )

    (
        stratum_mean_table,
        fallback_mean_delta,
    ) = fit_stratum_mean_delta(
        val_frame,
        val_delta,
    )

    np.save(
        out
        / "validation_global_mean_delta.npy",
        global_mean_delta,
    )

    np.savez_compressed(
        out
        / "validation_stratum_mean_delta.npz",
        **{
            key.replace(
                "|",
                "__",
            ):
                value

            for key, value
            in stratum_mean_table.items()
        },
    )

    ###########################################################################
    # Apply frozen controls to validation and test.
    ###########################################################################

    results = {}

    row_artifacts = {}

    for split in (
        "val",
        "test",
    ):

        pre = predictions[
            split
        ][
            "pre"
        ]

        post = predictions[
            split
        ][
            "post"
        ]

        bounded = predictions[
            split
        ][
            "bounded"
        ]

        frame = pre[
            "frame"
        ]

        global_delta = np.broadcast_to(
            global_mean_delta,
            (
                len(
                    frame
                ),
                MONTHS,
                4,
            ),
        )

        stratum_delta = (
            apply_stratum_mean_delta(
                frame,
                stratum_mean_table,
                fallback_mean_delta,
            )
        )

        global_control = (
            ckpt6a.prediction_from_logits(
                ckpt5,
                frame,
                (
                    pre[
                        "logits"
                    ]
                    + ALPHA
                    * global_delta
                ),
            )
        )

        stratum_control = (
            ckpt6a.prediction_from_logits(
                ckpt5,
                frame,
                (
                    pre[
                        "logits"
                    ]
                    + ALPHA
                    * stratum_delta
                ),
            )
        )

        pre_metric = (
            ckpt6a.metric_bundle(
                pre,
                censoring_km,
            )
        )

        bounded_metric = (
            ckpt6a.metric_bundle(
                bounded,
                censoring_km,
            )
        )

        global_metric = (
            ckpt6a.metric_bundle(
                global_control,
                censoring_km,
            )
        )

        stratum_metric = (
            ckpt6a.metric_bundle(
                stratum_control,
                censoring_km,
            )
        )

        bounded_vs_global_nll = (
            ckpt6a.bootstrap_nll_gain(
                frame,
                global_control[
                    "nll"
                ],
                bounded[
                    "nll"
                ],
                repetitions=2000,
            )
        )

        bounded_vs_stratum_nll = (
            ckpt6a.bootstrap_nll_gain(
                frame,
                stratum_control[
                    "nll"
                ],
                bounded[
                    "nll"
                ],
                repetitions=2000,
            )
        )

        bounded_vs_global_ibs = (
            ckpt6b.bootstrap_integrated_brier_gain(
                ckpt6a,
                frame,
                global_control[
                    "curves"
                ][
                    "pfs"
                ],
                bounded[
                    "curves"
                ][
                    "pfs"
                ],
                censoring_km,
                repetitions=2000,
            )
        )

        bounded_vs_stratum_ibs = (
            ckpt6b.bootstrap_integrated_brier_gain(
                ckpt6a,
                frame,
                stratum_control[
                    "curves"
                ][
                    "pfs"
                ],
                bounded[
                    "curves"
                ][
                    "pfs"
                ],
                censoring_km,
                repetitions=2000,
            )
        )

        results[
            split
        ] = {
            "pre":
                pre_metric,

            "bounded_individualized":
                bounded_metric,

            "validation_global_mean_delta":
                global_metric,

            "validation_stratum_mean_delta":
                stratum_metric,

            "global_mean_minus_individualized_nll":
                bounded_vs_global_nll,

            "stratum_mean_minus_individualized_nll":
                bounded_vs_stratum_nll,

            "global_mean_minus_individualized_ibs":
                bounded_vs_global_ibs,

            "stratum_mean_minus_individualized_ibs":
                bounded_vs_stratum_ibs,
        }

        output = frame[
            [
                column

                for column in (
                    "patient_id",
                    "scan_episode_id",
                    "landmark_day",
                    "line",
                    "scan_state",
                    "split",
                    "survival_time_days",
                    "survival_cause",
                    "target_type",
                    "scan_number_patient",
                    "scan_number_line",
                    "genomic_available",
                    "genomic_age_days",
                )

                if column
                in frame.columns
            ]
        ].copy()

        output[
            "pre_nll"
        ] = pre[
            "nll"
        ]

        output[
            "bounded_individualized_nll"
        ] = bounded[
            "nll"
        ]

        output[
            "global_mean_delta_nll"
        ] = global_control[
            "nll"
        ]

        output[
            "stratum_mean_delta_nll"
        ] = stratum_control[
            "nll"
        ]

        for horizon in HORIZONS:

            position = (
                horizon - 1
            )

            output[
                f"pre_pfs_{horizon}m"
            ] = (
                pre[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    position
                ]
            )

            output[
                f"bounded_pfs_{horizon}m"
            ] = (
                bounded[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    position
                ]
            )

            output[
                f"global_mean_pfs_{horizon}m"
            ] = (
                global_control[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    position
                ]
            )

            output[
                f"stratum_mean_pfs_{horizon}m"
            ] = (
                stratum_control[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    position
                ]
            )

        row_artifacts[
            split
        ] = output

        atomic_parquet(
            out
            / f"{split}_specificity_predictions.parquet",
            output,
        )

    ###########################################################################
    # Test permutation null.
    #
    # Delta values are shuffled only within observable baseline strata.
    ###########################################################################

    permutation = permutation_null(
        ckpt6a,
        predictions[
            "test"
        ][
            "pre"
        ][
            "frame"
        ],
        predictions[
            "test"
        ][
            "pre"
        ][
            "logits"
        ],
        predictions[
            "test"
        ][
            "delta"
        ],
        censoring_km,
        repetitions=PERMUTATIONS,
    )

    atomic_json(
        out
        / "permutation_negative_control.json",
        permutation,
    )

    ###########################################################################
    # Decision.
    ###########################################################################

    test = results[
        "test"
    ]

    pre_ibs = (
        test[
            "pre"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    bounded_ibs = (
        test[
            "bounded_individualized"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    pre_nll = (
        test[
            "pre"
        ][
            "patient_mean_nll"
        ]
    )

    bounded_nll = (
        test[
            "bounded_individualized"
        ][
            "patient_mean_nll"
        ]
    )

    stratum_nll_ci = (
        test[
            "stratum_mean_minus_individualized_nll"
        ]
    )

    stratum_ibs_ci = (
        test[
            "stratum_mean_minus_individualized_ibs"
        ]
    )

    permutation_nll_specific = (
        permutation[
            "empirical_one_sided_p"
        ][
            "nll"
        ]
        < 0.05
    )

    permutation_ibs_specific = (
        permutation[
            "empirical_one_sided_p"
        ][
            "ibs"
        ]
        < 0.05
    )

    beats_stratum_nll = (
        stratum_nll_ci[
            "ci_low"
        ]
        > 0
    )

    beats_stratum_ibs = (
        stratum_ibs_ci[
            "ci_low"
        ]
        > 0
    )

    primary_point_improves = (
        bounded_ibs
        < pre_ibs
    )

    nll_improves = (
        bounded_nll
        < pre_nll
    )

    specificity_nll = (
        beats_stratum_nll
        and permutation_nll_specific
    )

    specificity_ibs = (
        beats_stratum_ibs
        and permutation_ibs_specific
    )

    if (
        primary_point_improves
        and nll_improves
        and (
            specificity_nll
            or specificity_ibs
        )
    ):

        status = (
            "PASS_INTERNAL_DYNAMIC_CANDIDATE_LOCK"
        )

        next_action = (
            "Keep the CKPT6B bounded candidate frozen. "
            "Proceed to CKPT6D W0/W3/W7 grouping robustness "
            "using the frozen encoders, then open CKPT7 external "
            "validation only if window sensitivity is acceptable."
        )

    else:

        status = (
            "NEEDS_TARGETED_DIRECT_SCAN_RETRAINING"
        )

        next_action = (
            "Do not open external validation. Train the targeted "
            "direct-scan residual-update model on train/validation "
            "only, with survival-only and multitask arms, because "
            "the current bounded improvement is not sufficiently "
            "individualized beyond calibration/stratum effects."
        )

    report = {
        "status":
            status,

        "candidate":
            (
                "CKPT6B alpha=0.75 bounded "
                "PRE-to-POST update"
            ),

        "test": {
            "pre_patient_ibs":
                pre_ibs,

            "bounded_patient_ibs":
                bounded_ibs,

            "pre_patient_nll":
                pre_nll,

            "bounded_patient_nll":
                bounded_nll,

            "global_mean_delta_patient_ibs":
                test[
                    "validation_global_mean_delta"
                ][
                    "patient_integrated_brier_4h"
                ],

            "stratum_mean_delta_patient_ibs":
                test[
                    "validation_stratum_mean_delta"
                ][
                    "patient_integrated_brier_4h"
                ],

            "global_mean_delta_patient_nll":
                test[
                    "validation_global_mean_delta"
                ][
                    "patient_mean_nll"
                ],

            "stratum_mean_delta_patient_nll":
                test[
                    "validation_stratum_mean_delta"
                ][
                    "patient_mean_nll"
                ],

            "stratum_mean_minus_individualized_nll":
                stratum_nll_ci,

            "stratum_mean_minus_individualized_ibs":
                stratum_ibs_ci,
        },

        "permutation_negative_control":
            permutation,

        "specificity_flags": {
            "bounded_point_ibs_improves":
                primary_point_improves,

            "bounded_nll_improves":
                nll_improves,

            "beats_validation_stratum_mean_nll":
                beats_stratum_nll,

            "beats_validation_stratum_mean_ibs":
                beats_stratum_ibs,

            "permutation_nll_specific":
                permutation_nll_specific,

            "permutation_ibs_specific":
                permutation_ibs_specific,

            "individualized_specificity_nll":
                specificity_nll,

            "individualized_specificity_ibs":
                specificity_ibs,
        },

        "methodological_guardrails": [
            (
                "alpha remains 0.75 selected by "
                "CKPT6A validation only"
            ),
            (
                "global and stratum mean controls are "
                "fit using validation deltas only"
            ),
            (
                "no candidate parameter is selected "
                "using test outcomes"
            ),
            (
                "permutation null preserves patient-scan-number, "
                "treatment-line, and genomic-availability strata"
            ),
            (
                "DFCI/VICC outcomes remain untouched"
            ),
        ],

        "next_action":
            next_action,
    }

    atomic_json(
        out
        / "specificity_report.json",
        report,
    )

    atomic_json(
        out
        / "control_metrics.json",
        results,
    )

    ###########################################################################
    # Project state and handoff
    ###########################################################################

    audit = f"""# Checkpoint 6C — Scan-update specificity / negative control

Status: **{status}**

## Frozen candidate

CKPT6B bounded update:

`PRE + 0.75 * (POST - PRE)`

Alpha remains frozen from CKPT6A validation patient-balanced PFS IBS.

No retraining and no test-based tuning occurred.

## Scientific question

Does the bounded update contain patient/scan-specific information, or can its
benefit be reproduced by a generic calibration shift whenever a scan occurs?

## Controls

1. Validation-global mean POST-PRE delta
2. Validation stratum-specific mean delta
   - patient scan number
   - treatment line
   - genomic availability
3. Test permutation null
   - individual POST-PRE deltas shuffled within those same observable strata

These controls preserve broad update magnitude/calibration effects while
destroying or removing patient-specific scan assignment.

## Test performance

PRE patient IBS:
{pre_ibs}

Individualized bounded patient IBS:
{bounded_ibs}

Validation-global-mean-delta patient IBS:
{test['validation_global_mean_delta']['patient_integrated_brier_4h']}

Validation-stratum-mean-delta patient IBS:
{test['validation_stratum_mean_delta']['patient_integrated_brier_4h']}

PRE patient NLL:
{pre_nll}

Individualized bounded patient NLL:
{bounded_nll}

Validation-global-mean-delta patient NLL:
{test['validation_global_mean_delta']['patient_mean_nll']}

Validation-stratum-mean-delta patient NLL:
{test['validation_stratum_mean_delta']['patient_mean_nll']}

Stratum mean minus individualized NLL bootstrap:
{stratum_nll_ci}

Stratum mean minus individualized IBS bootstrap:
{stratum_ibs_ci}

Permutation test:
{permutation}

## Decision

{next_action}

## Deferred

W0/W7 performance is deliberately not approximated by remapping W3 predictions.
If CKPT6C passes, alternate scan-window states should be regenerated using the
frozen temporal encoder so the sensitivity analysis is scientifically exact.

DFCI/VICC remain outcome-frozen.
"""

    atomic_text(
        out
        / "audit.md",
        audit,
    )

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
        "<!-- CKPT6C_UPDATE_SPECIFICITY -->"
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
## Checkpoint 6C — Scan-update specificity

Status: **{status}**

Frozen candidate remains:

`PRE + 0.75 * (POST - PRE)`

No retraining or test-based tuning was performed.

Negative controls:
- validation global mean scan update
- validation observable-stratum mean scan update
- within-stratum permutation of individual test scan updates

Test PRE patient IBS:
{pre_ibs}

Test individualized bounded patient IBS:
{bounded_ibs}

Test PRE patient NLL:
{pre_nll}

Test individualized bounded patient NLL:
{bounded_nll}

Permutation control:
{permutation}

Specificity flags:
{report['specificity_flags']}

Next:
{next_action}

DFCI/VICC outcomes remain untouched.

Artifacts:
- artifacts/checkpoint6c/specificity_report.json
- artifacts/checkpoint6c/control_metrics.json
- artifacts/checkpoint6c/permutation_negative_control.json
- artifacts/checkpoint6c/val_specificity_predictions.parquet
- artifacts/checkpoint6c/test_specificity_predictions.parquet
- artifacts/checkpoint6c/audit.md
- artifacts/handoff/checkpoint_06C.json
"""

    atomic_text(
        state_path,
        existing.rstrip()
        + "\n\n"
        + section.strip()
        + "\n",
    )

    handoff = {
        "checkpoint":
            "6C",

        "name":
            "scan_update_specificity_negative_control",

        "status":
            status,

        "candidate": {
            "checkpoint":
                "artifacts/checkpoint6b/bounded_dynamic_scan_candidate.pt",

            "alpha":
                ALPHA,

            "formula":
                "PRE + 0.75 * (POST - PRE)",
        },

        "test": {
            "pre_patient_ibs":
                pre_ibs,

            "bounded_patient_ibs":
                bounded_ibs,

            "pre_patient_nll":
                pre_nll,

            "bounded_patient_nll":
                bounded_nll,

            "stratum_mean_minus_individualized_nll":
                stratum_nll_ci,

            "stratum_mean_minus_individualized_ibs":
                stratum_ibs_ci,

            "permutation":
                permutation,
        },

        "specificity_flags":
            report[
                "specificity_flags"
            ],

        "external_validation":
            "DFCI/VICC outcomes remain untouched",

        "next_action":
            next_action,
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_06C.json",
        handoff,
    )

    ###########################################################################
    # Console output
    ###########################################################################

    print("")
    print(
        "========== CKPT6C SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        f"frozen_alpha={ALPHA}"
    )

    print(
        f"test_PRE_patient_IBS={pre_ibs}"
    )

    print(
        "test_INDIVIDUALIZED_patient_IBS="
        f"{bounded_ibs}"
    )

    print(
        "test_GLOBAL_MEAN_DELTA_patient_IBS="
        f"{test['validation_global_mean_delta']['patient_integrated_brier_4h']}"
    )

    print(
        "test_STRATUM_MEAN_DELTA_patient_IBS="
        f"{test['validation_stratum_mean_delta']['patient_integrated_brier_4h']}"
    )

    print(
        f"test_PRE_patient_NLL={pre_nll}"
    )

    print(
        "test_INDIVIDUALIZED_patient_NLL="
        f"{bounded_nll}"
    )

    print(
        "test_GLOBAL_MEAN_DELTA_patient_NLL="
        f"{test['validation_global_mean_delta']['patient_mean_nll']}"
    )

    print(
        "test_STRATUM_MEAN_DELTA_patient_NLL="
        f"{test['validation_stratum_mean_delta']['patient_mean_nll']}"
    )

    print(
        "STRATUM_MEAN_minus_INDIVIDUALIZED_NLL_bootstrap="
        f"{stratum_nll_ci}"
    )

    print(
        "STRATUM_MEAN_minus_INDIVIDUALIZED_IBS_bootstrap="
        f"{stratum_ibs_ci}"
    )

    print(
        "permutation_empirical_p="
        f"{permutation['empirical_one_sided_p']}"
    )

    print(
        "specificity_flags="
        f"{report['specificity_flags']}"
    )

    print(
        f"next_action={next_action}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_06C.json"
    )

    print(
        "========== CKPT6C SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT6C DECISION PACKET =========="
    )

    print(
        "validation_controls="
        f"{results['val']}"
    )

    print(
        "test_controls="
        f"{results['test']}"
    )

    print(
        "permutation_negative_control="
        f"{permutation}"
    )

    print(
        "specificity_flags="
        f"{report['specificity_flags']}"
    )

    print(
        f"status={status}"
    )

    print(
        f"next_action={next_action}"
    )

    print(
        "========== CKPT6C DECISION PACKET END =========="
    )


###############################################################################
# CLI
###############################################################################


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "self-test",
            "run",
        ],
    )

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint6c",
    )

    args = parser.parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    output_arg = Path(
        args.output_dir
    )

    out = (
        output_arg.resolve()
        if output_arg.is_absolute()
        else (
            repo
            / output_arg
        ).resolve()
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.command == "self-test":

        self_test()

    else:

        run(
            repo,
            out,
        )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
