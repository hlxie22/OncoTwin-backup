#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SEED = 20260927

# Frozen from CKPT6A VALIDATION patient-balanced PFS IBS.
FROZEN_ALPHA = 0.75

HORIZONS = (
    3,
    6,
    12,
    18,
)

MONTH_DAYS = 730.0 / 24.0

CAUSE_CENSOR = 0


###############################################################################
# IO helpers
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
            f"Parquet row count mismatch: {path}"
        )

    tmp.replace(
        path
    )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


###############################################################################
# Frozen module import
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
            f"Unable to import {path}"
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
# Patient-bootstrap Brier gain
###############################################################################


def patient_brier_components(
    ckpt6a,
    frame: pd.DataFrame,
    survival: np.ndarray,
    horizon: int,
    censoring_km,
) -> pd.DataFrame:

    (
        y,
        weight,
    ) = ckpt6a.horizon_status_and_weight(
        frame,
        horizon * MONTH_DAYS,
        censoring_km,
        patient_balanced=True,
    )

    risk = (
        1.0
        - np.asarray(
            survival,
            dtype=float,
        )
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

    temporary = pd.DataFrame(
        {
            "patient_id":
                frame.loc[
                    valid,
                    "patient_id",
                ]
                .astype(str)
                .to_numpy(),

            "numerator":
                (
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
                ),

            "denominator":
                weight[
                    valid
                ],
        }
    )

    return (
        temporary.groupby(
            "patient_id",
            observed=True,
        )[
            [
                "numerator",
                "denominator",
            ]
        ]
        .sum()
    )


def bootstrap_integrated_brier_gain(
    ckpt6a,
    frame: pd.DataFrame,
    earlier_survival: np.ndarray,
    later_survival: np.ndarray,
    censoring_km,
    repetitions: int = 2000,
) -> dict[
    str,
    Any,
]:

    patients = sorted(
        frame[
            "patient_id"
        ]
        .astype(str)
        .unique()
    )

    if len(
        patients
    ) < 20:

        raise RuntimeError(
            "Too few patients for Brier bootstrap."
        )

    earlier_components = {}
    later_components = {}

    observed_earlier = []
    observed_later = []

    for horizon in HORIZONS:

        earlier = patient_brier_components(
            ckpt6a,
            frame,
            earlier_survival[
                :,
                horizon - 1,
            ],
            horizon,
            censoring_km,
        )

        later = patient_brier_components(
            ckpt6a,
            frame,
            later_survival[
                :,
                horizon - 1,
            ],
            horizon,
            censoring_km,
        )

        shared = sorted(
            set(
                earlier.index
            )
            & set(
                later.index
            )
        )

        earlier = earlier.loc[
            shared
        ]

        later = later.loc[
            shared
        ]

        earlier_components[
            horizon
        ] = earlier

        later_components[
            horizon
        ] = later

        observed_earlier.append(
            float(
                earlier[
                    "numerator"
                ].sum()
                / earlier[
                    "denominator"
                ].sum()
            )
        )

        observed_later.append(
            float(
                later[
                    "numerator"
                ].sum()
                / later[
                    "denominator"
                ].sum()
            )
        )

    observed_gain = float(
        np.mean(
            observed_earlier
        )
        - np.mean(
            observed_later
        )
    )

    common_patients = set(
        patients
    )

    for horizon in HORIZONS:

        common_patients &= set(
            earlier_components[
                horizon
            ].index
        )

        common_patients &= set(
            later_components[
                horizon
            ].index
        )

    common_patients = sorted(
        common_patients
    )

    if len(
        common_patients
    ) < 20:

        raise RuntimeError(
            "Too few complete patients for integrated Brier bootstrap."
        )

    rng = np.random.default_rng(
        SEED
    )

    bootstrap = np.empty(
        repetitions,
        dtype=float,
    )

    n_patients = len(
        common_patients
    )

    patient_array = np.asarray(
        common_patients,
        dtype=object,
    )

    for iteration in range(
        repetitions
    ):

        sampled = rng.choice(
            patient_array,
            size=n_patients,
            replace=True,
        )

        earlier_horizon = []
        later_horizon = []

        for horizon in HORIZONS:

            early = (
                earlier_components[
                    horizon
                ]
            )

            late = (
                later_components[
                    horizon
                ]
            )

            early_selected = early.loc[
                sampled
            ]

            late_selected = late.loc[
                sampled
            ]

            earlier_horizon.append(
                float(
                    early_selected[
                        "numerator"
                    ].sum()
                    / early_selected[
                        "denominator"
                    ].sum()
                )
            )

            later_horizon.append(
                float(
                    late_selected[
                        "numerator"
                    ].sum()
                    / late_selected[
                        "denominator"
                    ].sum()
                )
            )

        bootstrap[
            iteration
        ] = (
            np.mean(
                earlier_horizon
            )
            - np.mean(
                later_horizon
            )
        )

    return {
        "mean":
            observed_gain,

        "ci_low":
            float(
                np.quantile(
                    bootstrap,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    bootstrap,
                    0.975,
                )
            ),

        "patients":
            int(
                len(
                    common_patients
                )
            ),

        "repetitions":
            repetitions,
    }


###############################################################################
# Stratified bounded-update diagnostics
###############################################################################


def strata_labels(
    frame: pd.DataFrame,
    mode: str,
) -> pd.Series:

    if mode == "patient_scan_number":

        value = (
            frame[
                "scan_number_patient"
            ]
            .astype(int)
            .to_numpy()
        )

        return pd.Series(
            np.where(
                value == 1,
                "1_FIRST",
                np.where(
                    value == 2,
                    "2_SECOND",
                    "3_PLUS",
                ),
            ),
            index=frame.index,
        )

    if mode == "line_scan_number":

        value = (
            frame[
                "scan_number_line"
            ]
            .astype(int)
            .to_numpy()
        )

        return pd.Series(
            np.where(
                value == 1,
                "1_FIRST",
                np.where(
                    value == 2,
                    "2_SECOND",
                    "3_PLUS",
                ),
            ),
            index=frame.index,
        )

    if mode == "treatment_line":

        value = (
            frame[
                "line"
            ]
            .astype(int)
            .to_numpy()
        )

        return pd.Series(
            np.where(
                value == 1,
                "LINE_1",
                np.where(
                    value == 2,
                    "LINE_2",
                    "LINE_3_PLUS",
                ),
            ),
            index=frame.index,
        )

    if mode == "genomics":

        return pd.Series(
            np.where(
                frame[
                    "genomic_available"
                ]
                .to_numpy(
                    dtype=float
                )
                > 0,
                "AVAILABLE",
                "UNAVAILABLE",
            ),
            index=frame.index,
        )

    if mode == "future_target_type":

        if "target_type" not in frame.columns:

            return pd.Series(
                "UNAVAILABLE",
                index=frame.index,
            )

        return (
            frame[
                "target_type"
            ]
            .astype(str)
            .str.upper()
        )

    raise ValueError(
        mode
    )


def subset_prediction(
    prediction: dict[
        str,
        Any,
    ],
    mask: np.ndarray,
) -> dict[
    str,
    Any,
]:

    return {
        "frame":
            prediction[
                "frame"
            ].loc[
                mask
            ].reset_index(
                drop=True
            ),

        "nll":
            prediction[
                "nll"
            ][
                mask
            ],

        "curves": {
            key:
                value[
                    mask
                ]

            for key, value
            in prediction[
                "curves"
            ].items()
        },
    }


def stratified_metrics(
    ckpt6a,
    pre,
    candidate,
    censoring_km,
) -> dict[
    str,
    Any,
]:

    frame = pre[
        "frame"
    ]

    result = {}

    for mode in (
        "patient_scan_number",
        "line_scan_number",
        "treatment_line",
        "genomics",
        "future_target_type",
    ):

        labels = strata_labels(
            frame,
            mode,
        )

        mode_result = {}

        for label in sorted(
            labels
            .astype(str)
            .unique()
        ):

            mask = (
                labels
                .astype(str)
                .to_numpy()
                == label
            )

            if mask.sum() < 30:
                continue

            pre_subset = subset_prediction(
                pre,
                mask,
            )

            candidate_subset = (
                subset_prediction(
                    candidate,
                    mask,
                )
            )

            pre_metric = (
                ckpt6a.metric_bundle(
                    pre_subset,
                    censoring_km,
                )
            )

            candidate_metric = (
                ckpt6a.metric_bundle(
                    candidate_subset,
                    censoring_km,
                )
            )

            mode_result[
                label
            ] = {
                "rows":
                    int(
                        mask.sum()
                    ),

                "patients":
                    int(
                        pre_subset[
                            "frame"
                        ][
                            "patient_id"
                        ]
                        .nunique()
                    ),

                "pre_patient_nll":
                    pre_metric[
                        "patient_mean_nll"
                    ],

                "candidate_patient_nll":
                    candidate_metric[
                        "patient_mean_nll"
                    ],

                "pre_minus_candidate_patient_nll":
                    (
                        pre_metric[
                            "patient_mean_nll"
                        ]
                        - candidate_metric[
                            "patient_mean_nll"
                        ]
                    ),

                "pre_patient_ibs":
                    pre_metric[
                        "patient_integrated_brier_4h"
                    ],

                "candidate_patient_ibs":
                    candidate_metric[
                        "patient_integrated_brier_4h"
                    ],

                "pre_minus_candidate_patient_ibs":
                    (
                        pre_metric[
                            "patient_integrated_brier_4h"
                        ]
                        - candidate_metric[
                            "patient_integrated_brier_4h"
                        ]
                    ),
            }

        result[
            mode
        ] = mode_result

    return result


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
        "frozen_ckpt6a",
    )

    alpha_report = json.loads(
        (
            repo
            / "artifacts"
            / "checkpoint6a"
            / "alpha_shrinkage.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if (
        alpha_report[
            "selection_population"
        ]
        != "validation_only"
    ):

        raise RuntimeError(
            "CKPT6A alpha was not validation-selected."
        )

    selected_alpha = float(
        alpha_report[
            "selected_alpha_by_patient_pfs_ibs"
        ]
    )

    if not np.isclose(
        selected_alpha,
        FROZEN_ALPHA,
    ):

        raise RuntimeError(
            "Frozen CKPT6B alpha does not equal "
            "CKPT6A validation IBS selection."
        )

    ###########################################################################
    # Load frozen model/data.
    ###########################################################################

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
    # Training-derived censoring KM.
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
    # Recompute PRE / full POST and frozen bounded candidate.
    ###########################################################################

    split_outputs = {}

    for split in (
        "val",
        "test",
    ):

        print(
            "[CKPT6B] infer",
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
                "PRE/POST rows differ."
            )

        bounded_logits = (
            pre[
                "logits"
            ]
            + FROZEN_ALPHA
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

        split_outputs[
            split
        ] = {
            "pre":
                pre,

            "post":
                post,

            "bounded":
                bounded,
        }

    ###########################################################################
    # Metrics.
    ###########################################################################

    report = {
        "status":
            "EVALUATED",

        "candidate_definition": {
            "base_checkpoint":
                "artifacts/checkpoint5/dynamic_scan_model.pt",

            "alpha":
                FROZEN_ALPHA,

            "alpha_selection_population":
                "CKPT6A validation only",

            "alpha_selection_metric":
                (
                    "patient-balanced integrated Brier score "
                    "over 3/6/12/18-month progression/death PFS"
                ),

            "formula":
                (
                    "bounded_logits = PRE_logits + 0.75 * "
                    "(POST_logits - PRE_logits)"
                ),
        },

        "splits":
            {},
    }

    prediction_frames = {}

    for split in (
        "val",
        "test",
    ):

        pre = split_outputs[
            split
        ][
            "pre"
        ]

        post = split_outputs[
            split
        ][
            "post"
        ]

        bounded = split_outputs[
            split
        ][
            "bounded"
        ]

        pre_metric = (
            ckpt6a.metric_bundle(
                pre,
                censoring_km,
            )
        )

        post_metric = (
            ckpt6a.metric_bundle(
                post,
                censoring_km,
            )
        )

        bounded_metric = (
            ckpt6a.metric_bundle(
                bounded,
                censoring_km,
            )
        )

        nll_vs_pre = (
            ckpt6a.bootstrap_nll_gain(
                pre[
                    "frame"
                ],
                pre[
                    "nll"
                ],
                bounded[
                    "nll"
                ],
                repetitions=2000,
            )
        )

        nll_vs_post = (
            ckpt6a.bootstrap_nll_gain(
                pre[
                    "frame"
                ],
                post[
                    "nll"
                ],
                bounded[
                    "nll"
                ],
                repetitions=2000,
            )
        )

        ibs_vs_pre = (
            bootstrap_integrated_brier_gain(
                ckpt6a,
                pre[
                    "frame"
                ],
                pre[
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

        ibs_vs_post = (
            bootstrap_integrated_brier_gain(
                ckpt6a,
                pre[
                    "frame"
                ],
                post[
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

        report[
            "splits"
        ][
            split
        ] = {
            "pre":
                pre_metric,

            "full_post":
                post_metric,

            "bounded_post":
                bounded_metric,

            "pre_minus_bounded_nll":
                nll_vs_pre,

            "full_post_minus_bounded_nll":
                nll_vs_post,

            "pre_minus_bounded_patient_ibs":
                ibs_vs_pre,

            "full_post_minus_bounded_patient_ibs":
                ibs_vs_post,
        }

        #######################################################################
        # Row-level artifact.
        #######################################################################

        frame = pre[
            "frame"
        ].copy()

        output = frame[
            [
                column
                for column
                in (
                    "patient_id",
                    "scan_episode_id",
                    "landmark_day",
                    "line",
                    "scan_state",
                    "split",
                    "target_type",
                    "survival_time_days",
                    "survival_cause",
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
            "full_post_nll"
        ] = post[
            "nll"
        ]

        output[
            "bounded_post_nll"
        ] = bounded[
            "nll"
        ]

        output[
            "alpha"
        ] = FROZEN_ALPHA

        for horizon in HORIZONS:

            position = (
                horizon
                - 1
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
                f"full_post_pfs_{horizon}m"
            ] = (
                post[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    position
                ]
            )

            output[
                f"bounded_post_pfs_{horizon}m"
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
                f"bounded_progression_cif_{horizon}m"
            ] = (
                bounded[
                    "curves"
                ][
                    "progression_cif"
                ][
                    :,
                    position
                ]
            )

            output[
                f"bounded_death_cif_{horizon}m"
            ] = (
                bounded[
                    "curves"
                ][
                    "death_cif"
                ][
                    :,
                    position
                ]
            )

            output[
                f"bounded_switch_cif_{horizon}m"
            ] = (
                bounded[
                    "curves"
                ][
                    "switch_cif"
                ][
                    :,
                    position
                ]
            )

        prediction_frames[
            split
        ] = output

        atomic_parquet(
            out
            / f"{split}_bounded_update_predictions.parquet",
            output,
        )

    ###########################################################################
    # Frozen test stratification.
    #
    # Diagnostic only. It cannot alter alpha or candidate selection.
    ###########################################################################

    test_strata = stratified_metrics(
        ckpt6a,
        split_outputs[
            "test"
        ][
            "pre"
        ],
        split_outputs[
            "test"
        ][
            "bounded"
        ],
        censoring_km,
    )

    atomic_json(
        out
        / "test_stratified_bounded_update.json",
        test_strata,
    )

    ###########################################################################
    # Decision.
    ###########################################################################

    val_result = report[
        "splits"
    ][
        "val"
    ]

    test_result = report[
        "splits"
    ][
        "test"
    ]

    val_ibs_improved = (
        val_result[
            "bounded_post"
        ][
            "patient_integrated_brier_4h"
        ]
        < val_result[
            "pre"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    test_ibs_improved = (
        test_result[
            "bounded_post"
        ][
            "patient_integrated_brier_4h"
        ]
        < test_result[
            "pre"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    test_nll_improved = (
        test_result[
            "bounded_post"
        ][
            "patient_mean_nll"
        ]
        < test_result[
            "pre"
        ][
            "patient_mean_nll"
        ]
    )

    test_ibs_ci_positive = (
        test_result[
            "pre_minus_bounded_patient_ibs"
        ][
            "ci_low"
        ]
        > 0
    )

    test_nll_ci_positive = (
        test_result[
            "pre_minus_bounded_nll"
        ][
            "ci_low"
        ]
        > 0
    )

    if (
        val_ibs_improved
        and test_ibs_improved
        and test_nll_improved
        and test_ibs_ci_positive
        and test_nll_ci_positive
    ):

        status = (
            "PASS_BOUNDED_UPDATE_INTERNAL_CONFIRMATION"
        )

        retraining_decision = (
            "Do not launch broad CKPT6 neural retraining. "
            "The low-capacity validation-selected bounded update "
            "repairs the CKPT5 over-update failure mode. Freeze it "
            "as the internal candidate and proceed to robustness/"
            "ablation confirmation before external validation."
        )

    elif (
        val_ibs_improved
        and test_ibs_improved
        and test_nll_improved
    ):

        status = (
            "PASS_BOUNDED_UPDATE_CANDIDATE_FREEZE"
        )

        retraining_decision = (
            "Freeze the bounded candidate, but retain targeted "
            "survival-only/direct-scan retraining as a secondary "
            "development ablation because one or more bootstrap "
            "intervals still include zero."
        )

    else:

        status = (
            "BOUNDED_UPDATE_NOT_CONFIRMED_RETRAIN_CKPT6B_REQUIRED"
        )

        retraining_decision = (
            "Proceed to the previously planned targeted retraining "
            "matrix: direct scan delta, direct+temporal delta, "
            "survival-only versus multitask, no-genomics, and "
            "patient-balanced versus landmark-balanced training."
        )

    report[
        "status"
    ] = status

    report[
        "retraining_decision"
    ] = retraining_decision

    report[
        "important_methodological_note"
    ] = (
        "Alpha=0.75 was selected before this checkpoint using "
        "CKPT6A validation patient-balanced PFS IBS. CKPT6B does "
        "not select or tune alpha using test outcomes."
    )

    atomic_json(
        out
        / "bounded_update_metrics.json",
        report,
    )

    ###########################################################################
    # Standalone frozen checkpoint.
    ###########################################################################

    candidate_checkpoint = dict(
        base_checkpoint
    )

    candidate_checkpoint[
        "ckpt6b_bounded_update"
    ] = {
        "alpha":
            FROZEN_ALPHA,

        "formula":
            (
                "PRE + alpha * (POST - PRE)"
            ),

        "alpha_selection_population":
            "CKPT6A validation only",

        "alpha_selection_metric":
            (
                "patient-balanced progression/death PFS "
                "integrated Brier across 3/6/12/18 months"
            ),

        "status":
            status,
    }

    tmp_checkpoint = (
        out
        / "bounded_dynamic_scan_candidate.pt.tmp"
    )

    torch.save(
        candidate_checkpoint,
        tmp_checkpoint,
    )

    verify = torch.load(
        tmp_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if not np.isclose(
        float(
            verify[
                "ckpt6b_bounded_update"
            ][
                "alpha"
            ]
        ),
        FROZEN_ALPHA,
    ):

        raise RuntimeError(
            "Candidate checkpoint verification failed."
        )

    tmp_checkpoint.replace(
        out
        / "bounded_dynamic_scan_candidate.pt"
    )

    ###########################################################################
    # Audit / project state / handoff.
    ###########################################################################

    test_pre_ibs = (
        test_result[
            "pre"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    test_post_ibs = (
        test_result[
            "full_post"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    test_bounded_ibs = (
        test_result[
            "bounded_post"
        ][
            "patient_integrated_brier_4h"
        ]
    )

    test_pre_nll = (
        test_result[
            "pre"
        ][
            "patient_mean_nll"
        ]
    )

    test_post_nll = (
        test_result[
            "full_post"
        ][
            "patient_mean_nll"
        ]
    )

    test_bounded_nll = (
        test_result[
            "bounded_post"
        ][
            "patient_mean_nll"
        ]
    )

    audit = f"""# Checkpoint 6B — Frozen bounded scan-update candidate

Status: **{status}**

## Why no broad retraining was performed

CKPT6A established that the existing CKPT5 model contains useful scan
information but applies too large a POST update.

The primary validation-selected alpha was:

**{FROZEN_ALPHA}**

Selection was performed using patient-balanced progression/death PFS
integrated Brier score over 3/6/12/18 months.

No test outcome was used to select alpha.

Candidate definition:

`bounded POST logits = PRE logits + 0.75 * (full POST logits - PRE logits)`

This is a low-capacity calibration/update repair, not a new flexible model.

## Validation

PRE patient IBS:
{val_result['pre']['patient_integrated_brier_4h']}

Full POST patient IBS:
{val_result['full_post']['patient_integrated_brier_4h']}

Bounded POST patient IBS:
{val_result['bounded_post']['patient_integrated_brier_4h']}

PRE patient NLL:
{val_result['pre']['patient_mean_nll']}

Full POST patient NLL:
{val_result['full_post']['patient_mean_nll']}

Bounded POST patient NLL:
{val_result['bounded_post']['patient_mean_nll']}

## Internal test confirmation

PRE patient IBS:
{test_pre_ibs}

Full POST patient IBS:
{test_post_ibs}

Bounded POST patient IBS:
{test_bounded_ibs}

PRE patient NLL:
{test_pre_nll}

Full POST patient NLL:
{test_post_nll}

Bounded POST patient NLL:
{test_bounded_nll}

Patient-bootstrap PRE minus bounded POST IBS:
{test_result['pre_minus_bounded_patient_ibs']}

Patient-bootstrap PRE minus bounded POST NLL:
{test_result['pre_minus_bounded_nll']}

Patient-bootstrap full POST minus bounded POST IBS:
{test_result['full_post_minus_bounded_patient_ibs']}

Patient-bootstrap full POST minus bounded POST NLL:
{test_result['full_post_minus_bounded_nll']}

## Interpretation

{retraining_decision}

## Frozen upstream rules

Unchanged:
- CKPT1 endpoint and scan landmark construction
- W3 episode-end landmarking
- progressive scans excluded from residual-PFS prediction landmarks
- patient-disjoint splits
- persistent longitudinal history across lines
- explicit missingness / unobserved != negative
- CKPT3 genomic encoder
- CKPT4 temporal encoder
- DFCI/VICC outcome quarantine

## Next checkpoint

If the bounded candidate confirms internally, CKPT6C should be a
predeclared robustness/ablation checkpoint rather than another model search:

1. W0/W3/W7 scan episode sensitivity
2. trained/controlled no-genomics ablation
3. survival-only versus multitask as a mechanistic development ablation
4. scan-number and treatment-line robustness
5. calibration table and bootstrap summary
6. final internal candidate lock

Only after CKPT6C should DFCI/VICC outcomes be opened.

Artifacts:
- artifacts/checkpoint6b/bounded_dynamic_scan_candidate.pt
- artifacts/checkpoint6b/bounded_update_metrics.json
- artifacts/checkpoint6b/val_bounded_update_predictions.parquet
- artifacts/checkpoint6b/test_bounded_update_predictions.parquet
- artifacts/checkpoint6b/test_stratified_bounded_update.json
- artifacts/checkpoint6b/audit.md
- artifacts/handoff/checkpoint_06B.json
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
        "<!-- CKPT6B_BOUNDED_SCAN_UPDATE -->"
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
## Checkpoint 6B — Bounded scan-update candidate

Status: **{status}**

CKPT6A diagnosed CKPT5 as an update-amplitude/calibration problem rather
than absence of current-scan signal.

Frozen candidate:

`POST_bounded = PRE + 0.75 * (POST_full - PRE)`

Alpha 0.75 was chosen using CKPT6A validation patient-balanced PFS IBS
only. It was not chosen using test outcomes.

Internal test:
- PRE patient IBS: {test_pre_ibs}
- full POST patient IBS: {test_post_ibs}
- bounded POST patient IBS: {test_bounded_ibs}
- PRE patient NLL: {test_pre_nll}
- full POST patient NLL: {test_post_nll}
- bounded POST patient NLL: {test_bounded_nll}
- PRE minus bounded IBS bootstrap: {test_result['pre_minus_bounded_patient_ibs']}
- PRE minus bounded NLL bootstrap: {test_result['pre_minus_bounded_nll']}

Decision:
{retraining_decision}

DFCI/VICC outcomes remain untouched.

Artifacts:
- artifacts/checkpoint6b/bounded_dynamic_scan_candidate.pt
- artifacts/checkpoint6b/bounded_update_metrics.json
- artifacts/checkpoint6b/test_bounded_update_predictions.parquet
- artifacts/checkpoint6b/audit.md
- artifacts/handoff/checkpoint_06B.json
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
            "6B",

        "name":
            "bounded_scan_update_candidate_freeze",

        "status":
            status,

        "candidate": {
            "base_checkpoint":
                "artifacts/checkpoint5/dynamic_scan_model.pt",

            "frozen_checkpoint":
                (
                    "artifacts/checkpoint6b/"
                    "bounded_dynamic_scan_candidate.pt"
                ),

            "alpha":
                FROZEN_ALPHA,

            "formula":
                "PRE + 0.75 * (POST - PRE)",

            "selection_population":
                "CKPT6A validation only",

            "selection_metric":
                (
                    "patient-balanced PFS integrated Brier "
                    "at 3/6/12/18 months"
                ),
        },

        "validation": {
            "pre_patient_ibs":
                val_result[
                    "pre"
                ][
                    "patient_integrated_brier_4h"
                ],

            "full_post_patient_ibs":
                val_result[
                    "full_post"
                ][
                    "patient_integrated_brier_4h"
                ],

            "bounded_post_patient_ibs":
                val_result[
                    "bounded_post"
                ][
                    "patient_integrated_brier_4h"
                ],
        },

        "test": {
            "pre_patient_ibs":
                test_pre_ibs,

            "full_post_patient_ibs":
                test_post_ibs,

            "bounded_post_patient_ibs":
                test_bounded_ibs,

            "pre_patient_nll":
                test_pre_nll,

            "full_post_patient_nll":
                test_post_nll,

            "bounded_post_patient_nll":
                test_bounded_nll,

            "pre_minus_bounded_ibs":
                test_result[
                    "pre_minus_bounded_patient_ibs"
                ],

            "pre_minus_bounded_nll":
                test_result[
                    "pre_minus_bounded_nll"
                ],
        },

        "retraining_decision":
            retraining_decision,

        "external_validation":
            "DFCI/VICC outcomes remain untouched",

        "next_action":
            (
                "If status confirms the bounded candidate, "
                "run CKPT6C predeclared robustness/ablation "
                "and final internal candidate lock. Otherwise "
                "launch targeted retraining matrix."
            ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_06B.json",
        handoff,
    )

    ###########################################################################
    # Summary
    ###########################################################################

    print("")
    print(
        "========== CKPT6B SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "frozen_alpha="
        f"{FROZEN_ALPHA}"
    )

    print(
        "alpha_selection="
        "validation_patient_balanced_PFS_IBS"
    )

    print(
        "validation_PRE_IBS="
        f"{val_result['pre']['patient_integrated_brier_4h']}"
    )

    print(
        "validation_FULL_POST_IBS="
        f"{val_result['full_post']['patient_integrated_brier_4h']}"
    )

    print(
        "validation_BOUNDED_POST_IBS="
        f"{val_result['bounded_post']['patient_integrated_brier_4h']}"
    )

    print(
        "test_PRE_IBS="
        f"{test_pre_ibs}"
    )

    print(
        "test_FULL_POST_IBS="
        f"{test_post_ibs}"
    )

    print(
        "test_BOUNDED_POST_IBS="
        f"{test_bounded_ibs}"
    )

    print(
        "test_PRE_NLL="
        f"{test_pre_nll}"
    )

    print(
        "test_FULL_POST_NLL="
        f"{test_post_nll}"
    )

    print(
        "test_BOUNDED_POST_NLL="
        f"{test_bounded_nll}"
    )

    print(
        "test_PRE_minus_BOUNDED_IBS_bootstrap="
        f"{test_result['pre_minus_bounded_patient_ibs']}"
    )

    print(
        "test_PRE_minus_BOUNDED_NLL_bootstrap="
        f"{test_result['pre_minus_bounded_nll']}"
    )

    print(
        "retraining_decision="
        f"{retraining_decision}"
    )

    print(
        "candidate_checkpoint="
        "artifacts/checkpoint6b/bounded_dynamic_scan_candidate.pt"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_06B.json"
    )

    print(
        "========== CKPT6B SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT6B DECISION PACKET =========="
    )

    print(
        "validation="
        f"{report['splits']['val']}"
    )

    print(
        "test="
        f"{report['splits']['test']}"
    )

    print(
        "test_strata="
        f"{test_strata}"
    )

    print(
        "status="
        f"{status}"
    )

    print(
        "next_action="
        f"{handoff['next_action']}"
    )

    print(
        "========== CKPT6B DECISION PACKET END =========="
    )


###############################################################################
# CLI
###############################################################################


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-root",
        default=".",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/checkpoint6b",
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

    run(
        repo,
        out,
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
