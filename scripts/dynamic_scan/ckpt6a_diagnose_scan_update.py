#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score


SEED = 20260927

MONTHS = 24
MONTH_DAYS = 730.0 / MONTHS

HORIZONS_MONTHS = (
    3,
    6,
    12,
    18,
)

CAUSE_CENSOR = 0
CAUSE_PROGRESSION = 1
CAUSE_DEATH = 2
CAUSE_SWITCH = 3

EPS = 1e-7


###############################################################################
# General utilities
###############################################################################


def norm(
    value: Any,
) -> str:

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(
            value
        ).strip().lower(),
    ).strip(
        "_"
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
            default=lambda value:
                int(
                    value
                )
                if isinstance(
                    value,
                    np.integer,
                )
                else float(
                    value
                )
                if isinstance(
                    value,
                    np.floating,
                )
                else bool(
                    value
                )
                if isinstance(
                    value,
                    np.bool_,
                )
                else str(
                    value
                ),
        ),
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

    check = pd.read_parquet(
        tmp
    )

    if len(
        check
    ) != len(
        frame
    ):

        raise RuntimeError(
            f"Parquet row-count verification failed: {path}"
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
            "Unable to find any requested column.\n"
            f"requested={list(candidates)}\n"
            f"available={list(columns)}"
        )

    return None


def import_ckpt5(
    repo: Path,
):

    path = (
        repo
        / "scripts"
        / "dynamic_scan"
        / "ckpt5_supervised_dynamic_model.py"
    )

    spec = (
        importlib.util.spec_from_file_location(
            "frozen_ckpt5",
            path,
        )
    )

    if (
        spec is None
        or spec.loader
        is None
    ):

        raise RuntimeError(
            "Unable to import frozen CKPT5 implementation."
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
# Censoring distribution
###############################################################################


def fit_censoring_km(
    time_days: np.ndarray,
    cause: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    time_days = np.asarray(
        time_days,
        dtype=float,
    )

    cause = np.asarray(
        cause,
        dtype=int,
    )

    order = np.argsort(
        time_days,
        kind="mergesort",
    )

    time_days = time_days[
        order
    ]

    cause = cause[
        order
    ]

    unique_times = np.unique(
        time_days
    )

    at_risk = int(
        len(
            time_days
        )
    )

    survival = 1.0

    output_times = []
    output_survival = []

    for current_time in unique_times:

        mask = (
            time_days
            == current_time
        )

        number_at_time = int(
            mask.sum()
        )

        censor_events = int(
            (
                cause[
                    mask
                ]
                == CAUSE_CENSOR
            ).sum()
        )

        if (
            at_risk
            > 0
            and censor_events
            > 0
        ):

            survival *= (
                1.0
                - censor_events
                / at_risk
            )

        output_times.append(
            float(
                current_time
            )
        )

        output_survival.append(
            float(
                max(
                    survival,
                    EPS,
                )
            )
        )

        at_risk -= (
            number_at_time
        )

    return (
        np.asarray(
            output_times,
            dtype=float,
        ),
        np.asarray(
            output_survival,
            dtype=float,
        ),
    )


def censoring_survival_at(
    km: tuple[
        np.ndarray,
        np.ndarray,
    ],
    time: float,
    *,
    left_limit: bool,
) -> float:

    times, values = km

    index = (
        np.searchsorted(
            times,
            float(
                time
            ),
            side=(
                "left"
                if left_limit
                else "right"
            ),
        )
        - 1
    )

    if index < 0:
        return 1.0

    return float(
        max(
            values[
                index
            ],
            EPS,
        )
    )


###############################################################################
# Horizon outcome / IPCW
###############################################################################


def patient_balance_factor(
    frame: pd.DataFrame,
) -> np.ndarray:

    counts = (
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
            counts,
            1.0,
        )
    )


def horizon_status_and_weight(
    frame: pd.DataFrame,
    horizon_days: float,
    censoring_km,
    *,
    patient_balanced: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

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
        .astype(
            int
        )
        .to_numpy()
    )

    y = np.zeros(
        len(
            frame
        ),
        dtype=float,
    )

    weight = np.zeros(
        len(
            frame
        ),
        dtype=float,
    )

    for index in range(
        len(
            frame
        )
    ):

        t = float(
            time_days[
                index
            ]
        )

        c = int(
            cause[
                index
            ]
        )

        if t > horizon_days:

            y[
                index
            ] = 0.0

            weight[
                index
            ] = (
                1.0
                / censoring_survival_at(
                    censoring_km,
                    horizon_days,
                    left_limit=False,
                )
            )

            continue

        if c == CAUSE_CENSOR:

            # Unknown target status after censoring.
            continue

        # Progression/death are the patient-facing PFS events.
        y[
            index
        ] = float(
            c
            in {
                CAUSE_PROGRESSION,
                CAUSE_DEATH,
            }
        )

        # Switch is an observed competing event and therefore an observed
        # non-case for the progression/death subdistribution estimand.
        weight[
            index
        ] = (
            1.0
            / censoring_survival_at(
                censoring_km,
                t,
                left_limit=True,
            )
        )

    if patient_balanced:

        weight *= (
            patient_balance_factor(
                frame
            )
        )

    return (
        y,
        weight,
    )


###############################################################################
# Calibration metrics
###############################################################################


def sigmoid(
    value: np.ndarray,
) -> np.ndarray:

    value = np.clip(
        value,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -value
            )
        )
    )


def weighted_logistic_calibration(
    predicted_risk: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
) -> dict[
    str,
    float,
]:

    predicted_risk = np.clip(
        np.asarray(
            predicted_risk,
            dtype=float,
        ),
        1e-5,
        1.0
        - 1e-5,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    weight = np.asarray(
        weight,
        dtype=float,
    )

    use = (
        np.isfinite(
            predicted_risk
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

    if (
        use.sum()
        < 20
    ):

        return {
            "intercept":
                float(
                    "nan"
                ),

            "slope":
                float(
                    "nan"
                ),
        }

    p = predicted_risk[
        use
    ]

    target = y[
        use
    ]

    w = weight[
        use
    ]

    logit = np.log(
        p
        / (
            1.0
            - p
        )
    )

    x = np.column_stack(
        [
            np.ones(
                len(
                    logit
                )
            ),
            logit,
        ]
    )

    beta = np.asarray(
        [
            0.0,
            1.0,
        ],
        dtype=float,
    )

    for _ in range(
        50
    ):

        eta = (
            x
            @ beta
        )

        probability = sigmoid(
            eta
        )

        gradient = (
            x.T
            @ (
                w
                * (
                    probability
                    - target
                )
            )
        )

        curvature = (
            w
            * probability
            * (
                1.0
                - probability
            )
        )

        hessian = (
            x.T
            @ (
                curvature[
                    :,
                    None
                ]
                * x
            )
        )

        hessian += (
            np.eye(
                2
            )
            * 1e-6
        )

        try:

            step = np.linalg.solve(
                hessian,
                gradient,
            )

        except np.linalg.LinAlgError:

            break

        beta_next = (
            beta
            - step
        )

        if np.max(
            np.abs(
                beta_next
                - beta
            )
        ) < 1e-8:

            beta = beta_next
            break

        beta = beta_next

    return {
        "intercept":
            float(
                beta[
                    0
                ]
            ),

        "slope":
            float(
                beta[
                    1
                ]
            ),
    }


def weighted_ece(
    predicted_risk: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bins: int = 10,
) -> float:

    predicted_risk = np.asarray(
        predicted_risk,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    weight = np.asarray(
        weight,
        dtype=float,
    )

    use = (
        np.isfinite(
            predicted_risk
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

    if use.sum() < 20:
        return float(
            "nan"
        )

    p = predicted_risk[
        use
    ]

    target = y[
        use
    ]

    w = weight[
        use
    ]

    order = np.argsort(
        p
    )

    groups = np.array_split(
        order,
        min(
            bins,
            len(
                order
            ),
        ),
    )

    total_weight = float(
        w.sum()
    )

    value = 0.0

    for group in groups:

        if len(
            group
        ) == 0:
            continue

        group_weight = float(
            w[
                group
            ].sum()
        )

        if group_weight <= 0:
            continue

        mean_prediction = float(
            np.sum(
                w[
                    group
                ]
                * p[
                    group
                ]
            )
            / group_weight
        )

        observed = float(
            np.sum(
                w[
                    group
                ]
                * target[
                    group
                ]
            )
            / group_weight
        )

        value += (
            group_weight
            / total_weight
            * abs(
                mean_prediction
                - observed
            )
        )

    return float(
        value
    )


###############################################################################
# Horizon metrics
###############################################################################


def horizon_metrics(
    frame: pd.DataFrame,
    predicted_survival: np.ndarray,
    horizon_months: int,
    censoring_km,
    *,
    patient_balanced: bool,
) -> dict[
    str,
    Any,
]:

    predicted_survival = np.asarray(
        predicted_survival,
        dtype=float,
    )

    risk = (
        1.0
        - predicted_survival
    )

    (
        y,
        weight,
    ) = horizon_status_and_weight(
        frame,
        horizon_months
        * MONTH_DAYS,
        censoring_km,
        patient_balanced=patient_balanced,
    )

    use = (
        np.isfinite(
            risk
        )
        & (
            weight
            > 0
        )
    )

    if use.sum() < 20:

        return {
            "n_known":
                int(
                    use.sum()
                ),
        }

    risk_use = risk[
        use
    ]

    y_use = y[
        use
    ]

    weight_use = weight[
        use
    ]

    denominator = float(
        weight_use.sum()
    )

    brier = float(
        np.sum(
            weight_use
            * (
                y_use
                - risk_use
            )
            ** 2
        )
        / denominator
    )

    if (
        len(
            np.unique(
                y_use
            )
        )
        >= 2
    ):

        auc = float(
            roc_auc_score(
                y_use,
                risk_use,
                sample_weight=weight_use,
            )
        )

    else:

        auc = float(
            "nan"
        )

    calibration = (
        weighted_logistic_calibration(
            risk_use,
            y_use,
            weight_use,
        )
    )

    ece = weighted_ece(
        risk_use,
        y_use,
        weight_use,
    )

    mean_predicted_risk = float(
        np.sum(
            weight_use
            * risk_use
        )
        / denominator
    )

    observed_risk = float(
        np.sum(
            weight_use
            * y_use
        )
        / denominator
    )

    return {
        "n_known":
            int(
                use.sum()
            ),

        "brier":
            brier,

        "auc":
            auc,

        "calibration_intercept":
            calibration[
                "intercept"
            ],

        "calibration_slope":
            calibration[
                "slope"
            ],

        "ece":
            ece,

        "mean_predicted_risk":
            mean_predicted_risk,

        "observed_risk":
            observed_risk,
    }


###############################################################################
# Prediction curves from competing-risk logits
###############################################################################


def curves_from_logits(
    logits: np.ndarray,
) -> dict[
    str,
    np.ndarray,
]:

    tensor = torch.from_numpy(
        np.asarray(
            logits,
            dtype=np.float32,
        )
    )

    conditional = torch.softmax(
        tensor,
        dim=-1,
    ).numpy()

    n = conditional.shape[
        0
    ]

    event_free = np.ones(
        n,
        dtype=np.float64,
    )

    cif_progression = np.zeros(
        n,
        dtype=np.float64,
    )

    cif_death = np.zeros(
        n,
        dtype=np.float64,
    )

    cif_switch = np.zeros(
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

    progression = np.zeros_like(
        pfs
    )

    death = np.zeros_like(
        pfs
    )

    switch = np.zeros_like(
        pfs
    )

    no_event = np.zeros_like(
        pfs
    )

    for month in range(
        MONTHS
    ):

        current = conditional[
            :,
            month,
        ]

        cif_progression += (
            event_free
            * current[
                :,
                CAUSE_PROGRESSION
            ]
        )

        cif_death += (
            event_free
            * current[
                :,
                CAUSE_DEATH
            ]
        )

        cif_switch += (
            event_free
            * current[
                :,
                CAUSE_SWITCH
            ]
        )

        event_free *= (
            current[
                :,
                CAUSE_CENSOR
            ]
        )

        pfs[
            :,
            month
        ] = (
            1.0
            - cif_progression
            - cif_death
        )

        progression[
            :,
            month
        ] = cif_progression

        death[
            :,
            month
        ] = cif_death

        switch[
            :,
            month
        ] = cif_switch

        no_event[
            :,
            month
        ] = event_free

    entropy = -np.sum(
        conditional
        * np.log(
            np.clip(
                conditional,
                EPS,
                1.0,
            )
        ),
        axis=-1,
    ).mean(
        axis=1
    )

    return {
        "pfs":
            pfs,

        "progression_cif":
            progression,

        "death_cif":
            death,

        "switch_cif":
            switch,

        "event_free":
            no_event,

        "entropy":
            entropy,
    }


###############################################################################
# Frozen CKPT5 model
###############################################################################


def load_model(
    repo: Path,
    device: torch.device,
):

    ckpt5 = import_ckpt5(
        repo
    )

    checkpoint = torch.load(
        repo
        / "artifacts"
        / "checkpoint5"
        / "dynamic_scan_model.pt",
        map_location=device,
        weights_only=False,
    )

    model = ckpt5.DynamicFusionModel(
        scan_dim=checkpoint[
            "scan_dim"
        ],
        context_dim=checkpoint[
            "context_dim"
        ],
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    return (
        ckpt5,
        checkpoint,
        model,
    )


###############################################################################
# Prepared data
###############################################################################


def load_prepared(
    repo: Path,
) -> dict[
    str,
    Any,
]:

    prepared = (
        repo
        / "artifacts"
        / "checkpoint5"
        / "prepared"
    )

    index = pd.read_parquet(
        prepared
        / "scan_index.parquet"
    ).reset_index(
        drop=True
    )

    temporal = np.load(
        prepared
        / "temporal_prepost_f16.npy",
        mmap_mode="r",
    )

    tumor = np.load(
        prepared
        / "tumor_embeddings_f16.npy",
        mmap_mode="r",
    )

    scan = np.load(
        prepared
        / "current_scan_features_f32.npy",
        mmap_mode="r",
    )

    context = np.load(
        prepared
        / "context_features_f32.npy",
        mmap_mode="r",
    )

    schema = json.loads(
        (
            prepared
            / "feature_schema.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if len(
        index
    ) != temporal.shape[
        0
    ]:

        raise RuntimeError(
            "CKPT5 index/temporal array mismatch."
        )

    if len(
        index
    ) != tumor.shape[
        0
    ]:

        raise RuntimeError(
            "CKPT5 index/tumor array mismatch."
        )

    if len(
        index
    ) != scan.shape[
        0
    ]:

        raise RuntimeError(
            "CKPT5 index/scan array mismatch."
        )

    if len(
        index
    ) != context.shape[
        0
    ]:

        raise RuntimeError(
            "CKPT5 index/context array mismatch."
        )

    if "survival_mask" not in index.columns:

        raise RuntimeError(
            "CKPT5 scan index lacks survival_mask."
        )

    return {
        "prepared":
            prepared,

        "index":
            index,

        "temporal":
            temporal,

        "tumor":
            tumor,

        "scan":
            scan,

        "context":
            context,

        "schema":
            schema,
    }


###############################################################################
# Model variants
###############################################################################


VARIANTS = (
    "PRE",
    "POST",
    "PRE_PLUS_SCAN",
    "POST_NO_SCAN",
    "PRE_NO_GENOMICS",
    "POST_NO_GENOMICS",
)


def genomic_context_columns(
    schema: dict[
        str,
        Any,
    ],
) -> list[int]:

    names = schema.get(
        "context",
        []
    )

    return [
        index
        for index, name
        in enumerate(
            names
        )
        if "genomic"
        in norm(
            name
        )
    ]


@torch.no_grad()
def predict_variant(
    ckpt5,
    model,
    data: dict[
        str,
        Any,
    ],
    split: str,
    variant: str,
    device: torch.device,
) -> dict[
    str,
    Any,
]:

    index = data[
        "index"
    ]

    selected = (
        (
            index[
                "split"
            ]
            == split
        )
        & index[
            "survival_mask"
        ].astype(
            bool
        )
    )

    rows = np.where(
        selected.to_numpy()
    )[
        0
    ]

    frame = (
        index.iloc[
            rows
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        frame
    ) == 0:

        raise RuntimeError(
            f"No survival rows for split={split}"
        )

    temporal_array = data[
        "temporal"
    ]

    tumor_array = data[
        "tumor"
    ]

    scan_array = data[
        "scan"
    ]

    context_array = data[
        "context"
    ]

    genomic_context_indices = (
        genomic_context_columns(
            data[
                "schema"
            ]
        )
    )

    output_logits = np.zeros(
        (
            len(
                rows
            ),
            MONTHS,
            4,
        ),
        dtype=np.float32,
    )

    output_nll = np.zeros(
        len(
            rows
        ),
        dtype=np.float64,
    )

    batch_size = 1024

    for start in range(
        0,
        len(
            rows
        ),
        batch_size,
    ):

        stop = min(
            start
            + batch_size,
            len(
                rows
            ),
        )

        local_rows = rows[
            start:
            stop
        ]

        if variant in {
            "PRE",
            "PRE_PLUS_SCAN",
            "PRE_NO_GENOMICS",
        }:

            temporal_view = 0

        else:

            temporal_view = 1

        temporal = torch.from_numpy(
            np.asarray(
                temporal_array[
                    local_rows,
                    temporal_view,
                ],
                dtype=np.float32,
            ).copy()
        ).to(
            device
        )

        tumor = np.asarray(
            tumor_array[
                local_rows
            ],
            dtype=np.float32,
        ).copy()

        context = np.asarray(
            context_array[
                local_rows
            ],
            dtype=np.float32,
        ).copy()

        if variant in {
            "PRE",
            "POST_NO_SCAN",
            "PRE_NO_GENOMICS",
        }:

            scan = np.zeros(
                (
                    len(
                        local_rows
                    ),
                    scan_array.shape[
                        1
                    ],
                ),
                dtype=np.float32,
            )

        else:

            scan = np.asarray(
                scan_array[
                    local_rows
                ],
                dtype=np.float32,
            ).copy()

        if variant in {
            "PRE_NO_GENOMICS",
            "POST_NO_GENOMICS",
        }:

            tumor[
                :
            ] = 0.0

            for column in genomic_context_indices:

                context[
                    :,
                    column,
                ] = 0.0

        output = model(
            temporal,
            torch.from_numpy(
                tumor
            ).to(
                device
            ),
            torch.from_numpy(
                scan
            ).to(
                device
            ),
            torch.from_numpy(
                context
            ).to(
                device
            ),
        )

        logits = (
            output[
                "survival_logits"
            ]
            .float()
        )

        output_logits[
            start:
            stop
        ] = (
            logits
            .cpu()
            .numpy()
        )

        loss = (
            ckpt5.competing_risk_nll_per_row(
                logits,
                torch.tensor(
                    frame.iloc[
                        start:
                        stop
                    ][
                        "survival_time_days"
                    ]
                    .to_numpy(
                        dtype=np.float32
                    ),
                    device=device,
                ),
                torch.tensor(
                    frame.iloc[
                        start:
                        stop
                    ][
                        "survival_cause"
                    ]
                    .astype(
                        int
                    )
                    .to_numpy(),
                    device=device,
                ),
            )
        )

        output_nll[
            start:
            stop
        ] = (
            loss
            .float()
            .cpu()
            .numpy()
        )

    curves = curves_from_logits(
        output_logits
    )

    return {
        "frame":
            frame,

        "rows":
            rows,

        "logits":
            output_logits,

        "nll":
            output_nll,

        "curves":
            curves,
    }


def prediction_from_logits(
    ckpt5,
    frame: pd.DataFrame,
    logits: np.ndarray,
) -> dict[
    str,
    Any,
]:

    tensor = torch.from_numpy(
        np.asarray(
            logits,
            dtype=np.float32,
        )
    )

    nll = (
        ckpt5.competing_risk_nll_per_row(
            tensor,
            torch.tensor(
                frame[
                    "survival_time_days"
                ]
                .to_numpy(
                    dtype=np.float32
                )
            ),
            torch.tensor(
                frame[
                    "survival_cause"
                ]
                .astype(
                    int
                )
                .to_numpy()
            ),
        )
        .detach()
        .cpu()
        .numpy()
        .astype(
            float
        )
    )

    return {
        "frame":
            frame,

        "logits":
            np.asarray(
                logits,
                dtype=np.float32,
            ),

        "nll":
            nll,

        "curves":
            curves_from_logits(
                logits
            ),
    }


###############################################################################
# Metrics
###############################################################################


def patient_mean_nll(
    frame: pd.DataFrame,
    nll: np.ndarray,
) -> float:

    temporary = pd.DataFrame(
        {
            "patient_id":
                frame[
                    "patient_id"
                ]
                .astype(str)
                .to_numpy(),

            "nll":
                np.asarray(
                    nll,
                    dtype=float,
                ),
        }
    )

    return float(
        temporary.groupby(
            "patient_id",
            observed=True,
        )[
            "nll"
        ]
        .mean()
        .mean()
    )


def metric_bundle(
    prediction: dict[
        str,
        Any,
    ],
    censoring_km,
) -> dict[
    str,
    Any,
]:

    frame = prediction[
        "frame"
    ]

    nll = prediction[
        "nll"
    ]

    curves = prediction[
        "curves"
    ]

    result = {
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
                ].nunique()
            ),

        "row_mean_nll":
            float(
                np.mean(
                    nll
                )
            ),

        "patient_mean_nll":
            patient_mean_nll(
                frame,
                nll,
            ),

        "mean_entropy":
            float(
                np.mean(
                    curves[
                        "entropy"
                    ]
                )
            ),

        "row_weighted_pfs":
            {},

        "patient_balanced_pfs":
            {},
    }

    row_briers = []
    patient_briers = []

    for horizon in HORIZONS_MONTHS:

        survival = curves[
            "pfs"
        ][
            :,
            horizon
            - 1,
        ]

        row_metric = horizon_metrics(
            frame,
            survival,
            horizon,
            censoring_km,
            patient_balanced=False,
        )

        patient_metric = horizon_metrics(
            frame,
            survival,
            horizon,
            censoring_km,
            patient_balanced=True,
        )

        result[
            "row_weighted_pfs"
        ][
            f"{horizon}m"
        ] = row_metric

        result[
            "patient_balanced_pfs"
        ][
            f"{horizon}m"
        ] = patient_metric

        if np.isfinite(
            row_metric.get(
                "brier",
                np.nan,
            )
        ):

            row_briers.append(
                row_metric[
                    "brier"
                ]
            )

        if np.isfinite(
            patient_metric.get(
                "brier",
                np.nan,
            )
        ):

            patient_briers.append(
                patient_metric[
                    "brier"
                ]
            )

    result[
        "row_integrated_brier_4h"
    ] = (
        float(
            np.mean(
                row_briers
            )
        )
        if row_briers
        else float(
            "nan"
        )
    )

    result[
        "patient_integrated_brier_4h"
    ] = (
        float(
            np.mean(
                patient_briers
            )
        )
        if patient_briers
        else float(
            "nan"
        )
    )

    return result


###############################################################################
# Bootstrap
###############################################################################


def bootstrap_nll_gain(
    frame: pd.DataFrame,
    earlier_nll: np.ndarray,
    later_nll: np.ndarray,
    repetitions: int = 1000,
) -> dict[
    str,
    Any,
]:

    temporary = pd.DataFrame(
        {
            "patient_id":
                frame[
                    "patient_id"
                ]
                .astype(str)
                .to_numpy(),

            "earlier":
                np.asarray(
                    earlier_nll,
                    dtype=float,
                ),

            "later":
                np.asarray(
                    later_nll,
                    dtype=float,
                ),
        }
    )

    patient = (
        temporary.groupby(
            "patient_id",
            observed=True,
        )[
            [
                "earlier",
                "later",
            ]
        ]
        .mean()
        .dropna()
    )

    delta = (
        patient[
            "earlier"
        ]
        - patient[
            "later"
        ]
    ).to_numpy(
        dtype=float
    )

    rng = np.random.default_rng(
        SEED
    )

    bootstrap = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(
        repetitions
    ):

        bootstrap[
            index
        ] = (
            rng.choice(
                delta,
                size=len(
                    delta
                ),
                replace=True,
            )
            .mean()
        )

    return {
        "mean":
            float(
                delta.mean()
            ),

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
                    delta
                )
            ),
    }


###############################################################################
# Stratified diagnostics
###############################################################################


def strata_label_series(
    frame: pd.DataFrame,
    mode: str,
) -> pd.Series:

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

    if mode == "genomics":

        return np.where(
            frame[
                "genomic_available"
            ]
            .to_numpy(
                dtype=float
            )
            > 0,
            "AVAILABLE",
            "UNAVAILABLE",
        )

    if mode == "patient_scan_number":

        value = frame[
            "scan_number_patient"
        ].to_numpy(
            dtype=int
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

        value = frame[
            "scan_number_line"
        ].to_numpy(
            dtype=int
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

        value = frame[
            "line"
        ].to_numpy(
            dtype=int
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

    if mode == "genomic_age":

        available = (
            frame[
                "genomic_available"
            ]
            .to_numpy(
                dtype=float
            )
            > 0
        )

        age = (
            frame[
                "genomic_age_days"
            ]
            .fillna(
                -1
            )
            .to_numpy(
                dtype=float
            )
        )

        label = np.full(
            len(
                frame
            ),
            "UNAVAILABLE",
            dtype=object,
        )

        label[
            available
            & (
                age
                <= 180
            )
        ] = "AVAILABLE_1_TO_180D"

        label[
            available
            & (
                age
                > 180
            )
            & (
                age
                <= 730
            )
        ] = "AVAILABLE_181_TO_730D"

        label[
            available
            & (
                age
                > 730
            )
        ] = "AVAILABLE_GT_730D"

        return pd.Series(
            label,
            index=frame.index,
        )

    raise ValueError(
        mode
    )


def stratified_prepost(
    pre: dict[
        str,
        Any,
    ],
    post: dict[
        str,
        Any,
    ],
    censoring_km,
) -> dict[
    str,
    Any,
]:

    frame = pre[
        "frame"
    ]

    if not frame[
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

    result = {}

    for mode in (
        "future_target_type",
        "genomics",
        "patient_scan_number",
        "line_scan_number",
        "treatment_line",
        "genomic_age",
    ):

        label = strata_label_series(
            frame,
            mode,
        )

        mode_result = {}

        for value in sorted(
            pd.Series(
                label
            )
            .astype(str)
            .unique()
        ):

            mask = (
                pd.Series(
                    label
                )
                .astype(str)
                .to_numpy()
                == value
            )

            if mask.sum() < 30:
                continue

            subframe = (
                frame.loc[
                    mask
                ]
                .reset_index(
                    drop=True
                )
            )

            pre_nll = pre[
                "nll"
            ][
                mask
            ]

            post_nll = post[
                "nll"
            ][
                mask
            ]

            pre_prediction = {
                "frame":
                    subframe,

                "nll":
                    pre_nll,

                "curves": {
                    key:
                        value_array[
                            mask
                        ]
                    for key, value_array
                    in pre[
                        "curves"
                    ].items()
                },
            }

            post_prediction = {
                "frame":
                    subframe,

                "nll":
                    post_nll,

                "curves": {
                    key:
                        value_array[
                            mask
                        ]
                    for key, value_array
                    in post[
                        "curves"
                    ].items()
                },
            }

            pre_metric = metric_bundle(
                pre_prediction,
                censoring_km,
            )

            post_metric = metric_bundle(
                post_prediction,
                censoring_km,
            )

            pre_risk_12 = (
                1.0
                - pre_prediction[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    11
                ]
            )

            post_risk_12 = (
                1.0
                - post_prediction[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    11
                ]
            )

            mode_result[
                value
            ] = {
                "rows":
                    int(
                        mask.sum()
                    ),

                "patients":
                    int(
                        subframe[
                            "patient_id"
                        ].nunique()
                    ),

                "pre_row_nll":
                    pre_metric[
                        "row_mean_nll"
                    ],

                "post_row_nll":
                    post_metric[
                        "row_mean_nll"
                    ],

                "pre_minus_post_row_nll":
                    (
                        pre_metric[
                            "row_mean_nll"
                        ]
                        - post_metric[
                            "row_mean_nll"
                        ]
                    ),

                "pre_patient_nll":
                    pre_metric[
                        "patient_mean_nll"
                    ],

                "post_patient_nll":
                    post_metric[
                        "patient_mean_nll"
                    ],

                "pre_minus_post_patient_nll":
                    (
                        pre_metric[
                            "patient_mean_nll"
                        ]
                        - post_metric[
                            "patient_mean_nll"
                        ]
                    ),

                "pre_patient_ibs":
                    pre_metric[
                        "patient_integrated_brier_4h"
                    ],

                "post_patient_ibs":
                    post_metric[
                        "patient_integrated_brier_4h"
                    ],

                "pre_minus_post_patient_ibs":
                    (
                        pre_metric[
                            "patient_integrated_brier_4h"
                        ]
                        - post_metric[
                            "patient_integrated_brier_4h"
                        ]
                    ),

                "mean_post_minus_pre_12m_risk":
                    float(
                        np.mean(
                            post_risk_12
                            - pre_risk_12
                        )
                    ),

                "mean_post_minus_pre_entropy":
                    float(
                        np.mean(
                            post_prediction[
                                "curves"
                            ][
                                "entropy"
                            ]
                            - pre_prediction[
                                "curves"
                            ][
                                "entropy"
                            ]
                        )
                    ),
            }

        result[
            mode
        ] = mode_result

    return result


###############################################################################
# Prediction dataframe
###############################################################################


def prediction_dataframe(
    variant_predictions: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
) -> pd.DataFrame:

    first = next(
        iter(
            variant_predictions.values()
        )
    )

    frame = first[
        "frame"
    ]

    columns = [
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
            "genomic_available",
            "genomic_age_days",
            "scan_number_patient",
            "scan_number_line",
        )
        if column in frame.columns
    ]

    output = frame[
        columns
    ].copy()

    for name, prediction in variant_predictions.items():

        slug = name.lower()

        output[
            f"{slug}_nll"
        ] = prediction[
            "nll"
        ]

        output[
            f"{slug}_entropy"
        ] = prediction[
            "curves"
        ][
            "entropy"
        ]

        for horizon in HORIZONS_MONTHS:

            output[
                f"{slug}_pfs_{horizon}m"
            ] = (
                prediction[
                    "curves"
                ][
                    "pfs"
                ][
                    :,
                    horizon
                    - 1
                ]
            )

            output[
                f"{slug}_progression_cif_{horizon}m"
            ] = (
                prediction[
                    "curves"
                ][
                    "progression_cif"
                ][
                    :,
                    horizon
                    - 1
                ]
            )

            output[
                f"{slug}_death_cif_{horizon}m"
            ] = (
                prediction[
                    "curves"
                ][
                    "death_cif"
                ][
                    :,
                    horizon
                    - 1
                ]
            )

            output[
                f"{slug}_switch_cif_{horizon}m"
            ] = (
                prediction[
                    "curves"
                ][
                    "switch_cif"
                ][
                    :,
                    horizon
                    - 1
                ]
            )

    return output


###############################################################################
# Stale CKPT2 comparison
###############################################################################


def metric_bundle_survival_only(
    frame: pd.DataFrame,
    survival_by_horizon: dict[
        int,
        np.ndarray,
    ],
    censoring_km,
) -> dict[
    str,
    Any,
]:

    result = {
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
                ].nunique()
            ),

        "row_weighted_pfs":
            {},

        "patient_balanced_pfs":
            {},
    }

    row_brier = []
    patient_brier = []

    for horizon in HORIZONS_MONTHS:

        survival = np.asarray(
            survival_by_horizon[
                horizon
            ],
            dtype=float,
        )

        row_metric = horizon_metrics(
            frame,
            survival,
            horizon,
            censoring_km,
            patient_balanced=False,
        )

        patient_metric = horizon_metrics(
            frame,
            survival,
            horizon,
            censoring_km,
            patient_balanced=True,
        )

        result[
            "row_weighted_pfs"
        ][
            f"{horizon}m"
        ] = row_metric

        result[
            "patient_balanced_pfs"
        ][
            f"{horizon}m"
        ] = patient_metric

        if np.isfinite(
            row_metric.get(
                "brier",
                np.nan,
            )
        ):

            row_brier.append(
                row_metric[
                    "brier"
                ]
            )

        if np.isfinite(
            patient_metric.get(
                "brier",
                np.nan,
            )
        ):

            patient_brier.append(
                patient_metric[
                    "brier"
                ]
            )

    result[
        "row_integrated_brier_4h"
    ] = float(
        np.mean(
            row_brier
        )
    )

    result[
        "patient_integrated_brier_4h"
    ] = float(
        np.mean(
            patient_brier
        )
    )

    return result


def stale_comparison(
    repo: Path,
    test_predictions: pd.DataFrame,
    censoring_km,
) -> dict[
    str,
    Any,
]:

    stale_path = (
        repo
        / "artifacts"
        / "checkpoint2"
        / "scan_test_predictions.parquet"
    )

    stale = pd.read_parquet(
        stale_path
    )

    stale_patient = find_col(
        stale.columns,
        [
            "patient_id",
        ],
    )

    stale_line = find_col(
        stale.columns,
        [
            "line",
        ],
    )

    stale_day = find_col(
        stale.columns,
        [
            "landmark_day",
        ],
    )

    stale = stale.copy()

    stale[
        "patient_id"
    ] = (
        stale[
            stale_patient
        ]
        .astype(str)
        .str.strip()
    )

    stale[
        "line"
    ] = pd.to_numeric(
        stale[
            stale_line
        ],
        errors="raise",
    ).astype(
        int
    )

    stale[
        "_day_key"
    ] = np.round(
        pd.to_numeric(
            stale[
                stale_day
            ],
            errors="raise",
        ).astype(
            float
        ),
        6,
    )

    current = test_predictions.copy()

    current[
        "_day_key"
    ] = np.round(
        current[
            "landmark_day"
        ].astype(
            float
        ),
        6,
    )

    key = [
        "patient_id",
        "line",
        "_day_key",
    ]

    if stale[
        key
    ].duplicated().any():

        raise RuntimeError(
            "CKPT2 stale comparison key is not unique."
        )

    if current[
        key
    ].duplicated().any():

        raise RuntimeError(
            "CKPT5 comparison key is not unique."
        )

    stale_columns = []

    for horizon in HORIZONS_MONTHS:

        candidates = [
            f"stale_survival_{horizon}m",
            f"stale_pfs_{horizon}m",
        ]

        resolved = find_col(
            stale.columns,
            candidates,
            required=False,
        )

        if resolved is None:

            raise RuntimeError(
                "CKPT2 stale prediction missing "
                f"{horizon}m horizon.\n"
                f"columns={list(stale.columns)}"
            )

        stale_columns.append(
            resolved
        )

    rename = {
        stale_columns[
            index
        ]:
            f"stale_pfs_{horizon}m"

        for index, horizon
        in enumerate(
            HORIZONS_MONTHS
        )
    }

    stale_subset = (
        stale[
            key
            + stale_columns
        ]
        .rename(
            columns=rename
        )
    )

    joined = current.merge(
        stale_subset,
        on=key,
        how="inner",
        validate="one_to_one",
    )

    if len(
        joined
    ) < 500:

        raise RuntimeError(
            "Unexpectedly small CKPT2/CKPT5 stale overlap: "
            f"{len(joined)}"
        )

    frame_columns = [
        "patient_id",
        "scan_episode_id",
        "landmark_day",
        "line",
        "survival_time_days",
        "survival_cause",
        "target_type",
        "genomic_available",
        "genomic_age_days",
        "scan_number_patient",
        "scan_number_line",
    ]

    frame_columns = [
        column
        for column
        in frame_columns
        if column
        in joined.columns
    ]

    frame = joined[
        frame_columns
    ].copy()

    stale_survival = {
        horizon:
            joined[
                f"stale_pfs_{horizon}m"
            ]
            .to_numpy(
                dtype=float
            )

        for horizon in HORIZONS_MONTHS
    }

    pre_survival = {
        horizon:
            joined[
                f"pre_pfs_{horizon}m"
            ]
            .to_numpy(
                dtype=float
            )

        for horizon in HORIZONS_MONTHS
    }

    post_survival = {
        horizon:
            joined[
                f"post_pfs_{horizon}m"
            ]
            .to_numpy(
                dtype=float
            )

        for horizon in HORIZONS_MONTHS
    }

    result = {
        "overlap_rows":
            int(
                len(
                    joined
                )
            ),

        "overlap_patients":
            int(
                joined[
                    "patient_id"
                ].nunique()
            ),

        "fraction_of_ckpt5_test_survival_rows":
            float(
                len(
                    joined
                )
                / max(
                    len(
                        current
                    ),
                    1,
                )
            ),

        "stale":
            metric_bundle_survival_only(
                frame,
                stale_survival,
                censoring_km,
            ),

        "pre":
            metric_bundle_survival_only(
                frame,
                pre_survival,
                censoring_km,
            ),

        "post":
            metric_bundle_survival_only(
                frame,
                post_survival,
                censoring_km,
            ),

        "important_note":
            (
                "All three forecasts are evaluated against the same "
                "CKPT5 primary residual-PFS labels and scan time origin. "
                "The stale forecast itself comes from CKPT2's frozen "
                "line-start model and therefore exists only on the exact "
                "CKPT2/CKPT5 overlap subset."
            ),
    }

    return result


###############################################################################
# Self test
###############################################################################


def self_test() -> None:

    frame = pd.DataFrame(
        {
            "patient_id":
                [
                    "a",
                    "a",
                    "b",
                    "c",
                ],

            "survival_time_days":
                [
                    50.0,
                    300.0,
                    100.0,
                    500.0,
                ],

            "survival_cause":
                [
                    CAUSE_PROGRESSION,
                    CAUSE_CENSOR,
                    CAUSE_SWITCH,
                    CAUSE_DEATH,
                ],
        }
    )

    km = fit_censoring_km(
        frame[
            "survival_time_days"
        ].to_numpy(),
        frame[
            "survival_cause"
        ].to_numpy(),
    )

    (
        y,
        weight,
    ) = horizon_status_and_weight(
        frame,
        180.0,
        km,
        patient_balanced=False,
    )

    # a: progression before horizon -> case
    assert y[
        0
    ] == 1

    assert weight[
        0
    ] > 0

    # a: censor after horizon -> known noncase at horizon
    assert y[
        1
    ] == 0

    assert weight[
        1
    ] > 0

    # b: switch before horizon -> observed competing noncase
    assert y[
        2
    ] == 0

    assert weight[
        2
    ] > 0

    logits = np.zeros(
        (
            4,
            MONTHS,
            4,
        ),
        dtype=np.float32,
    )

    curves = curves_from_logits(
        logits
    )

    assert curves[
        "pfs"
    ].shape == (
        4,
        MONTHS,
    )

    assert bool(
        np.all(
            curves[
                "pfs"
            ][
                :,
                1:
            ]
            <= curves[
                "pfs"
            ][
                :,
                :-1
            ]
            + 1e-8
        )
    )

    # Calibration deliberately requires >=20 usable observations.
    # Use a non-perfectly-separated 40-observation synthetic fixture.
    calibration_prediction = np.tile(
        np.asarray(
            [
                0.10,
                0.20,
                0.30,
                0.40,
                0.60,
                0.70,
                0.80,
                0.90,
            ],
            dtype=float,
        ),
        5,
    )

    calibration_target = np.tile(
        np.asarray(
            [
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
            ],
            dtype=float,
        ),
        5,
    )

    calibration = (
        weighted_logistic_calibration(
            calibration_prediction,
            calibration_target,
            np.ones(
                len(
                    calibration_prediction
                ),
                dtype=float,
            ),
        )
    )

    assert np.isfinite(
        calibration[
            "intercept"
        ]
    )

    assert np.isfinite(
        calibration[
            "slope"
        ]
    )

    print(
        "[CKPT6A_SELF_TEST_PASS]"
    )


###############################################################################
# Main diagnostic pass
###############################################################################


def run(
    repo: Path,
    out: Path,
) -> None:

    np.random.seed(
        SEED
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "[CKPT6A] device=",
        device,
        flush=True,
    )

    data = load_prepared(
        repo
    )

    index = data[
        "index"
    ]

    ###########################################################################
    # Persistent split/leakage checks.
    ###########################################################################

    split_sets = {
        split:
            set(
                group[
                    "patient_id"
                ]
            )

        for split, group
        in index.groupby(
            "split"
        )
    }

    if (
        split_sets.get(
            "train",
            set(),
        )
        & split_sets.get(
            "val",
            set(),
        )
    ):

        raise RuntimeError(
            "train/val patient overlap"
        )

    if (
        split_sets.get(
            "train",
            set(),
        )
        & split_sets.get(
            "test",
            set(),
        )
    ):

        raise RuntimeError(
            "train/test patient overlap"
        )

    if (
        split_sets.get(
            "val",
            set(),
        )
        & split_sets.get(
            "test",
            set(),
        )
    ):

        raise RuntimeError(
            "val/test patient overlap"
        )

    progressive_primary = (
        index[
            "survival_mask"
        ].astype(
            bool
        )
        & (
            index[
                "scan_state"
            ]
            .astype(str)
            .str.upper()
            == "PROGRESSIVE"
        )
    )

    if progressive_primary.any():

        raise RuntimeError(
            "Progressive scan leaked into primary residual-PFS landmark set."
        )

    if (
        "genomic_available"
        in index.columns
        and "genomic_age_days"
        in index.columns
    ):

        genomic_leak = (
            (
                index[
                    "genomic_available"
                ]
                > 0
            )
            & (
                index[
                    "genomic_age_days"
                ]
                <= 0
            )
        )

        if genomic_leak.any():

            raise RuntimeError(
                "Same-day/future genomic observation detected."
            )

    ###########################################################################
    # Training-set censoring distribution.
    ###########################################################################

    train_survival = index[
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

    censoring_km = fit_censoring_km(
        train_survival[
            "survival_time_days"
        ].to_numpy(
            dtype=float
        ),
        train_survival[
            "survival_cause"
        ]
        .astype(
            int
        )
        .to_numpy(),
    )

    ###########################################################################
    # Frozen model predictions.
    ###########################################################################

    (
        ckpt5,
        checkpoint,
        model,
    ) = load_model(
        repo,
        device,
    )

    predictions = {
        "val": {},
        "test": {},
    }

    metrics = {
        "val": {},
        "test": {},
    }

    for split in (
        "val",
        "test",
    ):

        for variant in VARIANTS:

            print(
                "[CKPT6A] predicting",
                {
                    "split":
                        split,

                    "variant":
                        variant,
                },
                flush=True,
            )

            prediction = predict_variant(
                ckpt5,
                model,
                data,
                split,
                variant,
                device,
            )

            predictions[
                split
            ][
                variant
            ] = prediction

            metrics[
                split
            ][
                variant
            ] = metric_bundle(
                prediction,
                censoring_km,
            )

    ###########################################################################
    # PRE -> POST patient bootstrap.
    ###########################################################################

    paired_bootstrap = {}

    for split in (
        "val",
        "test",
    ):

        paired_bootstrap[
            split
        ] = bootstrap_nll_gain(
            predictions[
                split
            ][
                "PRE"
            ][
                "frame"
            ],
            predictions[
                split
            ][
                "PRE"
            ][
                "nll"
            ],
            predictions[
                split
            ][
                "POST"
            ][
                "nll"
            ],
        )

    ###########################################################################
    # Additive logit-update shrinkage.
    #
    # IMPORTANT:
    # alpha is selected from VALIDATION ONLY.
    ###########################################################################

    alpha_grid = [
        0.0,
        0.125,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
    ]

    alpha_validation = {}

    val_pre = predictions[
        "val"
    ][
        "PRE"
    ]

    val_post = predictions[
        "val"
    ][
        "POST"
    ]

    for alpha in alpha_grid:

        mixed_logits = (
            val_pre[
                "logits"
            ]
            + alpha
            * (
                val_post[
                    "logits"
                ]
                - val_pre[
                    "logits"
                ]
            )
        )

        mixed = prediction_from_logits(
            ckpt5,
            val_pre[
                "frame"
            ],
            mixed_logits,
        )

        alpha_validation[
            str(
                alpha
            )
        ] = metric_bundle(
            mixed,
            censoring_km,
        )

    finite_pfs_candidates = [
        alpha
        for alpha in alpha_grid
        if np.isfinite(
            alpha_validation[
                str(
                    alpha
                )
            ][
                "patient_integrated_brier_4h"
            ]
        )
    ]

    if not finite_pfs_candidates:

        raise RuntimeError(
            "No finite validation PFS Brier metric for alpha selection."
        )

    selected_alpha_pfs = min(
        finite_pfs_candidates,
        key=lambda alpha:
            alpha_validation[
                str(
                    alpha
                )
            ][
                "patient_integrated_brier_4h"
            ],
    )

    selected_alpha_nll = min(
        alpha_grid,
        key=lambda alpha:
            alpha_validation[
                str(
                    alpha
                )
            ][
                "patient_mean_nll"
            ],
    )

    test_pre = predictions[
        "test"
    ][
        "PRE"
    ]

    test_post = predictions[
        "test"
    ][
        "POST"
    ]

    alpha_test = {}

    for label, alpha in (
        (
            "validation_selected_pfs_alpha",
            selected_alpha_pfs,
        ),
        (
            "validation_selected_nll_alpha",
            selected_alpha_nll,
        ),
    ):

        mixed_logits = (
            test_pre[
                "logits"
            ]
            + alpha
            * (
                test_post[
                    "logits"
                ]
                - test_pre[
                    "logits"
                ]
            )
        )

        mixed = prediction_from_logits(
            ckpt5,
            test_pre[
                "frame"
            ],
            mixed_logits,
        )

        alpha_test[
            label
        ] = {
            "alpha":
                alpha,

            "metrics":
                metric_bundle(
                    mixed,
                    censoring_km,
                ),

            "pre_minus_mixed_patient_nll_gain":
                bootstrap_nll_gain(
                    test_pre[
                        "frame"
                    ],
                    test_pre[
                        "nll"
                    ],
                    mixed[
                        "nll"
                    ],
                ),

            "full_post_minus_mixed_patient_nll":
                bootstrap_nll_gain(
                    test_pre[
                        "frame"
                    ],
                    test_post[
                        "nll"
                    ],
                    mixed[
                        "nll"
                    ],
                ),
        }

    alpha_report = {
        "selection_population":
            "validation_only",

        "grid":
            alpha_grid,

        "definition":
            (
                "mixed_logits = PRE_logits + alpha * "
                "(POST_logits - PRE_logits)"
            ),

        "validation":
            alpha_validation,

        "selected_alpha_by_patient_pfs_ibs":
            selected_alpha_pfs,

        "selected_alpha_by_patient_nll":
            selected_alpha_nll,

        "test":
            alpha_test,

        "interpretation":
            (
                "alpha < 1 selected on validation supports the hypothesis "
                "that CKPT5's full scan update is too strong and should be "
                "implemented as a bounded/additive correction in CKPT6B."
            ),
    }

    atomic_json(
        out
        / "alpha_shrinkage.json",
        alpha_report,
    )

    ###########################################################################
    # Mechanistic component ablations.
    ###########################################################################

    mechanistic = {
        "important_note":
            (
                "PRE_PLUS_SCAN and POST_NO_SCAN are inference-time "
                "mechanistic perturbations. They were not paired exactly "
                "this way during CKPT5 training and therefore should not "
                "be treated as final candidate-model estimates."
            ),

        "val":
            {
                variant:
                    metrics[
                        "val"
                    ][
                        variant
                    ]

                for variant in VARIANTS
            },

        "test":
            {
                variant:
                    metrics[
                        "test"
                    ][
                        variant
                    ]

                for variant in VARIANTS
            },

        "paired_nll_gains": {},
    }

    for split in (
        "val",
        "test",
    ):

        frame = predictions[
            split
        ][
            "PRE"
        ][
            "frame"
        ]

        mechanistic[
            "paired_nll_gains"
        ][
            split
        ] = {
            "POST_vs_PRE":
                bootstrap_nll_gain(
                    frame,
                    predictions[
                        split
                    ][
                        "PRE"
                    ][
                        "nll"
                    ],
                    predictions[
                        split
                    ][
                        "POST"
                    ][
                        "nll"
                    ],
                ),

            "PRE_PLUS_SCAN_vs_PRE":
                bootstrap_nll_gain(
                    frame,
                    predictions[
                        split
                    ][
                        "PRE"
                    ][
                        "nll"
                    ],
                    predictions[
                        split
                    ][
                        "PRE_PLUS_SCAN"
                    ][
                        "nll"
                    ],
                ),

            "POST_NO_SCAN_vs_PRE":
                bootstrap_nll_gain(
                    frame,
                    predictions[
                        split
                    ][
                        "PRE"
                    ][
                        "nll"
                    ],
                    predictions[
                        split
                    ][
                        "POST_NO_SCAN"
                    ][
                        "nll"
                    ],
                ),

            "POST_vs_POST_NO_SCAN":
                bootstrap_nll_gain(
                    frame,
                    predictions[
                        split
                    ][
                        "POST_NO_SCAN"
                    ][
                        "nll"
                    ],
                    predictions[
                        split
                    ][
                        "POST"
                    ][
                        "nll"
                    ],
                ),

            "POST_NO_GENOMICS_vs_PRE_NO_GENOMICS":
                bootstrap_nll_gain(
                    frame,
                    predictions[
                        split
                    ][
                        "PRE_NO_GENOMICS"
                    ][
                        "nll"
                    ],
                    predictions[
                        split
                    ][
                        "POST_NO_GENOMICS"
                    ][
                        "nll"
                    ],
                ),
        }

    atomic_json(
        out
        / "mechanistic_ablations.json",
        mechanistic,
    )

    ###########################################################################
    # Stratified PRE/POST diagnostic.
    ###########################################################################

    stratified = {
        split:
            stratified_prepost(
                predictions[
                    split
                ][
                    "PRE"
                ],
                predictions[
                    split
                ][
                    "POST"
                ],
                censoring_km,
            )

        for split in (
            "val",
            "test",
        )
    }

    atomic_json(
        out
        / "stratified_metrics.json",
        stratified,
    )

    ###########################################################################
    # Prediction artifacts.
    ###########################################################################

    prediction_frames = {}

    for split in (
        "val",
        "test",
    ):

        frame = prediction_dataframe(
            predictions[
                split
            ]
        )

        prediction_frames[
            split
        ] = frame

        atomic_parquet(
            out
            / f"{split}_diagnostic_predictions.parquet",
            frame,
        )

    ###########################################################################
    # Stale CKPT2 comparison on exact test overlap.
    ###########################################################################

    stale = stale_comparison(
        repo,
        prediction_frames[
            "test"
        ],
        censoring_km,
    )

    atomic_json(
        out
        / "stale_pre_post_comparison.json",
        stale,
    )

    ###########################################################################
    # Validation-driven interpretation.
    ###########################################################################

    val_pre_metric = metrics[
        "val"
    ][
        "PRE"
    ]

    val_post_metric = metrics[
        "val"
    ][
        "POST"
    ]

    val_pre_plus_scan = metrics[
        "val"
    ][
        "PRE_PLUS_SCAN"
    ]

    val_post_no_scan = metrics[
        "val"
    ][
        "POST_NO_SCAN"
    ]

    val_pre_no_gen = metrics[
        "val"
    ][
        "PRE_NO_GENOMICS"
    ]

    val_post_no_gen = metrics[
        "val"
    ][
        "POST_NO_GENOMICS"
    ]

    bounded_update_signal = bool(
        selected_alpha_pfs
        < 0.999
        or selected_alpha_nll
        < 0.999
    )

    explicit_scan_direct_signal = bool(
        val_pre_plus_scan[
            "patient_integrated_brier_4h"
        ]
        < val_pre_metric[
            "patient_integrated_brier_4h"
        ]
    )

    post_temporal_signal = bool(
        val_post_no_scan[
            "patient_integrated_brier_4h"
        ]
        < val_pre_metric[
            "patient_integrated_brier_4h"
        ]
    )

    explicit_scan_double_count_signal = bool(
        val_post_no_scan[
            "patient_integrated_brier_4h"
        ]
        < val_post_metric[
            "patient_integrated_brier_4h"
        ]
    )

    full_scan_update_pfs_signal = bool(
        val_post_metric[
            "patient_integrated_brier_4h"
        ]
        < val_pre_metric[
            "patient_integrated_brier_4h"
        ]
    )

    no_genomics_update_delta = (
        val_pre_no_gen[
            "patient_integrated_brier_4h"
        ]
        - val_post_no_gen[
            "patient_integrated_brier_4h"
        ]
    )

    full_update_delta = (
        val_pre_metric[
            "patient_integrated_brier_4h"
        ]
        - val_post_metric[
            "patient_integrated_brier_4h"
        ]
    )

    priority = []

    if bounded_update_signal:

        priority.append(
            "Train explicit additive/bounded PRE->POST update architecture."
        )

    if explicit_scan_double_count_signal:

        priority.append(
            "Test removing the separate explicit-scan pathway when the "
            "POST temporal state already contains the current scan."
        )

    if (
        explicit_scan_direct_signal
        and not post_temporal_signal
    ):

        priority.append(
            "Prioritize PRE-state + direct scan-delta model over full "
            "POST temporal re-encoding."
        )

    elif (
        post_temporal_signal
        and not explicit_scan_direct_signal
    ):

        priority.append(
            "Prioritize temporal POST delta and reduce/remove explicit "
            "scan feature duplication."
        )

    else:

        priority.append(
            "Retain both temporal-delta and explicit-scan candidate arms "
            "for CKPT6B because validation does not cleanly isolate one."
        )

    priority.extend(
        [
            (
                "Train survival-only versus full multitask models to test "
                "negative transfer."
            ),
            (
                "Train a controlled no-genomics ablation on identical "
                "landmarks rather than interpreting genomic availability "
                "strata causally."
            ),
            (
                "Compare patient-balanced versus landmark-balanced "
                "supervised training."
            ),
        ]
    )

    decision = {
        "status":
            "DIAGNOSTIC_COMPLETE_RETRAIN_REQUIRED",

        "selection_policy":
            (
                "All CKPT6B architectural decisions below are driven by "
                "validation metrics. Test results are reported only as "
                "held-out diagnostic confirmation."
            ),

        "validation_signals": {
            "bounded_update_signal":
                bounded_update_signal,

            "selected_alpha_by_patient_pfs_ibs":
                selected_alpha_pfs,

            "selected_alpha_by_patient_nll":
                selected_alpha_nll,

            "full_scan_update_improves_patient_pfs_ibs":
                full_scan_update_pfs_signal,

            "pre_plus_scan_improves_patient_pfs_ibs":
                explicit_scan_direct_signal,

            "post_temporal_without_explicit_scan_improves_patient_pfs_ibs":
                post_temporal_signal,

            "post_without_explicit_scan_beats_full_post":
                explicit_scan_double_count_signal,

            "validation_full_pre_minus_post_patient_pfs_ibs":
                full_update_delta,

            "validation_no_genomics_pre_minus_post_patient_pfs_ibs":
                no_genomics_update_delta,
        },

        "ckpt6b_priority":
            priority,

        "deferred_until_candidate_freeze": [
            "W0/W3/W7 scan grouping robustness",
            "external BPC DFCI/VICC validation",
            "encoder unfreezing",
        ],
    }

    atomic_json(
        out
        / "decision_packet.json",
        decision,
    )

    ###########################################################################
    # Overall metrics summary.
    ###########################################################################

    metrics_summary = {
        "status":
            decision[
                "status"
            ],

        "model_checkpoint":
            "artifacts/checkpoint5/dynamic_scan_model.pt",

        "model_parameter_count":
            checkpoint.get(
                "parameter_count"
            ),

        "splits": {
            split: {
                variant:
                    metrics[
                        split
                    ][
                        variant
                    ]

                for variant in VARIANTS
            }

            for split in (
                "val",
                "test",
            )
        },

        "paired_pre_post_nll_bootstrap":
            paired_bootstrap,

        "alpha_shrinkage": {
            "selected_alpha_by_patient_pfs_ibs":
                selected_alpha_pfs,

            "selected_alpha_by_patient_nll":
                selected_alpha_nll,

            "test":
                alpha_test,
        },

        "stale_overlap":
            stale,

        "decision":
            decision,
    }

    atomic_json(
        out
        / "metrics_summary.json",
        metrics_summary,
    )

    ###########################################################################
    # Human-readable audit.
    ###########################################################################

    audit = f"""# Checkpoint 6A — Diagnose CKPT5 scan-update failure mode

Status: **DIAGNOSTIC_COMPLETE_RETRAIN_REQUIRED**

## Why this checkpoint exists

CKPT2 established incremental current-scan signal with a simpler gradient model.
CKPT5 changed the representation after the scan but did not establish a robust
aggregate residual-PFS likelihood gain.

CKPT6A performs no retraining. It localizes the failure before CKPT6B.

## Frozen evaluation policy

- CKPT1 endpoint and landmark rules unchanged.
- CKPT3 genomic encoder unchanged.
- CKPT4 temporal encoder unchanged.
- CKPT5 supervised checkpoint unchanged.
- DFCI/VICC outcomes untouched.
- Architectural decisions use validation metrics only.
- Test is diagnostic confirmation, not model-selection input.

## Primary PFS estimand

At each horizon, the target event is progression or death.

Treatment switch remains a separately observed competing event.

Censoring is handled with training-derived IPCW.

Metrics are reported both:
- landmark weighted
- patient balanced

Primary diagnostic metric for selecting update strength:
- patient-balanced integrated Brier score across 3/6/12/18 months

Full competing-risk NLL remains a secondary diagnostic.

## Validation PRE versus POST

PRE patient NLL:
{metrics['val']['PRE']['patient_mean_nll']}

POST patient NLL:
{metrics['val']['POST']['patient_mean_nll']}

PRE patient 4-horizon integrated Brier:
{metrics['val']['PRE']['patient_integrated_brier_4h']}

POST patient 4-horizon integrated Brier:
{metrics['val']['POST']['patient_integrated_brier_4h']}

## Test PRE versus POST

PRE patient NLL:
{metrics['test']['PRE']['patient_mean_nll']}

POST patient NLL:
{metrics['test']['POST']['patient_mean_nll']}

PRE patient 4-horizon integrated Brier:
{metrics['test']['PRE']['patient_integrated_brier_4h']}

POST patient 4-horizon integrated Brier:
{metrics['test']['POST']['patient_integrated_brier_4h']}

Patient-bootstrap PRE minus POST NLL:
{paired_bootstrap['test']}

## Bounded update diagnostic

Definition:

POST_alpha logits =
PRE logits + alpha * (POST logits - PRE logits)

Alpha is selected on validation only.

Selected alpha by patient-balanced PFS Brier:
{selected_alpha_pfs}

Selected alpha by patient-balanced NLL:
{selected_alpha_nll}

A selected alpha below 1 supports over-updating / over-sharpening as a
specific mechanism and motivates a bounded residual-update architecture.

## Mechanistic perturbations

PRE:
- PRE temporal state
- no current-scan explicit feature

POST:
- POST temporal state
- explicit current-scan feature

PRE_PLUS_SCAN:
- PRE temporal state
- explicit current-scan feature

POST_NO_SCAN:
- POST temporal state
- explicit current-scan pathway zeroed

PRE/POST_NO_GENOMICS:
- same landmarks and states
- tumor representation and genomic context zeroed

PRE_PLUS_SCAN and POST_NO_SCAN are mechanistic diagnostics, not final
candidate-model estimates, because CKPT5 was not trained on those exact
combinations.

## Stale comparison

Exact overlap rows:
{stale['overlap_rows']}

Exact overlap patients:
{stale['overlap_patients']}

Stale patient IBS:
{stale['stale']['patient_integrated_brier_4h']}

PRE patient IBS:
{stale['pre']['patient_integrated_brier_4h']}

POST patient IBS:
{stale['post']['patient_integrated_brier_4h']}

All three are evaluated against the same CKPT5 residual-PFS labels and scan
time origin on the overlap subset.

## Validation-driven CKPT6B priorities

{chr(10).join('- ' + item for item in priority)}

## Deliberately deferred

- W0/W3/W7 robustness until one candidate architecture is selected.
- DFCI/VICC external outcomes remain untouched.
- No encoder unfreezing before the supervised update mechanism is resolved.
"""

    atomic_text(
        out
        / "audit.md",
        audit,
    )

    ###########################################################################
    # PROJECT_STATE
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
        "<!-- CKPT6A_SCAN_UPDATE_DIAGNOSIS -->"
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
## Checkpoint 6A — Scan-update mechanism diagnosis

Status: **DIAGNOSTIC_COMPLETE_RETRAIN_REQUIRED**

CKPT5 remains frozen as:
`TRAINED_EVALUATED_PRIMARY_SCAN_GAIN_NOT_ESTABLISHED`

No upstream cohort, endpoint, encoder, or external-validation rule changed.

CKPT6A now makes the patient-facing progression/death PFS estimand explicit:
- progression + death are target PFS events
- switch is a separately observed competing event
- censoring uses training-derived IPCW
- primary diagnostic aggregation is patient balanced
- 3/6/12/18-month Brier, discrimination, and calibration are reported

Update-shrinkage rule tested without retraining:

`PRE_logits + alpha * (POST_logits - PRE_logits)`

Alpha was selected from validation only.

Selected validation alpha by patient PFS IBS:
**{selected_alpha_pfs}**

Selected validation alpha by patient NLL:
**{selected_alpha_nll}**

Validation signals:
`{decision['validation_signals']}`

CKPT6B should now retrain only the targeted candidate architectures identified
in `artifacts/checkpoint6a/decision_packet.json`.

DFCI/VICC outcomes remain untouched.

Artifacts:
- artifacts/checkpoint6a/metrics_summary.json
- artifacts/checkpoint6a/alpha_shrinkage.json
- artifacts/checkpoint6a/mechanistic_ablations.json
- artifacts/checkpoint6a/stratified_metrics.json
- artifacts/checkpoint6a/stale_pre_post_comparison.json
- artifacts/checkpoint6a/val_diagnostic_predictions.parquet
- artifacts/checkpoint6a/test_diagnostic_predictions.parquet
- artifacts/checkpoint6a/decision_packet.json
- artifacts/checkpoint6a/audit.md
- artifacts/handoff/checkpoint_06A.json
"""

    atomic_text(
        state_path,
        existing.rstrip()
        + "\n\n"
        + section.strip()
        + "\n",
    )

    ###########################################################################
    # Handoff
    ###########################################################################

    handoff = {
        "checkpoint":
            "6A",

        "name":
            "scan_update_mechanism_diagnosis",

        "status":
            decision[
                "status"
            ],

        "ckpt5_status":
            "TRAINED_EVALUATED_PRIMARY_SCAN_GAIN_NOT_ESTABLISHED",

        "selection_policy":
            decision[
                "selection_policy"
            ],

        "validation_signals":
            decision[
                "validation_signals"
            ],

        "test_pre_post": {
            "pre_patient_nll":
                metrics[
                    "test"
                ][
                    "PRE"
                ][
                    "patient_mean_nll"
                ],

            "post_patient_nll":
                metrics[
                    "test"
                ][
                    "POST"
                ][
                    "patient_mean_nll"
                ],

            "pre_patient_ibs":
                metrics[
                    "test"
                ][
                    "PRE"
                ][
                    "patient_integrated_brier_4h"
                ],

            "post_patient_ibs":
                metrics[
                    "test"
                ][
                    "POST"
                ][
                    "patient_integrated_brier_4h"
                ],

            "nll_bootstrap":
                paired_bootstrap[
                    "test"
                ],
        },

        "alpha_shrinkage": {
            "selected_alpha_by_patient_pfs_ibs":
                selected_alpha_pfs,

            "selected_alpha_by_patient_nll":
                selected_alpha_nll,

            "test":
                alpha_test,
        },

        "stale_overlap":
            stale,

        "ckpt6b_priority":
            priority,

        "external_validation":
            "DFCI/VICC outcomes remain untouched",

        "next_action":
            (
                "Use CKPT6A validation results to implement CKPT6B targeted "
                "retraining: bounded additive scan update, survival-only "
                "versus multitask, component pathway variants, controlled "
                "no-genomics training ablation, and patient-balancing "
                "sensitivity."
            ),
    }

    atomic_json(
        repo
        / "artifacts"
        / "handoff"
        / "checkpoint_06A.json",
        handoff,
    )

    ###########################################################################
    # Compact output
    ###########################################################################

    print("")
    print(
        "========== CKPT6A SUMMARY =========="
    )

    print(
        "status="
        f"{decision['status']}"
    )

    print(
        "validation_PRE_patient_NLL="
        f"{metrics['val']['PRE']['patient_mean_nll']}"
    )

    print(
        "validation_POST_patient_NLL="
        f"{metrics['val']['POST']['patient_mean_nll']}"
    )

    print(
        "validation_PRE_patient_IBS="
        f"{metrics['val']['PRE']['patient_integrated_brier_4h']}"
    )

    print(
        "validation_POST_patient_IBS="
        f"{metrics['val']['POST']['patient_integrated_brier_4h']}"
    )

    print(
        "test_PRE_patient_NLL="
        f"{metrics['test']['PRE']['patient_mean_nll']}"
    )

    print(
        "test_POST_patient_NLL="
        f"{metrics['test']['POST']['patient_mean_nll']}"
    )

    print(
        "test_PRE_patient_IBS="
        f"{metrics['test']['PRE']['patient_integrated_brier_4h']}"
    )

    print(
        "test_POST_patient_IBS="
        f"{metrics['test']['POST']['patient_integrated_brier_4h']}"
    )

    print(
        "test_PRE_minus_POST_NLL_bootstrap="
        f"{paired_bootstrap['test']}"
    )

    print(
        "selected_alpha_by_validation_patient_PFS_IBS="
        f"{selected_alpha_pfs}"
    )

    print(
        "selected_alpha_by_validation_patient_NLL="
        f"{selected_alpha_nll}"
    )

    print(
        "stale_overlap_rows="
        f"{stale['overlap_rows']}"
    )

    print(
        "stale_overlap_patients="
        f"{stale['overlap_patients']}"
    )

    print(
        "stale_overlap_patient_IBS="
        f"{stale['stale']['patient_integrated_brier_4h']}"
    )

    print(
        "pre_overlap_patient_IBS="
        f"{stale['pre']['patient_integrated_brier_4h']}"
    )

    print(
        "post_overlap_patient_IBS="
        f"{stale['post']['patient_integrated_brier_4h']}"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_06A.json"
    )

    print(
        "========== CKPT6A SUMMARY END =========="
    )

    print("")
    print(
        "========== CKPT6A DECISION PACKET =========="
    )

    print(
        "validation_signals="
        f"{decision['validation_signals']}"
    )

    print(
        "validation_PRE="
        f"{metrics['val']['PRE']}"
    )

    print(
        "validation_POST="
        f"{metrics['val']['POST']}"
    )

    print(
        "validation_PRE_PLUS_SCAN="
        f"{metrics['val']['PRE_PLUS_SCAN']}"
    )

    print(
        "validation_POST_NO_SCAN="
        f"{metrics['val']['POST_NO_SCAN']}"
    )

    print(
        "validation_PRE_NO_GENOMICS="
        f"{metrics['val']['PRE_NO_GENOMICS']}"
    )

    print(
        "validation_POST_NO_GENOMICS="
        f"{metrics['val']['POST_NO_GENOMICS']}"
    )

    print(
        "alpha_validation="
        f"{alpha_validation}"
    )

    print(
        "alpha_test="
        f"{alpha_test}"
    )

    print(
        "test_strata_future_target="
        f"{stratified['test']['future_target_type']}"
    )

    print(
        "test_strata_patient_scan_number="
        f"{stratified['test']['patient_scan_number']}"
    )

    print(
        "test_strata_line_scan_number="
        f"{stratified['test']['line_scan_number']}"
    )

    print(
        "test_strata_treatment_line="
        f"{stratified['test']['treatment_line']}"
    )

    print(
        "test_strata_genomics="
        f"{stratified['test']['genomics']}"
    )

    print(
        "ckpt6b_priority="
        f"{priority}"
    )

    print(
        "========== CKPT6A DECISION PACKET END =========="
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
        default="artifacts/checkpoint6a",
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
