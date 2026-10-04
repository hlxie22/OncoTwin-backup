#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ALPHA = 0.75
HORIZONS = [3, 6, 12, 18]
BOOTSTRAP_REPETITIONS = 2000
SEED = 20260927


###############################################################################
# IO
###############################################################################


def json_default(value: Any):

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    raise TypeError(
        type(value).__name__
    )


def atomic_json(
    path: Path,
    obj: Any,
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
            obj,
            indent=2,
            sort_keys=True,
            default=json_default,
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
        path.suffix
        + ".tmp"
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
        path.suffix
        + ".tmp"
    )

    frame.to_parquet(
        tmp,
        index=False,
    )

    if (
        not tmp.exists()
        or tmp.stat().st_size == 0
    ):
        raise RuntimeError(
            f"empty parquet write: {path}"
        )

    check = pd.read_parquet(
        tmp
    )

    if len(check) != len(frame):

        raise RuntimeError(
            f"parquet row verification failed: {path}"
        )

    tmp.replace(
        path
    )


def sha256(
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


def load_module(
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
            f"cannot import {path}"
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
# Keys
###############################################################################


def normalize_patient(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.strip()
    )


def integer_series(
    series: pd.Series,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="raise",
    )

    rounded = np.rint(
        values.to_numpy(
            dtype=float
        )
    )

    if not np.allclose(
        values.to_numpy(
            dtype=float
        ),
        rounded,
        atol=1e-8,
        rtol=0.0,
    ):

        raise RuntimeError(
            "expected integer-valued key"
        )

    return pd.Series(
        rounded.astype(
            np.int64
        ),
        index=series.index,
    )


def prediction_key_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    output = frame.copy()

    output[
        "patient_id"
    ] = normalize_patient(
        output[
            "patient_id"
        ]
    )

    output[
        "_key_day"
    ] = integer_series(
        output[
            "landmark_day"
        ]
    )

    output[
        "_key_line"
    ] = integer_series(
        output[
            "line"
        ]
    )

    return output


def alternate_key_frame(
    frame: pd.DataFrame,
    day_column: str,
    line_column: str,
) -> pd.DataFrame:

    output = frame.copy()

    output[
        "patient_id"
    ] = normalize_patient(
        output[
            "patient_id"
        ]
    )

    output[
        "_key_day"
    ] = integer_series(
        output[
            day_column
        ]
    )

    output[
        "_key_line"
    ] = integer_series(
        output[
            line_column
        ]
    )

    return output


def tuple_keys(
    frame: pd.DataFrame,
) -> list[
    tuple[str, int, int]
]:

    return list(
        zip(
            frame[
                "patient_id"
            ].astype(str),

            frame[
                "_key_day"
            ].astype(int),

            frame[
                "_key_line"
            ].astype(int),
        )
    )


###############################################################################
# CKPT5 current-scan feature schema adapter
###############################################################################


def adapt_ckpt5_scan_feature_schema(
    frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:

    output = frame.copy()

    ###########################################################################
    # CKPT1 canonical radiology uses progression_state_3.
    # CKPT5 scan_features() consumes the same semantic quantity under the
    # literal column name scan_state.
    ###########################################################################

    if "scan_state" not in output.columns:

        candidate = None

        for column in (
            "progression_state_3",
            "progression_state_3_scan",
        ):

            if column in output.columns:

                candidate = column
                break

        if candidate is None:

            raise RuntimeError(
                f"{label}: cannot provide CKPT5 scan_state; "
                f"columns={list(output.columns)}"
            )

        output[
            "scan_state"
        ] = output[
            candidate
        ]

    ###########################################################################
    # If both canonical and cache states are present, do not replace the
    # canonical state with the numeric cache code. The numeric cache state was
    # already independently checked against progression_state_3 upstream.
    ###########################################################################

    required = [
        "scan_state",
        "coverage_chest",
        "coverage_abdomen",
        "coverage_pelvis",
        "coverage_head",
        "coverage_other",
        "modalities_json",
        "tumor_sites_json",
    ]

    missing = [
        column
        for column in required
        if column not in output.columns
    ]

    if missing:

        raise RuntimeError(
            f"{label}: CKPT5 current-scan feature schema missing "
            f"{missing}; columns={list(output.columns)}"
        )

    if (
        output[
            "scan_state"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            f"{label}: scan_state contains missing values"
        )

    return output



###############################################################################
# Frozen model forward compatibility
###############################################################################


def model_forward(
    model,
    temporal: torch.Tensor,
    tumor: torch.Tensor,
    scan: torch.Tensor,
    context: torch.Tensor,
):

    signature = inspect.signature(
        model.forward
    )

    parameters = [
        parameter
        for parameter
        in signature.parameters.values()
        if parameter.name
        != "self"
    ]

    kwargs = {}

    unresolved = []

    for parameter in parameters:

        name = (
            parameter.name
            .lower()
        )

        if (
            "temporal"
            in name
            or "history"
            in name
        ):

            kwargs[
                parameter.name
            ] = temporal

        elif (
            "tumor"
            in name
            or "genomic"
            in name
        ):

            kwargs[
                parameter.name
            ] = tumor

        elif (
            "scan"
            in name
        ):

            kwargs[
                parameter.name
            ] = scan

        elif (
            "context"
            in name
            or "clinical"
            in name
        ):

            kwargs[
                parameter.name
            ] = context

        elif (
            parameter.default
            is not inspect._empty
        ):

            continue

        else:

            unresolved.append(
                parameter.name
            )

    if unresolved:

        raise RuntimeError(
            "Could not map DynamicFusionModel.forward arguments: "
            f"{unresolved}; signature={signature}"
        )

    return model(
        **kwargs
    )


def flatten_tensors(
    obj: Any,
    prefix: str = "",
) -> list[
    tuple[str, torch.Tensor]
]:

    output = []

    if torch.is_tensor(
        obj
    ):

        output.append(
            (
                prefix,
                obj,
            )
        )

        return output

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            output.extend(
                flatten_tensors(
                    value,
                    (
                        f"{prefix}.{key}"
                        if prefix
                        else str(
                            key
                        )
                    ),
                )
            )

        return output

    if isinstance(
        obj,
        (
            tuple,
            list,
        ),
    ):

        for index, value in enumerate(
            obj
        ):

            output.extend(
                flatten_tensors(
                    value,
                    (
                        f"{prefix}.{index}"
                        if prefix
                        else str(
                            index
                        )
                    ),
                )
            )

    return output


def extract_survival_logits(
    output: Any,
    months: int,
) -> torch.Tensor:

    tensors = flatten_tensors(
        output
    )

    preferred = []

    fallback = []

    for name, tensor in tensors:

        if (
            tensor.ndim == 3
            and tensor.shape[
                1
            ] == months
        ):

            fallback.append(
                (
                    name,
                    tensor,
                )
            )

            if (
                "survival"
                in name.lower()
                or "hazard"
                in name.lower()
            ):

                preferred.append(
                    (
                        name,
                        tensor,
                    )
                )

    candidates = (
        preferred
        if len(
            preferred
        ) == 1
        else fallback
    )

    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            "Could not uniquely identify competing-risk logits. "
            f"candidates="
            f"{[(name, tuple(t.shape)) for name, t in candidates]}"
        )

    return candidates[
        0
    ][
        1
    ]


def extract_pfs_survival(
    ckpt5,
    logits: torch.Tensor,
) -> tuple[
    torch.Tensor,
    str,
]:

    curves = ckpt5.probability_curves(
        logits
    )

    if not isinstance(
        curves,
        dict,
    ):

        raise RuntimeError(
            "probability_curves() did not return dict"
        )

    pfs_candidates = []

    for key, value in curves.items():

        if (
            torch.is_tensor(
                value
            )
            and value.ndim == 2
            and value.shape[
                0
            ] == logits.shape[
                0
            ]
            and value.shape[
                1
            ] == logits.shape[
                1
            ]
            and "pfs"
            in key.lower()
        ):

            pfs_candidates.append(
                (
                    key,
                    value,
                )
            )

    if len(
        pfs_candidates
    ) == 1:

        return (
            pfs_candidates[
                0
            ][
                1
            ],
            pfs_candidates[
                0
            ][
                0
            ],
        )

    progression = None
    death = None
    progression_key = None
    death_key = None

    for key, value in curves.items():

        if not (
            torch.is_tensor(
                value
            )
            and value.ndim == 2
            and value.shape[
                :2
            ]
            == logits.shape[
                :2
            ]
        ):

            continue

        lower = key.lower()

        if (
            "progress"
            in lower
            and (
                "cif"
                in lower
                or "cumulative"
                in lower
            )
        ):

            progression = value
            progression_key = key

        if (
            "death"
            in lower
            and (
                "cif"
                in lower
                or "cumulative"
                in lower
            )
        ):

            death = value
            death_key = key

    if (
        progression
        is None
        or death
        is None
    ):

        raise RuntimeError(
            "Could not recover PFS survival from probability_curves keys: "
            f"{list(curves.keys())}"
        )

    pfs = (
        1.0
        - progression
        - death
    ).clamp(
        min=0.0,
        max=1.0,
    )

    return (
        pfs,
        (
            "derived:"
            f"1-{progression_key}-{death_key}"
        ),
    )


###############################################################################
# Inference
###############################################################################


def infer(
    ckpt5,
    model,
    temporal_prepost: np.ndarray,
    tumor: np.ndarray,
    scan_features: np.ndarray,
    context: np.ndarray,
    time_days: np.ndarray,
    cause: np.ndarray,
    device: torch.device,
    months: int,
    batch_size: int = 1024,
) -> dict[str, Any]:

    count = len(
        time_days
    )

    if not (
        temporal_prepost.shape
        == (
            count,
            2,
            192,
        )
    ):

        raise RuntimeError(
            f"unexpected temporal shape "
            f"{temporal_prepost.shape}"
        )

    if tumor.shape != (
        count,
        128,
    ):

        raise RuntimeError(
            f"unexpected tumor shape "
            f"{tumor.shape}"
        )

    if scan_features.shape[
        0
    ] != count:

        raise RuntimeError(
            "scan feature row mismatch"
        )

    if context.shape[
        0
    ] != count:

        raise RuntimeError(
            "context row mismatch"
        )

    pre_nll_parts = []
    post_nll_parts = []
    bounded_nll_parts = []

    pre_survival_parts = []
    post_survival_parts = []
    bounded_survival_parts = []

    curve_source = None

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            count,
            batch_size,
        ):

            stop = min(
                start
                + batch_size,
                count,
            )

            temporal_np = np.asarray(
                temporal_prepost[
                    start:
                    stop
                ],
                dtype=np.float32,
            )

            tumor_t = torch.from_numpy(
                np.asarray(
                    tumor[
                        start:
                        stop
                    ],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            scan_t = torch.from_numpy(
                np.asarray(
                    scan_features[
                        start:
                        stop
                    ],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            context_t = torch.from_numpy(
                np.asarray(
                    context[
                        start:
                        stop
                    ],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            temporal_pre_t = torch.from_numpy(
                temporal_np[
                    :,
                    0,
                    :,
                ]
            ).to(
                device
            )

            temporal_post_t = torch.from_numpy(
                temporal_np[
                    :,
                    1,
                    :,
                ]
            ).to(
                device
            )

            # PRE contract:
            # before the current scan is revealed,
            # explicit current-scan features are zero.
            zero_scan_t = torch.zeros_like(
                scan_t
            )

            pre_output = model_forward(
                model,
                temporal_pre_t,
                tumor_t,
                zero_scan_t,
                context_t,
            )

            post_output = model_forward(
                model,
                temporal_post_t,
                tumor_t,
                scan_t,
                context_t,
            )

            pre_logits = extract_survival_logits(
                pre_output,
                months,
            )

            post_logits = extract_survival_logits(
                post_output,
                months,
            )

            bounded_logits = (
                pre_logits
                + ALPHA
                * (
                    post_logits
                    - pre_logits
                )
            )

            time_t = torch.from_numpy(
                np.asarray(
                    time_days[
                        start:
                        stop
                    ],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            cause_t = torch.from_numpy(
                np.asarray(
                    cause[
                        start:
                        stop
                    ],
                    dtype=np.int64,
                )
            ).to(
                device
            )

            pre_nll = (
                ckpt5.competing_risk_nll_per_row(
                    pre_logits,
                    time_t,
                    cause_t,
                )
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            post_nll = (
                ckpt5.competing_risk_nll_per_row(
                    post_logits,
                    time_t,
                    cause_t,
                )
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            bounded_nll = (
                ckpt5.competing_risk_nll_per_row(
                    bounded_logits,
                    time_t,
                    cause_t,
                )
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            (
                pre_pfs,
                source_pre,
            ) = extract_pfs_survival(
                ckpt5,
                pre_logits,
            )

            (
                post_pfs,
                source_post,
            ) = extract_pfs_survival(
                ckpt5,
                post_logits,
            )

            (
                bounded_pfs,
                source_bounded,
            ) = extract_pfs_survival(
                ckpt5,
                bounded_logits,
            )

            if curve_source is None:

                curve_source = {
                    "pre":
                        source_pre,

                    "post":
                        source_post,

                    "bounded":
                        source_bounded,
                }

            pre_nll_parts.append(
                pre_nll
            )

            post_nll_parts.append(
                post_nll
            )

            bounded_nll_parts.append(
                bounded_nll
            )

            pre_survival_parts.append(
                pre_pfs
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            post_survival_parts.append(
                post_pfs
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            bounded_survival_parts.append(
                bounded_pfs
                .detach()
                .float()
                .cpu()
                .numpy()
            )

    return {
        "pre_nll":
            np.concatenate(
                pre_nll_parts
            ),

        "post_nll":
            np.concatenate(
                post_nll_parts
            ),

        "bounded_nll":
            np.concatenate(
                bounded_nll_parts
            ),

        "pre_survival":
            np.concatenate(
                pre_survival_parts,
                axis=0,
            ),

        "post_survival":
            np.concatenate(
                post_survival_parts,
                axis=0,
            ),

        "bounded_survival":
            np.concatenate(
                bounded_survival_parts,
                axis=0,
            ),

        "curve_source":
            curve_source,
    }


###############################################################################
# Metrics
###############################################################################


def patient_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:

    return float(
        frame
        .groupby(
            "patient_id",
            sort=False,
        )[
            column
        ]
        .mean()
        .mean()
    )


def nll_bootstrap(
    frame: pd.DataFrame,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:

    by_patient = (
        frame
        .groupby(
            "patient_id",
            sort=False,
        )
        .agg(
            pre=(
                "pre_nll",
                "mean",
            ),
            bounded=(
                "bounded_nll",
                "mean",
            ),
        )
    )

    gains = (
        by_patient[
            "pre"
        ].to_numpy(
            dtype=float
        )
        - by_patient[
            "bounded"
        ].to_numpy(
            dtype=float
        )
    )

    rng = np.random.default_rng(
        SEED
    )

    n = len(
        gains
    )

    sampled = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(
        repetitions
    ):

        choice = rng.integers(
            0,
            n,
            size=n,
        )

        sampled[
            index
        ] = float(
            gains[
                choice
            ].mean()
        )

    return {
        "mean":
            float(
                gains.mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    sampled,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    sampled,
                    0.975,
                )
            ),

        "patients":
            int(
                n
            ),

        "repetitions":
            int(
                repetitions
            ),
    }


def patient_brier_score_from_components(
    components: pd.DataFrame,
) -> float:

    ###########################################################################
    # Exact CKPT6B aggregation.
    #
    # patient_brier_components() has already applied CKPT6A's row-level
    # patient-balance factor before aggregating numerator and denominator by
    # patient.
    #
    # Therefore the correct point estimate is:
    #
    #       sum(patient numerators) / sum(patient denominators)
    #
    # NOT:
    #
    #       mean(patient numerator / patient denominator)
    #
    ###########################################################################

    if not isinstance(
        components,
        pd.DataFrame,
    ):

        raise RuntimeError(
            "patient_brier_components did not return DataFrame"
        )

    required = {
        "numerator",
        "denominator",
    }

    missing = (
        required
        - set(
            components.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Unexpected patient_brier_components schema: "
            f"missing={sorted(missing)} "
            f"columns={list(components.columns)}"
        )

    numerator = pd.to_numeric(
        components[
            "numerator"
        ],
        errors="coerce",
    )

    denominator = pd.to_numeric(
        components[
            "denominator"
        ],
        errors="coerce",
    )

    valid = (
        numerator.notna()
        & denominator.notna()
        & (
            denominator
            > 0
        )
    )

    if not valid.any():

        return float(
            "nan"
        )

    denominator_sum = float(
        denominator[
            valid
        ].sum()
    )

    if denominator_sum <= 0:

        return float(
            "nan"
        )

    return float(
        numerator[
            valid
        ].sum()
        / denominator_sum
    )


def patient_ibs(
    ckpt6a,
    ckpt6b,
    frame: pd.DataFrame,
    survival: np.ndarray,
    censoring_km,
) -> tuple[
    float,
    dict[str, float],
]:

    ###########################################################################
    # Exact CKPT6A metric_bundle point-estimate path.
    #
    # We invoke horizon_metrics() directly rather than reconstructing its
    # weighting logic.
    ###########################################################################

    survival = np.asarray(
        survival,
        dtype=float,
    )

    if (
        survival.ndim != 2
        or survival.shape[
            0
        ] != len(
            frame
        )
    ):

        raise RuntimeError(
            "patient_ibs survival shape mismatch: "
            f"{survival.shape} vs frame rows={len(frame)}"
        )

    horizon_scores = {}

    for horizon in HORIZONS:

        if horizon - 1 >= survival.shape[
            1
        ]:

            raise RuntimeError(
                f"survival curve lacks horizon {horizon}m"
            )

        metric = ckpt6a.horizon_metrics(
            frame,
            survival[
                :,
                horizon - 1,
            ],
            horizon,
            censoring_km,
            patient_balanced=True,
        )

        value = float(
            metric.get(
                "brier",
                float(
                    "nan"
                ),
            )
        )

        if not np.isfinite(
            value
        ):

            raise RuntimeError(
                f"non-finite patient Brier at {horizon}m: {metric}"
            )

        horizon_scores[
            f"{horizon}m"
        ] = value

    return (
        float(
            np.mean(
                list(
                    horizon_scores.values()
                )
            )
        ),
        horizon_scores,
    )


###############################################################################
# Window data preparation
###############################################################################


def encode_semantic_line(
    values,
    mode: str,
) -> np.ndarray:

    array = np.asarray(
        values,
        dtype=float,
    )

    rounded = np.rint(
        array
    ).astype(
        np.int64
    )

    if not np.allclose(
        array,
        rounded,
        atol=1e-8,
        rtol=0.0,
    ):

        raise RuntimeError(
            "semantic treatment line is not integer-valued"
        )

    if mode == "IDENTITY":

        return rounded

    if mode == "CLIP_0_15":

        return np.clip(
            rounded,
            0,
            15,
        ).astype(
            np.int64
        )

    raise RuntimeError(
        f"unknown CKPT4 line encoding mode: {mode}"
    )


def validate_ckpt4_line_encoding(
    root: Path,
) -> dict[str, Any]:

    semantic = pd.read_parquet(
        root
        / "artifacts"
        / "checkpoint6d"
        / "mapped_scans"
        / "w3_mbc_authoritative.parquet"
    ).copy()

    cache = pd.read_parquet(
        root
        / "artifacts"
        / "checkpoint6d"
        / "regenerated"
        / "w3_replay"
        / "breast_scan_prepost_index.parquet"
    ).copy()

    semantic[
        "patient_id"
    ] = normalize_patient(
        semantic[
            "patient_id"
        ]
    )

    cache[
        "patient_id"
    ] = normalize_patient(
        cache[
            "patient_id"
        ]
    )

    semantic[
        "scan_episode_id"
    ] = semantic[
        "scan_episode_id"
    ].astype(str)

    cache[
        "scan_episode_id"
    ] = cache[
        "scan_episode_id"
    ].astype(str)

    semantic = semantic[
        [
            "patient_id",
            "scan_episode_id",
            "landmark_day",
            "treatment_line",
        ]
    ].rename(
        columns={
            "landmark_day":
                "semantic_landmark_day",

            "treatment_line":
                "semantic_treatment_line",
        }
    )

    cache = cache[
        [
            "patient_id",
            "scan_episode_id",
            "landmark_day",
            "line",
        ]
    ].rename(
        columns={
            "landmark_day":
                "cache_landmark_day",

            "line":
                "cache_line",
        }
    )

    joined = semantic.merge(
        cache,
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(
        joined
    ) != 49189:

        raise RuntimeError(
            "W3 line-encoding validation did not recover "
            f"49,189 rows: {len(joined)}"
        )

    semantic_day = integer_series(
        joined[
            "semantic_landmark_day"
        ]
    ).to_numpy()

    cache_day = integer_series(
        joined[
            "cache_landmark_day"
        ]
    ).to_numpy()

    if not (
        semantic_day
        == cache_day
    ).all():

        raise RuntimeError(
            "W3 cache landmark days differ from authoritative W3."
        )

    semantic_line = integer_series(
        joined[
            "semantic_treatment_line"
        ]
    ).to_numpy()

    cache_line = integer_series(
        joined[
            "cache_line"
        ]
    ).to_numpy()

    identity = (
        semantic_line
        == cache_line
    )

    clipped = (
        np.clip(
            semantic_line,
            0,
            15,
        )
        == cache_line
    )

    if identity.all():

        mode = "IDENTITY"

    elif clipped.all():

        mode = "CLIP_0_15"

    else:

        bad = joined.loc[
            ~clipped,
            [
                "patient_id",
                "scan_episode_id",
                "semantic_treatment_line",
                "cache_line",
            ],
        ].head(
            30
        )

        raise RuntimeError(
            "Frozen CKPT4 line encoding is neither identity nor "
            "clip(line,0,15). "
            f"preview={bad.to_dict(orient='records')}"
        )

    encoded = encode_semantic_line(
        semantic_line,
        mode,
    )

    if not (
        encoded
        == cache_line
    ).all():

        raise RuntimeError(
            "Internal line-encoding verification failed."
        )

    report = {
        "mode":
            mode,

        "rows":
            int(
                len(
                    joined
                )
            ),

        "patients":
            int(
                joined[
                    "patient_id"
                ].nunique()
            ),

        "semantic_line_min":
            int(
                semantic_line.min()
            ),

        "semantic_line_max":
            int(
                semantic_line.max()
            ),

        "cache_line_min":
            int(
                cache_line.min()
            ),

        "cache_line_max":
            int(
                cache_line.max()
            ),

        "rows_semantic_line_gt_15":
            int(
                (
                    semantic_line
                    > 15
                ).sum()
            ),

        "rows_identity":
            int(
                identity.sum()
            ),

        "rows_changed_by_encoding":
            int(
                (
                    semantic_line
                    != cache_line
                ).sum()
            ),

        "exact_contract_match":
            True,
    }

    return report


def load_alt_window(
    root: Path,
    label: str,
    line_encoding_mode: str,
) -> dict[str, Any]:

    if label == "w0":

        mapped_path = (
            root
            / "artifacts"
            / "checkpoint6d"
            / "mapped_scans"
            / "w0_mbc_exact_line.parquet"
        )

        regenerated = (
            root
            / "artifacts"
            / "checkpoint6d"
            / "regenerated"
            / "w0"
        )

    elif label == "w7":

        mapped_path = (
            root
            / "artifacts"
            / "checkpoint6d"
            / "mapped_scans"
            / "w7_mbc_exact_line.parquet"
        )

        regenerated = (
            root
            / "artifacts"
            / "checkpoint6d"
            / "regenerated"
            / "w7"
        )

    else:

        raise ValueError(
            label
        )

    mapped = pd.read_parquet(
        mapped_path
    ).copy()

    mapped[
        "patient_id"
    ] = normalize_patient(
        mapped[
            "patient_id"
        ]
    )

    mapped[
        "scan_episode_id"
    ] = mapped[
        "scan_episode_id"
    ].astype(str)

    mapped[
        "_semantic_day"
    ] = integer_series(
        mapped[
            "landmark_day"
        ]
    )

    mapped[
        "_semantic_line"
    ] = integer_series(
        mapped[
            "treatment_line"
        ]
    )

    index = pd.read_parquet(
        regenerated
        / "breast_scan_prepost_index.parquet"
    ).copy()

    index[
        "patient_id"
    ] = normalize_patient(
        index[
            "patient_id"
        ]
    )

    index[
        "scan_episode_id"
    ] = index[
        "scan_episode_id"
    ].astype(str)

    required_cache_columns = {
        "patient_id",
        "scan_episode_id",
        "landmark_day",
        "line",
        "scan_state",
        "split",
        "embedding_row",
    }

    missing_cache_columns = (
        required_cache_columns
        - set(
            index.columns
        )
    )

    if missing_cache_columns:

        raise RuntimeError(
            f"{label}: regenerated PRE/POST cache index missing "
            f"{sorted(missing_cache_columns)}"
        )

    embeddings = np.load(
        regenerated
        / "breast_scan_prepost_embeddings_f16.npy",
        mmap_mode="r",
    )

    if embeddings.shape != (
        len(
            index
        ),
        2,
        192,
    ):

        raise RuntimeError(
            f"{label}: temporal/index mismatch "
            f"{embeddings.shape} vs {len(index)}"
        )

    index[
        "_embedding_row"
    ] = np.arange(
        len(
            index
        ),
        dtype=np.int64,
    )

    if (
        mapped[
            [
                "patient_id",
                "scan_episode_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{label}: mapped scan IDs not unique"
        )

    if (
        index[
            [
                "patient_id",
                "scan_episode_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{label}: regenerated scan IDs not unique"
        )

    joined = index.merge(
        mapped,
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="inner",
        suffixes=(
            "_cache",
            "_scan",
        ),
        validate="one_to_one",
    )

    if len(
        joined
    ) != len(
        index
    ):

        raise RuntimeError(
            f"{label}: regenerated cache does not map "
            "one-to-one to mapped scans"
        )

    cache_day_column = (
        "landmark_day_cache"
        if "landmark_day_cache"
        in joined.columns
        else "landmark_day"
    )

    mapped_day_column = (
        "landmark_day_scan"
        if "landmark_day_scan"
        in joined.columns
        else "landmark_day"
    )

    cache_day = integer_series(
        joined[
            cache_day_column
        ]
    )

    mapped_day = integer_series(
        joined[
            mapped_day_column
        ]
    )

    if not (
        cache_day.to_numpy()
        == mapped_day.to_numpy()
    ).all():

        mismatch = joined.loc[
            cache_day.to_numpy()
            != mapped_day.to_numpy(),
            [
                "patient_id",
                "scan_episode_id",
                cache_day_column,
                mapped_day_column,
            ],
        ].head(
            20
        )

        raise RuntimeError(
            f"{label}: cache/mapped landmark-day mismatch: "
            f"{mismatch.to_dict(orient='records')}"
        )

    cache_line = integer_series(
        joined[
            "line"
        ]
    ).to_numpy()

    semantic_line = integer_series(
        joined[
            "treatment_line"
        ]
    ).to_numpy()

    expected_cache_line = encode_semantic_line(
        semantic_line,
        line_encoding_mode,
    )

    line_match = (
        cache_line
        == expected_cache_line
    )

    if not line_match.all():

        mismatch = joined.loc[
            ~line_match,
            [
                "patient_id",
                "scan_episode_id",
                "line",
                "treatment_line",
            ],
        ].head(
            30
        )

        raise RuntimeError(
            f"{label}: CKPT4 encoded line does not follow "
            f"frozen W3 encoding contract {line_encoding_mode}. "
            f"preview={mismatch.to_dict(orient='records')}"
        )

    # IMPORTANT:
    #
    # Semantic exact matching uses the real CKPT1 treatment line.
    # The capped/encoded CKPT4 line remains inside the temporal representation.
    joined[
        "_key_day"
    ] = mapped_day.to_numpy(
        dtype=np.int64
    )

    joined[
        "_key_line"
    ] = semantic_line.astype(
        np.int64
    )

    joined[
        "_cache_encoded_line"
    ] = cache_line.astype(
        np.int64
    )

    keys = tuple_keys(
        joined
    )

    if len(
        keys
    ) != len(
        set(
            keys
        )
    ):

        raise RuntimeError(
            f"{label}: semantic patient/day/line key is not unique"
        )

    lookup = {
        key:
            int(
                row
            )
        for key, row
        in zip(
            keys,
            joined[
                "_embedding_row"
            ],
        )
    }

    return {
        "mapped":
            mapped,

        "joined":
            joined,

        "embeddings":
            embeddings,

        "lookup":
            lookup,

        "line_encoding_mode":
            line_encoding_mode,

        "semantic_line_rows":
            int(
                len(
                    joined
                )
            ),

        "encoded_line_rows_changed":
            int(
                (
                    semantic_line
                    != cache_line
                ).sum()
            ),
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
        "--output-dir",
        default="artifacts/checkpoint6d",
    )

    args = parser.parse_args()

    root = Path(
        args.repo_root
    ).resolve()

    out = (
        root
        / args.output_dir
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "[CKPT6D2_DEVICE]",
        device,
        flush=True,
    )

    ###########################################################################
    # Import frozen implementations.
    ###########################################################################

    ckpt5 = load_module(
        "ckpt5_for_ckpt6d2",
        root
        / "scripts"
        / "dynamic_scan"
        / "ckpt5_supervised_dynamic_model.py",
    )

    ckpt6a = load_module(
        "ckpt6a_for_ckpt6d2",
        root
        / "scripts"
        / "dynamic_scan"
        / "ckpt6a_diagnose_scan_update.py",
    )

    ckpt6b = load_module(
        "ckpt6b_for_ckpt6d2",
        root
        / "scripts"
        / "dynamic_scan"
        / "ckpt6b_freeze_bounded_update.py",
    )

    required_functions = [
        (
            ckpt5,
            "scan_features",
        ),
        (
            ckpt5,
            "competing_risk_nll_per_row",
        ),
        (
            ckpt5,
            "probability_curves",
        ),
        (
            ckpt6a,
            "fit_censoring_km",
        ),
        (
            ckpt6b,
            "patient_brier_components",
        ),
        (
            ckpt6b,
            "bootstrap_integrated_brier_gain",
        ),
    ]

    for module, name in required_functions:

        if not hasattr(
            module,
            name,
        ):

            raise RuntimeError(
                f"required frozen API missing: {name}"
            )

    ###########################################################################
    # Frozen candidate.
    ###########################################################################

    checkpoint_path = (
        root
        / "artifacts"
        / "checkpoint6b"
        / "bounded_dynamic_scan_candidate.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    alpha = float(
        checkpoint[
            "ckpt6b_bounded_update"
        ][
            "alpha"
        ]
    )

    if abs(
        alpha
        - ALPHA
    ) > 1e-12:

        raise RuntimeError(
            f"candidate alpha changed: {alpha}"
        )

    model = ckpt5.DynamicFusionModel(
        scan_dim=int(
            checkpoint[
                "scan_dim"
            ]
        ),
        context_dim=int(
            checkpoint[
                "context_dim"
            ]
        ),
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    months = int(
        checkpoint[
            "months"
        ]
    )

    if months != 24:

        raise RuntimeError(
            f"unexpected survival months={months}"
        )

    print(
        "[CKPT6D2_CANDIDATE_PASS]",
        {
            "alpha":
                alpha,

            "parameters":
                int(
                    checkpoint[
                        "parameter_count"
                    ]
                ),

            "months":
                months,

            "forward_signature":
                str(
                    inspect.signature(
                        model.forward
                    )
                ),

            "pre_post_contract":
                checkpoint.get(
                    "pre_post_contract"
                ),
        },
        flush=True,
    )

    ###########################################################################
    # Original CKPT5 prepared arrays.
    ###########################################################################

    prepared = (
        root
        / "artifacts"
        / "checkpoint5"
        / "prepared"
    )

    base_index = pd.read_parquet(
        prepared
        / "scan_index.parquet"
    ).copy()

    base_index[
        "patient_id"
    ] = normalize_patient(
        base_index[
            "patient_id"
        ]
    )

    original_temporal = np.load(
        prepared
        / "temporal_prepost_f16.npy",
        mmap_mode="r",
    )

    tumor = np.load(
        prepared
        / "tumor_embeddings_f16.npy",
        mmap_mode="r",
    )

    frozen_scan_features = np.load(
        prepared
        / "current_scan_features_f32.npy",
        mmap_mode="r",
    )

    context = np.load(
        prepared
        / "context_features_f32.npy",
        mmap_mode="r",
    )

    expected_rows = len(
        base_index
    )

    if original_temporal.shape != (
        expected_rows,
        2,
        192,
    ):

        raise RuntimeError(
            "CKPT5 temporal prepared-array mismatch"
        )

    if tumor.shape != (
        expected_rows,
        128,
    ):

        raise RuntimeError(
            "CKPT5 tumor prepared-array mismatch"
        )

    if frozen_scan_features.shape != (
        expected_rows,
        18,
    ):

        raise RuntimeError(
            "CKPT5 current-scan prepared-array mismatch"
        )

    if context.shape != (
        expected_rows,
        6,
    ):

        raise RuntimeError(
            "CKPT5 context prepared-array mismatch"
        )

    base_index[
        "_base_row"
    ] = np.arange(
        expected_rows,
        dtype=np.int64,
    )

    if (
        base_index[
            [
                "patient_id",
                "scan_episode_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "CKPT5 scan index keys not unique"
        )

    base_row_lookup = {
        (
            str(
                patient
            ),
            str(
                scan
            ),
        ):
            int(
                row
            )
        for patient, scan, row
        in zip(
            base_index[
                "patient_id"
            ],
            base_index[
                "scan_episode_id"
            ].astype(str),
            base_index[
                "_base_row"
            ],
        )
    }

    ###########################################################################
    # Frozen CKPT4 treatment-line encoding contract.
    #
    # CKPT1 retains the true treatment-line number. CKPT4 may use a bounded
    # categorical representation internally. Infer this only from W3, whose
    # semantic treatment lines and frozen cache are both authoritative.
    ###########################################################################

    line_encoding_report = (
        validate_ckpt4_line_encoding(
            root
        )
    )

    line_encoding_mode = (
        line_encoding_report[
            "mode"
        ]
    )

    print(
        "[CKPT6D2_LINE_ENCODING_CONTRACT_PASS]",
        line_encoding_report,
        flush=True,
    )

    ###########################################################################
    # Locked CKPT6B held-out primary predictions.
    ###########################################################################

    locked = pd.read_parquet(
        root
        / "artifacts"
        / "checkpoint6b"
        / "test_bounded_update_predictions.parquet"
    )

    locked = prediction_key_frame(
        locked
    )

    # CKPT6B prediction metadata preserves the TRUE CKPT1 semantic
    # treatment-line number. It must not be confused with CKPT4's bounded
    # categorical line representation inside the temporal encoder/cache.
    locked[
        "_prediction_semantic_line"
    ] = locked[
        "_key_line"
    ].astype(
        np.int64
    )

    w3_semantic = pd.read_parquet(
        root
        / "artifacts"
        / "checkpoint6d"
        / "mapped_scans"
        / "w3_mbc_authoritative.parquet"
    ).copy()

    w3_semantic[
        "patient_id"
    ] = normalize_patient(
        w3_semantic[
            "patient_id"
        ]
    )

    w3_semantic[
        "scan_episode_id"
    ] = w3_semantic[
        "scan_episode_id"
    ].astype(str)

    w3_semantic = w3_semantic[
        [
            "patient_id",
            "scan_episode_id",
            "landmark_day",
            "treatment_line",
        ]
    ].rename(
        columns={
            "landmark_day":
                "_w3_semantic_day",

            "treatment_line":
                "_w3_semantic_line",
        }
    )

    locked[
        "scan_episode_id"
    ] = locked[
        "scan_episode_id"
    ].astype(str)

    locked = locked.merge(
        w3_semantic,
        on=[
            "patient_id",
            "scan_episode_id",
        ],
        how="left",
        validate="one_to_one",
    )

    if (
        locked[
            "_w3_semantic_day"
        ]
        .isna()
        .any()
        or locked[
            "_w3_semantic_line"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Some locked CKPT6B W3 scans lack authoritative "
            "CKPT1 semantic line metadata."
        )

    locked_semantic_day = integer_series(
        locked[
            "_w3_semantic_day"
        ]
    ).to_numpy(
        dtype=np.int64
    )

    locked_prediction_day = locked[
        "_key_day"
    ].to_numpy(
        dtype=np.int64
    )

    if not (
        locked_semantic_day
        == locked_prediction_day
    ).all():

        raise RuntimeError(
            "Locked CKPT6B landmark day differs from "
            "authoritative CKPT1 W3 landmark day."
        )

    locked_semantic_line = integer_series(
        locked[
            "_w3_semantic_line"
        ]
    ).to_numpy(
        dtype=np.int64
    )

    prediction_semantic_line = locked[
        "_prediction_semantic_line"
    ].to_numpy(
        dtype=np.int64
    )

    if not (
        prediction_semantic_line
        == locked_semantic_line
    ).all():

        mismatch = locked.loc[
            prediction_semantic_line
            != locked_semantic_line,
            [
                "patient_id",
                "scan_episode_id",
                "line",
                "_w3_semantic_line",
            ],
        ].head(
            20
        )

        raise RuntimeError(
            "CKPT6B prediction line disagrees with authoritative "
            "CKPT1 semantic treatment line. "
            f"preview={mismatch.to_dict(orient='records')}"
        )

    # Cross-window cohort matching uses the true CKPT1 semantic line.
    locked[
        "_semantic_line"
    ] = locked_semantic_line

    locked[
        "_key_line"
    ] = locked_semantic_line

    print(
        "[CKPT6D2_LOCKED_SEMANTIC_LINE_PASS]",
        {
            "rows":
                int(
                    len(
                        locked
                    )
                ),

            "rows_line_gt15":
                int(
                    (
                        locked_semantic_line
                        > 15
                    ).sum()
                ),

            "semantic_line_max":
                int(
                    locked_semantic_line.max()
                ),

            "note":
                (
                    "CKPT6B line is semantic CKPT1 line; "
                    "CKPT4 temporal cache alone uses bounded encoding"
                ),
        },
        flush=True,
    )

    if len(
        locked
    ) != 3102:

        raise RuntimeError(
            f"locked CKPT6B test row count changed: {len(locked)}"
        )

    if (
        locked[
            "patient_id"
        ].nunique()
        != 437
    ):

        raise RuntimeError(
            "locked CKPT6B test patient count changed"
        )

    if not (
        locked[
            "scan_state"
        ].astype(str)
        == "NON_PROGRESSIVE"
    ).all():

        raise RuntimeError(
            "CKPT6B primary rows are not all NON_PROGRESSIVE"
        )

    locked[
        "_base_row"
    ] = [
        base_row_lookup.get(
            (
                str(
                    patient
                ),
                str(
                    scan
                ),
            ),
            -1,
        )
        for patient, scan
        in zip(
            locked[
                "patient_id"
            ],
            locked[
                "scan_episode_id"
            ].astype(str),
        )
    ]

    if (
        locked[
            "_base_row"
        ]
        < 0
    ).any():

        raise RuntimeError(
            "CKPT6B held-out rows missing from CKPT5 prepared index"
        )

    ###########################################################################
    # W0/W7 exact same-day/same-line maps.
    ###########################################################################

    w0 = load_alt_window(
        root,
        "w0",
        line_encoding_mode,
    )

    w7 = load_alt_window(
        root,
        "w7",
        line_encoding_mode,
    )

    locked_keys = tuple_keys(
        locked
    )

    w0_key_set = set(
        w0[
            "lookup"
        ]
    )

    w7_key_set = set(
        w7[
            "lookup"
        ]
    )

    locked[
        "_w0_exact"
    ] = [
        key
        in w0_key_set
        for key in locked_keys
    ]

    locked[
        "_w7_exact"
    ] = [
        key
        in w7_key_set
        for key in locked_keys
    ]

    ###########################################################################
    # State preservation.
    #
    # For primary residual-PFS comparison a scan must still be NON_PROGRESSIVE
    # under the sensitivity grouping rule.
    ###########################################################################

    def normalize_scan_state_value(
        value: Any,
    ) -> str:

        if pd.isna(
            value
        ):

            return "UNKNOWN"

        if isinstance(
            value,
            (
                int,
                np.integer,
                float,
                np.floating,
            ),
        ):

            numeric = float(
                value
            )

            if np.isfinite(
                numeric
            ):

                rounded = int(
                    round(
                        numeric
                    )
                )

                if abs(
                    numeric
                    - rounded
                ) < 1e-8:

                    if rounded == 0:
                        return "NON_PROGRESSIVE"

                    if rounded == 1:
                        return "PROGRESSIVE"

                    if rounded == 2:
                        return "INDETERMINATE"

        text_value = (
            str(
                value
            )
            .strip()
            .upper()
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        )

        aliases = {
            "NON_PROGRESSIVE":
                "NON_PROGRESSIVE",

            "NONPROGRESSIVE":
                "NON_PROGRESSIVE",

            "N":
                "NON_PROGRESSIVE",

            "NO":
                "NON_PROGRESSIVE",

            "PROGRESSIVE":
                "PROGRESSIVE",

            "Y":
                "PROGRESSIVE",

            "YES":
                "PROGRESSIVE",

            "INDETERMINATE":
                "INDETERMINATE",

            "INDETERMINATE/UNKNOWN":
                "INDETERMINATE",

            "UNKNOWN":
                "INDETERMINATE",
        }

        return aliases.get(
            text_value,
            text_value,
        )


    def state_map(
        window: dict[str, Any],
    ) -> dict[
        tuple[str, int, int],
        str
    ]:

        joined = window[
            "joined"
        ]

        keys = tuple_keys(
            joined
        )

        # Canonical radiology semantics are primary.
        mapped_state_column = None

        for candidate in (
            "progression_state_3",
            "progression_state_3_scan",
            "scan_state_scan",
        ):

            if candidate in joined.columns:

                mapped_state_column = candidate
                break

        if mapped_state_column is None:

            raise RuntimeError(
                "Alternate window lacks canonical scan-state field. "
                f"columns={list(joined.columns)}"
            )

        mapped_states = [
            normalize_scan_state_value(
                value
            )
            for value in joined[
                mapped_state_column
            ]
        ]

        # Independently verify CKPT4 cached numeric state where available.
        cache_state_column = None

        for candidate in (
            "scan_state_cache",
            "scan_state",
        ):

            if candidate in joined.columns:

                cache_state_column = candidate
                break

        if cache_state_column is not None:

            cache_states = [
                normalize_scan_state_value(
                    value
                )
                for value in joined[
                    cache_state_column
                ]
            ]

            disagreements = [
                index
                for index, (
                    mapped_state,
                    cache_state,
                )
                in enumerate(
                    zip(
                        mapped_states,
                        cache_states,
                    )
                )
                if (
                    mapped_state
                    != cache_state
                )
            ]

            if disagreements:

                preview_rows = disagreements[
                    :20
                ]

                preview = []

                for index in preview_rows:

                    preview.append(
                        {
                            "patient_id":
                                str(
                                    joined.iloc[
                                        index
                                    ][
                                        "patient_id"
                                    ]
                                ),

                            "scan_episode_id":
                                str(
                                    joined.iloc[
                                        index
                                    ][
                                        "scan_episode_id"
                                    ]
                                ),

                            "mapped":
                                mapped_states[
                                    index
                                ],

                            "cache":
                                cache_states[
                                    index
                                ],
                        }
                    )

                raise RuntimeError(
                    "Canonical scan state disagrees with CKPT4 "
                    f"cached scan state for {len(disagreements)} rows. "
                    f"preview={preview}"
                )

        return {
            key:
                state
            for key, state
            in zip(
                keys,
                mapped_states,
            )
        }


    w0_state_map = state_map(
        w0
    )

    w7_state_map = state_map(
        w7
    )

    locked[
        "_w0_state"
    ] = [
        w0_state_map.get(
            key
        )
        for key in locked_keys
    ]

    locked[
        "_w7_state"
    ] = [
        w7_state_map.get(
            key
        )
        for key in locked_keys
    ]

    locked[
        "_w0_nonprogressive"
    ] = (
        locked[
            "_w0_state"
        ]
        == "NON_PROGRESSIVE"
    )

    locked[
        "_w7_nonprogressive"
    ] = (
        locked[
            "_w7_state"
        ]
        == "NON_PROGRESSIVE"
    )

    common_mask = (
        locked[
            "_w0_exact"
        ]
        & locked[
            "_w7_exact"
        ]
        & locked[
            "_w0_nonprogressive"
        ]
        & locked[
            "_w7_nonprogressive"
        ]
    )

    common = (
        locked[
            common_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(
        common
    ) < 0.80 * len(
        locked
    ):

        raise RuntimeError(
            "Common W0/W3/W7 primary sensitivity cohort "
            f"is unexpectedly small: {len(common)}/{len(locked)}"
        )

    print(
        "[CKPT6D2_COMMON_COHORT]",
        {
            "locked_test_rows":
                int(
                    len(
                        locked
                    )
                ),

            "locked_test_patients":
                int(
                    locked[
                        "patient_id"
                    ].nunique()
                ),

            "w0_exact_rows":
                int(
                    locked[
                        "_w0_exact"
                    ].sum()
                ),

            "w7_exact_rows":
                int(
                    locked[
                        "_w7_exact"
                    ].sum()
                ),

            "w0_nonprogressive_rows":
                int(
                    (
                        locked[
                            "_w0_exact"
                        ]
                        & locked[
                            "_w0_nonprogressive"
                        ]
                    ).sum()
                ),

            "w7_nonprogressive_rows":
                int(
                    (
                        locked[
                            "_w7_exact"
                        ]
                        & locked[
                            "_w7_nonprogressive"
                        ]
                    ).sum()
                ),

            "common_rows":
                int(
                    len(
                        common
                    )
                ),

            "common_patients":
                int(
                    common[
                        "patient_id"
                    ].nunique()
                ),

            "common_fraction":
                float(
                    len(
                        common
                    )
                    / len(
                        locked
                    )
                ),
        },
        flush=True,
    )

    ###########################################################################
    # Validate exact CKPT5 scan_features() implementation on W3 before using it
    # for W0/W7.
    ###########################################################################

    w3_mapped = pd.read_parquet(
        root
        / "artifacts"
        / "checkpoint6d"
        / "mapped_scans"
        / "w3_mbc_authoritative.parquet"
    )

    w3_mapped[
        "patient_id"
    ] = normalize_patient(
        w3_mapped[
            "patient_id"
        ]
    )

    w3_scan_lookup = {
        (
            str(
                patient
            ),
            str(
                scan
            ),
        ):
            index
        for index, (
            patient,
            scan,
        )
        in enumerate(
            zip(
                w3_mapped[
                    "patient_id"
                ],
                w3_mapped[
                    "scan_episode_id"
                ].astype(str),
            )
        )
    }

    locked_w3_scan_rows = np.asarray(
        [
            w3_scan_lookup[
                (
                    str(
                        patient
                    ),
                    str(
                        scan
                    ),
                )
            ]
            for patient, scan
            in zip(
                locked[
                    "patient_id"
                ],
                locked[
                    "scan_episode_id"
                ].astype(str),
            )
        ],
        dtype=np.int64,
    )

    w3_feature_frame = (
        w3_mapped.iloc[
            locked_w3_scan_rows
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    w3_feature_frame = adapt_ckpt5_scan_feature_schema(
        w3_feature_frame,
        "w3_feature_replay",
    )

    (
        reconstructed_scan_features,
        scan_feature_names,
    ) = ckpt5.scan_features(
        w3_feature_frame
    )

    original_locked_rows = locked[
        "_base_row"
    ].to_numpy(
        dtype=np.int64
    )

    original_locked_scan_features = np.asarray(
        frozen_scan_features[
            original_locked_rows
        ],
        dtype=np.float32,
    )

    scan_feature_max_abs = float(
        np.max(
            np.abs(
                np.asarray(
                    reconstructed_scan_features,
                    dtype=np.float32,
                )
                - original_locked_scan_features
            )
        )
    )

    if scan_feature_max_abs > 1e-6:

        raise RuntimeError(
            "Frozen CKPT5 scan_features() does not reproduce "
            "prepared W3 scan features: "
            f"max_abs={scan_feature_max_abs}"
        )

    print(
        "[CKPT6D2_SCAN_FEATURE_REPLAY_PASS]",
        {
            "dim":
                int(
                    reconstructed_scan_features.shape[
                        1
                    ]
                ),

            "max_abs":
                scan_feature_max_abs,

            "names":
                scan_feature_names,
        },
        flush=True,
    )

    ###########################################################################
    # Full-W3 CKPT6B reconstruction gate.
    ###########################################################################

    full_rows = original_locked_rows

    full_result = infer(
        ckpt5=ckpt5,
        model=model,
        temporal_prepost=np.asarray(
            original_temporal[
                full_rows
            ],
            dtype=np.float32,
        ),
        tumor=np.asarray(
            tumor[
                full_rows
            ],
            dtype=np.float32,
        ),
        scan_features=np.asarray(
            frozen_scan_features[
                full_rows
            ],
            dtype=np.float32,
        ),
        context=np.asarray(
            context[
                full_rows
            ],
            dtype=np.float32,
        ),
        time_days=locked[
            "survival_time_days"
        ].to_numpy(
            dtype=float
        ),
        cause=locked[
            "survival_cause"
        ].to_numpy(
            dtype=int
        ),
        device=device,
        months=months,
    )

    replay_checks = {
        "pre_nll_max_abs":
            float(
                np.max(
                    np.abs(
                        full_result[
                            "pre_nll"
                        ]
                        - locked[
                            "pre_nll"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
            ),

        "full_post_nll_max_abs":
            float(
                np.max(
                    np.abs(
                        full_result[
                            "post_nll"
                        ]
                        - locked[
                            "full_post_nll"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
            ),

        "bounded_nll_max_abs":
            float(
                np.max(
                    np.abs(
                        full_result[
                            "bounded_nll"
                        ]
                        - locked[
                            "bounded_post_nll"
                        ].to_numpy(
                            dtype=float
                        )
                    )
                )
            ),
    }

    for horizon in HORIZONS:

        month_index = (
            horizon
            - 1
        )

        replay_checks[
            f"pre_pfs_{horizon}m_max_abs"
        ] = float(
            np.max(
                np.abs(
                    full_result[
                        "pre_survival"
                    ][
                        :,
                        month_index
                    ]
                    - locked[
                        f"pre_pfs_{horizon}m"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        replay_checks[
            f"bounded_pfs_{horizon}m_max_abs"
        ] = float(
            np.max(
                np.abs(
                    full_result[
                        "bounded_survival"
                    ][
                        :,
                        month_index
                    ]
                    - locked[
                        f"bounded_post_pfs_{horizon}m"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

    if max(
        replay_checks.values()
    ) > 5e-4:

        raise RuntimeError(
            "Frozen CKPT6B prediction replay failed: "
            f"{replay_checks}"
        )

    print(
        "[CKPT6D2_CKPT6B_PREDICTION_REPLAY_PASS]",
        replay_checks,
        flush=True,
    )

    ###########################################################################
    # Training censoring KM exactly from frozen CKPT5 primary survival rows.
    ###########################################################################

    required_survival_columns = {
        "split",
        "survival_mask",
        "survival_time_days",
        "survival_cause",
    }

    missing_survival = (
        required_survival_columns
        - set(
            base_index.columns
        )
    )

    if missing_survival:

        raise RuntimeError(
            "CKPT5 scan index missing survival fields: "
            f"{sorted(missing_survival)}"
        )

    ###########################################################################
    # EXACT frozen CKPT6B censoring-KM contract.
    #
    # CKPT6B used:
    #
    #   split == train
    #   AND survival_mask == True
    #
    # and passed the ORIGINAL multicause survival_cause directly into
    # ckpt6a.fit_censoring_km().
    #
    # Do NOT collapse progression/death vs everything else here. In
    # particular, treatment SWITCH is a competing cause, not censoring.
    ###########################################################################

    train_survival = base_index[
        (
            base_index[
                "split"
            ].astype(str)
            == "train"
        )
        & base_index[
            "survival_mask"
        ].astype(
            bool
        )
    ].copy()

    if len(
        train_survival
    ) < 1000:

        raise RuntimeError(
            "too few frozen CKPT6B training survival landmarks"
        )

    if (
        train_survival[
            "survival_time_days"
        ]
        .isna()
        .any()
        or train_survival[
            "survival_cause"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Frozen CKPT6B survival-mask rows contain missing "
            "survival time/cause."
        )

    censoring_km = (
        ckpt6a.fit_censoring_km(
            train_survival[
                "survival_time_days"
            ]
            .to_numpy(
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
    )

    print(
        "[CKPT6D2_CENSORING_CONTRACT_PASS]",
        {
            "training_rows":
                int(
                    len(
                        train_survival
                    )
                ),

            "training_patients":
                int(
                    train_survival[
                        "patient_id"
                    ].nunique()
                ),

            "cause_counts":
                {
                    str(
                        int(
                            key
                        )
                    ):
                        int(
                            value
                        )
                    for key, value
                    in train_survival[
                        "survival_cause"
                    ]
                    .astype(
                        int
                    )
                    .value_counts()
                    .sort_index()
                    .to_dict()
                    .items()
                },

            "contract":
                (
                    "CKPT6B exact: train & survival_mask; "
                    "raw multicause survival_cause passed to fit_censoring_km"
                ),
        },
        flush=True,
    )

    ###########################################################################
    # Reproduce locked full-test patient metrics before sensitivity analysis.
    ###########################################################################

    replay_frame = locked.copy()

    replay_frame[
        "pre_nll_reconstructed"
    ] = full_result[
        "pre_nll"
    ]

    replay_frame[
        "bounded_nll_reconstructed"
    ] = full_result[
        "bounded_nll"
    ]

    full_pre_patient_nll = patient_mean(
        replay_frame.rename(
            columns={
                "pre_nll_reconstructed":
                    "_metric"
            }
        ),
        "_metric",
    )

    full_bounded_patient_nll = patient_mean(
        replay_frame.rename(
            columns={
                "bounded_nll_reconstructed":
                    "_metric"
            }
        ),
        "_metric",
    )

    (
        full_pre_ibs,
        full_pre_brier,
    ) = patient_ibs(
        ckpt6a,
        ckpt6b,
        replay_frame,
        full_result[
            "pre_survival"
        ],
        censoring_km,
    )

    (
        full_bounded_ibs,
        full_bounded_brier,
    ) = patient_ibs(
        ckpt6a,
        ckpt6b,
        replay_frame,
        full_result[
            "bounded_survival"
        ],
        censoring_km,
    )

    locked_metric_replay = {
        "pre_patient_nll":
            full_pre_patient_nll,

        "bounded_patient_nll":
            full_bounded_patient_nll,

        "pre_patient_ibs":
            full_pre_ibs,

        "bounded_patient_ibs":
            full_bounded_ibs,

        "expected_pre_patient_nll":
            2.7634854849024917,

        "expected_bounded_patient_nll":
            2.7314647641077596,

        "expected_pre_patient_ibs":
            0.1673434533268599,

        "expected_bounded_patient_ibs":
            0.16352614872814603,
    }

    metric_diffs = [
        abs(
            full_pre_patient_nll
            - 2.7634854849024917
        ),
        abs(
            full_bounded_patient_nll
            - 2.7314647641077596
        ),
        abs(
            full_pre_ibs
            - 0.1673434533268599
        ),
        abs(
            full_bounded_ibs
            - 0.16352614872814603
        ),
    ]

    if max(
        metric_diffs
    ) > 5e-5:

        raise RuntimeError(
            "CKPT6B metric replay failed: "
            f"{locked_metric_replay}"
        )

    print(
        "[CKPT6D2_CKPT6B_METRIC_REPLAY_PASS]",
        locked_metric_replay,
        flush=True,
    )

    ###########################################################################
    # Common-cohort arrays.
    ###########################################################################

    common_keys = tuple_keys(
        common
    )

    base_rows = common[
        "_base_row"
    ].to_numpy(
        dtype=np.int64
    )

    common_tumor = np.asarray(
        tumor[
            base_rows
        ],
        dtype=np.float32,
    )

    # Controlled window-sensitivity design:
    #
    # tumor/genomics and structured non-radiology clinical context are held
    # fixed because patient, landmark day, and treatment line are exactly
    # matched. The grouping-window perturbation is applied only through:
    #   1. the fully regenerated causal temporal PRE/POST history, and
    #   2. explicit current-scan features.
    #
    common_context = np.asarray(
        context[
            base_rows
        ],
        dtype=np.float32,
    )

    common_time = common[
        "survival_time_days"
    ].to_numpy(
        dtype=float
    )

    common_cause = common[
        "survival_cause"
    ].to_numpy(
        dtype=int
    )

    ###########################################################################
    # Window builder.
    ###########################################################################

    def build_window_inputs(
        label: str,
    ):

        if label == "w3":

            temporal = np.asarray(
                original_temporal[
                    base_rows
                ],
                dtype=np.float32,
            )

            scan = np.asarray(
                frozen_scan_features[
                    base_rows
                ],
                dtype=np.float32,
            )

            return (
                temporal,
                scan,
                common[
                    "scan_episode_id"
                ]
                .astype(str)
                .to_numpy(),
            )

        window = (
            w0
            if label
            == "w0"
            else w7
        )

        embedding_rows = np.asarray(
            [
                window[
                    "lookup"
                ][
                    key
                ]
                for key
                in common_keys
            ],
            dtype=np.int64,
        )

        temporal = np.asarray(
            window[
                "embeddings"
            ][
                embedding_rows
            ],
            dtype=np.float32,
        )

        joined = window[
            "joined"
        ]

        joined_lookup = {
            key:
                index
            for index, key
            in enumerate(
                tuple_keys(
                    joined
                )
            )
        }

        joined_rows = np.asarray(
            [
                joined_lookup[
                    key
                ]
                for key
                in common_keys
            ],
            dtype=np.int64,
        )

        scan_frame = (
            joined.iloc[
                joined_rows
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        # scan_features() needs original canonical scan-column names.
        #
        # After the index/mapped merge the mapped scan columns may carry
        # suffix "_scan". Restore those names where necessary.
        for column in list(
            scan_frame.columns
        ):

            if (
                column.endswith(
                    "_scan"
                )
                and column[
                    :-5
                ]
                not in scan_frame.columns
            ):

                scan_frame[
                    column[
                        :-5
                    ]
                ] = scan_frame[
                    column
                ]

        scan_frame = adapt_ckpt5_scan_feature_schema(
            scan_frame,
            f"{label}_current_scan_features",
        )

        (
            scan,
            feature_names,
        ) = ckpt5.scan_features(
            scan_frame
        )

        if list(
            feature_names
        ) != list(
            scan_feature_names
        ):

            raise RuntimeError(
                f"{label}: explicit scan feature names changed"
            )

        if scan.shape != (
            len(
                common
            ),
            18,
        ):

            raise RuntimeError(
                f"{label}: unexpected scan feature shape "
                f"{scan.shape}"
            )

        scan_id_column = (
            "scan_episode_id"
            if "scan_episode_id"
            in scan_frame.columns
            else "scan_episode_id_scan"
        )

        return (
            temporal,
            np.asarray(
                scan,
                dtype=np.float32,
            ),
            scan_frame[
                scan_id_column
            ]
            .astype(str)
            .to_numpy(),
        )

    ###########################################################################
    # Evaluate common cohort under W0/W3/W7.
    ###########################################################################

    results = {}

    prediction_output = common[
        [
            "patient_id",
            "scan_episode_id",
            "landmark_day",
            "line",
            "_semantic_line",
            "target_type",
            "survival_time_days",
            "survival_cause",
        ]
    ].copy()

    prediction_output = prediction_output.rename(
        columns={
            "_semantic_line":
                "semantic_treatment_line"
        }
    )

    survival_by_window = {}

    for label in (
        "w0",
        "w3",
        "w7",
    ):

        print(
            f"[CKPT6D2_INFER] {label}",
            flush=True,
        )

        (
            temporal_window,
            scan_window,
            alternate_scan_ids,
        ) = build_window_inputs(
            label
        )

        inference = infer(
            ckpt5=ckpt5,
            model=model,
            temporal_prepost=temporal_window,
            tumor=common_tumor,
            scan_features=scan_window,
            context=common_context,
            time_days=common_time,
            cause=common_cause,
            device=device,
            months=months,
        )

        frame = common[
            [
                "patient_id",
                "target_type",
                "survival_time_days",
                "survival_cause",
            ]
        ].copy()

        frame[
            "pre_nll"
        ] = inference[
            "pre_nll"
        ]

        frame[
            "bounded_nll"
        ] = inference[
            "bounded_nll"
        ]

        pre_patient_nll = patient_mean(
            frame,
            "pre_nll",
        )

        bounded_patient_nll = patient_mean(
            frame,
            "bounded_nll",
        )

        (
            pre_ibs,
            pre_brier_horizons,
        ) = patient_ibs(
            ckpt6a,
            ckpt6b,
            frame,
            inference[
                "pre_survival"
            ],
            censoring_km,
        )

        (
            bounded_ibs,
            bounded_brier_horizons,
        ) = patient_ibs(
            ckpt6a,
            ckpt6b,
            frame,
            inference[
                "bounded_survival"
            ],
            censoring_km,
        )

        nll_ci = ckpt6a.bootstrap_nll_gain(
            frame,
            inference[
                "pre_nll"
            ],
            inference[
                "bounded_nll"
            ],
            repetitions=BOOTSTRAP_REPETITIONS,
        )

        ibs_ci = (
            ckpt6b.bootstrap_integrated_brier_gain(
                ckpt6a,
                frame,
                inference[
                    "pre_survival"
                ],
                inference[
                    "bounded_survival"
                ],
                censoring_km,
                repetitions=(
                    BOOTSTRAP_REPETITIONS
                ),
            )
        )

        results[
            label
        ] = {
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

            "pre_patient_nll":
                pre_patient_nll,

            "bounded_patient_nll":
                bounded_patient_nll,

            "pre_minus_bounded_patient_nll":
                (
                    pre_patient_nll
                    - bounded_patient_nll
                ),

            "pre_patient_ibs":
                pre_ibs,

            "bounded_patient_ibs":
                bounded_ibs,

            "pre_minus_bounded_patient_ibs":
                (
                    pre_ibs
                    - bounded_ibs
                ),

            "pre_brier_horizons":
                pre_brier_horizons,

            "bounded_brier_horizons":
                bounded_brier_horizons,

            "nll_gain_bootstrap":
                nll_ci,

            "ibs_gain_bootstrap":
                ibs_ci,

            "pfs_curve_source":
                inference[
                    "curve_source"
                ],
        }

        prediction_output[
            f"{label}_scan_episode_id"
        ] = alternate_scan_ids

        prediction_output[
            f"{label}_pre_nll"
        ] = inference[
            "pre_nll"
        ]

        prediction_output[
            f"{label}_bounded_nll"
        ] = inference[
            "bounded_nll"
        ]

        for horizon in HORIZONS:

            column = (
                horizon
                - 1
            )

            prediction_output[
                f"{label}_pre_pfs_{horizon}m"
            ] = (
                inference[
                    "pre_survival"
                ][
                    :,
                    column
                ]
            )

            prediction_output[
                f"{label}_bounded_pfs_{horizon}m"
            ] = (
                inference[
                    "bounded_survival"
                ][
                    :,
                    column
                ]
            )

        survival_by_window[
            label
        ] = inference[
            "bounded_survival"
        ]

    ###########################################################################
    # Direct prediction sensitivity to grouping.
    ###########################################################################

    prediction_sensitivity = {}

    for comparison in (
        (
            "w0",
            "w3",
        ),
        (
            "w7",
            "w3",
        ),
        (
            "w0",
            "w7",
        ),
    ):

        left, right = comparison

        horizon_metrics = {}

        for horizon in HORIZONS:

            column = (
                horizon
                - 1
            )

            difference = np.abs(
                survival_by_window[
                    left
                ][
                    :,
                    column
                ]
                - survival_by_window[
                    right
                ][
                    :,
                    column
                ]
            )

            horizon_metrics[
                f"{horizon}m"
            ] = {
                "mean_absolute_pfs_difference":
                    float(
                        difference.mean()
                    ),

                "median_absolute_pfs_difference":
                    float(
                        np.median(
                            difference
                        )
                    ),

                "p95_absolute_pfs_difference":
                    float(
                        np.quantile(
                            difference,
                            0.95,
                        )
                    ),
            }

        prediction_sensitivity[
            f"{left}_vs_{right}"
        ] = horizon_metrics

    ###########################################################################
    # Predeclared robustness decision.
    #
    # We do NOT require statistical significance in every sensitivity window.
    # W3 IBS itself was directionally positive but narrowly uncertain.
    #
    # PASS requires:
    #   1. positive patient-balanced NLL gain in every window;
    #   2. no window has a 95% IBS interval proving harm;
    #   3. at least 2/3 windows have nonnegative IBS point gain;
    #   4. absolute bounded-IBS spread across W0/W3/W7 <= 0.015.
    ###########################################################################

    nll_positive_all = all(
        results[
            label
        ][
            "pre_minus_bounded_patient_nll"
        ]
        > 0
        for label
        in (
            "w0",
            "w3",
            "w7",
        )
    )

    ibs_nonnegative_count = sum(
        results[
            label
        ][
            "pre_minus_bounded_patient_ibs"
        ]
        >= 0
        for label
        in (
            "w0",
            "w3",
            "w7",
        )
    )

    ibs_no_supported_harm = all(
        float(
            results[
                label
            ][
                "ibs_gain_bootstrap"
            ][
                "ci_high"
            ]
        )
        >= 0
        for label
        in (
            "w0",
            "w3",
            "w7",
        )
    )

    bounded_ibs_values = [
        float(
            results[
                label
            ][
                "bounded_patient_ibs"
            ]
        )
        for label
        in (
            "w0",
            "w3",
            "w7",
        )
    ]

    bounded_ibs_spread = (
        max(
            bounded_ibs_values
        )
        - min(
            bounded_ibs_values
        )
    )

    robustness_flags = {
        "nll_gain_positive_all_windows":
            bool(
                nll_positive_all
            ),

        "ibs_point_gain_nonnegative_windows":
            int(
                ibs_nonnegative_count
            ),

        "ibs_no_statistically_supported_harm_all_windows":
            bool(
                ibs_no_supported_harm
            ),

        "bounded_ibs_spread":
            float(
                bounded_ibs_spread
            ),

        "bounded_ibs_spread_le_0_015":
            bool(
                bounded_ibs_spread
                <= 0.015
            ),
    }

    passed = (
        nll_positive_all
        and ibs_nonnegative_count
        >= 2
        and ibs_no_supported_harm
        and bounded_ibs_spread
        <= 0.015
    )

    status = (
        "PASS_WINDOW_GROUPING_ROBUSTNESS"
        if passed
        else "WINDOW_GROUPING_ROBUSTNESS_REVIEW_REQUIRED"
    )

    ###########################################################################
    # Persist.
    ###########################################################################

    cohort_report = {
        "locked_w3_test_rows":
            int(
                len(
                    locked
                )
            ),

        "locked_w3_test_patients":
            int(
                locked[
                    "patient_id"
                ].nunique()
            ),

        "w0_exact_rows":
            int(
                locked[
                    "_w0_exact"
                ].sum()
            ),

        "w7_exact_rows":
            int(
                locked[
                    "_w7_exact"
                ].sum()
            ),

        "w0_state_counts_on_locked_rows":
            {
                str(key):
                    int(value)
                for key, value
                in locked[
                    "_w0_state"
                ]
                .fillna(
                    "NO_EXACT_MATCH"
                )
                .value_counts()
                .to_dict()
                .items()
            },

        "w7_state_counts_on_locked_rows":
            {
                str(key):
                    int(value)
                for key, value
                in locked[
                    "_w7_state"
                ]
                .fillna(
                    "NO_EXACT_MATCH"
                )
                .value_counts()
                .to_dict()
                .items()
            },

        "common_rows":
            int(
                len(
                    common
                )
            ),

        "common_patients":
            int(
                common[
                    "patient_id"
                ].nunique()
            ),

        "common_fraction_of_locked_test":
            float(
                len(
                    common
                )
                / len(
                    locked
                )
            ),

        "primary_comparison_definition":
            (
                "Exact same patient + landmark day + treatment line, "
                "and NON_PROGRESSIVE under W0/W3/W7."
            ),
    }

    report = {
        "status":
            status,

        "frozen_candidate": {
            "alpha":
                ALPHA,

            "formula":
                "PRE + 0.75 * (FULL_POST - PRE)",

            "candidate_checkpoint":
                str(
                    checkpoint_path.relative_to(
                        root
                    )
                ),

            "candidate_checkpoint_sha256":
                sha256(
                    checkpoint_path
                ),

            "parameter_count":
                int(
                    checkpoint[
                        "parameter_count"
                    ]
                ),
        },

        "controlled_sensitivity_design": {
            "changed_by_window": [
                (
                    "causal temporal PRE/POST representations "
                    "regenerated from the full event history"
                ),
                (
                    "explicit current-scan features reconstructed "
                    "from each W0/W3/W7 episode"
                ),
            ],

            "held_fixed_at_exact_landmark": [
                "patient identity",
                "landmark day",
                "treatment line",
                "residual-PFS target/time origin",
                "tumor/genomic state",
                "structured non-radiology clinical context",
                "supervised model weights",
                "alpha=0.75",
            ],
        },

        "cohort":
            cohort_report,

        "w3_prediction_replay":
            replay_checks,

        "w3_metric_replay":
            locked_metric_replay,

        "ckpt4_line_encoding_contract":
            line_encoding_report,

        "scan_feature_replay_max_abs":
            scan_feature_max_abs,

        "scan_feature_names":
            list(
                scan_feature_names
            ),

        "results":
            results,

        "prediction_sensitivity":
            prediction_sensitivity,

        "robustness_flags":
            robustness_flags,

        "decision_rule": {
            "nll":
                (
                    "patient-balanced PRE-minus-bounded NLL "
                    "must be positive in W0/W3/W7"
                ),

            "ibs":
                (
                    "at least two of three point gains nonnegative "
                    "and no window shows bootstrap-supported harm"
                ),

            "absolute_performance":
                (
                    "max bounded patient IBS spread across "
                    "W0/W3/W7 <= 0.015"
                ),
        },

        "external_validation_outcomes_accessed":
            False,

        "next_action":
            (
                "Freeze CKPT6E external-validation protocol and immutable "
                "candidate manifest before opening DFCI/VICC outcomes."
                if passed
                else (
                    "Do not open DFCI/VICC outcomes. Review W0/W7 "
                    "sensitivity failure before external validation."
                )
            ),
    }

    atomic_json(
        out
        / "window_robustness.json",
        report,
    )

    atomic_parquet(
        out
        / "window_robustness_predictions.parquet",
        prediction_output,
    )

    atomic_json(
        out
        / "window_common_cohort.json",
        cohort_report,
    )

    ###########################################################################
    # Audit.
    ###########################################################################

    audit = f"""
# Checkpoint 6D — scan grouping robustness

Status: `{status}`

Frozen supervised candidate:

- alpha: 0.75
- update: PRE + 0.75 * (FULL_POST - PRE)
- no retraining
- no alpha retuning
- no DFCI/VICC outcome access

## Exact W3 replay

Prediction replay maximum absolute deviations:

{json.dumps(replay_checks, indent=2)}

Locked patient-metric replay:

{json.dumps(locked_metric_replay, indent=2)}

## Controlled W0/W3/W7 design

Primary comparison uses the same patient, same landmark day, same treatment
line, and scans that remain NON_PROGRESSIVE under all three grouping rules.

Radiology-sensitive channels changed:

1. causal temporal PRE/POST history state;
2. explicit current-scan features.

The genomic/tumor representation and structured non-radiology clinical
context remain fixed because the patient, landmark day, and line are exact
matches.

## Common cohort

{json.dumps(cohort_report, indent=2)}

## Results

{json.dumps(results, indent=2)}

## Robustness decision

{json.dumps(robustness_flags, indent=2)}

## External validation

DFCI/VICC outcomes remain untouched.

Next:

{report["next_action"]}
""".strip() + "\n"

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
            "6D",

        "name":
            "scan_grouping_window_robustness",

        "status":
            status,

        "frozen_alpha":
            ALPHA,

        "ckpt4_line_encoding_contract":
            line_encoding_report,

        "common_test_rows":
            int(
                len(
                    common
                )
            ),

        "common_test_patients":
            int(
                common[
                    "patient_id"
                ].nunique()
            ),

        "window_results": {
            label: {
                "pre_patient_nll":
                    results[
                        label
                    ][
                        "pre_patient_nll"
                    ],

                "bounded_patient_nll":
                    results[
                        label
                    ][
                        "bounded_patient_nll"
                    ],

                "nll_gain":
                    results[
                        label
                    ][
                        "pre_minus_bounded_patient_nll"
                    ],

                "nll_gain_ci":
                    results[
                        label
                    ][
                        "nll_gain_bootstrap"
                    ],

                "pre_patient_ibs":
                    results[
                        label
                    ][
                        "pre_patient_ibs"
                    ],

                "bounded_patient_ibs":
                    results[
                        label
                    ][
                        "bounded_patient_ibs"
                    ],

                "ibs_gain":
                    results[
                        label
                    ][
                        "pre_minus_bounded_patient_ibs"
                    ],

                "ibs_gain_ci":
                    results[
                        label
                    ][
                        "ibs_gain_bootstrap"
                    ],
            }
            for label
            in (
                "w0",
                "w3",
                "w7",
            )
        },

        "robustness_flags":
            robustness_flags,

        "external_validation_outcomes_accessed":
            False,

        "next_action":
            report[
                "next_action"
            ],
    }

    atomic_json(
        root
        / "artifacts"
        / "handoff"
        / "checkpoint_06D.json",
        handoff,
    )

    ###########################################################################
    # PROJECT_STATE only after a completed scientific evaluation.
    ###########################################################################

    state_path = (
        root
        / "PROJECT_STATE.md"
    )

    existing = (
        state_path.read_text(
            encoding="utf-8"
        )
        if state_path.exists()
        else ""
    )

    marker = (
        "# CHECKPOINT 6D — scan grouping robustness"
    )

    if marker not in existing:

        section = f"""
{marker}

Status: `{status}`

- frozen candidate: `PRE + 0.75 × (POST - PRE)`
- no retraining or parameter selection
- W0/W3/W7 temporal histories regenerated with the frozen CKPT4 encoder
- authoritative CKPT1 treatment-line mapping preserved exactly
- primary robustness cohort is exact patient/day/line matched and remains
  NON_PROGRESSIVE under all three grouping windows
- common test rows: {len(common)}
- common test patients: {common["patient_id"].nunique()}
- external DFCI/VICC outcomes remain untouched

Window results:

- W0:
  - NLL gain: {results["w0"]["pre_minus_bounded_patient_nll"]}
  - IBS gain: {results["w0"]["pre_minus_bounded_patient_ibs"]}
- W3:
  - NLL gain: {results["w3"]["pre_minus_bounded_patient_nll"]}
  - IBS gain: {results["w3"]["pre_minus_bounded_patient_ibs"]}
- W7:
  - NLL gain: {results["w7"]["pre_minus_bounded_patient_nll"]}
  - IBS gain: {results["w7"]["pre_minus_bounded_patient_ibs"]}

Next action:

{report["next_action"]}
""".strip()

        atomic_text(
            state_path,
            existing.rstrip()
            + "\n\n"
            + section
            + "\n",
        )

    ###########################################################################
    # Summary.
    ###########################################################################

    print("")
    print(
        "========== CKPT6D2 SUMMARY =========="
    )

    print(
        f"status={status}"
    )

    print(
        "common_test_rows="
        f"{len(common)}"
    )

    print(
        "common_test_patients="
        f"{common['patient_id'].nunique()}"
    )

    print(
        "common_fraction="
        f"{len(common) / len(locked)}"
    )

    for label in (
        "w0",
        "w3",
        "w7",
    ):

        item = results[
            label
        ]

        print(
            f"{label.upper()}_PRE_patient_NLL="
            f"{item['pre_patient_nll']}"
        )

        print(
            f"{label.upper()}_BOUNDED_patient_NLL="
            f"{item['bounded_patient_nll']}"
        )

        print(
            f"{label.upper()}_NLL_gain="
            f"{item['pre_minus_bounded_patient_nll']}"
        )

        print(
            f"{label.upper()}_NLL_gain_CI="
            f"{item['nll_gain_bootstrap']}"
        )

        print(
            f"{label.upper()}_PRE_patient_IBS="
            f"{item['pre_patient_ibs']}"
        )

        print(
            f"{label.upper()}_BOUNDED_patient_IBS="
            f"{item['bounded_patient_ibs']}"
        )

        print(
            f"{label.upper()}_IBS_gain="
            f"{item['pre_minus_bounded_patient_ibs']}"
        )

        print(
            f"{label.upper()}_IBS_gain_CI="
            f"{item['ibs_gain_bootstrap']}"
        )

    print(
        "robustness_flags="
        f"{robustness_flags}"
    )

    print(
        "prediction_sensitivity="
        f"{prediction_sensitivity}"
    )

    print(
        "external_validation_outcomes_accessed=False"
    )

    print(
        "report="
        "artifacts/checkpoint6d/window_robustness.json"
    )

    print(
        "predictions="
        "artifacts/checkpoint6d/window_robustness_predictions.parquet"
    )

    print(
        "handoff="
        "artifacts/handoff/checkpoint_06D.json"
    )

    print(
        "next_action="
        f"{report['next_action']}"
    )

    print(
        "========== CKPT6D2 SUMMARY END =========="
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
