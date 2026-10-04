#!/usr/bin/env python3
"""Exploratory Layer 8 incremental-value baseline screen."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ANALYSIS_VERSION = (
    "oncotwin_layer8_incremental_baseline_screen_v0_1"
)

EVALUATION_CLASS = (
    "exploratory_screening_not_locked_confirmatory"
)

EXPECTED_LANDMARK_SHA256 = (
    "7d47c12ddcdddded13abbcb1ea56512e2f27a8aeb90ef62590ab792c5187d87d"
)

EXPECTED_REPEAT_SHA256 = (
    "9b7f241a8a040632163c3d53b35bc688ce7eb79c377f19d0a5a7974df0cf786a"
)

EXPECTED_OOF_EXPORT_SHA256 = (
    "5b31aea86a0d4e32dec52597901011b36db36e908213642b84484fc052bcddc7"
)

N_REPEATS = 20
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4

BASE_SEED = 820260
BOOTSTRAP_SEED = 820261
BOOTSTRAP_REPLICATES = 5000

RIDGE_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
)

MODEL_NAMES = (
    "no_update",
    "direct_t2",
    "trend_half",
    "trend_full",
    "ridge_incremental",
    "ridge_context",
)

RIDGE_FEATURES = {
    "ridge_incremental": (
        "log_t2_over_t1",
        "log_t2_over_prior",
    ),
    "ridge_context": (
        "log_t2_over_t1",
        "log_t2_over_prior",
        "log_t1_over_baseline",
        "log_prior_ml",
        "prior_cv",
    ),
}

EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predefined T2 update baselines using "
            "repeated Layer 6 OOF priors and nested-CV ridge updates."
        )
    )

    parser.add_argument(
        "--landmark-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repeat-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--oof-export-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--summary-markdown",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repeat-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--case-output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_module(
    name: str,
    path: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create module specification for {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return dict(payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            payload = json.loads(line)

            if not isinstance(payload, Mapping):
                raise TypeError(
                    f"{path}:{line_number} is not a JSON object"
                )

            rows.append(dict(payload))

    return rows


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    text = "".join(
        json.dumps(
            dict(row),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )

    if not text:
        raise RuntimeError(
            f"Refusing to write empty JSONL: {path}"
        )

    atomic_write_text(path, text)


def positive_float(
    value: object,
    *,
    name: str,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number) or number <= 0:
        raise ValueError(
            f"{name} must be finite and positive"
        )

    return number


def finite_float(
    value: object,
    *,
    name: str,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be finite"
        )

    return number


def quantile(
    values: Sequence[float],
    probability: float,
) -> float:
    if not values:
        raise ValueError(
            "Cannot take a quantile of an empty sequence"
        )

    return float(
        np.quantile(
            np.asarray(values, dtype=float),
            probability,
        )
    )


def summarize_values(
    values: Sequence[float],
) -> dict[str, float | int]:
    data = [float(value) for value in values]

    if not data:
        raise ValueError(
            "Cannot summarize an empty sequence"
        )

    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "sd": (
            statistics.stdev(data)
            if len(data) > 1
            else 0.0
        ),
        "p10": quantile(data, 0.10),
        "p90": quantile(data, 0.90),
        "minimum": min(data),
        "maximum": max(data),
    }


def prediction_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    observed = np.asarray(
        observed,
        dtype=float,
    )
    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if observed.shape != predicted.shape:
        raise ValueError(
            "Observed and predicted shapes differ"
        )

    if observed.ndim != 1 or observed.size == 0:
        raise ValueError(
            "Expected nonempty one-dimensional arrays"
        )

    if not np.all(np.isfinite(observed)):
        raise ValueError(
            "Observed values contain non-finite entries"
        )

    if not np.all(np.isfinite(predicted)):
        raise ValueError(
            "Predictions contain non-finite entries"
        )

    if np.any(observed <= 0) or np.any(predicted <= 0):
        raise ValueError(
            "Observed values and predictions must be positive"
        )

    errors = predicted - observed
    log_errors = np.log(predicted) - np.log(observed)

    return {
        "n": int(observed.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(
            np.sqrt(np.mean(np.square(errors)))
        ),
        "log_rmse": float(
            np.sqrt(np.mean(np.square(log_errors)))
        ),
        "bias": float(np.mean(errors)),
    }


def pearson(
    left: np.ndarray,
    right: np.ndarray,
) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    if left.size != right.size or left.size < 2:
        return None

    if np.std(left) <= 0 or np.std(right) <= 0:
        return None

    return float(
        np.corrcoef(left, right)[0, 1]
    )


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    features = np.asarray(
        features,
        dtype=float,
    )
    target = np.asarray(
        target,
        dtype=float,
    )

    if features.ndim != 2:
        raise ValueError(
            "Ridge features must be two-dimensional"
        )

    if target.ndim != 1:
        raise ValueError(
            "Ridge target must be one-dimensional"
        )

    if features.shape[0] != target.size:
        raise ValueError(
            "Ridge feature and target row counts differ"
        )

    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)

    scales = np.where(
        scales > 1e-12,
        scales,
        1.0,
    )

    standardized = (
        features - means
    ) / scales

    design = np.column_stack(
        [
            np.ones(features.shape[0]),
            standardized,
        ]
    )

    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0

    system = (
        design.T @ design
        + penalty
    )
    right = design.T @ target

    try:
        coefficients = np.linalg.solve(
            system,
            right,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(
            system
        ) @ right

    return {
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "alpha": float(alpha),
    }


def predict_ridge(
    fitted: Mapping[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    features = np.asarray(
        features,
        dtype=float,
    )

    means = np.asarray(
        fitted["means"],
        dtype=float,
    )
    scales = np.asarray(
        fitted["scales"],
        dtype=float,
    )
    coefficients = np.asarray(
        fitted["coefficients"],
        dtype=float,
    )

    standardized = (
        features - means
    ) / scales

    design = np.column_stack(
        [
            np.ones(features.shape[0]),
            standardized,
        ]
    )

    return design @ coefficients


def select_ridge_alpha(
    *,
    feature_matrix: np.ndarray,
    residual_target: np.ndarray,
    fold_records: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    locked_runner: ModuleType,
    seed: int,
) -> tuple[float, dict[str, float]]:
    inner_folds = locked_runner.make_folds(
        fold_records,
        list(train_indices),
        N_INNER_FOLDS,
        seed,
    )

    flattened = [
        index
        for fold in inner_folds
        for index in fold
    ]

    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != set(train_indices)
    ):
        raise RuntimeError(
            "Inner folds do not cover the outer-training cases exactly"
        )

    scores: dict[str, float] = {}

    for alpha in RIDGE_ALPHAS:
        inner_predictions: dict[int, float] = {}

        for validation_indices in inner_folds:
            if not validation_indices:
                raise RuntimeError(
                    "Encountered an empty inner validation fold"
                )

            validation_set = set(validation_indices)

            fit_indices = [
                index
                for index in train_indices
                if index not in validation_set
            ]

            fitted = fit_ridge(
                feature_matrix[fit_indices],
                residual_target[fit_indices],
                alpha,
            )

            predicted = predict_ridge(
                fitted,
                feature_matrix[validation_indices],
            )

            for position, index in enumerate(
                validation_indices
            ):
                inner_predictions[index] = float(
                    predicted[position]
                )

        ordered_prediction = np.asarray(
            [
                inner_predictions[index]
                for index in train_indices
            ],
            dtype=float,
        )
        ordered_target = residual_target[
            list(train_indices)
        ]

        score = float(
            np.sqrt(
                np.mean(
                    np.square(
                        ordered_prediction
                        - ordered_target
                    )
                )
            )
        )

        scores[str(alpha)] = score

    minimum_score = min(scores.values())

    eligible = [
        alpha
        for alpha in RIDGE_ALPHAS
        if scores[str(alpha)]
        <= minimum_score + 1e-12
    ]

    # Conservative tie break: select stronger regularization.
    selected = max(eligible)

    return float(selected), scores


def bootstrap_comparison(
    *,
    observed: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)

    rng = np.random.default_rng(seed)
    case_count = observed.size

    delta_mae_values: list[float] = []
    delta_rmse_values: list[float] = []
    delta_log_rmse_values: list[float] = []

    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(
            0,
            case_count,
            size=case_count,
        )

        y = observed[indices]
        base = baseline[indices]
        cand = candidate[indices]

        delta_mae_values.append(
            float(
                np.mean(np.abs(cand - y))
                - np.mean(np.abs(base - y))
            )
        )

        delta_rmse_values.append(
            float(
                np.sqrt(
                    np.mean(np.square(cand - y))
                )
                - np.sqrt(
                    np.mean(np.square(base - y))
                )
            )
        )

        delta_log_rmse_values.append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.log(cand)
                            - np.log(y)
                        )
                    )
                )
                - np.sqrt(
                    np.mean(
                        np.square(
                            np.log(base)
                            - np.log(y)
                        )
                    )
                )
            )
        )

    absolute_error_delta = (
        np.abs(candidate - observed)
        - np.abs(baseline - observed)
    )

    tolerance = 1e-12

    wins = int(
        np.sum(absolute_error_delta < -tolerance)
    )
    losses = int(
        np.sum(absolute_error_delta > tolerance)
    )
    ties = int(case_count - wins - losses)

    baseline_metrics = prediction_metrics(
        observed,
        baseline,
    )
    candidate_metrics = prediction_metrics(
        observed,
        candidate,
    )

    return {
        "interpretation": (
            "Negative metric deltas favor the candidate. "
            "Intervals are exploratory paired case-bootstrap "
            "intervals without multiplicity correction."
        ),
        "case_count": int(case_count),
        "candidate_minus_no_update": {
            "mae": (
                float(candidate_metrics["mae"])
                - float(baseline_metrics["mae"])
            ),
            "rmse": (
                float(candidate_metrics["rmse"])
                - float(baseline_metrics["rmse"])
            ),
            "log_rmse": (
                float(candidate_metrics["log_rmse"])
                - float(baseline_metrics["log_rmse"])
            ),
        },
        "bootstrap_95_ci": {
            "delta_mae": [
                quantile(delta_mae_values, 0.025),
                quantile(delta_mae_values, 0.975),
            ],
            "delta_rmse": [
                quantile(delta_rmse_values, 0.025),
                quantile(delta_rmse_values, 0.975),
            ],
            "delta_log_rmse": [
                quantile(
                    delta_log_rmse_values,
                    0.025,
                ),
                quantile(
                    delta_log_rmse_values,
                    0.975,
                ),
            ],
        },
        "absolute_error_comparison": {
            "candidate_better": wins,
            "candidate_worse": losses,
            "tie": ties,
        },
    }


def main() -> None:
    args = parse_args()

    hashes = {
        "landmark_input": sha256_file(
            args.landmark_input
        ),
        "repeat_input": sha256_file(
            args.repeat_input
        ),
        "oof_export_json": sha256_file(
            args.oof_export_json
        ),
    }

    expected = {
        "landmark_input": EXPECTED_LANDMARK_SHA256,
        "repeat_input": EXPECTED_REPEAT_SHA256,
        "oof_export_json": EXPECTED_OOF_EXPORT_SHA256,
    }

    for name, expected_hash in expected.items():
        if hashes[name] != expected_hash:
            raise RuntimeError(
                f"{name} hash mismatch: "
                f"expected={expected_hash} "
                f"actual={hashes[name]}"
            )

    oof_export = load_json_object(
        args.oof_export_json
    )

    if not oof_export["locked_reproduction"][
        "all_repeats_reproduced"
    ]:
        raise RuntimeError(
            "Layer 6 OOF export did not reproduce all locked repeats"
        )

    landmark_rows = load_jsonl(
        args.landmark_input
    )
    repeat_input_rows = load_jsonl(
        args.repeat_input
    )

    if len(landmark_rows) != 220:
        raise RuntimeError(
            f"Expected 220 landmark rows, got {len(landmark_rows)}"
        )

    landmark_by_case = {
        str(row["case_id"]): row
        for row in landmark_rows
    }

    if len(landmark_by_case) != 220:
        raise RuntimeError(
            "Landmark input contains duplicate case IDs"
        )

    relevant_repeat_rows = [
        row
        for row in repeat_input_rows
        if str(row["case_id"]) in landmark_by_case
    ]

    if len(relevant_repeat_rows) != 4400:
        raise RuntimeError(
            "Expected 4400 landmark repeat predictions, "
            f"got {len(relevant_repeat_rows)}"
        )

    repeat_rows_by_number: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    case_prior_values: dict[
        str,
        list[float],
    ] = defaultdict(list)

    for row in relevant_repeat_rows:
        repeat_number = int(row["repeat_number"])
        repeat_rows_by_number[repeat_number].append(row)

        case_id = str(row["case_id"])
        case_prior_values[case_id].append(
            positive_float(
                row["layer6_oof_point_ml"],
                name=(
                    f"{case_id}.repeat[{repeat_number}]."
                    "layer6_oof_point_ml"
                ),
            )
        )

    if sorted(repeat_rows_by_number) != list(
        range(1, N_REPEATS + 1)
    ):
        raise RuntimeError(
            "Repeat input does not contain repeats 1 through 20"
        )

    maximum_prior_mean_relative_error = 0.0

    for case_id, values in case_prior_values.items():
        if len(values) != N_REPEATS:
            raise RuntimeError(
                f"Case {case_id} has {len(values)} repeat priors"
            )

        exported_mean = positive_float(
            landmark_by_case[case_id][
                "layer6_oof_prior_point_ml"
            ],
            name=(
                f"{case_id}.layer6_oof_prior_point_ml"
            ),
        )
        calculated_mean = statistics.fmean(values)

        error = abs(
            calculated_mean - exported_mean
        ) / max(abs(exported_mean), EPSILON)

        maximum_prior_mean_relative_error = max(
            maximum_prior_mean_relative_error,
            error,
        )

    if maximum_prior_mean_relative_error > 1e-12:
        raise RuntimeError(
            "Repeat-prior mean does not reproduce merged landmark prior: "
            f"max_relative_error={maximum_prior_mean_relative_error}"
        )

    locked_runner = load_module(
        "_oncotwin_layer8_locked_splitter",
        (
            REPO_ROOT
            / "scripts/evals/run_layer6_locked_confirmatory.py"
        ),
    )

    repeat_output_rows: list[dict[str, Any]] = []
    repeat_results: list[dict[str, Any]] = []

    alpha_selection_counts: dict[
        str,
        Counter[float],
    ] = {
        name: Counter()
        for name in RIDGE_FEATURES
    }

    alpha_score_records: list[dict[str, Any]] = []

    for repeat_number in range(
        1,
        N_REPEATS + 1,
    ):
        source_rows = sorted(
            repeat_rows_by_number[repeat_number],
            key=lambda row: str(row["case_id"]),
        )

        if len(source_rows) != 220:
            raise RuntimeError(
                f"Repeat {repeat_number} has "
                f"{len(source_rows)} landmark rows"
            )

        case_ids = [
            str(row["case_id"])
            for row in source_rows
        ]

        if len(case_ids) != len(set(case_ids)):
            raise RuntimeError(
                f"Repeat {repeat_number} contains duplicate cases"
            )

        baseline = np.asarray(
            [
                positive_float(
                    landmark_by_case[case_id][
                        "baseline_volume_ml"
                    ],
                    name=f"{case_id}.baseline_volume_ml",
                )
                for case_id in case_ids
            ],
            dtype=float,
        )

        t1 = np.asarray(
            [
                positive_float(
                    landmark_by_case[case_id][
                        "t1_early_volume_ml"
                    ],
                    name=f"{case_id}.t1_early_volume_ml",
                )
                for case_id in case_ids
            ],
            dtype=float,
        )

        t2 = np.asarray(
            [
                positive_float(
                    landmark_by_case[case_id][
                        "t2_intermediate_volume_ml"
                    ],
                    name=(
                        f"{case_id}."
                        "t2_intermediate_volume_ml"
                    ),
                )
                for case_id in case_ids
            ],
            dtype=float,
        )

        observed = np.asarray(
            [
                positive_float(
                    landmark_by_case[case_id][
                        "t3_final_volume_ml"
                    ],
                    name=f"{case_id}.t3_final_volume_ml",
                )
                for case_id in case_ids
            ],
            dtype=float,
        )

        prior = np.asarray(
            [
                positive_float(
                    row["layer6_oof_point_ml"],
                    name=(
                        f"{row['case_id']}."
                        "layer6_oof_point_ml"
                    ),
                )
                for row in source_rows
            ],
            dtype=float,
        )

        prior_cross_repeat_sd = np.asarray(
            [
                finite_float(
                    landmark_by_case[case_id][
                        "layer6_oof_prior_sd_ml"
                    ],
                    name=(
                        f"{case_id}."
                        "layer6_oof_prior_sd_ml"
                    ),
                )
                for case_id in case_ids
            ],
            dtype=float,
        )

        final_day = np.asarray(
            [
                finite_float(
                    landmark_by_case[case_id][
                        "t3_day"
                    ],
                    name=f"{case_id}.t3_day",
                )
                for case_id in case_ids
            ],
            dtype=float,
        )

        outer_fold_indices = np.asarray(
            [
                int(row["outer_fold_index"])
                for row in source_rows
            ],
            dtype=int,
        )

        if set(outer_fold_indices.tolist()) != set(
            range(N_OUTER_FOLDS)
        ):
            raise RuntimeError(
                f"Repeat {repeat_number} does not contain all outer folds"
            )

        fold_records = [
            {
                "baseline_volume_ml": float(
                    baseline[index]
                ),
                "early_volume_ml": float(t1[index]),
                "final_day": float(final_day[index]),
            }
            for index in range(len(case_ids))
        ]

        log_t2_over_t1 = np.log(t2 / t1)
        log_t2_over_prior = np.log(t2 / prior)
        log_t1_over_baseline = np.log(
            t1 / baseline
        )
        log_prior_ml = np.log(prior)
        prior_cv = prior_cross_repeat_sd / np.maximum(
            prior,
            EPSILON,
        )

        feature_values = {
            "log_t2_over_t1": log_t2_over_t1,
            "log_t2_over_prior": log_t2_over_prior,
            "log_t1_over_baseline": (
                log_t1_over_baseline
            ),
            "log_prior_ml": log_prior_ml,
            "prior_cv": prior_cv,
        }

        feature_matrices = {
            model_name: np.column_stack(
                [
                    feature_values[feature_name]
                    for feature_name
                    in RIDGE_FEATURES[model_name]
                ]
            )
            for model_name in RIDGE_FEATURES
        }

        residual_target = (
            np.log(observed)
            - np.log(prior)
        )

        predictions = {
            "no_update": prior.copy(),
            "direct_t2": t2.copy(),
            "trend_half": (
                prior
                * np.exp(
                    0.5 * log_t2_over_t1
                )
            ),
            "trend_full": (
                prior
                * np.exp(log_t2_over_t1)
            ),
            "ridge_incremental": np.full(
                observed.shape,
                np.nan,
                dtype=float,
            ),
            "ridge_context": np.full(
                observed.shape,
                np.nan,
                dtype=float,
            ),
        }

        selected_alpha_by_case: dict[
            str,
            np.ndarray,
        ] = {
            model_name: np.full(
                observed.shape,
                np.nan,
                dtype=float,
            )
            for model_name in RIDGE_FEATURES
        }

        for outer_fold_index in range(
            N_OUTER_FOLDS
        ):
            test_indices = np.flatnonzero(
                outer_fold_indices
                == outer_fold_index
            ).tolist()

            train_indices = np.flatnonzero(
                outer_fold_indices
                != outer_fold_index
            ).tolist()

            if not test_indices or not train_indices:
                raise RuntimeError(
                    f"Repeat {repeat_number} fold "
                    f"{outer_fold_index} is empty"
                )

            for model_offset, model_name in enumerate(
                RIDGE_FEATURES,
                start=1,
            ):
                selected_alpha, alpha_scores = (
                    select_ridge_alpha(
                        feature_matrix=(
                            feature_matrices[model_name]
                        ),
                        residual_target=residual_target,
                        fold_records=fold_records,
                        train_indices=train_indices,
                        locked_runner=locked_runner,
                        seed=(
                            BASE_SEED
                            + repeat_number * 10000
                            + outer_fold_index * 100
                            + model_offset
                        ),
                    )
                )

                alpha_selection_counts[
                    model_name
                ][selected_alpha] += 1

                alpha_score_records.append(
                    {
                        "repeat_number": repeat_number,
                        "outer_fold_index": (
                            outer_fold_index
                        ),
                        "model": model_name,
                        "selected_alpha": (
                            selected_alpha
                        ),
                        "inner_log_rmse_by_alpha": (
                            alpha_scores
                        ),
                    }
                )

                fitted = fit_ridge(
                    feature_matrices[model_name][
                        train_indices
                    ],
                    residual_target[train_indices],
                    selected_alpha,
                )

                predicted_residual = predict_ridge(
                    fitted,
                    feature_matrices[model_name][
                        test_indices
                    ],
                )

                # Clip only to avoid floating-point overflow.
                predicted_residual = np.clip(
                    predicted_residual,
                    -20.0,
                    20.0,
                )

                predicted_volume = (
                    prior[test_indices]
                    * np.exp(predicted_residual)
                )

                predictions[model_name][
                    test_indices
                ] = predicted_volume

                selected_alpha_by_case[model_name][
                    test_indices
                ] = selected_alpha

        for model_name in MODEL_NAMES:
            if not np.all(
                np.isfinite(predictions[model_name])
            ):
                raise RuntimeError(
                    f"Repeat {repeat_number} model "
                    f"{model_name} has non-finite predictions"
                )

            if np.any(predictions[model_name] <= 0):
                raise RuntimeError(
                    f"Repeat {repeat_number} model "
                    f"{model_name} has nonpositive predictions"
                )

        metrics_by_model = {
            model_name: prediction_metrics(
                observed,
                predictions[model_name],
            )
            for model_name in MODEL_NAMES
        }

        repeat_results.append(
            {
                "repeat_number": repeat_number,
                "outer_fold_sizes": {
                    str(fold_index): int(
                        np.sum(
                            outer_fold_indices
                            == fold_index
                        )
                    )
                    for fold_index in range(
                        N_OUTER_FOLDS
                    )
                },
                "metrics": metrics_by_model,
            }
        )

        for index, case_id in enumerate(case_ids):
            repeat_output_rows.append(
                {
                    "analysis_version": (
                        ANALYSIS_VERSION
                    ),
                    "evaluation_class": (
                        EVALUATION_CLASS
                    ),
                    "case_id": case_id,
                    "repeat_number": repeat_number,
                    "outer_fold_index": int(
                        outer_fold_indices[index]
                    ),
                    "held_out_from_layer8_training": True,
                    "layer6_prior_held_out_for_own_case": True,
                    "prior_information_cutoff": "T1",
                    "new_evidence": "T2",
                    "held_out_target": "T3",
                    "baseline_volume_ml": float(
                        baseline[index]
                    ),
                    "t1_volume_ml": float(t1[index]),
                    "t2_volume_ml": float(t2[index]),
                    "observed_t3_volume_ml": float(
                        observed[index]
                    ),
                    "layer6_oof_prior_ml": float(
                        prior[index]
                    ),
                    "prediction_no_update_ml": float(
                        predictions["no_update"][index]
                    ),
                    "prediction_direct_t2_ml": float(
                        predictions["direct_t2"][index]
                    ),
                    "prediction_trend_half_ml": float(
                        predictions["trend_half"][index]
                    ),
                    "prediction_trend_full_ml": float(
                        predictions["trend_full"][index]
                    ),
                    "prediction_ridge_incremental_ml": float(
                        predictions[
                            "ridge_incremental"
                        ][index]
                    ),
                    "prediction_ridge_context_ml": float(
                        predictions[
                            "ridge_context"
                        ][index]
                    ),
                    "ridge_incremental_alpha": float(
                        selected_alpha_by_case[
                            "ridge_incremental"
                        ][index]
                    ),
                    "ridge_context_alpha": float(
                        selected_alpha_by_case[
                            "ridge_context"
                        ][index]
                    ),
                    "target_used_as_feature": False,
                    "deployable_full_fit_prior_used": False,
                }
            )

        print(
            "BASELINE_REPEAT_COMPLETE "
            f"repeat={repeat_number}/{N_REPEATS} "
            f"no_update_log_rmse="
            f"{metrics_by_model['no_update']['log_rmse']:.8f} "
            f"ridge_incremental_log_rmse="
            f"{metrics_by_model['ridge_incremental']['log_rmse']:.8f} "
            f"ridge_context_log_rmse="
            f"{metrics_by_model['ridge_context']['log_rmse']:.8f}",
            flush=True,
        )

    expected_repeat_rows = 220 * N_REPEATS

    if len(repeat_output_rows) != expected_repeat_rows:
        raise RuntimeError(
            "Unexpected repeat-output row count: "
            f"actual={len(repeat_output_rows)} "
            f"expected={expected_repeat_rows}"
        )

    per_case_model_values: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in repeat_output_rows:
        case_id = str(row["case_id"])

        for model_name in MODEL_NAMES:
            per_case_model_values[case_id][
                model_name
            ].append(
                float(
                    row[
                        f"prediction_{model_name}_ml"
                    ]
                )
            )

    case_output_rows: list[dict[str, Any]] = []

    for case_id in sorted(landmark_by_case):
        source = landmark_by_case[case_id]
        row: dict[str, Any] = {
            "analysis_version": ANALYSIS_VERSION,
            "evaluation_class": EVALUATION_CLASS,
            "case_id": case_id,
            "observed_t3_volume_ml": positive_float(
                source["t3_final_volume_ml"],
                name=f"{case_id}.t3_final_volume_ml",
            ),
            "prior_information_cutoff": "T1",
            "new_evidence": "T2",
            "held_out_target": "T3",
            "repeat_prediction_count": N_REPEATS,
        }

        for model_name in MODEL_NAMES:
            values = per_case_model_values[
                case_id
            ][model_name]

            if len(values) != N_REPEATS:
                raise RuntimeError(
                    f"Case {case_id} model {model_name} "
                    f"has {len(values)} predictions"
                )

            summary = summarize_values(values)

            row[
                f"{model_name}_mean_prediction_ml"
            ] = float(summary["mean"])
            row[
                f"{model_name}_prediction_sd_ml"
            ] = float(summary["sd"])
            row[
                f"{model_name}_minimum_prediction_ml"
            ] = float(summary["minimum"])
            row[
                f"{model_name}_maximum_prediction_ml"
            ] = float(summary["maximum"])

        case_output_rows.append(row)

    observed_case = np.asarray(
        [
            row["observed_t3_volume_ml"]
            for row in case_output_rows
        ],
        dtype=float,
    )

    aggregate_predictions = {
        model_name: np.asarray(
            [
                row[
                    f"{model_name}_mean_prediction_ml"
                ]
                for row in case_output_rows
            ],
            dtype=float,
        )
        for model_name in MODEL_NAMES
    }

    aggregate_metrics = {
        model_name: prediction_metrics(
            observed_case,
            aggregate_predictions[model_name],
        )
        for model_name in MODEL_NAMES
    }

    repeat_metric_summary: dict[
        str,
        dict[str, Any],
    ] = {}

    for model_name in MODEL_NAMES:
        repeat_metric_summary[model_name] = {}

        for metric_name in (
            "mae",
            "rmse",
            "log_rmse",
            "bias",
        ):
            values = [
                float(
                    repeat_result["metrics"][
                        model_name
                    ][metric_name]
                )
                for repeat_result in repeat_results
            ]

            repeat_metric_summary[model_name][
                metric_name
            ] = summarize_values(values)

    comparisons: dict[str, Any] = {}

    for model_index, model_name in enumerate(
        MODEL_NAMES
    ):
        if model_name == "no_update":
            continue

        comparisons[model_name] = (
            bootstrap_comparison(
                observed=observed_case,
                baseline=aggregate_predictions[
                    "no_update"
                ],
                candidate=aggregate_predictions[
                    model_name
                ],
                seed=(
                    BOOTSTRAP_SEED
                    + model_index * 10000
                ),
            )
        )

    source_case_order = [
        row["case_id"]
        for row in case_output_rows
    ]

    mean_prior = np.asarray(
        [
            positive_float(
                landmark_by_case[case_id][
                    "layer6_oof_prior_point_ml"
                ],
                name=(
                    f"{case_id}."
                    "layer6_oof_prior_point_ml"
                ),
            )
            for case_id in source_case_order
        ],
        dtype=float,
    )

    t1_case = np.asarray(
        [
            positive_float(
                landmark_by_case[case_id][
                    "t1_early_volume_ml"
                ],
                name=f"{case_id}.t1_early_volume_ml",
            )
            for case_id in source_case_order
        ],
        dtype=float,
    )

    t2_case = np.asarray(
        [
            positive_float(
                landmark_by_case[case_id][
                    "t2_intermediate_volume_ml"
                ],
                name=(
                    f"{case_id}."
                    "t2_intermediate_volume_ml"
                ),
            )
            for case_id in source_case_order
        ],
        dtype=float,
    )

    residual_case = (
        np.log(observed_case)
        - np.log(mean_prior)
    )

    signal_diagnostics = {
        "pearson_log_t2_over_t1_vs_log_target_residual": (
            pearson(
                np.log(t2_case / t1_case),
                residual_case,
            )
        ),
        "pearson_log_t2_over_prior_vs_log_target_residual": (
            pearson(
                np.log(t2_case / mean_prior),
                residual_case,
            )
        ),
        "interpretation": (
            "These are descriptive full-cohort correlations "
            "used only to assess incremental T2 signal."
        ),
    }

    candidate_models = [
        model_name
        for model_name in MODEL_NAMES
        if model_name != "no_update"
    ]

    best_model = min(
        candidate_models,
        key=lambda model_name: (
            float(
                aggregate_metrics[model_name][
                    "log_rmse"
                ]
            ),
            float(
                aggregate_metrics[model_name][
                    "mae"
                ]
            ),
        ),
    )

    best_comparison = comparisons[best_model]
    best_deltas = best_comparison[
        "candidate_minus_no_update"
    ]
    best_cis = best_comparison[
        "bootstrap_95_ci"
    ]

    both_point_metrics_improve = (
        float(best_deltas["mae"]) < 0
        and float(best_deltas["log_rmse"]) < 0
    )

    at_least_one_interval_excludes_zero = (
        float(best_cis["delta_mae"][1]) < 0
        or float(
            best_cis["delta_log_rmse"][1]
        ) < 0
    )

    if (
        both_point_metrics_improve
        and at_least_one_interval_excludes_zero
    ):
        decision = (
            "incremental_t2_signal_detected_"
            "proceed_to_uncertainty_and_ablation"
        )
    elif both_point_metrics_improve:
        decision = (
            "weak_incremental_t2_signal_"
            "continue_with_ablation_before_escalation"
        )
    else:
        decision = (
            "no_reliable_incremental_t2_gain_"
            "do_not_escalate_model_complexity"
        )

    repeat_output_rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            int(row["repeat_number"]),
        )
    )

    case_output_rows.sort(
        key=lambda row: str(row["case_id"])
    )

    atomic_write_jsonl(
        args.repeat_output,
        repeat_output_rows,
    )
    atomic_write_jsonl(
        args.case_output,
        case_output_rows,
    )

    output_hashes = {
        str(args.repeat_output): sha256_file(
            args.repeat_output
        ),
        str(args.case_output): sha256_file(
            args.case_output
        ),
    }

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "exploratory_incremental_baseline_screen"
        ),
        "evaluation_class": EVALUATION_CLASS,
        "inputs": {
            "landmark_input": str(
                args.landmark_input
            ),
            "landmark_input_sha256": hashes[
                "landmark_input"
            ],
            "repeat_input": str(
                args.repeat_input
            ),
            "repeat_input_sha256": hashes[
                "repeat_input"
            ],
            "oof_export_json": str(
                args.oof_export_json
            ),
            "oof_export_json_sha256": hashes[
                "oof_export_json"
            ],
            "landmark_case_count": 220,
            "repeat_count": N_REPEATS,
            "relevant_repeat_row_count": len(
                relevant_repeat_rows
            ),
            "maximum_prior_mean_relative_error": (
                maximum_prior_mean_relative_error
            ),
        },
        "temporal_design": {
            "layer6_prior_information_cutoff": "T1",
            "new_update_evidence": "T2",
            "held_out_target": "T3",
            "t1_reused_as_update_evidence": False,
            "target_used_as_feature": False,
            "deployable_full_fit_prior_used": False,
        },
        "models": {
            "no_update": {
                "type": "fixed",
                "definition": (
                    "unchanged repeated-OOF Layer 6 prior"
                ),
            },
            "direct_t2": {
                "type": "fixed",
                "definition": (
                    "T2 enhancing volume carried directly "
                    "forward as the T3 prediction"
                ),
            },
            "trend_half": {
                "type": "fixed",
                "definition": (
                    "Layer 6 prior multiplied by "
                    "(T2/T1)^0.5"
                ),
            },
            "trend_full": {
                "type": "fixed",
                "definition": (
                    "Layer 6 prior multiplied by T2/T1"
                ),
            },
            "ridge_incremental": {
                "type": (
                    "nested_cv_log_residual_ridge"
                ),
                "target": (
                    "log(T3) - log(Layer6 prior)"
                ),
                "features": list(
                    RIDGE_FEATURES[
                        "ridge_incremental"
                    ]
                ),
                "candidate_alphas": list(
                    RIDGE_ALPHAS
                ),
            },
            "ridge_context": {
                "type": (
                    "nested_cv_log_residual_ridge"
                ),
                "target": (
                    "log(T3) - log(Layer6 prior)"
                ),
                "features": list(
                    RIDGE_FEATURES[
                        "ridge_context"
                    ]
                ),
                "candidate_alphas": list(
                    RIDGE_ALPHAS
                ),
            },
        },
        "cross_validation": {
            "outer_repeats": N_REPEATS,
            "outer_folds": N_OUTER_FOLDS,
            "outer_fold_source": (
                "exact fold assignment carried by each "
                "Layer 6 repeated-OOF prediction"
            ),
            "inner_folds": N_INNER_FOLDS,
            "inner_splitter": (
                "locked Layer 6 response-category and "
                "final-day stratified splitter"
            ),
            "ridge_selection_metric": (
                "inner-OOF log residual RMSE"
            ),
            "ridge_tie_break": (
                "largest alpha within 1e-12 of minimum"
            ),
        },
        "repeat_metrics": repeat_results,
        "repeat_metric_summary": (
            repeat_metric_summary
        ),
        "aggregate_case_mean_metrics": (
            aggregate_metrics
        ),
        "paired_comparisons_vs_no_update": (
            comparisons
        ),
        "ridge_alpha_selection_counts": {
            model_name: {
                str(alpha): count
                for alpha, count
                in sorted(counts.items())
            }
            for model_name, counts
            in alpha_selection_counts.items()
        },
        "ridge_inner_score_records": (
            alpha_score_records
        ),
        "signal_diagnostics": signal_diagnostics,
        "outputs": {
            "repeat_predictions": {
                "path": str(args.repeat_output),
                "row_count": len(
                    repeat_output_rows
                ),
                "sha256": output_hashes[
                    str(args.repeat_output)
                ],
            },
            "case_summary": {
                "path": str(args.case_output),
                "row_count": len(
                    case_output_rows
                ),
                "sha256": output_hashes[
                    str(args.case_output)
                ],
            },
        },
        "exploratory_selection": {
            "best_candidate_by_aggregate_log_rmse": (
                best_model
            ),
            "best_candidate_comparison": (
                best_comparison
            ),
            "multiplicity_adjusted": False,
            "may_be_used_as_locked_confirmation": False,
        },
        "stacking_limitation": {
            "strict_full_stack_nesting": False,
            "reason": (
                "Each patient's Layer 6 prior excludes that "
                "patient's target, but the base learner was not "
                "refitted wholly inside each Layer 8 outer-training "
                "partition. This is acceptable for exploratory "
                "signal screening, not for a final locked stacked "
                "evaluation."
            ),
            "required_for_final_lock": (
                "Refit or replay the complete Layer 6 base learner "
                "inside each final Layer 8 outer-training partition."
            ),
        },
        "decision": {
            "status": decision,
            "best_exploratory_candidate": (
                best_model
            ),
            "next_step": (
                "Run feature and model ablations, then evaluate "
                "uncertainty calibration only if incremental signal "
                "is supported."
            ),
        },
    }

    atomic_write_json(
        args.summary_json,
        payload,
    )

    report_lines = [
        "# Layer 8 incremental baseline screen",
        "",
        f"- Evaluation class: `{EVALUATION_CLASS}`",
        "- Landmark cases: `220`",
        "- Outer evaluation: `20 repeats x 5 folds`",
        "- Ridge tuning: `4-fold inner CV`",
        (
            "- Best exploratory candidate: "
            f"`{best_model}`"
        ),
        f"- Decision: `{decision}`",
        "",
        "## Aggregate case-mean metrics",
        "",
    ]

    for model_name in MODEL_NAMES:
        result = aggregate_metrics[model_name]

        report_lines.append(
            (
                f"- `{model_name}`: "
                f"MAE={result['mae']:.8f}, "
                f"RMSE={result['rmse']:.8f}, "
                f"log RMSE={result['log_rmse']:.8f}, "
                f"bias={result['bias']:.8f}"
            )
        )

    report_lines.extend(
        [
            "",
            "## Temporal boundary",
            "",
            "- Prior information cutoff: `T1`",
            "- New evidence: `T2`",
            "- Held-out target: `T3`",
            "- T1 reused as update evidence: `false`",
            "",
            "## Interpretation constraint",
            "",
            (
                "This is an exploratory screening evaluation. "
                "Candidate selection and bootstrap comparisons "
                "use the same 220-case development cohort and are "
                "not independent confirmation."
            ),
            "",
            "## Final-stack requirement",
            "",
            (
                "A locked evaluation must refit or replay the "
                "complete Layer 6 base learner within every Layer 8 "
                "outer-training partition."
            ),
        ]
    )

    atomic_write_text(
        args.summary_markdown,
        "\n".join(report_lines) + "\n",
    )

    print(
        "BASELINE_SCREEN_COUNTS "
        f"cases={len(case_output_rows)} "
        f"repeats={N_REPEATS} "
        f"outer_folds={N_OUTER_FOLDS} "
        f"inner_folds={N_INNER_FOLDS} "
        f"repeat_rows={len(repeat_output_rows)}"
    )

    print(
        "PRIOR_MEAN_REPRODUCTION "
        f"max_relative_error="
        f"{maximum_prior_mean_relative_error:.12g} "
        "status=pass"
    )

    for model_name in MODEL_NAMES:
        result = aggregate_metrics[model_name]

        print(
            "AGGREGATE_BASELINE_METRICS "
            f"model={model_name} "
            f"n={result['n']} "
            f"mae={result['mae']} "
            f"rmse={result['rmse']} "
            f"log_rmse={result['log_rmse']} "
            f"bias={result['bias']}"
        )

    for model_name, comparison in comparisons.items():
        deltas = comparison[
            "candidate_minus_no_update"
        ]
        intervals = comparison[
            "bootstrap_95_ci"
        ]
        counts = comparison[
            "absolute_error_comparison"
        ]

        print(
            "PAIRED_BASELINE_COMPARISON "
            f"model={model_name} "
            f"delta_mae={deltas['mae']} "
            f"delta_mae_ci="
            f"{intervals['delta_mae']} "
            f"delta_log_rmse="
            f"{deltas['log_rmse']} "
            f"delta_log_rmse_ci="
            f"{intervals['delta_log_rmse']} "
            f"better={counts['candidate_better']} "
            f"worse={counts['candidate_worse']} "
            f"ties={counts['tie']}"
        )

    for model_name, counts in (
        alpha_selection_counts.items()
    ):
        print(
            "RIDGE_ALPHA_SELECTION "
            f"model={model_name} "
            f"counts="
            f"{dict(sorted(counts.items()))}"
        )

    print(
        "T2_SIGNAL_DIAGNOSTICS "
        f"log_t2_over_t1_correlation="
        f"{signal_diagnostics['pearson_log_t2_over_t1_vs_log_target_residual']} "
        f"log_t2_over_prior_correlation="
        f"{signal_diagnostics['pearson_log_t2_over_prior_vs_log_target_residual']}"
    )

    print(
        "LAYER8_BASELINE_SCREEN_DECISION "
        f"status={decision} "
        f"best_candidate={best_model}"
    )

    print(
        "LAYER8_EVALUATION_CLASS "
        "value=exploratory_screening_not_locked_confirmatory"
    )

    print(
        "LAYER8_STACKING_GUARD "
        "strict_full_stack_nesting=false "
        "locked_use_allowed=false"
    )

    print(
        f"BASELINE_SCREEN_JSON path={args.summary_json}"
    )
    print(
        "BASELINE_SCREEN_REPORT "
        f"path={args.summary_markdown}"
    )
    print(
        "BASELINE_REPEAT_OUTPUT "
        f"path={args.repeat_output}"
    )
    print(
        f"BASELINE_CASE_OUTPUT path={args.case_output}"
    )


if __name__ == "__main__":
    main()
