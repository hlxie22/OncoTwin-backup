#!/usr/bin/env python3
"""Layer 8 feature-ablation, robustness, and uncertainty screen."""

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
    "oncotwin_layer8_ablation_uncertainty_screen_v0_2"
)

EVALUATION_CLASS = (
    "exploratory_ablation_uncertainty_not_locked_confirmatory"
)

EXPECTED_HASHES = {
    "landmark_input": (
        "7d47c12ddcdddded13abbcb1ea56512e2f27a8aeb90ef62590ab792c5187d87d"
    ),
    "repeat_input": (
        "9b7f241a8a040632163c3d53b35bc688ce7eb79c377f19d0a5a7974df0cf786a"
    ),
    "baseline_json": (
        "35d4ca48b9d9676fdc5bbb0baad4135c1e2eb0f40b85ef70fc17bc20a4b503fe"
    ),
    "baseline_repeat": (
        "fd41b7878a46d1e1d619a76634dba1da183f5ac8f5276ad561024f113869603f"
    ),
    "baseline_script": (
        "f2496e0ee62f899a5649ad854895970ecf25d170de41e88f13f8c102be91e785"
    ),
}

N_REPEATS = 20
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4

BASE_SEED = 830260
BOOTSTRAP_SEED = 830261

PARsimony_LOG_RMSE_TOLERANCE = 0.01
SUBGROUP_HARM_LOG_RMSE = 0.02
MIN_SUBGROUP_SIZE = 20
EPSILON = 1e-12

FIXED_MODELS = (
    "no_update",
    "trend_half",
)

ABLATION_FEATURES = {
    "ridge_t2_t1_only": (
        "log_t2_over_t1",
    ),
    "ridge_t2_prior_only": (
        "log_t2_over_prior",
    ),
    "ridge_incremental": (
        "log_t2_over_t1",
        "log_t2_over_prior",
    ),
    "ridge_plus_history": (
        "log_t2_over_t1",
        "log_t2_over_prior",
        "log_t1_over_baseline",
    ),
    "ridge_plus_prior_level": (
        "log_t2_over_t1",
        "log_t2_over_prior",
        "log_prior_ml",
    ),
    "ridge_plus_prior_cv": (
        "log_t2_over_t1",
        "log_t2_over_prior",
        "prior_cv",
    ),
    "ridge_context": (
        "log_t2_over_t1",
        "log_t2_over_prior",
        "log_t1_over_baseline",
        "log_prior_ml",
        "prior_cv",
    ),
}

# Stable offsets preserve exact reproduction of models already
# evaluated in the v0_1 incremental baseline screen. New
# ablations use a separate namespace so dictionary insertion
# order cannot change their inner-CV splits.
MODEL_SEED_OFFSETS = {
    "ridge_incremental": 1,
    "ridge_context": 2,
    "ridge_t2_t1_only": 101,
    "ridge_t2_prior_only": 102,
    "ridge_plus_history": 103,
    "ridge_plus_prior_level": 104,
    "ridge_plus_prior_cv": 105,
}

if set(MODEL_SEED_OFFSETS) != set(ABLATION_FEATURES):
    raise RuntimeError(
        "MODEL_SEED_OFFSETS must exactly cover "
        "ABLATION_FEATURES"
    )

MODEL_NAMES = FIXED_MODELS + tuple(
    ABLATION_FEATURES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--baseline-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-repeat",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-script",
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
            f"Unable to load module specification: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_json(path: Path) -> dict[str, Any]:
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
            dict(payload),
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


def interval_metrics(
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal_coverage: float,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if not (
        observed.shape == lower.shape == upper.shape
    ):
        raise ValueError(
            "Interval arrays have different shapes"
        )

    if (
        np.any(observed <= 0)
        or np.any(lower <= 0)
        or np.any(upper < lower)
    ):
        raise ValueError(
            "Invalid positive interval values"
        )

    alpha = 1.0 - nominal_coverage
    covered = (
        (observed >= lower)
        & (observed <= upper)
    )
    width = upper - lower

    score = width.copy()

    below = observed < lower
    above = observed > upper

    score[below] += (
        2.0 / alpha
    ) * (lower[below] - observed[below])

    score[above] += (
        2.0 / alpha
    ) * (observed[above] - upper[above])

    return {
        "n": int(observed.size),
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": float(
            np.mean(covered)
        ),
        "mean_width_ml": float(
            np.mean(width)
        ),
        "median_width_ml": float(
            np.median(width)
        ),
        "mean_relative_width": float(
            np.mean(
                width
                / np.maximum(
                    observed,
                    EPSILON,
                )
            )
        ),
        "mean_interval_score": float(
            np.mean(score)
        ),
    }


def subgroup_label(
    ratio: float,
) -> str:
    if ratio < 0.90:
        return "decrease_over_10pct"

    if ratio > 1.10:
        return "increase_over_10pct"

    return "stable_within_10pct"


def safe_prediction_metrics(
    baseline_module: ModuleType,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    if observed.size == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "log_rmse": None,
            "bias": None,
        }

    return baseline_module.prediction_metrics(
        observed,
        predicted,
    )


def main() -> None:
    args = parse_args()

    actual_hashes = {
        "landmark_input": sha256_file(
            args.landmark_input
        ),
        "repeat_input": sha256_file(
            args.repeat_input
        ),
        "baseline_json": sha256_file(
            args.baseline_json
        ),
        "baseline_repeat": sha256_file(
            args.baseline_repeat
        ),
        "baseline_script": sha256_file(
            args.baseline_script
        ),
    }

    for name, expected in EXPECTED_HASHES.items():
        actual = actual_hashes[name]

        if actual != expected:
            raise RuntimeError(
                f"{name} hash mismatch: "
                f"expected={expected} actual={actual}"
            )

    baseline_module = load_module(
        "_oncotwin_layer8_baseline_screen",
        args.baseline_script,
    )

    locked_runner = load_module(
        "_oncotwin_layer8_locked_splitter",
        (
            REPO_ROOT
            / "scripts/evals/"
            "run_layer6_locked_confirmatory.py"
        ),
    )

    baseline_summary = load_json(
        args.baseline_json
    )

    if baseline_summary["decision"]["status"] != (
        "incremental_t2_signal_detected_"
        "proceed_to_uncertainty_and_ablation"
    ):
        raise RuntimeError(
            "Baseline screen did not route to ablation"
        )

    landmark_rows = load_jsonl(
        args.landmark_input
    )
    repeat_rows = load_jsonl(
        args.repeat_input
    )
    baseline_repeat_rows = load_jsonl(
        args.baseline_repeat
    )

    if len(landmark_rows) != 220:
        raise RuntimeError(
            f"Expected 220 landmarks, got {len(landmark_rows)}"
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
        for row in repeat_rows
        if str(row["case_id"]) in landmark_by_case
    ]

    if len(relevant_repeat_rows) != 4400:
        raise RuntimeError(
            "Expected 4400 Layer 6 repeat rows, "
            f"got {len(relevant_repeat_rows)}"
        )

    baseline_reference = {
        (
            str(row["case_id"]),
            int(row["repeat_number"]),
        ): row
        for row in baseline_repeat_rows
    }

    if len(baseline_reference) != 4400:
        raise RuntimeError(
            "Baseline repeat reference does not contain "
            "4400 unique case/repeat rows"
        )

    rows_by_repeat: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in relevant_repeat_rows:
        rows_by_repeat[
            int(row["repeat_number"])
        ].append(row)

    if sorted(rows_by_repeat) != list(
        range(1, N_REPEATS + 1)
    ):
        raise RuntimeError(
            "Repeat rows do not contain repeats 1 through 20"
        )

    repeat_output_rows: list[dict[str, Any]] = []
    repeat_results: list[dict[str, Any]] = []

    alpha_selection_counts = {
        model_name: Counter()
        for model_name in ABLATION_FEATURES
    }

    baseline_reproduction_max_difference = 0.0

    for repeat_number in range(
        1,
        N_REPEATS + 1,
    ):
        source_rows = sorted(
            rows_by_repeat[repeat_number],
            key=lambda row: str(row["case_id"]),
        )

        if len(source_rows) != 220:
            raise RuntimeError(
                f"Repeat {repeat_number} has "
                f"{len(source_rows)} rows"
            )

        case_ids = [
            str(row["case_id"])
            for row in source_rows
        ]

        baseline = np.asarray(
            [
                positive_float(
                    landmark_by_case[case_id][
                        "baseline_volume_ml"
                    ],
                    name=f"{case_id}.baseline",
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
                    name=f"{case_id}.t1",
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
                    name=f"{case_id}.t2",
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
                    name=f"{case_id}.t3",
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
                        f"{row['case_id']}.prior"
                    ),
                )
                for row in source_rows
            ],
            dtype=float,
        )

        prior_sd = np.asarray(
            [
                finite_float(
                    landmark_by_case[case_id][
                        "layer6_oof_prior_sd_ml"
                    ],
                    name=f"{case_id}.prior_sd",
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

        outer_fold = np.asarray(
            [
                int(row["outer_fold_index"])
                for row in source_rows
            ],
            dtype=int,
        )

        if set(outer_fold.tolist()) != set(
            range(N_OUTER_FOLDS)
        ):
            raise RuntimeError(
                f"Repeat {repeat_number} lacks an outer fold"
            )

        feature_values = {
            "log_t2_over_t1": np.log(t2 / t1),
            "log_t2_over_prior": np.log(
                t2 / prior
            ),
            "log_t1_over_baseline": np.log(
                t1 / baseline
            ),
            "log_prior_ml": np.log(prior),
            "prior_cv": (
                prior_sd
                / np.maximum(
                    prior,
                    EPSILON,
                )
            ),
        }

        feature_matrices = {
            model_name: np.column_stack(
                [
                    feature_values[feature_name]
                    for feature_name
                    in feature_names
                ]
            )
            for model_name, feature_names
            in ABLATION_FEATURES.items()
        }

        residual_target = (
            np.log(observed)
            - np.log(prior)
        )

        predictions = {
            "no_update": prior.copy(),
            "trend_half": (
                prior
                * np.exp(
                    0.5
                    * feature_values[
                        "log_t2_over_t1"
                    ]
                )
            ),
        }

        selected_alphas = {}

        for model_name in ABLATION_FEATURES:
            predictions[model_name] = np.full(
                observed.shape,
                np.nan,
                dtype=float,
            )
            selected_alphas[model_name] = np.full(
                observed.shape,
                np.nan,
                dtype=float,
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

        for outer_fold_index in range(
            N_OUTER_FOLDS
        ):
            test_indices = np.flatnonzero(
                outer_fold == outer_fold_index
            ).tolist()

            train_indices = np.flatnonzero(
                outer_fold != outer_fold_index
            ).tolist()

            if not test_indices or not train_indices:
                raise RuntimeError(
                    f"Repeat {repeat_number} fold "
                    f"{outer_fold_index} is empty"
                )

            for model_name in ABLATION_FEATURES:
                model_seed_offset = MODEL_SEED_OFFSETS[
                    model_name
                ]
                alpha, _ = (
                    baseline_module.select_ridge_alpha(
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
                            + model_seed_offset
                        ),
                    )
                )

                alpha_selection_counts[
                    model_name
                ][alpha] += 1

                fitted = baseline_module.fit_ridge(
                    feature_matrices[model_name][
                        train_indices
                    ],
                    residual_target[train_indices],
                    alpha,
                )

                residual_prediction = (
                    baseline_module.predict_ridge(
                        fitted,
                        feature_matrices[model_name][
                            test_indices
                        ],
                    )
                )

                residual_prediction = np.clip(
                    residual_prediction,
                    -20.0,
                    20.0,
                )

                predictions[model_name][
                    test_indices
                ] = (
                    prior[test_indices]
                    * np.exp(residual_prediction)
                )

                selected_alphas[model_name][
                    test_indices
                ] = alpha

        for model_name in MODEL_NAMES:
            values = predictions[model_name]

            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Non-finite predictions for {model_name}"
                )

            if np.any(values <= 0):
                raise RuntimeError(
                    f"Nonpositive predictions for {model_name}"
                )

        interval_values = {
            model_name: {
                "lower80": np.full(
                    observed.shape,
                    np.nan,
                ),
                "upper80": np.full(
                    observed.shape,
                    np.nan,
                ),
                "lower95": np.full(
                    observed.shape,
                    np.nan,
                ),
                "upper95": np.full(
                    observed.shape,
                    np.nan,
                ),
            }
            for model_name in MODEL_NAMES
        }

        for outer_fold_index in range(
            N_OUTER_FOLDS
        ):
            test_indices = np.flatnonzero(
                outer_fold == outer_fold_index
            ).tolist()

            calibration_indices = np.flatnonzero(
                outer_fold != outer_fold_index
            ).tolist()

            for model_name in MODEL_NAMES:
                absolute_log_residual = np.abs(
                    np.log(
                        predictions[model_name][
                            calibration_indices
                        ]
                    )
                    - np.log(
                        observed[
                            calibration_indices
                        ]
                    )
                )

                q80 = float(
                    np.quantile(
                        absolute_log_residual,
                        0.80,
                    )
                )
                q95 = float(
                    np.quantile(
                        absolute_log_residual,
                        0.95,
                    )
                )

                test_prediction = predictions[
                    model_name
                ][test_indices]

                interval_values[model_name][
                    "lower80"
                ][test_indices] = (
                    test_prediction * math.exp(-q80)
                )
                interval_values[model_name][
                    "upper80"
                ][test_indices] = (
                    test_prediction * math.exp(q80)
                )
                interval_values[model_name][
                    "lower95"
                ][test_indices] = (
                    test_prediction * math.exp(-q95)
                )
                interval_values[model_name][
                    "upper95"
                ][test_indices] = (
                    test_prediction * math.exp(q95)
                )

        for model_name in MODEL_NAMES:
            for interval_name, values in (
                interval_values[model_name].items()
            ):
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        f"Invalid interval {model_name} "
                        f"{interval_name}"
                    )

        for index, case_id in enumerate(case_ids):
            reference = baseline_reference[
                (case_id, repeat_number)
            ]

            comparisons = {
                "no_update": (
                    "prediction_no_update_ml"
                ),
                "trend_half": (
                    "prediction_trend_half_ml"
                ),
                "ridge_incremental": (
                    "prediction_ridge_incremental_ml"
                ),
                "ridge_context": (
                    "prediction_ridge_context_ml"
                ),
            }

            for model_name, field in comparisons.items():
                expected_value = float(
                    reference[field]
                )
                actual_value = float(
                    predictions[model_name][index]
                )

                difference = abs(
                    expected_value - actual_value
                )

                baseline_reproduction_max_difference = max(
                    baseline_reproduction_max_difference,
                    difference,
                )

                tolerance = 1e-11 * max(
                    1.0,
                    abs(expected_value),
                    abs(actual_value),
                )

                if difference > tolerance:
                    raise RuntimeError(
                        "Baseline prediction reproduction "
                        f"failed for {case_id} repeat "
                        f"{repeat_number} model {model_name}: "
                        f"expected={expected_value} "
                        f"actual={actual_value}"
                    )

            output_row = {
                "analysis_version": ANALYSIS_VERSION,
                "evaluation_class": EVALUATION_CLASS,
                "case_id": case_id,
                "repeat_number": repeat_number,
                "outer_fold_index": int(
                    outer_fold[index]
                ),
                "prior_information_cutoff": "T1",
                "new_evidence": "T2",
                "held_out_target": "T3",
                "held_out_from_layer8_training": True,
                "target_used_as_feature": False,
                "layer6_prior_held_out_for_own_case": True,
                "observed_t3_volume_ml": float(
                    observed[index]
                ),
            }

            for model_name in MODEL_NAMES:
                output_row[
                    f"prediction_{model_name}_ml"
                ] = float(
                    predictions[model_name][index]
                )

                output_row[
                    f"{model_name}_lower80_ml"
                ] = float(
                    interval_values[model_name][
                        "lower80"
                    ][index]
                )
                output_row[
                    f"{model_name}_upper80_ml"
                ] = float(
                    interval_values[model_name][
                        "upper80"
                    ][index]
                )
                output_row[
                    f"{model_name}_lower95_ml"
                ] = float(
                    interval_values[model_name][
                        "lower95"
                    ][index]
                )
                output_row[
                    f"{model_name}_upper95_ml"
                ] = float(
                    interval_values[model_name][
                        "upper95"
                    ][index]
                )

            for model_name in ABLATION_FEATURES:
                output_row[
                    f"{model_name}_alpha"
                ] = float(
                    selected_alphas[model_name][
                        index
                    ]
                )

            repeat_output_rows.append(output_row)

        repeat_metrics = {
            model_name: (
                baseline_module.prediction_metrics(
                    observed,
                    predictions[model_name],
                )
            )
            for model_name in MODEL_NAMES
        }

        repeat_intervals = {
            model_name: {
                "coverage80": interval_metrics(
                    observed,
                    interval_values[model_name][
                        "lower80"
                    ],
                    interval_values[model_name][
                        "upper80"
                    ],
                    nominal_coverage=0.80,
                ),
                "coverage95": interval_metrics(
                    observed,
                    interval_values[model_name][
                        "lower95"
                    ],
                    interval_values[model_name][
                        "upper95"
                    ],
                    nominal_coverage=0.95,
                ),
            }
            for model_name in MODEL_NAMES
        }

        repeat_results.append(
            {
                "repeat_number": repeat_number,
                "metrics": repeat_metrics,
                "intervals": repeat_intervals,
            }
        )

        print(
            "ABLATION_REPEAT_COMPLETE "
            f"repeat={repeat_number}/{N_REPEATS} "
            f"no_update={repeat_metrics['no_update']['log_rmse']:.8f} "
            f"ridge_incremental="
            f"{repeat_metrics['ridge_incremental']['log_rmse']:.8f} "
            f"ridge_context="
            f"{repeat_metrics['ridge_context']['log_rmse']:.8f}",
            flush=True,
        )

    if len(repeat_output_rows) != 4400:
        raise RuntimeError(
            "Expected 4400 repeat output rows, "
            f"got {len(repeat_output_rows)}"
        )

    per_case_predictions = defaultdict(
        lambda: defaultdict(list)
    )

    for row in repeat_output_rows:
        case_id = str(row["case_id"])

        for model_name in MODEL_NAMES:
            per_case_predictions[case_id][
                model_name
            ].append(
                float(
                    row[
                        f"prediction_{model_name}_ml"
                    ]
                )
            )

    case_output_rows = []

    for case_id in sorted(landmark_by_case):
        source = landmark_by_case[case_id]

        row = {
            "analysis_version": ANALYSIS_VERSION,
            "evaluation_class": EVALUATION_CLASS,
            "case_id": case_id,
            "baseline_volume_ml": float(
                source["baseline_volume_ml"]
            ),
            "t1_volume_ml": float(
                source["t1_early_volume_ml"]
            ),
            "t2_volume_ml": float(
                source["t2_intermediate_volume_ml"]
            ),
            "observed_t3_volume_ml": float(
                source["t3_final_volume_ml"]
            ),
            "layer6_oof_prior_sd_ml": float(
                source["layer6_oof_prior_sd_ml"]
            ),
            "repeat_prediction_count": N_REPEATS,
        }

        for model_name in MODEL_NAMES:
            values = per_case_predictions[
                case_id
            ][model_name]

            if len(values) != N_REPEATS:
                raise RuntimeError(
                    f"Case {case_id} model {model_name} "
                    "does not have 20 predictions"
                )

            row[
                f"{model_name}_mean_prediction_ml"
            ] = statistics.fmean(values)

            row[
                f"{model_name}_prediction_sd_ml"
            ] = statistics.stdev(values)

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
        model_name: (
            baseline_module.prediction_metrics(
                observed_case,
                aggregate_predictions[model_name],
            )
        )
        for model_name in MODEL_NAMES
    }

    paired_comparisons = {
        model_name: (
            baseline_module.bootstrap_comparison(
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
        for model_index, model_name in enumerate(
            MODEL_NAMES
        )
        if model_name != "no_update"
    }

    context_vs_incremental = (
        baseline_module.bootstrap_comparison(
            observed=observed_case,
            baseline=aggregate_predictions[
                "ridge_incremental"
            ],
            candidate=aggregate_predictions[
                "ridge_context"
            ],
            seed=BOOTSTRAP_SEED + 990000,
        )
    )

    pooled_intervals = {}

    for model_name in MODEL_NAMES:
        pooled_observed = np.asarray(
            [
                row["observed_t3_volume_ml"]
                for row in repeat_output_rows
            ],
            dtype=float,
        )

        pooled_intervals[model_name] = {
            "coverage80": interval_metrics(
                pooled_observed,
                np.asarray(
                    [
                        row[
                            f"{model_name}_lower80_ml"
                        ]
                        for row in repeat_output_rows
                    ]
                ),
                np.asarray(
                    [
                        row[
                            f"{model_name}_upper80_ml"
                        ]
                        for row in repeat_output_rows
                    ]
                ),
                nominal_coverage=0.80,
            ),
            "coverage95": interval_metrics(
                pooled_observed,
                np.asarray(
                    [
                        row[
                            f"{model_name}_lower95_ml"
                        ]
                        for row in repeat_output_rows
                    ]
                ),
                np.asarray(
                    [
                        row[
                            f"{model_name}_upper95_ml"
                        ]
                        for row in repeat_output_rows
                    ]
                ),
                nominal_coverage=0.95,
            ),
        }

    case_ids = [
        str(row["case_id"])
        for row in case_output_rows
    ]

    baseline_case = np.asarray(
        [
            row["baseline_volume_ml"]
            for row in case_output_rows
        ],
        dtype=float,
    )
    t1_case = np.asarray(
        [
            row["t1_volume_ml"]
            for row in case_output_rows
        ],
        dtype=float,
    )
    t2_case = np.asarray(
        [
            row["t2_volume_ml"]
            for row in case_output_rows
        ],
        dtype=float,
    )
    prior_case = aggregate_predictions[
        "no_update"
    ]
    prior_cv_case = np.asarray(
        [
            row["layer6_oof_prior_sd_ml"]
            for row in case_output_rows
        ],
        dtype=float,
    ) / np.maximum(prior_case, EPSILON)

    baseline_median = float(
        np.median(baseline_case)
    )
    prior_cv_median = float(
        np.median(prior_cv_case)
    )

    subgroup_labels = {
        "t1_response": [
            subgroup_label(
                float(t1_case[index] / baseline_case[index])
            )
            for index in range(len(case_ids))
        ],
        "t2_response": [
            subgroup_label(
                float(t2_case[index] / t1_case[index])
            )
            for index in range(len(case_ids))
        ],
        "baseline_size": [
            (
                "baseline_low"
                if baseline_case[index]
                <= baseline_median
                else "baseline_high"
            )
            for index in range(len(case_ids))
        ],
        "prior_variability": [
            (
                "prior_cv_low"
                if prior_cv_case[index]
                <= prior_cv_median
                else "prior_cv_high"
            )
            for index in range(len(case_ids))
        ],
        "t2_vs_prior": [
            (
                "t2_below_prior"
                if t2_case[index] < prior_case[index]
                else "t2_at_or_above_prior"
            )
            for index in range(len(case_ids))
        ],
    }

    subgroup_results = {}

    for dimension, labels in subgroup_labels.items():
        subgroup_results[dimension] = {}

        for label in sorted(set(labels)):
            indices = np.asarray(
                [
                    index
                    for index, value in enumerate(labels)
                    if value == label
                ],
                dtype=int,
            )

            observed_subset = observed_case[indices]
            no_update_metrics = safe_prediction_metrics(
                baseline_module,
                observed_subset,
                aggregate_predictions[
                    "no_update"
                ][indices],
            )

            model_results = {}

            for model_name in MODEL_NAMES:
                metrics = safe_prediction_metrics(
                    baseline_module,
                    observed_subset,
                    aggregate_predictions[
                        model_name
                    ][indices],
                )

                model_results[model_name] = {
                    "metrics": metrics,
                    "delta_log_rmse_vs_no_update": (
                        (
                            float(metrics["log_rmse"])
                            - float(
                                no_update_metrics[
                                    "log_rmse"
                                ]
                            )
                        )
                        if metrics["log_rmse"]
                        is not None
                        else None
                    ),
                    "delta_mae_vs_no_update": (
                        (
                            float(metrics["mae"])
                            - float(
                                no_update_metrics["mae"]
                            )
                        )
                        if metrics["mae"] is not None
                        else None
                    ),
                }

            subgroup_results[dimension][label] = {
                "n": int(indices.size),
                "models": model_results,
            }

    repeat_stability = {}

    for model_name in MODEL_NAMES:
        if model_name == "no_update":
            continue

        repeat_stability[model_name] = {
            "beats_no_update_log_rmse": sum(
                float(
                    result["metrics"][model_name][
                        "log_rmse"
                    ]
                )
                < float(
                    result["metrics"]["no_update"][
                        "log_rmse"
                    ]
                )
                for result in repeat_results
            ),
            "beats_trend_half_log_rmse": sum(
                float(
                    result["metrics"][model_name][
                        "log_rmse"
                    ]
                )
                < float(
                    result["metrics"]["trend_half"][
                        "log_rmse"
                    ]
                )
                for result in repeat_results
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

    best_log_rmse = float(
        aggregate_metrics[best_model]["log_rmse"]
    )

    robust_candidates = []

    for model_name in candidate_models:
        comparison = paired_comparisons[
            model_name
        ]
        intervals = comparison[
            "bootstrap_95_ci"
        ]

        both_intervals_exclude_zero = (
            float(intervals["delta_mae"][1]) < 0
            and float(
                intervals["delta_log_rmse"][1]
            ) < 0
        )

        within_parsimony_band = (
            float(
                aggregate_metrics[model_name][
                    "log_rmse"
                ]
            )
            <= best_log_rmse
            + PARsimony_LOG_RMSE_TOLERANCE
        )

        if (
            both_intervals_exclude_zero
            and within_parsimony_band
        ):
            robust_candidates.append(model_name)

    complexity = {
        "trend_half": 0,
        **{
            model_name: len(feature_names)
            for model_name, feature_names
            in ABLATION_FEATURES.items()
        },
    }

    if robust_candidates:
        recommended_candidate = min(
            robust_candidates,
            key=lambda model_name: (
                complexity[model_name],
                float(
                    aggregate_metrics[model_name][
                        "log_rmse"
                    ]
                ),
            ),
        )
    else:
        recommended_candidate = best_model

    harmful_subgroups = []

    for dimension, groups in subgroup_results.items():
        for label, result in groups.items():
            if int(result["n"]) < MIN_SUBGROUP_SIZE:
                continue

            delta = result["models"][
                recommended_candidate
            ]["delta_log_rmse_vs_no_update"]

            if (
                delta is not None
                and float(delta)
                > SUBGROUP_HARM_LOG_RMSE
            ):
                harmful_subgroups.append(
                    {
                        "dimension": dimension,
                        "label": label,
                        "n": result["n"],
                        "delta_log_rmse": delta,
                    }
                )

    candidate_intervals = pooled_intervals[
        recommended_candidate
    ]

    coverage80 = float(
        candidate_intervals["coverage80"][
            "empirical_coverage"
        ]
    )
    coverage95 = float(
        candidate_intervals["coverage95"][
            "empirical_coverage"
        ]
    )

    uncertainty_reasonable = (
        0.74 <= coverage80 <= 0.88
        and 0.89 <= coverage95 <= 0.99
    )

    recommended_comparison = paired_comparisons[
        recommended_candidate
    ]
    recommended_cis = recommended_comparison[
        "bootstrap_95_ci"
    ]

    robust_gain = (
        float(recommended_cis["delta_mae"][1]) < 0
        and float(
            recommended_cis["delta_log_rmse"][1]
        ) < 0
    )

    if (
        robust_gain
        and not harmful_subgroups
        and uncertainty_reasonable
    ):
        decision = (
            "parsimonious_candidate_ready_for_"
            "strict_full_stack_nested_evaluation"
        )
    elif robust_gain:
        decision = (
            "incremental_signal_confirmed_but_"
            "resolve_robustness_or_uncertainty_before_lock"
        )
    else:
        decision = (
            "ablation_does_not_support_model_escalation"
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
            "exploratory_feature_ablation_"
            "robustness_uncertainty_screen"
        ),
        "evaluation_class": EVALUATION_CLASS,
        "inputs": {
            "hashes": actual_hashes,
            "landmark_case_count": 220,
            "repeat_count": N_REPEATS,
            "repeat_row_count": 4400,
            "baseline_prediction_reproduction_"
            "maximum_absolute_difference": (
                baseline_reproduction_max_difference
            ),
        },
        "temporal_design": {
            "prior_information_cutoff": "T1",
            "new_evidence": "T2",
            "held_out_target": "T3",
            "target_used_as_feature": False,
            "deployable_full_fit_prior_used": False,
        },
        "models": {
            "no_update": {
                "type": "fixed",
            },
            "trend_half": {
                "type": "fixed",
                "definition": (
                    "Layer 6 prior multiplied by "
                    "(T2/T1)^0.5"
                ),
            },
            **{
                model_name: {
                    "type": (
                        "nested_cv_log_residual_ridge"
                    ),
                    "features": list(feature_names),
                }
                for model_name, feature_names
                in ABLATION_FEATURES.items()
            },
        },
        "aggregate_case_mean_metrics": (
            aggregate_metrics
        ),
        "paired_comparisons_vs_no_update": (
            paired_comparisons
        ),
        "context_vs_incremental": (
            context_vs_incremental
        ),
        "repeat_stability": repeat_stability,
        "ridge_alpha_selection_counts": {
            model_name: {
                str(alpha): count
                for alpha, count
                in sorted(counts.items())
            }
            for model_name, counts
            in alpha_selection_counts.items()
        },
        "cross_fitted_residual_intervals": {
            "method": (
                "For each repeat and outer test fold, "
                "absolute log residual quantiles are estimated "
                "from OOF predictions in the other four folds."
            ),
            "formal_conformal_claim": False,
            "pooled_metrics": pooled_intervals,
            "repeat_results": repeat_results,
        },
        "predictor_defined_subgroups": {
            "target_defined_subgroups_used": False,
            "baseline_size_median": baseline_median,
            "prior_cv_median": prior_cv_median,
            "minimum_subgroup_size_for_harm_flag": (
                MIN_SUBGROUP_SIZE
            ),
            "harm_threshold_delta_log_rmse": (
                SUBGROUP_HARM_LOG_RMSE
            ),
            "results": subgroup_results,
            "recommended_candidate_harm_flags": (
                harmful_subgroups
            ),
        },
        "exploratory_selection": {
            "best_raw_model": best_model,
            "best_raw_log_rmse": best_log_rmse,
            "parsimony_log_rmse_tolerance": (
                PARsimony_LOG_RMSE_TOLERANCE
            ),
            "robust_candidates_with_both_"
            "bootstrap_intervals_below_zero": (
                robust_candidates
            ),
            "recommended_candidate": (
                recommended_candidate
            ),
            "recommended_feature_count": (
                complexity[recommended_candidate]
            ),
            "recommended_interval_coverage80": (
                coverage80
            ),
            "recommended_interval_coverage95": (
                coverage95
            ),
            "uncertainty_reasonable": (
                uncertainty_reasonable
            ),
        },
        "stacking_limitation": {
            "strict_full_stack_nesting": False,
            "locked_use_allowed": False,
            "required_next_evaluation": (
                "Replay or refit the complete Layer 6 base learner "
                "inside every final Layer 8 outer-training partition."
            ),
        },
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
        "decision": {
            "status": decision,
            "recommended_candidate": (
                recommended_candidate
            ),
            "next_step": (
                "Construct a strictly nested full-stack "
                "evaluation for the selected parsimonious updater."
            ),
        },
    }

    atomic_write_json(
        args.summary_json,
        payload,
    )

    report_lines = [
        "# Layer 8 ablation, robustness, and uncertainty screen",
        "",
        f"- Evaluation class: `{EVALUATION_CLASS}`",
        "- Cases: `220`",
        "- Repeats: `20`",
        "- Outer folds: `5`",
        "- Inner ridge folds: `4`",
        (
            "- Baseline reproduction maximum difference: "
            f"`{baseline_reproduction_max_difference:.12g}`"
        ),
        f"- Best raw model: `{best_model}`",
        (
            "- Parsimonious recommended candidate: "
            f"`{recommended_candidate}`"
        ),
        (
            "- Harmful predictor-defined subgroups: "
            f"`{len(harmful_subgroups)}`"
        ),
        (
            "- 80% interval coverage: "
            f"`{coverage80:.6f}`"
        ),
        (
            "- 95% interval coverage: "
            f"`{coverage95:.6f}`"
        ),
        f"- Decision: `{decision}`",
        "",
        "## Aggregate metrics",
        "",
    ]

    for model_name in MODEL_NAMES:
        metrics = aggregate_metrics[model_name]

        report_lines.append(
            (
                f"- `{model_name}`: "
                f"MAE={metrics['mae']:.8f}, "
                f"RMSE={metrics['rmse']:.8f}, "
                f"log RMSE={metrics['log_rmse']:.8f}"
            )
        )

    report_lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            (
                "Subgroups are defined only from information "
                "available through T2. No subgroup is defined "
                "using the T3 target or model error."
            ),
            "",
            (
                "The residual intervals are exploratory "
                "cross-fitted calibration intervals, not a formal "
                "finite-sample conformal guarantee."
            ),
            "",
            (
                "No model from this screen may be called locked "
                "or confirmatory. The next evaluation must nest "
                "the complete Layer 6 training procedure inside "
                "the Layer 8 outer split."
            ),
        ]
    )

    atomic_write_text(
        args.summary_markdown,
        "\n".join(report_lines) + "\n",
    )

    print(
        "ABLATION_SCREEN_COUNTS "
        f"cases={len(case_output_rows)} "
        f"repeats={N_REPEATS} "
        f"repeat_rows={len(repeat_output_rows)} "
        f"models={len(MODEL_NAMES)}"
    )

    print(
        "BASELINE_PREDICTION_REPRODUCTION "
        f"max_absolute_difference="
        f"{baseline_reproduction_max_difference:.12g} "
        "status=pass"
    )

    for model_name in MODEL_NAMES:
        metrics = aggregate_metrics[model_name]

        print(
            "ABLATION_AGGREGATE_METRICS "
            f"model={model_name} "
            f"mae={metrics['mae']} "
            f"rmse={metrics['rmse']} "
            f"log_rmse={metrics['log_rmse']} "
            f"bias={metrics['bias']}"
        )

    for model_name, stability in (
        repeat_stability.items()
    ):
        print(
            "ABLATION_REPEAT_STABILITY "
            f"model={model_name} "
            f"beats_no_update="
            f"{stability['beats_no_update_log_rmse']}/20 "
            f"beats_trend_half="
            f"{stability['beats_trend_half_log_rmse']}/20"
        )

    for model_name, metrics in (
        pooled_intervals.items()
    ):
        print(
            "UNCERTAINTY_CALIBRATION "
            f"model={model_name} "
            f"coverage80="
            f"{metrics['coverage80']['empirical_coverage']} "
            f"coverage95="
            f"{metrics['coverage95']['empirical_coverage']} "
            f"width80="
            f"{metrics['coverage80']['median_width_ml']} "
            f"width95="
            f"{metrics['coverage95']['median_width_ml']}"
        )

    print(
        "CONTEXT_INCREMENTAL_VALUE "
        f"delta_mae="
        f"{context_vs_incremental['candidate_minus_no_update']['mae']} "
        f"delta_mae_ci="
        f"{context_vs_incremental['bootstrap_95_ci']['delta_mae']} "
        f"delta_log_rmse="
        f"{context_vs_incremental['candidate_minus_no_update']['log_rmse']} "
        f"delta_log_rmse_ci="
        f"{context_vs_incremental['bootstrap_95_ci']['delta_log_rmse']}"
    )

    print(
        "SUBGROUP_ROBUSTNESS "
        f"recommended={recommended_candidate} "
        f"harm_flags={len(harmful_subgroups)} "
        f"details={harmful_subgroups}"
    )

    print(
        "PARSIMONIOUS_SELECTION "
        f"best_raw={best_model} "
        f"robust_candidates={robust_candidates} "
        f"recommended={recommended_candidate} "
        f"feature_count={complexity[recommended_candidate]}"
    )

    print(
        "LAYER8_ABLATION_DECISION "
        f"status={decision} "
        f"recommended_candidate={recommended_candidate}"
    )

    print(
        "LAYER8_EVALUATION_GUARD "
        "strict_full_stack_nesting=false "
        "locked_use_allowed=false"
    )

    print(
        f"ABLATION_JSON path={args.summary_json}"
    )
    print(
        f"ABLATION_REPORT path={args.summary_markdown}"
    )
    print(
        f"ABLATION_REPEAT_OUTPUT path={args.repeat_output}"
    )
    print(
        f"ABLATION_CASE_OUTPUT path={args.case_output}"
    )


if __name__ == "__main__":
    main()
