#!/usr/bin/env python3
"""Strictly nested post-selection Layer 8 internal evaluation."""

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
    "oncotwin_layer8_strict_nested_internal_v0_1"
)

EVALUATION_CLASS = (
    "strict_full_stack_nested_post_selection_internal"
)

EXPECTED_HASHES = {
    "landmark_input": (
        "7d47c12ddcdddded13abbcb1ea56512e2f27a8aeb90ef62590ab792c5187d87d"
    ),
    "ablation_json": (
        "1331d9b5b30e8950690bfaa270bd63efc9c3d4635a91b256224471415a56bb26"
    ),
    "ablation_script": (
        "ef2aeabe368a79dbfe1022021f3e19754de9bae9635e4aab1be098fe697a8c15"
    ),
    "baseline_script": (
        "f2496e0ee62f899a5649ad854895970ecf25d170de41e88f13f8c102be91e785"
    ),
    "locked_runner": (
        "a1dd82906d811615996e7a0786b1bbd0a81462d17484ccac9daf1aa803cf4b74"
    ),
    "locked_evaluation": (
        "c910c99eb8b129794faa0db7215cebf669ad922bf6dfe5dd2df09e70146eec48"
    ),
}

CANDIDATE_NAME = "ridge_t2_prior_only"
CANDIDATE_FEATURE = "log_t2_over_layer6_prior"

N_OUTER_FOLDS = 5
N_STACK_FOLDS = 4
N_UPDATER_FOLDS = 4

OUTER_SPLIT_SEED = 840260
SEED_NAMESPACE = 84_000_000

BOOTSTRAP_SEED = 840261

EPSILON = 1e-12
MIN_SUBGROUP_SIZE = 20
SUBGROUP_HARM_LOG_RMSE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly nest Layer 6 and the frozen one-feature "
            "Layer 8 updater inside each outer evaluation fold."
        )
    )

    parser.add_argument(
        "--landmark-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ablation-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--ablation-script",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-script",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--locked-runner",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--locked-evaluation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
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
            f"Could not load module specification: {path}"
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
            f"Expected a JSON object: {path}"
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
            "Invalid interval values"
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


def subgroup_label(ratio: float) -> str:
    if ratio < 0.90:
        return "decrease_over_10pct"

    if ratio > 1.10:
        return "increase_over_10pct"

    return "stable_within_10pct"


def layer6_predict(
    *,
    locked: ModuleType,
    records: Sequence[Mapping[str, Any]],
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    early_candidate_predictions: Mapping[int, float],
    train_indices: Sequence[int],
    prediction_indices: Sequence[int],
    calibration_seed: int,
    prediction_seed: int,
) -> tuple[
    dict[int, float],
    dict[str, Any],
]:
    if not train_indices:
        raise ValueError(
            "Layer 6 training indices are empty"
        )

    if not prediction_indices:
        raise ValueError(
            "Layer 6 prediction indices are empty"
        )

    train_set = set(train_indices)
    prediction_set = set(prediction_indices)

    if train_set & prediction_set:
        raise RuntimeError(
            "Layer 6 train/prediction overlap detected"
        )

    calibration = locked.estimate_training_calibration(
        records,
        feature_matrix,
        targets,
        list(train_indices),
        early_candidate_predictions,
        seed=calibration_seed,
    )

    mean, sd = locked.train_ensemble_predict(
        feature_matrix,
        targets,
        list(train_indices),
        list(prediction_indices),
        seed=prediction_seed,
    )

    if len(mean) != len(prediction_indices):
        raise RuntimeError(
            "Layer 6 mean prediction count mismatch"
        )

    if len(sd) != len(prediction_indices):
        raise RuntimeError(
            "Layer 6 SD prediction count mismatch"
        )

    predictions: dict[int, float] = {}

    for position, index in enumerate(
        prediction_indices
    ):
        neural_point = locked.decode_rate(
            records[index],
            mean[position],
        )

        blended_point = locked.blend_volume(
            neural_point,
            early_candidate_predictions[index],
        )

        point = locked.apply_log_calibration(
            blended_point,
            calibration["log_shift"],
        )

        predictions[index] = positive_float(
            point,
            name=(
                f"layer6 prediction index={index}"
            ),
        )

    return predictions, {
        "log_shift": float(
            calibration["log_shift"]
        ),
        "q80": float(calibration["q80"]),
        "q95": float(calibration["q95"]),
    }


def make_inner_layer6_oof(
    *,
    locked: ModuleType,
    records: Sequence[Mapping[str, Any]],
    feature_matrix: np.ndarray,
    targets: np.ndarray,
    early_candidate_predictions: Mapping[int, float],
    outer_train_indices: Sequence[int],
    split_seed: int,
    seed_base: int,
) -> tuple[
    dict[int, float],
    list[dict[str, Any]],
]:
    inner_folds = locked.make_folds(
        records,
        list(outer_train_indices),
        N_STACK_FOLDS,
        split_seed,
    )

    flattened = [
        index
        for fold in inner_folds
        for index in fold
    ]

    if (
        len(flattened) != len(set(flattened))
        or set(flattened)
        != set(outer_train_indices)
    ):
        raise RuntimeError(
            "Inner Layer 6 folds do not cover outer training "
            "cases exactly"
        )

    oof_predictions: dict[int, float] = {}
    fold_audit: list[dict[str, Any]] = []

    for inner_fold_index, validation_indices in enumerate(
        inner_folds
    ):
        validation_set = set(validation_indices)

        fit_indices = [
            index
            for index in outer_train_indices
            if index not in validation_set
        ]

        if not validation_indices or not fit_indices:
            raise RuntimeError(
                "Empty inner Layer 6 fold"
            )

        predictions, calibration = layer6_predict(
            locked=locked,
            records=records,
            feature_matrix=feature_matrix,
            targets=targets,
            early_candidate_predictions=(
                early_candidate_predictions
            ),
            train_indices=fit_indices,
            prediction_indices=validation_indices,
            calibration_seed=(
                seed_base
                + 200_000
                + inner_fold_index * 20_000
            ),
            prediction_seed=(
                seed_base
                + 400_000
                + inner_fold_index * 20_000
            ),
        )

        overlap = (
            set(oof_predictions)
            & set(predictions)
        )

        if overlap:
            raise RuntimeError(
                "Duplicate inner Layer 6 OOF predictions"
            )

        oof_predictions.update(predictions)

        fold_audit.append(
            {
                "inner_fold_index": inner_fold_index,
                "training_case_count": len(
                    fit_indices
                ),
                "validation_case_count": len(
                    validation_indices
                ),
                "calibration": calibration,
            }
        )

    if set(oof_predictions) != set(
        outer_train_indices
    ):
        raise RuntimeError(
            "Missing inner Layer 6 OOF predictions"
        )

    return oof_predictions, fold_audit


def cross_fitted_updater_predictions(
    *,
    baseline_module: ModuleType,
    locked: ModuleType,
    local_records: Sequence[Mapping[str, Any]],
    feature_matrix: np.ndarray,
    residual_target: np.ndarray,
    alpha: float,
    split_seed: int,
) -> np.ndarray:
    local_indices = list(
        range(len(local_records))
    )

    folds = locked.make_folds(
        local_records,
        local_indices,
        N_UPDATER_FOLDS,
        split_seed,
    )

    flattened = [
        index
        for fold in folds
        for index in fold
    ]

    if (
        len(flattened) != len(set(flattened))
        or set(flattened)
        != set(local_indices)
    ):
        raise RuntimeError(
            "Updater calibration folds do not cover "
            "outer training cases exactly"
        )

    predictions = np.full(
        len(local_records),
        np.nan,
        dtype=float,
    )

    for validation_indices in folds:
        validation_set = set(validation_indices)

        fit_indices = [
            index
            for index in local_indices
            if index not in validation_set
        ]

        fitted = baseline_module.fit_ridge(
            feature_matrix[fit_indices],
            residual_target[fit_indices],
            alpha,
        )

        predictions[validation_indices] = (
            baseline_module.predict_ridge(
                fitted,
                feature_matrix[
                    validation_indices
                ],
            )
        )

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError(
            "Updater OOF residual prediction is incomplete"
        )

    return predictions


def main() -> None:
    args = parse_args()

    if args.repeats < 1:
        raise ValueError(
            "--repeats must be positive"
        )

    actual_hashes = {
        "landmark_input": sha256_file(
            args.landmark_input
        ),
        "ablation_json": sha256_file(
            args.ablation_json
        ),
        "ablation_script": sha256_file(
            args.ablation_script
        ),
        "baseline_script": sha256_file(
            args.baseline_script
        ),
        "locked_runner": sha256_file(
            args.locked_runner
        ),
        "locked_evaluation": sha256_file(
            args.locked_evaluation
        ),
        "trainer": sha256_file(
            args.trainer
        ),
    }

    for name, expected in EXPECTED_HASHES.items():
        actual = actual_hashes[name]

        if actual != expected:
            raise RuntimeError(
                f"{name} hash mismatch: "
                f"expected={expected} actual={actual}"
            )

    ablation = load_json(
        args.ablation_json
    )

    if ablation["decision"]["status"] != (
        "parsimonious_candidate_ready_for_"
        "strict_full_stack_nested_evaluation"
    ):
        raise RuntimeError(
            "Ablation artifact does not authorize "
            "strict nested evaluation"
        )

    if ablation["decision"][
        "recommended_candidate"
    ] != CANDIDATE_NAME:
        raise RuntimeError(
            "Unexpected frozen candidate: "
            f"{ablation['decision']['recommended_candidate']}"
        )

    baseline_module = load_module(
        "_oncotwin_layer8_strict_baseline",
        args.baseline_script,
    )

    trainer = load_module(
        "_oncotwin_layer8_strict_trainer",
        args.trainer,
    )

    trainer._verify_locked_artifact(
        args.locked_evaluation
    )

    locked = trainer.LOCKED
    training_data = trainer._build_training_data()

    records = list(training_data["records"])
    feature_matrix = np.asarray(
        training_data["feature_matrix"],
        dtype=float,
    )
    targets = np.asarray(
        training_data["targets"],
        dtype=float,
    )
    active_indices = set(
        training_data["active_indices"]
    )
    early_candidate_predictions = dict(
        training_data[
            "early_candidate_predictions"
        ]
    )

    if len(records) != 270:
        raise RuntimeError(
            f"Expected 270 Layer 6 records, got {len(records)}"
        )

    if len(active_indices) != 252:
        raise RuntimeError(
            "Expected 252 Layer 6-active records"
        )

    record_index_by_case = {
        str(record["case_id"]): index
        for index, record in enumerate(records)
    }

    if len(record_index_by_case) != len(records):
        raise RuntimeError(
            "Duplicate Layer 6 case IDs"
        )

    landmark_rows = load_jsonl(
        args.landmark_input
    )

    landmark_by_case = {
        str(row["case_id"]): row
        for row in landmark_rows
    }

    if (
        len(landmark_rows) != 220
        or len(landmark_by_case) != 220
    ):
        raise RuntimeError(
            "Expected 220 unique landmark cases"
        )

    evaluation_indices = []

    for case_id, landmark in landmark_by_case.items():
        index = record_index_by_case.get(case_id)

        if index is None:
            raise RuntimeError(
                f"Landmark case absent from Layer 6: {case_id}"
            )

        if index not in active_indices:
            raise RuntimeError(
                f"Landmark case is not Layer 6-active: {case_id}"
            )

        landmark_target = positive_float(
            landmark["t3_final_volume_ml"],
            name=f"{case_id}.t3_final_volume_ml",
        )
        layer6_target = positive_float(
            records[index]["observed"],
            name=f"{case_id}.layer6_observed",
        )

        difference = abs(
            landmark_target - layer6_target
        )
        tolerance = 1e-12 * max(
            1.0,
            abs(landmark_target),
            abs(layer6_target),
        )

        if difference > tolerance:
            raise RuntimeError(
                f"T3 target mismatch for {case_id}"
            )

        evaluation_indices.append(index)

    if len(set(evaluation_indices)) != 220:
        raise RuntimeError(
            "Evaluation indices are not unique"
        )

    evaluation_set = set(evaluation_indices)

    non_landmark_active_indices = (
        active_indices - evaluation_set
    )

    if len(non_landmark_active_indices) != 32:
        raise RuntimeError(
            "Expected 32 active non-landmark cases"
        )

    repeat_output_rows: list[dict[str, Any]] = []
    repeat_results: list[dict[str, Any]] = []
    alpha_counts: Counter[float] = Counter()

    for repeat_index in range(args.repeats):
        repeat_number = repeat_index + 1

        outer_folds = locked.make_folds(
            records,
            evaluation_indices,
            N_OUTER_FOLDS,
            OUTER_SPLIT_SEED + repeat_index,
        )

        flattened = [
            index
            for fold in outer_folds
            for index in fold
        ]

        if (
            len(flattened) != len(set(flattened))
            or set(flattened) != evaluation_set
        ):
            raise RuntimeError(
                f"Repeat {repeat_number}: outer folds do not "
                "cover the 220 landmark cases exactly"
            )

        repeat_predictions = {
            "no_update": {},
            "trend_half": {},
            CANDIDATE_NAME: {},
        }

        repeat_intervals = {
            model_name: {
                "lower80": {},
                "upper80": {},
                "lower95": {},
                "upper95": {},
            }
            for model_name in repeat_predictions
        }

        fold_results = []

        for outer_fold_index, test_indices in enumerate(
            outer_folds
        ):
            test_set = set(test_indices)

            outer_train_indices = [
                index
                for index in evaluation_indices
                if index not in test_set
            ]

            if (
                set(outer_train_indices)
                & non_landmark_active_indices
            ):
                raise RuntimeError(
                    "Non-landmark auxiliary cases entered "
                    "Layer 6 training"
                )

            seed_base = (
                SEED_NAMESPACE
                + repeat_index * 10_000_000
                + outer_fold_index * 1_000_000
            )

            test_prior, outer_calibration = layer6_predict(
                locked=locked,
                records=records,
                feature_matrix=feature_matrix,
                targets=targets,
                early_candidate_predictions=(
                    early_candidate_predictions
                ),
                train_indices=outer_train_indices,
                prediction_indices=test_indices,
                calibration_seed=seed_base + 10_000,
                prediction_seed=seed_base + 100_000,
            )

            train_oof_prior, inner_base_audit = (
                make_inner_layer6_oof(
                    locked=locked,
                    records=records,
                    feature_matrix=feature_matrix,
                    targets=targets,
                    early_candidate_predictions=(
                        early_candidate_predictions
                    ),
                    outer_train_indices=(
                        outer_train_indices
                    ),
                    split_seed=seed_base + 150_000,
                    seed_base=seed_base,
                )
            )

            local_records = [
                {
                    "baseline_volume_ml": float(
                        records[index][
                            "baseline_volume_ml"
                        ]
                    ),
                    "early_volume_ml": float(
                        records[index][
                            "early_volume_ml"
                        ]
                    ),
                    "final_day": float(
                        records[index]["final_day"]
                    ),
                }
                for index in outer_train_indices
            ]

            train_feature = np.asarray(
                [
                    [
                        math.log(
                            positive_float(
                                landmark_by_case[
                                    str(records[index][
                                        "case_id"
                                    ])
                                ][
                                    "t2_intermediate_volume_ml"
                                ],
                                name=(
                                    f"{records[index]['case_id']}.t2"
                                ),
                            )
                            / train_oof_prior[index]
                        )
                    ]
                    for index in outer_train_indices
                ],
                dtype=float,
            )

            train_residual_target = np.asarray(
                [
                    math.log(
                        positive_float(
                            records[index]["observed"],
                            name=(
                                f"{records[index]['case_id']}."
                                "observed"
                            ),
                        )
                    )
                    - math.log(
                        train_oof_prior[index]
                    )
                    for index in outer_train_indices
                ],
                dtype=float,
            )

            local_indices = list(
                range(len(outer_train_indices))
            )

            selected_alpha, alpha_scores = (
                baseline_module.select_ridge_alpha(
                    feature_matrix=train_feature,
                    residual_target=(
                        train_residual_target
                    ),
                    fold_records=local_records,
                    train_indices=local_indices,
                    locked_runner=locked,
                    seed=seed_base + 600_000,
                )
            )

            alpha_counts[selected_alpha] += 1

            fitted_updater = baseline_module.fit_ridge(
                train_feature,
                train_residual_target,
                selected_alpha,
            )

            updater_oof_residual = (
                cross_fitted_updater_predictions(
                    baseline_module=baseline_module,
                    locked=locked,
                    local_records=local_records,
                    feature_matrix=train_feature,
                    residual_target=(
                        train_residual_target
                    ),
                    alpha=selected_alpha,
                    split_seed=seed_base + 700_000,
                )
            )

            train_prior_array = np.asarray(
                [
                    train_oof_prior[index]
                    for index in outer_train_indices
                ],
                dtype=float,
            )

            train_t1_array = np.asarray(
                [
                    positive_float(
                        landmark_by_case[
                            str(records[index]["case_id"])
                        ]["t1_early_volume_ml"],
                        name=(
                            f"{records[index]['case_id']}.t1"
                        ),
                    )
                    for index in outer_train_indices
                ],
                dtype=float,
            )

            train_t2_array = np.asarray(
                [
                    positive_float(
                        landmark_by_case[
                            str(records[index]["case_id"])
                        ]["t2_intermediate_volume_ml"],
                        name=(
                            f"{records[index]['case_id']}.t2"
                        ),
                    )
                    for index in outer_train_indices
                ],
                dtype=float,
            )

            train_observed_array = np.asarray(
                [
                    positive_float(
                        records[index]["observed"],
                        name=(
                            f"{records[index]['case_id']}."
                            "observed"
                        ),
                    )
                    for index in outer_train_indices
                ],
                dtype=float,
            )

            train_prediction_for_interval = {
                "no_update": train_prior_array,
                "trend_half": (
                    train_prior_array
                    * np.sqrt(
                        train_t2_array
                        / train_t1_array
                    )
                ),
                CANDIDATE_NAME: (
                    train_prior_array
                    * np.exp(
                        np.clip(
                            updater_oof_residual,
                            -20.0,
                            20.0,
                        )
                    )
                ),
            }

            residual_quantiles = {}

            for model_name, train_prediction in (
                train_prediction_for_interval.items()
            ):
                absolute_log_residual = np.abs(
                    np.log(train_prediction)
                    - np.log(train_observed_array)
                )

                residual_quantiles[model_name] = {
                    "q80": float(
                        np.quantile(
                            absolute_log_residual,
                            0.80,
                            method="higher",
                        )
                    ),
                    "q95": float(
                        np.quantile(
                            absolute_log_residual,
                            0.95,
                            method="higher",
                        )
                    ),
                }

            fold_observed = []
            fold_model_predictions = {
                model_name: []
                for model_name in repeat_predictions
            }

            for index in test_indices:
                case_id = str(
                    records[index]["case_id"]
                )
                landmark = landmark_by_case[case_id]

                prior = test_prior[index]
                t1 = positive_float(
                    landmark["t1_early_volume_ml"],
                    name=f"{case_id}.t1",
                )
                t2 = positive_float(
                    landmark[
                        "t2_intermediate_volume_ml"
                    ],
                    name=f"{case_id}.t2",
                )
                observed = positive_float(
                    landmark["t3_final_volume_ml"],
                    name=f"{case_id}.t3",
                )

                updater_feature = np.asarray(
                    [[math.log(t2 / prior)]],
                    dtype=float,
                )

                predicted_residual = float(
                    baseline_module.predict_ridge(
                        fitted_updater,
                        updater_feature,
                    )[0]
                )

                predicted_residual = float(
                    np.clip(
                        predicted_residual,
                        -20.0,
                        20.0,
                    )
                )

                predictions = {
                    "no_update": prior,
                    "trend_half": (
                        prior
                        * math.sqrt(t2 / t1)
                    ),
                    CANDIDATE_NAME: (
                        prior
                        * math.exp(
                            predicted_residual
                        )
                    ),
                }

                fold_observed.append(observed)

                output_row = {
                    "analysis_version": (
                        ANALYSIS_VERSION
                    ),
                    "evaluation_class": (
                        EVALUATION_CLASS
                    ),
                    "case_id": case_id,
                    "repeat_index": repeat_index,
                    "repeat_number": repeat_number,
                    "outer_fold_index": (
                        outer_fold_index
                    ),
                    "outer_fold_number": (
                        outer_fold_index + 1
                    ),
                    "layer6_training_case_count": len(
                        outer_train_indices
                    ),
                    "layer6_test_case_count": len(
                        test_indices
                    ),
                    "layer6_non_landmark_"
                    "auxiliary_cases_used": False,
                    "outer_test_patient_excluded_"
                    "from_all_layer6_fits": True,
                    "outer_test_patient_excluded_"
                    "from_updater_fit": True,
                    "updater_training_prior_is_oof": True,
                    "candidate_name": CANDIDATE_NAME,
                    "candidate_feature": (
                        CANDIDATE_FEATURE
                    ),
                    "selected_ridge_alpha": (
                        selected_alpha
                    ),
                    "prior_information_cutoff": "T1",
                    "new_evidence": "T2",
                    "held_out_target": "T3",
                    "target_used_as_feature": False,
                    "baseline_volume_ml": float(
                        landmark["baseline_volume_ml"]
                    ),
                    "t1_volume_ml": t1,
                    "t2_volume_ml": t2,
                    "observed_t3_volume_ml": observed,
                    "strict_layer6_prior_ml": prior,
                }

                for model_name, point in (
                    predictions.items()
                ):
                    point = positive_float(
                        point,
                        name=(
                            f"{case_id}.{model_name}"
                        ),
                    )

                    repeat_predictions[
                        model_name
                    ][index] = point

                    fold_model_predictions[
                        model_name
                    ].append(point)

                    q80 = residual_quantiles[
                        model_name
                    ]["q80"]
                    q95 = residual_quantiles[
                        model_name
                    ]["q95"]

                    lower80 = point * math.exp(-q80)
                    upper80 = point * math.exp(q80)
                    lower95 = point * math.exp(-q95)
                    upper95 = point * math.exp(q95)

                    repeat_intervals[
                        model_name
                    ]["lower80"][index] = lower80
                    repeat_intervals[
                        model_name
                    ]["upper80"][index] = upper80
                    repeat_intervals[
                        model_name
                    ]["lower95"][index] = lower95
                    repeat_intervals[
                        model_name
                    ]["upper95"][index] = upper95

                    output_row[
                        f"prediction_{model_name}_ml"
                    ] = point
                    output_row[
                        f"{model_name}_lower80_ml"
                    ] = lower80
                    output_row[
                        f"{model_name}_upper80_ml"
                    ] = upper80
                    output_row[
                        f"{model_name}_lower95_ml"
                    ] = lower95
                    output_row[
                        f"{model_name}_upper95_ml"
                    ] = upper95

                repeat_output_rows.append(
                    output_row
                )

            fold_observed_array = np.asarray(
                fold_observed,
                dtype=float,
            )

            fold_results.append(
                {
                    "outer_fold_index": (
                        outer_fold_index
                    ),
                    "outer_training_case_count": len(
                        outer_train_indices
                    ),
                    "outer_test_case_count": len(
                        test_indices
                    ),
                    "outer_layer6_calibration": (
                        outer_calibration
                    ),
                    "inner_layer6_fold_audit": (
                        inner_base_audit
                    ),
                    "selected_alpha": (
                        selected_alpha
                    ),
                    "alpha_scores": alpha_scores,
                    "residual_quantiles": (
                        residual_quantiles
                    ),
                    "metrics": {
                        model_name: (
                            baseline_module.prediction_metrics(
                                fold_observed_array,
                                np.asarray(
                                    fold_model_predictions[
                                        model_name
                                    ],
                                    dtype=float,
                                ),
                            )
                        )
                        for model_name
                        in fold_model_predictions
                    },
                }
            )

            print(
                "STRICT_OUTER_FOLD_COMPLETE "
                f"repeat={repeat_number}/{args.repeats} "
                f"fold={outer_fold_index + 1}/"
                f"{N_OUTER_FOLDS} "
                f"train={len(outer_train_indices)} "
                f"test={len(test_indices)} "
                f"alpha={selected_alpha}",
                flush=True,
            )

        if any(
            set(repeat_predictions[model_name])
            != evaluation_set
            for model_name in repeat_predictions
        ):
            raise RuntimeError(
                f"Repeat {repeat_number}: incomplete predictions"
            )

        observed_array = np.asarray(
            [
                float(records[index]["observed"])
                for index in evaluation_indices
            ],
            dtype=float,
        )

        metrics_by_model = {
            model_name: (
                baseline_module.prediction_metrics(
                    observed_array,
                    np.asarray(
                        [
                            repeat_predictions[
                                model_name
                            ][index]
                            for index
                            in evaluation_indices
                        ],
                        dtype=float,
                    ),
                )
            )
            for model_name in repeat_predictions
        }

        intervals_by_model = {}

        for model_name in repeat_predictions:
            intervals_by_model[model_name] = {
                "coverage80": interval_metrics(
                    observed_array,
                    np.asarray(
                        [
                            repeat_intervals[
                                model_name
                            ]["lower80"][index]
                            for index
                            in evaluation_indices
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        [
                            repeat_intervals[
                                model_name
                            ]["upper80"][index]
                            for index
                            in evaluation_indices
                        ],
                        dtype=float,
                    ),
                    nominal_coverage=0.80,
                ),
                "coverage95": interval_metrics(
                    observed_array,
                    np.asarray(
                        [
                            repeat_intervals[
                                model_name
                            ]["lower95"][index]
                            for index
                            in evaluation_indices
                        ],
                        dtype=float,
                    ),
                    np.asarray(
                        [
                            repeat_intervals[
                                model_name
                            ]["upper95"][index]
                            for index
                            in evaluation_indices
                        ],
                        dtype=float,
                    ),
                    nominal_coverage=0.95,
                ),
            }

        repeat_results.append(
            {
                "repeat_index": repeat_index,
                "repeat_number": repeat_number,
                "outer_fold_sizes": [
                    len(fold)
                    for fold in outer_folds
                ],
                "metrics": metrics_by_model,
                "intervals": intervals_by_model,
                "folds": fold_results,
            }
        )

        print(
            "STRICT_REPEAT_COMPLETE "
            f"repeat={repeat_number}/{args.repeats} "
            f"no_update_log_rmse="
            f"{metrics_by_model['no_update']['log_rmse']:.8f} "
            f"trend_half_log_rmse="
            f"{metrics_by_model['trend_half']['log_rmse']:.8f} "
            f"candidate_log_rmse="
            f"{metrics_by_model[CANDIDATE_NAME]['log_rmse']:.8f}",
            flush=True,
        )

    expected_repeat_rows = (
        len(evaluation_indices)
        * args.repeats
    )

    if len(repeat_output_rows) != expected_repeat_rows:
        raise RuntimeError(
            "Unexpected strict repeat row count: "
            f"actual={len(repeat_output_rows)} "
            f"expected={expected_repeat_rows}"
        )

    case_model_values: dict[
        str,
        dict[str, list[float]],
    ] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in repeat_output_rows:
        case_id = str(row["case_id"])

        for model_name in (
            "no_update",
            "trend_half",
            CANDIDATE_NAME,
        ):
            case_model_values[case_id][
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
        landmark = landmark_by_case[case_id]

        row = {
            "analysis_version": ANALYSIS_VERSION,
            "evaluation_class": EVALUATION_CLASS,
            "case_id": case_id,
            "baseline_volume_ml": float(
                landmark["baseline_volume_ml"]
            ),
            "t1_volume_ml": float(
                landmark["t1_early_volume_ml"]
            ),
            "t2_volume_ml": float(
                landmark[
                    "t2_intermediate_volume_ml"
                ]
            ),
            "observed_t3_volume_ml": float(
                landmark["t3_final_volume_ml"]
            ),
            "strict_repeat_prediction_count": (
                args.repeats
            ),
        }

        for model_name in (
            "no_update",
            "trend_half",
            CANDIDATE_NAME,
        ):
            values = case_model_values[
                case_id
            ][model_name]

            if len(values) != args.repeats:
                raise RuntimeError(
                    f"Case {case_id} model {model_name} "
                    "has an unexpected prediction count"
                )

            row[
                f"{model_name}_mean_prediction_ml"
            ] = statistics.fmean(values)

            row[
                f"{model_name}_prediction_sd_ml"
            ] = (
                statistics.stdev(values)
                if len(values) > 1
                else 0.0
            )

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
        for model_name in (
            "no_update",
            "trend_half",
            CANDIDATE_NAME,
        )
    }

    aggregate_metrics = {
        model_name: (
            baseline_module.prediction_metrics(
                observed_case,
                predictions,
            )
        )
        for model_name, predictions
        in aggregate_predictions.items()
    }

    candidate_vs_no_update = (
        baseline_module.bootstrap_comparison(
            observed=observed_case,
            baseline=aggregate_predictions[
                "no_update"
            ],
            candidate=aggregate_predictions[
                CANDIDATE_NAME
            ],
            seed=BOOTSTRAP_SEED,
        )
    )

    candidate_vs_trend_half = (
        baseline_module.bootstrap_comparison(
            observed=observed_case,
            baseline=aggregate_predictions[
                "trend_half"
            ],
            candidate=aggregate_predictions[
                CANDIDATE_NAME
            ],
            seed=BOOTSTRAP_SEED + 100_000,
        )
    )

    pooled_observed = np.asarray(
        [
            row["observed_t3_volume_ml"]
            for row in repeat_output_rows
        ],
        dtype=float,
    )

    pooled_intervals = {}

    for model_name in (
        "no_update",
        "trend_half",
        CANDIDATE_NAME,
    ):
        pooled_intervals[model_name] = {
            "coverage80": interval_metrics(
                pooled_observed,
                np.asarray(
                    [
                        row[
                            f"{model_name}_lower80_ml"
                        ]
                        for row in repeat_output_rows
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        row[
                            f"{model_name}_upper80_ml"
                        ]
                        for row in repeat_output_rows
                    ],
                    dtype=float,
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
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        row[
                            f"{model_name}_upper95_ml"
                        ]
                        for row in repeat_output_rows
                    ],
                    dtype=float,
                ),
                nominal_coverage=0.95,
            ),
        }

    repeat_stability = {
        "candidate_beats_no_update_log_rmse": sum(
            float(
                repeat_result["metrics"][
                    CANDIDATE_NAME
                ]["log_rmse"]
            )
            < float(
                repeat_result["metrics"][
                    "no_update"
                ]["log_rmse"]
            )
            for repeat_result in repeat_results
        ),
        "candidate_beats_trend_half_log_rmse": sum(
            float(
                repeat_result["metrics"][
                    CANDIDATE_NAME
                ]["log_rmse"]
            )
            < float(
                repeat_result["metrics"][
                    "trend_half"
                ]["log_rmse"]
            )
            for repeat_result in repeat_results
        ),
    }

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
    no_update_case = aggregate_predictions[
        "no_update"
    ]

    baseline_median = float(
        np.median(baseline_case)
    )

    subgroup_labels = {
        "t2_response": [
            subgroup_label(
                float(t2_case[index] / t1_case[index])
            )
            for index in range(len(case_output_rows))
        ],
        "baseline_size": [
            (
                "baseline_low"
                if baseline_case[index]
                <= baseline_median
                else "baseline_high"
            )
            for index in range(len(case_output_rows))
        ],
        "t2_vs_prior": [
            (
                "t2_below_prior"
                if t2_case[index]
                < no_update_case[index]
                else "t2_at_or_above_prior"
            )
            for index in range(len(case_output_rows))
        ],
    }

    subgroup_results = {}
    harmful_subgroups = []

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

            no_update_metrics = (
                baseline_module.prediction_metrics(
                    observed_case[indices],
                    aggregate_predictions[
                        "no_update"
                    ][indices],
                )
            )

            candidate_metrics = (
                baseline_module.prediction_metrics(
                    observed_case[indices],
                    aggregate_predictions[
                        CANDIDATE_NAME
                    ][indices],
                )
            )

            delta_log_rmse = (
                float(
                    candidate_metrics["log_rmse"]
                )
                - float(
                    no_update_metrics["log_rmse"]
                )
            )

            subgroup_results[dimension][label] = {
                "n": int(indices.size),
                "no_update": no_update_metrics,
                "candidate": candidate_metrics,
                "candidate_minus_no_update": {
                    "mae": (
                        float(candidate_metrics["mae"])
                        - float(no_update_metrics["mae"])
                    ),
                    "log_rmse": delta_log_rmse,
                },
            }

            if (
                int(indices.size) >= MIN_SUBGROUP_SIZE
                and delta_log_rmse
                > SUBGROUP_HARM_LOG_RMSE
            ):
                harmful_subgroups.append(
                    {
                        "dimension": dimension,
                        "label": label,
                        "n": int(indices.size),
                        "delta_log_rmse": (
                            delta_log_rmse
                        ),
                    }
                )

    candidate_no_update_ci = (
        candidate_vs_no_update[
            "bootstrap_95_ci"
        ]
    )

    robust_gain_vs_no_update = (
        float(
            candidate_no_update_ci[
                "delta_mae"
            ][1]
        )
        < 0
        and float(
            candidate_no_update_ci[
                "delta_log_rmse"
            ][1]
        )
        < 0
    )

    no_material_disadvantage_vs_trend = (
        float(
            aggregate_metrics[CANDIDATE_NAME][
                "log_rmse"
            ]
        )
        <= float(
            aggregate_metrics["trend_half"][
                "log_rmse"
            ]
        )
        + 0.01
        and float(
            aggregate_metrics[CANDIDATE_NAME][
                "mae"
            ]
        )
        <= float(
            aggregate_metrics["trend_half"][
                "mae"
            ]
        )
        + 0.01
    )

    candidate_coverage80 = float(
        pooled_intervals[CANDIDATE_NAME][
            "coverage80"
        ]["empirical_coverage"]
    )
    candidate_coverage95 = float(
        pooled_intervals[CANDIDATE_NAME][
            "coverage95"
        ]["empirical_coverage"]
    )

    uncertainty_reasonable = (
        0.74 <= candidate_coverage80 <= 0.88
        and 0.89 <= candidate_coverage95 <= 0.99
    )

    repeat_support = (
        repeat_stability[
            "candidate_beats_no_update_log_rmse"
        ]
        >= max(
            1,
            math.ceil(args.repeats * 0.80),
        )
    )

    if (
        robust_gain_vs_no_update
        and no_material_disadvantage_vs_trend
        and repeat_support
        and uncertainty_reasonable
        and not harmful_subgroups
    ):
        decision = (
            "strict_nested_internal_gain_supported_"
            "external_or_temporal_validation_required"
        )
    elif (
        float(
            aggregate_metrics[CANDIDATE_NAME][
                "log_rmse"
            ]
        )
        < float(
            aggregate_metrics["no_update"][
                "log_rmse"
            ]
        )
    ):
        decision = (
            "strict_nested_internal_signal_inconclusive_"
            "do_not_lock"
        )
    else:
        decision = (
            "strict_nested_internal_gain_not_supported_"
            "do_not_promote"
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
            "strict_full_stack_nested_"
            "post_selection_internal_evaluation"
        ),
        "evaluation_class": EVALUATION_CLASS,
        "candidate": {
            "name": CANDIDATE_NAME,
            "feature": CANDIDATE_FEATURE,
            "target": (
                "log(T3) - log(strict nested Layer 6 prior)"
            ),
            "alpha_grid": list(
                baseline_module.RIDGE_ALPHAS
            ),
            "selection_source": str(
                args.ablation_json
            ),
            "selection_source_sha256": (
                actual_hashes["ablation_json"]
            ),
            "candidate_changed_during_evaluation": False,
        },
        "inputs": {
            "hashes": actual_hashes,
            "landmark_case_count": len(
                evaluation_indices
            ),
            "layer6_active_case_count": len(
                active_indices
            ),
            "active_non_landmark_case_count": len(
                non_landmark_active_indices
            ),
        },
        "design": {
            "repeats": args.repeats,
            "outer_folds": N_OUTER_FOLDS,
            "inner_layer6_oof_folds": (
                N_STACK_FOLDS
            ),
            "updater_tuning_folds": (
                N_UPDATER_FOLDS
            ),
            "layer6_outer_training_population": (
                "only landmark cases assigned to the "
                "current outer-training partition"
            ),
            "non_landmark_active_cases_used": False,
            "outer_test_case_used_in_any_layer6_fit": False,
            "outer_test_case_used_in_updater_fit": False,
            "updater_training_priors_are_layer6_oof": True,
            "strict_full_stack_nesting": True,
            "prior_information_cutoff": "T1",
            "new_evidence": "T2",
            "held_out_target": "T3",
        },
        "repeat_results": repeat_results,
        "ridge_alpha_selection_counts": {
            str(alpha): count
            for alpha, count
            in sorted(alpha_counts.items())
        },
        "aggregate_case_mean_metrics": (
            aggregate_metrics
        ),
        "paired_comparisons": {
            "candidate_vs_no_update": (
                candidate_vs_no_update
            ),
            "candidate_vs_trend_half": (
                candidate_vs_trend_half
            ),
        },
        "repeat_stability": repeat_stability,
        "pooled_cross_fitted_intervals": (
            pooled_intervals
        ),
        "predictor_defined_subgroups": {
            "target_defined_subgroups_used": False,
            "minimum_size_for_harm_flag": (
                MIN_SUBGROUP_SIZE
            ),
            "harm_threshold_delta_log_rmse": (
                SUBGROUP_HARM_LOG_RMSE
            ),
            "results": subgroup_results,
            "candidate_harm_flags": (
                harmful_subgroups
            ),
        },
        "selection_bias_constraint": {
            "independent_confirmation": False,
            "reason": (
                "The same 220-patient cohort was used during "
                "exploratory feature and candidate selection."
            ),
            "model_lock_allowed": False,
            "external_or_temporal_validation_required": True,
        },
        "decision_checks": {
            "robust_gain_vs_no_update": (
                robust_gain_vs_no_update
            ),
            "no_material_disadvantage_vs_trend_half": (
                no_material_disadvantage_vs_trend
            ),
            "repeat_support": repeat_support,
            "uncertainty_reasonable": (
                uncertainty_reasonable
            ),
            "harmful_subgroup_count": len(
                harmful_subgroups
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
            "model_lock_allowed": False,
            "next_step": (
                "If strict nested gain is supported, scale the "
                "prespecified evaluation or move directly to an "
                "untouched external or temporal validation set."
            ),
        },
    }

    atomic_write_json(
        args.summary_json,
        payload,
    )

    report_lines = [
        "# Layer 8 strict nested internal evaluation",
        "",
        f"- Evaluation class: `{EVALUATION_CLASS}`",
        f"- Cases: `{len(evaluation_indices)}`",
        f"- Repeats: `{args.repeats}`",
        f"- Outer folds: `{N_OUTER_FOLDS}`",
        (
            "- Inner Layer 6 OOF folds: "
            f"`{N_STACK_FOLDS}`"
        ),
        (
            "- Candidate: "
            f"`{CANDIDATE_NAME}`"
        ),
        (
            "- Strict full-stack nesting: "
            "`true`"
        ),
        (
            "- Non-landmark auxiliary cases used: "
            "`false`"
        ),
        (
            "- Independent confirmation: "
            "`false`"
        ),
        (
            "- Model lock allowed: "
            "`false`"
        ),
        f"- Decision: `{decision}`",
        "",
        "## Aggregate case-mean metrics",
        "",
    ]

    for model_name, metrics in (
        aggregate_metrics.items()
    ):
        report_lines.append(
            (
                f"- `{model_name}`: "
                f"MAE={metrics['mae']:.8f}, "
                f"RMSE={metrics['rmse']:.8f}, "
                f"log RMSE={metrics['log_rmse']:.8f}, "
                f"bias={metrics['bias']:.8f}"
            )
        )

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This evaluation removes base-learner stacking "
                "leakage by retraining Layer 6 within every "
                "Layer 8 outer-training partition."
            ),
            "",
            (
                "It remains post-selection internal validation "
                "because the candidate was selected using this "
                "same 220-patient cohort. An untouched external "
                "or temporal evaluation is required before any "
                "confirmatory or model-lock claim."
            ),
        ]
    )

    atomic_write_text(
        args.summary_markdown,
        "\n".join(report_lines) + "\n",
    )

    print(
        "STRICT_NESTED_COUNTS "
        f"cases={len(evaluation_indices)} "
        f"repeats={args.repeats} "
        f"outer_folds={N_OUTER_FOLDS} "
        f"repeat_rows={len(repeat_output_rows)} "
        f"non_landmark_active_used=0"
    )

    for model_name, metrics in (
        aggregate_metrics.items()
    ):
        print(
            "STRICT_NESTED_AGGREGATE_METRICS "
            f"model={model_name} "
            f"n={metrics['n']} "
            f"mae={metrics['mae']} "
            f"rmse={metrics['rmse']} "
            f"log_rmse={metrics['log_rmse']} "
            f"bias={metrics['bias']}"
        )

    for comparison_name, comparison in (
        (
            "candidate_vs_no_update",
            candidate_vs_no_update,
        ),
        (
            "candidate_vs_trend_half",
            candidate_vs_trend_half,
        ),
    ):
        delta = comparison[
            "candidate_minus_no_update"
        ]
        intervals = comparison[
            "bootstrap_95_ci"
        ]
        counts = comparison[
            "absolute_error_comparison"
        ]

        print(
            "STRICT_NESTED_PAIRED_COMPARISON "
            f"comparison={comparison_name} "
            f"delta_mae={delta['mae']} "
            f"delta_mae_ci="
            f"{intervals['delta_mae']} "
            f"delta_log_rmse="
            f"{delta['log_rmse']} "
            f"delta_log_rmse_ci="
            f"{intervals['delta_log_rmse']} "
            f"better={counts['candidate_better']} "
            f"worse={counts['candidate_worse']} "
            f"ties={counts['tie']}"
        )

    candidate_intervals = pooled_intervals[
        CANDIDATE_NAME
    ]

    print(
        "STRICT_NESTED_UNCERTAINTY "
        f"coverage80="
        f"{candidate_intervals['coverage80']['empirical_coverage']} "
        f"coverage95="
        f"{candidate_intervals['coverage95']['empirical_coverage']} "
        f"median_width80="
        f"{candidate_intervals['coverage80']['median_width_ml']} "
        f"median_width95="
        f"{candidate_intervals['coverage95']['median_width_ml']}"
    )

    print(
        "STRICT_NESTED_REPEAT_STABILITY "
        f"candidate_beats_no_update="
        f"{repeat_stability['candidate_beats_no_update_log_rmse']}"
        f"/{args.repeats} "
        f"candidate_beats_trend_half="
        f"{repeat_stability['candidate_beats_trend_half_log_rmse']}"
        f"/{args.repeats}"
    )

    print(
        "STRICT_NESTED_ALPHA_SELECTION "
        f"counts="
        f"{dict(sorted(alpha_counts.items()))}"
    )

    print(
        "STRICT_NESTED_SUBGROUP_ROBUSTNESS "
        f"harm_flags={len(harmful_subgroups)} "
        f"details={harmful_subgroups}"
    )

    print(
        "STRICT_NESTING_AUDIT "
        "strict_full_stack_nesting=true "
        "outer_test_in_layer6_fit=false "
        "outer_test_in_updater_fit=false "
        "updater_training_priors_oof=true "
        "non_landmark_auxiliary_cases_used=false"
    )

    print(
        "STRICT_NESTED_CONFIRMATION_GUARD "
        "independent_confirmation=false "
        "model_lock_allowed=false "
        "external_or_temporal_validation_required=true"
    )

    print(
        "LAYER8_STRICT_NESTED_DECISION "
        f"status={decision}"
    )

    print(
        f"STRICT_NESTED_JSON path={args.summary_json}"
    )
    print(
        "STRICT_NESTED_REPORT "
        f"path={args.summary_markdown}"
    )
    print(
        "STRICT_NESTED_REPEAT_OUTPUT "
        f"path={args.repeat_output}"
    )
    print(
        "STRICT_NESTED_CASE_OUTPUT "
        f"path={args.case_output}"
    )


if __name__ == "__main__":
    main()
