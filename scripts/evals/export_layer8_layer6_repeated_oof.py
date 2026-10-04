#!/usr/bin/env python3
"""Export repeated out-of-fold Layer 6 predictions for Layer 8."""

from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ANALYSIS_VERSION = (
    "oncotwin_layer8_layer6_repeated_oof_export_v0_1"
)

EXPECTED_CONFIRMATORY_SHA256 = (
    "c910c99eb8b129794faa0db7215cebf669ad922bf6dfe5dd2df09e70146eec48"
)

EXPECTED_LANDMARK_SHA256 = (
    "f37605d875f6de23042f2e5137933fa7822208bb1ed21146bfa94db23170a82b"
)

METRIC_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the locked Layer 6 repeated outer-CV procedure "
            "and export patient-level held-out predictions."
        )
    )

    parser.add_argument(
        "--landmark-input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--locked-evaluation",
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
        "--repeat-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--case-jsonl",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--merged-landmark-jsonl",
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
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"

    atomic_write_text(path, text)


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


def verify_numeric_mapping(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> float:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{context}: metric key mismatch: "
            f"actual={sorted(actual)} expected={sorted(expected)}"
        )

    maximum_difference = 0.0

    for key in sorted(actual):
        actual_value = actual[key]
        expected_value = expected[key]

        if isinstance(actual_value, bool) or isinstance(
            expected_value,
            bool,
        ):
            if actual_value != expected_value:
                raise RuntimeError(
                    f"{context}.{key}: value mismatch"
                )
            continue

        try:
            actual_number = float(actual_value)
            expected_number = float(expected_value)
        except (TypeError, ValueError):
            if actual_value != expected_value:
                raise RuntimeError(
                    f"{context}.{key}: value mismatch"
                )
            continue

        if not math.isfinite(actual_number):
            raise RuntimeError(
                f"{context}.{key}: actual value is non-finite"
            )

        if not math.isfinite(expected_number):
            raise RuntimeError(
                f"{context}.{key}: expected value is non-finite"
            )

        difference = abs(
            actual_number - expected_number
        )
        maximum_difference = max(
            maximum_difference,
            difference,
        )

        tolerance = METRIC_TOLERANCE * max(
            1.0,
            abs(actual_number),
            abs(expected_number),
        )

        if difference > tolerance:
            raise RuntimeError(
                f"{context}.{key}: metric mismatch: "
                f"actual={actual_number} "
                f"expected={expected_number} "
                f"difference={difference} "
                f"tolerance={tolerance}"
            )

    return maximum_difference


def finite_positive(
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


def summarize_predictions(
    values: Sequence[float],
) -> dict[str, float | int]:
    if not values:
        raise ValueError(
            "Cannot summarize an empty prediction sequence"
        )

    data = [float(value) for value in values]

    if not all(math.isfinite(value) for value in data):
        raise ValueError(
            "Prediction sequence contains a non-finite value"
        )

    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "sample_sd": (
            statistics.stdev(data)
            if len(data) > 1
            else 0.0
        ),
        "minimum": min(data),
        "maximum": max(data),
    }


def main() -> None:
    args = parse_args()

    landmark_sha256 = sha256_file(
        args.landmark_input
    )
    locked_sha256 = sha256_file(
        args.locked_evaluation
    )

    if landmark_sha256 != EXPECTED_LANDMARK_SHA256:
        raise RuntimeError(
            "Landmark input hash mismatch: "
            f"expected {EXPECTED_LANDMARK_SHA256}, "
            f"got {landmark_sha256}"
        )

    if locked_sha256 != EXPECTED_CONFIRMATORY_SHA256:
        raise RuntimeError(
            "Locked evaluation hash mismatch: "
            f"expected {EXPECTED_CONFIRMATORY_SHA256}, "
            f"got {locked_sha256}"
        )

    runner_path = (
        REPO_ROOT
        / "scripts/evals/run_layer6_locked_confirmatory.py"
    )
    trainer_path = (
        REPO_ROOT
        / "scripts/train_layer6_deployable.py"
    )

    trainer = load_module(
        "_oncotwin_layer8_layer6_trainer",
        trainer_path,
    )
    locked = trainer.LOCKED

    locked_payload = trainer._verify_locked_artifact(
        args.locked_evaluation
    )

    training_data = trainer._build_training_data()

    records = list(training_data["records"])
    feature_matrix = training_data["feature_matrix"]
    targets = training_data["targets"]
    active_indices = list(
        training_data["active_indices"]
    )
    inactive_indices = list(
        training_data["inactive_indices"]
    )
    early_candidate_predictions = dict(
        training_data["early_candidate_predictions"]
    )

    if len(records) != 270:
        raise RuntimeError(
            f"Expected 270 Layer 6 records, got {len(records)}"
        )

    if len(active_indices) != 252:
        raise RuntimeError(
            "Expected 252 Layer 6-active records, "
            f"got {len(active_indices)}"
        )

    if len(inactive_indices) != 18:
        raise RuntimeError(
            "Expected 18 inactive records, "
            f"got {len(inactive_indices)}"
        )

    expected_repeat_results = locked_payload.get(
        "repeat_results"
    )

    if not isinstance(expected_repeat_results, list):
        raise RuntimeError(
            "Locked artifact is missing repeat_results"
        )

    if len(expected_repeat_results) != locked.N_REPEATS:
        raise RuntimeError(
            "Locked repeat count does not match current runner"
        )

    record_index_by_case = {
        str(record["case_id"]): index
        for index, record in enumerate(records)
    }

    if len(record_index_by_case) != len(records):
        raise RuntimeError(
            "Duplicate case IDs in Layer 6 training records"
        )

    repeat_rows: list[dict[str, Any]] = []
    predictions_by_case: dict[str, list[float]] = {
        str(records[index]["case_id"]): []
        for index in active_indices
    }

    repeat_reproduction: list[dict[str, Any]] = []
    global_maximum_metric_difference = 0.0

    for repeat in range(locked.N_REPEATS):
        print(
            "OOF_REPEAT_BEGIN "
            f"repeat={repeat + 1}/{locked.N_REPEATS}",
            flush=True,
        )

        outer_folds = locked.make_folds(
            records,
            active_indices,
            locked.N_OUTER_FOLDS,
            locked.BASE_SEED + repeat,
        )

        if len(outer_folds) != locked.N_OUTER_FOLDS:
            raise RuntimeError(
                f"Repeat {repeat}: unexpected outer-fold count"
            )

        flattened = [
            index
            for fold in outer_folds
            for index in fold
        ]

        if len(flattened) != len(set(flattened)):
            raise RuntimeError(
                f"Repeat {repeat}: duplicate test index across folds"
            )

        if set(flattened) != set(active_indices):
            raise RuntimeError(
                f"Repeat {repeat}: outer folds do not cover "
                "the active population exactly"
            )

        locked_predictions = dict(
            early_candidate_predictions
        )
        held_out_indices: set[int] = set()

        for outer_fold_index, test_indices in enumerate(
            outer_folds
        ):
            test_set = set(test_indices)

            if held_out_indices & test_set:
                raise RuntimeError(
                    f"Repeat {repeat}: case appeared in two "
                    "outer test folds"
                )

            held_out_indices.update(test_set)

            train_indices = [
                index
                for index in active_indices
                if index not in test_set
            ]

            calibration_seed = (
                locked.BASE_SEED
                + repeat * 100000
                + outer_fold_index * 1000
            )
            prediction_seed = (
                locked.BASE_SEED
                + 500000
                + repeat * 100000
                + outer_fold_index * 1000
            )

            calibration = (
                locked.estimate_training_calibration(
                    records,
                    feature_matrix,
                    targets,
                    train_indices,
                    early_candidate_predictions,
                    seed=calibration_seed,
                )
            )

            test_mean, test_sd = (
                locked.train_ensemble_predict(
                    feature_matrix,
                    targets,
                    train_indices,
                    test_indices,
                    seed=prediction_seed,
                )
            )

            if len(test_mean) != len(test_indices):
                raise RuntimeError(
                    "Outer-fold prediction count mismatch"
                )

            if len(test_sd) != len(test_indices):
                raise RuntimeError(
                    "Outer-fold uncertainty count mismatch"
                )

            for position, index in enumerate(
                test_indices
            ):
                record = records[index]
                case_id = str(record["case_id"])

                scaled_rate_mean = float(
                    test_mean[position]
                )
                scaled_rate_sd = max(
                    float(test_sd[position]),
                    1e-8,
                )

                neural_point = locked.decode_rate(
                    record,
                    scaled_rate_mean,
                )

                blended_point = locked.blend_volume(
                    neural_point,
                    early_candidate_predictions[index],
                )

                point = locked.apply_log_calibration(
                    blended_point,
                    calibration["log_shift"],
                )

                point = finite_positive(
                    point,
                    name=(
                        f"repeat {repeat} case {case_id} "
                        "OOF prediction"
                    ),
                )

                locked_predictions[index] = point
                predictions_by_case[case_id].append(point)

                repeat_rows.append(
                    {
                        "analysis_version": ANALYSIS_VERSION,
                        "case_id": case_id,
                        "repeat_index": repeat,
                        "repeat_number": repeat + 1,
                        "outer_fold_index": outer_fold_index,
                        "outer_fold_number": (
                            outer_fold_index + 1
                        ),
                        "held_out_from_training": True,
                        "train_case_count": len(train_indices),
                        "test_case_count": len(test_indices),
                        "calibration_seed": calibration_seed,
                        "prediction_seed": prediction_seed,
                        "ensemble_size": locked.ENSEMBLE_SIZE,
                        "fixed_epochs": locked.FIXED_EPOCHS,
                        "scaled_late_rate_mean": (
                            scaled_rate_mean
                        ),
                        "scaled_late_rate_sd": (
                            scaled_rate_sd
                        ),
                        "neural_point_ml": float(
                            neural_point
                        ),
                        "early_candidate_point_ml": float(
                            early_candidate_predictions[index]
                        ),
                        "calibration_log_shift": float(
                            calibration["log_shift"]
                        ),
                        "calibration_q80": float(
                            calibration["q80"]
                        ),
                        "calibration_q95": float(
                            calibration["q95"]
                        ),
                        "layer6_oof_point_ml": point,
                        "observed_t3_volume_ml": float(
                            record["observed"]
                        ),
                        "prediction_information_cutoff": "T1",
                        "held_out_target": "T3",
                        "patient_t3_excluded_from_training": True,
                    }
                )

        if held_out_indices != set(active_indices):
            raise RuntimeError(
                f"Repeat {repeat}: incomplete held-out coverage"
            )

        actual_overall = locked.metrics(
            records,
            locked_predictions,
            list(range(len(records))),
        )
        actual_active = locked.metrics(
            records,
            locked_predictions,
            active_indices,
        )

        expected_repeat = expected_repeat_results[repeat]

        if not isinstance(expected_repeat, Mapping):
            raise RuntimeError(
                f"Locked repeat {repeat} is not a mapping"
            )

        expected_overall = expected_repeat[
            "overall"
        ]["locked_layer6"]
        expected_active = expected_repeat[
            "early_available"
        ]["locked_layer6"]

        overall_difference = verify_numeric_mapping(
            actual_overall,
            expected_overall,
            context=(
                f"repeat[{repeat}].overall.locked_layer6"
            ),
        )
        active_difference = verify_numeric_mapping(
            actual_active,
            expected_active,
            context=(
                f"repeat[{repeat}]."
                "early_available.locked_layer6"
            ),
        )

        repeat_maximum_difference = max(
            overall_difference,
            active_difference,
        )

        global_maximum_metric_difference = max(
            global_maximum_metric_difference,
            repeat_maximum_difference,
        )

        repeat_reproduction.append(
            {
                "repeat_index": repeat,
                "repeat_number": repeat + 1,
                "outer_fold_sizes": [
                    len(fold)
                    for fold in outer_folds
                ],
                "held_out_case_count": len(
                    held_out_indices
                ),
                "overall_metrics": actual_overall,
                "early_available_metrics": actual_active,
                "maximum_absolute_metric_difference": (
                    repeat_maximum_difference
                ),
            }
        )

        print(
            "OOF_REPEAT_REPRODUCED "
            f"repeat={repeat + 1} "
            f"fold_sizes="
            f"{','.join(str(len(fold)) for fold in outer_folds)} "
            f"max_metric_difference="
            f"{repeat_maximum_difference:.12g}",
            flush=True,
        )

    expected_prediction_row_count = (
        len(active_indices) * locked.N_REPEATS
    )

    if len(repeat_rows) != expected_prediction_row_count:
        raise RuntimeError(
            "Unexpected repeated-OOF row count: "
            f"actual={len(repeat_rows)} "
            f"expected={expected_prediction_row_count}"
        )

    case_summaries: list[dict[str, Any]] = []
    aggregate_predictions: dict[int, float] = {}

    for case_id in sorted(predictions_by_case):
        values = predictions_by_case[case_id]

        if len(values) != locked.N_REPEATS:
            raise RuntimeError(
                f"Case {case_id} has {len(values)} OOF predictions; "
                f"expected {locked.N_REPEATS}"
            )

        index = record_index_by_case[case_id]
        record = records[index]
        summary = summarize_predictions(values)

        aggregate_predictions[index] = float(
            summary["mean"]
        )

        case_summaries.append(
            {
                "analysis_version": ANALYSIS_VERSION,
                "case_id": case_id,
                "activation_status": str(
                    record["activation_status"]
                ),
                "layer6_active": bool(record["active"]),
                "oof_prediction_count": int(
                    summary["count"]
                ),
                "layer6_oof_mean_point_ml": float(
                    summary["mean"]
                ),
                "layer6_oof_median_point_ml": float(
                    summary["median"]
                ),
                "layer6_oof_sample_sd_ml": float(
                    summary["sample_sd"]
                ),
                "layer6_oof_minimum_point_ml": float(
                    summary["minimum"]
                ),
                "layer6_oof_maximum_point_ml": float(
                    summary["maximum"]
                ),
                "early_candidate_point_ml": float(
                    record["early_candidate_point"]
                ),
                "observed_t3_volume_ml": float(
                    record["observed"]
                ),
                "prediction_information_cutoff": "T1",
                "held_out_target": "T3",
                "all_repeat_predictions_held_out": True,
            }
        )

    aggregate_active_metrics = locked.metrics(
        records,
        aggregate_predictions,
        active_indices,
    )

    landmark_rows = load_jsonl(
        args.landmark_input
    )

    if len(landmark_rows) != 220:
        raise RuntimeError(
            f"Expected 220 landmark rows, got {len(landmark_rows)}"
        )

    summary_by_case = {
        row["case_id"]: row
        for row in case_summaries
    }

    merged_rows: list[dict[str, Any]] = []
    landmark_indices: list[int] = []

    for landmark_row in landmark_rows:
        case_id = str(landmark_row["case_id"])

        case_summary = summary_by_case.get(case_id)

        if case_summary is None:
            raise RuntimeError(
                f"Missing OOF prediction for landmark case {case_id}"
            )

        index = record_index_by_case.get(case_id)

        if index is None:
            raise RuntimeError(
                f"Landmark case absent from Layer 6 records: {case_id}"
            )

        landmark_indices.append(index)

        landmark_target = finite_positive(
            landmark_row["t3_final_volume_ml"],
            name=f"{case_id}.t3_final_volume_ml",
        )
        layer6_target = finite_positive(
            records[index]["observed"],
            name=f"{case_id}.observed",
        )

        target_difference = abs(
            landmark_target - layer6_target
        )
        target_tolerance = 1e-12 * max(
            1.0,
            abs(landmark_target),
            abs(layer6_target),
        )

        if target_difference > target_tolerance:
            raise RuntimeError(
                f"T3 target mismatch for {case_id}: "
                f"landmark={landmark_target} "
                f"layer6={layer6_target}"
            )

        merged = dict(landmark_row)
        merged.update(
            {
                "layer6_oof_export_version": (
                    ANALYSIS_VERSION
                ),
                "layer6_oof_prior_point_ml": (
                    case_summary[
                        "layer6_oof_mean_point_ml"
                    ]
                ),
                "layer6_oof_prior_median_ml": (
                    case_summary[
                        "layer6_oof_median_point_ml"
                    ]
                ),
                "layer6_oof_prior_sd_ml": (
                    case_summary[
                        "layer6_oof_sample_sd_ml"
                    ]
                ),
                "layer6_oof_prediction_count": (
                    case_summary[
                        "oof_prediction_count"
                    ]
                ),
                "layer6_oof_all_predictions_held_out": True,
                "layer6_oof_information_cutoff": "T1",
                "layer8_update_evidence": "T2",
                "layer8_held_out_target": "T3",
                "layer6_oof_target_excluded_from_own_training": True,
            }
        )

        merged_rows.append(merged)

    if len(set(landmark_indices)) != 220:
        raise RuntimeError(
            "Merged landmark dataset contains duplicate Layer 6 indices"
        )

    landmark_aggregate_metrics = locked.metrics(
        records,
        aggregate_predictions,
        landmark_indices,
    )

    repeat_rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            int(row["repeat_index"]),
        )
    )
    case_summaries.sort(
        key=lambda row: str(row["case_id"])
    )
    merged_rows.sort(
        key=lambda row: str(row["case_id"])
    )

    atomic_write_jsonl(
        args.repeat_jsonl,
        repeat_rows,
    )
    atomic_write_jsonl(
        args.case_jsonl,
        case_summaries,
    )
    atomic_write_jsonl(
        args.merged_landmark_jsonl,
        merged_rows,
    )

    output_hashes = {
        str(args.repeat_jsonl): sha256_file(
            args.repeat_jsonl
        ),
        str(args.case_jsonl): sha256_file(
            args.case_jsonl
        ),
        str(args.merged_landmark_jsonl): sha256_file(
            args.merged_landmark_jsonl
        ),
    }

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "exploratory_repeated_oof_prior_export"
        ),
        "inputs": {
            "landmark_dataset": str(
                args.landmark_input
            ),
            "landmark_sha256": landmark_sha256,
            "locked_evaluation": str(
                args.locked_evaluation
            ),
            "locked_evaluation_sha256": (
                locked_sha256
            ),
            "locked_runner": str(
                runner_path.relative_to(REPO_ROOT)
            ),
            "locked_runner_sha256": sha256_file(
                runner_path
            ),
            "deployable_trainer": str(
                trainer_path.relative_to(REPO_ROOT)
            ),
            "deployable_trainer_sha256": (
                sha256_file(trainer_path)
            ),
        },
        "design": {
            "source_procedure": (
                "locked Layer 6 confirmatory evaluation"
            ),
            "repeats": locked.N_REPEATS,
            "outer_folds": locked.N_OUTER_FOLDS,
            "inner_folds": locked.N_INNER_FOLDS,
            "ensemble_size": locked.ENSEMBLE_SIZE,
            "fixed_epochs": locked.FIXED_EPOCHS,
            "base_seed": locked.BASE_SEED,
            "neural_weight": locked.NEURAL_WEIGHT,
            "baseline_weight": locked.BASELINE_WEIGHT,
            "point_calibration": (
                "outer-training-only inner-OOF "
                "log-multiplicative calibration"
            ),
            "one_prediction_per_case_per_repeat": True,
            "each_case_excluded_from_own_model_training": True,
        },
        "cohort": {
            "total_case_count": len(records),
            "active_case_count": len(active_indices),
            "inactive_case_count": len(
                inactive_indices
            ),
            "landmark_case_count": len(
                merged_rows
            ),
            "repeat_prediction_row_count": len(
                repeat_rows
            ),
            "case_summary_row_count": len(
                case_summaries
            ),
        },
        "locked_reproduction": {
            "repeat_count_verified": len(
                repeat_reproduction
            ),
            "metric_tolerance": METRIC_TOLERANCE,
            "maximum_absolute_metric_difference": (
                global_maximum_metric_difference
            ),
            "all_repeats_reproduced": (
                len(repeat_reproduction)
                == locked.N_REPEATS
                and global_maximum_metric_difference
                <= METRIC_TOLERANCE
            ),
            "repeats": repeat_reproduction,
        },
        "aggregate_oof_metrics": {
            "active_252": aggregate_active_metrics,
            "layer8_landmark_220": (
                landmark_aggregate_metrics
            ),
            "interpretation": (
                "Metrics of each patient's mean across 20 "
                "independently held-out Layer 6 predictions. "
                "These are exploratory and are not the locked "
                "confirmatory metrics."
            ),
        },
        "outputs": {
            "repeat_predictions": {
                "path": str(args.repeat_jsonl),
                "row_count": len(repeat_rows),
                "sha256": output_hashes[
                    str(args.repeat_jsonl)
                ],
            },
            "case_summary": {
                "path": str(args.case_jsonl),
                "row_count": len(
                    case_summaries
                ),
                "sha256": output_hashes[
                    str(args.case_jsonl)
                ],
            },
            "merged_landmarks": {
                "path": str(
                    args.merged_landmark_jsonl
                ),
                "row_count": len(merged_rows),
                "sha256": output_hashes[
                    str(args.merged_landmark_jsonl)
                ],
            },
        },
        "leakage_guards": {
            "layer6_prior_information_cutoff": "T1",
            "new_layer8_evidence": "T2",
            "held_out_target": "T3",
            "patient_t3_used_in_own_layer6_training": False,
            "deployable_full_development_predictions_used": False,
            "warning": (
                "These repeated OOF priors support exploratory "
                "Layer 8 signal analysis. A final locked updater "
                "evaluation should preserve outer-fold isolation "
                "for the complete stacked pipeline."
            ),
        },
        "decision": {
            "status": (
                "repeated_oof_layer6_priors_ready_for_"
                "exploratory_layer8_baselines"
            ),
            "next_step": (
                "compare no-update, direct T2, heuristic residual, "
                "and regularized residual-update baselines"
            ),
        },
    }

    atomic_write_json(
        args.summary_json,
        payload,
    )

    markdown = "\n".join(
        [
            "# Layer 8 repeated-OOF Layer 6 prior export",
            "",
            (
                f"- Total Layer 6 cases: `{len(records)}`"
            ),
            (
                f"- Layer 6-active cases: `{len(active_indices)}`"
            ),
            (
                f"- Layer 8 landmark cases: `{len(merged_rows)}`"
            ),
            (
                "- Held-out predictions per active case: "
                f"`{locked.N_REPEATS}`"
            ),
            (
                "- Total held-out prediction rows: "
                f"`{len(repeat_rows)}`"
            ),
            (
                "- Maximum locked-metric reproduction error: "
                f"`{global_maximum_metric_difference:.12g}`"
            ),
            (
                "- All locked repeats reproduced: "
                f"`{str(payload['locked_reproduction']['all_repeats_reproduced']).lower()}`"
            ),
            "",
            "## Temporal boundary",
            "",
            "- Layer 6 prior information cutoff: `T1`",
            "- New Layer 8 evidence: `T2`",
            "- Held-out evaluation target: `T3`",
            "",
            "## Next step",
            "",
            (
                "Compare no-update, direct-T2, heuristic residual, "
                "and regularized residual-update baselines."
            ),
            "",
            "## Evaluation caution",
            "",
            (
                "This repeated-OOF export is suitable for exploratory "
                "Layer 8 signal analysis. The final locked stacked "
                "evaluation must preserve outer-fold isolation across "
                "both the Layer 6 prior and Layer 8 updater."
            ),
        ]
    ) + "\n"

    atomic_write_text(
        args.summary_markdown,
        markdown,
    )

    print(
        "OOF_EXPORT_COUNTS "
        f"total_cases={len(records)} "
        f"active_cases={len(active_indices)} "
        f"landmark_cases={len(merged_rows)} "
        f"repeat_rows={len(repeat_rows)} "
        f"predictions_per_case={locked.N_REPEATS}"
    )

    print(
        "LOCKED_REPRODUCTION_RESULT "
        f"repeats={len(repeat_reproduction)} "
        f"max_metric_difference="
        f"{global_maximum_metric_difference:.12g} "
        f"tolerance={METRIC_TOLERANCE} "
        "status=pass"
    )

    print(
        "AGGREGATE_OOF_ACTIVE_METRICS "
        + " ".join(
            f"{key}={value}"
            for key, value
            in aggregate_active_metrics.items()
        )
    )

    print(
        "AGGREGATE_OOF_LANDMARK_METRICS "
        + " ".join(
            f"{key}={value}"
            for key, value
            in landmark_aggregate_metrics.items()
        )
    )

    print(
        "LAYER8_INFORMATION_BOUNDARY "
        "prior_through=T1 evidence=T2 target=T3 "
        "own_target_in_layer6_training=false"
    )

    print(
        "OOF_EXPORT_DECISION "
        "status=repeated_oof_layer6_priors_ready_for_"
        "exploratory_layer8_baselines"
    )

    print(
        f"OOF_REPEAT_OUTPUT path={args.repeat_jsonl}"
    )
    print(
        f"OOF_CASE_OUTPUT path={args.case_jsonl}"
    )
    print(
        "OOF_LANDMARK_OUTPUT "
        f"path={args.merged_landmark_jsonl}"
    )
    print(
        f"OOF_EXPORT_JSON path={args.summary_json}"
    )
    print(
        "OOF_EXPORT_REPORT "
        f"path={args.summary_markdown}"
    )


if __name__ == "__main__":
    main()
