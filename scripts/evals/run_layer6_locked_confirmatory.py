from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

import numpy as np

import evals.prior_stack.v1_real_data_eval as evalmod
from experiments.prior_builder.layer6_late_response import (
    LAYER6_FEATURE_NAMES,
    late_response_target_per_day,
    layer6_activation,
    layer6_feature_vector,
    predict_volume_from_late_rate,
)
from experiments.prior_builder.layer6_neural_inference import (
    Layer6NeuralConfig,
    combine_gaussian_ensemble,
    fit_layer6_neural_model,
    predict_layer6_distribution,
)


COHORT = Path(
    "data/processed/v1_prior_stack/"
    "ispy2_v1_prior_eval_cohort.jsonl"
)

STATIC_CALIBRATOR = Path(
    "configs/prior/"
    "v1_d1_constant_log_radius_posthoc_v0_1.json"
)

EARLY_UPDATER = Path(
    "configs/prior/"
    "v1_d1_continuous_early_volume_posthoc_v0_1.json"
)

EARLY_CALIBRATOR = Path(
    "configs/prior/"
    "v1_d1_early_response_constant_log_radius_"
    "posthoc_v0_1.json"
)

BASE_SEED = 2026
TARGET_RATE_REFERENCE_DAYS = 84.0

N_REPEATS = 20
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
ENSEMBLE_SIZE = 5

NEURAL_WEIGHT = 0.75
BASELINE_WEIGHT = 0.25
FIXED_EPOCHS = 28

LOCKED_CONFIG = Layer6NeuralConfig(
    hidden_dim=16,
    dropout=0.10,
    weight_decay=0.05,
    learning_rate=0.01,
    max_epochs=FIXED_EPOCHS,
    patience=30,
    min_epochs=20,
)


def metrics(
    records,
    predictions,
    indices,
):
    errors = []
    log_errors = []

    for index in indices:
        observed = max(
            float(records[index]["observed"]),
            1e-12,
        )
        predicted = max(
            float(predictions[index]),
            1e-12,
        )

        errors.append(predicted - observed)
        log_errors.append(
            math.log(predicted)
            - math.log(observed)
        )

    return {
        "n": len(indices),
        "mae": statistics.mean(
            abs(error) for error in errors
        ),
        "rmse": math.sqrt(
            statistics.mean(
                error * error for error in errors
            )
        ),
        "log_rmse": math.sqrt(
            statistics.mean(
                error * error
                for error in log_errors
            )
        ),
        "bias": statistics.mean(errors),
    }


def summarize(values):
    values = sorted(
        float(value) for value in values
    )

    def quantile(probability):
        position = (
            len(values) - 1
        ) * probability

        lower = math.floor(position)
        upper = math.ceil(position)

        if lower == upper:
            return values[lower]

        weight = position - lower

        return (
            values[lower] * (1.0 - weight)
            + values[upper] * weight
        )

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sd": (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "p10": quantile(0.10),
        "p90": quantile(0.90),
    }


def response_category(record):
    ratio = (
        record["early_volume_ml"]
        / record["baseline_volume_ml"]
    )

    if ratio < 0.50:
        return "major_shrinkage"

    if ratio < 0.85:
        return "mild_shrinkage"

    if ratio <= 1.15:
        return "stable"

    if ratio <= 1.50:
        return "mild_growth"

    return "major_growth"


def make_folds(
    records,
    indices,
    n_folds,
    seed,
):
    rng = random.Random(seed)
    groups = defaultdict(list)

    for index in indices:
        record = records[index]

        groups[
            (
                response_category(record),
                record["final_day"],
            )
        ].append(index)

    folds = [
        []
        for _ in range(n_folds)
    ]

    for group_indices in groups.values():
        rng.shuffle(group_indices)

        for position, index in enumerate(
            group_indices
        ):
            folds[
                position % n_folds
            ].append(index)

    for fold in folds:
        rng.shuffle(fold)

    return folds


def decode_rate(
    record,
    scaled_rate,
):
    return predict_volume_from_late_rate(
        record["case"],
        float(scaled_rate)
        / TARGET_RATE_REFERENCE_DAYS,
    )


def blend_volume(
    neural_prediction,
    baseline_prediction,
):
    return max(
        NEURAL_WEIGHT
        * float(neural_prediction)
        + BASELINE_WEIGHT
        * float(baseline_prediction),
        1e-12,
    )


def apply_log_calibration(
    prediction,
    log_shift,
):
    return max(
        float(prediction)
        * math.exp(float(log_shift)),
        1e-12,
    )


def train_ensemble_predict(
    feature_matrix,
    targets,
    train_indices,
    prediction_indices,
    seed,
):
    ensemble_means = []
    ensemble_sds = []

    for ensemble_index in range(
        ENSEMBLE_SIZE
    ):
        fitted = fit_layer6_neural_model(
            feature_matrix[train_indices],
            targets[train_indices],
            config=LOCKED_CONFIG,
            seed=(
                seed
                + ensemble_index
            ),
            fixed_epochs=FIXED_EPOCHS,
        )

        mean, sd = (
            predict_layer6_distribution(
                fitted,
                feature_matrix[
                    prediction_indices
                ],
            )
        )

        ensemble_means.append(mean)
        ensemble_sds.append(sd)

    return combine_gaussian_ensemble(
        ensemble_means,
        ensemble_sds,
    )


def estimate_training_calibration(
    records,
    feature_matrix,
    targets,
    train_indices,
    early_candidate_predictions,
    seed,
):
    inner_folds = make_folds(
        records,
        train_indices,
        N_INNER_FOLDS,
        seed,
    )

    oof_rate_mean = {}
    oof_rate_sd = {}
    oof_blended_point = {}

    for inner_fold_index, validation_indices in enumerate(
        inner_folds
    ):
        validation_set = set(
            validation_indices
        )

        fit_indices = [
            index
            for index in train_indices
            if index not in validation_set
        ]

        mean, sd = train_ensemble_predict(
            feature_matrix,
            targets,
            fit_indices,
            validation_indices,
            seed=(
                seed
                + 10000
                + inner_fold_index * 100
            ),
        )

        for position, index in enumerate(
            validation_indices
        ):
            oof_rate_mean[index] = float(
                mean[position]
            )
            oof_rate_sd[index] = max(
                float(sd[position]),
                1e-8,
            )

            neural_point = decode_rate(
                records[index],
                mean[position],
            )

            oof_blended_point[index] = (
                blend_volume(
                    neural_point,
                    early_candidate_predictions[
                        index
                    ],
                )
            )

    log_shift = statistics.mean(
        math.log(
            max(
                records[index]["observed"],
                1e-12,
            )
        )
        - math.log(
            max(
                oof_blended_point[index],
                1e-12,
            )
        )
        for index in train_indices
    )

    standardized_errors = [
        abs(
            targets[index]
            - oof_rate_mean[index]
        )
        / max(
            oof_rate_sd[index],
            1e-8,
        )
        for index in train_indices
    ]

    q80 = float(
        np.quantile(
            standardized_errors,
            0.80,
            method="higher",
        )
    )

    q95 = float(
        np.quantile(
            standardized_errors,
            0.95,
            method="higher",
        )
    )

    q80 = min(
        max(q80, 0.25),
        6.0,
    )
    q95 = min(
        max(q95, 0.50),
        10.0,
    )

    return {
        "log_shift": float(log_shift),
        "q80": q80,
        "q95": q95,
    }


def interval_metrics(
    records,
    indices,
    lower80,
    upper80,
    lower95,
    upper95,
):
    coverage80 = []
    coverage95 = []
    widths80 = []
    widths95 = []

    for index in indices:
        observed = records[index]["observed"]

        coverage80.append(
            lower80[index]
            <= observed
            <= upper80[index]
        )

        coverage95.append(
            lower95[index]
            <= observed
            <= upper95[index]
        )

        widths80.append(
            upper80[index]
            - lower80[index]
        )

        widths95.append(
            upper95[index]
            - lower95[index]
        )

    return {
        "n": len(indices),
        "coverage_80": (
            sum(coverage80)
            / len(coverage80)
        ),
        "coverage_95": (
            sum(coverage95)
            / len(coverage95)
        ),
        "mean_width_80_ml": (
            statistics.mean(widths80)
        ),
        "mean_width_95_ml": (
            statistics.mean(widths95)
        ),
    }


def main(output_path):
    print(
        "Loading frozen D1 predictions and "
        "building locked Layer 6 data..."
    )

    evaluation = evalmod.run_real_data_eval(
        COHORT,
        n_samples=2000,
        seed=BASE_SEED,
        interval_calibrator_path=(
            STATIC_CALIBRATOR
        ),
        allow_candidate_interval_calibrator=True,
        early_response_updater_path=(
            EARLY_UPDATER
        ),
        early_response_interval_calibrator_path=(
            EARLY_CALIBRATOR
        ),
        allow_candidate_early_response_updater=True,
    )

    evaluated_by_id = {
        str(row["case_id"]): row
        for row in evaluation[
            "case_predictions"
        ]
    }

    cases = evalmod.load_real_cohort(
        COHORT,
        allow_demo_data=False,
    )

    records = []
    feature_rows = []
    target_rows = []

    for case in cases:
        case_id = str(case["case_id"])
        evaluated = evaluated_by_id[
            case_id
        ]
        predictions = evaluated[
            "predictions"
        ]

        activation = layer6_activation(
            case
        )

        record = {
            "case": case,
            "case_id": case_id,
            "baseline_volume_ml": float(
                case["baseline_volume_ml"]
            ),
            "early_volume_ml": (
                float(case["early_volume_ml"])
                if activation["active"]
                else None
            ),
            "final_day": int(
                round(float(case["final_day"]))
            ),
            "observed": float(
                evaluated[
                    "observed_final_volume_ml"
                ]
            ),
            "active": bool(
                activation["active"]
            ),
            "current_point": float(
                predictions[
                    "layer4_mri_qc"
                ]["point_ml"]
            ),
            "early_candidate_point": float(
                predictions[
                    evalmod.EARLY_RESPONSE_CANDIDATE
                ]["point_ml"]
            ),
            "early_candidate_lower_80": float(
                predictions[
                    evalmod.EARLY_RESPONSE_CANDIDATE
                ]["lower_80_ml"]
            ),
            "early_candidate_upper_80": float(
                predictions[
                    evalmod.EARLY_RESPONSE_CANDIDATE
                ]["upper_80_ml"]
            ),
            "early_candidate_lower_95": float(
                predictions[
                    evalmod.EARLY_RESPONSE_CANDIDATE
                ]["lower_95_ml"]
            ),
            "early_candidate_upper_95": float(
                predictions[
                    evalmod.EARLY_RESPONSE_CANDIDATE
                ]["upper_95_ml"]
            ),
        }

        records.append(record)

        if not activation["active"]:
            feature_rows.append(
                [float("nan")]
                * len(LAYER6_FEATURE_NAMES)
            )
            target_rows.append(
                float("nan")
            )
            continue

        static_prediction = predictions[
            evalmod.CALIBRATED_LAYER4_CANDIDATE
        ]

        feature_rows.append(
            layer6_feature_vector(
                case,
                static_prediction,
            )
        )

        target_rows.append(
            late_response_target_per_day(
                case,
                record["observed"],
            )
            * TARGET_RATE_REFERENCE_DAYS
        )

    feature_matrix = np.asarray(
        feature_rows,
        dtype=float,
    )

    targets = np.asarray(
        target_rows,
        dtype=float,
    )

    all_indices = list(
        range(len(records))
    )

    active_indices = [
        index
        for index, record in enumerate(
            records
        )
        if record["active"]
    ]

    inactive_indices = [
        index
        for index, record in enumerate(
            records
        )
        if not record["active"]
    ]

    current_predictions = {
        index: record["current_point"]
        for index, record in enumerate(
            records
        )
    }

    early_candidate_predictions = {
        index: record[
            "early_candidate_point"
        ]
        for index, record in enumerate(
            records
        )
    }

    print("\n=== LOCKED SPECIFICATION ===")
    print(
        "architecture:",
        "hidden=16 dropout=0.10 "
        "weight_decay=0.05",
    )
    print("epochs:", FIXED_EPOCHS)
    print("ensemble size:", ENSEMBLE_SIZE)
    print(
        "volume blend:",
        f"{NEURAL_WEIGHT:.2f} neural + "
        f"{BASELINE_WEIGHT:.2f} D1",
    )
    print(
        "calibration:",
        "inner-OOF log multiplicative",
    )
    print(
        "intervals:",
        "inner-OOF standardized residuals",
    )

    print("\n=== DATASET ===")
    print("cases:", len(records))
    print(
        "Layer 6 active:",
        len(active_indices),
    )
    print(
        "Layer 6 inactive:",
        len(inactive_indices),
    )

    repeat_results = []
    interval_results = []
    calibration_results = []

    for repeat in range(
        N_REPEATS
    ):
        print(
            f"Confirmatory repeat "
            f"{repeat + 1}/{N_REPEATS}"
        )

        outer_folds = make_folds(
            records,
            active_indices,
            N_OUTER_FOLDS,
            BASE_SEED + repeat,
        )

        locked_predictions = dict(
            early_candidate_predictions
        )

        lower80 = {}
        upper80 = {}
        lower95 = {}
        upper95 = {}

        repeat_calibration = []

        for outer_fold_index, test_indices in enumerate(
            outer_folds
        ):
            print(
                f"  outer fold "
                f"{outer_fold_index + 1}/"
                f"{N_OUTER_FOLDS}"
            )

            test_set = set(
                test_indices
            )

            train_indices = [
                index
                for index in active_indices
                if index not in test_set
            ]

            calibration = (
                estimate_training_calibration(
                    records,
                    feature_matrix,
                    targets,
                    train_indices,
                    early_candidate_predictions,
                    seed=(
                        BASE_SEED
                        + repeat * 100000
                        + outer_fold_index * 1000
                    ),
                )
            )

            repeat_calibration.append(
                calibration
            )

            test_mean, test_sd = (
                train_ensemble_predict(
                    feature_matrix,
                    targets,
                    train_indices,
                    test_indices,
                    seed=(
                        BASE_SEED
                        + 500000
                        + repeat * 100000
                        + outer_fold_index * 1000
                    ),
                )
            )

            for position, index in enumerate(
                test_indices
            ):
                neural_point = decode_rate(
                    records[index],
                    test_mean[position],
                )

                blended_point = blend_volume(
                    neural_point,
                    early_candidate_predictions[
                        index
                    ],
                )

                locked_predictions[index] = (
                    apply_log_calibration(
                        blended_point,
                        calibration[
                            "log_shift"
                        ],
                    )
                )

                neural_lower80 = decode_rate(
                    records[index],
                    (
                        test_mean[position]
                        - calibration["q80"]
                        * test_sd[position]
                    ),
                )

                neural_upper80 = decode_rate(
                    records[index],
                    (
                        test_mean[position]
                        + calibration["q80"]
                        * test_sd[position]
                    ),
                )

                neural_lower95 = decode_rate(
                    records[index],
                    (
                        test_mean[position]
                        - calibration["q95"]
                        * test_sd[position]
                    ),
                )

                neural_upper95 = decode_rate(
                    records[index],
                    (
                        test_mean[position]
                        + calibration["q95"]
                        * test_sd[position]
                    ),
                )

                lower80[index] = (
                    apply_log_calibration(
                        blend_volume(
                            neural_lower80,
                            records[index][
                                "early_candidate_lower_80"
                            ],
                        ),
                        calibration[
                            "log_shift"
                        ],
                    )
                )

                upper80[index] = (
                    apply_log_calibration(
                        blend_volume(
                            neural_upper80,
                            records[index][
                                "early_candidate_upper_80"
                            ],
                        ),
                        calibration[
                            "log_shift"
                        ],
                    )
                )

                lower95[index] = (
                    apply_log_calibration(
                        blend_volume(
                            neural_lower95,
                            records[index][
                                "early_candidate_lower_95"
                            ],
                        ),
                        calibration[
                            "log_shift"
                        ],
                    )
                )

                upper95[index] = (
                    apply_log_calibration(
                        blend_volume(
                            neural_upper95,
                            records[index][
                                "early_candidate_upper_95"
                            ],
                        ),
                        calibration[
                            "log_shift"
                        ],
                    )
                )

        repeat_results.append(
            {
                "overall": {
                    "current": metrics(
                        records,
                        current_predictions,
                        all_indices,
                    ),
                    "early_candidate": metrics(
                        records,
                        early_candidate_predictions,
                        all_indices,
                    ),
                    "locked_layer6": metrics(
                        records,
                        locked_predictions,
                        all_indices,
                    ),
                },
                "early_available": {
                    "current": metrics(
                        records,
                        current_predictions,
                        active_indices,
                    ),
                    "early_candidate": metrics(
                        records,
                        early_candidate_predictions,
                        active_indices,
                    ),
                    "locked_layer6": metrics(
                        records,
                        locked_predictions,
                        active_indices,
                    ),
                },
            }
        )

        interval_results.append(
            interval_metrics(
                records,
                active_indices,
                lower80,
                upper80,
                lower95,
                upper95,
            )
        )

        calibration_results.extend(
            repeat_calibration
        )

    def print_group(group_name):
        print(
            f"\n=== CONFIRMATORY: "
            f"{group_name.upper()} ==="
        )

        for model_name in (
            "current",
            "early_candidate",
            "locked_layer6",
        ):
            print(f"\n{model_name}:")

            for metric_name in (
                "mae",
                "rmse",
                "log_rmse",
                "bias",
            ):
                result = summarize(
                    [
                        repeat[group_name][
                            model_name
                        ][metric_name]
                        for repeat
                        in repeat_results
                    ]
                )

                print(
                    f"  {metric_name:8s} "
                    f"mean={result['mean']:.5f} "
                    f"sd={result['sd']:.5f} "
                    f"p10={result['p10']:.5f} "
                    f"p90={result['p90']:.5f}"
                )

        print(
            "\nLocked Layer 6 gain versus "
            "early candidate:"
        )

        for metric_name in (
            "mae",
            "rmse",
            "log_rmse",
        ):
            gains = [
                repeat[group_name][
                    "early_candidate"
                ][metric_name]
                - repeat[group_name][
                    "locked_layer6"
                ][metric_name]
                for repeat
                in repeat_results
            ]

            result = summarize(gains)

            print(
                f"  {metric_name:8s} "
                f"gain={result['mean']:+.5f} "
                f"p10={result['p10']:+.5f} "
                f"p90={result['p90']:+.5f} "
                f"win={sum(value > 0 for value in gains) / len(gains):.3f}"
            )

    print_group("overall")
    print_group("early_available")

    print(
        "\n=== CONFIRMATORY INTERVAL CALIBRATION ==="
    )

    for metric_name in (
        "coverage_80",
        "coverage_95",
        "mean_width_80_ml",
        "mean_width_95_ml",
    ):
        result = summarize(
            [
                item[metric_name]
                for item in interval_results
            ]
        )

        print(
            f"{metric_name:22s} "
            f"mean={result['mean']:.5f} "
            f"sd={result['sd']:.5f} "
            f"p10={result['p10']:.5f} "
            f"p90={result['p90']:.5f}"
        )

    print(
        "\n=== TRAINING-FOLD CALIBRATION ==="
    )

    for calibration_name in (
        "log_shift",
        "q80",
        "q95",
    ):
        result = summarize(
            [
                item[calibration_name]
                for item
                in calibration_results
            ]
        )

        print(
            f"{calibration_name:12s} "
            f"mean={result['mean']:+.5f} "
            f"sd={result['sd']:.5f} "
            f"p10={result['p10']:+.5f} "
            f"p90={result['p90']:+.5f}"
        )

    artifact = {
        "analysis_version": (
            "oncotwin_layer6_locked_"
            "confirmatory_v1"
        ),
        "status": (
            "locked_confirmatory_evaluation"
        ),
        "cohort_path": str(COHORT),
        "cohort_sha256": hashlib.sha256(
            COHORT.read_bytes()
        ).hexdigest(),
        "feature_names": list(
            LAYER6_FEATURE_NAMES
        ),
        "locked_specification": {
            "neural_config": asdict(
                LOCKED_CONFIG
            ),
            "fixed_epochs": FIXED_EPOCHS,
            "ensemble_size": ENSEMBLE_SIZE,
            "neural_weight": NEURAL_WEIGHT,
            "baseline_weight": (
                BASELINE_WEIGHT
            ),
            "blend_space": "volume",
            "point_calibration": (
                "inner_oof_log_multiplicative"
            ),
            "interval_calibration": (
                "inner_oof_standardized_"
                "absolute_rate_residual"
            ),
            "inactive_fallback": (
                evalmod.EARLY_RESPONSE_CANDIDATE
            ),
        },
        "design": {
            "repeats": N_REPEATS,
            "outer_folds": N_OUTER_FOLDS,
            "inner_folds": N_INNER_FOLDS,
            "selection_performed": False,
        },
        "dataset": {
            "case_count": len(records),
            "active_count": len(
                active_indices
            ),
            "inactive_count": len(
                inactive_indices
            ),
        },
        "repeat_results": repeat_results,
        "interval_results": (
            interval_results
        ),
        "calibration_results": (
            calibration_results
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n=== CONFIRMATORY ARTIFACT ==="
    )
    print(output_path)
    print(
        "sha256:",
        hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest(),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: "
            "run_layer6_locked_confirmatory.py "
            "<output.json>"
        )

    main(Path(sys.argv[1]))
