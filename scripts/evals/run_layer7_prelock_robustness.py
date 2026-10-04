#!/usr/bin/env python3
"""Run Layer 7 fail-closed robustness and benchmark analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.prior_stack.v1_layer7_real_data_eval import (
    run_layer7_real_data_eval,
)
from evals.prior_stack.v1_real_data_eval import (
    CALIBRATED_LAYER4_CANDIDATE,
    EARLY_RESPONSE_CANDIDATE,
    LAYER6_CANDIDATE,
    run_real_data_eval,
)
from experiments.twin_runtime.layer7_candidate import (
    LAYER7_FAIL_CLOSED_CANDIDATE_NAME,
    apply_layer7_fail_closed_candidate,
)


CANDIDATE_POLICY = Path(
    "configs/prior/"
    "layer7_fail_closed_candidate_v0_2.json"
)
PARTICLE_POLICY = Path(
    "configs/prior/"
    "layer7_baseline_particle_policy_v0_1.json"
)
EVAL_POLICY = Path(
    "configs/prior/"
    "layer7_exploratory_eval_policy_v0_1.json"
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
LAYER6_ARTIFACT = Path(
    "artifacts/prior_builder/layer6/"
    "v1_late_response_ensemble"
)
LAYER6_CLOSURE = Path(
    "artifacts/prior_builder/layer6/"
    "layer6_runtime_closure_v1.json"
)

OUTPUT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_prelock_robustness_v0_2.json"
)
REPORT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_prelock_robustness_v0_2.md"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def finite(
    value: object,
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


def metrics(
    rows: Sequence[Mapping[str, object]],
    prediction_key: str,
) -> dict[str, float | int]:
    if not rows:
        raise ValueError(
            "metrics requires at least one row"
        )

    absolute_errors = []
    squared_errors = []
    squared_log_errors = []
    percentage_errors = []
    widths80 = []
    widths95 = []
    covered80 = 0
    covered95 = 0

    for row in rows:
        observed = finite(
            row["observed_final_volume_ml"],
            "observed final",
        )
        prediction = row[prediction_key]

        point = finite(
            prediction["point_ml"],
            "point",
        )
        lower80 = finite(
            prediction["lower_80_ml"],
            "lower80",
        )
        upper80 = finite(
            prediction["upper_80_ml"],
            "upper80",
        )
        lower95 = finite(
            prediction["lower_95_ml"],
            "lower95",
        )
        upper95 = finite(
            prediction["upper_95_ml"],
            "upper95",
        )

        error = point - observed
        absolute_errors.append(abs(error))
        squared_errors.append(error * error)
        squared_log_errors.append(
            (
                math.log(max(point, 1e-8))
                - math.log(
                    max(observed, 1e-8)
                )
            )
            ** 2
        )
        percentage_errors.append(
            abs(error)
            / max(observed, 1e-8)
        )
        widths80.append(
            upper80 - lower80
        )
        widths95.append(
            upper95 - lower95
        )
        covered80 += int(
            lower80 <= observed <= upper80
        )
        covered95 += int(
            lower95 <= observed <= upper95
        )

    n = len(rows)

    return {
        "n": n,
        "mae_ml": statistics.fmean(
            absolute_errors
        ),
        "rmse_ml": math.sqrt(
            statistics.fmean(squared_errors)
        ),
        "log_volume_rmse": math.sqrt(
            statistics.fmean(
                squared_log_errors
            )
        ),
        "mape": statistics.fmean(
            percentage_errors
        ),
        "median_absolute_error_ml": (
            statistics.median(
                absolute_errors
            )
        ),
        "coverage_80": covered80 / n,
        "coverage_95": covered95 / n,
        "mean_width_80_ml": (
            statistics.fmean(widths80)
        ),
        "mean_width_95_ml": (
            statistics.fmean(widths95)
        ),
        "maximum_absolute_error_ml": max(
            absolute_errors
        ),
    }


def quantile(
    values: Sequence[float],
    probability: float,
) -> float:
    ordered = sorted(values)
    index = round(
        probability
        * (len(ordered) - 1)
    )
    return ordered[index]


def distribution(
    values: Sequence[float],
) -> dict[str, float]:
    if not values:
        raise ValueError(
            "distribution requires values"
        )

    return {
        "minimum": min(values),
        "p05": quantile(values, 0.05),
        "median": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


candidate_policy = json.loads(
    CANDIDATE_POLICY.read_text(
        encoding="utf-8"
    )
)
particle_counts = [
    int(value)
    for value in candidate_policy[
        "robustness_grid"
    ]["particle_counts"]
]
seeds = [
    int(value)
    for value in candidate_policy[
        "robustness_grid"
    ]["seeds"]
]
threshold = finite(
    candidate_policy["safety"][
        "ess_threshold_fraction"
    ],
    "ESS threshold",
)
accepted_statuses = tuple(
    str(value)
    for value in candidate_policy[
        "safety"
    ]["accepted_health_statuses"]
)

run_summaries = []
case_statuses: dict[
    str,
    list[dict[str, object]],
] = defaultdict(list)
case_points_by_particle_count: dict[
    int,
    dict[str, list[float]],
] = defaultdict(
    lambda: defaultdict(list)
)

for particle_count in particle_counts:
    for seed in seeds:
        print(
            "RUN_START:",
            {
                "particle_count": particle_count,
                "seed": seed,
            },
            flush=True,
        )

        raw_result = (
            run_layer7_real_data_eval(
                COHORT,
                particle_count=particle_count,
                seed=seed,
                particle_policy_path=(
                    PARTICLE_POLICY
                ),
                eval_policy_path=(
                    EVAL_POLICY
                ),
            )
        )

        candidate_rows = []
        status_counts = Counter()
        fallback_ids = []
        exact_missing_count = 0

        for row in raw_result[
            "case_results"
        ]:
            candidate = (
                apply_layer7_fail_closed_candidate(
                    row,
                    ess_threshold_fraction=(
                        threshold
                    ),
                    accepted_health_statuses=(
                        accepted_statuses
                    ),
                )
            )

            status = candidate[
                "layer7"
            ]["status"]
            status_counts[status] += 1

            if candidate[
                "layer7"
            ]["fallback_used"]:
                fallback_ids.append(
                    str(row["case_id"])
                )

            if row["early_day"] is None:
                if {
                    key: candidate[key]
                    for key in (
                        "point_ml",
                        "lower_80_ml",
                        "upper_80_ml",
                        "lower_95_ml",
                        "upper_95_ml",
                    )
                } != row["prior_prediction"]:
                    raise RuntimeError(
                        "Early-missing case did not "
                        "exactly retain prior: "
                        f"{row['case_id']}"
                    )
                exact_missing_count += 1

            candidate_rows.append(
                {
                    "case_id": row["case_id"],
                    "observed_final_volume_ml": (
                        row[
                            "observed_final_volume_ml"
                        ]
                    ),
                    "early_available": (
                        row["early_day"] is not None
                    ),
                    "prior": row[
                        "prior_prediction"
                    ],
                    "raw": row[
                        "posterior_prediction"
                    ],
                    "safe": candidate,
                }
            )

            if row["early_day"] is not None:
                case_id = str(
                    row["case_id"]
                )
                point = finite(
                    candidate["point_ml"],
                    "safe point",
                )
                case_points_by_particle_count[
                    particle_count
                ][case_id].append(point)

                case_statuses[case_id].append(
                    {
                        "particle_count": (
                            particle_count
                        ),
                        "seed": seed,
                        "status": status,
                        "fallback_used": (
                            candidate[
                                "layer7"
                            ]["fallback_used"]
                        ),
                        "raw_ess_fraction": (
                            candidate[
                                "layer7"
                            ][
                                "raw_ess_fraction"
                            ]
                        ),
                    }
                )

        if exact_missing_count != 18:
            raise RuntimeError(
                "Expected 18 exact no-update cases, "
                f"got {exact_missing_count}"
            )

        updated_rows = [
            row
            for row in candidate_rows
            if row["early_available"]
        ]

        summary = {
            "particle_count": particle_count,
            "seed": seed,
            "case_count": len(
                candidate_rows
            ),
            "updated_case_count": len(
                updated_rows
            ),
            "exact_no_update_count": (
                exact_missing_count
            ),
            "status_counts": dict(
                sorted(
                    status_counts.items()
                )
            ),
            "fallback_case_ids": sorted(
                fallback_ids
            ),
            "metrics": {
                "prior": metrics(
                    updated_rows,
                    "prior",
                ),
                "raw_posterior": metrics(
                    updated_rows,
                    "raw",
                ),
                "fail_closed": metrics(
                    updated_rows,
                    "safe",
                ),
            },
        }
        run_summaries.append(summary)

        print(
            "RUN_COMPLETE:",
            {
                "particle_count": (
                    particle_count
                ),
                "seed": seed,
                "fallback_count": len(
                    fallback_ids
                )
                - exact_missing_count,
                "fail_closed_metrics": (
                    summary["metrics"][
                        "fail_closed"
                    ]
                ),
            },
            flush=True,
        )

        del raw_result
        del candidate_rows
        gc.collect()

expected_runs = (
    len(particle_counts) * len(seeds)
)
if len(run_summaries) != expected_runs:
    raise RuntimeError(
        "Robustness grid did not complete"
    )

metric_names = (
    "mae_ml",
    "rmse_ml",
    "log_volume_rmse",
    "mape",
    "coverage_80",
    "coverage_95",
    "mean_width_80_ml",
    "mean_width_95_ml",
    "maximum_absolute_error_ml",
)

aggregate_by_particle_count = {}

for particle_count in particle_counts:
    selected = [
        run
        for run in run_summaries
        if run["particle_count"]
        == particle_count
    ]

    aggregate_by_particle_count[
        str(particle_count)
    ] = {
        "run_count": len(selected),
        "fallback_count": distribution(
            [
                float(
                    run["status_counts"].get(
                        "fallback_to_prior_"
                        "unhealthy_posterior",
                        0,
                    )
                )
                for run in selected
            ]
        ),
        "fail_closed_metrics": {
            metric: distribution(
                [
                    finite(
                        run["metrics"][
                            "fail_closed"
                        ][metric],
                        metric,
                    )
                    for run in selected
                ]
            )
            for metric in metric_names
        },
    }

prediction_stability = {}

for particle_count in particle_counts:
    ranges = []
    standard_deviations = []
    unstable_cases = []

    for case_id, points in (
        case_points_by_particle_count[
            particle_count
        ].items()
    ):
        if len(points) != len(seeds):
            raise RuntimeError(
                "Missing seed result for "
                f"{case_id} at "
                f"{particle_count} particles"
            )

        point_range = max(points) - min(
            points
        )
        point_sd = statistics.pstdev(
            points
        )
        ranges.append(point_range)
        standard_deviations.append(
            point_sd
        )

        unstable_cases.append(
            {
                "case_id": case_id,
                "minimum_point_ml": min(
                    points
                ),
                "maximum_point_ml": max(
                    points
                ),
                "range_ml": point_range,
                "population_sd_ml": (
                    point_sd
                ),
            }
        )

    unstable_cases.sort(
        key=lambda row: row["range_ml"],
        reverse=True,
    )

    prediction_stability[
        str(particle_count)
    ] = {
        "case_count": len(ranges),
        "point_range_ml": distribution(
            ranges
        ),
        "point_population_sd_ml": (
            distribution(
                standard_deviations
            )
        ),
        "top_20_unstable_cases": (
            unstable_cases[:20]
        ),
    }

fallback_frequency = []

for case_id, statuses in (
    case_statuses.items()
):
    fallback_count = sum(
        bool(row["fallback_used"])
        for row in statuses
    )
    status_names = sorted(
        {
            str(row["status"])
            for row in statuses
        }
    )

    fallback_frequency.append(
        {
            "case_id": case_id,
            "fallback_count": (
                fallback_count
            ),
            "run_count": len(statuses),
            "fallback_fraction": (
                fallback_count
                / len(statuses)
            ),
            "status_names": (
                status_names
            ),
            "status_changed_across_runs": (
                len(status_names) > 1
            ),
            "minimum_raw_ess_fraction": min(
                finite(
                    row["raw_ess_fraction"],
                    "raw ESS",
                )
                for row in statuses
            ),
            "maximum_raw_ess_fraction": max(
                finite(
                    row["raw_ess_fraction"],
                    "raw ESS",
                )
                for row in statuses
            ),
        }
    )

fallback_frequency.sort(
    key=lambda row: (
        row["fallback_fraction"],
        -row[
            "minimum_raw_ess_fraction"
        ],
    ),
    reverse=True,
)

layer6_eval = run_real_data_eval(
    COHORT,
    n_samples=2000,
    seed=2026,
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
    layer6_artifact_dir=(
        LAYER6_ARTIFACT
    ),
)

benchmark_rows = []

for row in layer6_eval[
    "case_predictions"
]:
    predictions = row["predictions"]
    layer6 = predictions[
        LAYER6_CANDIDATE
    ]

    if (
        layer6["layer6"]["status"]
        != "active"
    ):
        continue

    benchmark_rows.append(
        {
            "case_id": row["case_id"],
            "observed_final_volume_ml": (
                row[
                    "observed_final_volume_ml"
                ]
            ),
            "static_layer4": predictions[
                CALIBRATED_LAYER4_CANDIDATE
            ],
            "d1_early_response": predictions[
                EARLY_RESPONSE_CANDIDATE
            ],
            "layer6": layer6,
        }
    )

if len(benchmark_rows) != 252:
    raise RuntimeError(
        "Expected 252 Layer 6 active benchmark cases, "
        f"got {len(benchmark_rows)}"
    )

benchmark_metrics = {
    "static_layer4": metrics(
        benchmark_rows,
        "static_layer4",
    ),
    "d1_early_response": metrics(
        benchmark_rows,
        "d1_early_response",
    ),
    "layer6": metrics(
        benchmark_rows,
        "layer6",
    ),
}

best_particle_summary = {
    particle_count: {
        metric: statistics.fmean(
            run["metrics"][
                "fail_closed"
            ][metric]
            for run in run_summaries
            if run["particle_count"]
            == particle_count
        )
        for metric in metric_names
    }
    for particle_count in particle_counts
}

result = {
    "analysis_version": (
        "oncotwin_layer7_prelock_robustness_v0_2"
    ),
    "status": (
        "layer7_prelock_robustness_complete"
    ),
    "interpretation": (
        "Exploratory Monte Carlo robustness and benchmark "
        "analysis for the predeclared fail-closed Layer 7 "
        "candidate. No confirmatory claim is made."
    ),
    "selection_performed": False,
    "configuration": {
        "candidate_policy_path": str(
            CANDIDATE_POLICY
        ),
        "candidate_policy_sha256": (
            sha256_file(
                CANDIDATE_POLICY
            )
        ),
        "particle_policy_path": str(
            PARTICLE_POLICY
        ),
        "particle_policy_sha256": (
            sha256_file(
                PARTICLE_POLICY
            )
        ),
        "eval_policy_path": str(
            EVAL_POLICY
        ),
        "eval_policy_sha256": (
            sha256_file(EVAL_POLICY)
        ),
        "cohort_path": str(COHORT),
        "cohort_sha256": (
            sha256_file(COHORT)
        ),
        "candidate_name": (
            LAYER7_FAIL_CLOSED_CANDIDATE_NAME
        ),
        "ess_threshold_fraction": (
            threshold
        ),
        "particle_counts": (
            particle_counts
        ),
        "seeds": seeds,
        "run_count": expected_runs,
    },
    "run_summaries": run_summaries,
    "aggregate_by_particle_count": (
        aggregate_by_particle_count
    ),
    "prediction_stability": (
        prediction_stability
    ),
    "fallback_frequency": (
        fallback_frequency
    ),
    "status_flip_case_count": sum(
        bool(
            row[
                "status_changed_across_runs"
            ]
        )
        for row in fallback_frequency
    ),
    "always_fallback_case_ids": [
        row["case_id"]
        for row in fallback_frequency
        if math.isclose(
            row["fallback_fraction"],
            1.0,
        )
    ],
    "never_fallback_case_ids": [
        row["case_id"]
        for row in fallback_frequency
        if math.isclose(
            row["fallback_fraction"],
            0.0,
        )
    ],
    "benchmark_same_252_cases": {
        "metrics": benchmark_metrics,
        "source_layer6_closure": {
            "path": str(
                LAYER6_CLOSURE
            ),
            "sha256": (
                sha256_file(
                    LAYER6_CLOSURE
                )
            ),
        },
    },
    "mean_fail_closed_metrics_by_particle_count": (
        best_particle_summary
    ),
    "requirements": {
        "all_robustness_runs_completed": (
            len(run_summaries)
            == expected_runs
        ),
        "all_runs_evaluated_270_cases": all(
            run["case_count"] == 270
            for run in run_summaries
        ),
        "all_runs_updated_252_cases": all(
            run[
                "updated_case_count"
            ]
            == 252
            for run in run_summaries
        ),
        "all_runs_exact_no_update_18": all(
            run[
                "exact_no_update_count"
            ]
            == 18
            for run in run_summaries
        ),
        "final_outcome_used_for_selection": (
            False
        ),
        "threshold_selected_from_outcomes": (
            False
        ),
        "tempering_selected": False,
        "confirmatory_evaluation": False,
    },
    "limitations": list(
        candidate_policy[
            "limitations"
        ]
    ),
}

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)
OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
)

lines = [
    "# Layer 7 pre-lock robustness",
    "",
    (
        "**Exploratory robustness analysis; not "
        "confirmatory evidence.**"
    ),
    "",
    "## Fail-closed metrics by particle count",
    "",
    "| Particles | MAE mean | RMSE mean | Log RMSE mean | 80% coverage mean | 95% coverage mean | Max error mean |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for particle_count in particle_counts:
    values = best_particle_summary[
        particle_count
    ]
    lines.append(
        f"| {particle_count} | "
        f"{values['mae_ml']} | "
        f"{values['rmse_ml']} | "
        f"{values['log_volume_rmse']} | "
        f"{values['coverage_80']} | "
        f"{values['coverage_95']} | "
        f"{values['maximum_absolute_error_ml']} |"
    )

lines.extend(
    [
        "",
        "## Existing-candidate benchmark on the same 252 cases",
        "",
        "| Candidate | MAE | RMSE | Log RMSE | 80% coverage | 95% coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
)

for name, values in (
    benchmark_metrics.items()
):
    lines.append(
        f"| {name} | "
        f"{values['mae_ml']} | "
        f"{values['rmse_ml']} | "
        f"{values['log_volume_rmse']} | "
        f"{values['coverage_80']} | "
        f"{values['coverage_95']} |"
    )

lines.extend(
    [
        "",
        "## Posterior-health stability",
        "",
        (
            "- Cases whose fallback status changed across "
            f"runs: {result['status_flip_case_count']}"
        ),
        (
            "- Cases that fell back in every run: "
            f"{result['always_fallback_case_ids']}"
        ),
        "",
        "## Monte Carlo prediction stability",
        "",
        "| Particles | Median point range | 95th-percentile point range | Maximum point range |",
        "| ---: | ---: | ---: | ---: |",
    ]
)

for particle_count in particle_counts:
    values = prediction_stability[
        str(particle_count)
    ]["point_range_ml"]
    lines.append(
        f"| {particle_count} | "
        f"{values['median']} | "
        f"{values['p95']} | "
        f"{values['maximum']} |"
    )

lines.extend(
    [
        "",
        "## Next decision",
        "",
        (
            "Use this artifact to choose the smallest "
            "particle count that provides stable metrics, "
            "stable fallback classification, and acceptable "
            "case-level prediction variation. After that "
            "choice is frozen, run the locked repeated "
            "confirmatory evaluation."
        ),
        "",
    ]
)

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print(
    "analysis_version:",
    result["analysis_version"],
)
print("status:", result["status"])
print(
    "configuration:",
    result["configuration"],
)
print(
    "mean_fail_closed_metrics_by_particle_count:"
)
for particle_count in particle_counts:
    print(
        particle_count,
        best_particle_summary[
            particle_count
        ],
    )

print(
    "fallback_stability:",
    {
        "status_flip_case_count": (
            result[
                "status_flip_case_count"
            ]
        ),
        "always_fallback_case_ids": (
            result[
                "always_fallback_case_ids"
            ]
        ),
    },
)

print(
    "prediction_stability:"
)
for particle_count in particle_counts:
    print(
        particle_count,
        prediction_stability[
            str(particle_count)
        ]["point_range_ml"],
    )

print(
    "benchmark_same_252_cases:",
    benchmark_metrics,
)
print("json_artifact:", OUTPUT)
print("markdown_report:", REPORT)
print(
    "json_sha256:",
    sha256_file(OUTPUT),
)
print(
    "markdown_sha256:",
    sha256_file(REPORT),
)
print(
    "LAYER7_PRELOCK_ROBUSTNESS_STATUS: PASS"
)
