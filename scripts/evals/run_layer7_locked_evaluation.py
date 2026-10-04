#!/usr/bin/env python3
"""Run and package the locked internal Layer 7 evaluation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
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
    LAYER7_FAIL_CLOSED_CANDIDATE_VERSION,
    apply_layer7_fail_closed_candidate,
)


LOCKED_SPEC = Path(
    "configs/prior/"
    "layer7_locked_specification_v1.json"
)
PARTICLE_POLICY = Path(
    "configs/prior/"
    "layer7_baseline_particle_policy_v0_1.json"
)
EVAL_POLICY = Path(
    "configs/prior/"
    "layer7_exploratory_eval_policy_v0_1.json"
)
CANDIDATE_POLICY = Path(
    "configs/prior/"
    "layer7_fail_closed_candidate_v0_2.json"
)
PRELOCK = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_prelock_robustness_v0_2.json"
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
    "locked_internal_evaluation_v1.json"
)
REPORT = Path(
    "artifacts/prior_builder/layer7/"
    "locked_internal_evaluation_v1.md"
)
RUNTIME_DIR = Path(
    "artifacts/prior_builder/layer7/"
    "posterior_runtime_v1"
)

PREDICTION_KEYS = (
    "point_ml",
    "lower_80_ml",
    "upper_80_ml",
    "lower_95_ml",
    "upper_95_ml",
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


def write_json(
    path: Path,
    value: Mapping[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


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


def validate_prediction(
    prediction: Mapping[str, object],
    name: str,
) -> None:
    values = [
        finite(
            prediction[key],
            f"{name}.{key}",
        )
        for key in (
            "lower_95_ml",
            "lower_80_ml",
            "point_ml",
            "upper_80_ml",
            "upper_95_ml",
        )
    ]

    if any(value <= 0 for value in values):
        raise ValueError(
            f"{name} must contain positive volumes"
        )

    if values != sorted(values):
        raise ValueError(
            f"{name} intervals are not nested"
        )


def metrics(
    rows: Sequence[Mapping[str, object]],
    prediction_key: str,
) -> dict[str, float | int]:
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
            "observed final volume",
        )
        prediction = row[prediction_key]
        validate_prediction(
            prediction,
            prediction_key,
        )

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
                - math.log(max(observed, 1e-8))
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
    if n == 0:
        raise ValueError(
            "metrics requires nonempty rows"
        )

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


def metric_value(
    rows: Sequence[Mapping[str, object]],
    prediction_key: str,
    metric: str,
) -> float:
    observed = [
        finite(
            row["observed_final_volume_ml"],
            "observed final",
        )
        for row in rows
    ]
    predicted = [
        finite(
            row[prediction_key][
                "point_ml"
            ],
            "predicted point",
        )
        for row in rows
    ]

    if metric == "mae_ml":
        return statistics.fmean(
            abs(prediction - truth)
            for prediction, truth
            in zip(predicted, observed)
        )

    if metric == "rmse_ml":
        return math.sqrt(
            statistics.fmean(
                (prediction - truth) ** 2
                for prediction, truth
                in zip(predicted, observed)
            )
        )

    if metric == "log_volume_rmse":
        return math.sqrt(
            statistics.fmean(
                (
                    math.log(
                        max(prediction, 1e-8)
                    )
                    - math.log(
                        max(truth, 1e-8)
                    )
                )
                ** 2
                for prediction, truth
                in zip(predicted, observed)
            )
        )

    raise KeyError(metric)


def percentile(
    values: Sequence[float],
    probability: float,
) -> float:
    ordered = sorted(values)
    position = probability * (
        len(ordered) - 1
    )
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def paired_bootstrap_delta(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    comparator: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    n = len(rows)

    point_delta = (
        metric_value(
            rows,
            candidate,
            metric,
        )
        - metric_value(
            rows,
            comparator,
            metric,
        )
    )

    deltas = []

    for _ in range(replicates):
        sample = [
            rows[rng.randrange(n)]
            for _ in range(n)
        ]

        delta = (
            metric_value(
                sample,
                candidate,
                metric,
            )
            - metric_value(
                sample,
                comparator,
                metric,
            )
        )
        deltas.append(delta)

    return {
        "metric": metric,
        "candidate": candidate,
        "comparator": comparator,
        "point_delta": point_delta,
        "ci95": [
            percentile(deltas, 0.025),
            percentile(deltas, 0.975),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "negative_favors_candidate": True,
    }


def win_loss_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    comparator: str,
) -> dict[str, int]:
    helped = harmed = tied = 0

    for row in rows:
        observed = finite(
            row["observed_final_volume_ml"],
            "observed final",
        )
        candidate_error = abs(
            finite(
                row[candidate]["point_ml"],
                "candidate point",
            )
            - observed
        )
        comparator_error = abs(
            finite(
                row[comparator]["point_ml"],
                "comparator point",
            )
            - observed
        )

        difference = (
            candidate_error
            - comparator_error
        )

        if difference < -1e-12:
            helped += 1
        elif difference > 1e-12:
            harmed += 1
        else:
            tied += 1

    return {
        "candidate_better": helped,
        "candidate_worse": harmed,
        "tied": tied,
    }


spec = json.loads(
    LOCKED_SPEC.read_text(
        encoding="utf-8"
    )
)

expected_hashes = spec[
    "expected_hashes"
]
hash_targets = {
    "cohort": COHORT,
    "particle_policy": PARTICLE_POLICY,
    "evaluation_policy": EVAL_POLICY,
    "candidate_policy": CANDIDATE_POLICY,
    "prelock_robustness": PRELOCK,
    "layer6_closure": LAYER6_CLOSURE,
}

verified_hashes = {}

for name, path in hash_targets.items():
    actual = sha256_file(path)
    expected = str(
        expected_hashes[name]
    )

    if actual != expected:
        raise RuntimeError(
            f"{name} hash mismatch: "
            f"expected {expected}, got {actual}"
        )

    verified_hashes[name] = actual

particle_count = int(
    spec["particle_runtime"][
        "particle_count"
    ]
)
particle_seed = int(
    spec["particle_runtime"]["seed"]
)
ess_threshold = finite(
    spec["posterior_update"][
        "ess_threshold_fraction"
    ],
    "ess threshold",
)
bootstrap_replicates = int(
    spec["locked_evaluation"][
        "bootstrap_replicates"
    ]
)
bootstrap_seed = int(
    spec["locked_evaluation"][
        "bootstrap_seed"
    ]
)

raw_layer7 = run_layer7_real_data_eval(
    COHORT,
    particle_count=particle_count,
    seed=particle_seed,
    particle_policy_path=(
        PARTICLE_POLICY
    ),
    eval_policy_path=EVAL_POLICY,
)

layer7_rows_by_id = {}
all_layer7_rows = []
layer7_status_counts = Counter()
updated_fallback_ids = []
exact_missing_count = 0

for raw_row in raw_layer7[
    "case_results"
]:
    candidate = (
        apply_layer7_fail_closed_candidate(
            raw_row,
            ess_threshold_fraction=(
                ess_threshold
            ),
            accepted_health_statuses=(
                "updated_healthy",
            ),
        )
    )

    status = str(
        candidate["layer7"]["status"]
    )
    layer7_status_counts[status] += 1

    if raw_row["early_day"] is None:
        exact_prior = {
            key: candidate[key]
            for key in PREDICTION_KEYS
        } == raw_row["prior_prediction"]

        if not exact_prior:
            raise RuntimeError(
                "Early-missing case changed: "
                f"{raw_row['case_id']}"
            )

        exact_missing_count += 1

    elif candidate["layer7"][
        "fallback_used"
    ]:
        updated_fallback_ids.append(
            str(raw_row["case_id"])
        )

    if candidate["layer7"][
        "final_outcome_used_for_selection"
    ]:
        raise RuntimeError(
            "Final outcome entered candidate selection"
        )

    row = {
        "case_id": str(
            raw_row["case_id"]
        ),
        "observed_final_volume_ml": (
            raw_row[
                "observed_final_volume_ml"
            ]
        ),
        "early_available": (
            raw_row["early_day"] is not None
        ),
        "mechanistic_prior": (
            raw_row["prior_prediction"]
        ),
        "raw_mechanistic_posterior": (
            raw_row[
                "posterior_prediction"
            ]
        ),
        "layer7_fail_closed": candidate,
        "raw_posterior_health": (
            raw_row["posterior_health"]
        ),
    }

    all_layer7_rows.append(row)
    layer7_rows_by_id[
        row["case_id"]
    ] = row

if len(all_layer7_rows) != 270:
    raise RuntimeError(
        "Expected 270 Layer 7 rows"
    )
if exact_missing_count != 18:
    raise RuntimeError(
        "Expected 18 exact no-update cases"
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

comparison_rows = []

for row in layer6_eval[
    "case_predictions"
]:
    case_id = str(row["case_id"])
    predictions = row["predictions"]
    layer6 = predictions[
        LAYER6_CANDIDATE
    ]

    if (
        layer6["layer6"]["status"]
        != "active"
    ):
        continue

    layer7_row = layer7_rows_by_id[
        case_id
    ]

    comparison_rows.append(
        {
            "case_id": case_id,
            "observed_final_volume_ml": (
                row[
                    "observed_final_volume_ml"
                ]
            ),
            "mechanistic_prior": (
                layer7_row[
                    "mechanistic_prior"
                ]
            ),
            "layer7_fail_closed": (
                layer7_row[
                    "layer7_fail_closed"
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

if len(comparison_rows) != 252:
    raise RuntimeError(
        "Expected 252 aligned comparison rows"
    )

candidate_names = (
    "mechanistic_prior",
    "layer7_fail_closed",
    "static_layer4",
    "d1_early_response",
    "layer6",
)

comparison_metrics = {
    name: metrics(
        comparison_rows,
        name,
    )
    for name in candidate_names
}

all_case_metrics = {
    "mechanistic_prior": metrics(
        all_layer7_rows,
        "mechanistic_prior",
    ),
    "layer7_fail_closed": metrics(
        all_layer7_rows,
        "layer7_fail_closed",
    ),
}

comparisons = {}

for comparison_index, comparator in enumerate(
    (
        "mechanistic_prior",
        "d1_early_response",
        "layer6",
    )
):
    metric_results = {}

    for metric_index, metric in enumerate(
        (
            "mae_ml",
            "rmse_ml",
            "log_volume_rmse",
        )
    ):
        metric_results[metric] = (
            paired_bootstrap_delta(
                comparison_rows,
                candidate=(
                    "layer7_fail_closed"
                ),
                comparator=comparator,
                metric=metric,
                replicates=(
                    bootstrap_replicates
                ),
                seed=(
                    bootstrap_seed
                    + comparison_index * 100
                    + metric_index
                ),
            )
        )

    comparisons[comparator] = {
        "paired_bootstrap": (
            metric_results
        ),
        "absolute_error_win_loss": (
            win_loss_summary(
                comparison_rows,
                candidate=(
                    "layer7_fail_closed"
                ),
                comparator=comparator,
            )
        ),
    }

mae_update_ci = comparisons[
    "mechanistic_prior"
]["paired_bootstrap"]["mae_ml"]["ci95"]

update_value_demonstrated = (
    mae_update_ci[1] < 0
)

layer6_remains_primary = (
    comparison_metrics["layer6"][
        "mae_ml"
    ]
    < comparison_metrics[
        "layer7_fail_closed"
    ]["mae_ml"]
    and comparison_metrics["layer6"][
        "rmse_ml"
    ]
    < comparison_metrics[
        "layer7_fail_closed"
    ]["rmse_ml"]
)

health_status_counts = Counter(
    str(
        row[
            "raw_posterior_health"
        ]["status"]
    )
    for row in all_layer7_rows
    if row["early_available"]
)

evaluation = {
    "analysis_version": (
        "oncotwin_layer7_locked_internal_evaluation_v1"
    ),
    "status": (
        "locked_internal_evaluation_complete"
    ),
    "closure_status": (
        "layer7_complete_ready_for_layer8"
    ),
    "evidence_class": (
        "internal_locked_evaluation_after_exploratory_selection"
    ),
    "independent_confirmatory_evidence": (
        False
    ),
    "selection_performed_in_locked_run": (
        False
    ),
    "development_selection_preceded_lock": (
        True
    ),
    "locked_specification": {
        "path": str(LOCKED_SPEC),
        "sha256": sha256_file(
            LOCKED_SPEC
        ),
    },
    "verified_source_hashes": (
        verified_hashes
    ),
    "design": {
        "particle_count": particle_count,
        "particle_seed": particle_seed,
        "ess_threshold_fraction": (
            ess_threshold
        ),
        "bootstrap_replicates": (
            bootstrap_replicates
        ),
        "bootstrap_seed": (
            bootstrap_seed
        ),
        "candidate_name": (
            LAYER7_FAIL_CLOSED_CANDIDATE_NAME
        ),
        "candidate_version": (
            LAYER7_FAIL_CLOSED_CANDIDATE_VERSION
        ),
        "patient_specific_schedule": (
            False
        ),
        "schedule_proxy": (
            spec["schedule"]["version"]
        ),
    },
    "dataset": {
        "case_count": 270,
        "update_eligible_count": 252,
        "early_missing_count": 18,
        "exact_no_update_count": (
            exact_missing_count
        ),
        "updated_fallback_count": len(
            updated_fallback_ids
        ),
        "updated_fallback_case_ids": (
            sorted(
                updated_fallback_ids
            )
        ),
        "candidate_status_counts": dict(
            sorted(
                layer7_status_counts.items()
            )
        ),
        "raw_health_status_counts": dict(
            sorted(
                health_status_counts.items()
            )
        ),
    },
    "metrics_same_252_cases": (
        comparison_metrics
    ),
    "metrics_all_270_cases": (
        all_case_metrics
    ),
    "paired_comparisons": comparisons,
    "scientific_conclusions": {
        "mechanistic_update_value_demonstrated": (
            update_value_demonstrated
        ),
        "mechanistic_update_value_definition": (
            "Layer 7 fail-closed MAE is lower than "
            "the pre-update mechanistic prior, with "
            "the paired bootstrap 95 percent interval "
            "for the MAE difference below zero."
        ),
        "layer6_remains_primary_predictive_candidate": (
            layer6_remains_primary
        ),
        "layer7_role": (
            "Mechanistic posterior updating, parameter "
            "uncertainty, sequential evidence assimilation, "
            "and posterior-health diagnostics."
        ),
        "layer7_does_not_replace_layer6": (
            True
        ),
    },
    "requirements": {
        "all_270_cases_evaluated": True,
        "all_252_update_eligible_cases_evaluated": (
            True
        ),
        "all_18_missing_cases_exact_no_update": (
            exact_missing_count == 18
        ),
        "fail_closed_policy_applied": True,
        "final_outcome_used_for_inference": (
            False
        ),
        "silent_resampling_performed": (
            False
        ),
        "tempering_used": False,
        "source_hashes_verified": True,
        "external_validation_complete": (
            False
        ),
        "prospective_validation_complete": (
            False
        ),
        "treatment_recommendation_supported": (
            False
        ),
    },
    "case_results": (
        all_layer7_rows
    ),
    "limitations": list(
        spec["known_limitations"]
    ),
}

write_json(
    OUTPUT,
    evaluation,
)

RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

runtime_files = {
    "locked_specification.json": (
        LOCKED_SPEC
    ),
    "particle_policy.json": (
        PARTICLE_POLICY
    ),
    "evaluation_policy.json": (
        EVAL_POLICY
    ),
    "candidate_policy.json": (
        CANDIDATE_POLICY
    ),
}

for destination_name, source_path in (
    runtime_files.items()
):
    shutil.copyfile(
        source_path,
        RUNTIME_DIR / destination_name,
    )

source_files = {
    "layer7_prior_particles.py": Path(
        "experiments/twin_runtime/"
        "layer7_prior_particles.py"
    ),
    "posterior.py": Path(
        "experiments/twin_runtime/"
        "posterior.py"
    ),
    "posterior_diagnostics.py": Path(
        "experiments/twin_runtime/"
        "posterior_diagnostics.py"
    ),
    "layer7_candidate.py": Path(
        "experiments/twin_runtime/"
        "layer7_candidate.py"
    ),
    "v1_layer7_real_data_eval.py": Path(
        "evals/prior_stack/"
        "v1_layer7_real_data_eval.py"
    ),
}

manifest = {
    "artifact_version": (
        "oncotwin_layer7_posterior_runtime_v1"
    ),
    "status": (
        "locked_internal_research_only"
    ),
    "candidate_name": (
        LAYER7_FAIL_CLOSED_CANDIDATE_NAME
    ),
    "candidate_version": (
        LAYER7_FAIL_CLOSED_CANDIDATE_VERSION
    ),
    "particle_count": particle_count,
    "particle_seed": particle_seed,
    "ess_threshold_fraction": (
        ess_threshold
    ),
    "accepted_health_statuses": [
        "updated_healthy"
    ],
    "fallback_policy": (
        "exact_mechanistic_prior_prediction"
    ),
    "schedule_proxy": (
        spec["schedule"]
    ),
    "locked_evaluation": {
        "path": str(OUTPUT),
        "sha256": sha256_file(OUTPUT),
        "evidence_class": evaluation[
            "evidence_class"
        ],
        "independent_confirmatory_evidence": (
            False
        ),
    },
    "source_artifacts": {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in {
            "cohort": COHORT,
            "prelock_robustness": PRELOCK,
            "layer6_closure": LAYER6_CLOSURE,
        }.items()
    },
    "source_code": {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in (
            source_files.items()
        )
    },
    "limitations": list(
        spec["known_limitations"]
    ),
}

manifest_path = (
    RUNTIME_DIR / "manifest.json"
)
write_json(
    manifest_path,
    manifest,
)

checksummed_names = [
    "manifest.json",
    *runtime_files.keys(),
]

checksums = {
    "artifact_version": (
        "oncotwin_layer7_posterior_runtime_checksums_v1"
    ),
    "files": {
        name: sha256_file(
            RUNTIME_DIR / name
        )
        for name in sorted(
            checksummed_names
        )
    },
}

checksums_path = (
    RUNTIME_DIR / "checksums.json"
)
write_json(
    checksums_path,
    checksums,
)

prior_metrics = comparison_metrics[
    "mechanistic_prior"
]
safe_metrics = comparison_metrics[
    "layer7_fail_closed"
]
d1_metrics = comparison_metrics[
    "d1_early_response"
]
layer6_metrics = comparison_metrics[
    "layer6"
]

lines = [
    "# Layer 7 locked internal evaluation",
    "",
    (
        "**Status: Layer 7 complete and ready for "
        "Layer 8.**"
    ),
    "",
    (
        "This is a locked internal evaluation after "
        "exploratory design selection on the same cohort. "
        "It is not independent external or prospective "
        "confirmation."
    ),
    "",
    "## Locked design",
    "",
    f"- Particles: {particle_count}",
    f"- Particle seed: {particle_seed}",
    (
        "- ESS fail-closed threshold: "
        f"{ess_threshold}"
    ),
    (
        "- Updated cases using prior fallback: "
        f"{len(updated_fallback_ids)}"
    ),
    (
        "- Early-missing exact no-update cases: "
        f"{exact_missing_count}"
    ),
    "",
    "## Metrics on the same 252 update-eligible cases",
    "",
    "| Candidate | MAE | RMSE | Log RMSE | 80% coverage | 95% coverage |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    (
        "| Mechanistic prior | "
        f"{prior_metrics['mae_ml']} | "
        f"{prior_metrics['rmse_ml']} | "
        f"{prior_metrics['log_volume_rmse']} | "
        f"{prior_metrics['coverage_80']} | "
        f"{prior_metrics['coverage_95']} |"
    ),
    (
        "| Layer 7 fail-closed posterior | "
        f"{safe_metrics['mae_ml']} | "
        f"{safe_metrics['rmse_ml']} | "
        f"{safe_metrics['log_volume_rmse']} | "
        f"{safe_metrics['coverage_80']} | "
        f"{safe_metrics['coverage_95']} |"
    ),
    (
        "| D1 early-response candidate | "
        f"{d1_metrics['mae_ml']} | "
        f"{d1_metrics['rmse_ml']} | "
        f"{d1_metrics['log_volume_rmse']} | "
        f"{d1_metrics['coverage_80']} | "
        f"{d1_metrics['coverage_95']} |"
    ),
    (
        "| Layer 6 candidate | "
        f"{layer6_metrics['mae_ml']} | "
        f"{layer6_metrics['rmse_ml']} | "
        f"{layer6_metrics['log_volume_rmse']} | "
        f"{layer6_metrics['coverage_80']} | "
        f"{layer6_metrics['coverage_95']} |"
    ),
    "",
    "## Scientific conclusion",
    "",
    (
        "- Mechanistic update value demonstrated: "
        f"{update_value_demonstrated}"
    ),
    (
        "- Layer 6 remains the primary predictive "
        f"candidate: {layer6_remains_primary}"
    ),
    (
        "- Layer 7 remains the mechanistic posterior "
        "and uncertainty layer."
    ),
    "",
    "## Scope",
    "",
    (
        "The runtime is for internal research only. "
        "Patient-specific treatment administration data, "
        "external validation, prospective validation, and "
        "treatment-recommendation support remain outside "
        "the completed V1 Layer 7 scope."
    ),
    "",
]

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print(
    "analysis_version:",
    evaluation["analysis_version"],
)
print(
    "status:",
    evaluation["status"],
)
print(
    "closure_status:",
    evaluation["closure_status"],
)
print(
    "design:",
    evaluation["design"],
)
print(
    "dataset:",
    evaluation["dataset"],
)
print(
    "metrics_same_252_cases:"
)
for name in candidate_names:
    print(
        name,
        comparison_metrics[name],
    )

print(
    "layer7_vs_mechanistic_prior:",
    comparisons[
        "mechanistic_prior"
    ],
)
print(
    "layer7_vs_d1:",
    comparisons[
        "d1_early_response"
    ],
)
print(
    "layer7_vs_layer6:",
    comparisons["layer6"],
)
print(
    "scientific_conclusions:",
    evaluation[
        "scientific_conclusions"
    ],
)
print("evaluation_json:", OUTPUT)
print("evaluation_markdown:", REPORT)
print(
    "evaluation_json_sha256:",
    sha256_file(OUTPUT),
)
print(
    "evaluation_markdown_sha256:",
    sha256_file(REPORT),
)
print(
    "runtime_manifest:",
    manifest_path,
)
print(
    "runtime_manifest_sha256:",
    sha256_file(manifest_path),
)
print(
    "runtime_checksums:",
    checksums_path,
)
print(
    "LAYER7_LOCKED_EVALUATION_STATUS: PASS"
)
