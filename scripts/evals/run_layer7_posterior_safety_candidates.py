#!/usr/bin/env python3
"""Compare fail-closed and likelihood-tempered Layer 7 safety candidates."""

from __future__ import annotations

from collections import Counter
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
    canonical_act_schedule,
)
from evals.prior_stack.v1_real_data_eval import (
    load_real_cohort,
)
from experiments.twin_runtime.layer7_prior_particles import (
    build_layer7_prior_particles,
)
from experiments.twin_runtime.posterior import (
    update_volume_posterior,
)
from experiments.twin_runtime.posterior_safety import (
    fail_closed_weights,
    tempered_likelihood_weights,
)


SOURCE = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_exploratory_real_data_v0_1.json"
)
COHORT = Path(
    "data/processed/v1_prior_stack/"
    "ispy2_v1_prior_eval_cohort.jsonl"
)
OUTPUT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_posterior_safety_candidates_v0_1.json"
)
REPORT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_posterior_safety_candidates_v0_1.md"
)

EXPECTED_ESS_THRESHOLD = 0.10


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


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    probability: float,
) -> float:
    total = sum(weights)
    if total <= 0:
        raise ValueError(
            "weights must contain positive mass"
        )

    ordered = sorted(
        zip(values, weights),
        key=lambda pair: pair[0],
    )
    cumulative = 0.0

    for value, weight in ordered:
        cumulative += weight / total
        if cumulative >= probability:
            return value

    return ordered[-1][0]


def particle_prediction(
    posterior: Mapping[str, object],
    *,
    prediction_day: float,
    weights: Sequence[float],
) -> dict[str, float]:
    rows = posterior["particle_trajectories"]
    times = [
        finite(value, "trajectory time")
        for value in posterior[
            "posterior_trajectory_summary"
        ]["times"]
    ]

    try:
        index = next(
            position
            for position, day in enumerate(times)
            if math.isclose(
                day,
                prediction_day,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    except StopIteration as exc:
        raise ValueError(
            "prediction day missing from posterior"
        ) from exc

    values = [
        finite(
            row["predicted_volume_ml"][index],
            "predicted volume",
        )
        for row in rows
    ]

    return {
        "point_ml": weighted_quantile(
            values,
            weights,
            0.50,
        ),
        "lower_80_ml": weighted_quantile(
            values,
            weights,
            0.10,
        ),
        "upper_80_ml": weighted_quantile(
            values,
            weights,
            0.90,
        ),
        "lower_95_ml": weighted_quantile(
            values,
            weights,
            0.025,
        ),
        "upper_95_ml": weighted_quantile(
            values,
            weights,
            0.975,
        ),
    }


def case_qc(
    case: Mapping[str, object],
) -> tuple[str, str, str]:
    context = case.get("context", {})
    if not isinstance(context, Mapping):
        context = {}

    source = str(
        context.get(
            "source",
            context.get(
                "data_origin",
                "unknown",
            ),
        )
    )
    qc = str(
        context.get(
            "segmentation_qc",
            "unknown",
        )
    )
    confidence = str(
        context.get(
            "confidence",
            qc,
        )
    )

    return source, confidence, qc


def metrics(
    records: Sequence[Mapping[str, object]],
    candidate: str,
) -> dict[str, float | int]:
    absolute_errors = []
    squared_errors = []
    squared_log_errors = []
    percentage_errors = []
    widths80 = []
    widths95 = []
    covered80 = 0
    covered95 = 0

    for record in records:
        observed = finite(
            record["observed_final_volume_ml"],
            "observed final",
        )
        prediction = record["predictions"][
            candidate
        ]

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
            abs(error) / max(observed, 1e-8)
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

    n = len(records)

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


source = json.loads(
    SOURCE.read_text(encoding="utf-8")
)
eval_policy_path = Path(
    source["configuration"][
        "eval_policy_path"
    ]
)
particle_policy_path = Path(
    source["configuration"][
        "particle_policy_path"
    ]
)

eval_policy = json.loads(
    eval_policy_path.read_text(
        encoding="utf-8"
    )
)

threshold = finite(
    eval_policy["likelihood"][
        "ess_threshold_fraction"
    ],
    "ESS threshold",
)

if not math.isclose(
    threshold,
    EXPECTED_ESS_THRESHOLD,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise RuntimeError(
        "This comparison must use the predeclared "
        "runtime ESS threshold of 0.10"
    )

cohort = load_real_cohort(
    COHORT,
    allow_demo_data=False,
)
cases_by_id = {
    str(case["case_id"]): case
    for case in cohort
}

source_rows = source["case_results"]
candidate_rows = []
safety_status_counts = Counter()
tempered_cases = []

for row in source_rows:
    case_id = str(row["case_id"])
    prior = dict(row["prior_prediction"])
    raw = dict(row["posterior_prediction"])

    if row["early_day"] is None:
        candidate_rows.append(
            {
                "case_id": case_id,
                "observed_final_volume_ml": (
                    row[
                        "observed_final_volume_ml"
                    ]
                ),
                "update_eligible": False,
                "raw_ess_fraction": None,
                "predictions": {
                    "prior": prior,
                    "raw_posterior": raw,
                    "fail_closed": dict(prior),
                    "tempered": dict(prior),
                },
                "fail_closed_status": (
                    "not_updated_no_valid_observation"
                ),
                "tempered_status": (
                    "not_updated_no_valid_observation"
                ),
                "likelihood_power": 0.0,
            }
        )
        continue

    health = row["posterior_health"]
    raw_ess_fraction = finite(
        health["weights"][
            "effective_sample_size_fraction"
        ],
        "raw ESS fraction",
    )

    if raw_ess_fraction >= threshold:
        safety_status_counts[
            "full_update_accepted"
        ] += 1

        candidate_rows.append(
            {
                "case_id": case_id,
                "observed_final_volume_ml": (
                    row[
                        "observed_final_volume_ml"
                    ]
                ),
                "update_eligible": True,
                "raw_ess_fraction": (
                    raw_ess_fraction
                ),
                "predictions": {
                    "prior": prior,
                    "raw_posterior": raw,
                    "fail_closed": dict(raw),
                    "tempered": dict(raw),
                },
                "fail_closed_status": (
                    "full_update_accepted"
                ),
                "tempered_status": (
                    "full_update_accepted"
                ),
                "likelihood_power": 1.0,
            }
        )
        continue

    case = cases_by_id[case_id]
    particle_count = int(
        row["particle_count"]
    )
    particle_seed = int(
        row["particle_seed"]
    )

    artifact = build_layer7_prior_particles(
        case,
        n_particles=particle_count,
        seed=particle_seed,
        policy_path=particle_policy_path,
    )

    source_name, confidence, qc = (
        case_qc(case)
    )

    posterior = update_volume_posterior(
        initial_volume_ml=finite(
            row["baseline_volume_ml"],
            "baseline volume",
        ),
        treatment_schedule=row["schedule"],
        parameter_particles=artifact[
            "parameter_particles"
        ],
        observations=[
            {
                "day": finite(
                    row["early_day"],
                    "early day",
                ),
                "tumor_volume_ml": finite(
                    row["early_volume_ml"],
                    "early volume",
                ),
                "source": source_name,
                "confidence": confidence,
                "segmentation_qc": qc,
                "observation_id": (
                    f"{case_id}_early"
                ),
            }
        ],
        prediction_days=[
            finite(
                row["prediction_day"],
                "prediction day",
            )
        ],
        dt_days=finite(
            eval_policy["simulation"][
                "dt_days"
            ],
            "dt_days",
        ),
        likelihood_noise_fraction=(
            eval_policy["likelihood"].get(
                "override_noise_fraction"
            )
        ),
        ess_threshold_fraction=threshold,
        include_failed_qc_observations=bool(
            eval_policy["likelihood"][
                "include_failed_qc_observations"
            ]
        ),
        allow_beyond_schedule=bool(
            eval_policy["simulation"][
                "allow_beyond_schedule"
            ]
        ),
    )

    particle_rows = posterior[
        "particle_trajectories"
    ]
    log_prior_weights = [
        finite(
            particle[
                "log_prior_weight"
            ],
            "log prior weight",
        )
        for particle in particle_rows
    ]
    log_likelihoods = [
        finite(
            particle["log_likelihood"],
            "log likelihood",
        )
        for particle in particle_rows
    ]

    fail_closed = fail_closed_weights(
        log_prior_weights=(
            log_prior_weights
        ),
        log_likelihoods=log_likelihoods,
        ess_threshold_fraction=threshold,
    )
    tempered = tempered_likelihood_weights(
        log_prior_weights=(
            log_prior_weights
        ),
        log_likelihoods=log_likelihoods,
        ess_target_fraction=threshold,
    )

    prediction_day = finite(
        row["prediction_day"],
        "prediction day",
    )

    fail_closed_prediction = (
        particle_prediction(
            posterior,
            prediction_day=prediction_day,
            weights=fail_closed["weights"],
        )
    )
    tempered_prediction = (
        particle_prediction(
            posterior,
            prediction_day=prediction_day,
            weights=tempered["weights"],
        )
    )

    if fail_closed_prediction != prior:
        raise RuntimeError(
            f"fail-closed prediction did not exactly "
            f"match stored prior for {case_id}"
        )

    safety_status_counts[
        str(tempered["status"])
    ] += 1

    tempered_case = {
        "case_id": case_id,
        "raw_ess_fraction": (
            raw_ess_fraction
        ),
        "tempered_ess_fraction": (
            tempered[
                "selected_ess_fraction"
            ]
        ),
        "likelihood_power": (
            tempered["likelihood_power"]
        ),
        "maximum_raw_weight": (
            health["weights"][
                "maximum_weight"
            ]
        ),
        "prior_point_ml": (
            prior["point_ml"]
        ),
        "raw_posterior_point_ml": (
            raw["point_ml"]
        ),
        "tempered_point_ml": (
            tempered_prediction[
                "point_ml"
            ]
        ),
        "observed_final_volume_ml": (
            row[
                "observed_final_volume_ml"
            ]
        ),
        "tempered_status": (
            tempered["status"]
        ),
    }
    tempered_cases.append(
        tempered_case
    )

    candidate_rows.append(
        {
            "case_id": case_id,
            "observed_final_volume_ml": (
                row[
                    "observed_final_volume_ml"
                ]
            ),
            "update_eligible": True,
            "raw_ess_fraction": (
                raw_ess_fraction
            ),
            "predictions": {
                "prior": prior,
                "raw_posterior": raw,
                "fail_closed": (
                    fail_closed_prediction
                ),
                "tempered": (
                    tempered_prediction
                ),
            },
            "fail_closed_status": (
                fail_closed["status"]
            ),
            "tempered_status": (
                tempered["status"]
            ),
            "likelihood_power": (
                tempered[
                    "likelihood_power"
                ]
            ),
        }
    )

updated_rows = [
    row
    for row in candidate_rows
    if row["update_eligible"]
]

candidates = (
    "prior",
    "raw_posterior",
    "fail_closed",
    "tempered",
)

updated_metrics = {
    candidate: metrics(
        updated_rows,
        candidate,
    )
    for candidate in candidates
}
all_case_metrics = {
    candidate: metrics(
        candidate_rows,
        candidate,
    )
    for candidate in candidates
}

result = {
    "analysis_version": (
        "oncotwin_layer7_posterior_safety_candidates_v0_1"
    ),
    "status": (
        "layer7_posterior_safety_candidates_complete"
    ),
    "interpretation": (
        "Exploratory candidate comparison. The ESS threshold "
        "of 0.10 was inherited from the pre-existing runtime "
        "policy and was not selected from outcome performance."
    ),
    "selection_performed": False,
    "source_artifact": {
        "path": str(SOURCE),
        "sha256": sha256_file(SOURCE),
    },
    "configuration": {
        "ess_threshold_fraction": threshold,
        "threshold_source": (
            "configs/prior/"
            "layer7_exploratory_eval_policy_v0_1.json"
        ),
        "tempering_rule": (
            "largest likelihood power with ESS fraction "
            "at least 0.10"
        ),
        "fail_closed_rule": (
            "return exact mechanistic prior prediction "
            "when raw ESS fraction is below 0.10"
        ),
    },
    "dataset": {
        "case_count": len(
            candidate_rows
        ),
        "updated_case_count": len(
            updated_rows
        ),
        "early_missing_count": (
            len(candidate_rows)
            - len(updated_rows)
        ),
        "low_ess_case_count": len(
            tempered_cases
        ),
    },
    "updated_case_metrics": (
        updated_metrics
    ),
    "all_case_metrics": (
        all_case_metrics
    ),
    "safety_status_counts": dict(
        sorted(
            safety_status_counts.items()
        )
    ),
    "tempered_cases": tempered_cases,
    "case_results": candidate_rows,
    "requirements": {
        "predeclared_threshold_used": True,
        "outcome_based_threshold_selection": False,
        "fail_closed_exact_prior_match": True,
        "healthy_updates_unchanged": True,
        "early_missing_exact_no_update": True,
        "tempering_uses_existing_particles": True,
        "resampling_performed": False,
    },
    "limitations": [
        (
            "This remains exploratory and cannot serve as "
            "confirmatory evidence."
        ),
        (
            "The canonical cohort-level A/C-T schedule proxy "
            "is still used."
        ),
        (
            "Only eight low-ESS cases require tempering, so "
            "comparisons between fail-closed and tempering "
            "have limited sample size."
        ),
    ],
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
    "# Layer 7 posterior safety candidates",
    "",
    (
        "**Exploratory comparison using the predeclared "
        "ESS threshold of 0.10.**"
    ),
    "",
    "## Updated-case metrics",
    "",
    "| Candidate | MAE | RMSE | Log RMSE | MAPE | 80% coverage | 95% coverage | Maximum error |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for candidate in candidates:
    values = updated_metrics[candidate]
    lines.append(
        f"| {candidate} | "
        f"{values['mae_ml']} | "
        f"{values['rmse_ml']} | "
        f"{values['log_volume_rmse']} | "
        f"{values['mape']} | "
        f"{values['coverage_80']} | "
        f"{values['coverage_95']} | "
        f"{values['maximum_absolute_error_ml']} |"
    )

lines.extend(
    [
        "",
        "## Tempered low-ESS cases",
        "",
        "| Case | Raw ESS fraction | Tempered ESS fraction | Likelihood power | Prior point | Raw point | Tempered point | Observed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
)

for row in tempered_cases:
    lines.append(
        f"| {row['case_id']} | "
        f"{row['raw_ess_fraction']} | "
        f"{row['tempered_ess_fraction']} | "
        f"{row['likelihood_power']} | "
        f"{row['prior_point_ml']} | "
        f"{row['raw_posterior_point_ml']} | "
        f"{row['tempered_point_ml']} | "
        f"{row['observed_final_volume_ml']} |"
    )

lines.extend(
    [
        "",
        "## Decision boundary",
        "",
        (
            "The next step should select the safety mechanism "
            "for the locked Layer 7 specification. Selection "
            "must consider predictive performance, continuity "
            "of updating, interpretability, and whether the "
            "tempered likelihood powers indicate severe "
            "prior-observation conflict."
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
print("dataset:", result["dataset"])
print(
    "updated_case_metrics:"
)
for candidate in candidates:
    print(
        candidate,
        updated_metrics[candidate],
    )
print(
    "safety_status_counts:",
    result["safety_status_counts"],
)
print("tempered_cases:")
for row in tempered_cases:
    print(row)
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
    "LAYER7_POSTERIOR_SAFETY_STATUS: PASS"
)
