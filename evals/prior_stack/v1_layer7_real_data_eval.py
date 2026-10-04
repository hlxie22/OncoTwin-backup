"""Exploratory real-cohort evaluation of Layer 7 Bayesian updating."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from evals.prior_stack.v1_real_data_eval import (
    load_real_cohort,
)
from experiments.twin_runtime.layer7_prior_particles import (
    DEFAULT_POLICY_PATH,
    build_layer7_prior_particles,
    sha256_file,
)
from experiments.twin_runtime.posterior import (
    normalize_log_weights,
    update_volume_posterior,
)
from experiments.twin_runtime.posterior_diagnostics import (
    summarize_posterior_health,
)
from experiments.v0.mechanistic_simulator.volume_ode import (
    simulate_volume_trajectory,
)


LAYER7_REAL_DATA_EVAL_VERSION = (
    "oncotwin_layer7_real_data_eval_v0_1"
)
DEFAULT_EVAL_POLICY = Path(
    "configs/prior/"
    "layer7_exploratory_eval_policy_v0_1.json"
)
DEFAULT_COHORT = Path(
    "data/processed/v1_prior_stack/"
    "ispy2_v1_prior_eval_cohort.jsonl"
)
DEFAULT_OUTPUT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_exploratory_real_data_v0_1.json"
)
DEFAULT_REPORT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_exploratory_real_data_v0_1.md"
)


def _finite(
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


def _load_policy(
    path: Path,
) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Layer 7 eval policy must be a JSON object"
        )
    return dict(payload)


def canonical_act_schedule(
    prediction_horizon_days: float,
    *,
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Construct the explicitly documented exploratory A/C-T proxy."""

    horizon = _finite(
        prediction_horizon_days,
        "prediction_horizon_days",
    )
    if horizon <= 0:
        raise ValueError(
            "prediction_horizon_days must be positive"
        )

    schedule_policy = policy.get(
        "schedule_proxy"
    )
    if not isinstance(
        schedule_policy,
        Mapping,
    ):
        raise ValueError(
            "policy.schedule_proxy must be a mapping"
        )

    anthracycline_days = [
        _finite(value, "anthracycline day")
        for value in schedule_policy[
            "anthracycline_days"
        ]
    ]
    taxane_start = _finite(
        schedule_policy["taxane_start_day"],
        "taxane_start_day",
    )
    taxane_interval = _finite(
        schedule_policy[
            "taxane_interval_days"
        ],
        "taxane_interval_days",
    )
    relative_dose = _finite(
        schedule_policy["relative_dose"],
        "relative_dose",
    )

    if taxane_interval <= 0:
        raise ValueError(
            "taxane_interval_days must be positive"
        )

    events = [
        {
            "drug": "anthracycline",
            "day": day,
            "relative_dose": relative_dose,
        }
        for day in anthracycline_days
        if day <= horizon
    ]

    day = taxane_start
    while day <= horizon + 1e-9:
        events.append(
            {
                "drug": "taxane",
                "day": day,
                "relative_dose": relative_dose,
            }
        )
        day += taxane_interval

    return {
        "schedule_id": (
            "layer7_canonical_act_proxy"
        ),
        "regimen_name": (
            "Exploratory canonical A/C-T proxy"
        ),
        "total_duration_days": horizon,
        "events": events,
    }


def _weighted_quantile(
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


def _prediction(
    *,
    point: float,
    lower80: float,
    upper80: float,
    lower95: float,
    upper95: float,
) -> dict[str, float]:
    values = {
        "point_ml": _finite(
            point,
            "point_ml",
        ),
        "lower_80_ml": _finite(
            lower80,
            "lower_80_ml",
        ),
        "upper_80_ml": _finite(
            upper80,
            "upper_80_ml",
        ),
        "lower_95_ml": _finite(
            lower95,
            "lower_95_ml",
        ),
        "upper_95_ml": _finite(
            upper95,
            "upper_95_ml",
        ),
    }

    ordered = [
        values["lower_95_ml"],
        values["lower_80_ml"],
        values["point_ml"],
        values["upper_80_ml"],
        values["upper_95_ml"],
    ]

    if ordered != sorted(ordered):
        raise ValueError(
            "prediction intervals must be nested"
        )

    return values


def _prediction_from_summary(
    summary: Mapping[str, object],
    prediction_day: float,
    *,
    particle_rows: Sequence[
        Mapping[str, object]
    ],
    weights: Sequence[float],
) -> dict[str, float]:
    """Summarize one prediction day directly from weighted particles.

    The current posterior runtime's compact trajectory summary contains
    median and 80% intervals only. Layer 7 evaluation also requires 95%
    intervals, so all quantiles are calculated from the same stored particle
    trajectories and weights rather than inferred from the 80% interval.
    """

    times = [
        _finite(value, "summary time")
        for value in summary["times"]
    ]

    try:
        index = next(
            position
            for position, value
            in enumerate(times)
            if math.isclose(
                value,
                prediction_day,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    except StopIteration as exc:
        raise ValueError(
            "prediction day missing from trajectory summary"
        ) from exc

    if len(particle_rows) != len(weights):
        raise ValueError(
            "particle_rows and weights must have equal length"
        )
    if not particle_rows:
        raise ValueError(
            "particle_rows must not be empty"
        )

    values = []
    for row_index, row in enumerate(
        particle_rows
    ):
        trajectories = row.get(
            "predicted_volume_ml"
        )
        if (
            not isinstance(
                trajectories,
                Sequence,
            )
            or isinstance(
                trajectories,
                (str, bytes),
            )
        ):
            raise ValueError(
                "particle trajectory "
                f"{row_index} is missing predicted volumes"
            )

        if index >= len(trajectories):
            raise ValueError(
                "particle trajectory "
                f"{row_index} is missing prediction day"
            )

        values.append(
            _finite(
                trajectories[index],
                (
                    "particle predicted volume "
                    f"{row_index}"
                ),
            )
        )

    numeric_weights = [
        _finite(
            weight,
            "particle weight",
        )
        for weight in weights
    ]

    if any(
        weight < 0
        for weight in numeric_weights
    ):
        raise ValueError(
            "particle weights must be nonnegative"
        )
    if sum(numeric_weights) <= 0:
        raise ValueError(
            "particle weights must contain positive mass"
        )

    return _prediction(
        point=_weighted_quantile(
            values,
            numeric_weights,
            0.50,
        ),
        lower80=_weighted_quantile(
            values,
            numeric_weights,
            0.10,
        ),
        upper80=_weighted_quantile(
            values,
            numeric_weights,
            0.90,
        ),
        lower95=_weighted_quantile(
            values,
            numeric_weights,
            0.025,
        ),
        upper95=_weighted_quantile(
            values,
            numeric_weights,
            0.975,
        ),
    )
def _prior_prediction_without_update(
    *,
    initial_volume_ml: float,
    schedule: Mapping[str, object],
    particles: Sequence[
        Mapping[str, object]
    ],
    prediction_day: float,
    dt_days: float,
) -> dict[str, float]:
    predictions = []

    for particle in particles:
        simulation = simulate_volume_trajectory(
            initial_volume_ml=(
                initial_volume_ml
            ),
            treatment_schedule=schedule,
            params=particle,
            output_days=[prediction_day],
            dt_days=dt_days,
            allow_beyond_schedule=False,
        )
        predictions.append(
            _finite(
                simulation["trajectory"][0][
                    "tumor_volume_ml"
                ],
                "prior predicted volume",
            )
        )

    weights = [
        1.0 / len(predictions)
        for _ in predictions
    ]

    return _prediction(
        point=_weighted_quantile(
            predictions,
            weights,
            0.50,
        ),
        lower80=_weighted_quantile(
            predictions,
            weights,
            0.10,
        ),
        upper80=_weighted_quantile(
            predictions,
            weights,
            0.90,
        ),
        lower95=_weighted_quantile(
            predictions,
            weights,
            0.025,
        ),
        upper95=_weighted_quantile(
            predictions,
            weights,
            0.975,
        ),
    )


def _case_qc(
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


def evaluate_layer7_case(
    case: Mapping[str, object],
    *,
    case_index: int,
    particle_count: int,
    seed: int,
    particle_policy_path: Path,
    eval_policy: Mapping[str, object],
) -> dict[str, object]:
    baseline_day = _finite(
        case["baseline_day"],
        "baseline_day",
    )
    baseline_volume = _finite(
        case["baseline_volume_ml"],
        "baseline_volume_ml",
    )
    final_day = _finite(
        case["final_day"],
        "final_day",
    )
    observed_final = _finite(
        case["final_volume_ml"],
        "final_volume_ml",
    )

    prediction_day = (
        final_day - baseline_day
    )
    if prediction_day <= 0:
        raise ValueError(
            "final day must follow baseline"
        )

    particle_seed = seed + case_index
    particle_artifact = (
        build_layer7_prior_particles(
            case,
            n_particles=particle_count,
            seed=particle_seed,
            policy_path=particle_policy_path,
        )
    )
    particles = particle_artifact[
        "parameter_particles"
    ]

    schedule = canonical_act_schedule(
        prediction_day,
        policy=eval_policy,
    )

    simulation_policy = eval_policy[
        "simulation"
    ]
    likelihood_policy = eval_policy[
        "likelihood"
    ]
    health_policy = eval_policy[
        "posterior_health"
    ]

    dt_days = _finite(
        simulation_policy["dt_days"],
        "dt_days",
    )

    early_day_raw = case.get("early_day")
    early_volume_raw = case.get(
        "early_volume_ml"
    )
    early_available = (
        early_day_raw is not None
        and early_volume_raw is not None
    )

    if not early_available:
        prior = _prior_prediction_without_update(
            initial_volume_ml=baseline_volume,
            schedule=schedule,
            particles=particles,
            prediction_day=prediction_day,
            dt_days=dt_days,
        )

        return {
            "case_id": str(case["case_id"]),
            "status": (
                "not_updated_no_valid_observation"
            ),
            "particle_seed": particle_seed,
            "particle_sha256": (
                particle_artifact[
                    "particle_sha256"
                ]
            ),
            "particle_count": particle_count,
            "baseline_day": baseline_day,
            "baseline_volume_ml": (
                baseline_volume
            ),
            "early_day": None,
            "early_volume_ml": None,
            "prediction_day": prediction_day,
            "observed_final_volume_ml": (
                observed_final
            ),
            "schedule": schedule,
            "prior_prediction": prior,
            "posterior_prediction": dict(
                prior
            ),
            "posterior_health": {
                "status": (
                    "not_updated_no_valid_observation"
                ),
                "particle_count": particle_count,
            },
            "exact_prior_posterior_match": True,
            "sampling": particle_artifact[
                "sampling"
            ],
            "future_information_used_for_prior": (
                False
            ),
        }

    early_day = (
        _finite(
            early_day_raw,
            "early_day",
        )
        - baseline_day
    )
    early_volume = _finite(
        early_volume_raw,
        "early_volume_ml",
    )

    if not 0 < early_day < prediction_day:
        raise ValueError(
            "early day must lie between baseline and final day"
        )

    source, confidence, qc = _case_qc(
        case
    )

    noise_override = likelihood_policy.get(
        "override_noise_fraction"
    )
    if noise_override is not None:
        noise_override = _finite(
            noise_override,
            "override_noise_fraction",
        )

    posterior = update_volume_posterior(
        initial_volume_ml=baseline_volume,
        treatment_schedule=schedule,
        parameter_particles=particles,
        observations=[
            {
                "day": early_day,
                "tumor_volume_ml": (
                    early_volume
                ),
                "source": source,
                "confidence": confidence,
                "segmentation_qc": qc,
                "observation_id": (
                    f"{case['case_id']}_early"
                ),
            }
        ],
        prediction_days=[prediction_day],
        dt_days=dt_days,
        likelihood_noise_fraction=(
            noise_override
        ),
        ess_threshold_fraction=_finite(
            likelihood_policy[
                "ess_threshold_fraction"
            ],
            "ess_threshold_fraction",
        ),
        include_failed_qc_observations=bool(
            likelihood_policy[
                "include_failed_qc_observations"
            ]
        ),
        allow_beyond_schedule=bool(
            simulation_policy[
                "allow_beyond_schedule"
            ]
        ),
    )

    particle_rows = posterior.get(
        "particle_trajectories"
    )
    if (
        not isinstance(
            particle_rows,
            Sequence,
        )
        or isinstance(
            particle_rows,
            (str, bytes),
        )
        or not particle_rows
    ):
        raise ValueError(
            "posterior runtime returned no particle trajectories"
        )

    typed_particle_rows = []
    for row_index, row in enumerate(
        particle_rows
    ):
        if not isinstance(row, Mapping):
            raise ValueError(
                "posterior particle "
                f"{row_index} must be a mapping"
            )
        typed_particle_rows.append(row)

    prior_weights = normalize_log_weights(
        [
            _finite(
                row.get("log_prior_weight"),
                "log prior weight",
            )
            for row in typed_particle_rows
        ]
    )
    posterior_weights = [
        _finite(
            row.get("weight"),
            "posterior particle weight",
        )
        for row in typed_particle_rows
    ]

    prior_prediction = (
        _prediction_from_summary(
            posterior[
                "prior_trajectory_summary"
            ],
            prediction_day,
            particle_rows=(
                typed_particle_rows
            ),
            weights=prior_weights,
        )
    )
    posterior_prediction = (
        _prediction_from_summary(
            posterior[
                "posterior_trajectory_summary"
            ],
            prediction_day,
            particle_rows=(
                typed_particle_rows
            ),
            weights=posterior_weights,
        )
    )

    health = summarize_posterior_health(
        posterior,
        boundary_fraction=_finite(
            health_policy[
                "boundary_fraction_of_hard_range"
            ],
            "boundary fraction",
        ),
        boundary_collapse_mass=_finite(
            health_policy[
                "boundary_collapse_mass"
            ],
            "boundary collapse mass",
        ),
        boundary_mass_increase=_finite(
            health_policy[
                "boundary_mass_increase"
            ],
            "boundary mass increase",
        ),
        material_mass_target=_finite(
            health_policy[
                "material_mass_target"
            ],
            "material mass target",
        ),
    )

    return {
        "case_id": str(case["case_id"]),
        "status": health["status"],
        "particle_seed": particle_seed,
        "particle_sha256": (
            particle_artifact[
                "particle_sha256"
            ]
        ),
        "particle_count": particle_count,
        "baseline_day": baseline_day,
        "baseline_volume_ml": (
            baseline_volume
        ),
        "early_day": early_day,
        "early_volume_ml": early_volume,
        "prediction_day": prediction_day,
        "observed_final_volume_ml": (
            observed_final
        ),
        "schedule": schedule,
        "prior_prediction": (
            prior_prediction
        ),
        "posterior_prediction": (
            posterior_prediction
        ),
        "posterior_health": health,
        "exact_prior_posterior_match": (
            prior_prediction
            == posterior_prediction
        ),
        "sampling": particle_artifact[
            "sampling"
        ],
        "future_information_used_for_prior": (
            False
        ),
    }


def _metrics(
    rows: Sequence[Mapping[str, object]],
    prediction_name: str,
) -> dict[str, float]:
    if not rows:
        return {
            "n": 0,
            "mae_ml": float("nan"),
            "rmse_ml": float("nan"),
            "log_volume_rmse": (
                float("nan")
            ),
            "mape": float("nan"),
            "coverage_80": float("nan"),
            "coverage_95": float("nan"),
            "mean_width_80_ml": (
                float("nan")
            ),
            "mean_width_95_ml": (
                float("nan")
            ),
        }

    absolute_errors = []
    squared_errors = []
    squared_log_errors = []
    percentage_errors = []
    covered80 = 0
    covered95 = 0
    widths80 = []
    widths95 = []

    for row in rows:
        observed = _finite(
            row["observed_final_volume_ml"],
            "observed final",
        )
        prediction = row[prediction_name]

        point = _finite(
            prediction["point_ml"],
            "prediction point",
        )
        lower80 = _finite(
            prediction["lower_80_ml"],
            "lower80",
        )
        upper80 = _finite(
            prediction["upper_80_ml"],
            "upper80",
        )
        lower95 = _finite(
            prediction["lower_95_ml"],
            "lower95",
        )
        upper95 = _finite(
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
        covered80 += int(
            lower80 <= observed <= upper80
        )
        covered95 += int(
            lower95 <= observed <= upper95
        )
        widths80.append(
            upper80 - lower80
        )
        widths95.append(
            upper95 - lower95
        )

    n = len(rows)

    return {
        "n": n,
        "mae_ml": (
            sum(absolute_errors) / n
        ),
        "rmse_ml": math.sqrt(
            sum(squared_errors) / n
        ),
        "log_volume_rmse": math.sqrt(
            sum(squared_log_errors) / n
        ),
        "mape": (
            sum(percentage_errors) / n
        ),
        "coverage_80": covered80 / n,
        "coverage_95": covered95 / n,
        "mean_width_80_ml": (
            sum(widths80) / n
        ),
        "mean_width_95_ml": (
            sum(widths95) / n
        ),
    }


def _quantiles(
    values: Sequence[float],
) -> dict[str, float] | None:
    if not values:
        return None

    ordered = sorted(values)

    def value(probability: float) -> float:
        index = round(
            probability
            * (len(ordered) - 1)
        )
        return ordered[index]

    return {
        "minimum": ordered[0],
        "p05": value(0.05),
        "median": value(0.50),
        "p95": value(0.95),
        "maximum": ordered[-1],
        "mean": statistics.fmean(
            ordered
        ),
    }


def run_layer7_real_data_eval(
    cohort_path: Path,
    *,
    particle_count: int,
    seed: int,
    particle_policy_path: Path,
    eval_policy_path: Path,
) -> dict[str, object]:
    if particle_count < 16:
        raise ValueError(
            "particle_count must be at least 16"
        )

    eval_policy = _load_policy(
        eval_policy_path
    )
    cases = load_real_cohort(
        cohort_path,
        allow_demo_data=False,
    )

    rows = []
    for index, case in enumerate(cases):
        rows.append(
            evaluate_layer7_case(
                case,
                case_index=index,
                particle_count=(
                    particle_count
                ),
                seed=seed,
                particle_policy_path=(
                    particle_policy_path
                ),
                eval_policy=eval_policy,
            )
        )

    status_counts = Counter(
        str(row["status"])
        for row in rows
    )

    update_rows = [
        row
        for row in rows
        if row["early_day"] is not None
    ]
    missing_rows = [
        row
        for row in rows
        if row["early_day"] is None
    ]

    prior_metrics_all = _metrics(
        rows,
        "prior_prediction",
    )
    posterior_metrics_all = _metrics(
        rows,
        "posterior_prediction",
    )
    prior_metrics_updated = _metrics(
        update_rows,
        "prior_prediction",
    )
    posterior_metrics_updated = _metrics(
        update_rows,
        "posterior_prediction",
    )

    helped = harmed = unchanged = 0
    case_deltas = []

    for row in update_rows:
        observed = _finite(
            row["observed_final_volume_ml"],
            "observed final",
        )
        prior_error = abs(
            _finite(
                row["prior_prediction"][
                    "point_ml"
                ],
                "prior point",
            )
            - observed
        )
        posterior_error = abs(
            _finite(
                row["posterior_prediction"][
                    "point_ml"
                ],
                "posterior point",
            )
            - observed
        )
        delta = (
            posterior_error - prior_error
        )

        if delta < -1e-12:
            outcome = "helped"
            helped += 1
        elif delta > 1e-12:
            outcome = "harmed"
            harmed += 1
        else:
            outcome = "unchanged"
            unchanged += 1

        case_deltas.append(
            {
                "case_id": row["case_id"],
                "outcome": outcome,
                "prior_abs_error_ml": (
                    prior_error
                ),
                "posterior_abs_error_ml": (
                    posterior_error
                ),
                "posterior_minus_prior_abs_error_ml": (
                    delta
                ),
                "status": row["status"],
            }
        )

    health_rows = [
        row["posterior_health"]
        for row in update_rows
    ]

    ess_fractions = [
        _finite(
            health["weights"][
                "effective_sample_size_fraction"
            ],
            "ESS fraction",
        )
        for health in health_rows
    ]
    maximum_weights = [
        _finite(
            health["weights"][
                "maximum_weight"
            ],
            "maximum weight",
        )
        for health in health_rows
    ]
    material_fractions = [
        _finite(
            health["weights"][
                "material_particle_fraction"
            ],
            "material particle fraction",
        )
        for health in health_rows
    ]
    kl_values = [
        _finite(
            health["weights"][
                "posterior_to_prior_kl"
            ],
            "posterior KL",
        )
        for health in health_rows
    ]

    contraction = {
        name: _quantiles(
            [
                _finite(
                    health["parameters"][name][
                        "contraction_ratio_80"
                    ],
                    f"{name} contraction",
                )
                for health in health_rows
            ]
        )
        for name in (
            "growth_rate_per_day",
            "active_treatment_sensitivity",
            "resistant_fraction",
        )
    }

    unexpected_missing_change = [
        row["case_id"]
        for row in missing_rows
        if not row[
            "exact_prior_posterior_match"
        ]
    ]
    future_leakage_cases = [
        row["case_id"]
        for row in rows
        if row[
            "future_information_used_for_prior"
        ]
    ]

    if unexpected_missing_change:
        raise RuntimeError(
            "early-missing cases changed after no update: "
            + ", ".join(
                unexpected_missing_change
            )
        )
    if future_leakage_cases:
        raise RuntimeError(
            "future information entered prior construction"
        )

    output = {
        "analysis_version": (
            LAYER7_REAL_DATA_EVAL_VERSION
        ),
        "status": (
            "layer7_exploratory_real_data_complete"
        ),
        "interpretation": list(
            eval_policy["interpretation"]
        ),
        "cohort": {
            "path": str(cohort_path),
            "sha256": sha256_file(
                cohort_path
            ),
            "case_count": len(rows),
        },
        "configuration": {
            "particle_count": (
                particle_count
            ),
            "seed": seed,
            "particle_policy_path": str(
                particle_policy_path
            ),
            "particle_policy_sha256": (
                sha256_file(
                    particle_policy_path
                )
            ),
            "eval_policy_path": str(
                eval_policy_path
            ),
            "eval_policy_sha256": (
                sha256_file(
                    eval_policy_path
                )
            ),
            "schedule_proxy": (
                eval_policy[
                    "schedule_proxy"
                ]
            ),
            "likelihood": (
                eval_policy["likelihood"]
            ),
        },
        "dataset": {
            "case_count": len(rows),
            "updated_case_count": len(
                update_rows
            ),
            "early_missing_count": len(
                missing_rows
            ),
            "status_counts": dict(
                sorted(
                    status_counts.items()
                )
            ),
            "exact_no_update_count": sum(
                bool(
                    row[
                        "exact_prior_posterior_match"
                    ]
                )
                for row in missing_rows
            ),
        },
        "metrics": {
            "all_cases": {
                "prior": prior_metrics_all,
                "posterior": (
                    posterior_metrics_all
                ),
            },
            "update_eligible_cases": {
                "prior": (
                    prior_metrics_updated
                ),
                "posterior": (
                    posterior_metrics_updated
                ),
                "posterior_minus_prior": {
                    key: (
                        posterior_metrics_updated[
                            key
                        ]
                        - prior_metrics_updated[
                            key
                        ]
                    )
                    for key in (
                        "mae_ml",
                        "rmse_ml",
                        "log_volume_rmse",
                        "mape",
                        "coverage_80",
                        "coverage_95",
                        "mean_width_80_ml",
                        "mean_width_95_ml",
                    )
                },
            },
        },
        "update_value": {
            "helped_count": helped,
            "harmed_count": harmed,
            "unchanged_count": unchanged,
            "mean_abs_error_change_ml": (
                statistics.fmean(
                    row[
                        "posterior_minus_prior_abs_error_ml"
                    ]
                    for row in case_deltas
                )
                if case_deltas
                else None
            ),
            "cases": case_deltas,
        },
        "posterior_health": {
            "status_counts": dict(
                sorted(
                    Counter(
                        health["status"]
                        for health in health_rows
                    ).items()
                )
            ),
            "effective_sample_size_fraction": (
                _quantiles(
                    ess_fractions
                )
            ),
            "maximum_weight": (
                _quantiles(
                    maximum_weights
                )
            ),
            "material_particle_fraction": (
                _quantiles(
                    material_fractions
                )
            ),
            "posterior_to_prior_kl": (
                _quantiles(
                    kl_values
                )
            ),
            "parameter_contraction_ratio_80": (
                contraction
            ),
            "boundary_collapse_case_count": (
                sum(
                    bool(
                        health[
                            "boundary_collapse_parameters"
                        ]
                    )
                    for health in health_rows
                )
            ),
        },
        "requirements": {
            "all_cases_evaluated": (
                len(rows) == len(cases)
            ),
            "early_missing_exact_no_update": (
                not unexpected_missing_change
            ),
            "future_outcome_used_for_inference": (
                False
            ),
            "silent_resampling_performed": (
                False
            ),
            "patient_specific_schedule_available": (
                False
            ),
            "confirmatory_evaluation": False,
        },
        "case_results": rows,
    }

    return output


def write_outputs(
    result: Mapping[str, object],
    *,
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    updated = result["metrics"][
        "update_eligible_cases"
    ]
    dataset = result["dataset"]
    health = result["posterior_health"]
    update_value = result["update_value"]

    lines = [
        "# Layer 7 exploratory real-cohort evaluation",
        "",
        (
            "**Exploratory only — this is not a locked "
            "confirmatory evaluation.**"
        ),
        "",
        "## Dataset",
        "",
        (
            f"- Cases: {dataset['case_count']}"
        ),
        (
            "- Updated with early MRI: "
            f"{dataset['updated_case_count']}"
        ),
        (
            "- Exact no-update cases: "
            f"{dataset['exact_no_update_count']}"
        ),
        "",
        "## Prior versus posterior",
        "",
        "| Metric | Prior | Posterior | Posterior − prior |",
        "| --- | ---: | ---: | ---: |",
    ]

    for key in (
        "mae_ml",
        "rmse_ml",
        "log_volume_rmse",
        "mape",
        "coverage_80",
        "coverage_95",
        "mean_width_80_ml",
        "mean_width_95_ml",
    ):
        lines.append(
            "| "
            + key
            + " | "
            + str(updated["prior"][key])
            + " | "
            + str(updated["posterior"][key])
            + " | "
            + str(
                updated[
                    "posterior_minus_prior"
                ][key]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Update value",
            "",
            (
                "- Helped cases: "
                f"{update_value['helped_count']}"
            ),
            (
                "- Harmed cases: "
                f"{update_value['harmed_count']}"
            ),
            (
                "- Unchanged cases: "
                f"{update_value['unchanged_count']}"
            ),
            (
                "- Mean posterior-minus-prior "
                "absolute error: "
                f"{update_value['mean_abs_error_change_ml']}"
            ),
            "",
            "## Posterior health",
            "",
            (
                "- Status counts: "
                f"{health['status_counts']}"
            ),
            (
                "- ESS fraction: "
                f"{health['effective_sample_size_fraction']}"
            ),
            (
                "- Maximum particle weight: "
                f"{health['maximum_weight']}"
            ),
            (
                "- Material particle fraction: "
                f"{health['material_particle_fraction']}"
            ),
            (
                "- Boundary-collapse cases: "
                f"{health['boundary_collapse_case_count']}"
            ),
            "",
            "## Scope warning",
            "",
            (
                "The current evaluation uses a canonical "
                "A/C-T schedule proxy because the normalized "
                "cohort does not contain patient-level treatment "
                "administration events. The likelihood, schedule, "
                "prior policy, and health thresholds remain "
                "exploratory."
            ),
            "",
        ]
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exploratory Layer 7 real-cohort "
            "prior-versus-posterior evaluation."
        )
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )
    parser.add_argument(
        "--particle-policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
    )
    parser.add_argument(
        "--eval-policy",
        type=Path,
        default=DEFAULT_EVAL_POLICY,
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=730000,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
    )
    args = parser.parse_args()

    result = run_layer7_real_data_eval(
        args.cohort,
        particle_count=args.particles,
        seed=args.seed,
        particle_policy_path=(
            args.particle_policy
        ),
        eval_policy_path=args.eval_policy,
    )
    write_outputs(
        result,
        output_path=args.output,
        report_path=args.report,
    )

    print(
        "analysis_version:",
        result["analysis_version"],
    )
    print("status:", result["status"])
    print("dataset:", result["dataset"])
    print(
        "updated_metrics:",
        result["metrics"][
            "update_eligible_cases"
        ],
    )
    print(
        "update_value:",
        {
            key: result["update_value"][key]
            for key in (
                "helped_count",
                "harmed_count",
                "unchanged_count",
                "mean_abs_error_change_ml",
            )
        },
    )
    print(
        "posterior_health:",
        result["posterior_health"],
    )
    print("json_artifact:", args.output)
    print("markdown_report:", args.report)
    print(
        "json_sha256:",
        _sha256(args.output),
    )
    print(
        "markdown_sha256:",
        _sha256(args.report),
    )
    print(
        "LAYER7_EXPLORATORY_EVAL_STATUS: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
