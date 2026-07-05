"""Recovery, identifiability, and held-out prediction sweep for v0."""

from __future__ import annotations

import copy
import math
from typing import Any

from .identifiability import analyze_identifiability
from .params import sample_volume_params
from .summarize import flatten_driver_params, quantile
from .synthetic_fit import make_noisy_observations, reweight_particles_from_observations
from .validation import schedule_drugs
from .volume_ode import simulate_volume_trajectory


DEFAULT_FIT_DAYS = [0.0, 42.0, 84.0]
DEFAULT_HELDOUT_DAY = 126.0
TRACKED_PARAMETERS = [
    "growth_rate",
    "resistant_fraction",
    "resistant_sensitivity_scale",
    "drug_sensitivity.anthracycline",
    "drug_sensitivity.taxane",
]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def rmse(predicted: list[float], observed: list[float]) -> float:
    return math.sqrt(
        sum((pred - obs) ** 2 for pred, obs in zip(predicted, observed)) / len(observed)
    )


def absolute_error(predicted: float, observed: float) -> float:
    return abs(float(predicted) - float(observed))


def make_prior_variant(
    base_prior: dict[str, Any],
    variant: str,
    active_drugs: list[str],
    assumed_noise_fraction: float,
) -> dict[str, Any]:
    """Return a prior variant for parameter-ablation experiments."""

    prior = copy.deepcopy(base_prior)
    prior.setdefault("fixed", {})
    prior.setdefault("distributions", {})

    if variant == "full":
        return prior

    if variant in {"active_drugs_only", "fixed_core", "shared_chemo_fixed_core"}:
        for mapping_name in ("drug_ec50", "drug_decay"):
            mapping = prior["fixed"].get(mapping_name, {})
            prior["fixed"][mapping_name] = {
                drug: value for drug, value in mapping.items() if drug in active_drugs
            }
        sensitivity_specs = prior["distributions"].get("drug_sensitivity", {})
        prior["distributions"]["drug_sensitivity"] = {
            drug: spec for drug, spec in sensitivity_specs.items() if drug in active_drugs
        }

    if variant in {"fixed_core", "shared_chemo_fixed_core"}:
        prior["fixed"]["carrying_capacity_ml"] = 260.0
        prior["fixed"]["resistant_sensitivity_scale"] = 0.1
        prior["fixed"]["observation_noise_fraction"] = assumed_noise_fraction
        for key in (
            "carrying_capacity_ml",
            "resistant_sensitivity_scale",
            "observation_noise_fraction",
        ):
            prior["distributions"].pop(key, None)

    if variant not in {"active_drugs_only", "fixed_core", "shared_chemo_fixed_core"}:
        raise ValueError(f"unknown prior variant: {variant}")

    return prior


def transform_particles_for_variant(
    particles: list[dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    """Apply variant-specific parameter tying after sampling."""

    transformed = copy.deepcopy(particles)
    if variant == "shared_chemo_fixed_core":
        for particle in transformed:
            sensitivities = particle.get("drug_sensitivity", {})
            if "anthracycline" in sensitivities and "taxane" in sensitivities:
                shared = 0.5 * (sensitivities["anthracycline"] + sensitivities["taxane"])
                sensitivities["anthracycline"] = shared
                sensitivities["taxane"] = shared
    return transformed


def linear_prediction(fit_days: list[float], fit_volumes: list[float], target_day: float) -> float:
    x_mean = mean(fit_days)
    y_mean = mean(fit_volumes)
    denom = sum((day - x_mean) ** 2 for day in fit_days)
    if denom == 0:
        return max(0.0, fit_volumes[-1])
    slope = sum((day - x_mean) * (volume - y_mean) for day, volume in zip(fit_days, fit_volumes))
    slope /= denom
    intercept = y_mean - slope * x_mean
    return max(0.0, intercept + slope * target_day)


def exponential_prediction(
    fit_days: list[float],
    fit_volumes: list[float],
    target_day: float,
) -> float:
    safe_volumes = [max(volume, 1e-6) for volume in fit_volumes]
    log_volumes = [math.log(value) for value in safe_volumes]
    x_mean = mean(fit_days)
    y_mean = mean(log_volumes)
    denom = sum((day - x_mean) ** 2 for day in fit_days)
    if denom == 0:
        return safe_volumes[-1]
    slope = sum((day - x_mean) * (volume - y_mean) for day, volume in zip(fit_days, log_volumes))
    slope /= denom
    intercept = y_mean - slope * x_mean
    log_prediction = intercept + slope * target_day
    return math.exp(log_prediction)


def last_slope_prediction(
    fit_days: list[float],
    fit_volumes: list[float],
    target_day: float,
) -> float:
    if len(fit_days) < 2 or fit_days[-1] == fit_days[-2]:
        return max(0.0, fit_volumes[-1])
    slope = (fit_volumes[-1] - fit_volumes[-2]) / (fit_days[-1] - fit_days[-2])
    return max(0.0, fit_volumes[-1] + slope * (target_day - fit_days[-1]))


def baseline_predictions(
    fit_observations: list[dict[str, Any]],
    target_day: float,
) -> dict[str, float]:
    fit_days = [float(obs["day"]) for obs in fit_observations]
    fit_volumes = [float(obs["tumor_volume_ml"]) for obs in fit_observations]
    return {
        "last_observation": max(0.0, fit_volumes[-1]),
        "last_slope": last_slope_prediction(fit_days, fit_volumes, target_day),
        "linear": linear_prediction(fit_days, fit_volumes, target_day),
        "exponential": exponential_prediction(fit_days, fit_volumes, target_day),
    }


def _posterior_mean_for_param(
    particle_trajectories: list[dict[str, Any]],
    parameter_name: str,
) -> float | None:
    values: list[float] = []
    weights: list[float] = []
    for particle in particle_trajectories:
        flattened = flatten_driver_params(particle["parameters"])
        if parameter_name not in flattened:
            continue
        values.append(flattened[parameter_name])
        weights.append(float(particle.get("weight") or 0.0))
    if not values:
        return None
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return mean(values)
    return sum(value * weight for value, weight in zip(values, weights)) / weight_sum


def run_single_sweep_case(
    case: dict[str, Any],
    schedule: dict[str, Any],
    base_prior: dict[str, Any],
    truth_params: dict[str, Any],
    variant: str,
    particle_count: int,
    seed: int,
    assumed_noise_fraction: float,
    generated_noise_fraction: float,
    fit_days: list[float] | None = None,
    heldout_day: float = DEFAULT_HELDOUT_DAY,
    dt_days: float = 0.5,
) -> dict[str, Any]:
    fit_days = fit_days or DEFAULT_FIT_DAYS
    all_days = sorted(set(fit_days + [float(heldout_day)]))
    initial_volume = float(case["baseline_measurement"]["tumor_volume_ml"])
    active_drugs = schedule_drugs(schedule)

    truth = simulate_volume_trajectory(
        initial_volume_ml=initial_volume,
        treatment_schedule=schedule,
        params=truth_params,
        output_days=all_days,
        dt_days=0.25,
    )
    all_observations = make_noisy_observations(
        truth,
        noise_fraction=generated_noise_fraction,
        seed=seed + 100_000,
    )
    observations_by_day = {float(obs["day"]): obs for obs in all_observations}
    fit_observations = [observations_by_day[day] for day in fit_days]
    heldout_observation = observations_by_day[float(heldout_day)]
    true_by_day = {
        float(record["day"]): float(record["tumor_volume_ml"])
        for record in truth["trajectory"]
    }

    prior = make_prior_variant(base_prior, variant, active_drugs, assumed_noise_fraction)
    particles = sample_volume_params(prior, n_particles=particle_count, seed=seed)
    particles = transform_particles_for_variant(particles, variant)
    fit = reweight_particles_from_observations(
        initial_volume_ml=initial_volume,
        treatment_schedule=schedule,
        parameter_particles=particles,
        observations=fit_observations,
        dt_days=dt_days,
        likelihood_noise_fraction=assumed_noise_fraction,
        prediction_days=[heldout_day],
    )
    identifiability = analyze_identifiability(
        fit["particle_trajectories"],
        observations=fit_observations,
        active_drugs=active_drugs,
    )

    heldout_index = fit["prediction_times"].index(float(heldout_day))
    prior_heldout = fit["prior_mean_all_times_ml"][heldout_index]
    posterior_heldout = fit["posterior_mean_all_times_ml"][heldout_index]
    baselines = baseline_predictions(fit_observations, heldout_day)
    true_heldout = true_by_day[float(heldout_day)]
    observed_heldout = float(heldout_observation["tumor_volume_ml"])

    methods = {
        "prior_particle_mean": prior_heldout,
        "posterior_particle_mean": posterior_heldout,
        **baselines,
    }
    heldout_errors = {
        method: {
            "absolute_error_to_true_ml": absolute_error(prediction, true_heldout),
            "absolute_error_to_observed_ml": absolute_error(prediction, observed_heldout),
            "prediction_ml": prediction,
        }
        for method, prediction in methods.items()
    }

    posterior_parameter_means = {
        parameter: _posterior_mean_for_param(fit["particle_trajectories"], parameter)
        for parameter in TRACKED_PARAMETERS
    }
    posterior_parameter_means = {
        key: value for key, value in posterior_parameter_means.items() if value is not None
    }

    return {
        "variant": variant,
        "particle_count": particle_count,
        "seed": seed,
        "assumed_noise_fraction": assumed_noise_fraction,
        "generated_noise_fraction": generated_noise_fraction,
        "fit_days": fit_days,
        "heldout_day": heldout_day,
        "prior_rmse_fit_ml": fit["prior_rmse"],
        "posterior_rmse_fit_ml": fit["posterior_rmse"],
        "rmse_improvement_fraction": (
            (fit["prior_rmse"] - fit["posterior_rmse"]) / fit["prior_rmse"]
            if fit["prior_rmse"] > 0
            else 0.0
        ),
        "effective_sample_size": fit["effective_sample_size"],
        "effective_sample_size_fraction": fit["effective_sample_size"] / particle_count,
        "heldout_true_volume_ml": true_heldout,
        "heldout_observed_volume_ml": observed_heldout,
        "heldout_errors": heldout_errors,
        "posterior_parameter_means": posterior_parameter_means,
        "classification_counts": {
            "constrained": len(identifiability["constrained_parameters"]),
            "weakly_constrained": len(identifiability["weakly_constrained_parameters"]),
            "prior_dominated": len(identifiability["prior_dominated_parameters"]),
        },
        "constrained_parameters": identifiability["constrained_parameters"],
        "weakly_constrained_parameters": identifiability["weakly_constrained_parameters"],
        "prior_dominated_parameters": identifiability["prior_dominated_parameters"],
        "similar_trajectory_pair_count": len(identifiability["similar_trajectory_pairs"]),
        "warnings": sorted(set(fit["warnings"] + identifiability["warnings"])),
    }


def _group_records(records: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return groups


def _median(values: list[float]) -> float:
    return quantile(values, 0.5)


def _summarize_records(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    summaries = []
    for value, group in sorted(_group_records(records, group_key).items(), key=lambda pair: str(pair[0])):
        summaries.append(
            {
                group_key: value,
                "n_runs": len(group),
                "median_ess": _median([row["effective_sample_size"] for row in group]),
                "median_ess_fraction": _median(
                    [row["effective_sample_size_fraction"] for row in group]
                ),
                "median_fit_prior_rmse_ml": _median([row["prior_rmse_fit_ml"] for row in group]),
                "median_fit_posterior_rmse_ml": _median(
                    [row["posterior_rmse_fit_ml"] for row in group]
                ),
                "median_rmse_improvement_fraction": _median(
                    [row["rmse_improvement_fraction"] for row in group]
                ),
                "median_posterior_heldout_true_error_ml": _median(
                    [
                        row["heldout_errors"]["posterior_particle_mean"][
                            "absolute_error_to_true_ml"
                        ]
                        for row in group
                    ]
                ),
                "median_prior_heldout_true_error_ml": _median(
                    [
                        row["heldout_errors"]["prior_particle_mean"][
                            "absolute_error_to_true_ml"
                        ]
                        for row in group
                    ]
                ),
            }
        )
    return summaries


def summarize_baseline_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(records[0]["heldout_errors"].keys())
    errors_by_method = {
        method: [
            row["heldout_errors"][method]["absolute_error_to_true_ml"]
            for row in records
        ]
        for method in methods
    }
    win_counts = {method: 0 for method in methods}
    for row in records:
        best_method = min(
            methods,
            key=lambda method: row["heldout_errors"][method]["absolute_error_to_true_ml"],
        )
        win_counts[best_method] += 1
    return {
        method: {
            "median_absolute_error_to_true_ml": _median(errors),
            "mean_absolute_error_to_true_ml": mean(errors),
            "win_count": win_counts[method],
        }
        for method, errors in errors_by_method.items()
    }


def summarize_parameter_stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    stability: dict[str, Any] = {}
    for variant, group in _group_records(records, "variant").items():
        stability[variant] = {}
        parameter_names = sorted(
            {
                parameter
                for row in group
                for parameter in row.get("posterior_parameter_means", {})
            }
        )
        for parameter in parameter_names:
            values = [
                row["posterior_parameter_means"][parameter]
                for row in group
                if parameter in row["posterior_parameter_means"]
            ]
            if not values:
                continue
            average = mean(values)
            stability[variant][parameter] = {
                "mean": average,
                "stddev": stddev(values),
                "min": min(values),
                "max": max(values),
                "relative_stddev": stddev(values) / abs(average) if average else 0.0,
            }
    return stability


def generate_insights(records: list[dict[str, Any]], summaries: dict[str, Any]) -> list[str]:
    insights: list[str] = []
    posterior_errors = [
        row["heldout_errors"]["posterior_particle_mean"]["absolute_error_to_true_ml"]
        for row in records
    ]
    prior_errors = [
        row["heldout_errors"]["prior_particle_mean"]["absolute_error_to_true_ml"]
        for row in records
    ]
    median_posterior_error = _median(posterior_errors)
    median_prior_error = _median(prior_errors)
    median_ess_fraction = _median([row["effective_sample_size_fraction"] for row in records])

    if median_posterior_error < median_prior_error:
        insights.append(
            "Particle reweighting improved median held-out prediction versus the prior ensemble."
        )
    else:
        insights.append(
            "Particle reweighting did not improve median held-out prediction versus the prior ensemble."
        )

    if median_ess_fraction < 0.05:
        insights.append(
            "Effective sample size stayed below 5% of particles, so posterior narrowing remains fragile."
        )
    else:
        insights.append(
            "Effective sample size was not extremely concentrated for the median run."
        )

    baseline_summary = summaries["heldout_baseline_comparison"]
    best_baseline = min(
        baseline_summary,
        key=lambda method: baseline_summary[method]["median_absolute_error_to_true_ml"],
    )
    if best_baseline != "posterior_particle_mean":
        insights.append(
            f"The best median held-out method was {best_baseline}, so mechanistic fitting "
            "should be compared against simple baselines before product claims."
        )
    else:
        insights.append(
            "The posterior particle mean was the best median held-out method in this sweep."
        )

    variant_summaries = summaries["by_variant"]
    best_variant = min(
        variant_summaries,
        key=lambda row: row["median_posterior_heldout_true_error_ml"],
    )
    insights.append(
        f"The lowest median posterior held-out error came from the {best_variant['variant']} variant."
    )

    return insights


def run_recovery_sweep(
    case: dict[str, Any],
    schedule: dict[str, Any],
    base_prior: dict[str, Any],
    truth_params: dict[str, Any],
    seeds: list[int],
    particle_counts: list[int],
    assumed_noise_levels: list[float],
    variants: list[str],
    generated_noise_fraction: float = 0.08,
    heldout_day: float = DEFAULT_HELDOUT_DAY,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for variant in variants:
        for particle_count in particle_counts:
            for assumed_noise_fraction in assumed_noise_levels:
                for seed in seeds:
                    records.append(
                        run_single_sweep_case(
                            case=case,
                            schedule=schedule,
                            base_prior=base_prior,
                            truth_params=truth_params,
                            variant=variant,
                            particle_count=particle_count,
                            seed=seed,
                            assumed_noise_fraction=assumed_noise_fraction,
                            generated_noise_fraction=generated_noise_fraction,
                            heldout_day=heldout_day,
                        )
                    )

    summaries = {
        "by_assumed_noise": _summarize_records(records, "assumed_noise_fraction"),
        "by_particle_count": _summarize_records(records, "particle_count"),
        "by_variant": _summarize_records(records, "variant"),
        "heldout_baseline_comparison": summarize_baseline_comparison(records),
        "parameter_stability": summarize_parameter_stability(records),
    }
    return {
        "case_id": case["case_id"],
        "truth_case": "synthetic high-response parameter fixture",
        "generated_noise_fraction": generated_noise_fraction,
        "fit_days": DEFAULT_FIT_DAYS,
        "heldout_day": heldout_day,
        "n_runs": len(records),
        "records": records,
        "summaries": summaries,
        "insights": generate_insights(records, summaries),
        "warnings": [
            "Recovery sweep uses synthetic/demo observations, not clinical evidence.",
            "Held-out comparisons are a feasibility signal only; real/manual data are still required.",
        ],
    }


def write_markdown_report(path: str, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# v0 Recovery and Identifiability Sweep")
    lines.append("")
    lines.append(f"Case: `{report['case_id']}`")
    lines.append(f"Runs: `{report['n_runs']}`")
    lines.append(f"Generated observation noise: `{report['generated_noise_fraction']}`")
    lines.append(f"Fit days: `{report['fit_days']}`")
    lines.append(f"Held-out day: `{report['heldout_day']}`")
    lines.append("")
    lines.append("## Insights")
    lines.extend(f"- {insight}" for insight in report["insights"])
    lines.append("")
    lines.append("## By Assumed Noise")
    lines.append(
        "| assumed noise | runs | median ESS | median ESS fraction | median fit posterior RMSE | median posterior held-out error |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["summaries"]["by_assumed_noise"]:
        lines.append(
            f"| {row['assumed_noise_fraction']:.2f} | {row['n_runs']} | "
            f"{row['median_ess']:.2f} | {row['median_ess_fraction']:.3f} | "
            f"{row['median_fit_posterior_rmse_ml']:.3f} | "
            f"{row['median_posterior_heldout_true_error_ml']:.3f} |"
        )
    lines.append("")
    lines.append("## By Particle Count")
    lines.append(
        "| particles | runs | median ESS | median ESS fraction | median posterior held-out error |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in report["summaries"]["by_particle_count"]:
        lines.append(
            f"| {row['particle_count']} | {row['n_runs']} | {row['median_ess']:.2f} | "
            f"{row['median_ess_fraction']:.3f} | "
            f"{row['median_posterior_heldout_true_error_ml']:.3f} |"
        )
    lines.append("")
    lines.append("## By Variant")
    lines.append(
        "| variant | runs | median ESS fraction | median posterior held-out error | median RMSE improvement |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["summaries"]["by_variant"]:
        lines.append(
            f"| {row['variant']} | {row['n_runs']} | {row['median_ess_fraction']:.3f} | "
            f"{row['median_posterior_heldout_true_error_ml']:.3f} | "
            f"{row['median_rmse_improvement_fraction']:.3f} |"
        )
    lines.append("")
    lines.append("## Held-Out Method Comparison")
    lines.append("| method | median absolute error to truth | mean absolute error | win count |")
    lines.append("| --- | ---: | ---: | ---: |")
    for method, row in sorted(report["summaries"]["heldout_baseline_comparison"].items()):
        lines.append(
            f"| {method} | {row['median_absolute_error_to_true_ml']:.3f} | "
            f"{row['mean_absolute_error_to_true_ml']:.3f} | {row['win_count']} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append(
        "Keep v0 as a research simulator. Use reduced parameter variants and held-out "
        "baseline checks before product-facing parameter explanations."
    )
    lines.append("")
    from pathlib import Path

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
