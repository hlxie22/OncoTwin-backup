"""Batch importance-sampling posterior update runtime for V1 twins.

This module is intentionally small and table/runtime oriented. It reweights a
static set of simulator particles from all observations to date, rather than
running a sequential particle filter that would resample and impoverish static
biology parameters after each scan.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from experiments.v0.mechanistic_simulator.volume_ode import simulate_volume_trajectory


POSTERIOR_RUNTIME_VERSION = "oncotwin_posterior_update_v1"
DEFAULT_NOISE_FRACTION = 0.12
MIN_ABSOLUTE_SIGMA_ML = 0.25
DEFAULT_ESS_THRESHOLD_FRACTION = 0.10
DEFAULT_LOW_RESIDUAL_BURDEN_ML = 1.0
FAILED_QC_LABELS = {"failed", "fail", "empty", "all_zero", "all zero", "no mask"}
LOW_QC_LABELS = {"low", "poor", "motion", "artifact", "manual review"}
MEDIUM_QC_LABELS = {"medium", "moderate", "unknown", "uncertain"}
HIGH_QC_LABELS = {"high", "good", "excellent", "pass", "passed"}


@dataclass(frozen=True)
class VolumeObservation:
    """One tumor-volume observation used for posterior reweighting."""

    day: float
    tumor_volume_ml: float
    source: str = "unknown"
    confidence: str = "unknown"
    segmentation_qc: str = "unknown"
    noise_fraction: float | None = None
    observation_id: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "VolumeObservation":
        return cls(
            day=_required_finite(row.get("day"), "day"),
            tumor_volume_ml=_required_nonnegative(
                row.get("tumor_volume_ml", row.get("volume_ml")),
                "tumor_volume_ml",
            ),
            source=str(row.get("source", "unknown")),
            confidence=str(row.get("confidence", "unknown")),
            segmentation_qc=str(row.get("segmentation_qc", row.get("qc", "unknown"))),
            noise_fraction=_optional_positive(row.get("noise_fraction")),
            observation_id=str(row["observation_id"]) if row.get("observation_id") else None,
        )

    @property
    def failed_qc(self) -> bool:
        return _normalize_label(self.segmentation_qc) in FAILED_QC_LABELS

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "day": self.day,
            "tumor_volume_ml": self.tumor_volume_ml,
            "source": self.source,
            "confidence": self.confidence,
            "segmentation_qc": self.segmentation_qc,
        }
        if self.noise_fraction is not None:
            payload["noise_fraction"] = self.noise_fraction
        if self.observation_id:
            payload["observation_id"] = self.observation_id
        return payload


def update_volume_posterior(
    *,
    initial_volume_ml: float,
    treatment_schedule: Mapping[str, Any],
    parameter_particles: Sequence[Mapping[str, Any]],
    observations: Sequence[VolumeObservation | Mapping[str, object]],
    prediction_days: Sequence[float] = (),
    dt_days: float = 0.5,
    likelihood_noise_fraction: float | None = None,
    ess_threshold_fraction: float = DEFAULT_ESS_THRESHOLD_FRACTION,
    residual_burden_threshold_ml: float = DEFAULT_LOW_RESIDUAL_BURDEN_ML,
    include_failed_qc_observations: bool = False,
    allow_beyond_schedule: bool = False,
) -> dict[str, object]:
    """Reweight prior particles against all tumor-volume observations to date."""

    initial_volume = _required_nonnegative(initial_volume_ml, "initial_volume_ml")
    if not parameter_particles:
        raise ValueError("parameter_particles must contain at least one particle")
    if ess_threshold_fraction <= 0 or ess_threshold_fraction > 1:
        raise ValueError("ess_threshold_fraction must be in (0, 1]")

    accepted_observations, skipped_observations, observation_warnings = _accepted_observations(
        observations,
        include_failed_qc_observations=include_failed_qc_observations,
    )
    if not accepted_observations:
        raise ValueError("at least one non-failed tumor-volume observation is required")

    observation_days = [observation.day for observation in accepted_observations]
    output_days = _sorted_unique([*observation_days, *[float(day) for day in prediction_days]])
    prior_log_weights = [
        _particle_log_prior_weight(particle, len(parameter_particles))
        for particle in parameter_particles
    ]
    prior_weights = normalize_log_weights(prior_log_weights)

    particle_rows: list[dict[str, object]] = []
    posterior_log_weights: list[float] = []
    simulator_warnings: list[str] = []

    for index, particle in enumerate(parameter_particles):
        particle_params = dict(particle)
        particle_id = str(particle_params.get("particle_id", f"p{index:06d}"))
        result = simulate_volume_trajectory(
            initial_volume_ml=initial_volume,
            treatment_schedule=dict(treatment_schedule),
            params=particle_params,
            output_days=output_days,
            dt_days=dt_days,
            allow_beyond_schedule=allow_beyond_schedule,
        )
        predicted = [float(record["tumor_volume_ml"]) for record in result["trajectory"]]
        predicted_by_day = dict(zip(output_days, predicted))
        log_likelihood = sum(
            tumor_volume_log_likelihood(
                predicted_volume_ml=predicted_by_day[observation.day],
                observation=observation,
                likelihood_noise_fraction=likelihood_noise_fraction,
            )
            for observation in accepted_observations
        )
        log_weight = prior_log_weights[index] + log_likelihood
        posterior_log_weights.append(log_weight)

        warnings = [str(warning) for warning in result.get("warnings", [])]
        simulator_warnings.extend(warnings)
        particle_rows.append(
            {
                "particle_id": particle_id,
                "parameters": particle_params,
                "times": output_days,
                "predicted_volume_ml": predicted,
                "predicted_longest_diameter_cm": [
                    float(record["predicted_longest_diameter_cm"])
                    for record in result["trajectory"]
                ],
                "log_prior_weight": prior_log_weights[index],
                "log_likelihood": log_likelihood,
                "log_weight": log_weight,
                "weight": None,
                "warnings": warnings,
            }
        )

    posterior_weights = normalize_log_weights(posterior_log_weights)
    for row, weight in zip(particle_rows, posterior_weights):
        row["weight"] = weight

    prior_trajectory = summarize_weighted_trajectories(
        particle_rows,
        output_days=output_days,
        weights=prior_weights,
        residual_burden_threshold_ml=residual_burden_threshold_ml,
    )
    posterior_trajectory = summarize_weighted_trajectories(
        particle_rows,
        output_days=output_days,
        weights=posterior_weights,
        residual_burden_threshold_ml=residual_burden_threshold_ml,
    )
    ess = effective_sample_size(posterior_weights)
    ess_threshold = ess_threshold_fraction * len(parameter_particles)
    fallback_status = "not_needed"
    fallback_reason = None
    if ess < ess_threshold:
        fallback_status = "tempered_smc_recommended"
        fallback_reason = (
            "posterior effective sample size is below threshold; "
            "future tempered SMC should move particles instead of resampling duplicates"
        )

    warnings = [
        *observation_warnings,
        *_stable_unique(simulator_warnings),
    ]
    if fallback_reason:
        warnings.append(fallback_reason)

    return {
        "posterior_runtime_version": POSTERIOR_RUNTIME_VERSION,
        "update_algorithm": "batch_importance_sampling_from_prior",
        "initial_volume_ml": initial_volume,
        "n_prior_particles": len(parameter_particles),
        "n_observations": len(accepted_observations),
        "observations": [observation.as_dict() for observation in accepted_observations],
        "skipped_observations": [observation.as_dict() for observation in skipped_observations],
        "prediction_days": output_days,
        "prior_effective_sample_size": effective_sample_size(prior_weights),
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / len(parameter_particles),
        "ess_threshold": ess_threshold,
        "fallback_status": fallback_status,
        "fallback_reason": fallback_reason,
        "prior_trajectory_summary": prior_trajectory,
        "posterior_trajectory_summary": posterior_trajectory,
        "parameter_summary": summarize_weighted_parameters(
            parameter_particles,
            prior_weights=prior_weights,
            posterior_weights=posterior_weights,
        ),
        "particle_trajectories": particle_rows,
        "uncertainty_summary": build_uncertainty_summary(
            observations=accepted_observations,
            skipped_observations=skipped_observations,
            effective_sample_size_fraction=ess / len(parameter_particles),
            posterior_trajectory_summary=posterior_trajectory,
            fallback_status=fallback_status,
        ),
        "update_explanation": build_update_explanation(
            observations=accepted_observations,
            prior_trajectory_summary=prior_trajectory,
            posterior_trajectory_summary=posterior_trajectory,
            effective_sample_size_fraction=ess / len(parameter_particles),
            fallback_status=fallback_status,
        ),
        "warnings": warnings,
    }


def tumor_volume_log_likelihood(
    *,
    predicted_volume_ml: float,
    observation: VolumeObservation | Mapping[str, object],
    likelihood_noise_fraction: float | None = None,
) -> float:
    """Gaussian log likelihood for one tumor-volume observation."""

    obs = _coerce_observation(observation)
    predicted = _required_nonnegative(predicted_volume_ml, "predicted_volume_ml")
    noise_fraction = (
        _required_positive(likelihood_noise_fraction, "likelihood_noise_fraction")
        if likelihood_noise_fraction is not None
        else resolve_volume_observation_noise_fraction(obs)
    )
    sigma = max(noise_fraction * max(obs.tumor_volume_ml, 1.0), MIN_ABSOLUTE_SIGMA_ML)
    residual = obs.tumor_volume_ml - predicted
    return -0.5 * (residual / sigma) ** 2 - math.log(sigma)


def resolve_volume_observation_noise_fraction(
    observation: VolumeObservation | Mapping[str, object],
    *,
    default_noise_fraction: float = DEFAULT_NOISE_FRACTION,
) -> float:
    """Resolve an observation-specific noise fraction from source and QC metadata."""

    obs = _coerce_observation(observation)
    if obs.noise_fraction is not None:
        return _required_positive(obs.noise_fraction, "noise_fraction")

    multiplier = 1.0
    source = _normalize_label(obs.source)
    if source in {"manual", "report", "radiology_report", "metadata"}:
        multiplier *= 1.5
    elif source in {"mask_derived", "mri_segmentation", "segmentation"}:
        multiplier *= 1.0
    else:
        multiplier *= 1.25

    confidence = _normalize_label(obs.confidence)
    qc = _normalize_label(obs.segmentation_qc)
    for label in (confidence, qc):
        if label in HIGH_QC_LABELS:
            multiplier *= 0.9
        elif label in MEDIUM_QC_LABELS:
            multiplier *= 1.25
        elif label in LOW_QC_LABELS:
            multiplier *= 2.0
        elif label in FAILED_QC_LABELS:
            multiplier *= 4.0

    return max(default_noise_fraction * multiplier, 0.03)


def normalize_log_weights(log_weights: Sequence[float]) -> list[float]:
    if not log_weights:
        raise ValueError("log_weights must not be empty")
    finite = [_required_finite(value, "log_weight") for value in log_weights]
    max_log_weight = max(finite)
    raw = [math.exp(value - max_log_weight) for value in finite]
    total = sum(raw)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("log_weights could not be normalized")
    return [value / total for value in raw]


def effective_sample_size(weights: Sequence[float]) -> float:
    if not weights:
        raise ValueError("weights must not be empty")
    total = sum(_required_nonnegative(weight, "weight") for weight in weights)
    if total <= 0:
        raise ValueError("weights must have positive total")
    normalized = [weight / total for weight in weights]
    return 1.0 / sum(weight * weight for weight in normalized)


def summarize_weighted_trajectories(
    particle_rows: Sequence[Mapping[str, object]],
    *,
    output_days: Sequence[float],
    weights: Sequence[float],
    residual_burden_threshold_ml: float = DEFAULT_LOW_RESIDUAL_BURDEN_ML,
) -> dict[str, object]:
    if len(particle_rows) != len(weights):
        raise ValueError("particle_rows and weights must have the same length")
    medians: list[float] = []
    lower80: list[float] = []
    upper80: list[float] = []
    means: list[float] = []

    for day_index, _day in enumerate(output_days):
        values = [
            float(row["predicted_volume_ml"][day_index])  # type: ignore[index]
            for row in particle_rows
        ]
        means.append(sum(value * weight for value, weight in zip(values, weights)))
        lower80.append(weighted_quantile(values, weights, 0.10))
        medians.append(weighted_quantile(values, weights, 0.50))
        upper80.append(weighted_quantile(values, weights, 0.90))

    final_values = [
        float(row["predicted_volume_ml"][-1])  # type: ignore[index]
        for row in particle_rows
    ]
    probability_low_residual = sum(
        weight
        for value, weight in zip(final_values, weights)
        if value <= residual_burden_threshold_ml
    )

    return {
        "times": [float(day) for day in output_days],
        "mean_volume_ml": means,
        "median_volume_ml": medians,
        "lower80_volume_ml": lower80,
        "upper80_volume_ml": upper80,
        "probability_low_residual_burden": probability_low_residual,
        "residual_burden_threshold_ml": residual_burden_threshold_ml,
    }


def summarize_weighted_parameters(
    parameter_particles: Sequence[Mapping[str, Any]],
    *,
    prior_weights: Sequence[float],
    posterior_weights: Sequence[float],
) -> dict[str, object]:
    keys = sorted({key for particle in parameter_particles for key in _flatten_numeric_params(particle)})
    summary: dict[str, object] = {}
    for key in keys:
        values = [_flatten_numeric_params(particle)[key] for particle in parameter_particles]
        prior_mean = sum(value * weight for value, weight in zip(values, prior_weights))
        posterior_mean = sum(value * weight for value, weight in zip(values, posterior_weights))
        prior_width80 = (
            weighted_quantile(values, prior_weights, 0.90)
            - weighted_quantile(values, prior_weights, 0.10)
        )
        posterior_width80 = (
            weighted_quantile(values, posterior_weights, 0.90)
            - weighted_quantile(values, posterior_weights, 0.10)
        )
        summary[key] = {
            "prior_mean": prior_mean,
            "posterior_mean": posterior_mean,
            "shift": posterior_mean - prior_mean,
            "prior_width80": prior_width80,
            "posterior_width80": posterior_width80,
            "width_ratio": posterior_width80 / prior_width80 if prior_width80 > 0 else 1.0,
        }
    return summary


def weighted_quantile(values: Sequence[float], weights: Sequence[float], quantile: float) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        raise ValueError("values must not be empty")
    if quantile < 0 or quantile > 1:
        raise ValueError("quantile must be in [0, 1]")

    pairs = sorted(
        (
            _required_finite(value, "value"),
            _required_nonnegative(weight, "weight"),
        )
        for value, weight in zip(values, weights)
    )
    total_weight = sum(weight for _value, weight in pairs)
    if total_weight <= 0:
        raise ValueError("weights must have positive total")
    threshold = quantile * total_weight
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def build_uncertainty_summary(
    *,
    observations: Sequence[VolumeObservation],
    skipped_observations: Sequence[VolumeObservation],
    effective_sample_size_fraction: float,
    posterior_trajectory_summary: Mapping[str, object],
    fallback_status: str,
) -> dict[str, object]:
    drivers: list[str] = []
    data_quality = "low"
    if len(observations) < 2:
        drivers.append("only one tumor-volume observation available")
        trajectory_uncertainty = "high"
    else:
        trajectory_uncertainty = "moderate"

    if any(_normalize_label(obs.segmentation_qc) in LOW_QC_LABELS for obs in observations):
        drivers.append("low-quality segmentation widened observation likelihood")
        data_quality = "moderate"
    if any(_normalize_label(obs.segmentation_qc) in MEDIUM_QC_LABELS for obs in observations):
        drivers.append("unknown or medium segmentation QC")
        data_quality = "moderate"
    if skipped_observations:
        drivers.append("failed-QC observations were ignored")
        data_quality = "high"
    if fallback_status != "not_needed":
        drivers.append("low effective sample size")
        trajectory_uncertainty = "high"

    lower = posterior_trajectory_summary.get("lower80_volume_ml", [])
    upper = posterior_trajectory_summary.get("upper80_volume_ml", [])
    if isinstance(lower, list) and isinstance(upper, list) and lower and upper:
        final_width = float(upper[-1]) - float(lower[-1])
        final_median = max(float(posterior_trajectory_summary["median_volume_ml"][-1]), 1.0)  # type: ignore[index]
        if final_width / final_median > 1.0:
            drivers.append("wide final-volume posterior interval")
            trajectory_uncertainty = "high"

    if effective_sample_size_fraction > 0.5 and trajectory_uncertainty != "high":
        parameter_uncertainty = "moderate"
    else:
        parameter_uncertainty = "high"

    return {
        "trajectory_uncertainty": trajectory_uncertainty,
        "parameter_uncertainty": parameter_uncertainty,
        "data_quality_uncertainty": data_quality,
        "effective_sample_size_fraction": effective_sample_size_fraction,
        "top_drivers": _stable_unique(drivers) or ("posterior update used available tumor-volume observations",),
    }


def build_update_explanation(
    *,
    observations: Sequence[VolumeObservation],
    prior_trajectory_summary: Mapping[str, object],
    posterior_trajectory_summary: Mapping[str, object],
    effective_sample_size_fraction: float,
    fallback_status: str,
) -> str:
    final_observation = observations[-1]
    times = list(prior_trajectory_summary["times"])  # type: ignore[index]
    day_index = times.index(final_observation.day)
    prior_median = float(prior_trajectory_summary["median_volume_ml"][day_index])  # type: ignore[index]
    posterior_median = float(posterior_trajectory_summary["median_volume_ml"][day_index])  # type: ignore[index]

    if final_observation.tumor_volume_ml < 0.9 * prior_median:
        direction = "toward particles with stronger tumor shrinkage"
    elif final_observation.tumor_volume_ml > 1.1 * prior_median:
        direction = "toward particles with weaker tumor shrinkage or resistant growth"
    else:
        direction = "toward particles close to the prior-predictive median"

    explanation = (
        f"The posterior shifted {direction} because the day {final_observation.day:g} "
        f"tumor volume was {final_observation.tumor_volume_ml:.3g} mL versus a "
        f"prior median prediction of {prior_median:.3g} mL. The posterior median "
        f"at that time is {posterior_median:.3g} mL."
    )
    if fallback_status != "not_needed":
        explanation += (
            " Effective sample size is low, so this update should be treated as "
            "a contradictory-observation diagnostic until a tempered SMC fallback is available."
        )
    elif effective_sample_size_fraction < 0.25:
        explanation += " Effective sample size is modest, so uncertainty should remain prominent."
    return explanation


def _accepted_observations(
    observations: Sequence[VolumeObservation | Mapping[str, object]],
    *,
    include_failed_qc_observations: bool,
) -> tuple[list[VolumeObservation], list[VolumeObservation], list[str]]:
    accepted: list[VolumeObservation] = []
    skipped: list[VolumeObservation] = []
    warnings: list[str] = []
    for observation in observations:
        obs = _coerce_observation(observation)
        if obs.failed_qc and not include_failed_qc_observations:
            skipped.append(obs)
            warnings.append(
                f"ignored failed-QC volume observation at day {obs.day:g}"
            )
            continue
        accepted.append(obs)
    accepted.sort(key=lambda obs: obs.day)
    skipped.sort(key=lambda obs: obs.day)
    return accepted, skipped, _stable_unique(warnings)


def _coerce_observation(observation: VolumeObservation | Mapping[str, object]) -> VolumeObservation:
    if isinstance(observation, VolumeObservation):
        return observation
    if isinstance(observation, Mapping):
        return VolumeObservation.from_mapping(observation)
    raise ValueError("observations must be VolumeObservation objects or mappings")


def _particle_log_prior_weight(particle: Mapping[str, object], n_particles: int) -> float:
    if particle.get("log_prior_weight") not in (None, ""):
        return _required_finite(particle["log_prior_weight"], "log_prior_weight")
    if particle.get("prior_weight") not in (None, ""):
        return math.log(_required_positive(particle["prior_weight"], "prior_weight"))
    if particle.get("weight") not in (None, ""):
        return math.log(_required_positive(particle["weight"], "weight"))
    return -math.log(n_particles)


def _flatten_numeric_params(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, item in value.items():
        if key in {"weight", "prior_weight", "log_prior_weight"}:
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            output.update(_flatten_numeric_params(item, name))
        elif isinstance(item, bool):
            continue
        else:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                output[name] = number
    return output


def _sorted_unique(values: Iterable[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return tuple(output)


def _normalize_label(value: object) -> str:
    return " ".join(str(value).lower().replace("-", "_").split())


def _optional_positive(value: object) -> float | None:
    if value in (None, ""):
        return None
    return _required_positive(value, "noise_fraction")


def _required_positive(value: object, name: str) -> float:
    number = _required_finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _required_nonnegative(value: object, name: str) -> float:
    number = _required_finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _required_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or value in (None, ""):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
