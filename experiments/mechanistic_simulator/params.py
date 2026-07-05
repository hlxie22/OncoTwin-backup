"""Parameter sampling for volume-only simulator ensembles."""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from .validation import SimulatorInputError, validate_params


def _sample_spec(spec: Any, rng: random.Random, name: str) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, dict):
        raise SimulatorInputError(f"{name} distribution must be a number or object.")

    dist_type = spec.get("type", spec.get("distribution", "fixed"))
    if dist_type == "fixed":
        if "value" not in spec:
            raise SimulatorInputError(f"{name} fixed distribution requires value.")
        return float(spec["value"])
    if dist_type == "uniform":
        lower = float(spec["min"])
        upper = float(spec["max"])
        if upper < lower:
            raise SimulatorInputError(f"{name} uniform max must be >= min.")
        return rng.uniform(lower, upper)
    if dist_type == "log_uniform":
        lower = float(spec["min"])
        upper = float(spec["max"])
        if lower <= 0 or upper < lower:
            raise SimulatorInputError(f"{name} log_uniform bounds must be positive and ordered.")
        return 10 ** rng.uniform(math.log10(lower), math.log10(upper))
    raise SimulatorInputError(f"{name} has unsupported distribution type {dist_type}.")


def _sample_mapping(specs: dict[str, Any], rng: random.Random, name: str) -> dict[str, float]:
    return {
        key: _sample_spec(spec, rng, f"{name}.{key}")
        for key, spec in specs.items()
    }


def sample_volume_params(
    prior_config: dict[str, Any],
    n_particles: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample simulator parameter particles from a bounded prior config."""

    if n_particles <= 0:
        raise SimulatorInputError("n_particles must be positive.")
    rng = random.Random(seed)

    fixed = copy.deepcopy(prior_config.get("fixed", {}))
    distributions = prior_config.get("distributions", {})
    if not isinstance(distributions, dict):
        raise SimulatorInputError("prior_config.distributions must be an object.")

    particles: list[dict[str, Any]] = []
    for index in range(n_particles):
        particle = copy.deepcopy(fixed)
        particle.setdefault("growth_law", "logistic")

        for key in (
            "growth_rate",
            "carrying_capacity_ml",
            "resistant_fraction",
            "resistant_sensitivity_scale",
            "observation_noise_fraction",
        ):
            if key in distributions:
                particle[key] = _sample_spec(distributions[key], rng, key)

        for mapping_name in ("drug_sensitivity", "drug_ec50", "drug_decay"):
            mapping = copy.deepcopy(particle.get(mapping_name, {}))
            if mapping_name in distributions:
                mapping.update(_sample_mapping(distributions[mapping_name], rng, mapping_name))
            particle[mapping_name] = mapping

        particle["particle_id"] = f"p{index:06d}"
        validate_params(particle)
        particles.append(particle)

    return particles
