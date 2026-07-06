"""Transformed-space utilities for V1 prior construction.

The V1 prior stack works in an unconstrained space so that correlated normal
priors can be sampled without repeatedly clipping biologic parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import NormalDist
from typing import Callable, Dict, List, Mapping, Optional, Sequence


LEARNABLE_PARAMETER_NAMES = (
    "growth_rate_per_day",
    "active_treatment_sensitivity",
    "resistant_fraction",
)

TRANSFORMED_PARAMETER_NAMES = (
    "log_growth_rate_per_day",
    "log_active_treatment_sensitivity",
    "logit_resistant_fraction",
)

FRACTION_EPSILON = 1e-9
POSITIVE_EPSILON = 1e-12
PSD_TOLERANCE = 1e-10


@dataclass(frozen=True)
class NormalApproximation:
    """Normal approximation for a median and central interval."""

    mean: float
    std: float


def _require_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def safe_log(value: float, *, epsilon: float = POSITIVE_EPSILON) -> float:
    """Return a finite log for positive parameters, with tiny values floored."""

    numeric = _require_finite(value, "value")
    if numeric < 0:
        raise ValueError("value must be non-negative for safe_log")
    return math.log(max(numeric, epsilon))


def safe_logit(value: float, *, epsilon: float = FRACTION_EPSILON) -> float:
    """Return a finite logit for fractions, clamping only at numeric edges."""

    numeric = _require_finite(value, "value")
    if numeric < 0 or numeric > 1:
        raise ValueError("value must be in [0, 1] for safe_logit")
    clipped = min(max(numeric, epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def safe_sigmoid(value: float) -> float:
    """Return a numerically stable sigmoid."""

    numeric = _require_finite(value, "value")
    if numeric >= 0:
        exp_negative = math.exp(-numeric)
        return 1.0 / (1.0 + exp_negative)
    exp_positive = math.exp(numeric)
    return exp_positive / (1.0 + exp_positive)


def to_transformed(params: Mapping[str, float]) -> Dict[str, float]:
    """Convert learnable V1 parameters to transformed space."""

    missing = [name for name in LEARNABLE_PARAMETER_NAMES if name not in params]
    if missing:
        raise KeyError(f"Missing learnable parameters: {', '.join(missing)}")

    return {
        "log_growth_rate_per_day": safe_log(params["growth_rate_per_day"]),
        "log_active_treatment_sensitivity": safe_log(
            params["active_treatment_sensitivity"]
        ),
        "logit_resistant_fraction": safe_logit(params["resistant_fraction"]),
    }


def from_transformed(z: Mapping[str, float]) -> Dict[str, float]:
    """Convert transformed V1 parameters back to simulator-facing values."""

    missing = [name for name in TRANSFORMED_PARAMETER_NAMES if name not in z]
    if missing:
        raise KeyError(f"Missing transformed parameters: {', '.join(missing)}")

    return {
        "growth_rate_per_day": math.exp(
            _require_finite(z["log_growth_rate_per_day"], "log_growth_rate_per_day")
        ),
        "active_treatment_sensitivity": math.exp(
            _require_finite(
                z["log_active_treatment_sensitivity"],
                "log_active_treatment_sensitivity",
            )
        ),
        "resistant_fraction": safe_sigmoid(z["logit_resistant_fraction"]),
    }


def median_interval_to_normal(
    median: float,
    lower: float,
    upper: float,
    *,
    interval_mass: float = 0.80,
    transform: Optional[str] = None,
) -> NormalApproximation:
    """Approximate a central interval as a normal distribution.

    ``transform`` may be ``"log"`` for positive rates or ``"logit"`` for
    fractions. The returned mean is the transformed median, and the standard
    deviation is inferred from the transformed interval width.
    """

    if not 0 < interval_mass < 1:
        raise ValueError("interval_mass must be between 0 and 1")
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    if not lower <= median <= upper:
        raise ValueError("median must fall inside the interval")

    transform_value = _get_transform(transform)
    transformed_median = transform_value(median)
    transformed_lower = transform_value(lower)
    transformed_upper = transform_value(upper)
    if transformed_lower >= transformed_upper:
        raise ValueError("transformed lower must be less than transformed upper")

    tail_mass = (1.0 - interval_mass) / 2.0
    z_score = NormalDist().inv_cdf(1.0 - tail_mass)
    std = (transformed_upper - transformed_lower) / (2.0 * z_score)
    if std <= 0 or not math.isfinite(std):
        raise ValueError("interval produces an invalid normal approximation")

    return NormalApproximation(mean=transformed_median, std=std)


def _get_transform(transform: Optional[str]) -> Callable[[float], float]:
    if transform is None:
        return lambda value: _require_finite(value, "value")
    if transform == "log":
        return safe_log
    if transform == "logit":
        return safe_logit
    raise ValueError(f"Unsupported transform: {transform}")


def validate_covariance(
    covariance: Sequence[Sequence[float]],
    *,
    dimension: Optional[int] = None,
    tolerance: float = PSD_TOLERANCE,
) -> List[List[float]]:
    """Validate and return a finite symmetric positive-semidefinite covariance."""

    matrix = [
        [_require_finite(value, "covariance value") for value in row]
        for row in covariance
    ]
    if not matrix:
        raise ValueError("covariance must not be empty")
    size = len(matrix)
    if dimension is not None and size != dimension:
        raise ValueError(f"covariance must be {dimension}x{dimension}")
    if any(len(row) != size for row in matrix):
        raise ValueError("covariance must be square")

    for row_index in range(size):
        if matrix[row_index][row_index] < -tolerance:
            raise ValueError("covariance variances must be non-negative")
        for column_index in range(row_index + 1, size):
            mismatch = abs(
                matrix[row_index][column_index] - matrix[column_index][row_index]
            )
            if mismatch > tolerance:
                raise ValueError("covariance must be symmetric")

    _cholesky_psd(matrix, tolerance=tolerance)
    return matrix


def sample_correlated_transformed_prior(
    means: Mapping[str, float],
    covariance: Sequence[Sequence[float]],
    *,
    n_samples: int,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Sample correlated normal particles in transformed parameter space."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    parameter_names = list(means.keys())
    if not parameter_names:
        raise ValueError("means must not be empty")
    mean_vector = [_require_finite(means[name], name) for name in parameter_names]
    matrix = validate_covariance(covariance, dimension=len(parameter_names))
    lower = _cholesky_psd(matrix)
    rng = random.Random(seed)

    samples: List[Dict[str, float]] = []
    for _ in range(n_samples):
        standard_normals = [rng.gauss(0.0, 1.0) for _ in parameter_names]
        values = []
        for row_index, mean in enumerate(mean_vector):
            offset = sum(
                lower[row_index][column_index] * standard_normals[column_index]
                for column_index in range(row_index + 1)
            )
            values.append(mean + offset)
        samples.append(dict(zip(parameter_names, values)))
    return samples


def _cholesky_psd(
    matrix: Sequence[Sequence[float]],
    *,
    tolerance: float = PSD_TOLERANCE,
) -> List[List[float]]:
    size = len(matrix)
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for row_index in range(size):
        for column_index in range(row_index + 1):
            residual = matrix[row_index][column_index] - sum(
                lower[row_index][k] * lower[column_index][k]
                for k in range(column_index)
            )
            if row_index == column_index:
                if residual < -tolerance:
                    raise ValueError("covariance must be positive semidefinite")
                lower[row_index][column_index] = math.sqrt(max(residual, 0.0))
            elif lower[column_index][column_index] <= tolerance:
                if abs(residual) > tolerance:
                    raise ValueError("covariance must be positive semidefinite")
                lower[row_index][column_index] = 0.0
            else:
                lower[row_index][column_index] = (
                    residual / lower[column_index][column_index]
                )
    return lower
