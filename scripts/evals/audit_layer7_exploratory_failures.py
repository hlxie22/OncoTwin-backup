#!/usr/bin/env python3
"""Diagnose catastrophic Layer 7 exploratory posterior forecasts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence


INPUT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_exploratory_real_data_v0_1.json"
)
OUTPUT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_exploratory_failure_audit_v0_1.json"
)
REPORT = Path(
    "artifacts/prior_builder/layer7/"
    "layer7_exploratory_failure_audit_v0_1.md"
)

ESS_THRESHOLDS = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
)

TOP_CASE_COUNT = 20


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


def prediction_metrics(
    records: Sequence[Mapping[str, object]],
    *,
    selector,
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
        prediction = selector(record)
        observed = finite(
            record["observed_final_volume_ml"],
            "observed final volume",
        )

        point = finite(
            prediction["point_ml"],
            "prediction point",
        )
        lower80 = finite(
            prediction["lower_80_ml"],
            "lower 80",
        )
        upper80 = finite(
            prediction["upper_80_ml"],
            "upper 80",
        )
        lower95 = finite(
            prediction["lower_95_ml"],
            "lower 95",
        )
        upper95 = finite(
            prediction["upper_95_ml"],
            "upper 95",
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
        widths80.append(upper80 - lower80)
        widths95.append(upper95 - lower95)

        covered80 += int(
            lower80 <= observed <= upper80
        )
        covered95 += int(
            lower95 <= observed <= upper95
        )

    n = len(records)
    if n == 0:
        raise ValueError(
            "cannot calculate metrics for zero cases"
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


def posterior_or_prior_for_ess(
    record: Mapping[str, object],
    threshold: float,
) -> Mapping[str, object]:
    ess_fraction = finite(
        record["posterior_health"]["weights"][
            "effective_sample_size_fraction"
        ],
        "ESS fraction",
    )

    if ess_fraction < threshold:
        return record["prior_prediction"]

    return record["posterior_prediction"]


payload = json.loads(
    INPUT.read_text(encoding="utf-8")
)
rows = payload["case_results"]

updated = [
    row
    for row in rows
    if row.get("early_day") is not None
]

if len(updated) != 252:
    raise RuntimeError(
        f"expected 252 updated cases, got {len(updated)}"
    )

case_audit = []

for row in updated:
    observed = finite(
        row["observed_final_volume_ml"],
        "observed final volume",
    )
    baseline = finite(
        row["baseline_volume_ml"],
        "baseline volume",
    )
    early = finite(
        row["early_volume_ml"],
        "early volume",
    )
    early_day = finite(
        row["early_day"],
        "early day",
    )
    prediction_day = finite(
        row["prediction_day"],
        "prediction day",
    )

    prior = row["prior_prediction"]
    posterior = row["posterior_prediction"]
    health = row["posterior_health"]
    weights = health["weights"]

    prior_point = finite(
        prior["point_ml"],
        "prior point",
    )
    posterior_point = finite(
        posterior["point_ml"],
        "posterior point",
    )

    prior_error = prior_point - observed
    posterior_error = (
        posterior_point - observed
    )
    prior_abs_error = abs(prior_error)
    posterior_abs_error = abs(
        posterior_error
    )

    contractions = {
        name: finite(
            health["parameters"][name][
                "contraction_ratio_80"
            ],
            f"{name} contraction",
        )
        for name in (
            "growth_rate_per_day",
            "active_treatment_sensitivity",
            "resistant_fraction",
        )
    }

    case_audit.append(
        {
            "case_id": row["case_id"],
            "status": row["status"],
            "baseline_volume_ml": baseline,
            "early_day": early_day,
            "early_volume_ml": early,
            "early_to_baseline_ratio": (
                early / baseline
            ),
            "early_log_change": math.log(
                max(early, 1e-8)
                / max(baseline, 1e-8)
            ),
            "remaining_horizon_days": (
                prediction_day - early_day
            ),
            "observed_final_volume_ml": observed,
            "prior_point_ml": prior_point,
            "posterior_point_ml": (
                posterior_point
            ),
            "prior_absolute_error_ml": (
                prior_abs_error
            ),
            "posterior_absolute_error_ml": (
                posterior_abs_error
            ),
            "posterior_minus_prior_absolute_error_ml": (
                posterior_abs_error
                - prior_abs_error
            ),
            "prior_squared_error": (
                prior_error * prior_error
            ),
            "posterior_squared_error": (
                posterior_error
                * posterior_error
            ),
            "posterior_minus_prior_squared_error": (
                posterior_error
                * posterior_error
                - prior_error * prior_error
            ),
            "posterior_to_observed_ratio": (
                posterior_point
                / max(observed, 1e-8)
            ),
            "prior_to_observed_ratio": (
                prior_point
                / max(observed, 1e-8)
            ),
            "ess_fraction": finite(
                weights[
                    "effective_sample_size_fraction"
                ],
                "ESS fraction",
            ),
            "effective_sample_size": finite(
                weights[
                    "effective_sample_size"
                ],
                "ESS",
            ),
            "maximum_weight": finite(
                weights["maximum_weight"],
                "maximum weight",
            ),
            "material_particle_fraction": finite(
                weights[
                    "material_particle_fraction"
                ],
                "material particle fraction",
            ),
            "posterior_to_prior_kl": finite(
                weights[
                    "posterior_to_prior_kl"
                ],
                "posterior KL",
            ),
            "log_likelihood_range": finite(
                weights[
                    "log_likelihood_range"
                ],
                "log likelihood range",
            ),
            "parameter_contraction_ratio_80": (
                contractions
            ),
            "boundary_collapse_parameters": list(
                health[
                    "boundary_collapse_parameters"
                ]
            ),
        }
    )

worst_by_harm = sorted(
    case_audit,
    key=lambda row: row[
        "posterior_minus_prior_absolute_error_ml"
    ],
    reverse=True,
)

worst_by_posterior_error = sorted(
    case_audit,
    key=lambda row: row[
        "posterior_absolute_error_ml"
    ],
    reverse=True,
)

worst_by_squared_error = sorted(
    case_audit,
    key=lambda row: row[
        "posterior_squared_error"
    ],
    reverse=True,
)

total_posterior_sse = sum(
    row["posterior_squared_error"]
    for row in case_audit
)
total_prior_sse = sum(
    row["prior_squared_error"]
    for row in case_audit
)
total_positive_harm = sum(
    max(
        row[
            "posterior_minus_prior_absolute_error_ml"
        ],
        0.0,
    )
    for row in case_audit
)

top_contributions = {}
for count in (1, 3, 5, 10):
    top_rows = worst_by_squared_error[:count]
    top_harm_rows = worst_by_harm[:count]

    top_contributions[str(count)] = {
        "posterior_sse_fraction": (
            sum(
                row["posterior_squared_error"]
                for row in top_rows
            )
            / total_posterior_sse
        ),
        "positive_absolute_harm_fraction": (
            sum(
                max(
                    row[
                        "posterior_minus_prior_absolute_error_ml"
                    ],
                    0.0,
                )
                for row in top_harm_rows
            )
            / total_positive_harm
            if total_positive_harm > 0
            else 0.0
        ),
        "case_ids_by_posterior_sse": [
            row["case_id"]
            for row in top_rows
        ],
        "case_ids_by_harm": [
            row["case_id"]
            for row in top_harm_rows
        ],
    }

raw_prior_metrics = prediction_metrics(
    updated,
    selector=lambda row: (
        row["prior_prediction"]
    ),
)
raw_posterior_metrics = prediction_metrics(
    updated,
    selector=lambda row: (
        row["posterior_prediction"]
    ),
)

healthy_rows = [
    row
    for row in updated
    if row["status"] == "updated_healthy"
]
low_ess_rows = [
    row
    for row in updated
    if row["status"] == "updated_low_ess"
]

status_metrics = {
    "updated_healthy": {
        "case_count": len(healthy_rows),
        "prior": prediction_metrics(
            healthy_rows,
            selector=lambda row: (
                row["prior_prediction"]
            ),
        ),
        "posterior": prediction_metrics(
            healthy_rows,
            selector=lambda row: (
                row["posterior_prediction"]
            ),
        ),
    },
    "updated_low_ess": {
        "case_count": len(low_ess_rows),
        "prior": prediction_metrics(
            low_ess_rows,
            selector=lambda row: (
                row["prior_prediction"]
            ),
        ),
        "posterior": prediction_metrics(
            low_ess_rows,
            selector=lambda row: (
                row["posterior_prediction"]
            ),
        ),
    },
}

threshold_sensitivity = []

for threshold in ESS_THRESHOLDS:
    fallback_count = sum(
        finite(
            row["posterior_health"]["weights"][
                "effective_sample_size_fraction"
            ],
            "ESS fraction",
        )
        < threshold
        for row in updated
    )

    metrics = prediction_metrics(
        updated,
        selector=lambda row, threshold=threshold: (
            posterior_or_prior_for_ess(
                row,
                threshold,
            )
        ),
    )

    threshold_sensitivity.append(
        {
            "ess_fraction_threshold": (
                threshold
            ),
            "fallback_count": (
                fallback_count
            ),
            "metrics": metrics,
            "mae_change_vs_raw_posterior": (
                metrics["mae_ml"]
                - raw_posterior_metrics[
                    "mae_ml"
                ]
            ),
            "rmse_change_vs_raw_posterior": (
                metrics["rmse_ml"]
                - raw_posterior_metrics[
                    "rmse_ml"
                ]
            ),
            "log_rmse_change_vs_raw_posterior": (
                metrics["log_volume_rmse"]
                - raw_posterior_metrics[
                    "log_volume_rmse"
                ]
            ),
        }
    )

trimmed_sensitivity = []

for count in (1, 3, 5, 10):
    excluded = {
        row["case_id"]
        for row in worst_by_posterior_error[
            :count
        ]
    }
    retained = [
        row
        for row in updated
        if row["case_id"] not in excluded
    ]

    trimmed_sensitivity.append(
        {
            "excluded_worst_posterior_error_count": (
                count
            ),
            "excluded_case_ids": sorted(
                excluded
            ),
            "retained_count": len(retained),
            "prior": prediction_metrics(
                retained,
                selector=lambda row: (
                    row["prior_prediction"]
                ),
            ),
            "posterior": prediction_metrics(
                retained,
                selector=lambda row: (
                    row[
                        "posterior_prediction"
                    ]
                ),
            ),
        }
    )

low_ess_case_ids = {
    row["case_id"]
    for row in low_ess_rows
}
top_harmed_case_ids = {
    row["case_id"]
    for row in worst_by_harm[:10]
}
top_sse_case_ids = {
    row["case_id"]
    for row in worst_by_squared_error[:10]
}

diagnosis = {
    "low_ess_case_count": len(
        low_ess_rows
    ),
    "low_ess_cases": sorted(
        low_ess_case_ids
    ),
    "low_ess_overlap_top_10_harmed": sorted(
        low_ess_case_ids
        & top_harmed_case_ids
    ),
    "low_ess_overlap_top_10_posterior_sse": (
        sorted(
            low_ess_case_ids
            & top_sse_case_ids
        )
    ),
    "low_ess_posterior_sse_fraction": (
        sum(
            row["posterior_squared_error"]
            for row in case_audit
            if row["case_id"]
            in low_ess_case_ids
        )
        / total_posterior_sse
    ),
    "low_ess_positive_harm_fraction": (
        sum(
            max(
                row[
                    "posterior_minus_prior_absolute_error_ml"
                ],
                0.0,
            )
            for row in case_audit
            if row["case_id"]
            in low_ess_case_ids
        )
        / total_positive_harm
        if total_positive_harm > 0
        else 0.0
    ),
    "catastrophic_weight_collapse_count": (
        sum(
            row["maximum_weight"] >= 0.90
            for row in case_audit
        )
    ),
    "maximum_posterior_point_ml": max(
        row["posterior_point_ml"]
        for row in case_audit
    ),
    "maximum_posterior_to_observed_ratio": (
        max(
            row[
                "posterior_to_observed_ratio"
            ]
            for row in case_audit
        )
    ),
}

result = {
    "analysis_version": (
        "oncotwin_layer7_exploratory_failure_audit_v0_1"
    ),
    "status": (
        "layer7_exploratory_failure_audit_complete"
    ),
    "interpretation": (
        "Diagnostic analysis of the already generated "
        "exploratory Layer 7 artifact. No model, prior, "
        "likelihood, schedule, or threshold was changed."
    ),
    "source_artifact": {
        "path": str(INPUT),
        "sha256": sha256_file(INPUT),
        "analysis_version": payload[
            "analysis_version"
        ],
    },
    "case_count": len(updated),
    "raw_metrics": {
        "prior": raw_prior_metrics,
        "posterior": raw_posterior_metrics,
        "posterior_minus_prior": {
            key: (
                raw_posterior_metrics[key]
                - raw_prior_metrics[key]
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
    "error_totals": {
        "prior_sum_squared_error": (
            total_prior_sse
        ),
        "posterior_sum_squared_error": (
            total_posterior_sse
        ),
        "total_positive_absolute_harm_ml": (
            total_positive_harm
        ),
    },
    "diagnosis": diagnosis,
    "top_contributions": (
        top_contributions
    ),
    "status_metrics": status_metrics,
    "ess_fallback_sensitivity": (
        threshold_sensitivity
    ),
    "trimmed_error_sensitivity": (
        trimmed_sensitivity
    ),
    "worst_cases_by_absolute_harm": (
        worst_by_harm[:TOP_CASE_COUNT]
    ),
    "worst_cases_by_posterior_error": (
        worst_by_posterior_error[
            :TOP_CASE_COUNT
        ]
    ),
    "worst_cases_by_posterior_sse": (
        worst_by_squared_error[
            :TOP_CASE_COUNT
        ]
    ),
    "all_case_diagnostics": case_audit,
    "limitations": [
        (
            "ESS thresholds are exploratory sensitivity "
            "analyses and are not selected or locked."
        ),
        (
            "Outcome-based comparisons in this audit cannot "
            "be used to tune and then confirm on the same "
            "cases."
        ),
        (
            "The underlying evaluation still uses the "
            "canonical cohort-level A/C-T schedule proxy."
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
    "# Layer 7 exploratory failure audit",
    "",
    (
        "**Diagnostic only. No inference policy was "
        "changed.**"
    ),
    "",
    "## Main diagnosis",
    "",
    (
        "- Updated cases: "
        f"{len(updated)}"
    ),
    (
        "- Low-ESS cases: "
        f"{diagnosis['low_ess_case_count']}"
    ),
    (
        "- Cases with maximum particle weight ≥ 0.90: "
        f"{diagnosis['catastrophic_weight_collapse_count']}"
    ),
    (
        "- Fraction of posterior SSE from low-ESS cases: "
        f"{diagnosis['low_ess_posterior_sse_fraction']}"
    ),
    (
        "- Fraction of positive MAE harm from low-ESS cases: "
        f"{diagnosis['low_ess_positive_harm_fraction']}"
    ),
    (
        "- Maximum posterior point: "
        f"{diagnosis['maximum_posterior_point_ml']} mL"
    ),
    (
        "- Maximum posterior/observed ratio: "
        f"{diagnosis['maximum_posterior_to_observed_ratio']}"
    ),
    "",
    "## Raw metrics",
    "",
    "| Metric | Prior | Posterior |",
    "| --- | ---: | ---: |",
]

for key in (
    "mae_ml",
    "rmse_ml",
    "log_volume_rmse",
    "mape",
    "median_absolute_error_ml",
    "coverage_80",
    "coverage_95",
):
    lines.append(
        f"| {key} | "
        f"{raw_prior_metrics[key]} | "
        f"{raw_posterior_metrics[key]} |"
    )

lines.extend(
    [
        "",
        "## Metrics by posterior-health status",
        "",
        "| Status | Cases | Prior MAE | Posterior MAE | Prior RMSE | Posterior RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
)

for status, values in status_metrics.items():
    lines.append(
        f"| {status} | "
        f"{values['case_count']} | "
        f"{values['prior']['mae_ml']} | "
        f"{values['posterior']['mae_ml']} | "
        f"{values['prior']['rmse_ml']} | "
        f"{values['posterior']['rmse_ml']} |"
    )

lines.extend(
    [
        "",
        "## Exploratory ESS fallback sensitivity",
        "",
        "| ESS threshold | Fallback cases | MAE | RMSE | Log RMSE | 80% coverage | 95% coverage |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
)

for row in threshold_sensitivity:
    metrics = row["metrics"]
    lines.append(
        f"| {row['ess_fraction_threshold']} | "
        f"{row['fallback_count']} | "
        f"{metrics['mae_ml']} | "
        f"{metrics['rmse_ml']} | "
        f"{metrics['log_volume_rmse']} | "
        f"{metrics['coverage_80']} | "
        f"{metrics['coverage_95']} |"
    )

lines.extend(
    [
        "",
        "## Worst cases by posterior harm",
        "",
        "| Case | Status | Observed | Prior point | Posterior point | Prior abs. error | Posterior abs. error | Harm | ESS fraction | Max weight |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
)

for row in worst_by_harm[:TOP_CASE_COUNT]:
    lines.append(
        f"| {row['case_id']} | "
        f"{row['status']} | "
        f"{row['observed_final_volume_ml']} | "
        f"{row['prior_point_ml']} | "
        f"{row['posterior_point_ml']} | "
        f"{row['prior_absolute_error_ml']} | "
        f"{row['posterior_absolute_error_ml']} | "
        f"{row['posterior_minus_prior_absolute_error_ml']} | "
        f"{row['ess_fraction']} | "
        f"{row['maximum_weight']} |"
    )

lines.extend(
    [
        "",
        "## Interpretation rule for the next step",
        "",
        (
            "If the catastrophic error is concentrated in "
            "low-ESS cases, the next implementation should "
            "compare an explicit fail-closed posterior policy "
            "with tempered updating. If healthy cases dominate "
            "the catastrophic error, the schedule proxy, "
            "likelihood, or mechanistic prior is misspecified "
            "and must be investigated before adding fallback "
            "logic."
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
print("case_count:", len(updated))
print(
    "raw_prior_metrics:",
    raw_prior_metrics,
)
print(
    "raw_posterior_metrics:",
    raw_posterior_metrics,
)
print("diagnosis:", diagnosis)
print(
    "top_5_harmed_cases:"
)
for row in worst_by_harm[:5]:
    print(
        {
            key: row[key]
            for key in (
                "case_id",
                "status",
                "observed_final_volume_ml",
                "prior_point_ml",
                "posterior_point_ml",
                "posterior_minus_prior_absolute_error_ml",
                "ess_fraction",
                "maximum_weight",
            )
        }
    )

print(
    "status_metrics:",
    status_metrics,
)
print(
    "ess_fallback_sensitivity:"
)
for row in threshold_sensitivity:
    print(
        {
            "threshold": (
                row[
                    "ess_fraction_threshold"
                ]
            ),
            "fallback_count": (
                row["fallback_count"]
            ),
            "mae_ml": (
                row["metrics"]["mae_ml"]
            ),
            "rmse_ml": (
                row["metrics"]["rmse_ml"]
            ),
            "log_volume_rmse": (
                row["metrics"][
                    "log_volume_rmse"
                ]
            ),
            "coverage_80": (
                row["metrics"][
                    "coverage_80"
                ]
            ),
            "coverage_95": (
                row["metrics"][
                    "coverage_95"
                ]
            ),
        }
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
    "LAYER7_FAILURE_AUDIT_STATUS: PASS"
)
