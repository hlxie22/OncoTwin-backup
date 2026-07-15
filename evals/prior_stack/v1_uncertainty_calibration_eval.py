"""Real-data uncertainty calibration eval for V1 prior-predictive intervals."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .common import EvalResult, print_result, unavailable
from .v1_real_data_eval import run_real_data_eval


DEFAULT_REPORT_PATH = Path("evals/reports/v1_uncertainty_calibration.md")
DEFAULT_MIN_SUBGROUP_SIZE = 5
OVERCONFIDENT_80_COVERAGE_THRESHOLD = 0.70
OVERCONFIDENT_95_COVERAGE_THRESHOLD = 0.88
FINAL_PRIOR_LAYER_CANDIDATES = ("layer5_ai_residual", "layer4_mri_qc")


def run_uncertainty_calibration_eval(
    cohort: Path | None,
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
    n_samples: int = 2000,
    seed: int = 2026,
    allow_demo_data: bool = False,
    min_subgroup_size: int = DEFAULT_MIN_SUBGROUP_SIZE,
) -> EvalResult:
    if cohort is None:
        return unavailable(
            "uncertainty_calibration",
            ("real_longitudinal_cohort",),
            "No cohort was provided, so real-data interval calibration could not be computed.",
        )
    if min_subgroup_size < 2:
        raise ValueError("min_subgroup_size must be at least 2")

    result = run_real_data_eval(
        cohort,
        n_samples=n_samples,
        seed=seed,
        allow_demo_data=allow_demo_data,
    )
    metrics_by_layer = result["metrics"]
    final_layer = _final_prior_layer(metrics_by_layer)
    metrics = metrics_by_layer.get(final_layer, {})
    layer4_metrics = metrics_by_layer.get("layer4_mri_qc", {})
    subgroup_calibration = _subgroup_calibration(
        result.get("case_predictions", []),
        layer_name=final_layer,
        min_subgroup_size=min_subgroup_size,
    )
    warnings = tuple(
        dict.fromkeys(
            [*result.get("warnings", []), *_calibration_warnings(metrics, final_layer)]
        )
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# V1 uncertainty calibration",
                "",
                f"- Cohort path: `{cohort}`",
                f"- In-scope cases: {result['in_scope_case_count']}",
                f"- Final prior layer: {final_layer}",
                f"- 80% coverage: {metrics.get('coverage_80')}",
                f"- 95% coverage: {metrics.get('coverage_95')}",
                f"- Mean 80% interval width ml: {metrics.get('width_80_ml')}",
                f"- Minimum subgroup size: {min_subgroup_size}",
                "",
                "## Aggregate calibration",
                "",
                _aggregate_calibration_table(metrics, final_layer),
                "",
                "## Subgroup calibration",
                "",
                _subgroup_calibration_table(subgroup_calibration),
                "",
                "## Calibration warnings",
                "",
                _warning_lines(warnings),
                "",
                "This uses prior-predictive intervals; posterior calibration requires the Bayesian update runtime.",
            ]
        ),
        encoding="utf-8",
    )

    return EvalResult(
        name="uncertainty_calibration",
        status="pass",
        summary="Computed prior-predictive interval coverage on real held-out observations.",
        metrics={
            "final_layer": final_layer,
            "final_layer_coverage_80": metrics.get("coverage_80"),
            "final_layer_coverage_95": metrics.get("coverage_95"),
            "final_layer_width_80_ml": metrics.get("width_80_ml"),
            "layer4_coverage_80": layer4_metrics.get("coverage_80"),
            "layer4_coverage_95": layer4_metrics.get("coverage_95"),
            "layer4_width_80_ml": layer4_metrics.get("width_80_ml"),
            "subgroup_calibration": subgroup_calibration,
        },
        warnings=warnings,
        report_path=str(report_path),
    )


def _final_prior_layer(metrics_by_layer: Mapping[str, object]) -> str:
    for layer_name in FINAL_PRIOR_LAYER_CANDIDATES:
        if layer_name in metrics_by_layer:
            return layer_name
    if metrics_by_layer:
        return sorted(metrics_by_layer)[-1]
    raise ValueError("no prior-layer metrics were available for calibration")


def _calibration_warnings(
    metrics: Mapping[str, object],
    layer_name: str,
) -> list[str]:
    warnings = []
    coverage_80 = _optional_float(metrics.get("coverage_80"))
    coverage_95 = _optional_float(metrics.get("coverage_95"))
    if coverage_80 is not None and coverage_80 < OVERCONFIDENT_80_COVERAGE_THRESHOLD:
        warnings.append(
            f"{layer_name} 80% interval coverage is below 70%; prior intervals may be overconfident."
        )
    if coverage_95 is not None and coverage_95 < OVERCONFIDENT_95_COVERAGE_THRESHOLD:
        warnings.append(
            f"{layer_name} 95% interval coverage is below 88%; prior intervals may be overconfident."
        )
    return warnings


def _subgroup_calibration(
    rows: Sequence[Mapping[str, object]],
    *,
    layer_name: str,
    min_subgroup_size: int,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        predictions = row.get("predictions")
        if not isinstance(predictions, Mapping) or layer_name not in predictions:
            continue
        groups = row.get("calibration_groups")
        if not isinstance(groups, Mapping):
            continue
        for dimension, label in groups.items():
            grouped[f"{dimension}={label}"].append(row)

    return {
        key: _coverage_metrics(subgroup, layer_name=layer_name)
        for key, subgroup in sorted(grouped.items())
        if len(subgroup) >= min_subgroup_size
    }


def _coverage_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    layer_name: str,
) -> dict[str, object]:
    intervals_80 = []
    intervals_95 = []
    widths_80 = []

    for row in rows:
        observed = _optional_float(row.get("observed_final_volume_ml"))
        predictions = row.get("predictions")
        if observed is None or not isinstance(predictions, Mapping):
            continue
        prediction = predictions.get(layer_name)
        if not isinstance(prediction, Mapping):
            continue

        lower_80 = _optional_float(prediction.get("lower_80_ml"))
        upper_80 = _optional_float(prediction.get("upper_80_ml"))
        lower_95 = _optional_float(prediction.get("lower_95_ml"))
        upper_95 = _optional_float(prediction.get("upper_95_ml"))
        if lower_80 is not None and upper_80 is not None:
            intervals_80.append(lower_80 <= observed <= upper_80)
            widths_80.append(upper_80 - lower_80)
        if lower_95 is not None and upper_95 is not None:
            intervals_95.append(lower_95 <= observed <= upper_95)

    return {
        "n": len(rows),
        "coverage_80": _mean_bool(intervals_80),
        "coverage_95": _mean_bool(intervals_95),
        "width_80_ml": sum(widths_80) / len(widths_80) if widths_80 else None,
    }


def _aggregate_calibration_table(
    metrics: Mapping[str, object],
    layer_name: str,
) -> str:
    return "\n".join(
        [
            "| Layer | n | 80% coverage | 95% coverage | 80% width ml |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| {layer_name} | {metrics.get('n', '-')} | "
                f"{_fmt(metrics.get('coverage_80'))} | "
                f"{_fmt(metrics.get('coverage_95'))} | "
                f"{_fmt(metrics.get('width_80_ml'))} |"
            ),
        ]
    )


def _subgroup_calibration_table(
    subgroup_calibration: Mapping[str, Mapping[str, object]],
) -> str:
    if not subgroup_calibration:
        return "No subgroup met the minimum size for stable subgroup calibration reporting."

    lines = [
        "| Subgroup | n | 80% coverage | 95% coverage | 80% width ml |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for subgroup, metrics in subgroup_calibration.items():
        lines.append(
            f"| {subgroup} | {metrics.get('n')} | "
            f"{_fmt(metrics.get('coverage_80'))} | "
            f"{_fmt(metrics.get('coverage_95'))} | "
            f"{_fmt(metrics.get('width_80_ml'))} |"
        )
    return "\n".join(lines)


def _warning_lines(warnings: Sequence[str]) -> str:
    if not warnings:
        return "No calibration warnings."
    return "\n".join(f"- {warning}" for warning in warnings)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_bool(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: object) -> str:
    numeric = _optional_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.3g}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run V1 uncertainty calibration on a real cohort."
    )
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-demo-data", action="store_true")
    parser.add_argument(
        "--min-subgroup-size",
        type=int,
        default=DEFAULT_MIN_SUBGROUP_SIZE,
        help="Minimum cases required before reporting a subgroup calibration row.",
    )
    args = parser.parse_args(argv)

    result = run_uncertainty_calibration_eval(
        args.cohort,
        report_path=args.report,
        n_samples=args.n_samples,
        seed=args.seed,
        allow_demo_data=args.allow_demo_data,
        min_subgroup_size=args.min_subgroup_size,
    )
    print_result(result)
    return 0 if result.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
