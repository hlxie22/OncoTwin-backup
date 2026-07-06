"""Real-cohort V1 prior-stack layer ablation eval."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from experiments.prior_builder.mri_feature_rules import (
    apply_mri_feature_rules,
    sample_mri_feature_prior,
)
from experiments.prior_builder.parameter_contract import (
    TNBC_CHEMO_CONTRACT_ID,
    resolve_parameter_contract,
)
from experiments.prior_builder.pathology_biomarker_rules import (
    apply_pathology_biomarker_rules,
    sample_pathology_biomarker_prior,
)
from experiments.prior_builder.population_prior import (
    resolve_population_prior,
    sample_population_prior,
)

from .common import EvalResult


DEFAULT_REPORT_PATH = Path("evals/reports/v1_real_data_prior_layer_eval.md")
FAKE_TOKENS = ("demo", "synthetic", "simulated", "toy", "fixture")
BASELINES = ("baseline_no_change", "linear_early", "exponential_early")
LAYERS = ("layer2_population", "layer3_pathology", "layer4_mri_qc")


def load_real_cohort(
    path: Path,
    *,
    allow_demo_data: bool = False,
) -> list[dict[str, object]]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError("cohort contains no cases")
    if not allow_demo_data and any(_fake(path, row) for row in rows):
        raise ValueError(
            "real-data eval refused demo/synthetic/fixture data; "
            "pass --allow-demo-data only for smoke tests"
        )
    return [_case(row) for row in rows]


def run_real_data_eval(
    cohort: Path,
    *,
    n_samples: int = 2000,
    seed: int = 2026,
    allow_demo_data: bool = False,
) -> dict[str, object]:
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")

    cases = load_real_cohort(cohort, allow_demo_data=allow_demo_data)
    rows = []
    skipped = []

    for index, case in enumerate(cases):
        predictions = _baseline_predictions(case)
        contract = resolve_parameter_contract(case["context"])
        warnings = list(contract.warnings)

        if contract.contract_id != TNBC_CHEMO_CONTRACT_ID:
            skipped.append(
                f"{case['case_id']}: out of V1-A scope ({contract.contract_id})"
            )
            rows.append(_row(case, predictions, warnings))
            continue

        prior2 = resolve_population_prior(contract)
        samples2 = sample_population_prior(
            prior2,
            n_samples=n_samples,
            seed=seed + index,
        ).samples
        predictions["layer2_population"] = _interval(case, samples2)

        prior3 = apply_pathology_biomarker_rules(prior2, case["context"])
        warnings += list(prior3.warnings)
        samples3 = sample_pathology_biomarker_prior(
            prior3,
            n_samples=n_samples,
            seed=seed + 1000 + index,
        ).samples
        predictions["layer3_pathology"] = _interval(case, samples3)

        prior4 = apply_mri_feature_rules(prior3, case["context"])
        warnings += list(prior4.warnings)
        samples4 = sample_mri_feature_prior(
            prior4,
            n_samples=n_samples,
            seed=seed + 2000 + index,
        ).samples
        predictions["layer4_mri_qc"] = _interval(case, samples4)

        rows.append(_row(case, predictions, warnings))

    in_scope = sum("layer2_population" in row["predictions"] for row in rows)
    if not in_scope:
        raise ValueError("no in-scope TNBC chemotherapy cases were available for V1 eval")

    warnings = []
    if len(cases) < 20:
        warnings.append(
            "Cohort has fewer than 20 cases; treat metrics as early evidence."
        )

    return {
        "cohort_path": str(cohort),
        "case_count": len(cases),
        "in_scope_case_count": in_scope,
        "metrics": _all_metrics(rows),
        "layer_delta": _deltas(rows),
        "case_predictions": rows,
        "skipped_cases": skipped,
        "warnings": warnings,
    }


def run_real_data_eval_result(
    cohort: Path,
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
    n_samples: int = 2000,
    seed: int = 2026,
    allow_demo_data: bool = False,
) -> EvalResult:
    result = run_real_data_eval(
        cohort,
        n_samples=n_samples,
        seed=seed,
        allow_demo_data=allow_demo_data,
    )
    write_markdown_report(result, report_path)
    layer4 = result["metrics"].get("layer4_mri_qc", {})
    return EvalResult(
        name="real_data_prior_layer_performance",
        status="pass",
        summary=(
            f"Evaluated {result['in_scope_case_count']}/{result['case_count']} "
            "in-scope cases against held-out final volumes."
        ),
        metrics={
            "layer4_mae_ml": layer4.get("mae_ml"),
            "layer4_log_rmse": layer4.get("log_volume_rmse"),
            "layer4_coverage_80": layer4.get("coverage_80"),
        },
        warnings=tuple(result["warnings"]),
        report_path=str(report_path),
    )


def write_markdown_report(result: Mapping[str, object], report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V1 real-data prior-layer evaluation",
        "",
        "Demo/synthetic/fixture data are rejected by default.",
        "",
        f"- Cohort path: `{result['cohort_path']}`",
        f"- Cases: {result['in_scope_case_count']}/{result['case_count']} in scope",
        "",
        "## Leaderboard",
        "",
        _metrics_table(result["metrics"]),
        "",
        "## Layer deltas",
        "",
        _delta_table(result["layer_delta"]),
        "",
        "## Case predictions",
        "",
        _case_table(result["case_predictions"]),
        "",
        "## Limits",
        "",
        (
            "This is prior-predictive layer ablation, not posterior-update "
            "performance. Missing posterior, update-value, scenario, and "
            "explanation evals are represented by separate unavailable suite entries."
        ),
    ]

    if result["skipped_cases"]:
        lines += ["", "## Skipped cases"]
        lines += [f"- {item}" for item in result["skipped_cases"]]
    if result["warnings"]:
        lines += ["", "## Warnings"]
        lines += [f"- {item}" for item in result["warnings"]]

    report.write_text("\n".join(lines), encoding="utf-8")


def _read_rows(path: Path) -> list[Mapping[str, object]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            cases = payload.get("cases")
            return cases if isinstance(cases, list) else [payload]
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("cohort must be .json, .jsonl, or .csv")


def _case(row: Mapping[str, object]) -> dict[str, object]:
    context = dict(row)
    for nested in ("pathology", "mri_features", "biomarkers"):
        if isinstance(row.get(nested), Mapping):
            context.update(row[nested])

    measurements = row.get("measurements")
    if isinstance(measurements, list):
        points = sorted(
            (
                _num(measurement, "day"),
                _num_any(measurement, ("tumor_volume_ml", "volume_ml")),
            )
            for measurement in measurements
            if isinstance(measurement, Mapping)
        )
        if len(points) < 2:
            raise ValueError("measurement-list cohorts require at least two points")
        baseline_day, baseline_volume = points[0]
        final_day, final_volume = points[-1]
        early_day, early_volume = points[1] if len(points) > 2 else (None, None)
    else:
        baseline_day = _num(row, "baseline_day", 0.0)
        baseline_volume = _num_any(
            row,
            ("baseline_volume_ml", "baseline_tumor_volume_ml", "initial_volume_ml"),
        )
        final_day = _num_any(row, ("final_day", "heldout_day", "outcome_day"))
        final_volume = _num_any(
            row,
            ("final_volume_ml", "heldout_volume_ml", "outcome_volume_ml"),
        )
        early_day = _optional(row, ("early_day", "followup_day"))
        early_volume = _optional(row, ("early_volume_ml", "followup_volume_ml"))

    if baseline_volume <= 0 or final_volume <= 0 or final_day <= baseline_day:
        raise ValueError("cases require positive volumes and final_day after baseline_day")
    if (early_day is None) != (early_volume is None):
        raise ValueError("early_day and early_volume_ml must be provided together")

    context.setdefault("volume_ml", baseline_volume)
    return {
        "case_id": str(row.get("case_id") or row.get("patient_id") or "case"),
        "context": context,
        "baseline_day": baseline_day,
        "baseline_volume_ml": baseline_volume,
        "early_day": early_day,
        "early_volume_ml": early_volume,
        "final_day": final_day,
        "final_volume_ml": final_volume,
    }


def _baseline_predictions(case: Mapping[str, object]) -> dict[str, dict[str, float]]:
    baseline = float(case["baseline_volume_ml"])
    if case["early_day"] is None:
        return {name: {"point_ml": baseline} for name in BASELINES}

    elapsed = float(case["early_day"]) - float(case["baseline_day"])
    horizon = float(case["final_day"]) - float(case["baseline_day"])
    early = float(case["early_volume_ml"])

    return {
        "baseline_no_change": {"point_ml": baseline},
        "linear_early": {
            "point_ml": max(baseline + (early - baseline) * horizon / elapsed, 1e-6)
        },
        "exponential_early": {
            "point_ml": max(
                baseline * math.exp(math.log(early / baseline) * horizon / elapsed),
                1e-6,
            )
        },
    }


def _interval(
    case: Mapping[str, object],
    samples: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    values = sorted(_predict(case, sample) for sample in samples)
    return {
        "point_ml": _q(values, 0.5),
        "lower_80_ml": _q(values, 0.1),
        "upper_80_ml": _q(values, 0.9),
        "lower_95_ml": _q(values, 0.025),
        "upper_95_ml": _q(values, 0.975),
    }


def _predict(case: Mapping[str, object], sample: Mapping[str, float]) -> float:
    treatment = sample["active_treatment_sensitivity"] * (
        1.0 - sample["resistant_fraction"]
    )
    days = float(case["final_day"]) - float(case["baseline_day"])
    log_change = (sample["growth_rate_per_day"] - treatment) * days
    return max(
        float(case["baseline_volume_ml"])
        * math.exp(min(max(log_change, -30.0), 30.0)),
        1e-6,
    )


def _row(
    case: Mapping[str, object],
    predictions: Mapping[str, object],
    warnings: Sequence[str],
) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "observed_final_volume_ml": case["final_volume_ml"],
        "predictions": dict(predictions),
        "warnings": list(warnings),
    }


def _all_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    names = sorted({name for row in rows for name in row["predictions"]})
    return {name: _metrics(rows, name) for name in names}


def _metrics(rows: Sequence[Mapping[str, object]], name: str) -> dict[str, object]:
    pairs = [
        (float(row["observed_final_volume_ml"]), row["predictions"][name])
        for row in rows
        if name in row["predictions"]
    ]
    errors = [prediction["point_ml"] - observed for observed, prediction in pairs]
    return {
        "n": len(pairs),
        "mae_ml": sum(abs(error) for error in errors) / len(errors),
        "rmse_ml": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "log_volume_rmse": math.sqrt(
            sum(
                (math.log(prediction["point_ml"]) - math.log(observed)) ** 2
                for observed, prediction in pairs
            )
            / len(pairs)
        ),
        "mape": _q(
            sorted(abs(error) / observed for error, (observed, _) in zip(errors, pairs)),
            0.5,
        ),
        "coverage_80": _coverage(pairs, "lower_80_ml", "upper_80_ml"),
        "coverage_95": _coverage(pairs, "lower_95_ml", "upper_95_ml"),
        "width_80_ml": _width(pairs, "lower_80_ml", "upper_80_ml"),
    }


def _coverage(
    pairs: Sequence[tuple[float, Mapping[str, float]]],
    lower: str,
    upper: str,
) -> float | None:
    intervals = [(observed, pred) for observed, pred in pairs if lower in pred]
    if not intervals:
        return None
    return sum(pred[lower] <= observed <= pred[upper] for observed, pred in intervals) / len(
        intervals
    )


def _width(
    pairs: Sequence[tuple[float, Mapping[str, float]]],
    lower: str,
    upper: str,
) -> float | None:
    widths = [pred[upper] - pred[lower] for _, pred in pairs if lower in pred]
    if not widths:
        return None
    return sum(widths) / len(widths)


def _deltas(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    output = {}
    comparisons = (
        ("layer3_vs_layer2", "layer3_pathology", "layer2_population"),
        ("layer4_vs_layer3", "layer4_mri_qc", "layer3_pathology"),
    )
    for label, new_name, old_name in comparisons:
        deltas = []
        for row in rows:
            predictions = row["predictions"]
            if new_name not in predictions or old_name not in predictions:
                continue
            observed = float(row["observed_final_volume_ml"])
            deltas.append(
                abs(predictions[old_name]["point_ml"] - observed)
                - abs(predictions[new_name]["point_ml"] - observed)
            )
        output[label] = {
            "n": len(deltas),
            "helped": sum(delta > 0 for delta in deltas),
            "harmed": sum(delta < 0 for delta in deltas),
            "mean_mae_delta_ml": sum(deltas) / len(deltas) if deltas else 0.0,
        }
    return output


def _metrics_table(metrics: Mapping[str, Mapping[str, object]]) -> str:
    lines = [
        "| Model | n | MAE ml | RMSE ml | log RMSE | MAPE | 80% cov | 95% cov | 80% width ml |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in (*BASELINES, *LAYERS):
        if name not in metrics:
            continue
        metric = metrics[name]
        lines.append(
            f"| {name} | {metric['n']} | {_fmt(metric['mae_ml'])} | "
            f"{_fmt(metric['rmse_ml'])} | {_fmt(metric['log_volume_rmse'])} | "
            f"{_fmt(metric['mape'])} | {_fmt(metric['coverage_80'])} | "
            f"{_fmt(metric['coverage_95'])} | {_fmt(metric['width_80_ml'])} |"
        )
    return "\n".join(lines)


def _delta_table(deltas: Mapping[str, Mapping[str, object]]) -> str:
    lines = [
        "| Comparison | n | helped | harmed | mean MAE delta ml |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in deltas.items():
        lines.append(
            f"| {name} | {row['n']} | {row['helped']} | {row['harmed']} | "
            f"{_fmt(row['mean_mae_delta_ml'])} |"
        )
    return "\n".join(lines)


def _case_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Case | Observed ml | no-change | linear | exponential | Layer 2 | Layer 3 | Layer 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pred = row["predictions"]
        lines.append(
            f"| {row['case_id']} | {_fmt(row['observed_final_volume_ml'])} | "
            f"{_point(pred, 'baseline_no_change')} | {_point(pred, 'linear_early')} | "
            f"{_point(pred, 'exponential_early')} | {_point(pred, 'layer2_population')} | "
            f"{_point(pred, 'layer3_pathology')} | {_point(pred, 'layer4_mri_qc')} |"
        )
    return "\n".join(lines)


def _point(predictions: Mapping[str, Mapping[str, float]], name: str) -> str:
    if name not in predictions:
        return "-"
    return _fmt(predictions[name]["point_ml"])


def _fake(path: Path, row: Mapping[str, object]) -> bool:
    haystack = str(path) + json.dumps(row, default=str)
    text = haystack.lower()
    return any(token in text for token in FAKE_TOKENS)


def _num(mapping: Mapping[str, object], field: str, default: float | None = None) -> float:
    if field not in mapping or mapping[field] in (None, ""):
        if default is not None:
            return default
        raise ValueError(f"missing required numeric field: {field}")
    if isinstance(mapping[field], bool):
        raise ValueError(f"{field} must be numeric")
    value = float(mapping[field])
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _num_any(mapping: Mapping[str, object], fields: Sequence[str]) -> float:
    for field in fields:
        if field in mapping and mapping[field] not in (None, ""):
            return _num(mapping, field)
    raise ValueError("missing required numeric field: " + " or ".join(fields))


def _optional(mapping: Mapping[str, object], fields: Sequence[str]) -> float | None:
    for field in fields:
        if field in mapping and mapping[field] not in (None, ""):
            return _num(mapping, field)
    return None


def _q(values: Sequence[float], q: float) -> float:
    pos = q * (len(values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4g}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run V1 prior-layer ablations on a real longitudinal cohort."
    )
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-demo-data", action="store_true")
    args = parser.parse_args(argv)

    result = run_real_data_eval_result(
        args.cohort,
        report_path=args.report,
        n_samples=args.n_samples,
        seed=args.seed,
        allow_demo_data=args.allow_demo_data,
    )
    print(f"Status: {result.status}")
    print(result.summary)
    print(f"Report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
