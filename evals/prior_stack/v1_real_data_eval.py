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
BASELINE_IN_SCOPE = tuple(f"{name}_in_scope" for name in BASELINES)
LAYERS = ("layer2_population", "layer3_pathology", "layer4_mri_qc")

# Diagnostic V1 fix: active_treatment_sensitivity was previously applied
# as a per-day kill rate. That made prior-predictive trajectories collapse
# to near-zero volume over normal treatment horizons. Treat it as a
# cycle-scale effect and convert to an effective daily pressure.
TREATMENT_EFFECT_TIMESCALE_DAYS = 12.0


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
        debug_by_layer = {}
        audit_by_layer = {}
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
        debug_by_layer["layer2_population"] = _debug_summary(case, samples2)

        prior3 = apply_pathology_biomarker_rules(prior2, case["context"])
        audit_by_layer["layer3_pathology"] = _layer3_audit_summary(case, prior3)
        warnings += list(prior3.warnings)
        samples3 = sample_pathology_biomarker_prior(
            prior3,
            n_samples=n_samples,
            seed=seed + 1000 + index,
        ).samples
        predictions["layer3_pathology"] = _interval(case, samples3)
        debug_by_layer["layer3_pathology"] = _debug_summary(case, samples3)

        prior4 = apply_mri_feature_rules(prior3, case["context"])
        audit_by_layer["layer4_mri_qc"] = _layer4_audit_summary(case, prior4)
        warnings += list(prior4.warnings)
        samples4 = sample_mri_feature_prior(
            prior4,
            n_samples=n_samples,
            seed=seed + 2000 + index,
        ).samples
        samples4, layer4_early_rule = _apply_layer4_early_response_rules(case, samples4)
        if layer4_early_rule:
            audit_by_layer.setdefault("layer4_mri_qc", {}).setdefault("rules", []).append(layer4_early_rule)
        predictions["layer4_mri_qc"] = _interval(case, samples4)
        debug_by_layer["layer4_mri_qc"] = _debug_summary(case, samples4)

        rows.append(_row(case, predictions, warnings, debug_by_layer, audit_by_layer))

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
        "## Layer debug",
        "",
        "Shows the first in-scope cases. `tx/day` is the effective treatment pressure after scaling.",
        "",
        _layer_debug_table(result["case_predictions"]),
        "",
        "## Layer 3 pathology/biomarker cohort summary",
        "",
        "Audit-only: counts pathology/biomarker availability and Layer 3 rule activity across all in-scope cases.",
        "",
        _layer3_audit_cohort_summary_table(result["case_predictions"]),
        "",
        "## Layer 3 pathology/biomarker audit",
        "",
        "Audit-only: shows pathology/biomarker field availability and Layer 3 rules fired. This section should not change metrics.",
        "",
        _layer3_audit_table(result["case_predictions"]),
        "",
        "## Layer 4 MRI/QC cohort summary",
        "",
        "Audit-only: counts early MRI availability, early-response categories, QC fields, and Layer 4 rule activity across all in-scope cases.",
        "",
        _layer4_audit_cohort_summary_table(result["case_predictions"]),
        "",
        "## Layer 4 early-response outcome diagnostics",
        "",
        "Compares final outcomes and model errors by early MRI response group.",
        "",
        _layer4_early_response_outcome_table(result["case_predictions"]),
        "",
        "## Layer 4 MRI/QC audit",
        "",
        "Shows early MRI response, QC field availability, and active Layer 4 rules fired.",
        "",
        _layer4_audit_table(result["case_predictions"]),
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

    case_id = str(row.get("case_id") or row.get("patient_id") or "").strip()
    if not case_id:
        raise ValueError("cases require case_id or patient_id")
    if baseline_volume <= 0 or final_volume <= 0 or final_day <= baseline_day:
        raise ValueError("cases require positive volumes and final_day after baseline_day")
    if (early_day is None) != (early_volume is None):
        raise ValueError("early_day and early_volume_ml must be provided together")
    if early_day is not None and early_volume is not None:
        if early_volume <= 0 or not (baseline_day < early_day < final_day):
            raise ValueError(
                "early follow-up must have positive volume and fall between baseline and final day"
            )

    context.setdefault("volume_ml", baseline_volume)
    return {
        "case_id": case_id,
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



def _effective_treatment_per_day(sample: Mapping[str, float]) -> float:
    return (
        sample["active_treatment_sensitivity"]
        * (1.0 - sample["resistant_fraction"])
        / TREATMENT_EFFECT_TIMESCALE_DAYS
    )


def _days(case: Mapping[str, object]) -> float:
    return float(case["final_day"]) - float(case["baseline_day"])


def _net_log_change(case: Mapping[str, object], sample: Mapping[str, float]) -> float:
    return (
        sample["growth_rate_per_day"] - _effective_treatment_per_day(sample)
    ) * _days(case)


def _predict(case: Mapping[str, object], sample: Mapping[str, float]) -> float:
    log_change = _net_log_change(case, sample)
    return max(
        float(case["baseline_volume_ml"])
        * math.exp(min(max(log_change, -30.0), 30.0)),
        1e-6,
    )


def _debug_summary(
    case: Mapping[str, object],
    samples: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    predictions = sorted(_predict(case, sample) for sample in samples)
    growth = sorted(sample["growth_rate_per_day"] for sample in samples)
    sensitivity = sorted(sample["active_treatment_sensitivity"] for sample in samples)
    resistant = sorted(sample["resistant_fraction"] for sample in samples)
    treatment_per_day = sorted(_effective_treatment_per_day(sample) for sample in samples)
    net_log_change = sorted(_net_log_change(case, sample) for sample in samples)

    return {
        "days": _days(case),
        "baseline_volume_ml": float(case["baseline_volume_ml"]),
        "observed_final_volume_ml": float(case["final_volume_ml"]),
        "growth_rate_per_day_p50": _q(growth, 0.5),
        "active_treatment_sensitivity_p50": _q(sensitivity, 0.5),
        "resistant_fraction_p50": _q(resistant, 0.5),
        "effective_treatment_per_day_p50": _q(treatment_per_day, 0.5),
        "net_log_change_p50": _q(net_log_change, 0.5),
        "predicted_volume_ml_p10": _q(predictions, 0.1),
        "predicted_volume_ml_p50": _q(predictions, 0.5),
        "predicted_volume_ml_p90": _q(predictions, 0.9),
    }


def _row(
    case: Mapping[str, object],
    predictions: Mapping[str, object],
    warnings: Sequence[str],
    debug_by_layer: Mapping[str, object] | None = None,
    audit_by_layer: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "observed_final_volume_ml": case["final_volume_ml"],
        "predictions": dict(predictions),
        "layer_debug": dict(debug_by_layer or {}),
        "layer_audit": dict(audit_by_layer or {}),
        "warnings": list(warnings),
    }


def _all_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    names = sorted({name for row in rows for name in row["predictions"]})
    output = {name: _metrics(rows, name) for name in names}

    # Fix 0: add apples-to-apples baseline metrics on exactly the same
    # V1-A in-scope rows used by Layer 2/3/4.
    in_scope_rows = [
        row for row in rows if "layer2_population" in row["predictions"]
    ]
    if in_scope_rows:
        for name in BASELINES:
            if name in output:
                output[f"{name}_in_scope"] = _metrics(in_scope_rows, name)

    return output


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
    for name in (*BASELINES, *BASELINE_IN_SCOPE, *LAYERS):
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




_LAYER3_AUDIT_FIELD_GROUPS = {
    "ki67": ("ki67_percent", "ki67", "ki_67", "ki67_index", "proliferation_index"),
    "grade": ("grade", "tumor_grade", "nottingham_grade", "histologic_grade"),
    "er": ("er_status", "estrogen_receptor_status", "er"),
    "pr": ("pr_status", "progesterone_receptor_status", "pr"),
    "her2": ("her2_status", "her2", "erbb2_status"),
    "brca_hrd": (
        "brca_status",
        "brca1_status",
        "brca2_status",
        "hrd_status",
        "homologous_recombination_deficiency",
        "hrd_positive",
    ),
}


def _layer3_audit_summary(
    case: Mapping[str, object],
    prior3: object,
) -> dict[str, object]:
    context = case.get("context", {})
    if not isinstance(context, Mapping):
        context = {}

    contribution = {}
    if hasattr(prior3, "layer_contribution"):
        contribution = prior3.layer_contribution()

    rules = contribution.get("rules", []) if isinstance(contribution, Mapping) else []
    warnings = contribution.get("warnings", []) if isinstance(contribution, Mapping) else []
    uncertainty_drivers = (
        contribution.get("uncertainty_drivers", [])
        if isinstance(contribution, Mapping)
        else []
    )

    available = []
    missing = []
    for group_name, fields in _LAYER3_AUDIT_FIELD_GROUPS.items():
        present_fields = [
            field
            for field in fields
            if field in context and context[field] not in (None, "")
        ]
        if present_fields:
            available.append(f"{group_name}:{','.join(present_fields)}")
        else:
            missing.append(group_name)

    return {
        "available_fields": available,
        "missing_field_groups": missing,
        "rules": list(rules),
        "warnings": list(warnings),
        "uncertainty_drivers": list(uncertainty_drivers),
    }




_LAYER4_QC_FIELDS = (
    "mri_qc",
    "qc_status",
    "qc_pass",
    "segmentation_qc",
    "mask_qc",
    "registration_qc",
    "motion_artifact",
    "image_quality",
    "dce_quality",
    "mri_quality",
    "scanner",
    "field_strength",
    "site_id",
    "manufacturer",
    "series_description",
)

_LAYER4_EARLY_FIELD_ALIASES = {
    "baseline_day": ("baseline_day", "t0_day"),
    "early_day": ("early_day", "t1_day", "interim_day", "mid_treatment_day"),
    "final_day": ("final_day", "t2_day"),
    "baseline_volume_ml": ("baseline_volume_ml", "baseline_enhancing_volume_ml"),
    "early_volume_ml": (
        "early_volume_ml",
        "early_enhancing_volume_ml",
        "interim_volume_ml",
        "mid_treatment_volume_ml",
    ),
    "final_volume_ml": ("final_volume_ml", "observed_final_volume_ml"),
}


def _layer4_audit_summary(
    case: Mapping[str, object],
    prior4: object,
) -> dict[str, object]:
    context = case.get("context", {})
    if not isinstance(context, Mapping):
        context = {}

    contribution = {}
    if hasattr(prior4, "layer_contribution"):
        contribution = prior4.layer_contribution()

    rules = contribution.get("rules", []) if isinstance(contribution, Mapping) else []
    warnings = contribution.get("warnings", []) if isinstance(contribution, Mapping) else []
    uncertainty_drivers = (
        contribution.get("uncertainty_drivers", [])
        if isinstance(contribution, Mapping)
        else []
    )

    values = {
        name: _layer4_first_present(case, context, aliases)
        for name, aliases in _LAYER4_EARLY_FIELD_ALIASES.items()
    }

    baseline_day = _layer4_float_or_none(values["baseline_day"])
    early_day = _layer4_float_or_none(values["early_day"])
    final_day = _layer4_float_or_none(values["final_day"])
    baseline_volume = _layer4_float_or_none(values["baseline_volume_ml"])
    early_volume = _layer4_float_or_none(values["early_volume_ml"])
    final_volume = _layer4_float_or_none(values["final_volume_ml"])

    derived = _layer4_early_response_features(
        baseline_day=baseline_day,
        early_day=early_day,
        final_day=final_day,
        baseline_volume=baseline_volume,
        early_volume=early_volume,
    )

    qc_fields = [
        field
        for field in _LAYER4_QC_FIELDS
        if field in context and context[field] not in (None, "")
    ]

    return {
        "baseline_day": baseline_day,
        "early_day": early_day,
        "final_day": final_day,
        "baseline_volume_ml": baseline_volume,
        "early_volume_ml": early_volume,
        "final_volume_ml": final_volume,
        "early_interval_days": derived["early_interval_days"],
        "early_ratio": derived["early_ratio"],
        "early_log_slope_per_day": derived["early_log_slope_per_day"],
        "early_validity": derived["validity"],
        "qc_fields": qc_fields,
        "rules": list(rules),
        "warnings": list(warnings),
        "uncertainty_drivers": list(uncertainty_drivers),
    }


def _layer4_first_present(
    case: Mapping[str, object],
    context: Mapping[str, object],
    aliases: Sequence[str],
) -> object | None:
    for name in aliases:
        if name in case and case[name] not in (None, ""):
            return case[name]
        if name in context and context[name] not in (None, ""):
            return context[name]
    return None


def _layer4_float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _layer4_early_response_features(
    *,
    baseline_day: float | None,
    early_day: float | None,
    final_day: float | None,
    baseline_volume: float | None,
    early_volume: float | None,
) -> dict[str, object]:
    validity = []

    if baseline_day is None:
        validity.append("missing_baseline_day")
    if early_day is None:
        validity.append("missing_early_day")
    if final_day is None:
        validity.append("missing_final_day")
    if baseline_volume is None:
        validity.append("missing_baseline_volume")
    if early_volume is None:
        validity.append("missing_early_volume")

    early_interval = None
    early_ratio = None
    early_log_slope = None

    if baseline_day is not None and early_day is not None:
        early_interval = early_day - baseline_day
        if early_interval <= 0:
            validity.append("early_not_after_baseline")
        elif early_interval < 14:
            validity.append("early_interval_lt_14d")

    if early_day is not None and final_day is not None and early_day >= final_day:
        validity.append("early_not_before_final")

    if baseline_volume is not None:
        if baseline_volume <= 0:
            validity.append("baseline_volume_nonpositive")
        elif baseline_volume < 0.05:
            validity.append("tiny_baseline_volume_lt_0.05ml")
        elif baseline_volume < 0.10:
            validity.append("small_baseline_volume_lt_0.10ml")

    if early_volume is not None and early_volume <= 0:
        validity.append("early_volume_nonpositive")

    if (
        baseline_volume is not None
        and early_volume is not None
        and baseline_volume > 0
        and early_volume > 0
    ):
        early_ratio = early_volume / baseline_volume
        if early_ratio < 0.05:
            validity.append("early_ratio_lt_0.05")
        elif early_ratio > 20:
            validity.append("early_ratio_gt_20")

        if early_interval is not None and early_interval > 0:
            early_log_slope = math.log(early_ratio) / early_interval

    usable = (
        baseline_day is not None
        and early_day is not None
        and final_day is not None
        and baseline_volume is not None
        and early_volume is not None
        and early_interval is not None
        and early_interval >= 14
        and early_day < final_day
        and baseline_volume > 0
        and early_volume > 0
        and early_ratio is not None
        and 0.05 <= early_ratio <= 20
    )

    if usable:
        validity.append("usable_early_response")
    elif not validity:
        validity.append("unusable_early_response")

    return {
        "early_interval_days": early_interval,
        "early_ratio": early_ratio,
        "early_log_slope_per_day": early_log_slope,
        "validity": validity,
    }




_LAYER4_EARLY_RESPONSE_EFFECTS = {
    "major_shrinkage_ratio_lt_0.50": {
        "growth_multiplier": 0.97,
        "sensitivity_multiplier": 1.08,
        "resistant_shift": -0.020,
    },
    "mild_shrinkage_0.50_to_0.85": {
        "growth_multiplier": 0.99,
        "sensitivity_multiplier": 1.04,
        "resistant_shift": -0.010,
    },
    "mild_growth_1.15_to_1.50": {
        "growth_multiplier": 1.015,
        "sensitivity_multiplier": 0.985,
        "resistant_shift": 0.0075,
    },
    "major_growth_gt_1.50": {
        "growth_multiplier": 1.06,
        "sensitivity_multiplier": 0.93,
        "resistant_shift": 0.030,
    },
}


def _apply_layer4_early_response_rules(
    case: Mapping[str, object],
    samples: Sequence[Mapping[str, float]],
) -> tuple[list[dict[str, float]], dict[str, object] | None]:
    context = case.get("context", {})
    if not isinstance(context, Mapping):
        context = {}

    values = {
        name: _layer4_first_present(case, context, aliases)
        for name, aliases in _LAYER4_EARLY_FIELD_ALIASES.items()
    }

    baseline_day = _layer4_float_or_none(values["baseline_day"])
    early_day = _layer4_float_or_none(values["early_day"])
    final_day = _layer4_float_or_none(values["final_day"])
    baseline_volume = _layer4_float_or_none(values["baseline_volume_ml"])
    early_volume = _layer4_float_or_none(values["early_volume_ml"])

    derived = _layer4_early_response_features(
        baseline_day=baseline_day,
        early_day=early_day,
        final_day=final_day,
        baseline_volume=baseline_volume,
        early_volume=early_volume,
    )

    validity = [str(item) for item in derived.get("validity", [])]
    if "usable_early_response" not in validity:
        return [dict(sample) for sample in samples], None

    early_ratio = _layer4_float_or_none(derived.get("early_ratio"))
    category = _layer4_early_response_category(early_ratio)
    effects = _LAYER4_EARLY_RESPONSE_EFFECTS.get(category)
    if not effects:
        return [dict(sample) for sample in samples], None

    strength = 1.0
    strength_reason = "full_strength"

    # Tiny/small tumors are more sensitive to segmentation noise, so do not
    # let early relative change drive the biology as strongly.
    if baseline_volume is not None and baseline_volume < 0.05:
        strength = 0.25
        strength_reason = "downweighted_tiny_baseline_volume"
    elif baseline_volume is not None and baseline_volume < 0.10:
        strength = 0.50
        strength_reason = "downweighted_small_baseline_volume"

    growth_multiplier = 1.0 + strength * (float(effects["growth_multiplier"]) - 1.0)
    sensitivity_multiplier = 1.0 + strength * (
        float(effects["sensitivity_multiplier"]) - 1.0
    )
    resistant_shift = strength * float(effects["resistant_shift"])

    adjusted = []
    for sample in samples:
        new_sample = dict(sample)

        if "growth_rate_per_day" in new_sample:
            new_sample["growth_rate_per_day"] = max(
                0.0,
                float(new_sample["growth_rate_per_day"]) * growth_multiplier,
            )

        if "active_treatment_sensitivity" in new_sample:
            new_sample["active_treatment_sensitivity"] = max(
                0.0,
                float(new_sample["active_treatment_sensitivity"])
                * sensitivity_multiplier,
            )

        if "resistant_fraction" in new_sample:
            new_sample["resistant_fraction"] = min(
                0.95,
                max(
                    0.0,
                    float(new_sample["resistant_fraction"]) + resistant_shift,
                ),
            )

        adjusted.append(new_sample)

    rule = {
        "rule_id": f"early_response_{category}_v1",
        "evidence_level": "early_mri_response",
        "effects": {
            "growth_multiplier": growth_multiplier,
            "sensitivity_multiplier": sensitivity_multiplier,
            "resistant_shift": resistant_shift,
            "strength": strength,
        },
        "early_ratio": early_ratio,
        "early_log_slope_per_day": derived.get("early_log_slope_per_day"),
        "strength_reason": strength_reason,
        "explanation": (
            "Conservative Layer 4 early-MRI response modifier. The early slope "
            "is not extrapolated directly; it only weakly nudges latent growth, "
            "treatment sensitivity, and resistance parameters."
        ),
    }

    return adjusted, rule


def _layer4_audit_cohort_summary_table(
    rows: Sequence[Mapping[str, object]],
) -> str:
    from collections import Counter

    total = 0
    validity_counts = Counter()
    qc_field_counts = Counter()
    rule_counts = Counter()
    uncertainty_driver_counts = Counter()
    warning_counts = Counter()
    early_category_counts = Counter()
    baseline_size_counts = Counter()
    early_available = 0
    usable_early = 0

    for row in rows:
        audit_by_layer = row.get("layer_audit") or {}
        audit = audit_by_layer.get("layer4_mri_qc")
        if not audit:
            continue

        total += 1

        early_day = audit.get("early_day")
        early_volume = audit.get("early_volume_ml")
        early_ratio = audit.get("early_ratio")
        baseline_volume = audit.get("baseline_volume_ml")
        validity = [str(item) for item in audit.get("early_validity", [])]

        if early_day is not None and early_volume is not None:
            early_available += 1
        if "usable_early_response" in validity:
            usable_early += 1

        for item in validity:
            validity_counts[item] += 1

        for field in audit.get("qc_fields", []):
            qc_field_counts[str(field)] += 1

        for rule in audit.get("rules", []):
            if isinstance(rule, Mapping):
                rule_id = str(rule.get("rule_id", "unknown_rule"))
            else:
                rule_id = str(rule)
            rule_counts[rule_id] += 1

        for driver in audit.get("uncertainty_drivers", []):
            uncertainty_driver_counts[str(driver)] += 1

        for warning in audit.get("warnings", []):
            warning_counts[str(warning)] += 1

        early_category_counts[_layer4_early_response_category(early_ratio)] += 1
        baseline_size_counts[_layer4_baseline_size_category(baseline_volume)] += 1

    if total == 0:
        return "No Layer 4 audit rows were available."

    lines = [
        f"- Audited in-scope cases: {total}",
        f"- Early MRI available: {early_available} / {total} ({_fmt(early_available / total)})",
        f"- Usable early response: {usable_early} / {total} ({_fmt(usable_early / total)})",
        (
            "- Layer 4 active rule signal: "
            + ("present" if rule_counts else "absent; Layer 4 is currently near-neutral relative to Layer 3.")
        ),
        "",
        "### Early-response categories",
        "",
        "| Category | n | % |",
        "| --- | ---: | ---: |",
    ]

    for category in (
        "missing",
        "major_shrinkage_ratio_lt_0.50",
        "mild_shrinkage_0.50_to_0.85",
        "stable_0.85_to_1.15",
        "mild_growth_1.15_to_1.50",
        "major_growth_gt_1.50",
    ):
        count = early_category_counts[category]
        lines.append(f"| {category} | {count} | {_fmt(count / total)} |")

    lines.extend(
        [
            "",
            "### Baseline-volume categories",
            "",
            "| Category | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    for category in (
        "missing",
        "tiny_lt_0.05ml",
        "small_0.05_to_0.10ml",
        "low_0.10_to_0.50ml",
        "medium_0.50_to_2.00ml",
        "large_ge_2.00ml",
    ):
        count = baseline_size_counts[category]
        lines.append(f"| {category} | {count} | {_fmt(count / total)} |")

    lines.extend(
        [
            "",
            "### Early-response validity flags",
            "",
            "| Flag | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    if validity_counts:
        for flag, count in validity_counts.most_common():
            lines.append(f"| {flag} | {count} | {_fmt(count / total)} |")
    else:
        lines.append("| - | 0 | 0 |")

    lines.extend(
        [
            "",
            "### MRI/QC field availability",
            "",
            "| Field | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    if qc_field_counts:
        for field, count in qc_field_counts.most_common():
            lines.append(f"| {field} | {count} | {_fmt(count / total)} |")
    else:
        lines.append("| - | 0 | 0 |")

    lines.extend(
        [
            "",
            "### Layer 4 rule frequencies",
            "",
            "| Rule | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    if rule_counts:
        for rule_id, count in rule_counts.most_common():
            lines.append(f"| {rule_id} | {count} | {_fmt(count / total)} |")
    else:
        lines.append("| - | 0 | 0 |")

    lines.extend(
        [
            "",
            "### Layer 4 uncertainty-driver frequencies",
            "",
            "| Uncertainty driver | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    if uncertainty_driver_counts:
        for driver, count in uncertainty_driver_counts.most_common():
            lines.append(f"| {driver} | {count} | {_fmt(count / total)} |")
    else:
        lines.append("| - | 0 | 0 |")

    if warning_counts:
        lines.extend(
            [
                "",
                "### Layer 4 warning frequencies",
                "",
                "| Warning | n | % |",
                "| --- | ---: | ---: |",
            ]
        )
        for warning, count in warning_counts.most_common():
            lines.append(f"| {warning} | {count} | {_fmt(count / total)} |")

    return "\n".join(lines)


def _layer4_early_response_category(early_ratio: object) -> str:
    value = _layer4_float_or_none(early_ratio)
    if value is None:
        return "missing"
    if value < 0.50:
        return "major_shrinkage_ratio_lt_0.50"
    if value < 0.85:
        return "mild_shrinkage_0.50_to_0.85"
    if value <= 1.15:
        return "stable_0.85_to_1.15"
    if value <= 1.50:
        return "mild_growth_1.15_to_1.50"
    return "major_growth_gt_1.50"


def _layer4_baseline_size_category(baseline_volume: object) -> str:
    value = _layer4_float_or_none(baseline_volume)
    if value is None:
        return "missing"
    if value < 0.05:
        return "tiny_lt_0.05ml"
    if value < 0.10:
        return "small_0.05_to_0.10ml"
    if value < 0.50:
        return "low_0.10_to_0.50ml"
    if value < 2.00:
        return "medium_0.50_to_2.00ml"
    return "large_ge_2.00ml"



def _layer4_early_response_outcome_table(
    rows: Sequence[Mapping[str, object]],
) -> str:
    groups: dict[str, list[Mapping[str, object]]] = {}

    for row in rows:
        audit_by_layer = row.get("layer_audit") or {}
        audit = audit_by_layer.get("layer4_mri_qc")
        if not audit:
            continue

        category = _layer4_early_response_category(audit.get("early_ratio"))
        groups.setdefault(category, []).append(row)

    ordered = (
        "missing",
        "major_shrinkage_ratio_lt_0.50",
        "mild_shrinkage_0.50_to_0.85",
        "stable_0.85_to_1.15",
        "mild_growth_1.15_to_1.50",
        "major_growth_gt_1.50",
    )

    lines = [
        "| Early-response group | n | median early ratio | median final/baseline | no-change MAE | Layer 3 MAE | Layer 4 MAE | Layer 4 80% cov | Layer 4 width 80 ml |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for category in ordered:
        subgroup = groups.get(category, [])
        if not subgroup:
            lines.append(f"| {category} | 0 | - | - | - | - | - | - | - |")
            continue

        early_ratios = []
        final_ratios = []
        for row in subgroup:
            audit = (row.get("layer_audit") or {}).get("layer4_mri_qc") or {}
            early_ratio = _layer4_float_or_none(audit.get("early_ratio"))
            baseline = _layer4_float_or_none(audit.get("baseline_volume_ml"))
            final = _layer4_float_or_none(audit.get("final_volume_ml"))

            if early_ratio is not None:
                early_ratios.append(early_ratio)
            if baseline is not None and final is not None and baseline > 0:
                final_ratios.append(final / baseline)

        metrics_l4 = _metrics_for_name(subgroup, "layer4_mri_qc")
        mae_no_change = _mae_for_point_name(subgroup, "baseline_no_change")
        mae_l3 = _mae_for_point_name(subgroup, "layer3_pathology")
        mae_l4 = _mae_for_point_name(subgroup, "layer4_mri_qc")

        lines.append(
            f"| {category} | {len(subgroup)} | "
            f"{_fmt(_median(early_ratios))} | "
            f"{_fmt(_median(final_ratios))} | "
            f"{_fmt(mae_no_change)} | "
            f"{_fmt(mae_l3)} | "
            f"{_fmt(mae_l4)} | "
            f"{_fmt(metrics_l4.get('coverage_80'))} | "
            f"{_fmt(metrics_l4.get('width_80_ml'))} |"
        )

    return "\n".join(lines)


def _mae_for_point_name(
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> float | None:
    errors = []
    for row in rows:
        observed = _layer4_float_or_none(row.get("observed_final_volume_ml"))
        pred = (row.get("predictions") or {}).get(name)
        if observed is None or not isinstance(pred, Mapping):
            continue

        point = _layer4_float_or_none(pred.get("point_ml"))
        if point is None:
            continue
        errors.append(abs(point - observed))

    if not errors:
        return None
    return sum(errors) / len(errors)


def _metrics_for_name(
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> dict[str, float | None]:
    covered_80 = 0
    covered_95 = 0
    widths_80 = []
    n80 = 0
    n95 = 0

    for row in rows:
        observed = _layer4_float_or_none(row.get("observed_final_volume_ml"))
        pred = (row.get("predictions") or {}).get(name)
        if observed is None or not isinstance(pred, Mapping):
            continue

        lo80 = _layer4_float_or_none(pred.get("p10_ml"))
        hi80 = _layer4_float_or_none(pred.get("p90_ml"))
        lo95 = _layer4_float_or_none(pred.get("p025_ml"))
        hi95 = _layer4_float_or_none(pred.get("p975_ml"))

        if lo80 is not None and hi80 is not None:
            n80 += 1
            widths_80.append(hi80 - lo80)
            if lo80 <= observed <= hi80:
                covered_80 += 1

        if lo95 is not None and hi95 is not None:
            n95 += 1
            if lo95 <= observed <= hi95:
                covered_95 += 1

    return {
        "coverage_80": covered_80 / n80 if n80 else None,
        "coverage_95": covered_95 / n95 if n95 else None,
        "width_80_ml": sum(widths_80) / len(widths_80) if widths_80 else None,
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _layer4_audit_table(
    rows: Sequence[Mapping[str, object]],
    *,
    case_limit: int = 40,
) -> str:
    lines = [
        "| Case | days b->e->f | volume b/e/f ml | early ratio | early log slope/day | early validity | QC fields | rules fired | uncertainty drivers | warnings |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]

    emitted = 0
    for row in rows:
        audit_by_layer = row.get("layer_audit") or {}
        audit = audit_by_layer.get("layer4_mri_qc")
        if not audit:
            continue

        emitted += 1
        if emitted > case_limit:
            break

        day_text = (
            f"{_fmt(audit.get('baseline_day'))}"
            f"->{_fmt(audit.get('early_day'))}"
            f"->{_fmt(audit.get('final_day'))}"
        )
        volume_text = (
            f"{_fmt(audit.get('baseline_volume_ml'))}"
            f"/{_fmt(audit.get('early_volume_ml'))}"
            f"/{_fmt(audit.get('final_volume_ml'))}"
        )

        rules = _summarize_layer4_rules(audit.get("rules", []))
        qc_fields = ", ".join(audit.get("qc_fields", [])) or "-"
        validity = ", ".join(audit.get("early_validity", [])) or "-"
        uncertainty = ", ".join(audit.get("uncertainty_drivers", [])) or "-"
        warnings = "; ".join(audit.get("warnings", [])) or "-"

        lines.append(
            f"| {row['case_id']} | {day_text} | {volume_text} | "
            f"{_fmt(audit.get('early_ratio'))} | "
            f"{_fmt(audit.get('early_log_slope_per_day'))} | "
            f"{validity} | {qc_fields} | {rules} | {uncertainty} | {warnings} |"
        )

    if len(lines) == 2:
        return "No Layer 4 audit rows were available."

    return "\n".join(lines)


def _summarize_layer4_rules(rules: Sequence[Mapping[str, object]]) -> str:
    if not rules:
        return "-"

    chunks = []
    for rule in rules[:6]:
        if isinstance(rule, Mapping):
            rule_id = str(rule.get("rule_id", "unknown_rule"))
            effects = rule.get("effects", {})
            if isinstance(effects, Mapping) and effects:
                effect_text = ",".join(
                    f"{key}={_fmt(value)}"
                    for key, value in sorted(effects.items())
                )
                chunks.append(f"{rule_id} ({effect_text})")
            else:
                chunks.append(rule_id)
        else:
            chunks.append(str(rule))

    if len(rules) > 6:
        chunks.append(f"+{len(rules) - 6} more")

    return "<br>".join(chunks)


def _layer3_audit_cohort_summary_table(
    rows: Sequence[Mapping[str, object]],
) -> str:
    from collections import Counter

    total = 0
    available_counts = Counter()
    missing_counts = Counter()
    rule_counts = Counter()
    uncertainty_driver_counts = Counter()
    warning_counts = Counter()
    pattern_counts = Counter()

    all_groups = tuple(_LAYER3_AUDIT_FIELD_GROUPS.keys())

    for row in rows:
        audit_by_layer = row.get("layer_audit") or {}
        audit = audit_by_layer.get("layer3_pathology")
        if not audit:
            continue

        total += 1

        available_groups = set()
        for item in audit.get("available_fields", []):
            group = str(item).split(":", 1)[0]
            if group:
                available_groups.add(group)
                available_counts[group] += 1

        missing_groups = set(str(item) for item in audit.get("missing_field_groups", []))
        for group in missing_groups:
            missing_counts[group] += 1

        for rule in audit.get("rules", []):
            if isinstance(rule, Mapping):
                rule_id = str(rule.get("rule_id", "unknown_rule"))
            else:
                rule_id = str(rule)
            rule_counts[rule_id] += 1

        for driver in audit.get("uncertainty_drivers", []):
            uncertainty_driver_counts[str(driver)] += 1

        for warning in audit.get("warnings", []):
            warning_counts[str(warning)] += 1

        if available_groups:
            pattern = "+".join(sorted(available_groups))
        else:
            pattern = "none"
        pattern_counts[pattern] += 1

    if total == 0:
        return "No Layer 3 audit rows were available."

    informative_groups = ("ki67", "grade", "brca_hrd")
    informative_available = sum(1 for group in informative_groups if available_counts[group] > 0)

    lines = [
        f"- Audited in-scope cases: {total}",
        (
            "- Layer 3 informative biomarker signal: "
            + (
                "present"
                if informative_available
                else "absent; Layer 3 is expected to remain close to Layer 2."
            )
        ),
        "",
        "### Pathology/biomarker field availability",
        "",
        "| Field group | available n | available % | missing n |",
        "| --- | ---: | ---: | ---: |",
    ]

    for group in all_groups:
        available = available_counts[group]
        missing = missing_counts[group]
        pct = available / total if total else 0.0
        lines.append(f"| {group} | {available} | {_fmt(pct)} | {missing} |")

    lines.extend(
        [
            "",
            "### Layer 3 evidence patterns",
            "",
            "| Available field pattern | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    for pattern, count in pattern_counts.most_common():
        lines.append(f"| {pattern} | {count} | {_fmt(count / total)} |")

    lines.extend(
        [
            "",
            "### Layer 3 rule frequencies",
            "",
            "| Rule | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    if rule_counts:
        for rule_id, count in rule_counts.most_common():
            lines.append(f"| {rule_id} | {count} | {_fmt(count / total)} |")
    else:
        lines.append("| - | 0 | 0 |")

    lines.extend(
        [
            "",
            "### Layer 3 uncertainty-driver frequencies",
            "",
            "| Uncertainty driver | n | % |",
            "| --- | ---: | ---: |",
        ]
    )

    if uncertainty_driver_counts:
        for driver, count in uncertainty_driver_counts.most_common():
            lines.append(f"| {driver} | {count} | {_fmt(count / total)} |")
    else:
        lines.append("| - | 0 | 0 |")

    if warning_counts:
        lines.extend(
            [
                "",
                "### Layer 3 warning frequencies",
                "",
                "| Warning | n | % |",
                "| --- | ---: | ---: |",
            ]
        )
        for warning, count in warning_counts.most_common():
            lines.append(f"| {warning} | {count} | {_fmt(count / total)} |")

    return "\n".join(lines)


def _layer3_audit_table(
    rows: Sequence[Mapping[str, object]],
    *,
    case_limit: int = 40,
) -> str:
    lines = [
        "| Case | available pathology fields | missing groups | rules fired | uncertainty drivers | warnings |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    emitted = 0
    for row in rows:
        audit_by_layer = row.get("layer_audit") or {}
        audit = audit_by_layer.get("layer3_pathology")
        if not audit:
            continue

        emitted += 1
        if emitted > case_limit:
            break

        rules = audit.get("rules", [])
        rule_text = _summarize_layer3_rules(rules)
        available = ", ".join(audit.get("available_fields", [])) or "-"
        missing = ", ".join(audit.get("missing_field_groups", [])) or "-"
        uncertainty = ", ".join(audit.get("uncertainty_drivers", [])) or "-"
        warnings = "; ".join(audit.get("warnings", [])) or "-"

        lines.append(
            f"| {row['case_id']} | {available} | {missing} | "
            f"{rule_text} | {uncertainty} | {warnings} |"
        )

    if len(lines) == 2:
        return "No Layer 3 audit rows were available."

    return "\n".join(lines)


def _summarize_layer3_rules(rules: Sequence[Mapping[str, object]]) -> str:
    if not rules:
        return "-"

    chunks = []
    for rule in rules[:6]:
        rule_id = str(rule.get("rule_id", "unknown_rule"))
        effects = rule.get("effects", {})
        if isinstance(effects, Mapping) and effects:
            effect_text = ",".join(
                f"{key}={_fmt(value)}"
                for key, value in sorted(effects.items())
            )
            chunks.append(f"{rule_id} ({effect_text})")
        else:
            chunks.append(rule_id)

    if len(rules) > 6:
        chunks.append(f"+{len(rules) - 6} more")

    return "<br>".join(chunks)


def _layer_debug_table(
    rows: Sequence[Mapping[str, object]],
    *,
    case_limit: int = 25,
) -> str:
    lines = [
        "| Case | Layer | days | baseline ml | observed ml | pred p50 ml | pred p10-p90 ml | growth/day p50 | sens p50 | resistant p50 | tx/day p50 | net logchg p50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    cases_emitted = 0
    for row in rows:
        debug = row.get("layer_debug") or {}
        if not debug:
            continue

        cases_emitted += 1
        if cases_emitted > case_limit:
            break

        for layer in LAYERS:
            if layer not in debug:
                continue
            item = debug[layer]
            interval = (
                f"{_fmt(item['predicted_volume_ml_p10'])}-"
                f"{_fmt(item['predicted_volume_ml_p90'])}"
            )
            lines.append(
                f"| {row['case_id']} | {layer} | {_fmt(item['days'])} | "
                f"{_fmt(item['baseline_volume_ml'])} | "
                f"{_fmt(item['observed_final_volume_ml'])} | "
                f"{_fmt(item['predicted_volume_ml_p50'])} | {interval} | "
                f"{_fmt(item['growth_rate_per_day_p50'])} | "
                f"{_fmt(item['active_treatment_sensitivity_p50'])} | "
                f"{_fmt(item['resistant_fraction_p50'])} | "
                f"{_fmt(item['effective_treatment_per_day_p50'])} | "
                f"{_fmt(item['net_log_change_p50'])} |"
            )

    if len(lines) == 2:
        return "No in-scope layer-debug rows were available."
    return "\n".join(lines)


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



def _case_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Case | Observed ml | no-change | linear | exponential | Layer 2 | Layer 3 | Layer 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pred = row["predictions"]
        lines.append(
            f"| {row['case_id']} | {_fmt(row['observed_final_volume_ml'])} | "
            f"{_point(pred, 'baseline_no_change')} | "
            f"{_point(pred, 'linear_early')} | "
            f"{_point(pred, 'exponential_early')} | "
            f"{_point(pred, 'layer2_population')} | "
            f"{_point(pred, 'layer3_pathology')} | "
            f"{_point(pred, 'layer4_mri_qc')} |"
        )
    return "\n".join(lines)


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
