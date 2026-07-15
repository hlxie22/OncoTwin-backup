"""Run all V1 prior-stack eval categories with graceful unavailable handling."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Sequence

from .common import EvalResult, unavailable, write_suite_report
from .v1_explanation_quality_eval import run_eval as run_explanation_quality
from .v1_posterior_health_eval import run_eval as run_posterior_health
from .v1_real_data_eval import run_real_data_eval_result
from .v1_scenario_lab_eval import run_eval as run_scenario_lab
from .v1_sequential_forecasting_eval import run_eval as run_sequential_forecasting
from .v1_uncertainty_calibration_eval import run_uncertainty_calibration_eval
from .v1_update_value_eval import run_eval as run_update_value


DEFAULT_REPORT_PATH = Path("evals/reports/v1_eval_suite.md")
SUMMARY_SUFFIX = ".summary.json"
EXCLUSIONS_SUFFIX = ".exclusions.jsonl"
ANALYSIS_SUFFIX = "_artifacts"


def run_suite(
    *,
    cohort_path: Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_path: Path | None = None,
    cohort_summary_path: Path | None = None,
    exclusions_path: Path | None = None,
    analysis_dir: Path | None = None,
    n_samples: int = 2000,
    seed: int = 2026,
    allow_demo_data: bool = False,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    resolved_analysis_dir = analysis_dir or report_path.parent / f"{report_path.stem}{ANALYSIS_SUFFIX}"
    cohort_evidence = load_cohort_curation_evidence(
        cohort_path,
        cohort_summary_path=cohort_summary_path,
        exclusions_path=exclusions_path,
    )

    if cohort_path is None:
        results.append(
            unavailable(
                "real_data_prior_layer_performance",
                ("real_longitudinal_cohort",),
                "No cohort was provided, so real-data prior-layer performance was not computed.",
            )
        )
    else:
        results.append(
            run_real_data_eval_result(
                cohort_path,
                report_path=report_path.with_name("v1_real_data_prior_layer_eval.md"),
                n_samples=n_samples,
                seed=seed,
                allow_demo_data=allow_demo_data,
            )
        )

    results.append(
        run_uncertainty_calibration_eval(
            cohort_path,
            report_path=report_path.with_name("v1_uncertainty_calibration.md"),
            n_samples=n_samples,
            seed=seed,
            allow_demo_data=allow_demo_data,
        )
    )
    results.extend(
        [
            run_posterior_health(
                report_path=report_path.with_name("v1_posterior_health.md"),
                analysis_path=resolved_analysis_dir / "v1_posterior_health.analysis.json",
            ),
            run_sequential_forecasting(
                report_path=report_path.with_name("v1_sequential_forecasting.md"),
                analysis_path=resolved_analysis_dir / "v1_sequential_forecasting.analysis.json",
            ),
            run_update_value(
                report_path=report_path.with_name("v1_update_value.md"),
                analysis_path=resolved_analysis_dir / "v1_update_value.analysis.json",
            ),
            run_scenario_lab(
                report_path=report_path.with_name("v1_scenario_lab_stability.md"),
                analysis_path=resolved_analysis_dir / "v1_scenario_lab_stability.analysis.json",
            ),
            run_explanation_quality(
                report_path=report_path.with_name("v1_explanation_quality.md"),
                analysis_path=resolved_analysis_dir / "v1_explanation_quality.analysis.json",
            ),
        ]
    )

    suite_summary = build_suite_summary(
        results,
        cohort_path=cohort_path,
        report_path=report_path,
        analysis_dir=resolved_analysis_dir,
        cohort_evidence=cohort_evidence,
        n_samples=n_samples,
        seed=seed,
        allow_demo_data=allow_demo_data,
    )
    write_suite_report(results, report_path, metadata=suite_summary)
    if summary_path is None:
        summary_path = report_path.with_suffix(SUMMARY_SUFFIX)
    write_suite_summary(suite_summary, summary_path)
    return results


def load_cohort_curation_evidence(
    cohort_path: Path | None,
    *,
    cohort_summary_path: Path | None = None,
    exclusions_path: Path | None = None,
) -> dict[str, object]:
    """Load optional V1-D1 cohort-builder sidecars for suite reporting."""
    inferred_summary, inferred_exclusions = _infer_sidecars(cohort_path)
    summary_path = cohort_summary_path or inferred_summary
    exclusion_report_path = exclusions_path or inferred_exclusions

    evidence: dict[str, object] = {
        "cohort_summary_path": str(summary_path) if summary_path is not None else None,
        "exclusions_path": str(exclusion_report_path)
        if exclusion_report_path is not None
        else None,
        "cohort_summary_found": False,
        "exclusions_found": False,
        "cohort_summary": {},
        "exclusion_report": {},
    }

    if summary_path is not None:
        if summary_path.exists():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"cohort summary must be a JSON object: {summary_path}")
            evidence["cohort_summary_found"] = True
            evidence["cohort_summary"] = dict(payload)
        elif cohort_summary_path is not None:
            raise FileNotFoundError(f"cohort summary not found: {summary_path}")

    if exclusion_report_path is not None:
        if exclusion_report_path.exists():
            evidence["exclusions_found"] = True
            evidence["exclusion_report"] = _summarize_exclusions(exclusion_report_path)
        elif exclusions_path is not None:
            raise FileNotFoundError(f"exclusions report not found: {exclusion_report_path}")

    return evidence


def build_suite_summary(
    results: Sequence[EvalResult],
    *,
    cohort_path: Path | None,
    report_path: Path,
    analysis_dir: Path,
    cohort_evidence: Mapping[str, object],
    n_samples: int,
    seed: int,
    allow_demo_data: bool,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cohort_path": str(cohort_path) if cohort_path is not None else None,
        "report_path": str(report_path),
        "analysis_dir": str(analysis_dir),
        "n_samples": n_samples,
        "seed": seed,
        "allow_demo_data": allow_demo_data,
        "v1_d1_status": _v1_d1_status(results),
        "results": [_result_payload(result) for result in results],
        "cohort_curation": dict(cohort_evidence),
    }


def write_suite_summary(summary: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _v1_d1_status(results: Sequence[EvalResult]) -> str:
    by_name = {result.name: result.status for result in results}
    required = (
        "real_data_prior_layer_performance",
        "uncertainty_calibration",
    )
    if all(by_name.get(name) == "pass" for name in required):
        return "pass"
    if any(by_name.get(name) == "unavailable" for name in required):
        return "unavailable"
    return "fail"


def _result_payload(result: EvalResult) -> dict[str, object]:
    return {
        "name": result.name,
        "status": result.status,
        "summary": result.summary,
        "metrics": dict(result.metrics),
        "warnings": list(result.warnings),
        "missing_components": list(result.missing_components),
        "report_path": result.report_path,
    }


def _infer_sidecars(cohort_path: Path | None) -> tuple[Path | None, Path | None]:
    if cohort_path is None:
        return None, None
    return (
        cohort_path.with_name(f"{cohort_path.stem}{SUMMARY_SUFFIX}"),
        cohort_path.with_name(f"{cohort_path.stem}{EXCLUSIONS_SUFFIX}"),
    )


def _summarize_exclusions(path: Path) -> dict[str, object]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    reasons: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, Mapping):
            reasons["malformed_exclusion_row"] += 1
            continue
        reason_value = (
            row.get("excluded_reason")
            or row.get("exclusion_reason")
            or row.get("reason")
            or row.get("excluded_reasons")
        )
        if isinstance(reason_value, str):
            reasons[reason_value] += 1
        elif isinstance(reason_value, Sequence) and not isinstance(reason_value, (str, bytes)):
            for reason in reason_value:
                reasons[str(reason)] += 1
        else:
            reasons["unknown"] += 1

    return {
        "excluded_rows": len(rows),
        "excluded_reason_counts": dict(sorted(reasons.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all V1 prior-stack eval categories.")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Machine-readable suite summary JSON. Defaults to the report path with .summary.json.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Directory for runtime-layer JSON analysis artifacts. Defaults to <report stem>_artifacts next to the suite report.",
    )
    parser.add_argument(
        "--cohort-summary",
        type=Path,
        help="Optional cohort-builder summary JSON. Defaults to <cohort>.summary.json when present.",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        help="Optional cohort-builder exclusions JSONL. Defaults to <cohort>.exclusions.jsonl when present.",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-demo-data", action="store_true")
    args = parser.parse_args(argv)

    analysis_dir = args.analysis_dir or args.report.parent / f"{args.report.stem}{ANALYSIS_SUFFIX}"
    results = run_suite(
        cohort_path=args.cohort,
        report_path=args.report,
        summary_path=args.summary,
        cohort_summary_path=args.cohort_summary,
        exclusions_path=args.exclusions,
        analysis_dir=analysis_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        allow_demo_data=args.allow_demo_data,
    )
    summary_path = args.summary or args.report.with_suffix(SUMMARY_SUFFIX)
    print(
        f"Status: pass ({sum(result.available for result in results)}/"
        f"{len(results)} eval categories produced results)"
    )
    print(f"Report: {args.report}")
    print(f"Summary: {summary_path}")
    print(f"Analysis directory: {analysis_dir}")
    for result in results:
        print(f"- {result.name}: {result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
