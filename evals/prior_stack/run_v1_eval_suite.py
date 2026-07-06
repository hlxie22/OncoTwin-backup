"""Run all V1 prior-stack eval categories with graceful unavailable handling."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import EvalResult, unavailable, write_suite_report
from .v1_explanation_quality_eval import run_eval as run_explanation_quality
from .v1_posterior_health_eval import run_eval as run_posterior_health
from .v1_real_data_eval import run_real_data_eval_result
from .v1_scenario_lab_eval import run_eval as run_scenario_lab
from .v1_sequential_forecasting_eval import run_eval as run_sequential_forecasting
from .v1_uncertainty_calibration_eval import run_uncertainty_calibration_eval
from .v1_update_value_eval import run_eval as run_update_value


DEFAULT_REPORT_PATH = Path("evals/reports/v1_eval_suite.md")


def run_suite(
    *,
    cohort_path: Path | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    n_samples: int = 2000,
    seed: int = 2026,
    allow_demo_data: bool = False,
) -> list[EvalResult]:
    results: list[EvalResult] = []

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
    results.append(run_posterior_health())
    results.append(run_sequential_forecasting())
    results.append(run_update_value())
    results.append(run_scenario_lab())
    results.append(run_explanation_quality())

    write_suite_report(results, report_path)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all V1 prior-stack eval categories.")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-demo-data", action="store_true")
    args = parser.parse_args(argv)

    results = run_suite(
        cohort_path=args.cohort,
        report_path=args.report,
        n_samples=args.n_samples,
        seed=args.seed,
        allow_demo_data=args.allow_demo_data,
    )
    print(
        f"Status: pass ({sum(result.available for result in results)}/"
        f"{len(results)} eval categories produced results)"
    )
    print(f"Report: {args.report}")
    for result in results:
        print(f"- {result.name}: {result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
