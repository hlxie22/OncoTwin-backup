"""Posterior-health smoke checks for the V1 Bayesian update layer."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from .common import EvalResult, print_result
from .v1_runtime_smoke_fixtures import (
    posterior_update_fixture,
    write_json,
    write_markdown_report,
)


NAME = "posterior_health"


def run_eval(
    *,
    report_path: Path | None = None,
    analysis_path: Path | None = None,
    strict: bool = False,
) -> EvalResult:
    posterior = posterior_update_fixture()
    ess_fraction = float(posterior["effective_sample_size_fraction"])
    final_median = float(
        posterior["posterior_trajectory_summary"]["median_volume_ml"][-1]  # type: ignore[index]
    )
    fallback_status = str(posterior.get("fallback_status", "unknown"))
    warnings = tuple(str(warning) for warning in posterior.get("warnings", []))
    status = "pass" if math.isfinite(ess_fraction) and ess_fraction > 0 else "fail"
    summary = (
        "Posterior update runtime smoke check completed with "
        f"ESS fraction {ess_fraction:.1%} and fallback_status={fallback_status}."
    )
    analysis = {
        "eval_name": NAME,
        "status": status,
        "posterior_runtime_version": posterior.get("posterior_runtime_version"),
        "n_prior_particles": posterior.get("n_prior_particles"),
        "n_observations": posterior.get("n_observations"),
        "effective_sample_size": posterior.get("effective_sample_size"),
        "effective_sample_size_fraction": ess_fraction,
        "fallback_status": fallback_status,
        "posterior_trajectory_summary": posterior.get("posterior_trajectory_summary"),
        "uncertainty_summary": posterior.get("uncertainty_summary"),
        "warnings": list(warnings),
    }
    written_analysis = write_json(analysis_path, analysis)
    metrics = {
        "effective_sample_size_fraction": ess_fraction,
        "fallback_status": fallback_status,
        "final_median_volume_ml": final_median,
    }
    if written_analysis:
        metrics["analysis_path"] = written_analysis
    written_report = write_markdown_report(
        report_path,
        title="V1 posterior-health smoke eval",
        summary=summary,
        metrics=metrics,
        warnings=warnings,
    )
    return EvalResult(
        name=NAME,
        status=status,
        summary=summary,
        metrics=metrics,
        warnings=warnings,
        report_path=written_report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = run_eval(report_path=args.report, analysis_path=args.analysis, strict=args.strict)
    print_result(result)
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
