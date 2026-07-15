"""Scenario-lab stability smoke eval for treatment what-if analyses."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .common import EvalResult, print_result
from .v1_runtime_smoke_fixtures import (
    scenario_lab_fixture,
    write_json,
    write_markdown_report,
)


NAME = "scenario_lab_stability"


def run_eval(
    *,
    report_path: Path | None = None,
    analysis_path: Path | None = None,
    strict: bool = False,
) -> EvalResult:
    scenario_lab = scenario_lab_fixture()
    scenarios = list(scenario_lab.get("scenarios", []))
    ok_scenarios = [scenario for scenario in scenarios if scenario.get("status") == "ok"]
    failed_scenarios = [scenario for scenario in scenarios if scenario.get("status") != "ok"]
    ranked = scenario_lab["comparison_summary"].get("ranked_scenario_ids_by_low_residual_probability", [])  # type: ignore[index]
    status = "pass" if ok_scenarios and failed_scenarios and ranked else "fail"
    summary = (
        "Scenario-lab smoke check ranked modeled scenarios and confirmed that "
        f"{len(failed_scenarios)} unsafe scenario(s) failed closed."
    )
    warnings = tuple(str(warning) for warning in scenario_lab.get("warnings", []))
    analysis = {
        "eval_name": NAME,
        "status": status,
        "scenario_lab_version": scenario_lab.get("scenario_lab_version"),
        "n_scenarios": scenario_lab.get("n_scenarios"),
        "ok_scenario_count": len(ok_scenarios),
        "failed_scenario_count": len(failed_scenarios),
        "comparison_summary": scenario_lab.get("comparison_summary"),
        "scenario_statuses": {
            str(scenario.get("scenario_id")): scenario.get("status")
            for scenario in scenarios
        },
        "warnings": list(warnings),
    }
    written_analysis = write_json(analysis_path, analysis)
    metrics = {
        "ok_scenario_count": len(ok_scenarios),
        "failed_scenario_count": len(failed_scenarios),
        "top_scenario_id": scenario_lab["comparison_summary"].get("top_scenario_id"),  # type: ignore[index]
    }
    if written_analysis:
        metrics["analysis_path"] = written_analysis
    written_report = write_markdown_report(
        report_path,
        title="V1 scenario-lab stability smoke eval",
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
