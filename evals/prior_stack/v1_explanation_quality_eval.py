"""Explanation-quality smoke audit for V1 runtime reports."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

from .common import EvalResult, print_result
from .v1_runtime_smoke_fixtures import (
    explanation_fixture,
    write_json,
    write_markdown_report,
)


NAME = "explanation_quality"


def run_eval(
    *,
    report_path: Path | None = None,
    analysis_path: Path | None = None,
    strict: bool = False,
) -> EvalResult:
    explanation = explanation_fixture()
    sections = explanation.get("sections", [])
    key_factors = explanation.get("key_factors", [])
    uncertainty_drivers = explanation.get("uncertainty_drivers", [])
    section_ids = [section.get("section_id") for section in sections if isinstance(section, Mapping)]
    required_sections = {"posterior_update", "scenario_comparison", "prior_context", "safety_and_scope"}
    guardrails_present = bool(explanation.get("not_a_treatment_recommendation")) and bool(
        explanation.get("safety_and_scope_note")
    )
    status = (
        "pass"
        if guardrails_present and required_sections.issubset(set(section_ids)) and key_factors and uncertainty_drivers
        else "fail"
    )
    summary = (
        "Explanation-quality smoke audit confirmed guardrails, prior context, "
        f"{len(key_factors)} key factor(s), and {len(uncertainty_drivers)} uncertainty driver(s)."
    )
    warnings = () if status == "pass" else ("explanation smoke artifact is missing expected structure",)
    analysis = {
        "eval_name": NAME,
        "status": status,
        "explanation_runtime_version": explanation.get("explanation_runtime_version"),
        "audience": explanation.get("audience"),
        "guardrails_present": guardrails_present,
        "section_ids": section_ids,
        "key_factor_count": len(key_factors),
        "uncertainty_driver_count": len(uncertainty_drivers),
        "summary": explanation.get("summary"),
        "warnings": list(warnings),
    }
    written_analysis = write_json(analysis_path, analysis)
    metrics = {
        "guardrails_present": guardrails_present,
        "section_count": len(section_ids),
        "key_factor_count": len(key_factors),
        "uncertainty_driver_count": len(uncertainty_drivers),
    }
    if written_analysis:
        metrics["analysis_path"] = written_analysis
    written_report = write_markdown_report(
        report_path,
        title="V1 explanation-quality smoke audit",
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
