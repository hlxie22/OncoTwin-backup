"""Shared utilities for V1 prior-stack eval runners."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EvalResult:
    name: str
    status: str
    summary: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    missing_components: Sequence[str] = ()
    report_path: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "pass"


class EvalUnavailable(RuntimeError):
    def __init__(self, name: str, missing: Sequence[str], summary: str):
        self.name = name
        self.missing = tuple(missing)
        self.summary = summary
        super().__init__(summary)

    def result(self) -> EvalResult:
        return EvalResult(
            name=self.name,
            status="unavailable",
            summary=self.summary,
            missing_components=self.missing,
        )


def require_modules(name: str, modules: Sequence[str], summary: str) -> None:
    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == module or module.startswith(exc.name + "."):
                missing.append(module)
            else:
                raise
    if missing:
        raise EvalUnavailable(name, missing, summary)


def unavailable(name: str, missing: Sequence[str], summary: str) -> EvalResult:
    return EvalResult(
        name=name,
        status="unavailable",
        summary=summary,
        missing_components=tuple(missing),
    )


def runtime_stub_result(
    name: str,
    modules: Sequence[str],
    summary: str,
    *,
    strict: bool = False,
) -> EvalResult:
    try:
        require_modules(name, modules, summary)
    except EvalUnavailable as exc:
        if strict:
            raise
        return exc.result()
    return unavailable(
        name,
        ("runner_wiring",),
        "Required modules import, but this eval is not wired to the runtime API yet.",
    )


def write_suite_report(
    results: Sequence[EvalResult],
    report: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V1 prior-stack evaluation suite",
        "",
        "Unavailable evals indicate missing runtime components, not a suite failure.",
        "",
        "| Eval | Status | Summary | Missing components |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        missing = ", ".join(result.missing_components) if result.missing_components else "-"
        lines.append(
            f"| {result.name} | {result.status} | {result.summary} | {missing} |"
        )

    if metadata is not None:
        lines.extend(_suite_metadata_lines(metadata))

    lines += ["", "## Metrics", ""]
    for result in results:
        if not result.metrics:
            continue
        lines.append(f"### {result.name}")
        for key, value in result.metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")


def _suite_metadata_lines(metadata: Mapping[str, object]) -> list[str]:
    lines = [
        "",
        "## V1-D1 cohort evidence",
        "",
        f"- Cohort path: `{metadata.get('cohort_path') or '-'}`",
        f"- Seed: {metadata.get('seed')}",
        f"- Prior-predictive samples: {metadata.get('n_samples')}",
        f"- V1-D1 status: {metadata.get('v1_d1_status')}",
    ]

    curation = metadata.get("cohort_curation")
    if not isinstance(curation, Mapping):
        return lines

    cohort_summary = curation.get("cohort_summary")
    exclusion_report = curation.get("exclusion_report")
    if not isinstance(cohort_summary, Mapping):
        cohort_summary = {}
    if not isinstance(exclusion_report, Mapping):
        exclusion_report = {}

    lines += [
        f"- Cohort summary: `{curation.get('cohort_summary_path') or '-'}` "
        f"({'found' if curation.get('cohort_summary_found') else 'not found'})",
        f"- Exclusions report: `{curation.get('exclusions_path') or '-'}` "
        f"({'found' if curation.get('exclusions_found') else 'not found'})",
    ]

    selected_summary_fields = (
        "total_input_rows",
        "included_rows",
        "excluded_rows",
        "v1a_in_scope_count",
        "tnbc_count",
        "non_tnbc_count",
        "baseline_volume_available_count",
        "final_volume_available_count",
        "early_followup_available_count",
    )
    present_fields = [field for field in selected_summary_fields if field in cohort_summary]
    if present_fields:
        lines += ["", "### Cohort summary fields", ""]
        for field in present_fields:
            lines.append(f"- {field}: {cohort_summary[field]}")

    _append_nested_mapping(
        lines,
        "Excluded reason counts",
        cohort_summary.get("excluded_reason_counts"),
    )
    _append_nested_mapping(
        lines,
        "Biomarker completeness",
        cohort_summary.get("biomarker_completeness"),
    )
    _append_nested_mapping(
        lines,
        "MRI feature completeness",
        cohort_summary.get("mri_feature_completeness"),
    )
    _append_nested_mapping(
        lines,
        "Exclusion report reason counts",
        exclusion_report.get("excluded_reason_counts"),
    )

    source_files = cohort_summary.get("source_files")
    if isinstance(source_files, Sequence) and not isinstance(source_files, (str, bytes)):
        lines += ["", "### Source files", ""]
        for source_file in source_files:
            lines.append(f"- `{source_file}`")

    return lines


def _append_nested_mapping(
    lines: list[str],
    title: str,
    value: object,
) -> None:
    if not isinstance(value, Mapping) or not value:
        return
    lines += ["", f"### {title}", ""]
    for key, item in sorted(value.items()):
        lines.append(f"- {key}: {item}")


def print_result(result: EvalResult) -> None:
    print(f"Status: {result.status}")
    print(result.summary)
    if result.missing_components:
        print("Missing components: " + ", ".join(result.missing_components))
    if result.report_path:
        print("Report: " + result.report_path)
