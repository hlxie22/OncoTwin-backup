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


def write_suite_report(results: Sequence[EvalResult], report: Path) -> None:
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

    lines += ["", "## Metrics", ""]
    for result in results:
        if not result.metrics:
            continue
        lines.append(f"### {result.name}")
        for key, value in result.metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")


def print_result(result: EvalResult) -> None:
    print(f"Status: {result.status}")
    print(result.summary)
    if result.missing_components:
        print("Missing components: " + ", ".join(result.missing_components))
    if result.report_path:
        print("Report: " + result.report_path)
