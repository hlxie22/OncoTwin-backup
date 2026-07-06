"""Explanation-quality audit for V1 layer contribution reports."""
from .common import runtime_stub_result
from .runtime_stub import main_for

NAME = "explanation_quality"
MODULES = ("experiments.explanation_engine",)
SUMMARY = "Explanation-quality audit requires the explanation engine and adjudicated rubric data."


def run_eval(*, strict: bool = False):
    return runtime_stub_result(NAME, MODULES, SUMMARY, strict=strict)


if __name__ == "__main__":
    raise SystemExit(main_for(NAME, MODULES, SUMMARY))
