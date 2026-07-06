"""Scenario-lab stability eval for treatment what-if analyses."""
from .common import runtime_stub_result
from .runtime_stub import main_for

NAME = "scenario_lab_stability"
MODULES = ("experiments.scenario_lab", "experiments.mechanistic_simulator")
SUMMARY = (
    "Scenario-lab stability requires the scenario-lab runtime and simulator "
    "intervention API."
)


def run_eval(*, strict: bool = False):
    return runtime_stub_result(NAME, MODULES, SUMMARY, strict=strict)


if __name__ == "__main__":
    raise SystemExit(main_for(NAME, MODULES, SUMMARY))
