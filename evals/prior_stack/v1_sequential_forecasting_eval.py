"""Sequential forecasting eval for the V1 patient-update stack."""
from .common import runtime_stub_result
from .runtime_stub import main_for

NAME = "sequential_forecasting"
MODULES = ("experiments.prior_builder.bayesian_update",)
SUMMARY = (
    "Sequential forecasting requires the Bayesian update runtime and a "
    "longitudinal forecast splitter."
)


def run_eval(*, strict: bool = False):
    return runtime_stub_result(NAME, MODULES, SUMMARY, strict=strict)


if __name__ == "__main__":
    raise SystemExit(main_for(NAME, MODULES, SUMMARY))
