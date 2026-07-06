"""Posterior-health checks for the V1 Bayesian update layer."""
from .common import runtime_stub_result
from .runtime_stub import main_for

NAME = "posterior_health"
MODULES = ("experiments.prior_builder.bayesian_update",)
SUMMARY = (
    "Posterior-health checks require the Bayesian update runtime and posterior "
    "particle diagnostics."
)


def run_eval(*, strict: bool = False):
    return runtime_stub_result(NAME, MODULES, SUMMARY, strict=strict)


if __name__ == "__main__":
    raise SystemExit(main_for(NAME, MODULES, SUMMARY))
