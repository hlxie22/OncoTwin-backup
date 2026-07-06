"""Update-value eval for benefit from patient-specific evidence."""
from .common import runtime_stub_result
from .runtime_stub import main_for

NAME = "update_value"
MODULES = ("experiments.prior_builder.bayesian_update",)
SUMMARY = "Update-value eval requires posterior update comparisons against population-only priors."


def run_eval(*, strict: bool = False):
    return runtime_stub_result(NAME, MODULES, SUMMARY, strict=strict)


if __name__ == "__main__":
    raise SystemExit(main_for(NAME, MODULES, SUMMARY))
