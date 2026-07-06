# Legacy V0 Mechanistic Simulator Experiment Harness

This folder contains the disposable V0 volume-only simulator described in `roadmap/MECHANISTIC_SIMULATOR_PLAN.md`.

The implementation is intentionally volume-only. It is for research sanity checks, baseline comparisons, stress testing, and regression testing. It is not clinical use, product-facing treatment guidance, or the main V1 architecture.

## Main entry points

```text
run_v0_single_trajectory.py   Run one parameter set against one fixture case.
run_v0_ensemble.py            Sample a prior, run an ensemble, write JSON and SVG.
run_v0_synthetic_fit.py       Reweight particles against synthetic observations.
run_v0_identifiability.py     Report weakly constrained parameters.
run_v0_stress_tests.py        Exercise validation and numerical guards.
run_v0_recovery_sweep.py      Run seed/noise/particle/ablation/held-out checks.
```

Core modules are pure-stdlib Python so tests and evaluations can run in a minimal environment.

## Current investigation outputs

The main recovery-sweep artifacts are:

```text
outputs/v0_recovery_sweep_report.json
outputs/v0_recovery_sweep_report.md
outputs/v0_large_particle_check_report.json
outputs/v0_large_particle_check_report.md
```

These reports are synthetic feasibility checks. They are not clinical validation.

## V1 relationship

Do not remove this harness yet. V1 should use it as:

```text
V0 generic-prior baseline
synthetic recovery baseline
held-out simple-baseline comparison target
identifiability and posterior-collapse regression fixture
```

New V1 work should live beside this harness, not inside it:

```text
experiments/prior_builder/   layered prior construction and simulator adapters
evals/                       predictive, calibration, personalization, update-value, scenario, and explanation benchmarks
configs/prior/               versioned prior bounds, population priors, and rule configs
```

The goal is to keep V0 available for comparison while avoiding the impression that the V0 volume-ODE harness is the main product architecture.