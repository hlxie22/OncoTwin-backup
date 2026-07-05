# Mechanistic Simulator Experiment Harness

This folder contains the disposable v0 simulator described in
`roadmap/MECHANISTIC_SIMULATOR_PLAN.md`.

The implementation is intentionally volume-only. It is for research sanity
checks, not clinical use or product-facing treatment guidance.

## Main entry points

```text
run_v0_single_trajectory.py   Run one parameter set against one fixture case.
run_v0_ensemble.py            Sample a prior, run an ensemble, write JSON and SVG.
run_v0_synthetic_fit.py       Reweight particles against synthetic observations.
run_v0_identifiability.py     Report weakly constrained parameters.
run_v0_stress_tests.py        Exercise validation and numerical guards.
run_v0_recovery_sweep.py      Run seed/noise/particle/ablation/held-out checks.
```

Core modules are pure-stdlib Python so tests and evaluations can run in a
minimal environment.

## Current investigation outputs

The main recovery-sweep artifacts are:

```text
outputs/v0_recovery_sweep_report.json
outputs/v0_recovery_sweep_report.md
outputs/v0_large_particle_check_report.json
outputs/v0_large_particle_check_report.md
```

These reports are synthetic feasibility checks. They are not clinical
validation.
