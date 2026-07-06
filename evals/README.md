# Evaluation Suite

This directory is reserved for V1 performance, calibration, insight, and safety evaluations.

The goal is to test whether OncoTwin becomes more useful after seeing patient-specific evidence, not merely whether the code runs.

Planned structure:

```text
baselines/
  no_change.py
  linear_response.py
  exponential_response.py
  subtype_average.py

metrics/
  prediction.py
  calibration.py
  uncertainty.py
  posterior_health.py
  personalization_lift.py
  explanation_quality.py
  scenario_stability.py
  safety.py

runners/
  run_v1_baseline_leaderboard.py
  run_v1_layer_ablation.py
  run_v1_forecasting_eval.py
  run_v1_update_value_eval.py
  run_v1_uncertainty_calibration.py
  run_v1_failure_mode_report.py
  run_v1_scenario_lab_eval.py
  run_v1_explanation_audit.py

reports/
  v1_leaderboard.md
  v1_layer_ablation.md
  v1_forecasting.md
  v1_uncertainty_calibration.md
  v1_update_value.md
  v1_posterior_health.md
  v1_failure_modes.md
  v1_scenario_lab_eval.md
  v1_explanation_audit.md
```

Every V1 report should include simple baselines, layer ablations, uncertainty calibration, posterior health, cases helped/harmed, and clear warnings about limits.

<!-- V1_PRIOR_STACK_EVALS_START -->
## V1 prior-stack evals

Run real-data prior-layer performance on a user-provided longitudinal cohort:

```bash
python3 -m evals.prior_stack.v1_real_data_eval --cohort path/to/real_cohort.json --report evals/reports/v1_real_data_prior_layer_eval.md
```

Run the full suite with graceful unavailable handling:

```bash
python3 -m evals.prior_stack.run_v1_eval_suite --cohort path/to/real_cohort.json --report evals/reports/v1_eval_suite.md
```

The real-data eval accepts `.json`, `.jsonl`, and `.csv` cohorts. Records need `case_id`, `subtype`, `treatment_regimen` or `treatment_context`, `baseline_day`, `baseline_volume_ml`, `final_day`, and `final_volume_ml`; optional fields include early follow-up volume, biomarkers, pathology, MRI volume features, and QC fields. JSON records may also use a `measurements` list with `day` and `tumor_volume_ml` or `volume_ml` values.

Demo, synthetic, simulated, toy, and fixture data are rejected by default. Use `--allow-demo-data` only for local smoke checks, not for performance claims.

Implemented categories:

- real-data prior-layer performance and Layer 2/3/4 ablation
- real-data uncertainty interval coverage
- posterior-health stub, unavailable until the Bayesian update runtime exists
- sequential-forecasting stub, unavailable until the Bayesian update runtime exists
- update-value stub, unavailable until posterior comparison runtime exists
- scenario-lab stability stub, unavailable until scenario-lab runtime exists
- explanation-quality audit stub, unavailable until explanation engine and rubric data exist

The suite records unavailable evals in the report instead of crashing. Individual stub runners exit non-zero with a clear missing-component message when invoked directly.

Every V1 report should include simple baselines, layer ablations, uncertainty calibration, posterior health, cases helped/harmed, and clear warnings about limits. Posterior-health, sequential forecasting, update-value, scenario-lab stability, and explanation-quality audits should produce unavailable results until their required runtime components exist.
<!-- V1_PRIOR_STACK_EVALS_END -->

