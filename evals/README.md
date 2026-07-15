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

Run the full suite and collect runtime-layer analysis artifacts:

```bash
python3 -m evals.prior_stack.run_v1_eval_suite \
  --cohort path/to/real_cohort.json \
  --report evals/reports/v1_eval_suite.md \
  --summary evals/reports/v1_eval_suite.summary.json \
  --analysis-dir evals/reports/v1_eval_suite_artifacts
```

The suite writes the Markdown report, a machine-readable summary JSON, and one JSON analysis artifact per runtime-layer smoke eval under `--analysis-dir`. When no cohort is supplied, real-data prior-layer performance and uncertainty calibration are reported as unavailable, but runtime-layer smoke evals still run.

The real-data eval accepts `.json`, `.jsonl`, and `.csv` cohorts. Records need `case_id`, `subtype`, `treatment_regimen` or `treatment_context`, `baseline_day`, `baseline_volume_ml`, `final_day`, and `final_volume_ml`; optional fields include early follow-up volume, biomarkers, pathology, MRI volume features, and QC fields. JSON records may also use a `measurements` list with `day` and `tumor_volume_ml` or `volume_ml` values.

Demo, synthetic, simulated, toy, and fixture data are rejected by default. Use `--allow-demo-data` only for local smoke checks, not for performance claims.

Uncertainty calibration reports aggregate final-prior-layer interval coverage and, when a group has enough in-scope cases, subgroup coverage for cohort provenance, early-follow-up availability, MRI/QC status, and baseline-volume bins. Low 80% or 95% coverage is flagged as a calibration warning rather than silently folded into the pass/fail status.

Implemented categories:

- real-data prior-layer performance and Layer 2/3/4/5 ablation
- real-data uncertainty interval coverage on the final available prior layer
- posterior-health smoke checks for the V1 posterior update runtime
- sequential-forecasting smoke checks for patient-specific updates
- update-value smoke checks comparing population-prior and posterior errors
- scenario-lab stability smoke checks, including fail-closed unsafe scenarios
- explanation-quality smoke audits for guardrails, sections, uncertainty drivers, and prior context

Runtime-layer smoke evals are deterministic wiring and artifact checks. They do not replace real-data performance claims, calibration checks, posterior-health checks on production cohorts, scenario-lab stability evals on curated scenario sets, or explanation-quality audits with adjudicated rubric data.
<!-- V1_PRIOR_STACK_EVALS_END -->

