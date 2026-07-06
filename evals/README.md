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