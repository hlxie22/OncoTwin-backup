# Mechanistic Simulator Fixtures

These fixtures implement the first mechanistic-simulator step: small, qualitative demo cases that future volume-only simulation experiments can load without depending on the product app.

The fixtures are intentionally synthetic and demo-only. They are not clinical evidence, treatment guidance, or validated response predictions.

## Layout

```text
fixtures/mechanistic_simulator/
  cases/      Demo tumor-response cases.
  schedules/  Coarse treatment schedules referenced by the cases.
  params/     Demo parameter prior and named anchor parameter sets.
```

## Case fixture conventions

Each case includes:

- `case_id`
- `subtype`
- `baseline_measurement.tumor_volume_ml`
- optional `baseline_measurement.longest_diameter_cm`
- `treatment_schedule.schedule_id` and `treatment_schedule.path`
- `expected_qualitative_behavior`

The expected behavior is qualitative on purpose. It should guide sanity checks for the first simulator harness without overfitting numeric trajectories before the model exists.

## Parameter fixture conventions

`params/generic_volume_prior.json` is a bounded sampling prior used for ensemble
and synthetic-recovery checks. The named parameter files are deterministic anchor
sets for strong-response, weak-response, and resistant-disease trajectories.

All parameter fixtures are synthetic, demo-only, and not clinically validated.
