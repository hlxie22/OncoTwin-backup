# Mechanistic simulator schemas

This directory intentionally separates fixture and output shapes:

- `resolved_params.schema.json`: fully resolved simulator parameters.
- `prior_config.schema.json`: bounded prior/sampling configuration.
- `tumor_measurement.schema.json`: one individual tumor measurement.
- `case_fixture.schema.json`: case wrapper with baseline and optional longitudinal measurements.
- `treatment_schedule.schema.json`: treatment schedule fixture.
- `ensemble_output.schema.json`: multi-particle ensemble simulator output.
- `single_trajectory_output.schema.json`: deterministic single-trajectory simulator output.
- `report_output.schema.json`: non-trajectory evaluation/report outputs.

Compatibility copies are retained for older references:

- `mechanistic_params.schema.json` has the same content as `resolved_params.schema.json`.
- `simulation_output.schema.json` has the same content as `ensemble_output.schema.json`.
