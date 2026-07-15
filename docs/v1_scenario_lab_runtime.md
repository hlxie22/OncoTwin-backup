# V1 scenario-lab runtime

This is the first implementation-layer scenario lab. It consumes a posterior
update JSON artifact and compares candidate schedules under the posterior
particle weights.

It is not an eval runner and it does not make treatment recommendations.

## Command

```bash
python3 scripts/run_v1_scenario_lab.py \
  --posterior-update /tmp/v1_volume_posterior_update.json \
  --scenarios /path/to/scenarios.json \
  --output-days 21 84 140 \
  --residual-burden-threshold-ml 1.0 \
  --output /tmp/v1_scenario_lab.json
```

## Scenario input

The scenarios file can be either a JSON array or an object with a `scenarios`
array:

```json
{
  "scenarios": [
    {
      "scenario_id": "current_plan",
      "label": "Current plan",
      "reference": true,
      "treatment_schedule": {
        "schedule_id": "current_plan",
        "regimen_name": "Current plan",
        "total_duration_days": 140,
        "events": []
      }
    }
  ]
}
```

Each schedule uses the existing V0 simulator treatment-schedule contract:
`schedule_id`, `regimen_name`, `total_duration_days`, and `events`.

## Safety behavior

The scenario lab fails closed per scenario. Missing schedule fields, malformed
events, negative doses, events beyond the schedule horizon, or relative doses
above the configured cap are marked `failed_safety` and excluded from ranking.

## Output

The JSON output includes:

```text
decision_support_disclaimer
not_a_treatment_recommendation
scenarios[].trajectory_summary
scenarios[].probabilities
scenarios[].comparison_to_reference
comparison_summary.ranked_scenario_ids_by_low_residual_probability
warnings
```

Ranking is based on modeled posterior outcomes only. It should be treated as a
research summary for clinician review, not as a treatment recommendation.
