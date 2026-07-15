# V1 explanation runtime

The V1 explanation runtime converts posterior-update and scenario-lab artifacts
into structured clinician/patient explanations.

It is implementation-layer code, not an eval runner. It does not make treatment
recommendations.

## Command

```bash
python3 scripts/explain_v1_twin_update.py \
  --posterior-update /tmp/v1_volume_posterior_update.json \
  --scenario-lab /tmp/v1_scenario_lab.json \
  --audience clinician \
  --format json \
  --output /tmp/v1_twin_explanation.json
```

For a reviewable Markdown version:

```bash
python3 scripts/explain_v1_twin_update.py \
  --posterior-update /tmp/v1_volume_posterior_update.json \
  --scenario-lab /tmp/v1_scenario_lab.json \
  --audience patient \
  --format markdown \
  --output /tmp/v1_twin_explanation.md
```

## Output contract

The JSON explanation includes:

```text
explanation_runtime_version
audience
summary
key_factors
uncertainty_drivers
posterior_update_explanation
scenario_comparison_explanation
prior_context_explanation
sections
safety_and_scope_note
not_a_treatment_recommendation
source_versions
```

## Guardrails

Every output includes:

```text
not_a_treatment_recommendation: true
safety_and_scope_note
```

The explanation layer summarizes model artifacts. It should not be used as a
diagnosis, prescription, or treatment recommendation.
