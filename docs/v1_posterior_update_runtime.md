# V1 posterior update runtime

This document describes the first implementation-layer posterior runtime. It is
not an eval runner and does not change the V1 suite status by itself.

## Scope

The runtime currently supports volume-only posterior updates using static
parameter particles and the existing V0 volume ODE simulator. The update is
batch importance sampling from the original prior particles against all
tumor-volume observations to date.

## Why batch reweighting

V1 parameters represent static tumor biology. Reweighting from the prior against
all observations avoids sequential resampling loops that duplicate particles and
create false posterior precision.

## Command

```bash
python3 scripts/run_v1_volume_posterior_update.py \
  --initial-volume-ml 28.0 \
  --schedule fixtures/mechanistic_simulator/schedules/chemotherapy_schedule.json \
  --particles /path/to/particles.jsonl \
  --observations /path/to/observations.jsonl \
  --prediction-days 21 84 140 \
  --output /tmp/v1_volume_posterior_update.json
```

## Observation behavior

Tumor-volume observations require:

```text
day
tumor_volume_ml
source
confidence
segmentation_qc
```

Failed-QC observations are ignored by default. Low-quality or uncertain
observations widen the likelihood instead of silently narrowing posterior
uncertainty.

## Output

The JSON output includes:

```text
effective_sample_size
effective_sample_size_fraction
fallback_status
prior_trajectory_summary
posterior_trajectory_summary
parameter_summary
uncertainty_summary
update_explanation
warnings
```

If ESS falls below the configured threshold, `fallback_status` is
`tempered_smc_recommended`. This is a diagnostic state; the current V1 runtime
does not fake diversity by resampling duplicate particles.
