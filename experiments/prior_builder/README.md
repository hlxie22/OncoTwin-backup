# V1 Prior Builder Experiments

This directory contains the V1 layered-prior implementation.

Implemented modules:

```text
parameter_contract.py        Layer 0 learnable vs fixed parameter rules
bounds.py                    Layer 1 biologic bounds and observation-noise policy
transforms.py                transformed parameter-space utilities
population_prior.py          Layer 2 TNBC chemotherapy population prior
pathology_biomarker_rules.py Layer 3 pathology/biomarker shifts and missingness rules
mri_feature_rules.py         Layer 4 MRI feature/QC rules and observation noise
adapter_to_volume_ode.py     V1 prior samples to current volume ODE parameters
```

Planned modules:

```text
ai_residual.py               disabled no-op residual first, learned residual later
assemble_prior.py            layer composition and traceable prior object
sample_prior.py              correlated particle sampling
smc_update.py                Bayesian-style particle reweighting and ESS policy
calibration_metrics.py       coverage, interval width, and log-likelihood helpers
```

The first V1 implementation should target TNBC + A/C-T style chemotherapy in volume-only mode. It should not learn every simulator parameter. It should personalize only growth, active treatment sensitivity, and resistant fraction unless a later evaluation proves that a larger parameter set is identifiable and useful.