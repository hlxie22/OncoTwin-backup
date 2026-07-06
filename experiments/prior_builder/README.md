# V1 Prior Builder Experiments

This directory contains the V1 layered-prior implementation.

Implemented modules:

```text
transforms.py                transformed parameter-space utilities
```

Planned modules:

```text
parameter_contract.py        learnable vs fixed parameter rules
bounds.py                    biologic/numeric bounds and warnings
population_prior.py          subtype/treatment population priors
pathology_rules.py           pathology shifts and uncertainty rules
biomarker_rules.py           biomarker shifts and missingness rules
mri_feature_rules.py         MRI feature and QC prior rules
ai_residual.py               disabled no-op residual first, learned residual later
assemble_prior.py            layer composition and traceable prior object
sample_prior.py              correlated particle sampling
adapter_to_volume_ode.py     V1 prior samples to current volume ODE parameters
smc_update.py                Bayesian-style particle reweighting and ESS policy
calibration_metrics.py       coverage, interval width, and log-likelihood helpers
```

The first V1 implementation should target TNBC + A/C-T style chemotherapy in volume-only mode. It should not learn every simulator parameter. It should personalize only growth, active treatment sensitivity, and resistant fraction unless a later evaluation proves that a larger parameter set is identifiable and useful.