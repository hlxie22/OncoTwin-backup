# v0 Recovery and Identifiability Sweep

Case: `demo_longitudinal_measurement_001`
Runs: `96`
Generated observation noise: `0.08`
Fit days: `[0.0, 42.0, 84.0]`
Held-out day: `126.0`

## Insights
- Particle reweighting improved median held-out prediction versus the prior ensemble.
- Effective sample size stayed below 5% of particles, so posterior narrowing remains fragile.
- The best median held-out method was exponential, so mechanistic fitting should be compared against simple baselines before product claims.
- The lowest median posterior held-out error came from the shared_chemo_fixed_core variant.

## By Assumed Noise
| assumed noise | runs | median ESS | median ESS fraction | median fit posterior RMSE | median posterior held-out error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.08 | 24 | 7.70 | 0.017 | 0.505 | 1.631 |
| 0.12 | 24 | 11.57 | 0.025 | 0.516 | 1.720 |
| 0.20 | 24 | 19.37 | 0.043 | 0.664 | 1.945 |
| 0.30 | 24 | 29.89 | 0.066 | 1.049 | 2.308 |

## By Particle Count
| particles | runs | median ESS | median ESS fraction | median posterior held-out error |
| ---: | ---: | ---: | ---: | ---: |
| 300 | 48 | 10.40 | 0.035 | 1.771 |
| 900 | 48 | 27.97 | 0.031 | 1.945 |

## By Variant
| variant | runs | median ESS fraction | median posterior held-out error | median RMSE improvement |
| --- | ---: | ---: | ---: | ---: |
| active_drugs_only | 24 | 0.039 | 1.975 | 0.940 |
| fixed_core | 24 | 0.036 | 2.114 | 0.930 |
| full | 24 | 0.039 | 2.312 | 0.938 |
| shared_chemo_fixed_core | 24 | 0.019 | 0.902 | 0.919 |

## Held-Out Method Comparison
| method | median absolute error to truth | mean absolute error | win count |
| --- | ---: | ---: | ---: |
| exponential | 1.220 | 1.177 | 71 |
| last_observation | 5.211 | 5.205 | 0 |
| last_slope | 3.263 | 3.426 | 0 |
| linear | 1.438 | 1.438 | 0 |
| posterior_particle_mean | 1.824 | 1.795 | 25 |
| prior_particle_mean | 13.525 | 13.598 | 0 |

## Recommendation
Keep v0 as a research simulator. Use reduced parameter variants and held-out baseline checks before product-facing parameter explanations.

