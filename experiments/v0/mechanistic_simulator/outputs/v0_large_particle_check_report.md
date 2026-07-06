# v0 Recovery and Identifiability Sweep

Case: `demo_longitudinal_measurement_001`
Runs: `9`
Generated observation noise: `0.08`
Fit days: `[0.0, 42.0, 84.0]`
Held-out day: `126.0`

## Insights
- Particle reweighting improved median held-out prediction versus the prior ensemble.
- Effective sample size stayed below 5% of particles, so posterior narrowing remains fragile.
- The best median held-out method was exponential, so mechanistic fitting should be compared against simple baselines before product claims.
- The lowest median posterior held-out error came from the full variant.

## By Assumed Noise
| assumed noise | runs | median ESS | median ESS fraction | median fit posterior RMSE | median posterior held-out error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.12 | 9 | 54.14 | 0.027 | 0.481 | 1.813 |

## By Particle Count
| particles | runs | median ESS | median ESS fraction | median posterior held-out error |
| ---: | ---: | ---: | ---: | ---: |
| 2000 | 3 | 54.14 | 0.027 | 1.847 |
| 5000 | 3 | 134.63 | 0.027 | 1.783 |
| 900 | 3 | 23.40 | 0.026 | 1.744 |

## By Variant
| variant | runs | median ESS fraction | median posterior held-out error | median RMSE improvement |
| --- | ---: | ---: | ---: | ---: |
| full | 9 | 0.027 | 1.813 | 0.960 |

## Held-Out Method Comparison
| method | median absolute error to truth | mean absolute error | win count |
| --- | ---: | ---: | ---: |
| exponential | 1.220 | 1.177 | 9 |
| last_observation | 5.211 | 5.205 | 0 |
| last_slope | 3.263 | 3.426 | 0 |
| linear | 1.438 | 1.438 | 0 |
| posterior_particle_mean | 1.813 | 1.882 | 0 |
| prior_particle_mean | 13.263 | 13.237 | 0 |

## Recommendation
Keep v0 as a research simulator. Use reduced parameter variants and held-out baseline checks before product-facing parameter explanations.

