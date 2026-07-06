# V1 Prior Stack Implementation and Evaluation Plan

## Purpose

This document defines the first serious V1 implementation path for OncoTwin's mechanistic simulation work.

V1 is not the full spatial MRI, CNN, molecular-graph, and patient-facing product. V1 is the first evidence-gated implementation that should prove whether a transparent layered prior plus Bayesian updating can produce useful, calibrated, explainable tumor-response simulations.

The core V1 question is:

```text
Did the digital twin become more useful after seeing patient-specific evidence?
```

V1 should answer:

```text
Can patient-specific evidence improve forecasts?
Does the posterior improve as follow-up measurements arrive?
Are uncertainty intervals calibrated?
Does the model know when it is likely to be wrong?
Do explanations respect identifiability?
Can scenario comparisons be useful without becoming treatment-prescriptive?
Where does the model fail, and does it warn users honestly?
```

## V1-A scope

Start narrow:

```text
Disease context: TNBC
Treatment context: A/C-T style neoadjuvant chemotherapy
Simulator mode: volume-only
Prior mode: layered, transparent, non-AI first
Update mode: Bayesian-style particle reweighting
Primary outputs: trajectory forecast, uncertainty bands, posterior health, explanation, failure-mode report
```

Inputs:

```text
case metadata
baseline tumor volume or diameter
subtype / ER / PR / HER2
grade
Ki-67 if available
BRCA / HRD if available
MRI-derived volume / FTV / QC if available
A/C-T chemotherapy schedule
follow-up tumor measurements when available
```

Personalized simulator parameters:

```text
growth_rate_per_day
active_treatment_sensitivity
resistant_fraction
```

Fixed or non-AI-derived parameters:

```text
carrying_capacity_ml
drug_decay
drug_ec50
resistant_sensitivity_scale
observation_noise_fraction
inactive drug sensitivities
```

Observation noise should come from measurement source and QC, not from AI personalization.

## Out of scope for V1-A

Delay these until the V1 prior stack and evaluation suite are stable:

```text
3D spatial reaction-diffusion simulation
CNN image encoder residuals
learned molecular graph attention
multi-regimen optimization
patient-facing treatment ranking
symptom-driven tumor-biology updates
```

Symptoms may affect person-burden summaries and care-team questions, but they must not update tumor-response biology directly.

## V1 architecture

Represent the learnable prior in transformed parameter space:

```text
z = [
  log(growth_rate_per_day),
  log(active_treatment_sensitivity),
  logit(resistant_fraction)
]
```

Assemble the prior in explicit layers:

```text
patient_prior =
    Layer 0 parameter contract
  + Layer 1 biologic/numeric bounds
  + Layer 2 subtype/treatment population prior
  + Layer 3 pathology and biomarker shifts
  + Layer 4 MRI feature and QC shifts
  + Layer 5 conservative AI residual, initially disabled
```

Then update with observations:

```text
posterior proportional_to patient_prior * likelihood(observed follow-up data | simulation)
```

Every prior object should expose its composition:

```json
{
  "prior_version": "oncotwin_prior_v1",
  "base_group": "tnbc_chemo",
  "learnable_parameters": [
    "growth_rate_per_day",
    "active_treatment_sensitivity",
    "resistant_fraction"
  ],
  "fixed_parameters": [
    "carrying_capacity_ml",
    "drug_decay",
    "drug_ec50",
    "resistant_sensitivity_scale",
    "observation_noise_fraction"
  ],
  "layer_contributions": {
    "parameter_contract": [],
    "bounds": [],
    "population_prior": {},
    "pathology_biomarker_rules": [],
    "mri_feature_rules": [],
    "ai_residual": {}
  },
  "warnings": [],
  "uncertainty_drivers": []
}
```

## Implementation plan

### Phase 0: Freeze the V0 baseline

Before changing simulator behavior, snapshot the existing V0 harness.

Capture:

```text
current generic-prior ensemble
current synthetic recovery report
current identifiability report
current recovery sweep
current large-particle check
current simple baseline comparisons
```

Required tests:

```text
existing unit tests pass
existing integration tests pass
fixed-seed V0 outputs are reproducible
baseline reports can be regenerated
```

Evaluation outputs:

```text
V0 held-out volume error
V0 uncertainty coverage
V0 posterior ESS summary
V0 identifiability summary
no-change / linear / exponential baseline leaderboard
```

Gate:

```text
V0 remains reproducible and can be used as a regression baseline.
Simple baselines are visible in every V1 comparison report.
```

Watch out for:

```text
Treating V0 as a product architecture.
Hiding simple baselines because they are inconveniently strong.
```

### Phase 1: Add transformed-space prior utilities

Implement:

```text
transforms.py
```

Functions:

```text
to_transformed(params)
from_transformed(z)
safe_log
safe_logit
safe_sigmoid
median_interval_to_normal
validate_covariance
sample_correlated_transformed_prior
```

Required tests:

```text
round-trip transform tests
near-zero resistant-fraction tests
near-one resistant-fraction tests
invalid covariance tests
positive-semidefinite covariance tests
sample quantile sanity tests
fixed-seed reproducibility tests
```

Evaluation outputs:

```text
sampled median vs configured median
sampled 80% interval vs configured 80% interval
bound-hit frequency
```

Gate:

```text
Sampling behaves predictably in transformed space.
Normal sampling does not require frequent silent clipping.
```

Watch out for:

```text
Samples repeatedly hitting hard bounds.
Unstable logit behavior near 0 or 1.
Overly narrow transformed-space variance.
```

### Phase 2: Layer 0 parameter contract

Implement context-specific learnable parameter sets.

For V1-A:

```text
TNBC + chemotherapy:
  learnable:
    growth_rate_per_day
    active_treatment_sensitivity
    resistant_fraction

  fixed:
    carrying_capacity_ml
    drug_decay
    drug_ec50
    resistant_sensitivity_scale
    observation_noise_fraction
    inactive drug sensitivities
```

Required tests:

```text
TNBC chemo activates only growth/sensitivity/resistance
inactive drug sensitivities cannot be personalized
unknown regimen falls back to conservative generic behavior
out-of-scope regimen emits warning
fixed parameters remain fixed after sampling
```

Evaluation outputs:

```text
parameter-count report
prior-dominated parameter report
out-of-scope case warning report
```

Gate:

```text
The model cannot accidentally personalize parameters outside the V1 contract.
```

Watch out for:

```text
Parameter creep.
Trying to learn anthracycline, taxane, carrying capacity, observation noise, and resistant sensitivity scale all at once.
```

### Phase 3: Layer 1 bounds and observation noise

Initial volume-ODE bounds:

```text
growth_rate_per_day:
  normal: 0.0005-0.020
  warning_high: >0.030
  hard_stop: >0.100

active_treatment_sensitivity:
  normal: 0.015-0.200
  warning_high: >0.300
  hard_stop: >0.500

resistant_fraction:
  normal: 0.00-0.65
  warning_high: >0.75
  hard_stop: >0.90
```

Observation noise policy:

```text
high-QC MRI volume:       0.08 log-scale noise
medium-QC MRI volume:     0.12
low-QC MRI volume:        0.20
manual volume:            0.25
diameter-derived volume:  0.35
```

Required tests:

```text
normal values pass without warning
warning values pass with warning
hard-stop values fail loudly
negative growth/sensitivity fails
low-QC MRI increases observation noise
diameter-derived volume uses larger noise than MRI volume
```

Evaluation outputs:

```text
large sampling stress test
trajectory stability report
bound-warning frequency
hard-stop frequency
measurement-noise sensitivity report
```

Gate:

```text
Sampled particles produce stable and plausible volume trajectories.
Noise assumptions widen low-quality observations as expected.
```

Watch out for:

```text
Warning thresholds that are never triggered.
Warning thresholds that are always triggered.
Measurement noise so large that updates become uninformative.
```

### Phase 4: Layer 2 TNBC chemotherapy population prior

Initial TNBC chemotherapy prior:

```text
growth_rate_per_day median:             0.0067/day
active_treatment_sensitivity median:    0.090/day
resistant_fraction median:              0.20
```

Starting correlations:

```text
growth vs treatment sensitivity:        +0.25
treatment sensitivity vs resistance:    -0.40
growth vs resistance:                   -0.05
```

Required tests:

```text
configured medians are reproduced by samples
configured 80% intervals are approximately reproduced
covariance matrix is positive semidefinite
correlation signs match config
unknown population group fails or falls back explicitly
```

Evaluation outputs:

```text
Layer 2 vs V0 generic-prior leaderboard
Layer 2 vs no-change / linear / exponential baselines
Layer 2 uncertainty coverage
Layer 2 trajectory distribution report
```

Gate:

```text
Layer 2 improves or better calibrates predictions compared with V0 generic prior.
Layer 2 remains honestly compared against simple baselines.
```

Watch out for:

```text
Better RMSE but worse coverage.
Prior too narrow for sparse longitudinal data.
Subtype prior overclaiming patient-specific behavior.
```

### Phase 5: Adapter from V1 prior samples to the volume ODE

For V1-A, map:

```text
active_treatment_sensitivity
  -> shared chemo sensitivity
  -> drug_sensitivity.anthracycline
  -> drug_sensitivity.taxane
```

Inactive drug sensitivities stay inactive/fixed.

Required tests:

```text
adapter emits valid simulator params
shared sensitivity maps consistently to active chemotherapy agents
inactive agents remain fixed
fixed nuisance parameters do not vary across particles
adapter output validates against simulator parameter schema
```

Evaluation outputs:

```text
adapter parity smoke test
simulator input distribution report
active-vs-inactive drug sensitivity report
```

Gate:

```text
The V1 prior builder can feed the current simulator without modifying simulator internals.
```

Watch out for:

```text
Accidentally reintroducing inactive drug sensitivity variation.
Breaking existing V0 tests.
```

### Phase 6: Layer 3 pathology and biomarker rules

Start with modest rules for:

```text
Ki-67
grade
ER/PR/HER2 consistency
BRCA/HRD status
BRCA/HRD missingness
```

Rules should shift or widen priors. They should not create false certainty.

Example rule shape:

```json
{
  "rule_id": "ki67_high_v1",
  "condition": "ki67_percent >= 30",
  "effects": {
    "growth_multiplier": 1.50,
    "chemo_sensitivity_multiplier": 1.10,
    "resistant_odds_multiplier": 0.90,
    "growth_variance_multiplier": 1.10
  },
  "evidence_level": "A/B",
  "explanation": "High Ki-67 shifts proliferation assumptions upward with modest chemotherapy-response effect."
}
```

Required tests:

```text
unknown Ki-67 widens growth variance and does not invent status
Ki-67 <=5 lowers growth prior
Ki-67 >=30 raises growth prior
intermediate Ki-67 causes only modest or no shift
grade 3 raises growth modestly
BRCA/HRD unknown is not treated as negative
every applied rule appears in layer_contributions
```

Evaluation outputs:

```text
Layer 2 only vs Layer 3 full
Layer 2 + Ki-67 only
Layer 2 + grade only
Layer 2 + biomarkers only
cases helped/harmed by each rule family
uncertainty change by missingness pattern
```

Gate:

```text
Pathology/biomarker rules provide measurable lift or calibrated uncertainty changes.
Rules that harm performance are weakened, removed, or kept only as uncertainty drivers.
```

Watch out for:

```text
Outcome leakage.
Unknown values treated as negative values.
Biomarker rules dominating observed follow-up data.
```

### Phase 7: Layer 4 MRI feature and QC rules

Initial MRI feature vector:

```text
volume_ml
functional_tumor_volume_ml
longest_diameter_cm
enhancement_mean
enhancement_std
low_enhancement_fraction
sphericity
compactness
segmentation_qc
registration_qc, if longitudinal imaging exists
```

In V1-A, MRI/QC should mostly affect:

```text
observation noise
resistance/delivery uncertainty
warnings
uncertainty drivers
```

Required tests:

```text
low segmentation QC inflates observation noise
high heterogeneity widens resistant-fraction uncertainty
low-enhancement fraction increases resistance/delivery uncertainty
missing MRI features do not fail the prior builder
FTV/anatomic-volume inconsistency creates QC warning
diameter-only case falls back to report-only mode
```

Evaluation outputs:

```text
Layer 3 vs Layer 4 leaderboard
coverage by MRI QC group
error-vs-uncertainty by MRI QC group
same-volume paired-case MRI feature directionality test
MRI feature ablation report
```

Gate:

```text
MRI features improve calibration or useful uncertainty stratification.
Low-quality imaging produces wider and better-calibrated uncertainty.
```

Watch out for:

```text
MRI feature rules making deterministic resistance claims.
MRI features dominating subtype/pathology priors.
QC widening intervals so much that predictions become useless.
```

### Phase 8: Bayesian update policy

Likelihood modes:

```text
log-volume likelihood
diameter-derived volume likelihood
manual-volume likelihood
MRI-volume likelihood
```

Track posterior health:

```text
ESS
ESS fraction
posterior/prior interval shrinkage
posterior predictive coverage
number of high-weight particles
seed-to-seed posterior stability
```

Suggested ESS policy:

```text
ESS >= 0.50N: accept
0.30-0.50N: accept with warning
0.10-0.30N: temper likelihood and report warning
0.05-0.10N: fragile posterior; temper/resample only for diagnostics
<0.05N: unreliable posterior; widen intervals and avoid parameter claims
```

Required tests:

```text
strong response downweights low-sensitivity particles
weak response downweights high-sensitivity particles
noisy observation does not collapse uncertainty
contradictory observation triggers warning
low ESS triggers tempering/fallback
posterior explanation avoids parameter claims when ESS is low
symptom/tolerance data does not update tumor biology parameters
```

Evaluation outputs:

```text
baseline-only vs early-update prediction
early-update vs mid-update prediction
posterior ESS report
posterior collapse report
posterior coverage report
posterior/prior shrinkage report
```

Gate:

```text
Follow-up measurements improve future prediction more often than they hurt it.
Posterior narrowing is accepted only when ESS and coverage remain healthy.
```

Watch out for:

```text
Posterior intervals getting narrower because nearly all particles died.
Parameter explanations after low-ESS updates.
Contradictory observations being treated as certainty.
```

### Phase 9: Evaluation suite before AI residuals

Build the evaluation suite before enabling learned residuals.

Planned report outputs:

```text
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

Gate:

```text
A single command can regenerate the V1 evaluation reports.
Every report includes simple baselines, sample size, metric definitions, and warnings.
```

Watch out for:

```text
Only reporting one aggregate metric.
Not reporting cases helped and harmed.
Not reporting simple baselines.
```

### Phase 10: AI residual, initially disabled

Implement the interface but keep it as a no-op until the non-AI prior stack is benchmarked.

Initial residual:

```json
{
  "delta_log_growth_rate": 0.0,
  "delta_log_active_treatment_sensitivity": 0.0,
  "delta_logit_resistant_fraction": 0.0,
  "uncertainty_multipliers": {
    "growth": 1.0,
    "sensitivity": 1.0,
    "resistance": 1.0
  },
  "ood_score": 0.0
}
```

Later residual limits:

```text
growth multiplier: 0.50-2.00
sensitivity multiplier: 0.75-1.35
resistant odds multiplier: 0.67-1.50
uncertainty multiplier: 0.75-1.50
severe OOD: residual = 0, uncertainty x2.00
```

Required tests:

```text
AI-disabled mode exactly reproduces Layer 4 prior
AI residual cannot exceed clip range
OOD input zeros residual and widens uncertainty
AI cannot modify fixed parameters
AI contribution is separately traceable
```

Evaluation outputs, when AI is enabled later:

```text
Layer 0-4 non-AI baseline
Layer 0-4 + AI growth residual
Layer 0-4 + AI sensitivity residual
Layer 0-4 + full AI residual
calibration comparison
subgroup comparison
failure-mode comparison
```

Gate:

```text
AI residuals improve the frozen non-AI baseline without worsening uncertainty calibration, posterior health, safety, or explanation quality.
```

Watch out for:

```text
AI improves AUC but worsens trajectory calibration.
AI learns site/scanner/cohort leakage.
AI makes outputs less explainable.
```

## Best V1 evaluations

### 1. Predictive performance leaderboard

Question:

```text
Does V1 predict tumor response better than alternatives?
```

Compare:

```text
no-change baseline
linear shrinkage baseline
exponential shrinkage baseline
subtype-average response baseline
V0 generic prior
V1 Layer 2 population prior
V1 Layer 3 pathology/biomarker prior
V1 Layer 4 MRI/QC prior
V1 posterior after early follow-up
```

Metrics:

```text
MAE final volume
RMSE final volume
log-volume RMSE
median absolute percentage error
rank correlation with final residual burden
pCR/non-pCR AUC, if labels exist
Brier score, if probabilities are emitted
```

Success signal:

```text
V1 beats V0 generic prior and remains competitive with simple baselines.
If simple baselines win, V1 provides better-calibrated uncertainty or useful mechanistic insight.
```

### 2. Sequential forecasting evaluation

Question:

```text
Does the twin improve as new observations arrive?
```

Protocol:

```text
Use T0 to predict T1/T2/T3.
Update with T1, then predict T2/T3.
Update with T2, then predict T3.
```

Metrics:

```text
forecast error by horizon
change in log likelihood after update
change in interval width after update
change in coverage after update
ESS after update
```

Success signal:

```text
Early and mid-treatment observations improve future prediction and calibration more often than they hurt it.
```

### 3. Uncertainty calibration evaluation

Question:

```text
Are uncertainty intervals honest?
```

Metrics:

```text
80% interval coverage
95% interval coverage
average interval width
coverage by timepoint
coverage by measurement type
coverage by MRI QC group
negative log likelihood
```

Success signal:

```text
Observed outcomes fall inside nominal intervals at approximately the claimed rates.
Low-quality or sparse cases show wider uncertainty.
```

### 4. Error-vs-uncertainty evaluation

Question:

```text
Does the model know when it is likely to be wrong?
```

Metrics:

```text
Spearman correlation between interval width and absolute error
error by uncertainty quintile
coverage by uncertainty quintile
```

Success signal:

```text
High-uncertainty cases have higher average error than low-uncertainty cases.
```

### 5. Personalization-lift ablation

Question:

```text
Which patient-specific evidence actually helps?
```

Run:

```text
population prior only
+ pathology
+ biomarkers
+ MRI features/QC
+ early follow-up observation
+ AI residual, later
```

Metrics:

```text
accuracy delta
calibration delta
interval-width delta
number of cases helped
number of cases harmed
subgroup-level lift
```

Success signal:

```text
The report can say which layer helped, which mainly changed uncertainty, and which harmed performance.
```

### 6. Value-of-information evaluation

Question:

```text
Which added data types make the twin meaningfully better?
```

Run counterfactual missingness tests:

```text
hide Ki-67
hide grade
hide BRCA/HRD
hide MRI QC
hide early follow-up volume
```

Metrics:

```text
prediction degradation
uncertainty widening
change in posterior ESS
change in scenario ranking or interval overlap
```

Success signal:

```text
Important missing data widens uncertainty and sometimes degrades performance.
Unknown values are not silently treated as negative values.
```

### 7. Posterior health and collapse evaluation

Question:

```text
Is Bayesian updating learning, or just killing particles?
```

Metrics:

```text
ESS fraction
number of high-weight particles
posterior/prior shrinkage ratio
coverage after shrinkage
seed-to-seed posterior stability
```

Success signal:

```text
Posterior narrowing is accepted only when ESS and coverage remain healthy.
```

### 8. Explanation-quality and identifiability audit

Question:

```text
Do explanations respect what the model can actually identify?
```

Audit cases where parameters are:

```text
well constrained
partially constrained
prior dominated
non-identifiable
```

Metrics:

```text
percentage of explanations that mention uncertainty when required
percentage of prior-dominated parameters incorrectly stated as known
driver-stability agreement across seeds
```

Success signal:

```text
The model explains trajectories, uncertainty, and evidence shifts without pretending weakly identified parameters are measured facts.
```

### 9. Scenario-lab stability evaluation

Question:

```text
Can research scenarios be compared without overclaiming?
```

Evaluate scenarios:

```text
current schedule
delayed dose event
reduced exposure event
strong early response observation
weak early response observation
missing biomarker becomes positive/negative/unknown
```

Metrics:

```text
scenario ranking stability
interval overlap
probability one scenario beats another
sensitivity to posterior uncertainty
required disclaimer presence
```

Success signal:

```text
Scenario outputs show uncertainty and avoid treatment-prescriptive language.
```

### 10. Failure-mode discovery report

Question:

```text
Where does V1 fail, and what do those failures teach us?
```

Create buckets:

```text
simple-baseline-wins cases
mechanistic-model-wins cases
posterior-collapse cases
unexpected responders
unexpected non-responders
low-QC imaging cases
high-missingness cases
```

Summarize:

```text
common features
uncertainty drivers
which prior layer helped or hurt
whether follow-up updating corrected the prior
whether warnings were appropriate
```

Success signal:

```text
The report produces actionable modeling insights, not just a leaderboard.
```

## Test cadence

### Every pull request

Run:

```text
unit tests
schema tests
snapshot tests for prior output
safety-language tests
small deterministic smoke simulation
```

### After each prior layer is implemented

Run:

```text
layer-specific unit tests
directionality tests
sampling sanity tests
small synthetic ensemble
ablation against previous layer
```

### After changes to bounds, covariance, or rules

Run:

```text
large sampling stress test
trajectory sanity sweep
identifiability report
coverage check on synthetic cases
```

### After changes to Bayesian update

Run:

```text
synthetic recovery tests
strong/weak/noisy/contradictory observation tests
ESS collapse tests
held-out prediction leaderboard
```

### Before adding AI residuals

Freeze:

```text
Layer 0-4 configs
train/validation/test split
baseline leaderboard
calibration report
safety criteria
```

### Before any product-facing demo

Run:

```text
full regression suite
full recovery sweep
baseline leaderboard
safety red-team tests
example output review
manual review of explanations
```

## V1 success criteria

V1 succeeds if it demonstrates a useful and honest modeling pattern, not if it claims clinical readiness.

Minimum success criteria:

```text
V1 beats the V0 generic prior on synthetic or semi-synthetic held-out forecasting.
V1 remains competitive with simple no-change, linear, and exponential baselines.
When simple baselines win, V1 exposes why and whether uncertainty is better calibrated.
Follow-up observations improve future prediction more often than they hurt it.
80% and 95% intervals are approximately calibrated.
Higher uncertainty correlates with higher error.
Missing data widens uncertainty rather than becoming fake negative evidence.
Explanations respect identifiability and posterior health.
Scenario comparisons include uncertainty and avoid treatment-directive claims.
Safety-language tests pass for all product-facing summaries.
```

## Main red flags

```text
Held-out error improves but interval coverage worsens.
Posterior ESS is very low but posterior intervals look narrow.
Simple baselines still win and the report hides that fact.
Parameter explanations appear before identifiability is earned.
Layer rules dominate follow-up observations.
Missing data behaves like negative data.
AI residual improves AUC but worsens calibration.
Scenario outputs sound like treatment recommendations.
Safety disclaimers disappear in summaries.
```

## Practical build order

```text
1. Freeze V0 baseline and simple baselines.
2. Add V1 prior-builder docs, config directories, and evaluation scaffolding.
3. Implement transformed-space utilities.
4. Implement Layer 0 parameter contract.
5. Implement Layer 1 bounds and observation-noise policy.
6. Implement adapter from V1 prior particles to current volume ODE params.
7. Implement Layer 2 TNBC chemotherapy population prior.
8. Run Layer 2 vs V0 vs simple-baseline evaluation.
9. Implement Layer 3 pathology/biomarker rules.
10. Run Layer 3 ablation and missingness evaluation.
11. Implement Layer 4 MRI/QC rules.
12. Run Layer 4 calibration, uncertainty, and QC subgroup evaluation.
13. Formalize Bayesian update policy and ESS gates.
14. Run sequential forecasting and update-value evaluations.
15. Add failure-mode, explanation-quality, and scenario-stability reports.
16. Add AI residual interface as a disabled no-op.
17. Enable learned AI residuals only after the non-AI V1 stack has a frozen benchmark.
```