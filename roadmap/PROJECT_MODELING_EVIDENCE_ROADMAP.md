# OncoTwin Project Modeling Evidence Roadmap

## Purpose

This roadmap reframes OncoTwin around the real project risk: whether the modeling works on real evidence.

Engineering completion is not success. A component is successful only if it improves model evidence, uncertainty quality, interpretability, or safety compared with a simpler alternative.

The project should not ask first:

~~~text
Can we build the digital-twin app?
~~~

It should ask:

~~~text
Can each modeling claim survive comparison against simpler baselines on real data?
~~~

The app should be built only around model behavior that has passed evidence gates.

## Core principle

Every new layer must earn its place.

A modeling layer should be added only if it improves at least one of:

~~~text
held-out prediction
uncertainty calibration
coverage of real observations
identifiability
interpretability
safety
clinical/research usefulness
~~~

over a simpler alternative.

## Roadmap overview

~~~text
0. Claims, tasks, and evidence standards
1. Real-data audit and modeling dataset construction
2. Baseline leaderboard
3. Tumor-response modeling core
4. Bayesian updating and longitudinal evidence
5. Imaging value test
6. Pathology and molecular personalization value test
7. Patient-burden and symptom modeling
8. Scenario Lab evidence and safety
9. Productization from proven model slices
10. Model cards, claims policy, and external validation
~~~

## Phase 0: Claims, tasks, and evidence standards

### Goal

Define exactly what the project is trying to prove before building more components.

### Build

Create a claim matrix with:

~~~text
claim_id
claim_text
intended_user
input_data_required
prediction_or_summary_target
baseline_to_beat
metric
minimum_success_threshold
required_dataset
known_failure_modes
fallback_if_failed
allowed_product_language
forbidden_product_language
~~~

### Candidate claims

~~~text
Claim A:
The model can forecast tumor-volume change with calibrated uncertainty.

Claim B:
An early follow-up measurement improves later tumor-response prediction.

Claim C:
MRI-derived features improve prediction beyond baseline volume, subtype, and treatment category.

Claim D:
Pathology or molecular features improve parameter priors or response prediction beyond generic or subtype-level priors.

Claim E:
Patient-reported symptoms can support burden tracking, daily planning, and care-team questions without being used as direct tumor-response evidence.

Claim F:
Scenario Lab outputs can expose assumptions and uncertainty without making unsafe treatment claims.
~~~

### Exit criteria

~~~text
Primary and secondary modeling targets are written down.
Each target has a baseline comparison.
Each target has metrics.
Each target has a fallback if it fails.
The project has a claims policy before model output appears in the product.
~~~

## Phase 1: Real-data audit and modeling dataset construction

### Goal

Determine whether real datasets can support the modeling tasks.

This phase should not stop at dataset inventory. It must produce modeling-ready manifests.

### Build

A real-data modeling manifest with:

~~~text
case_id
dataset_source
measurement_timepoints
measurement_days_relative_to_treatment
tumor_volume_ml
longest_diameter_cm
measurement_source
measurement_confidence
subtype
er_status
pr_status
her2_status
grade
ki67
molecular_markers_if_available
treatment_category
treatment_schedule_detail_if_available
outcome_label_if_available
missingness_flags
data_quality_flags
~~~

### Questions to answer

~~~text
Are there enough cases with longitudinal tumor measurements?
Are measurement days aligned with treatment?
Are treatment categories reliable enough for modeling?
Are subtype/pathology labels available?
Are MRI masks or volumes available, or must volumes be derived?
Are follow-up timepoints dense enough to test updating?
Can data be split into train/validation/test without leakage?
~~~

### Exit criteria

~~~text
At least one modeling-ready manifest exists.
The manifest supports at least one real prediction task.
Missingness and quality flags are explicit.
The team knows which claims are currently testable.
~~~

### Failure criteria

~~~text
If real longitudinal measurements cannot be assembled,
the tumor-response model cannot be validated yet.
Use demo/synthetic mode only.
~~~

## Phase 2: Baseline leaderboard

### Goal

Build the simple baselines before building or trusting mechanistic complexity.

No advanced model is useful unless it beats a simpler alternative or provides better calibrated uncertainty.

### Baselines

~~~text
No-change baseline:
future volume = baseline volume

Last-observation-carried-forward:
future volume = most recent measured volume

Linear trend baseline:
fit a line through available measurements

Population-average response curve:
average volume-change curve from training data

Subtype/regimen-average response curve:
average curve conditioned on subtype and treatment category

Simple statistical regression:
predict percent volume change from baseline features

Bayesian hierarchical response model:
estimate population, subtype, and regimen-level response distributions with uncertainty
~~~

### Metrics

~~~text
mean absolute error in tumor volume
relative volume error
error in percent change from baseline
negative log likelihood
prediction interval coverage
calibration of 50%, 80%, and 95% intervals
sharpness of uncertainty intervals
performance by subtype/regimen/data-source subgroup
~~~

### Exit criteria

~~~text
Baseline leaderboard exists.
Metrics are computed on held-out data.
Uncertainty baselines are included, not only point predictions.
Failure cases are inspected and documented.
~~~

### Gate

~~~text
No model component is considered useful unless it beats a simpler baseline on held-out real data
or improves uncertainty calibration, interpretability, or safety in a documented way.
~~~

## Phase 3: Tumor-response modeling core

### Goal

Test whether a mechanistic tumor-response model adds value beyond the baseline leaderboard.

The first model should be volume-only and low-dimensional.

### Starting model

~~~text
Input:
baseline tumor volume
treatment category or schedule
subtype if available
optional follow-up measurement

Output:
future tumor-volume distribution
uncertainty intervals
parameter uncertainty
identifiability warnings
~~~

### Modeling constraints

~~~text
Use few parameters.
Prefer broad uncertainty to false precision.
Do not expose patient-specific parameter explanations until identifiability is studied.
Do not claim clinical prediction.
Compare against the baseline leaderboard before adding complexity.
~~~

### Success metrics

~~~text
held-out prediction error improves over baselines
negative log likelihood improves over baselines
prediction interval coverage is reasonable
uncertainty is wider when data are sparse
model can identify when parameters are weakly constrained
~~~

### Gate

~~~text
If the mechanistic model does not beat naive or statistical baselines,
keep it as an educational scenario tool only.
Do not call it personalized tumor-response modeling.
~~~

## Phase 4: Bayesian updating and longitudinal evidence

### Goal

Test the central digital-twin claim: the model improves as new observations arrive.

### Core question

~~~text
Does adding T1 improve prediction of T2/T3 compared with using T0 alone?
~~~

### Experiments

~~~text
baseline-only prediction of future timepoints
baseline + early follow-up prediction
baseline + multiple follow-ups prediction
noisy-measurement robustness
contradictory-measurement robustness
posterior collapse detection
~~~

### Metrics

~~~text
change in held-out prediction error
change in negative log likelihood
change in uncertainty coverage
effective sample size
posterior collapse rate
calibration before and after update
frequency of overconfident wrong predictions
~~~

### Gate

~~~text
If updating does not improve later prediction or creates false precision,
do not say the twin learns.
Use observation overlays or scenario recalibration language instead.
~~~

## Phase 5: Imaging value test

### Goal

Test whether MRI-derived information adds modeling value.

Do not assume imaging complexity is valuable. Prove it.

### Imaging feature ladder

~~~text
Level 1:
baseline volume and longest diameter

Level 2:
mask-derived shape features

Level 3:
simple enhancement or intensity summaries

Level 4:
regional imaging features

Level 5:
deep imaging encoder features

Level 6:
spatial or voxelwise simulation
~~~

### Core question

~~~text
Do MRI-derived features improve tumor-response prediction beyond volume, subtype, and treatment category?
~~~

### Metrics

~~~text
held-out prediction improvement
calibration improvement
subgroup performance
robustness to segmentation uncertainty
sensitivity to image-quality flags
~~~

### Gate

~~~text
If imaging features do not improve held-out prediction or uncertainty calibration,
do not build expensive imaging personalization yet.
~~~

## Phase 6: Pathology and molecular personalization value test

### Goal

Test whether pathology or molecular features improve model priors or response prediction.

### Feature ladder

~~~text
Level 1:
ER, PR, HER2, subtype

Level 2:
grade and Ki-67

Level 3:
BRCA, HRD, TP53, PIK3CA, ESR1, ERBB2 if available

Level 4:
curated pathway rules

Level 5:
regularized molecular feature model

Level 6:
graph-based molecular explanation model
~~~

### Core question

~~~text
Do pathology or molecular features improve real-data performance over generic, regimen-level, or subtype-level priors?
~~~

### Metrics

~~~text
trajectory prediction improvement
uncertainty calibration improvement
feature ablation impact
missing-data behavior
subgroup calibration
~~~

### Gate

~~~text
If molecular/pathology features do not improve real-data performance,
keep explanations rule-based and avoid claiming personalized biological inference.
~~~

## Phase 7: Patient-burden and symptom modeling

### Goal

Validate the patient-facing symptom and burden layer separately from tumor-response modeling.

Symptoms must not directly update tumor biology.

### Allowed symptom uses

~~~text
burden summaries
trend detection
daily check-in selection
care-team questions
visit summaries
planning support
treatment-context interpretation
~~~

### Forbidden symptom uses

~~~text
direct tumor-response inference
progression claims
treatment efficacy claims
recommendations to start, stop, or change cancer treatment
invented urgent thresholds
~~~

### Evidence targets

~~~text
trend detection correctness
summary faithfulness to structured logs
safety-rule compliance
usefulness of care-team questions
no symptom-to-tumor-response claims
fallback to templates when LLM output fails validation
~~~

### Gate

~~~text
If LLM summaries are unsafe or unfaithful,
use deterministic templates and limit the LLM to editable care-team question drafts.
~~~

## Phase 8: Scenario Lab evidence and safety

### Goal

Build Scenario Lab only around model claims that have evidence.

### Scenario types

~~~text
Patient-planning scenarios:
safe preparation and care-team questions

Measurement-update scenarios:
how a new observation changes uncertainty

Missing-data scenarios:
which missing measurement could reduce uncertainty

Research-only treatment comparison scenarios:
clearly labeled, uncertainty-bounded, not patient-facing recommendations
~~~

### Scenario requirements

Every scenario must declare:

~~~text
model evidence supporting it
assumptions changed
data used
uncertainty shown
allowed interpretation
forbidden interpretation
safety disclaimer
~~~

### Gate

~~~text
If the underlying model has not passed real-data evaluation,
Scenario Lab may show educational ranges but not personalized rankings.
~~~

## Phase 9: Productization from proven model slices

### Goal

Build the app around evidence that passed, not around the original aspirational architecture.

### Productization rules

~~~text
If only symptom tracking passes:
build patient co-pilot and care-team summaries.

If volume-only modeling passes:
add tumor-volume trajectory dashboard.

If Bayesian updating passes:
add twin update timeline.

If imaging adds value:
add MRI-derived model inputs.

If molecular/pathology personalization passes:
add prior explanations.

If treatment comparison does not pass:
do not productize treatment ranking.
~~~

### Exit criteria

~~~text
Every product surface maps to a passed evidence gate.
Every model output has allowed and forbidden claims.
Every uncertainty visualization has a documented meaning.
Every patient-facing output has safety checks.
~~~

## Phase 10: Model cards, claims policy, and external validation

### Goal

Make the prototype honest, reviewable, and externally testable.

### Build

~~~text
dataset cards
baseline leaderboard report
tumor-response model card
Bayesian update model card
imaging value report
pathology/molecular value report
LLM safety report
scenario safety report
claims matrix
external validation plan
limitations page
~~~

### Each model card should include

~~~text
intended use
not intended use
training/evaluation data
metrics
calibration status
subgroup limitations
missing-data behavior
known failure modes
version
audit trail
allowed claims
forbidden claims
~~~

### Gate

~~~text
No model output should appear in the product without a matching model card or status note.
~~~

## Complexity ladder

Use this ladder for tumor-response modeling:

~~~text
Level 0: no-change / last-observation baseline
Level 1: population-average response curve
Level 2: subtype/regimen response curve
Level 3: simple statistical model
Level 4: volume-only mechanistic model
Level 5: volume-only Bayesian update model
Level 6: imaging-feature-informed prior
Level 7: pathology/molecular-informed prior
Level 8: regional or spatial model
Level 9: learned AI parameter amortizer
~~~

Rule:

~~~text
Do not move up the ladder unless the current level beats the simpler level
on held-out real data or clearly improves uncertainty, interpretability, or safety.
~~~

## Modeling MVP

The modeling MVP is not the product MVP.

The modeling MVP is:

~~~text
real-data measurement manifest
baseline leaderboard
volume-only mechanistic model
held-out evaluation report
uncertainty calibration report
identifiability report
Bayesian-update value test
failure-mode report
claims matrix
~~~

Only after this exists should the product MVP be finalized.

## Product MVP depends on evidence

The product MVP should be selected after evidence gates.

Possible product MVPs:

~~~text
If tumor modeling is weak:
tracking, education, observation overlays, and care-team summaries

If volume modeling is promising:
manual measurement entry, trajectory chart, uncertainty bands, and update timeline

If imaging adds value:
MRI-derived inputs and QC warnings

If Bayesian updating adds value:
learning twin timeline

If Scenario Lab is supported:
safe scenario planning with assumptions and uncertainty
~~~

## Global failure rules

~~~text
If real longitudinal data is unavailable:
do not claim tumor-response validation.

If the mechanistic model does not beat baselines:
do not call it personalized response modeling.

If Bayesian updating does not improve held-out prediction:
do not say the twin learns.

If imaging features do not improve performance:
do not prioritize expensive imaging personalization.

If molecular features do not improve performance:
keep explanations rule-based and conservative.

If LLM outputs are unsafe:
use deterministic templates or editable draft questions only.

If treatment-comparison evidence is weak:
do not rank treatments in patient-facing mode.
~~~

## Summary

This roadmap makes OncoTwin evidence-shaped rather than architecture-shaped.

The goal is not to build every proposed component. The goal is to discover which model claims are actually supported, then build a product only around those claims.
