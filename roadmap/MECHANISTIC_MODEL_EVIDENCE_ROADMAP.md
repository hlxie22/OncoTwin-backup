# Mechanistic Model Evidence Roadmap

## Purpose

This roadmap replaces a build-first simulator plan with a model-evidence-first plan.

The hard part is not implementing an ODE solver, parameter sampler, or plotting harness. The hard part is proving that a mechanistic model can explain or predict real tumor-response measurements better than simpler alternatives.

The central question is:

~~~text
Can a mechanistic tumor-response model improve held-out prediction, uncertainty calibration,
or interpretability compared with naive and statistical baselines on real longitudinal data?
~~~

If the answer is no, the simulator should remain an educational or exploratory scenario tool. It should not be presented as a personalized digital twin.

## Core principle

Synthetic tests are required but not sufficient.

Synthetic tests can show:

~~~text
the code solves the equations correctly
the treatment exposure function behaves as expected
uncertainty intervals are calculated correctly
the updater moves in the expected direction when truth is known
the model avoids impossible numerical outputs
~~~

Synthetic tests cannot show:

~~~text
clinical usefulness
real-world prediction quality
generalization
personalized response validity
biological correctness of fitted parameters
~~~

Passing synthetic tests only permits real-data evaluation.

## Roadmap overview

~~~text
0. Define the modeling question and success metrics
1. Assemble real longitudinal tumor-response data
2. Build the baseline leaderboard
3. Define the simplest mechanistic candidate
4. Run synthetic code and identifiability checks
5. Evaluate the volume-only model on real data
6. Test Bayesian updating value
7. Study identifiability and parameter meaning
8. Add complexity only if earned
9. Decide allowed product claims
10. Promote or downgrade the simulator
~~~

## Phase 0: Define the modeling question and success metrics

### Goal

Make the modeling target precise before implementing or tuning the simulator.

### Primary modeling task

~~~text
Given baseline tumor volume, treatment category or schedule, subtype if available,
and optional early follow-up measurements, predict future tumor volume with uncertainty.
~~~

### Candidate prediction targets

~~~text
future tumor volume at next observed timepoint
percent volume change from baseline
volume trajectory across treatment-relative days
probability of crossing a research residual-burden threshold
uncertainty interval coverage for observed follow-up volumes
~~~

### Candidate update target

~~~text
Given T0 alone versus T0 + T1, predict T2/T3.
Measure whether the update improves held-out prediction and calibration.
~~~

### Metrics

~~~text
mean absolute error in tumor volume
relative volume error
percent-change error
negative log likelihood
50%, 80%, and 95% interval coverage
calibration curve
sharpness of uncertainty intervals
posterior collapse rate
performance by subtype/regimen/data-source subgroup
~~~

### Exit criteria

~~~text
The prediction target is written down.
The update target is written down.
Metrics are selected before model tuning.
The minimum useful performance threshold is defined.
Baselines to beat are listed.
~~~

## Phase 1: Assemble real longitudinal tumor-response data

### Goal

Create a modeling dataset before trusting simulator behavior.

### Required manifest fields

~~~text
case_id
dataset_source
timepoint_id
day_relative_to_treatment_start
tumor_volume_ml
longest_diameter_cm
measurement_source
measurement_confidence
subtype
er_status
pr_status
her2_status
treatment_category
treatment_schedule_detail_if_available
outcome_label_if_available
missingness_flags
quality_flags
~~~

### Minimum viable real-data task

~~~text
At least two tumor measurements per case:
T0 baseline and T1 follow-up.

Preferred:
T0, T1, T2, and T3 with treatment-relative timing.
~~~

### Data checks

~~~text
measurement timepoints are ordered
volumes are positive
treatment-relative days are available or inferable
subtype labels are usable
treatment categories are consistent
missingness is explicit
cases can be split without leakage
~~~

### Gate

~~~text
If real longitudinal tumor measurements cannot be assembled,
the mechanistic model cannot be validated.
Use synthetic/demo mode only.
~~~

## Phase 2: Build the baseline leaderboard

### Goal

Establish what simple methods can already do.

The mechanistic model must beat these baselines or provide better calibrated uncertainty.

### Required baselines

~~~text
No-change baseline:
future volume = baseline volume

Last-observation-carried-forward:
future volume = most recent measured volume

Linear trend baseline:
fit linear trend through available measurements

Population-average curve:
average response trajectory from training data

Subtype/regimen-average curve:
average trajectory conditioned on subtype and treatment category

Simple statistical model:
regularized regression or mixed-effects model predicting volume change

Bayesian hierarchical baseline:
population + subgroup-level response distributions with uncertainty
~~~

### Leaderboard outputs

~~~text
prediction error by horizon
negative log likelihood
interval coverage
calibration
subgroup performance
failure-case examples
~~~

### Gate

~~~text
The mechanistic model cannot be considered useful until it is compared against this leaderboard.
~~~

## Phase 3: Define the simplest mechanistic candidate

### Goal

Start with the smallest mechanistic model that can test the project thesis.

### Initial model

~~~text
Volume-only tumor model with:
growth term
treatment-effect term
optional resistant or residual component
observation noise
broad priors
~~~

### Candidate equation

~~~text
dV/dt = growth(V, theta_growth) - treatment_effect(t, theta_treatment) * V
~~~

Use this only as a modeling candidate, not as a product claim.

### Parameter discipline

Use the fewest parameters possible.

Candidate parameters:

~~~text
effective growth rate
effective treatment sensitivity
residual or resistant fraction
observation noise
~~~

Avoid early overparameterization.

Do not separately estimate parameters that real measurements cannot identify.

### Output

~~~text
predicted volume distribution
trajectory uncertainty intervals
parameter uncertainty
identifiability warnings
prior-dominated parameter flags
~~~

## Phase 4: Synthetic code and identifiability checks

### Goal

Use synthetic tests only to verify math, code, and known-truth behavior.

### Synthetic tests should answer

~~~text
Does the ODE solver behave correctly?
Does exposure turn on and decay correctly?
Do uncertainty intervals compute correctly?
Does the model avoid negative volume?
Does parameter recovery work in an idealized known-truth setting?
Does sparse data produce uncertainty?
~~~

### Synthetic tests should not answer

~~~text
Does the model predict real response?
Is the model clinically meaningful?
Are parameters biologically valid?
Does the model generalize?
~~~

### Required synthetic experiments

~~~text
no-treatment growth sanity check
strong-response synthetic case
weak-response synthetic case
residual/resistant synthetic case
high-noise synthetic case
sparse-observation synthetic case
contradictory-observation synthetic case
known-parameter recovery
practical non-identifiability example
~~~

### Gate

~~~text
Passing this phase only allows real-data evaluation.
It does not validate the model.
~~~

## Phase 5: Evaluate the volume-only model on real data

### Goal

Determine whether the volume-only mechanistic model adds value on real measurements.

### Evaluation design

Use train/validation/test splits.

Example tasks:

~~~text
Use T0 to predict T1.
Use T0 to predict T2.
Use T0 + T1 to predict T2.
Use T0 + T1 to predict T3.
~~~

### Compare against

~~~text
no-change baseline
last-observation-carried-forward
linear trend baseline
population-average response curve
subtype/regimen-average response curve
simple statistical model
Bayesian hierarchical baseline
~~~

### Metrics

~~~text
MAE
relative error
percent-change error
negative log likelihood
prediction interval coverage
calibration
subgroup performance
~~~

### Pass criteria

The model should pass only if it does at least one of:

~~~text
improves held-out prediction over strong baselines
improves negative log likelihood
improves uncertainty calibration
provides useful uncertainty flags without false precision
performs better in an important subgroup with enough data
~~~

### Failure response

~~~text
Do not call it personalized tumor-response modeling.
Keep it as educational scenario visualization or observation overlay.
Reduce parameters or return to baseline/statistical models.
~~~

## Phase 6: Test Bayesian updating value

### Goal

Test whether the model actually becomes more useful when new observations arrive.

### Core experiment

~~~text
Compare:
Prediction using T0 only
versus
Prediction using T0 + T1

Target:
T2 or T3 held-out measurements
~~~

### Metrics

~~~text
prediction error improvement
negative log likelihood improvement
interval coverage before and after update
effective sample size
posterior collapse frequency
overconfident wrong prediction frequency
~~~

### Pass criteria

~~~text
Adding early measurements improves later prediction or uncertainty calibration.
The posterior does not collapse under noisy measurements.
The model remains uncertain when observations are sparse or contradictory.
~~~

### Failure response

~~~text
Do not say the twin learns.
Use language such as:
new observation overlay
scenario recalibration
updated uncertainty display

Avoid:
the twin has learned your tumor
personalized adaptive prediction
~~~

## Phase 7: Study identifiability and parameter meaning

### Goal

Determine whether fitted parameters have meaningful support from the data.

### Questions

~~~text
Can growth rate and treatment sensitivity be distinguished?
Can resistant fraction be identified from available timepoints?
Is observation noise absorbing biological variation?
Are parameters stable across folds?
Are parameter estimates correlated or interchangeable?
Are posterior intervals broad enough?
~~~

### Required outputs

~~~text
parameter correlation report
profile likelihood or posterior sensitivity report
prior-vs-posterior comparison
identifiability flags
parameter-ablation study
simplified-parameter recommendation
~~~

### Gate

~~~text
If parameters are weakly identified,
do not expose patient-specific parameter explanations.
Show trajectory-level uncertainty instead.
~~~

## Phase 8: Add complexity only if earned

### Goal

Prevent complexity from hiding weak modeling.

### Complexity ladder

~~~text
Level 0: no-change baseline
Level 1: population-average curve
Level 2: subtype/regimen-average curve
Level 3: statistical response model
Level 4: volume-only mechanistic model
Level 5: volume-only Bayesian update model
Level 6: imaging-feature-informed prior
Level 7: pathology/molecular-informed prior
Level 8: regional model
Level 9: spatial model
Level 10: learned AI parameter amortizer
~~~

### Rule

~~~text
Move up the ladder only if the new level improves held-out real-data performance,
uncertainty calibration, identifiability, or interpretability over the previous level.
~~~

### Examples

~~~text
If imaging features do not improve prediction,
do not add spatial MRI modeling yet.

If pathology/molecular features do not improve priors,
keep molecular explanations rule-based.

If volume-only Bayesian updating fails,
do not build a spatial updating system.
~~~

## Phase 9: Decide allowed product claims

### Goal

Translate evidence into product language.

### Claim levels

~~~text
Level 0:
Educational simulation only.

Level 1:
Research trajectory visualization from manual measurements.

Level 2:
Model-estimated tumor-volume range with uncertainty.

Level 3:
Updated trajectory after follow-up measurement.

Level 4:
Evidence-supported imaging-informed model.

Level 5:
Evidence-supported personalized prior or parameter explanation.
~~~

### Forbidden unless externally validated

~~~text
guaranteed prediction
clinical treatment recommendation
treatment will work
treatment will cure
your cancer is gone
clinically validated pCR prediction
best treatment for you
~~~

### Gate

~~~text
No product language should exceed the evidence level reached by the model.
~~~

## Phase 10: Promote or downgrade the simulator

### Promote if

~~~text
real-data evaluation beats relevant baselines
uncertainty is calibrated enough for research use
Bayesian updating improves future prediction
identifiability limitations are explicit
failure modes are documented
allowed claims are conservative
~~~

### Downgrade if

~~~text
the model does not beat baselines
uncertainty is poorly calibrated
parameters are not identifiable
updates create false precision
performance is subgroup-fragile
~~~

### Promotion target

Only after evidence gates pass should reusable code move toward:

~~~text
src/oncotwin/simulation/
~~~

Before that, the simulator should remain experimental.

## Recommended repository placement

Use this roadmap as the main mechanistic modeling plan:

~~~text
roadmap/MECHANISTIC_MODEL_EVIDENCE_ROADMAP.md
~~~

Keep engineering experiments separate:

~~~text
experiments/mechanistic_model/
fixtures/mechanistic_model/
reports/mechanistic_model/
tests/mechanistic_model/
~~~

Suggested reports:

~~~text
reports/mechanistic_model/baseline_leaderboard.md
reports/mechanistic_model/real_data_evaluation.md
reports/mechanistic_model/update_value_report.md
reports/mechanistic_model/identifiability_report.md
reports/mechanistic_model/claims_status.md
~~~

## First 10 tasks

~~~text
1. Define the primary prediction task.
2. Define the update prediction task.
3. Assemble a small real-data measurement manifest.
4. Implement no-change and last-observation baselines.
5. Implement population-average and subtype/regimen-average baselines.
6. Build the baseline leaderboard report.
7. Implement the simplest volume-only mechanistic candidate.
8. Run synthetic code and known-truth checks.
9. Evaluate the volume-only model against the leaderboard on real data.
10. Decide whether the model earns the right to support a product claim.
~~~

## Summary

The mechanistic model should not be judged by whether it can generate plausible-looking curves.

It should be judged by whether it improves real-data prediction, calibration, updating, or interpretability compared with simpler baselines.

If it cannot, that is not an engineering failure. It is a modeling result, and the product should adapt accordingly.
