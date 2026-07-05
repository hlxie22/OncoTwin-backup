# Mechanistic Simulator Implementation, Experimentation, and Testing Plan

## Purpose

This document turns the mechanistic simulator roadmap phase into an executable engineering plan.

The goal is not to build the full spatial MRI-backed tumor simulator first. The goal is to prove, with small components and disposable experiments, that a simple mechanistic simulator can produce plausible, stable, explainable tumor-volume trajectories before the app depends on it.

The simulator should start as a research harness, not a product service. It should only become part of the durable backend after it passes basic numerical, biological-plausibility, uncertainty, and identifiability checks.

## Placement in the repository

Place this document at:

```text
roadmap/MECHANISTIC_SIMULATOR_PLAN.md
```

During implementation, use this structure:

```text
experiments/
  mechanistic_simulator/
    README.md
    run_v0_single_trajectory.py
    run_v0_ensemble.py
    run_v0_synthetic_fit.py
    run_v0_identifiability.py
    run_v0_stress_tests.py
    outputs/

fixtures/
  mechanistic_simulator/
    cases/
      hr_positive_demo_case.json
      her2_positive_demo_case.json
      tnbc_demo_case.json
      longitudinal_measurement_demo_case.json
    schedules/
      endocrine_schedule.json
      her2_directed_schedule.json
      chemotherapy_schedule.json
    params/
      generic_volume_prior.json
      high_response_params.json
      weak_response_params.json
      resistant_disease_params.json

schemas/
  mechanistic_simulator/
    tumor_measurement.schema.json
    treatment_schedule.schema.json
    mechanistic_params.schema.json
    simulation_output.schema.json

tests/
  mechanistic_simulator/
    test_volume_ode.py
    test_drug_exposure.py
    test_parameter_sampling.py
    test_trajectory_summary.py
    test_synthetic_recovery.py
    test_identifiability_flags.py
    test_safety_language.py
```

Once the simulator passes the feasibility gates, promote the reusable code into a durable package such as:

```text
src/
  oncotwin/
    simulation/
      __init__.py
      types.py
      exposure.py
      volume_ode.py
      ensemble.py
      summarize.py
      calibration.py
      identifiability.py
      warnings.py
```

Until then, keep the implementation in `experiments/mechanistic_simulator/` so the team can change or discard it without breaking product code.

## Strategy

Build the simulator in layers:

```text
v0: volume-only ODE simulator
v1: volume-only ensemble simulator with uncertainty bands
v2: synthetic fitting and parameter recovery
v3: practical identifiability checks
v4: Bayesian-update-ready particle output
v5: regional or spatial simulator spike, only after v0-v4 pass
```

The first milestone is not "a digital twin." The first milestone is:

```text
Given a baseline tumor volume, a treatment schedule, and sampled parameters,
produce plausible tumor-volume trajectories with uncertainty summaries,
clear assumptions, and warnings when the model is underdetermined.
```

## Non-goals for the first simulator phase

Do not build these first:

```text
MRI segmentation integration
voxelwise PDE solver
spatial residual-risk maps
AI-personalized priors
molecular graph parameter modifiers
clinical pCR prediction
treatment recommendation ranking
patient-facing medical language
```

Those depend on the simulator behaving sensibly at the simpler volume-only level.

## Core model for v0

Start with a volume-only ODE:

```text
dV/dt = rV(1 - V/K) - kill(t) * V
```

Where:

```text
V = tumor volume in mL
r = effective growth rate
K = effective carrying capacity
kill(t) = treatment effect at time t
```

Use a simple drug exposure curve:

```text
C_drug(t) = sum over dose events:
  relative_dose * exp(-decay_rate * (t - dose_day))
  for t >= dose_day
```

Then:

```text
kill(t) = sum over drugs:
  sensitivity_drug * C_drug(t) / (C_drug(t) + EC50_drug)
```

For v0, keep the model intentionally small:

```text
free parameters:
  growth_rate
  treatment_sensitivity
  resistant_fraction
  observation_noise
```

Avoid fitting too many parameters from sparse longitudinal measurements. The first implementation should make underidentification obvious rather than hiding it.

## Data contracts

### Tumor measurement

```json
{
  "case_id": "demo_tnbc_001",
  "day": 0,
  "tumor_volume_ml": 18.4,
  "longest_diameter_cm": 3.2,
  "source": "manual",
  "confidence": "demo"
}
```

Required fields:

```text
case_id
day
tumor_volume_ml OR longest_diameter_cm
source
confidence
```

Allowed `source` values:

```text
manual
metadata
mask_derived
synthetic
demo
```

### Treatment schedule

```json
{
  "schedule_id": "demo_chemotherapy_schedule",
  "regimen_name": "Demo chemotherapy schedule",
  "total_duration_days": 168,
  "events": [
    {
      "drug": "anthracycline",
      "day": 0,
      "relative_dose": 1.0,
      "label": "Cycle 1"
    },
    {
      "drug": "taxane",
      "day": 84,
      "relative_dose": 1.0,
      "label": "Taxane start"
    }
  ]
}
```

Required fields:

```text
schedule_id
regimen_name
total_duration_days
events
```

Each event requires:

```text
drug
day
relative_dose
```

### Mechanistic parameters

```json
{
  "growth_law": "logistic",
  "growth_rate": 0.006,
  "carrying_capacity_ml": 250.0,
  "drug_sensitivity": {
    "anthracycline": 0.08,
    "taxane": 0.07
  },
  "drug_ec50": {
    "anthracycline": 0.5,
    "taxane": 0.5
  },
  "drug_decay": {
    "anthracycline": 0.25,
    "taxane": 0.2
  },
  "resistant_fraction": 0.08,
  "resistant_sensitivity_scale": 0.1,
  "observation_noise_fraction": 0.12
}
```

### Simulation output

```json
{
  "case_id": "demo_tnbc_001",
  "simulation_version": "volume_ode_v0",
  "times": [0, 7, 14, 21],
  "median_volume_ml": [18.4, 17.1, 15.3, 13.8],
  "interval_80_volume_ml": [
    [16.5, 20.6],
    [14.8, 19.7],
    [12.1, 18.4],
    [10.2, 17.9]
  ],
  "n_particles": 1000,
  "warnings": [
    "Demo-only simulation.",
    "Parameters are not clinically validated."
  ],
  "driver_summary": {
    "dominant_factors": [
      "treatment_sensitivity",
      "resistant_fraction"
    ],
    "prior_dominated_parameters": [
      "carrying_capacity_ml"
    ]
  }
}
```

## Implementation plan

### Step 1: Create simulator fixtures

Create four small demo cases:

```text
HR-positive / HER2-negative case
HER2-positive case
TNBC chemotherapy case
Longitudinal tumor-measurement case with T0/T1/T2/T3 volumes
```

Each case should include:

```text
case_id
subtype
baseline tumor volume
optional longest diameter
treatment schedule reference
expected qualitative behavior
```

Example expected behaviors:

```text
strong responder: volume decreases steadily
weak responder: volume decreases slowly or plateaus
resistant case: early decrease followed by residual burden
no-treatment control: volume grows toward carrying capacity
```

The expected behavior should be qualitative at first. Do not overfit numeric values before the simulator exists.

### Step 2: Implement treatment exposure

Implement:

```text
compute_exposure(schedule, params, times)
```

It should return exposure by drug and time.

Minimum behavior:

```text
before first dose: exposure is zero
at dose day: exposure increases
after dose day: exposure decays
multiple dose events: exposure accumulates
relative_dose = 0: no exposure contribution
```

This should be tested independently before it is connected to tumor-volume dynamics.

### Step 3: Implement single-trajectory volume simulation

Implement:

```text
simulate_volume_trajectory(
  initial_volume_ml,
  treatment_schedule,
  params,
  output_days,
  dt_days
)
```

Minimum output:

```text
day
tumor_volume_ml
total_exposure_by_drug
growth_component
kill_component
warnings
```

Start with a simple numerical integrator. For the v0 ODE, fixed-step integration is acceptable if the tests verify stability across smaller `dt` values.

Do not add spatial diffusion yet.

### Step 4: Add biological and numerical guards

The simulator should fail loudly or emit warnings when inputs are unsafe or misleading.

Add guards for:

```text
negative initial volume
zero or negative carrying capacity
growth rate outside allowed range
drug sensitivity outside allowed range
negative relative dose
output days beyond schedule duration without explicit permission
volume becoming negative
volume exceeding carrying capacity by an implausible amount
dt too large for stable behavior
```

The simulator should never silently emit impossible trajectories.

### Step 5: Add parameter sampling

Implement:

```text
sample_volume_params(prior_config, n_particles, seed)
```

For the first version, use simple bounded distributions:

```text
growth_rate
treatment_sensitivity by regimen category
resistant_fraction
observation_noise_fraction
```

The sampler must be deterministic when given a seed.

The output should preserve each particle's parameter values so Bayesian updating can later reweight them.

### Step 6: Add ensemble simulation

Implement:

```text
simulate_volume_ensemble(
  initial_volume_ml,
  treatment_schedule,
  parameter_particles,
  output_days,
  dt_days
)
```

Output:

```text
all particle trajectories
median trajectory
80% interval
95% interval
final residual-burden distribution
uncertainty score
warnings
```

Keep particle-level output available for the Bayesian updater. Do not only return summary curves.

### Step 7: Add trajectory summarization

Implement:

```text
summarize_trajectories(particle_trajectories)
```

The summary should include:

```text
median volume by day
80% interval by day
95% interval by day
probability final volume is below research threshold
uncertainty width by day
maximum uncertainty day
driver summary
warnings
```

The output should avoid clinical claims. Use labels such as:

```text
research residual-burden threshold
model-estimated low residual burden
simulation uncertainty
```

Do not use:

```text
cure
guaranteed response
clinical pCR prediction
best treatment
should take this regimen
```

### Step 8: Add synthetic fitting

Implement a simple fitting harness:

```text
run_v0_synthetic_fit.py
```

It should:

```text
choose known parameters
simulate a synthetic ground-truth trajectory
add observation noise
fit or reweight candidate parameters
compare recovered parameters to known parameters
report which parameters were recoverable
```

This is not full Bayesian updating yet. It is a sanity check that the model can learn something from simple observations.

### Step 9: Add identifiability experiment

Implement:

```text
run_v0_identifiability.py
```

It should test whether different parameter combinations can produce nearly indistinguishable tumor-volume trajectories.

The output should report:

```text
parameters that are constrained by the observations
parameters that are weakly constrained
parameters that are prior-dominated
trajectory pairs that look similar despite different parameters
recommendation to reduce free parameters when needed
```

This is a key gate. If many parameter sets produce the same trajectory, the app must not pretend those parameters are known.

### Step 10: Add Bayesian-update-ready output

Before implementing the Bayesian updater itself, make simulator output compatible with it.

Each particle trajectory should include:

```text
particle_id
parameter values
predicted volume at each output day
predicted longest diameter if available
likelihood placeholder
weight placeholder
warnings
```

This makes Phase 4 easier and prevents rework.

## Experimentation plan

### Experiment 1: No-treatment growth sanity check

Purpose:

```text
Confirm that the model grows monotonically toward carrying capacity when no treatment is applied.
```

Input:

```text
baseline volume
growth_rate
carrying_capacity
empty treatment schedule
```

Pass criteria:

```text
volume never becomes negative
volume increases when below carrying capacity
volume approaches carrying capacity without exploding
smaller dt produces similar trajectory
```

Failure response:

```text
fix numerical integration or parameter bounds before continuing
```

### Experiment 2: Exposure sanity check

Purpose:

```text
Confirm that treatment exposure behaves correctly.
```

Pass criteria:

```text
exposure is zero before first dose
exposure jumps or rises after a dose event
exposure decays over time
repeated doses accumulate
zero-dose events do not affect exposure
```

Failure response:

```text
do not connect exposure to tumor dynamics until exposure is correct
```

### Experiment 3: Strong-response trajectory

Purpose:

```text
Confirm that high treatment sensitivity can produce tumor shrinkage.
```

Pass criteria:

```text
tumor volume decreases after treatment exposure
volume does not become negative
resistant fraction prevents impossible total disappearance unless explicitly allowed
trajectory shape is smooth enough for app display
```

Failure response:

```text
inspect kill term, resistant fraction handling, and numerical step size
```

### Experiment 4: Weak-response trajectory

Purpose:

```text
Confirm that low treatment sensitivity produces plateau or slow shrinkage.
```

Pass criteria:

```text
trajectory differs clearly from strong-response case
model can represent partial response or stable disease
uncertainty remains visible
```

Failure response:

```text
adjust prior ranges and sensitivity scale
```

### Experiment 5: Resistant-disease trajectory

Purpose:

```text
Confirm that resistant fraction creates residual burden.
```

Pass criteria:

```text
sensitive component shrinks more than resistant component
final volume remains above residual threshold in a meaningful subset of particles
driver summary identifies resistant fraction as influential
```

Failure response:

```text
separate sensitive and resistant effects more clearly, or defer resistant modeling if v0 is too complex
```

### Experiment 6: Ensemble uncertainty

Purpose:

```text
Confirm that sampled parameters produce uncertainty bands.
```

Pass criteria:

```text
narrow priors produce narrower bands
broad priors produce wider bands
uncertainty increases when observations are sparse
summary intervals are ordered correctly: lower <= median <= upper
```

Failure response:

```text
fix parameter sampling, summarization, or prior ranges
```

### Experiment 7: Synthetic parameter recovery

Purpose:

```text
Confirm that simple longitudinal observations can shift parameter estimates in the right direction.
```

Pass criteria:

```text
known strong-response parameters are recovered as high-sensitivity or low-resistant-fraction cases
known weak-response parameters are recovered as low-sensitivity or higher-resistant-fraction cases
posterior or fitted trajectories are closer to observations than generic prior trajectories
```

Failure response:

```text
reduce number of free parameters
increase observation noise honestly
document that parameters are weakly identifiable
```

### Experiment 8: Practical identifiability

Purpose:

```text
Detect whether several parameter combinations can explain the same observations.
```

Pass criteria:

```text
the report identifies at least which parameters are constrained and which are prior-dominated
the simulator does not emit false precision for weakly constrained parameters
```

Failure response:

```text
remove weakly identifiable parameters from fitting
fix some parameters to curated priors
broaden uncertainty intervals
```

### Experiment 9: Stress-test parameter ranges

Purpose:

```text
Make sure bad parameter values do not produce polished nonsense.
```

Test:

```text
very high growth
very high kill
very high resistant fraction
very low carrying capacity
large dt
missing drug decay
unknown drug category
```

Pass criteria:

```text
invalid inputs fail validation
dangerous-but-allowed inputs produce warnings
trajectories remain finite
no negative volumes
no hidden clipping without warning
```

Failure response:

```text
add validation and warnings before product integration
```

### Experiment 10: App-readiness dry run

Purpose:

```text
Confirm that the simulator can produce a product-facing summary without the full app.
```

Input:

```text
one demo case
one treatment schedule
one parameter prior
one output-day grid
```

Output:

```text
trajectory JSON
summary JSON
warnings JSON
optional static plot
```

Pass criteria:

```text
a future frontend could render the trajectory chart from the JSON
warnings are clear
no unsafe patient-facing language is present
particle-level output is preserved for Bayesian updating
```

Failure response:

```text
stabilize output schema before adding backend API
```

## Testing plan

### Unit tests

#### Drug exposure tests

Test file:

```text
tests/mechanistic_simulator/test_drug_exposure.py
```

Cases:

```text
exposure is zero before first dose
single dose decays over time
multiple doses accumulate
relative dose scales exposure
unknown drug raises validation error or warning
negative dose fails validation
```

#### ODE dynamics tests

Test file:

```text
tests/mechanistic_simulator/test_volume_ode.py
```

Cases:

```text
no-treatment volume grows toward carrying capacity
zero growth and zero treatment keeps volume constant
high treatment sensitivity causes shrinkage
volume never becomes negative
volume remains finite
smaller dt gives similar result
```

#### Parameter sampling tests

Test file:

```text
tests/mechanistic_simulator/test_parameter_sampling.py
```

Cases:

```text
sampler returns requested number of particles
same seed gives same particles
different seed gives different particles
all sampled values are inside bounds
required parameters are present
```

#### Ensemble summary tests

Test file:

```text
tests/mechanistic_simulator/test_trajectory_summary.py
```

Cases:

```text
median is computed correctly
80% interval contains median
95% interval contains 80% interval
uncertainty score increases for wider trajectories
summary preserves warning messages
```

#### Synthetic recovery tests

Test file:

```text
tests/mechanistic_simulator/test_synthetic_recovery.py
```

Cases:

```text
strong-response synthetic data favors high-sensitivity particles
weak-response synthetic data favors low-sensitivity particles
noisy observations do not collapse uncertainty too aggressively
```

#### Identifiability warning tests

Test file:

```text
tests/mechanistic_simulator/test_identifiability_flags.py
```

Cases:

```text
sparse observations mark parameters as prior-dominated
more observations reduce but do not eliminate uncertainty
similar trajectories from different parameters trigger identifiability warning
```

#### Safety-language tests

Test file:

```text
tests/mechanistic_simulator/test_safety_language.py
```

Reject output containing:

```text
will cure
guaranteed
best treatment
you should take
clinical pCR prediction
cancer is gone
```

Allow output containing:

```text
exploratory simulation
uncertain estimate
model-estimated
research threshold
discuss with care team
```

### Integration tests

Add one integration test that runs:

```text
fixture case
fixture treatment schedule
sampled parameter prior
ensemble simulation
trajectory summary
JSON schema validation
```

Expected result:

```text
valid simulation output
valid summary output
warnings included
particle trajectories preserved
no unsafe language
```

### Regression tests

For each demo case, save a small expected-output snapshot containing broad ranges, not exact floating-point trajectories.

Do not assert exact values unless the numerical method is fully deterministic and stable. Prefer assertions like:

```text
final median volume is lower than baseline for strong-response case
final median volume is near baseline or higher for weak-response case
80% interval width is greater than zero
no particle has negative volume
```

## Pass/fail gates

### Gate 1: v0 single trajectory

Pass if:

```text
single trajectories behave plausibly for no-treatment, strong-response, weak-response, and resistant cases
drug exposure tests pass
volume never becomes negative
outputs are finite
```

Fail means:

```text
do not build ensemble simulation yet
fix ODE or exposure first
```

### Gate 2: ensemble uncertainty

Pass if:

```text
parameter sampling works
uncertainty bands are sensible
broad priors produce broader uncertainty than narrow priors
summary output validates against schema
```

Fail means:

```text
do not connect to Bayesian update yet
fix priors, sampling, or summaries
```

### Gate 3: synthetic recovery

Pass if:

```text
synthetic observations move estimates in the expected direction
posterior/fitted trajectories improve over generic prior trajectories
uncertainty remains when observations are sparse or noisy
```

Fail means:

```text
reduce free parameters
widen uncertainty
avoid personalized claims
```

### Gate 4: identifiability

Pass if:

```text
the simulator can identify which parameters are data-informed versus prior-dominated
driver summaries do not overstate weakly identified parameters
```

Fail means:

```text
do not expose parameter-level explanations
show only trajectory-level uncertainty
```

### Gate 5: app-readiness

Pass if:

```text
future frontend can render JSON output
future Bayesian updater can reweight particle output
warnings and assumptions are explicit
no unsafe language appears in simulator outputs
```

Fail means:

```text
do not productize simulator service yet
keep it in experiments
```

## First 10 implementation tasks

1. Create the simulator plan document:

```text
roadmap/MECHANISTIC_SIMULATOR_PLAN.md
```

2. Create experiment folders:

```text
experiments/mechanistic_simulator/
fixtures/mechanistic_simulator/
schemas/mechanistic_simulator/
tests/mechanistic_simulator/
```

3. Add JSON fixtures for:

```text
one baseline tumor measurement
one longitudinal tumor-measurement case
one simple treatment schedule
one generic parameter-prior config
```

4. Add JSON schemas for:

```text
tumor measurement
treatment schedule
mechanistic params
simulation output
```

5. Implement and test `compute_exposure`.

6. Implement and test a no-treatment logistic volume trajectory.

7. Add treatment kill term and test strong-response versus weak-response behavior.

8. Add parameter sampling with deterministic seeds.

9. Add ensemble simulation and trajectory summarization.

10. Add the first experiment runner:

```text
experiments/mechanistic_simulator/run_v0_ensemble.py
```

It should load a fixture case, sample parameters, run the ensemble, write JSON output, and save a simple plot to:

```text
experiments/mechanistic_simulator/outputs/
```

## Promotion criteria

Keep simulator code in `experiments/` until all of these are true:

```text
drug exposure tests pass
single-trajectory tests pass
ensemble summary tests pass
synthetic recovery experiment passes
identifiability report exists
output schemas are stable
unsafe-language tests pass
```

Only then promote reusable modules into:

```text
src/oncotwin/simulation/
```

At that point, the simulator can become a backend service candidate.

## Documentation updates after completion

When v0 passes, update:

```text
roadmap/IMPLEMENTATION_ROADMAP.md
docs/05_mechanistic_tumor_simulator.md
```

Add a short status note:

```text
Volume-only simulator v0 has passed synthetic feasibility checks.
It supports demo/manual tumor-volume trajectories, ensemble uncertainty,
and Bayesian-update-ready particle outputs. It is not clinically validated
and should be shown only as an exploratory research simulation.
```

If v0 fails, update the roadmap with the fallback:

```text
Volume-only simulator did not pass feasibility checks.
The app should temporarily frame tumor modeling as educational scenario ranges
or observation overlays rather than a personalized digital twin.
```

## Safety constraints

The simulator is not allowed to produce patient-facing treatment advice.

Allowed language:

```text
model-estimated trajectory
exploratory simulation
uncertain residual-burden estimate
research threshold
simulation assumption
prior-dominated parameter
```

Forbidden language:

```text
this treatment will work
this treatment will cure
you should choose this treatment
your cancer is gone
clinical pCR prediction
guaranteed response
```

Any treatment comparison, ranking, or recommendation-style output must be handled by the Scenario Lab or clinical/research-facing layer with explicit uncertainty and disclaimer language.

## Expected outcome

At the end of this phase, the team should know whether the simplest simulator is worth building on.

A successful outcome means:

```text
the simulator can produce plausible volume trajectories
uncertainty behaves sensibly
sparse data does not create false precision
particle outputs are ready for Bayesian updating
the model's limitations are explicit
```

An unsuccessful outcome is still useful. It means the product should avoid personalized digital-twin claims and instead focus on tracking, education, observation overlays, and carefully labeled research scenarios until the modeling improves.
