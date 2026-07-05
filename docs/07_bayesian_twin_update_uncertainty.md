# Bayesian Twin Updating and Uncertainty

## Purpose

The Bayesian update loop is what makes OncoTwin a real digital twin instead of a one-time simulation. The twin begins with a prior distribution over parameters, simulates possible tumor trajectories, and updates that distribution whenever new observations arrive.

New observations may include:

```text
early-treatment MRI
mid-treatment MRI
presurgery MRI
tumor size from report
pathology update
BRCA/HRD result
actual treatment delay or dose modification
patient-reported symptoms/tolerance
```

## Core update rule

```math
p(\theta \mid y_{1:t}) \propto p(y_t \mid \theta)p(\theta \mid y_{1:t-1})
```

Where:

```text
θ = patient-specific mechanistic parameters
y_t = new observation at time t
p(θ | y_1:t-1) = previous twin belief
p(θ | y_1:t) = updated twin belief
```

## Particle representation

Maintain an ensemble of possible twins.

```typescript
type TwinParticle = {
  particleId: string;
  params: MechanisticParams;
  weight: number;
  trajectorySummary?: TumorSimulationOutput;
  residualRiskMapUri?: string;
  toxicityTrajectory?: ToxicityTrajectory;
};
```

A posterior is a weighted particle set:

```typescript
type TwinPosterior = {
  particles: TwinParticle[];
  sourceObservations: string[];
  createdAt: string;
  uncertaintySummary: UncertaintySummary;
};
```

## Initialization

At baseline:

```text
1. AI parameter amortizer outputs parameter distributions.
2. Sample N parameter particles.
3. Assign equal weights.
4. Simulate each particle forward under the current treatment context.
5. Summarize median trajectory and uncertainty band.
```

## Observation likelihoods

Different observations require different likelihood functions.

### Tumor volume observation

If observed tumor volume is available:

```math
p(y_t \mid \theta) \propto \exp\left(-\frac{(V_{obs}(t)-V_{pred}(t;\theta))^2}{2\sigma_V^2}\right)
```

### Longest diameter observation

If only diameter is available, compare diameter or convert to approximate volume with high uncertainty.

```text
report-only measurement → larger σ
MRI-derived volume → smaller σ
low-confidence segmentation → larger σ
```

### Spatial mask observation

If a follow-up tumor mask exists, compare predicted residual-risk map with observed residual mask.

Possible losses:

```text
Dice-like overlap
voxelwise cross-entropy
centroid distance
regional volume mismatch
```

### Biomarker observation

If a new molecular result arrives, update parameter priors or molecular graph modifiers.

Example:

```text
BRCA/HRD positive → shift DNA-repair sensitivity prior
BRCA/HRD negative → reduce uncertainty around DNA-repair sensitivity modifier
```

### Treatment event observation

If treatment was delayed or modified, update the schedule and resimulate.

```text
new schedule does not directly update biology
but it changes future simulated exposure and outcomes
```

### Symptom/tolerance observation

Symptoms should primarily update the person-burden/toxicity twin, not the tumor-response parameters, unless the symptom implies treatment interruption or dose change.

The patient-facing LLM co-pilot uses these symptom observations differently from the Bayesian tumor update. It can select daily check-in questions, explain symptom trends, generate care-team questions, and draft visit summaries from structured logs and deterministic trend flags. It must not treat symptoms as evidence that the tumor is responding or progressing.

## Update algorithm

The update is **batch importance sampling from the amortizer prior**, not a sequential particle filter.

This choice is driven by the structure of the problem:

```text
parameters θ are static biology, not a time-evolving state
  (proliferation rate does not random-walk between scans)
observations are few — typically 3–4 MRI timepoints over the
  whole neoadjuvant course (T0/T1/T2/T3; see 03)
every update re-simulates each particle anyway
```

Under these conditions a sequential reweight → resample → jitter loop buys nothing: it pays the full re-simulation cost on every update *and* accumulates particle impoverishment (the cloud collapses to duplicates after repeated resampling), which then has to be patched with parameter-noise jitter that distorts the posterior. Because the parameters are static, sequentially compounding weights over observations is mathematically equivalent to weighting once by the joint likelihood — so we do the latter, statelessly, from the original prior particles each time a new observation arrives.

```python
def update_posterior(prior_particles, observations):
    # observations = ALL observations to date (T0..T_now), not just the newest.
    # prior_particles come from the amortizer; we always re-weight from the
    # prior, never from a previously resampled set, so there is no
    # impoverishment to accumulate.
    weighted = []

    for particle in prior_particles:
        trajectory = simulate_full_course(particle)
        log_likelihood = sum(
            compute_log_likelihood(trajectory, obs) for obs in observations
        )
        weighted.append({**particle, "log_weight": particle.log_prior_weight + log_likelihood})

    normalized = normalize_log_weights(weighted)

    if effective_sample_size(normalized) < threshold:
        # Prior is far from the posterior (e.g. data contradict the amortizer).
        # Escalate to a tempered SMC sampler with MCMC move steps, which
        # regenerates genuine diversity instead of faking it with jitter.
        normalized = smc_sampler_with_tempering(prior_particles, observations)

    return summarize_posterior(normalized)
```

Biomarker and treatment-event observations are not reweights (see above): a biomarker result shifts the prior, so it re-primes `prior_particles` from the amortizer before the volume/mask observations are weighted in; a treatment event only changes the schedule fed to `simulate_full_course`.

## Effective sample size and the SMC fallback

```math
ESS = \frac{1}{\sum_i w_i^2}
```

A low ESS means few particles carry most of the weight — here it is a **diagnostic that the prior is far from the posterior**, not a cue to jitter. When ESS falls below threshold, escalate to a **tempered SMC sampler**: anneal from the prior to the posterior through a sequence of intermediate distributions, running an MCMC move step at each rung so particles are *moved* toward the posterior rather than duplicated. This subsumes the particle filter, regenerates real diversity, and is the principled path for the contradictory-observation case in "Handling contradictory observations" below. Reserve it for the low-ESS path; the plain batch importance-sampling weighting is the default.

A fully amortized alternative — train the SBI/NPE estimator (`06`) to condition on baseline features *plus* observations-to-date and emit the posterior in one forward pass, with no runtime simulation — is a later optimization, not the v1 path, because conditioning on variable-length irregularly-timed observations is a substantial training effort and the per-particle simulations are wanted for visualization anyway.

## Posterior summaries

The app should summarize the posterior as:

```text
median tumor-volume trajectory
80% uncertainty band
90% uncertainty band
probability of low residual burden
most likely parameter shifts
residual-risk map
uncertainty drivers
missing-data value ranking
```

## Uncertainty score

Do not use a fake confidence score. Use an uncertainty summary with multiple components.

```typescript
type UncertaintySummary = {
  trajectoryUncertainty: "low" | "moderate" | "high";
  parameterUncertainty: "low" | "moderate" | "high";
  dataQualityUncertainty: "low" | "moderate" | "high";
  missingDataUncertainty: "low" | "moderate" | "high";
  outOfDistributionWarning: boolean;
  topDrivers: string[];
};
```

## Value of information

The system should estimate what additional data would reduce uncertainty most.

Candidates:

```text
early-treatment MRI
mid-treatment MRI
BRCA/HRD status
Ki-67
HER2 confirmation
treatment schedule details
manual review of segmentation
```

Simple first version:

```text
rank missing variables based on hand-coded importance and current uncertainty
```

Advanced version:

```text
simulate expected posterior entropy reduction from adding each observation type
```

Output example:

```text
Most useful next data:
1. Early-treatment MRI: would reduce drug-sensitivity uncertainty.
2. BRCA/HRD result: would reduce DNA-repair sensitivity uncertainty.
3. Manual segmentation review: would reduce baseline tumor-volume uncertainty.
```

## Update explanation

Every update should generate a “why the twin changed” explanation.

Example:

```text
The twin shifted toward higher treatment sensitivity because the T1 MRI tumor volume was 42% lower than the median baseline simulation expected. Particles with low taxane/anthracycline sensitivity no longer matched the observed shrinkage and were downweighted.
```

Another example:

```text
The twin's uncertainty decreased because a mid-treatment MRI was added. However, molecular uncertainty remains high because BRCA/HRD status is still unknown.
```

## Handling contradictory observations

If new observations do not match any particles well:

```text
1. Raise data-quality warning.
2. Check segmentation confidence.
3. Check timepoint/treatment-date mismatch.
4. Expand posterior uncertainty.
5. Reinitialize particles around broader priors if needed.
```

UI language:

```text
The new measurement is difficult for the current twin to explain. This may reflect measurement differences, segmentation uncertainty, treatment-date mismatch, or biology not captured by the current model.
```

## API endpoint

```text
POST /cases/{case_id}/twin/update-observation
```

Request:

```json
{
  "observation_type": "tumor_volume",
  "time_days": 42,
  "value": 12.4,
  "units": "mL",
  "source": "MRI_segmentation",
  "confidence": 0.83
}
```

Response:

```json
{
  "posterior_id": "posterior_456",
  "trajectory_summary_uri": "s3://...",
  "uncertainty_summary": {
    "trajectoryUncertainty": "moderate",
    "topDrivers": ["only one follow-up scan", "BRCA/HRD unknown"]
  },
  "update_explanation": "The observed T1 tumor volume was lower than expected under low-sensitivity particles, so those particles were downweighted."
}
```

## Implementation milestones

1. Weighted particle representation.
2. Volume-observation likelihood.
3. Batch importance-sampling update from the prior over all observations.
4. ESS-gated tempered SMC-sampler fallback for low-ESS / contradictory observations.
5. Uncertainty-band computation.
6. Update explanation generator.
7. Segmentation-confidence-aware likelihoods.
8. Spatial residual-mask likelihood.
9. Value-of-information module.
10. Molecular/bio-marker update hooks.
