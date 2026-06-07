# AI Personalization and Parameter Amortizer

## Purpose

The AI personalization layer maps baseline patient data into a distribution over mechanistic model parameters. This lets the system initialize a patient-specific twin even before multiple follow-up measurements are available.

The central design principle:

> The AI does not directly replace the mechanistic simulator. It predicts the simulator's patient-specific parameter priors.

## Inputs

```text
baseline MRI volume
baseline tumor mask
imaging feature vector
pathology biomarkers
molecular markers
age and patient context
treatment context
```

## Outputs

The AI should output parameter distributions, not single point estimates.

A naive per-parameter mean/std (a diagonal Gaussian) is **insufficient** and contradicts the rest of this design: Family A below produces posteriors that are deliberately wide and often multimodal, and the parameters are not independent — under a reaction-diffusion fit `k` (proliferation) and `θ_cap` (carrying capacity), and `D` vs `k`, trade off strongly. Sampling marginals independently produces twins that are each plausible but jointly impossible. The output must therefore carry the **joint** distribution.

Choose one of the following, in increasing order of fidelity:

```text
1. Multivariate Gaussian with full covariance, emitted as a Cholesky factor.
   Cheapest upgrade; captures parameter correlations.
2. Mixture density (a few Gaussian components).
   Adds multimodality.
3. Preferred: the conditioning context of a small normalizing flow, so the
   deployed output is the SAME distribution family as the NPE posterior in
   Family A. No information is discarded between calibration and deployment.
```

Working schema (Cholesky variant; `params` is the fixed-order parameter vector
`[k, D, θ_cap, α_*, β_*, deliveryScale, heterogeneityScale]`):

```typescript
type ParameterDistribution = {
  paramOrder: string[];          // canonical ordering of the parameter vector
  mean: number[];                // in the modeling space (see Distribution head)
  choleskyL: number[][];         // lower-triangular L, covariance = L Lᵀ
  // OR, for the mixture / flow variants:
  // components?: { weight: number; mean: number[]; choleskyL: number[][] }[];
  // flowContext?: number[];     // conditioning vector for the deployed flow
};
```

The simulator samples from this **joint** distribution to create an ensemble of possible twins (see `Sampling parameter particles`).

## Architecture

```text
MRI encoder
  +
Tumor feature encoder
  +
Pathology encoder
  +
Molecular graph encoder
  +
Patient-context encoder
        ↓
Fusion transformer / MLP fusion
        ↓
Distribution head
        ↓
Mechanistic parameter distributions
```

## MRI encoder

Start with a practical 3D encoder:

```text
MedicalNet 3D ResNet baseline
or MONAI 3D DenseNet/ResNet
or custom DCE-MRI CNN/ViT trained on BreastDCEDL
```

Input options:

```text
tumor crop only
whole breast crop
multi-channel DCE phases
subtraction volume + tumor mask channel
```

Recommended first version:

```text
channels = [postcontrast_image, subtraction_image, tumor_mask]
input shape = fixed 3D crop around tumor
```

## Pathology encoder

Pathology features:

```text
ER status
PR status
HER2 status
subtype
grade
Ki-67
node status if available
```

Simple implementation:

```python
pathology_embedding = MLP(one_hot_and_numeric_pathology_features)
```

## Molecular encoder

For early versions, use a tabular molecular encoder:

```text
BRCA1/2
HRD
TP53
PIK3CA
ESR1
ERBB2
PTEN
unknown/missing flags
```

Later, replace or augment it with the molecular graph attention model described in `08_molecular_graph_explanation_engine.md`.

## Fusion model

The fusion model combines embeddings:

```python
z = concatenate([
    image_embedding,
    imaging_feature_embedding,
    pathology_embedding,
    molecular_embedding,
    patient_context_embedding,
])

parameter_distribution = distribution_head(fusion_mlp(z))
```

A transformer-based fusion model can be added later if there are enough data and modalities.

## Distribution head

Use a parameter head that outputs constrained distributions.

**Work in log space.** Predicting `mu` in linear space via `softplus` and then sampling with a log-normal is a parameterization mismatch: a log-normal interprets its location/scale as those of the *underlying normal* (log space), so a linear-space `mu` would be silently exponentiated. Instead, model every positive parameter in log space — the head emits `mu` with an **identity** activation (no softplus) and `sigma` as the log-space scale. Positivity is then automatic and consistent with the sampler.

**Bound the range.** A softplus/exp with no upper bound can emit implausibly large `k` or `D`. Map into the Family-E literature ranges with a scaled sigmoid so the head and the biological priors agree by construction.

Example:

```python
class ParameterHead(nn.Module):
    def forward(self, z):
        # log-space location, bounded into [log_lo, log_hi] from Family-E ranges
        u = torch.sigmoid(self.mu_layer(z))
        log_mu = self.log_lo + u * (self.log_hi - self.log_lo)

        # log-space scale; full-covariance variant emits a Cholesky factor instead
        log_sigma = softplus(self.sigma_layer(z)) + 1e-4
        return log_mu, log_sigma     # consumed by a log-normal / joint sampler
```

Positivity is guaranteed because sampling happens in log space (`x = exp(normal)`),
so no separate positivity transform is needed:

```text
proliferation > 0
diffusion > 0
drug sensitivity > 0
drug decay > 0
carrying capacity > 0
```

**Training loss.** Plain Gaussian/log-normal NLL is known to underfit the mean
and inflate variance on hard cases. Train the head with **β-NLL**
(Seitzer et al. 2022) or a two-stage mean-then-variance fit, plus an explicit
calibration term (the `L_calibration` referenced in Family B):

```math
L_{head} = \text{β-NLL}(\theta^* \mid \mu, \Sigma) + \lambda\, L_{calibration}
```

For the full-covariance / mixture / flow variants, replace β-NLL with the
multivariate NLL (or the flow's negative log-density) of the joint posterior.

## Training under data scarcity

The amortizer is the most data-starved component in the system. Public data give abundant segmentation masks and baseline imaging, but the targets this model needs — patient-specific mechanistic parameters paired with full multimodal baseline data — exist for only a few hundred partial-modality cases.

"Not enough data" is really two separate problems, with different fixes:

```text
Problem 1: we lack targets.
  Fitted parameters θ* require expensive per-patient calibration on the
  few hundred longitudinal cases, and are point estimates that ignore
  parameter identifiability.

Problem 2: we must learn the baseline → parameter map from few examples.
  Even with perfect targets, mapping (MRI + pathology + molecular) to a
  distribution over k, D, θ, α, β is high-dimensional regression with
  only ~hundreds of partial-modality examples.
```

The simulator solves Problem 1 well and Problem 2 barely at all — Problem 2 is the biology we are trying to learn and cannot synthesize. The solution is a layered stack, ordered below by leverage.

### 0. Identifiability analysis (decide what to learn at all)

Before spending any scarce real data on regression, ask which parameters are even *identifiable* from a realistic baseline plus 1–2 follow-up MRIs. The reaction-diffusion model is "sloppy": many parameters (and combinations) are practically unconstrained by the data we actually observe, so attempting to learn all of them wastes the few hundred real cases on directions the data cannot pin down.

```text
run a sensitivity / identifiability analysis on the simulator
  (Fisher information, profile likelihood, or active subspaces)
  given a realistic observation schedule (baseline + 1–2 follow-ups)
LEARN only the identifiable directions
PIN the rest to their Family-E priors (θ_cap is already fixed; D is weak)
```

This directly attacks Problem 2 by shrinking the regression dimensionality *before* any data is spent — it generalizes the instinct already applied to carrying capacity and diffusion in Family E.

### A. Amortized calibration via simulation-based inference (make the targets)

The simulator is a generative model: sample `θ ~ p(θ)`, simulate forward, obtain a trajectory `y`. Generate millions of `(θ, y)` pairs and train a Neural Posterior Estimator (a normalizing flow) `q(θ | y)` to invert the simulator.

Apply the trained estimator to the real longitudinal cases to obtain a full posterior over `θ` per patient in one forward pass — cheaply, consistently, and with honest uncertainty where the data are underdetermined.

```text
sample θ ~ p(θ)  →  simulate  →  y          (unlimited synthetic pairs)
train q(θ | y) once on synthetic pairs       (normalizing-flow NPE)
apply q to each real longitudinal case       → posterior targets θ* for the amortizer
```

Benefits:

```text
target generation is decoupled from cohort size
identifiability is represented honestly (wide/multimodal posteriors)
posteriors, not point estimates, become amortizer targets
```

Requirements:

```text
realistic observation/noise model (calibrate from RIDER test-retest data)
prior p(θ) must cover the real parameter space
  (literature ranges for decay β, GDSC/DepMap for drug-sensitivity α)
```

Tooling: `sbi` or BayesFlow; sequential NPE; flow-based posteriors. This subsumes the older two-stage "pseudo-label" plan: the per-patient fit becomes one amortized forward pass that returns a posterior instead of a point estimate.

**Use the NPE posterior's location, not its width, as the amortizer target.** A subtle but important mismatch: NPE computes `q(θ | y)` conditioned on the *full longitudinal trajectory* `y`, whereas the deployed amortizer predicts from *baseline-only* `x`. The posterior *mean* is a sound regression target, but the NPE posterior is narrow precisely because it has seen the trajectory. Training the amortizer to reproduce that width would make it overconfident — a baseline-only twin genuinely knows less and must have *wider* bands. So: take the location from A, but let the predictive *spread* be re-learned for the baseline-only information set (via Family B's end-to-end loss and a held-out coverage calibration), never copied from A.

### B. Differentiable end-to-end training (no explicit targets)

Make the simulator differentiable (see `05_mechanistic_tumor_simulator.md`, v3) and train the amortizer through it, using the physics as the decoder:

```text
x (baseline) → amortizer → θ-distribution → sample → simulator → predicted trajectory
                                                                      ↓
                                            loss vs the patient's observed trajectory
```

This needs no pre-fitted `θ*`; every real longitudinal case supervises the network directly.

Loss:

```math
L = L_{trajectory} + L_{pCR} + L_{calibration} + L_{regularization}
```

Requirements and risks:

```text
reduced-order model or adjoint method for tractable, stable PDE gradients
regularize toward the biological prior (KL) to control non-identifiability
```

### C. Synthetic-pretraining curriculum (combine A and B)

```text
1. Pretrain on synthetic: simulate millions of (θ, y) pairs and train the
   amortizer (B's trajectory loss and/or A's NPE targets) on them.
   Teaches the dynamics and the parameter geometry.
2. Fine-tune on real: switch to the few hundred real longitudinal cases,
   using A's posteriors as targets and/or B's end-to-end trajectory loss.
   Teaches the biology → parameter correlations (Problem 2).
```

This is the highest-leverage move: the synthetic phase has unlimited data, the real phase carries the biology.

### D. Representation transfer (shrink the parameters trained on scarce data)

`f(x) = head(encoder(x))`, and the encoder holds almost all the weights. The goal is for the encoder to come from abundant data so only the small head is fit on the scarce `(x → θ)` pairs. Public state of the art already covers this, so the rule is **adopt, do not build**.

**Spatial encoder — adopt a public model, frozen.** Use an off-the-shelf 3D MRI foundation model (e.g. Triad, Decipher-MR, or MRI-CORE) or the MAMA-MIA segmentation backbone as the image encoder. Keep it frozen (or lightly fine-tune) and train only the fusion + distribution head on the few hundred multimodal `(x, θ)` pairs. Bespoke self-supervised pretraining (SwinUNETR-style SSL, Models Genesis) is a **fallback only** if the public backbones underperform on breast DCE-MRI — not the default.

**DCE kinetics — deterministic feature maps, not a learned model.** The public encoders ingest a single 3D volume and do not model contrast dynamics, but the dynamics we need are computed analytically, not learned. From the multi-phase DCE series, compute per-voxel kinetic maps (wash-in slope, peak enhancement, washout, normalized AUC; see `04_mri_ingestion_segmentation_feature_extraction.md`) and feed them as extra input channels. The **normalized-AUC map is used directly as the simulator's spatial `delivery(x)` field**; the learned `deliveryScale` parameter is a single global multiplier on that field, not a competing estimate of delivery — they are not redundant. Wash-in/AUC are scanner- and protocol-dependent, so **harmonize these channels across sites** (e.g. ComBat) before they enter the encoder. No bespoke 4D pretraining is required.

**Watch for pretraining → evaluation leakage.** The public backbones above are trained on overlapping public breast-MRI corpora, several of which (Duke, I-SPY, MAMA-MIA) are also our evaluation cohorts — so a "held-out cohort" may not be truly held out. Document each backbone's pretraining corpus, enforce **patient- and site-level** splits, and confirm no overlap between the pretraining data and the evaluation cohorts before reporting generalization.

**Other encoders.** Pretrain the molecular graph on TCGA-BRCA / METABRIC (masked-feature / link-prediction / subtype tasks); reuse the segmentation backbone as an additional imaging feature source.

With a frozen public backbone plus deterministic kinetic channels, the only weights trained on the scarce pairs are the fusion + distribution head.

### E. Biology-informed residual priors (shrink what must be learned)

Rather than learn `θ = f(x)` from scratch, predict a **residual** from a biological prior built from established oncology:

```math
\theta = prior(subtype, biomarkers) + \Delta_{learned}
```

The prior is itself a distribution; its **width encodes confidence** and widens when a biomarker is unknown (e.g. BRCA/HRD unknown → wide DNA-repair-related drug-sensitivity prior — "unknown" is not "negative"). Concrete anchors per parameter:

| Parameter | Anchored by | Concrete anchor |
|---|---|---|
| proliferation `k` | Ki-67, grade | high if Ki-67 ≳ 20% (intermediate 14–20%); grade 3 → higher `k` |
| drug sensitivity `α` | subtype + biomarkers, calibrated so the ensemble reproduces subtype pCR base rates | population pCR ≈ luminal A 9%, luminal B 11%, TNBC 47%, HER2+ 55%; HER2+ → high anti-HER2 `α`; HR+ → low chemo / endocrine `α`; **BRCA/HRD+ → elevated platinum & anthracycline `α`** |
| drug decay `β` | pharmacokinetics (literature ranges) | doxorubicin β ∈ [0.01, 0.6] /day; cyclophosphamide β ∈ [1.0, 5.4] /day; paclitaxel β ≈ mean of the two |
| carrying capacity `θ_cap` | cell-packing physics | ≈ fixed constant (packing density 0.7405, ~10 µm cell radius) — not learned |
| diffusion `D` | weak (grade / infiltrative histology) | broad prior; `Δ_learned` does most of the work here |

Notes:

```text
pCR rates anchor the expected response magnitude, not a single parameter:
  tune the α / k priors so simulated ensembles reproduce these base rates per subtype.
subtype → parameter shifts reuse the curated rules in
  08_molecular_graph_explanation_engine.md.
subtype-conditioned α scaling can be calibrated against GDSC/DepMap, but treat
  these as the WEAKER anchor: they are in-vitro cell-line IC50s, and the
  in-vitro → in-vivo α leap is large. Use them only for the RELATIVE ordering
  of drug sensitivities; the subtype pCR base rates are the trustworthy anchor.
enforce monotonicity (e.g. higher Ki-67 ⇒ higher k) ARCHITECTURALLY, not as a
  soft penalty: use a monotonic network (positive-weight / lattice) so the
  constraint cannot be violated at inference. This matters for trust and for
  the explanation surfaced by the molecular-graph engine.
```

Strong inductive bias means less data is needed, and the prior doubles as the explanation surfaced by the molecular-graph engine.

### F. Use every case (missingness, pooling, multi-task)

```text
modality dropout: train so the net works with any subset of modalities,
  letting non-aligned cohorts each contribute
  (Duke = imaging+genomics; I-SPY = longitudinal; TCGA = molecular)
hierarchical/partial pooling: population prior + per-patient deviation;
  thin-data patients shrink toward the population
multi-task auxiliary heads (pCR, subtype, residual volume) using the
  label-rich BreastDCEDL cohort to regularize the shared encoder
```

## Recommended training curriculum

The families above compose into one end-to-end recipe:

```text
0. Run the identifiability analysis; fix unidentifiable params to priors    (0)
1. Pretrain encoders self-supervised on all imaging/molecular data          (D)
2. Build an SBI estimator; turn the longitudinal cohort into
   uncertainty-aware parameter targets (use the LOCATION as target)         (A)
3. Pretrain the amortizer on unlimited synthetic dynamics                   (C)
4. Fine-tune end-to-end through the differentiable simulator on
   real trajectories; re-learn predictive spread for baseline-only input    (B)
5. Apply biology-informed priors, modality dropout, and pooling
   throughout                                                            (E, F)
6. Gate on sim-to-real and calibration checks (below)
```

## Sim-to-real risk and required checks

Every step above leans on the simulator, so the dominant failure mode is the **sim-to-real gap / model misspecification**: if synthetic trajectories do not resemble real ones, pretraining injects a confidently wrong prior — and the AI cannot override the physics to fix it.

Required checks before trusting any synthetic pretraining:

```text
discriminator / two-sample test between synthetic and real trajectories;
  if separable, fix p(θ) or the noise model first
held-out-cohort validation (I-SPY1 or a held-out site): do the learned
  priors improve real trajectory prediction over a generic population prior?
posterior coverage / simulation-based calibration: do the credible
  intervals contain the truth at the nominal rate?
conformal prediction wrapper on the DEPLOYED parameter/trajectory
  intervals: gives distribution-free, finite-sample coverage on the scarce
  real cohort. Complements SBC, which validates only the SIMULATOR/NPE,
  not the deployed baseline-only amortizer.
```

## Uncertainty output

The amortizer should support:

```text
aleatoric uncertainty: patient data are inherently ambiguous
missing-data uncertainty: important features are unknown
model uncertainty: model is unsure or out-of-distribution
```

Implementation options:

```text
ensembles
Monte Carlo dropout
deep evidential regression
normal/inverse-gamma output heads
Bayesian neural network approximations
```

First practical version:

```text
train 5 small model seeds
average predictions
use between-model variance as model uncertainty
```

Caveats for the practical version:

```text
5 seeds is the low end for epistemic variance — treat as a starting point,
  and revisit if coverage checks are unstable.
between-seed variance does NOT give the out-of-distribution signal promised
  above. For OOD, add a density model on the fused latent z (or use the
  deployed flow's own log-likelihood) and flag low-density inputs explicitly.
keep ensembles and MC-dropout as distinct mechanisms; do not conflate them.
```

## Sampling parameter particles

Sample from the **joint** distribution, not per-parameter marginals — sampling
each parameter independently destroys the `k`/`θ_cap` and `D`/`k` correlations
and yields jointly impossible twins. Draw in log space (positivity is then free)
and exponentiate. Use a seeded, logged RNG so a clinical twin's ensemble is
reproducible and auditable.

```python
def sample_parameter_particles(param_dist, n=256, seed=0):
    rng = np.random.default_rng(seed)                  # seeded + logged
    # log_mean: (k,) vector; L: (k, k) Cholesky factor of the log-space cov
    log_mean, L = param_dist.log_mean, param_dist.cholesky_L
    z = rng.standard_normal((n, log_mean.shape[0]))
    log_samples = log_mean + z @ L.T                   # correlated draws
    samples = np.exp(log_samples)                      # back to positive params
    return [dict(zip(param_dist.param_order, row)) for row in samples]
```

For the mixture / flow output variants, replace the single
`log_mean + z @ Lᵀ` draw with a draw from the mixture or a flow sample. Each
parameter sample becomes one possible twin.

## Missing-data handling

Missing biomarkers should not be silently imputed without explanation. The model should include missingness flags:

```text
BRCA unknown
Ki-67 unknown
HER2 unknown
node status unknown
```

The output should include:

```text
This twin has high uncertainty because BRCA/HRD status is missing.
Adding BRCA/HRD data would most reduce uncertainty in drug-sensitivity parameters.
```

**Compute this as Value-of-Information, not by hand.** "Adding BRCA/HRD would most reduce uncertainty" is exactly the expected reduction in posterior entropy from observing a currently-missing feature. Rank the missing modalities by it so the recommendation is principled and orderable — which test to send the patient for next, quantified:

```math
VoI(m) = H[\theta \mid x] - \mathbb{E}_{m}\big[H[\theta \mid x, m]\big]
```

where `m` is a missing feature and the expectation is over its imputed values. Surface the top-ranked `m` as the recommended next measurement.

## Evaluation

Evaluate the amortizer by asking:

```text
Does it predict calibrated mechanistic parameters?
Does it improve trajectory prediction over generic priors?
Are uncertainty bands calibrated?
Does it generalize across sites/scanners/subtypes?
Does it behave sensibly with missing data?
```

Metrics:

```text
parameter RMSE against calibrated parameters
trajectory RMSE after simulation
pCR AUC as secondary endpoint
Brier score
uncertainty coverage
negative log likelihood
```

## Service interface

```text
POST /cases/{case_id}/parameter-amortizer/run
```

Request:

```json
{
  "case_id": "case_123",
  "imaging_features_uri": "s3://...",
  "tumor_crop_uri": "s3://...",
  "pathology": {},
  "molecular": {},
  "patient_context": {}
}
```

Response:

```json
{
  "parameter_distribution_uri": "s3://...",
  "n_particles_recommended": 512,
  "uncertainty_summary": {
    "overall": "moderate",
    "drivers": ["BRCA/HRD unknown", "only baseline MRI available"]
  }
}
```

## Implementation milestones

1. Tabular-only parameter prior baseline.
2. Run the identifiability analysis; fix unidentifiable parameters to priors (family 0).
3. Add imaging features.
4. Adopt a frozen public 3D MRI foundation-model encoder + deterministic DCE kinetic channels, with site harmonization and a leakage-free split (family D).
5. Add molecular missingness flags and modality dropout (family F).
6. Add molecular graph encoder.
7. Move the distribution head to a joint output (full-covariance → mixture → flow) with log-space, bounded, β-NLL training.
8. Build the SBI amortized-calibration estimator for posterior targets; use the location as target (family A).
9. Pretrain on synthetic dynamics, then fine-tune end-to-end on real trajectories, re-learning predictive spread for baseline-only input (families C, B).
10. Add biology-informed residual priors with architectural monotonicity (family E).
11. Add the OOD density model and the Value-of-Information ranking for missing data.
12. Add uncertainty calibration, a conformal-prediction wrapper, and sim-to-real / coverage checks.
