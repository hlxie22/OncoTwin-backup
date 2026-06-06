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

```typescript
type ParameterDistribution = {
  proliferationRate: { mean: number; std: number };
  diffusionCoefficient: { mean: number; std: number };
  carryingCapacity: { mean: number; std: number };
  drugSensitivity: Record<string, { mean: number; std: number }>;
  drugDecay: Record<string, { mean: number; std: number }>;
  deliveryScale: { mean: number; std: number };
  heterogeneityScale: { mean: number; std: number };
};
```

The simulator samples from these distributions to create an ensemble of possible twins.

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

Example:

```python
class ParameterHead(nn.Module):
    def forward(self, z):
        raw_mu = self.mu_layer(z)
        raw_sigma = self.sigma_layer(z)

        mu = softplus(raw_mu)
        sigma = softplus(raw_sigma) + 1e-4
        return mu, sigma
```

Use transformations so parameters stay valid:

```text
proliferation > 0
diffusion > 0
drug sensitivity > 0
drug decay > 0
carrying capacity > 0
```

## Training targets

There are two ways to train the amortizer.

### Option A: two-stage pseudo-label training

First, calibrate mechanistic parameters for each longitudinal training case. Then train the AI to predict those fitted parameters from baseline data.

```text
baseline data → AI → parameter estimate
fitted mechanistic parameters → target
```

Loss:

```math
L_{param}=\sum_j \left\|\hat{\theta}_j-\theta_j^*\right\|^2
```

### Option B: simulation-aware training

Train the AI by running the simulator and comparing simulated trajectories to observed trajectories.

```text
baseline data → AI → parameter distribution → simulator → trajectory → compare with observed MRI response
```

Loss:

```math
L = L_{trajectory} + L_{pCR} + L_{calibration} + L_{regularization}
```

This is more faithful but harder to implement.

## Recommended path

Use both:

```text
1. Fit mechanistic parameters for cases with longitudinal MRI.
2. Train the AI amortizer to predict those parameters.
3. Fine-tune with simulation-aware losses.
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

## Sampling parameter particles

```python
def sample_parameter_particles(param_dist, n=256):
    particles = []
    for _ in range(n):
        sample = {}
        for name, dist in param_dist.items():
            sample[name] = lognormal_sample(dist.mean, dist.std)
        particles.append(sample)
    return particles
```

Each parameter sample becomes one possible twin.

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
2. Add imaging features.
3. Add 3D MRI encoder.
4. Add molecular missingness flags.
5. Add molecular graph encoder.
6. Train with pseudo-label fitted parameters.
7. Fine-tune with simulation-aware trajectory loss.
8. Add uncertainty calibration.
