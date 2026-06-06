# Mechanistic Tumor Simulator

## Purpose

The mechanistic simulator is the scientific core of OncoTwin. It should model how a tumor changes over time under treatment using biologically interpretable parameters.

The simulator should not be a fixed set of canned response curves. It should simulate an ensemble of possible patient-specific tumors, each with different plausible parameters.

## Relationship to the 2025 digital-twin paper

The simulator should use the same general family as the 2025 MRI-based digital-twin paper: a reaction-diffusion tumor model with proliferation, migration/invasion, and treatment-induced death.

However, OncoTwin should extend that model by adding:

```text
AI-estimated parameter priors
Bayesian posterior updates
molecular graph parameter modifiers
patient-reported toxicity/person-burden coupling
app-facing uncertainty and explanations
```

## State variable

The core state variable is tumor cell density:

```text
N(x, t)
```

where:

```text
x = spatial location or region
N = tumor cell density
 t = time, usually days from treatment start
```

The model can be implemented at multiple resolutions.

### Version 1: volume-only model

Use when only tumor diameter or volume is available.

```text
V(t) = tumor volume over time
```

### Version 2: regional model

Divide the tumor into regions, such as enhancing core, low-enhancement region, and invasive boundary.

```text
N_region(t)
```

### Version 3: voxelwise spatial model

Use when MRI masks and registered images are available.

```text
N(x, y, z, t)
```

The app can support all three, with uncertainty increasing when less data are available.

## Core equation

A practical spatial model:

```math
\frac{\partial N(\mathbf{x},t)}{\partial t}
= \nabla \cdot \left(D(\mathbf{x}) \nabla N(\mathbf{x},t)\right)
+ k(\mathbf{x})N(\mathbf{x},t)\left(1 - \frac{N(\mathbf{x},t)}{\theta}\right)
- N(\mathbf{x},t)\sum_i \alpha_i C_i(\mathbf{x},t)
```

Where:

```text
N(x,t) = tumor cell density
D(x) = local diffusion/invasion coefficient
k(x) = local proliferation rate
θ = carrying capacity
α_i = sensitivity to drug i
C_i(x,t) = effective exposure of drug i at location x and time t
```

## Drug exposure model

A simple drug exposure model for each drug event:

```math
C_i(t) = \sum_j dose_{i,j}\exp(-\beta_i(t - t_{i,j}))\mathbf{1}_{t \ge t_{i,j}}
```

Where:

```text
dose_i,j = dose of drug i at event j
t_i,j = time of dose event j
β_i = drug decay parameter
```

A spatial version multiplies by delivery:

```math
C_i(x,t) = delivery(x) \times C_i(t)
```

where `delivery(x)` can be estimated from DCE enhancement.

## Parameters

```typescript
type MechanisticParams = {
  proliferationRate: number;
  diffusionCoefficient: number;
  carryingCapacity: number;
  drugSensitivity: Record<string, number>;
  drugDecay: Record<string, number>;
  deliveryScale: number;
  spatialHeterogeneityScale: number;
};
```

Voxelwise extensions:

```typescript
type SpatialMechanisticParams = {
  proliferationMapUri: string;
  diffusionMapUri: string;
  deliveryMapUri: string;
  drugSensitivityScalars: Record<string, number>;
  drugDecayScalars: Record<string, number>;
  carryingCapacity: number;
};
```

## Treatment schedule representation

```typescript
type DrugDoseEvent = {
  drug: "anthracycline" | "cyclophosphamide" | "taxane" | "platinum" | "immunotherapy" | "other";
  day: number;
  relativeDose: number;
  label?: string;
};
```

A regimen is a list of dose events:

```typescript
type TreatmentSchedule = {
  regimenName: string;
  events: DrugDoseEvent[];
  totalDurationDays: number;
};
```

## Numerical implementation

### First implementation

Start with a coarse grid or regional model to avoid overengineering.

```text
input mask → downsample to manageable 3D grid → solve PDE → upsample result for visualization
```

### Solver steps

```text
1. Initialize N(x,0) from tumor mask and imaging features.
2. For each time step:
   a. compute diffusion term
   b. compute proliferation term
   c. compute drug exposure
   d. compute drug kill
   e. update N(x,t)
   f. enforce nonnegative density and carrying capacity bounds
3. At requested timepoints, compute volume and residual-risk maps.
```

### Pseudocode

```python
def simulate_tumor(initial_density, params, treatment_schedule, times, dt=0.25):
    state = initial_density.copy()
    outputs = []
    current_time = 0.0

    for target_time in times:
        while current_time < target_time:
            diffusion = compute_diffusion(state, params.diffusion)
            proliferation = params.k * state * (1 - state / params.theta)
            exposure = compute_drug_exposure(treatment_schedule, params, current_time)
            kill = state * compute_drug_kill(exposure, params)

            state = state + dt * (diffusion + proliferation - kill)
            state = clip_state(state, lower=0, upper=params.theta)
            current_time += dt

        outputs.append(summarize_state(state, current_time))

    return outputs
```

## Output summaries

For each simulation run:

```typescript
type TumorSimulationOutput = {
  times: number[];
  tumorVolumeMl: number[];
  longestDiameterCm?: number[];
  residualRiskMapUri?: string;
  pcrProxy?: number;
  finalTumorBurden: number;
  notes: string[];
};
```

## pCR proxy

A simple research proxy:

```text
pCR-like response if final viable tumor burden < threshold
```

Do not present this as a clinical pCR prediction unless properly validated.

UI language:

```text
Model-estimated residual tumor burden is low in 68% of simulated twins. This is a research simulation, not a clinical pCR prediction.
```

## Calibration targets

Fit the simulator to observations:

```text
observed tumor volume at T1/T2/T3
longest diameter if volume unavailable
residual mask if available
pCR/residual outcome at surgery
functional tumor volume if available
```

Example loss:

```math
L = \lambda_V L_{volume} + \lambda_S L_{spatial} + \lambda_R L_{response} + \lambda_P L_{prior}
```

Where:

```text
L_volume = error between observed and predicted volume
L_spatial = residual-risk map mismatch
L_response = final outcome classification loss
L_prior = parameter plausibility regularization
```

## Model versions

### v0: report-only ODE

Use a volume-only logistic/drug-kill model:

```math
\frac{dV}{dt}=rV\left(1-\frac{V}{K}\right)-\alpha C(t)V
```

This is useful for testing app logic.

### v1: regional model

Use 3 to 5 tumor compartments:

```text
enhancing core
low-enhancement region
boundary/invasive rim
optional residual-risk region
```

### v2: downsampled 3D spatial PDE

Use a downsampled tumor grid with diffusion, proliferation, drug kill, and spatial delivery.

### v3: differentiable simulator

Implement in PyTorch so gradients can train the AI amortizer end to end.

## Implementation priorities

1. Implement v0 ODE to test API and UI.
2. Implement v1 regional model for stable calibration.
3. Implement v2 spatial model for MRI-backed cases.
4. Add ensemble simulation.
5. Add Bayesian update hooks.
6. Add reduced-order modeling if spatial simulation is too slow.
7. Add differentiable PyTorch version for fine-tuning.

## Safety constraints

The simulator should never produce treatment instructions. It should produce:

```text
research scenario comparison
uncertainty bands
model assumptions
missing-data warnings
questions to ask care team
```

Never produce:

```text
recommended treatment
recommended dose
recommended delay
claim of guaranteed response
```
