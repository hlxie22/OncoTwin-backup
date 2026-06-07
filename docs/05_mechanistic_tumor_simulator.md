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

This equation is the baseline. The refinements below are the parts that matter most for neoadjuvant breast cancer; they are introduced incrementally across the model versions (see `## Model versions`).

### Nondimensionalize the density

Solve in the normalized variable `u = N/θ ∈ [0, 1]` rather than raw density. This conditions the numerics, removes one free parameter, and lets the carrying-capacity bound be enforced with a smooth map (see `## Numerical implementation`) instead of a hard clip — important for the differentiable v3 path.

### Growth term: logistic or Gompertz

Logistic growth is the default. Gompertz often fits tumor kinetics better and is a drop-in replacement for the proliferation term:

```math
\text{logistic: } k\,u(1-u)
\qquad
\text{Gompertz: } k\,u\,\ln\!\left(\frac{1}{u}\right)
```

Expose the growth law as a parameter so calibration can compare both.

### Kill term: saturating and proliferation-coupled

Linear log-kill (`u · Σ_i α_i C_i`) is unsaturated and acts uniformly on all cells. Two corrections:

1. **Saturating (Emax/Hill) pharmacodynamics** so large doses do not produce unphysical kill:

```math
\text{kill} = u \sum_i \alpha_i \frac{C_i(\mathbf{x},t)}{C_i(\mathbf{x},t) + EC_{50,i}}
```

2. **Proliferation coupling** — cytotoxics (taxanes, anthracyclines) preferentially kill cycling cells, so scale the kill by the proliferating fraction `(1 - u)`:

```math
\text{kill} = u\,(1-u) \sum_i \alpha_i \frac{C_i}{C_i + EC_{50,i}}
```

The `Σ_i` assumes drugs act additively (Bliss/Loewe independence). AC-T sequencing and synergy are **not** additive — document this as a known limitation, and treat any interaction term as a later extension.

### Resistant subpopulation (v1+)

Residual disease at surgery is typically the resistant fraction, so the dominant pCR-failure mode cannot be captured by a single density. Split the state into sensitive and resistant compartments with shared growth/diffusion but very different drug sensitivity (`α_r ≪ α_s`):

```math
\frac{\partial u_s}{\partial t} = \text{(growth)} + \text{(diffusion)} - u_s\sum_i \alpha_{s,i}\,\tilde C_i
\qquad
\frac{\partial u_r}{\partial t} = \text{(growth)} + \text{(diffusion)} - u_r\sum_i \alpha_{r,i}\,\tilde C_i
```

This two-compartment split is also the natural attachment point for the molecular-graph parameter modifiers (see `08_molecular_graph_explanation_engine.md`). Optionally allow a small switching/induced-resistance rate from `u_s` to `u_r`.

### Immunotherapy is not a cytotoxic

Immunotherapy is listed as a drug, but its dynamics are delayed, immune-cell-mediated, and non-monotonic — it should not share the linear cytotoxic kill term. For v0–v2, either give it a separate effector term or explicitly flag it as out-of-scope and warn when an immunotherapy event is present.

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

Treat the DCE → `delivery(x)` mapping as a **validated assumption, not ground truth**, and surface it as a model assumption in outputs. Note also that this delivery field is static: it does not capture vascular normalization or perfusion changes as the tumor responds during treatment. A time-varying `delivery(x, t)` is a later extension.

## Parameters

```typescript
type MechanisticParams = {
  growthLaw: "logistic" | "gompertz";
  proliferationRate: number;
  diffusionCoefficient: number;
  carryingCapacity: number;
  drugSensitivity: Record<string, number>;        // α for the sensitive compartment
  drugEC50: Record<string, number>;               // Hill EC50 per drug
  drugDecay: Record<string, number>;
  deliveryScale: number;
  spatialHeterogeneityScale: number;
  // resistant compartment (v1+); omit for single-compartment v0
  resistantFraction0?: number;                     // initial fraction in u_r
  resistantSensitivity?: Record<string, number>;   // α_r ≪ α_s
  inducedResistanceRate?: number;                  // optional u_s → u_r switching
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

### Time integration and stability

Do **not** integrate the diffusion term with a fixed-step forward Euler scheme. Explicit diffusion is only conditionally stable, requiring `dt ≤ dx² / (2 · ndim · D_max)`; at `dt = 0.25` days with a plausible invasion coefficient on a downsampled grid this will oscillate or blow up.

```text
1. Treat diffusion implicitly: backward Euler or Crank–Nicolson (unconditionally stable),
   or operator/Strang-split diffusion (implicit) from reaction + kill (explicit) — an IMEX scheme.
2. Add an explicit CFL/stability guard so the engine fails loud rather than emitting
   garbage trajectories that then corrupt calibration. Support adaptive dt.
3. Specify boundary conditions: no-flux (Neumann) at the tissue/breast boundary.
4. Sub-step dosing: dose events are at integer days but dt < 1, so integrate exposure
   correctly across sub-steps instead of only sampling on event days.
```

### Smooth bound enforcement

Solve in the normalized variable `u = N/θ ∈ [0, 1]` and enforce the bounds with a **smooth** map (e.g. clamp via softplus, or a sigmoid reparameterization), not a hard `clip`. A hard clip has zero/undefined gradient at the bounds and silently breaks the v3 differentiable path.

### Solver steps

```text
1. Initialize u(x,0) = N(x,0)/θ from the tumor mask and imaging features
   (and the resistant fraction for v1+).
2. For each time step:
   a. compute drug exposure (sub-step accurate)
   b. compute reaction terms (growth + saturating, proliferation-coupled kill) explicitly
   c. advance diffusion implicitly (IMEX), applying no-flux boundary conditions
   d. apply the smooth [0,1] projection
3. At requested timepoints, compute volume and residual-risk maps.
```

### Batched ensembles

The product needs ensembles (the "% of simulated twins" framing and the Bayesian layer), so vectorize over a leading ensemble/parameter axis from the start rather than looping per realization — retrofitting this later is painful. In the differentiable v3 build this is a `vmap` over parameters.

### Pseudocode

`state` carries a leading ensemble axis (and a compartment axis for sensitive/resistant in v1+); all ops below are batched over it.

```python
def simulate_tumor(initial_state, params, treatment_schedule, times, dt=0.25):
    # initial_state is u = N / theta in [0, 1], shape (ensemble, [compartment,] *grid)
    state = initial_state.clone()
    outputs = []
    current_time = 0.0

    dt = stability_limited_dt(dt, grid_spacing, params)  # CFL guard / adaptive dt

    for target_time in times:
        while current_time < target_time:
            # exposure integrated across the sub-step, not just at event days
            exposure = compute_drug_exposure(treatment_schedule, params, current_time, dt)

            # reaction handled explicitly: growth (logistic or gompertz) + saturating,
            # proliferation-coupled kill, summed over compartments
            reaction = growth_term(state, params) - kill_term(state, exposure, params)
            state = state + dt * reaction

            # diffusion advanced implicitly (IMEX) with no-flux boundary conditions
            state = implicit_diffusion_step(state, params.diffusion, dt, bc="neumann")

            state = smooth_project_unit_interval(state)  # softplus/sigmoid, not hard clip
            current_time += dt

        outputs.append(summarize_state(state, current_time, params.theta))

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
  detectionThreshold: number;   // density above which a voxel counts as "tumor present"
  notes: string[];
};
```

The density-to-volume conversion depends on a **detection threshold** (the `u` above which a voxel is counted as tumor). The pCR proxy is highly sensitive to this value, so it must be explicit, recorded in the output, and propagated into the uncertainty bands rather than hard-coded silently.

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

## Parameter identifiability

With ~7+ parameters (`D, k, θ, α_i, EC50_i, β_i, delivery, heterogeneity`) and typically only three imaging timepoints (T1/T2/T3), the model is badly under-determined. If ignored, the Bayesian posterior and any "% of simulated twins" statement will mostly echo the priors rather than the data — a false-confidence trap.

Mitigations:

```text
1. Run a practical-identifiability analysis (profile likelihood, or Fisher-information /
   sloppy-model eigen-analysis) and document which parameters the data actually constrain.
2. Tier the free-parameter set to the available data, matching the v0/v1/v2 ladder:
     volume-only  → fit ~2 params (effective growth, effective kill); fix the rest to priors
     regional     → add per-compartment sensitivity
     voxelwise    → fit spatial maps
   Never fit voxelwise maps from a single diameter measurement.
3. State plainly in outputs when a parameter is prior-dominated (data-uninformed).
```

## Numerical verification

Before trusting calibration, verify the solver itself with cheap analytic checks (these belong in the test suite):

```text
pure diffusion (no reaction/kill) → matches the analytic Gaussian-spreading solution
pure logistic / Gompertz growth   → matches the closed-form trajectory
no treatment                       → monotone growth to carrying capacity
grid + timestep refinement         → convergence (solution stable as dx, dt shrink)
no-flux boundary                   → total mass changes only via reaction/kill, not flux
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

Introduce the sensitive/resistant split and the saturating, proliferation-coupled kill here — regional data is the cheapest place to calibrate them stably.

### v2: downsampled 3D spatial PDE

Use a downsampled tumor grid with diffusion, proliferation, drug kill, and spatial delivery. Integrate diffusion with the IMEX/implicit scheme and no-flux boundaries described above; do not use fixed-step explicit diffusion.

### v3: differentiable simulator

Implement in PyTorch so gradients can train the AI amortizer end to end. Use the smooth `[0,1]` projection (not hard clipping) and `vmap` over the parameter ensemble so gradients flow cleanly through the whole trajectory.

## Implementation priorities

1. Implement v0 ODE (nondimensionalized) with the numerical-verification checks to test API and UI.
2. Implement v1 regional model with the sensitive/resistant split and saturating kill for stable calibration.
3. Implement v2 spatial model (IMEX diffusion, no-flux BCs) for MRI-backed cases.
4. Add batched ensemble simulation.
5. Run the identifiability analysis and tier the free-parameter set to the data.
6. Add Bayesian update hooks.
7. Add reduced-order modeling if spatial simulation is too slow.
8. Add differentiable PyTorch version (smooth projection, vmap) for fine-tuning.

## Safety constraints

The simulator is a low-level engine: it produces model outputs, not patient-facing language. Any exploratory ranking or recommendation framing is assembled at the Scenario Lab / product layer (see `09_scenario_lab_toxicity_twin_and_safety.md`), where the standard not-guaranteed / not-medical-advice disclaimer is attached.

The simulator should produce:

```text
response trajectories and scenario comparisons
exploratory tradeoff / ranking scores across candidate schedules
uncertainty bands
model assumptions
missing-data warnings
questions to ask care team
```

Consistent with the app's lightweight-disclaimer policy, exploratory treatment suggestions and rankings are permitted downstream as long as they carry uncertainty and the standard disclaimer. The one hard rule is to avoid **false certainty** — the simulator must never emit:

```text
a claim of guaranteed, definite, or curative response
a deterministic "this schedule will work" outcome
any output presented as a directive rather than an exploratory, uncertain estimate
```
