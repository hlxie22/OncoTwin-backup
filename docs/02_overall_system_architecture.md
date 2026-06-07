# Overall System Architecture

## System summary

OncoTwin has two major loops:

1. **Offline training loop**: public datasets are used to train segmentation, imaging encoders, parameter-amortization networks, molecular graph models, uncertainty calibration, and validation reports.
2. **Online app loop**: a user or researcher creates a case, the system initializes a patient-specific twin, simulates response, updates the twin as new observations arrive, and displays scenario results with uncertainty and explanations.

The most important architecture rule is:

> AI personalizes the mechanistic simulator. It does not replace it.

## High-level pipeline

```text
Public datasets / user case
        ↓
Data ingestion and harmonization
        ↓
MRI preprocessing and segmentation
        ↓
Imaging feature extraction
        ↓
Pathology + molecular + patient-context encoding
        ↓
AI parameter amortizer
        ↓
Mechanistic tumor simulator
        ↓
Bayesian update / posterior ensemble
        ↓
Scenario Lab + uncertainty + explanations
        ↓
Patient-facing and research-facing outputs
```

## Offline training loop

The offline training system should be organized into stages.

### Stage 1: segmentation pretraining

```text
MAMA-MIA / BreastDCEDL images + tumor masks
        ↓
3D segmentation model
        ↓
Tumor masks, volumes, segmentation confidence
```

Primary output: a robust DCE-MRI tumor segmentation service.

### Stage 2: imaging representation learning

```text
Baseline MRI + tumor mask + pCR/residual outcome labels
        ↓
MRI encoder
        ↓
Latent tumor phenotype embedding
```

The encoder should learn features useful for response modeling, not just segmentation.

### Stage 3: mechanistic parameter fitting

```text
Longitudinal MRI timepoints + treatment schedule + outcomes
        ↓
Patient-level calibration
        ↓
Estimated mechanistic parameters per case
```

These fitted parameter sets become pseudo-labels for the AI parameter amortizer.

### Stage 4: multimodal parameter amortization

```text
Baseline MRI + pathology + molecular data + age/context
        ↓
Fusion model
        ↓
Distribution over mechanistic parameters
```

This model lets the app initialize a personalized twin before many follow-up scans are available.

### Stage 5: end-to-end fine-tuning and validation

```text
Parameter priors + simulator + observed response
        ↓
Differentiable or simulation-aware training
        ↓
Calibrated response trajectories and uncertainty bands
```

Validation should include tumor-volume trajectories, pCR/residual classification, calibration, segmentation quality, and uncertainty quality.

## Online app loop

The online app loop starts when a case is created.

```text
1. Create case.
2. Enter pathology and treatment context.
3. Upload MRI or enter tumor measurements.
4. Segment tumor if MRI exists.
5. Extract imaging features.
6. Estimate patient-specific parameter prior.
7. Generate posterior ensemble of possible twins.
8. Simulate response trajectories.
9. Display uncertainty, residual-risk heatmap, and explanations.
10. Update the twin when new data arrive.
11. Run safe research scenarios in Twin Scenario Lab.
```

## Main services

### Case service

Stores case metadata, structured pathology, treatment context, uploaded scans, user-facing events, and permissions.

### Imaging service

Handles DICOM/NIfTI ingestion, preprocessing, registration, segmentation, mask postprocessing, and quality control.

### Feature extraction service

Computes tumor volume, longest diameter, shape descriptors, enhancement features, radiomics-style features, and longitudinal change features.

### Parameter amortizer service

Runs the multimodal AI model that predicts parameter distributions for the mechanistic simulator.

### Mechanistic solver service

Runs reaction-diffusion treatment-response simulations for an ensemble of parameter samples.

### Bayesian update service

Computes the posterior by batch importance sampling from the amortizer prior over all observations to date, escalating to a tempered SMC sampler when the effective sample size is low, and produces uncertainty summaries (see `07`).

### Molecular graph service

Builds patient-specific molecular pathway graphs and produces mechanism embeddings, parameter modifiers, pathway attention, and missing-data explanations.

### Scenario Lab service

Runs simulations under research scenario templates such as current regimen, delayed measurement, alternative timing templates, missing-biomarker assumptions, or toxicity-sensitive scenarios.

### Explanation service

Turns model outputs into patient-facing and researcher-facing explanations, including safety labels.

## Data object model

```typescript
type OncoTwinCase = {
  caseId: string;
  mode: "research" | "demo" | "user-entered";

  imagingTimepoints: ImagingTimepoint[];
  pathology: PathologyProfile;
  molecular: MolecularProfile;
  treatment: TreatmentPlan;
  patientContext: PatientContext;

  twinState?: TwinState;
  posterior?: TwinPosterior;
  scenarioRuns: ScenarioRun[];
  auditLog: AuditEvent[];
};
```

```typescript
type ImagingTimepoint = {
  timepointId: string;
  label: "T0" | "T1" | "T2" | "T3" | "custom";
  daysFromTreatmentStart: number;
  mriVolumeUri?: string;
  tumorMaskUri?: string;
  tumorVolumeMl?: number;
  longestDiameterCm?: number;
  segmentationConfidence?: number;
  featureVectorUri?: string;
};
```

```typescript
type TwinPosterior = {
  particlesUri: string;
  createdAt: string;
  sourceObservations: string[];
  summary: {
    medianTrajectory: number[];
    lowerTrajectory: number[];
    upperTrajectory: number[];
    residualRiskMapUri?: string;
    pcrProbability?: number;
    uncertaintyScore: number;
  };
};
```

## Deployment architecture

```text
Frontend:
  Next.js / React
  Plotly or D3 for response curves
  vtk.js / Three.js for 3D tumor and residual-risk visualization

Backend API:
  FastAPI or Node/Express
  Auth and case-management layer
  Synchronous metadata endpoints
  Asynchronous simulation endpoints

Model services:
  PyTorch + MONAI
  SimpleITK / ANTsPy
  NumPy / SciPy
  pyABC / particles or a custom importance-sampling + SMC-sampler updater

Storage:
  PostgreSQL for structured metadata
  S3-compatible object storage for MRI volumes, masks, particles, and result arrays
  Redis/Celery or similar for background jobs

Monitoring:
  model-version registry
  simulation-run audit log
  data quality logs
  safety-message logs
```

## Core API endpoints

```text
POST /cases
GET  /cases/{case_id}
POST /cases/{case_id}/pathology
POST /cases/{case_id}/molecular
POST /cases/{case_id}/treatment
POST /cases/{case_id}/imaging/upload
POST /cases/{case_id}/imaging/preprocess
POST /cases/{case_id}/imaging/segment
POST /cases/{case_id}/features/extract
POST /cases/{case_id}/twin/initialize
POST /cases/{case_id}/twin/simulate
POST /cases/{case_id}/twin/update-observation
POST /cases/{case_id}/scenario-lab/run
GET  /cases/{case_id}/uncertainty
GET  /cases/{case_id}/explanation
GET  /cases/{case_id}/summary/doctor
```

## Model registry

Every model output should record:

```text
model_name
model_version
training_dataset
training_date
input schema version
output schema version
calibration report URI
known limitations
```

This is essential because the app makes scientific claims and must be auditable.

## Safety architecture

The system should enforce safety at the backend and frontend levels:

- Present treatment rankings or suggestions only as **exploratory, model-based options** — never as guaranteed, definitive, or certain.
- Attach the standard not-guaranteed / not-medical-advice disclaimer to any recommendation-style output.
- Label scenario results as research simulations.
- Show uncertainty whenever outcomes or rankings are shown.
- Show missing-data warnings.
- Show data-quality warnings for low-confidence segmentation or out-of-distribution inputs.
- Frame outputs as decision-support for discussion with an oncology team, not as instructions to follow.
- Keep model outputs clearly distinguished from clinical advice.

