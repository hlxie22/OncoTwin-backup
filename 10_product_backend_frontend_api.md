# Product, Backend, Frontend, and API Implementation

## Product goal

OncoTwin should feel like a sophisticated, interactive research simulator. The app should let a user create a case, build a virtual tumor state, run mechanistic-AI simulations, update the twin over time, and understand the sources of uncertainty.

## Primary product surfaces

### 1. Build My Twin

Inputs:

```text
MRI upload or sample public-data case
pathology biomarkers
molecular markers if known
age/context
treatment stage/current plan
optional symptom baseline
```

Outputs:

```text
case created
available data summary
missing-data checklist
simulation readiness score
```

### 2. Virtual Tumor State

Shows:

```text
tumor volume
longest diameter
tumor mask preview
segmentation confidence
imaging features
subtype/pathology summary
initial parameter prior summary
```

### 3. Mechanistic Simulation

Shows:

```text
simulated tumor trajectory
observed tumor measurements
uncertainty bands
residual-risk heatmap
parameter summary
```

### 4. Why Did the Twin Think That?

Shows:

```text
imaging drivers
pathology drivers
molecular graph drivers
missing-data drivers
uncertainty explanation
```

### 5. Twin Update Timeline

Shows:

```text
baseline twin
early-treatment update
mid-treatment update
presurgery update
how parameters changed
how uncertainty changed
```

### 6. Twin Scenario Lab

Shows:

```text
current plan simulation
hypothetical research scenarios
missing-biomarker scenarios
measurement-update scenarios
toxicity-sensitive scenarios
```

### 7. Doctor / Research Summary

Generates:

```text
one-page summary
observed measurements
simulated trajectories
uncertainty drivers
questions to discuss with care team
safety disclaimer
```

## Backend services

```text
case_service
pathology_service
molecular_service
imaging_ingestion_service
preprocessing_service
segmentation_service
feature_extraction_service
parameter_amortizer_service
mechanistic_solver_service
bayesian_update_service
scenario_lab_service
toxicity_twin_service
explanation_service
summary_service
```

## Database schema

### cases

```sql
CREATE TABLE cases (
  id UUID PRIMARY KEY,
  user_id UUID,
  mode TEXT NOT NULL,
  title TEXT,
  created_at TIMESTAMP NOT NULL,
  updated_at TIMESTAMP NOT NULL
);
```

### pathology_profiles

```sql
CREATE TABLE pathology_profiles (
  id UUID PRIMARY KEY,
  case_id UUID REFERENCES cases(id),
  er_status TEXT,
  pr_status TEXT,
  her2_status TEXT,
  subtype TEXT,
  grade INTEGER,
  ki67 NUMERIC,
  node_status TEXT,
  created_at TIMESTAMP NOT NULL
);
```

### molecular_profiles

```sql
CREATE TABLE molecular_profiles (
  id UUID PRIMARY KEY,
  case_id UUID REFERENCES cases(id),
  brca1 TEXT,
  brca2 TEXT,
  hrd TEXT,
  tp53 TEXT,
  pik3ca TEXT,
  esr1 TEXT,
  erbb2 TEXT,
  raw_json JSONB,
  created_at TIMESTAMP NOT NULL
);
```

### imaging_timepoints

```sql
CREATE TABLE imaging_timepoints (
  id UUID PRIMARY KEY,
  case_id UUID REFERENCES cases(id),
  label TEXT,
  days_from_treatment_start INTEGER,
  raw_uri TEXT,
  preprocessed_uri TEXT,
  tumor_mask_uri TEXT,
  tumor_volume_ml NUMERIC,
  longest_diameter_cm NUMERIC,
  segmentation_confidence NUMERIC,
  qc_flags TEXT[],
  created_at TIMESTAMP NOT NULL
);
```

### twin_posteriors

```sql
CREATE TABLE twin_posteriors (
  id UUID PRIMARY KEY,
  case_id UUID REFERENCES cases(id),
  particles_uri TEXT,
  trajectory_summary_uri TEXT,
  uncertainty_summary JSONB,
  source_observation_ids TEXT[],
  model_versions JSONB,
  created_at TIMESTAMP NOT NULL
);
```

### scenario_runs

```sql
CREATE TABLE scenario_runs (
  id UUID PRIMARY KEY,
  case_id UUID REFERENCES cases(id),
  posterior_id UUID REFERENCES twin_posteriors(id),
  scenario_type TEXT,
  assumptions JSONB,
  result_uri TEXT,
  safety_label TEXT,
  created_at TIMESTAMP NOT NULL
);
```

### patient_reported_outcomes

```sql
CREATE TABLE patient_reported_outcomes (
  id UUID PRIMARY KEY,
  case_id UUID REFERENCES cases(id),
  date DATE,
  fatigue INTEGER,
  nausea INTEGER,
  neuropathy INTEGER,
  pain INTEGER,
  sleep_quality INTEGER,
  appetite INTEGER,
  activity_level INTEGER,
  notes TEXT,
  created_at TIMESTAMP NOT NULL
);
```

## Core endpoints

### Create case

```text
POST /cases
```

### Add pathology

```text
POST /cases/{case_id}/pathology
```

### Add molecular profile

```text
POST /cases/{case_id}/molecular
```

### Upload MRI

```text
POST /cases/{case_id}/imaging/upload
```

### Run segmentation

```text
POST /cases/{case_id}/imaging/{timepoint_id}/segment
```

### Extract features

```text
POST /cases/{case_id}/imaging/{timepoint_id}/features
```

### Initialize twin

```text
POST /cases/{case_id}/twin/initialize
```

### Run simulation

```text
POST /cases/{case_id}/twin/simulate
```

### Update observation

```text
POST /cases/{case_id}/twin/update-observation
```

### Run scenario

```text
POST /cases/{case_id}/scenario-lab/run
```

### Get explanation

```text
GET /cases/{case_id}/explanation
```

### Generate summary

```text
GET /cases/{case_id}/summary
```

## Job queue design

Long-running operations should be asynchronous:

```text
MRI preprocessing
segmentation
feature extraction
parameter amortization
large ensemble simulation
Bayesian update
scenario lab run
summary generation with rendered figures
```

Job schema:

```typescript
type Job = {
  jobId: string;
  caseId: string;
  jobType: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: number;
  resultUri?: string;
  errorMessage?: string;
};
```

Endpoints:

```text
POST /jobs
GET /jobs/{job_id}
```

## Frontend components

```text
CaseDataCard
MissingDataChecklist
MRIUploadPanel
SegmentationViewer
TumorVolumeChart
TwinTrajectoryChart
UncertaintyBandChart
ResidualRiskViewer3D
MolecularGraphPanel
ParameterSummaryPanel
TwinUpdateTimeline
ScenarioComparisonTable
ToxicityBurdenPanel
CareTeamQuestionsCard
SafetyDisclaimerBanner
```

## Visualization requirements

### Tumor trajectory chart

Show:

```text
observed points
posterior median
80% uncertainty band
scenario comparison line if selected
```

### Residual-risk viewer

Show:

```text
baseline tumor mask
predicted residual-risk heatmap
follow-up observed residual region if available
```

### Molecular graph panel

Show:

```text
active pathways
missing biomarkers
attention/importance score
plain-language explanation
```

### Update timeline

Show:

```text
what changed
why it changed
which parameters moved
whether uncertainty increased or decreased
```

## Safety filter tests

Add automated tests that fail if generated explanations contain language like:

```text
You should take...
You should stop...
Best treatment for you...
Guaranteed response...
Will cure...
Doctor should...
```

Allowed language:

```text
In this research simulation...
The model estimates...
Uncertainty remains...
Discuss with an oncology team...
This is not a treatment recommendation...
```

## Development milestones

### Milestone 1: documentation and data design

```text
finalize schemas
finalize datasets
create sample case format
create UI wireframes
```

### Milestone 2: imaging MVP

```text
NIfTI upload
segmentation inference
tumor volume extraction
QC display
```

### Milestone 3: simulator MVP

```text
volume-only model
parameter sampling
trajectory chart
uncertainty band
```

### Milestone 4: spatial simulator

```text
mask-based initialization
regional or voxelwise model
residual-risk map
```

### Milestone 5: Bayesian update

```text
new tumor measurement
posterior particle update
update explanation
uncertainty change
```

### Milestone 6: AI personalization

```text
baseline imaging encoder
pathology encoder
parameter prior prediction
ensemble simulation
```

### Milestone 7: molecular explanations

```text
curated molecular rules
missing biomarker ranking
pathway explanation panel
```

### Milestone 8: Scenario Lab

```text
current-plan simulation
missing-biomarker scenario
measurement-update scenario
safe scenario comparison
```

### Milestone 9: toxicity/person-burden twin

```text
symptom tracker
toxicity-burden score
trend explanation
scenario coupling
```

### Milestone 10: validation and safety report

```text
segmentation validation
trajectory validation
uncertainty calibration
safety-language audit
model card
limitations page
```

## Model card requirements

Each model should have a model card including:

```text
training data
intended use
non-intended use
performance metrics
subgroup analysis
limitations
known failure modes
input requirements
uncertainty behavior
version history
```

## Final product principle

The app should feel powerful, but it must remain honest:

> OncoTwin simulates plausible response trajectories under explicit assumptions. It does not determine the correct treatment.
