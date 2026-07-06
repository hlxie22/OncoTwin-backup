# V1 Prior Stack Implementation and Evaluation Plan

Status: V1-A implementation and evaluation plan  
Scope: prior-builder layers, real-data evaluation, uncertainty calibration, and MRI-feature integration  
Primary goal: prove that the V1 prior stack can be evaluated on real longitudinal breast-cancer treatment-response data before building heavier posterior, scenario, or imaging-training infrastructure.

---

## 1. Executive Summary

V1 should be implemented as a **table-first prior-stack evaluation system**.

The first V1 milestone is not full MRI model training and not full raw DICOM ingestion. The first milestone is a real patient-level cohort table that allows the existing prior-builder layers to be evaluated against held-out tumor-volume observations.

The V1 system should:

1. Resolve a parameter contract for a patient and treatment context.
2. Apply parameter bounds and observation-noise policy.
3. Build a population prior.
4. Apply pathology and biomarker prior shifts.
5. Apply MRI-feature and QC rules.
6. Convert prior samples into simulator-compatible parameters.
7. Evaluate predictions, uncertainty, and layer deltas on real longitudinal data.

The MRI portion should be integrated through cached features, not through direct raw-image reads during every eval. The recommended V1 imaging approach is to use a pretrained tumor segmenter, specifically **MAMA-MIA pretrained nnU-Net**, and then reduce images into feature tables used by Layer 4.

Full raw I-SPY2 DICOM download and full raw DICOM preprocessing are not required for V1 success.

---

## 2. Current V1 Design Principles

### 2.1 Table-first evaluation

V1 evals should consume patient-level JSONL/CSV tables with real longitudinal measurements.

Raw MRI images are optional inputs to a separate feature-extraction lane. The prior-stack evals should read only:

```text
data/processed/v1_prior_stack/*.jsonl
data/curated/features/*.jsonl
```

The evals should not repeatedly read raw DICOM or NIfTI files.

### 2.2 Fail closed

V1 should fail closed when data is missing, synthetic, out of scope, or too low quality.

Examples:

```text
synthetic/demo/fixture rows rejected by default
missing baseline or final volume excluded with reason
out-of-scope tumor subtype excluded from V1-A performance claims
low-quality MRI features widen uncertainty or become report-only
missing posterior/scenario/explanation runtimes marked unavailable, not passed
```

### 2.3 Separate data readiness from runtime readiness

The V1 plan should distinguish:

```text
real-data cohort readiness
MRI-feature readiness
posterior-update runtime readiness
scenario-lab runtime readiness
explanation runtime readiness
```

A missing runtime should not block real-data prior-layer evaluation.

### 2.4 Use pretrained imaging models first

Do not train a new breast MRI segmentation model during V1.

Use MAMA-MIA pretrained nnU-Net as the first segmentation adapter. Treat it as an external inference dependency that produces tumor masks and cached MRI features. Future versions may compare against other pretrained breast MRI segmenters or fine-tune models, but that is not a V1 requirement.

---

## 3. Existing Prior-Builder Layer Stack

The V1 prior-builder stack is organized as follows.

### Layer 0: Parameter contract

Module:

```text
experiments/prior_builder/parameter_contract.py
```

Responsibilities:

```text
define learnable simulator parameters
define treatment/subtype scope
resolve whether a patient is in V1-A scope
provide conservative fallback for unsupported contexts
```

V1-A intended scope:

```text
TNBC or resolvable TNBC-like receptor profile
neoadjuvant chemotherapy
A/C-T-like treatment context when available
```

Primary learnable parameters:

```text
growth_rate_per_day
active_treatment_sensitivity
resistant_fraction
```

### Layer 1: Bounds and observation-noise policy

Module:

```text
experiments/prior_builder/bounds.py
```

Responsibilities:

```text
define hard and soft bounds for transformed parameters
define observation-noise policy
apply conservative behavior when quality or scope is uncertain
```

### Layer 2: Population prior

Module:

```text
experiments/prior_builder/population_prior.py
```

Responsibilities:

```text
sample prior distributions from population-level assumptions
produce prior predictive samples before patient-specific pathology/MRI shifts
support prior predictive evaluation on held-out tumor-volume observations
```

### Layer 3: Pathology and biomarker rules

Module:

```text
experiments/prior_builder/pathology_biomarker_rules.py
```

Responsibilities:

```text
shift transformed-space priors using pathology and biomarker fields
handle receptor status, grade, Ki-67, BRCA/HRD, and related fields
avoid claiming precision when biomarkers are missing
```

### Layer 4: MRI feature and QC rules

Module:

```text
experiments/prior_builder/mri_feature_rules.py
```

Responsibilities:

```text
use MRI-derived features as weak evidence
widen uncertainty for poor QC or conflicting features
mark report-only behavior when MRI quality is insufficient
avoid overconfident parameter shifts from noisy segmentations
```

Layer 4 should consume cached features only. It should not call the image segmenter or open MRI files during prior-stack evaluation.

### Adapter: simulator conversion

Module:

```text
experiments/prior_builder/adapter_to_volume_ode.py
```

Responsibilities:

```text
convert V1 prior samples to simulator parameters
enforce simulator-compatible parameter naming and bounds
support prior predictive forecasting on longitudinal tumor-volume data
```

---

## 4. V1 Data Readiness Levels

V1 should explicitly track data readiness levels.

| Level | Name | Required for V1? | Description |
| --- | --- | --- | --- |
| D0 | Metadata inventory | Yes | Known datasets, local paths, licenses, columns, source provenance |
| D1 | Real longitudinal cohort table | Yes | Patient-level rows with baseline and held-out tumor volume |
| D2 | Curated MRI feature table | Recommended | Tumor volume, enhancement, QC, and segmentation-derived features |
| D3 | Raw MRI subset | Optional | Small DICOM/NIfTI subset for ingestion and segmentation testing |
| D4 | Full raw MRI archive | No for V1 | Multi-TB reproducibility path; defer unless needed |

V1 success should require D1. D2 is recommended for MRI-layer evaluation. D3 is useful for integration tests. D4 is not a V1 blocker.

---

## 5. Dataset Policy

### 5.1 I-SPY2

I-SPY2 is the primary target for real longitudinal neoadjuvant breast cancer response evaluation.

Use I-SPY2 in this order:

```text
1. Processed clinical, outcome, volume, and FTV metadata
2. BreastDCEDL_ISPY2 curated NIfTI/metadata, where useful
3. Small raw I-SPY2 subset for raw-ingestion testing
4. Full raw I-SPY2 only for later reproducibility work
```

Do not require full raw I-SPY2 DICOM for V1.

Known source links:

```text
TCIA I-SPY2:
https://www.cancerimagingarchive.net/collection/ispy2/

IDC I-SPY2:
https://portal.imaging.datacommons.cancer.gov/collections/ispy2/
```

Notes:

```text
TCIA describes I-SPY2 as a public DCE-MRI breast-cancer collection.
IDC warns that full I-SPY2 image download can exceed 1 TB.
A local NBIA query may return millions of DICOM instances if all MR series are requested.
```

### 5.2 BreastDCEDL / BreastDCEDL_ISPY2

BreastDCEDL and BreastDCEDL_ISPY2 should be preferred over full raw DICOM where possible because they provide curated, deep-learning-ready NIfTI volumes and harmonized metadata.

Use BreastDCEDL_ISPY2 for:

```text
curated I-SPY2 imaging baseline
NIfTI ingestion tests
tumor-volume and pCR metadata alignment
future image-model benchmarking
pretraining or pCR baseline comparisons
```

Known source links:

```text
TCIA BreastDCEDL_ISPY2:
https://www.cancerimagingarchive.net/analysis-result/breastdcedl_ispy2/

BreastDCEDL GitHub:
https://github.com/naomifridman/BreastDCEDL
```

### 5.3 MAMA-MIA

MAMA-MIA should be the default V1 segmentation integration target.

Use MAMA-MIA for:

```text
pretrained breast tumor segmentation
tumor-mask generation
tumor-volume extraction
segmentation QC
feature-generation pipeline validation
future segmentation benchmarking
```

Known source link:

```text
MAMA-MIA GitHub:
https://github.com/LidiaGarrucho/MAMA-MIA
```

Policy:

```text
Do not train a segmentation model from scratch during V1.
Use pretrained MAMA-MIA nnU-Net inference first.
Cache masks and features.
Spot-check masks manually before using MRI features in performance claims.
```

### 5.4 QIN Breast DCE-MRI

Use QIN Breast DCE-MRI as an external sanity-check dataset if storage allows.

Use cases:

```text
longitudinal response sanity check
calibration check outside I-SPY2 conventions
MRI preprocessing integration test
```

### 5.5 RIDER Breast MRI

Use RIDER Breast MRI as a small measurement-noise and test-retest style dataset.

Use cases:

```text
observation-noise calibration
repeatability checks
MRI feature stability checks
```

RIDER should not be the primary treatment-response cohort.

### 5.6 Full raw DICOM policy

Full raw DICOM archives are optional and deferred.

Default policy:

```text
Do not require full raw I-SPY2 or full raw Duke for V1.
Use curated metadata and curated NIfTI where possible.
Use raw DICOM subsets only when testing raw-ingestion code.
```

---

## 6. Recommended Storage and Compute Policy

### 6.1 Storage modes

| Mode | Expected contents | Approximate scale |
| --- | --- | --- |
| Table-only V1 | cohort JSONL, reports, summaries | <5 GB |
| Curated eval plus QIN/RIDER | metadata, smaller DICOM/NIfTI, features | 20-150 GB |
| Practical MRI V1/V2 | BreastDCEDL_ISPY2, MAMA-MIA weights, masks, features | 150-500 GB |
| Serious imaging development | multiple curated datasets, masks, checkpoints | 500 GB-1.5 TB |
| Full raw reproducibility | raw I-SPY2/Duke plus processed copies | multi-TB |

### 6.2 Guidance for 2x L40 GPUs and 150 GB disk

Two L40 GPUs are sufficient for practical MAMA-MIA inference, moderate fine-tuning, feature extraction workflows, and future deep response-model experiments.

The limiting resource is disk, not GPU.

With approximately 150 GB disk:

```text
keep cohort tables
keep MAMA-MIA weights
keep RIDER
keep QIN if final size fits
use BreastDCEDL_ISPY2 or a subset, not multiple full curated datasets
avoid full raw I-SPY2
avoid full raw Duke
delete intermediate nnU-Net preprocessing caches after feature extraction
store only final masks, features, summaries, and reports when possible
```

Recommended minimum for comfortable MRI development:

```text
500 GB SSD minimum workable
1-2 TB SSD recommended
3-6+ TB only for full raw archive reproducibility work
```

Known source link:

```text
NVIDIA L40 datasheet:
https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf
```

---

## 7. Data Directory Standard

Use this layout:

```text
data/
  raw/
    tcia/
    mamamia/
    breastdcedl/
  curated/
    manifests/
    cohorts/
    images_nifti/
    nnunet_inputs/
    segmentations/
    features/
  processed/
    v1_prior_stack/
```

Directory responsibilities:

```text
data/raw/
  original downloads, never edited

data/curated/manifests/
  download inventories, case maps, provenance, expected series counts

data/curated/cohorts/
  intermediate cleaned cohort tables

data/curated/images_nifti/
  converted or curated NIfTI image files

data/curated/nnunet_inputs/
  model-ready inputs for MAMA-MIA/nnU-Net

data/curated/segmentations/
  predicted or provided tumor masks

data/curated/features/
  small feature tables derived from MRI and masks

data/processed/v1_prior_stack/
  final eval-ready JSONL files
```

Invariant:

```text
V1 prior evals read only processed cohort JSONL and cached feature tables.
V1 prior evals do not read raw DICOM directly.
```

---

## 8. Primary V1 Cohort Artifact

The primary real-data artifact is:

```text
data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.jsonl
```

Each row should represent one real patient/case.

### 8.1 Minimum required fields

```text
case_id
data_origin
subtype
treatment_regimen or treatment_context
baseline_day
baseline_volume_ml
final_day
final_volume_ml
```

### 8.2 Recommended fields

```text
early_day
early_volume_ml
er_status
pr_status
her2_status
grade
ki67_percent
brca_status
hrd_status
pathologic_complete_response
volume_ml
functional_tumor_volume_ml
enhancement_mean
enhancement_std
low_enhancement_fraction
segmentation_qc
registration_qc
mri_source
source_metadata_file
```

### 8.3 Example row

```json
{
  "case_id": "ISPY2_001",
  "data_origin": "ISPY2",
  "subtype": "TNBC",
  "treatment_regimen": "A/C-T neoadjuvant chemotherapy",
  "baseline_day": 0,
  "baseline_volume_ml": 28.0,
  "early_day": 21,
  "early_volume_ml": 18.5,
  "final_day": 126,
  "final_volume_ml": 7.5,
  "er_status": "negative",
  "pr_status": "negative",
  "her2_status": "negative",
  "grade": 3,
  "ki67_percent": 45,
  "volume_ml": 28.0,
  "functional_tumor_volume_ml": 24.0,
  "segmentation_qc": "high",
  "registration_qc": "medium",
  "mri_source": "curated_metadata"
}
```

---

## 9. Required Curation Scripts

Add or maintain these scripts.

### Required for V1-D1

```text
scripts/build_v1_prior_eval_cohort.py
scripts/inspect_v1_data_columns.py
```

### Required for V1-D2 MRI-feature integration

```text
scripts/merge_mri_features_into_v1_cohort.py
scripts/mri_preprocessing/convert_dicom_to_nifti.py
scripts/mri_preprocessing/prepare_mamamia_nnunet_inputs.py
scripts/mri_preprocessing/run_mamamia_inference.sh
scripts/mri_preprocessing/extract_mri_features.py
scripts/mri_preprocessing/qc_mri_features.py
```

### 9.1 Cohort builder outputs

The cohort builder should write:

```text
data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.jsonl
data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.exclusions.jsonl
data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.summary.json
```

### 9.2 Cohort summary fields

The summary JSON should include:

```text
total_input_rows
included_rows
excluded_rows
tnbc_count
non_tnbc_count
v1a_in_scope_count
baseline_volume_available_count
final_volume_available_count
early_followup_available_count
biomarker_completeness
mri_feature_completeness
excluded_reason_counts
source_files
created_at
```

### 9.3 Exclusion reasons

Use stable exclusion reason labels:

```text
missing_case_id
missing_baseline_volume
missing_final_volume
non_positive_baseline_volume
non_positive_final_volume
invalid_time_order
unresolved_subtype
out_of_v1a_scope
synthetic_or_demo_data
duplicate_case_id
unsupported_treatment_context
```

---

## 10. Curation Checks

The cohort builder must enforce:

```text
real data only by default
no demo/synthetic/fixture paths unless explicitly allowed
patient-level rows only
baseline volume > 0
final volume > 0
final_day > baseline_day
case_id present and stable
subtype resolvable
treatment context resolvable
V1-A in-scope flag for TNBC plus A/C-T-like chemotherapy
clear reason for excluded rows
```

Recommended assertions:

```text
no duplicated case_id in included cohort
no train/test leakage by scan/timepoint
no synthetic/demo tokens in path or row content unless allow_demo_data=true
all volumes in mL or explicitly converted to mL
all days relative to treatment baseline or explicitly documented
```

---

## 11. MRI Preprocessing Lane

MRI preprocessing should be implemented as a separate cached lane.

### 11.1 Pipeline

```text
DICOM or curated NIfTI
  -> normalized NIfTI
  -> MAMA-MIA/nnU-Net input format
  -> pretrained nnU-Net tumor mask
  -> feature extraction
  -> feature JSONL
  -> merge into V1 cohort
```

### 11.2 Preferred input sources

Use this order:

```text
1. curated NIfTI datasets where available
2. BreastDCEDL_ISPY2 curated imaging
3. MAMA-MIA-compatible data
4. QIN/RIDER selected cases
5. raw I-SPY2 selected subset
6. full raw I-SPY2 only in later reproducibility phase
```

### 11.3 MAMA-MIA adapter

Add a replaceable adapter around MAMA-MIA inference. Suggested structure:

```text
experiments/mri_ingestion/
  segmenters/
    mamamia_nnunet.py
  features/
    tumor_volume.py
    enhancement_features.py
    qc.py
  schemas.py
```

The adapter should expose a stable interface:

```text
input:
  case_id
  image_path_or_phase_paths
  output_dir
  model_config

output:
  segmentation_path
  inference_metadata
  warnings
```

Do not couple the prior stack directly to MAMA-MIA internals.

### 11.4 MAMA-MIA inference command shape

The exact command should be maintained in:

```text
scripts/mri_preprocessing/run_mamamia_inference.sh
```

Expected command shape:

```bash
nnUNetv2_predict \
  -i data/curated/nnunet_inputs \
  -o data/curated/segmentations/mamamia_nnunet \
  -d 101 \
  -c 3d_fullres
```

The script should validate that:

```text
nnUNetv2_predict is available
nnUNet_results is set
model weights exist
input files are compressed NIfTI when required
output directory is writable
```

---

## 12. MRI Feature Table Contract

The MRI feature table should be:

```text
data/curated/features/mri_features.jsonl
```

Each row should contain one case-level feature record.

Example:

```json
{
  "case_id": "ISPY2_001",
  "source_image": "data/curated/images_nifti/ISPY2_001.nii.gz",
  "source_mask": "data/curated/segmentations/mamamia_nnunet/ISPY2_001.nii.gz",
  "tumor_volume_ml": 28.0,
  "functional_tumor_volume_ml": 24.0,
  "mask_voxels": 28000,
  "voxel_volume_ml": 0.001,
  "enhancement_mean": 1.35,
  "enhancement_std": 0.22,
  "low_enhancement_fraction": 0.18,
  "connected_component_count": 1,
  "segmentation_qc": "medium",
  "registration_qc": "unknown",
  "warnings": []
}
```

Initial feature set:

```text
tumor_volume_ml
functional_tumor_volume_ml
enhancement_mean
enhancement_std
low_enhancement_fraction
mask_voxels
voxel_volume_ml
connected_component_count
segmentation_qc
registration_qc
```

Feature naming should align with the fields consumed by Layer 4.

---

## 13. MRI QC Gates

Before using MRI-derived features in V1 evals, require:

```text
non-empty tumor mask
clinically plausible tumor volume
valid voxel spacing
valid image affine
no obvious all-zero image
no obvious full-volume mask
connected component count recorded
manual spot-check on at least 10-20 cases
feature provenance saved
```

Recommended QC labels:

```text
high
medium
low
failed
unknown
```

Layer 4 behavior by QC:

| QC | Layer 4 behavior |
| --- | --- |
| high | feature may influence uncertainty according to rules |
| medium | conservative feature use; mild widening allowed |
| low | uncertainty widening only or report-only |
| failed | ignore MRI feature for parameter shifts; report failure |
| unknown | conservative/report-only behavior |

Layer 4 should never silently treat low-quality MRI-derived features as reliable.

---

## 14. Evaluation Categories

V1 should include the following eval categories.

### 14.1 Real-data prior-layer performance

Runner:

```text
evals/prior_stack/v1_real_data_eval.py
```

Input:

```text
real longitudinal cohort JSONL/CSV
```

Required behavior:

```text
reject demo/synthetic data by default
compute non-model baselines
compute Layer 2, Layer 3, and Layer 4 prior-predictive metrics
report Layer 3 vs Layer 2 deltas
report Layer 4 vs Layer 3 deltas
report cases helped and harmed
```

Metrics:

```text
MAE
RMSE
log-volume RMSE
MAPE
80% interval coverage
95% interval coverage
80% interval width
```

Baselines:

```text
baseline_no_change
linear_early
exponential_early
```

### 14.2 Uncertainty calibration

Runner:

```text
evals/prior_stack/v1_uncertainty_calibration_eval.py
```

Required behavior:

```text
evaluate 80% and 95% coverage
report interval widths
report subgroup calibration when sample size allows
flag overconfident behavior
```

### 14.3 Posterior health

Runner:

```text
evals/prior_stack/v1_posterior_health_eval.py
```

V1 status:

```text
unavailable until posterior-update runtime exists
```

Future requirements:

```text
particle degeneracy diagnostics
effective sample size
posterior contraction
posterior predictive checks
failure-mode reporting
```

### 14.4 Sequential forecasting

Runner:

```text
evals/prior_stack/v1_sequential_forecasting_eval.py
```

V1 status:

```text
unavailable until sequential update runtime exists
```

Future requirements:

```text
baseline-only prediction
baseline plus early follow-up prediction
forecast horizon metrics
patient-level temporal splits
```

### 14.5 Update value

Runner:

```text
evals/prior_stack/v1_update_value_eval.py
```

V1 status:

```text
unavailable until posterior-update runtime exists
```

Future requirements:

```text
value of biomarkers
value of early volume
value of MRI features
value of QC-filtered MRI features
```

### 14.6 Scenario lab

Runner:

```text
evals/prior_stack/v1_scenario_lab_eval.py
```

V1 status:

```text
unavailable until scenario runtime exists
```

Future requirements:

```text
alternative treatment schedule simulation
posterior-particle scenario summaries
stability diagnostics
report uncertainty rather than point recommendation
```

### 14.7 Explanation quality

Runner:

```text
evals/prior_stack/v1_explanation_quality_eval.py
```

V1 status:

```text
unavailable until explanation runtime exists
```

Future requirements:

```text
explanation faithfulness
evidence provenance
uncertainty wording
missing-data transparency
```

---

## 15. Eval Suite Runner

Runner:

```text
evals/prior_stack/run_v1_eval_suite.py
```

Expected behavior:

```text
run all available V1 eval categories
mark missing cohort as unavailable, not failed
mark missing runtime-dependent categories as unavailable, not failed
write markdown report
write machine-readable summary if supported
return nonzero only on actual unexpected failure
```

Example command:

```bash
python3 -m evals.prior_stack.run_v1_eval_suite \
  --cohort data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.jsonl \
  --report evals/reports/v1_eval_suite.md \
  --n-samples 2000 \
  --seed 2026
```

Expected V1-D1 result:

```text
real_data_prior_layer_performance: pass
uncertainty_calibration: pass
posterior_health: unavailable
sequential_forecasting: unavailable
update_value: unavailable
scenario_lab: unavailable
explanation_quality: unavailable
```

---

## 16. Implementation Milestones

### Milestone V1-D0: dataset inventory

Goal:

```text
know what data exists locally and what columns are available
```

Required outputs:

```text
data/curated/manifests/dataset_inventory.json
data/curated/manifests/local_download_status.json
```

Acceptance criteria:

```text
I-SPY2 metadata source identified
QIN download status known
RIDER download status known
MAMA-MIA/BreastDCEDL acquisition plan documented
full raw I-SPY2 explicitly marked optional/deferred
```

### Milestone V1-D1: real-data cohort eval

Goal:

```text
run V1 prior-layer evals on a real longitudinal cohort table
```

Required artifact:

```text
data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.jsonl
```

Acceptance criteria:

```text
at least 50 real in-scope TNBC chemotherapy cases preferred
baseline and held-out final volume available
real_data_prior_layer_performance passes
uncertainty_calibration passes
demo/synthetic rows rejected by default
exclusion report written
cohort summary written
```

Minimum acceptable development threshold:

```text
10-25 real cases for pipeline smoke test only
50+ real cases for initial V1 performance claims
150-300+ real cases for stronger benchmark claims
```

### Milestone V1-D2: MRI-feature cohort eval

Goal:

```text
re-run V1 evals with MRI-derived cached features available
```

Required artifacts:

```text
data/curated/features/mri_features.jsonl
data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort_with_mri.jsonl
```

Acceptance criteria:

```text
feature completeness reported
MRI QC flags propagate into eval reports
Layer 4 vs Layer 3 deltas reported
cases helped/harmed listed
low/failed QC features do not create overconfident shifts
```

### Milestone V1-D3: MAMA-MIA segmentation adapter smoke test

Goal:

```text
run MAMA-MIA pretrained nnU-Net on a small curated subset
```

Suggested scale:

```text
5-10 cases first
25-50 cases after smoke test
50-100 cases before broad MRI-feature eval claims
```

Acceptance criteria:

```text
nnU-Net inference completes
masks are non-empty
feature extractor produces valid rows
manual spot-check passes on sampled cases
features merge cleanly into cohort JSONL
```

### Milestone V1-D4: posterior/scenario runtime planning

Goal:

```text
define runtime interfaces for future posterior updates and scenario lab
```

Acceptance criteria:

```text
posterior-update interface documented
scenario-lab interface documented
explanation interface documented
current eval stubs continue to report unavailable until implemented
```

---

## 17. Recommended Development Order

1. Ensure eval runners exist and use real data only by default.
2. Add cohort builder summary and exclusion reports.
3. Build the I-SPY2 V1 cohort JSONL from metadata.
4. Run V1-D1 evals.
5. Add MRI feature table contract.
6. Add feature merge script.
7. Add MAMA-MIA segmentation adapter.
8. Run MAMA-MIA inference on 5-10 cases.
9. Extract MRI features and write JSONL.
10. Merge MRI features into V1 cohort.
11. Run V1-D2 evals.
12. Scale to 50-100 MRI-feature cases.
13. Defer full posterior/scenario/explanation evals until runtimes exist.
14. Defer full raw I-SPY2/Duke processing until storage and specific use case justify it.

---

## 18. Non-Goals for V1

V1 should not require:

```text
training a new MRI segmentation model
full raw I-SPY2 DICOM download
full raw Duke DICOM download
multi-GPU segmentation training
manual segmentation at scale
end-to-end image-to-treatment recommendation
real-time clinical deployment
```

V1 should prove:

```text
real data can be curated into a stable cohort contract
prior layers can be evaluated on held-out tumor volumes
uncertainty calibration is measurable
MRI features can be integrated as cached QC-aware inputs
the system fails closed when data quality is poor
the eval suite distinguishes unavailable runtimes from failures
```

---

## 19. Test Requirements

### 19.1 Cohort loader tests

Required tests:

```text
loads JSONL cohort
loads CSV cohort
rejects synthetic/demo data by default
allows synthetic/demo data only with explicit flag
rejects missing baseline/final volume
rejects invalid time order
records exclusion reasons
```

### 19.2 Prior-layer eval tests

Required tests:

```text
Layer 2 prior predictive metrics computed
Layer 3 deltas computed
Layer 4 deltas computed when MRI fields exist
out-of-scope cases excluded or reported separately
metrics stable under fixed seed
```

### 19.3 Uncertainty calibration tests

Required tests:

```text
80% coverage computed
95% coverage computed
interval width computed
empty input handled clearly
small cohort warnings emitted
```

### 19.4 MRI feature tests

Required tests:

```text
feature JSONL schema validated
non-empty masks produce positive volume
empty masks marked failed
low QC maps to conservative Layer 4 behavior
feature merge preserves patient-level rows
missing feature rows do not drop cohort rows unless requested
```

### 19.5 Suite tests

Required tests:

```text
no cohort -> real-data eval unavailable
valid cohort -> real-data eval passes
missing posterior runtime -> posterior eval unavailable
suite report written
unexpected exceptions fail loudly
```

---

## 20. Report Requirements

The V1 suite report should include:

```text
cohort path
number of input rows
number of included rows
number of excluded rows
exclusion reason counts
in-scope V1-A case count
biomarker completeness
MRI feature completeness
baseline metrics
Layer 2 metrics
Layer 3 metrics
Layer 4 metrics
uncertainty coverage
cases helped/harmed
unavailable eval categories and why
source provenance
seed
n_samples
created_at
```

The report should clearly separate:

```text
performance results
calibration results
data-quality warnings
runtime-unavailable categories
```

---

## 21. Acceptance Criteria for V1-A

V1-A is successful when:

```text
a real-data cohort JSONL exists
the cohort builder writes summary and exclusion artifacts
the eval suite runs on the real cohort
real_data_prior_layer_performance produces metrics
uncertainty_calibration produces coverage metrics
demo/synthetic data is rejected by default
Layer 2/3/4 behavior is separately reported
Layer 4 uses cached MRI features only
runtime-dependent evals are marked unavailable until implemented
```

V1-A is not required to:

```text
train MRI models
download full raw I-SPY2
implement full posterior updating
implement full scenario lab
implement clinical explanations
```

---

## 22. Source Links

The following sources informed the data and imaging strategy:

```text
TCIA I-SPY2:
https://www.cancerimagingarchive.net/collection/ispy2/

IDC I-SPY2:
https://portal.imaging.datacommons.cancer.gov/collections/ispy2/

TCIA BreastDCEDL_ISPY2:
https://www.cancerimagingarchive.net/analysis-result/breastdcedl_ispy2/

BreastDCEDL GitHub:
https://github.com/naomifridman/BreastDCEDL

MAMA-MIA GitHub:
https://github.com/LidiaGarrucho/MAMA-MIA

NVIDIA L40 datasheet:
https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf
```

---

## 23. Practical Current Recommendation

For the current project state:

```text
keep RIDER
finish QIN if storage allows
do not continue full raw I-SPY2 unless multi-TB storage is available
build the processed I-SPY2 cohort table first
use MAMA-MIA pretrained nnU-Net for segmentation integration
cache final masks and feature JSONL
delete large intermediate preprocessing caches when disk is constrained
```

With approximately 150 GB disk, prioritize:

```text
cohort tables
QIN/RIDER if they fit
MAMA-MIA pretrained weights
one curated imaging dataset or selected subset
final masks
feature JSONL
eval reports
```

Avoid:

```text
full raw I-SPY2
full raw Duke
duplicated raw plus NIfTI plus nnU-Net preprocessed copies at scale
```