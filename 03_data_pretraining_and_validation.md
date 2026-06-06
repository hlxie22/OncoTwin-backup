# Data, Pretraining, and Validation Plan

## Dataset strategy

No single public dataset gives everything needed for the full product. The strongest approach is to combine public datasets by role:

| Role | Primary dataset(s) | Why it matters |
|---|---|---|
| Longitudinal treatment-response modeling | I-SPY2 | Longitudinal breast MRI during neoadjuvant therapy, treatment response setting, pCR/outcome labels |
| Deep-learning-ready baseline imaging | BreastDCEDL / BreastDCEDL_ISPY2 | Standardized NIfTI DCE-MRI volumes, segmentations, harmonized metadata |
| Tumor segmentation pretraining | MAMA-MIA | Large breast DCE-MRI segmentation benchmark with expert masks and pretrained nnU-Net weights |
| Secondary validation | I-SPY1 | Smaller but relevant neoadjuvant breast DCE-MRI dataset with segmentations and molecular/follow-up data |
| DWI/ADC cellularity features | ACRIN-6698, if included | Diffusion-weighted imaging features can support cellularity proxies |
| Molecular graph pretraining | TCGA-BRCA, METABRIC | Mutation/expression/pathway relationships and subtype biology |

## Recommended public datasets

### I-SPY2

Use I-SPY2 as the anchor dataset for longitudinal response modeling. It is a multi-center breast DCE-MRI collection in the neoadjuvant chemotherapy setting with imaging, segmentations, demographic, molecular test, follow-up, treatment, radiomic feature, and measurement data.

Use it for:

```text
T0 baseline MRI
T1 early-treatment MRI
T2 inter-regimen/mid-treatment MRI
T3 presurgery MRI
pathologic complete response or residual disease outcome
```

### BreastDCEDL_ISPY2

Use this for deep-learning-ready pretraining and benchmarking. It contains standardized 3D NIfTI volumes, tumor annotations, voxel-based tumor volumes, and harmonized clinicopathologic metadata including HR status, HER2 status, and pCR outcomes.

Use it for:

```text
baseline DCE-MRI encoder training
pCR baseline model
tumor phenotype embedding
parameter-amortizer pretraining
```

### MAMA-MIA

Use MAMA-MIA for segmentation. It contains 1,506 breast DCE-MRI cases with expert tumor segmentations and pretrained nnU-Net weights.

Use it for:

```text
tumor segmentation pretraining
segmentation confidence estimation
MRI quality-control experiments
tumor-volume extraction pipeline validation
```

### TCGA-BRCA and METABRIC

Use these for molecular graph pretraining and pathway-level biology. They are not ideal for longitudinal neoadjuvant MRI response, but they are valuable for learning relationships between breast cancer subtype, mutation patterns, expression signatures, and pathways.

Use them for:

```text
molecular graph node embeddings
subtype representations
pathway priors
mutation-to-phenotype parameter modifiers
```

## Pretrained models to start with

### 1. MAMA-MIA pretrained nnU-Net

This should be the first segmentation backbone. It is breast DCE-MRI specific and directly aligned with the tumor-segmentation problem.

Expected role:

```text
MRI volume → tumor mask → tumor volume → spatial initial condition for the twin
```

### 2. TNBC_DigitalTwins codebase

Use this as a reference for mechanistic digital-twin implementation patterns. It includes files such as `DigitalTwin.py`, `ForwardModels.py`, `Calibrations.py`, `ReducedModel.py`, and `Optimize.py`, and can guide the design of calibration and simulation workflows.

Expected role:

```text
reaction-diffusion solver structure
Bayesian calibration ideas
reduced-order modeling approach
schedule simulation logic
```

### 3. MedicalNet 3D ResNet

Use as a general volumetric medical-imaging encoder baseline. It is not breast-specific, but it provides pretrained 3D ResNet weights that can be fine-tuned on DCE-MRI.

Expected role:

```text
baseline 3D image encoder
initial representation model
comparison against self-supervised DCE-MRI pretraining
```

### 4. MONAI Model Zoo / Swin UNETR

Use MONAI as the medical-imaging framework and experiment infrastructure. Use Swin UNETR or other MONAI bundles as secondary experiments, not as the first breast tumor segmentation model.

Expected role:

```text
model bundles
training/inference utilities
3D transformer experiments
segmentation alternatives
```

## Training stages

### Stage A: data harmonization

Inputs:

```text
DICOM/NIfTI MRI
clinical CSV/metadata
tumor masks if available
pathology labels
outcome labels
```

Outputs:

```text
standardized NIfTI volumes
timepoint labels
harmonized pathology table
treatment-time metadata
case-level splits
```

### Stage B: segmentation model

Train/fine-tune tumor segmentation.

Inputs:

```text
DCE-MRI volumes
expert tumor masks
```

Outputs:

```text
tumor mask
tumor volume
segmentation confidence
quality-control flags
```

Metrics:

```text
Dice score
Hausdorff distance
absolute volume error
relative volume error
false-negative tumor rate
confidence calibration
```

### Stage C: response representation model

Train an MRI encoder using baseline imaging, tumor masks, and outcomes.

Tasks:

```text
pCR classification
residual tumor volume prediction
subtype prediction as auxiliary task
functional tumor volume regression if available
```

The goal is to learn a useful latent representation, not to replace the simulator.

### Stage D: mechanistic calibration

For each longitudinal case, fit parameters that make the mechanistic simulator match observed tumor response.

Inputs:

```text
baseline tumor mask or volume
timepoint tumor volumes
optional spatial residual masks
treatment schedule
outcome label
```

Outputs:

```text
patient-specific proliferation estimate
patient-specific diffusion/invasion estimate
patient-specific drug-sensitivity estimates
uncertainty over parameters
```

### Stage E: AI parameter amortization

Train an AI model to map baseline multimodal data to the fitted mechanistic parameter distributions.

Inputs:

```text
baseline MRI embedding
pathology features
molecular features
age/context features
```

Targets:

```text
posterior parameter means
posterior parameter variances
or sampled parameter ensembles
```

### Stage F: end-to-end simulation-aware fine-tuning

Train the AI amortizer and simulator together, where possible, using a combined loss:

```text
volume trajectory loss
spatial residual-risk loss
pCR / residual outcome loss
parameter regularization
uncertainty calibration loss
molecular consistency loss
```

## Data splits

Use patient-level splits only. Never split by scan when scans from the same patient can appear in multiple sets.

Recommended split strategy:

```text
Train: 70%
Validation: 15%
Test: 15%
External validation: I-SPY1 or held-out site/cohort where possible
```

Also evaluate out-of-distribution behavior by site, scanner, subtype, race/ethnicity if labels are available, age group, and tumor size.

## Validation targets

The system should be evaluated at multiple layers.

### Segmentation validation

```text
Dice score
Hausdorff distance
volume error
quality-control sensitivity
```

### Simulator validation

```text
observed-vs-predicted tumor volume at T1/T2/T3
trajectory RMSE
residual volume error
spatial overlap of residual-risk map if spatial labels exist
```

### Classification validation

```text
pCR AUC
balanced accuracy
sensitivity/specificity
calibration curve
Brier score
```

### Uncertainty validation

```text
coverage of 80% / 90% prediction intervals
expected calibration error
negative log likelihood
uncertainty-error correlation
```

### Product validation

```text
Does the explanation match the model drivers?
Does the UI avoid treatment recommendations?
Do users understand uncertainty?
Do safety labels appear wherever needed?
```

## Known limitations

- Public datasets may not contain every molecular marker needed for full graph personalization.
- MRI acquisition protocols and timepoint definitions vary by cohort.
- Treatment regimen details may be incomplete or simplified in public datasets.
- pCR labels are endpoints and do not always describe spatial residual disease.
- Patient-reported toxicity data may require separate datasets or app-native data collection.
- Public cohorts may not represent all populations equally.

## Source references

- I-SPY2 TCIA: https://www.cancerimagingarchive.net/collection/ispy2/
- BreastDCEDL_ISPY2 TCIA: https://www.cancerimagingarchive.net/analysis-result/breastdcedl_ispy2/
- BreastDCEDL Nature Scientific Data: https://www.nature.com/articles/s41597-026-06589-6
- BreastDCEDL GitHub: https://github.com/naomifridman/BreastDCEDL
- MAMA-MIA GitHub: https://github.com/LidiaGarrucho/MAMA-MIA
- MAMA-MIA Scientific Data: https://www.nature.com/articles/s41597-025-04707-4
- TNBC_DigitalTwins: https://github.com/cchristenson2/TNBC_DigitalTwins
- MedicalNet: https://github.com/Tencent/MedicalNet
- MONAI Model Zoo: https://project-monai.github.io/model-zoo.html
