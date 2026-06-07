# Data, Pretraining, and Validation Plan

## Dataset strategy

No single public dataset gives everything needed for the full product. The strongest approach is to combine public datasets by role:

| Role | Primary dataset(s) | Why it matters |
|---|---|---|
| Longitudinal treatment-response modeling | I-SPY2 | Longitudinal breast MRI during neoadjuvant therapy, treatment response setting, pCR/outcome labels |
| DWI/ADC cellularity (the simulator's state variable) | ACRIN-6698 | Serial diffusion-weighted MRI in I-SPY2; ADC gives the tumor-cell-density signal `N(x,t)` the mechanistic model actually evolves |
| Deep-learning-ready baseline imaging | BreastDCEDL / BreastDCEDL_ISPY2 | Standardized NIfTI DCE-MRI volumes (~2,070 across I-SPY1/I-SPY2/Duke), segmentations, harmonized metadata |
| Baseline imaging + radiogenomics | Duke-Breast-Cancer-MRI | 922 patients with baseline DCE-MRI plus clinical/pathology/treatment/outcome/genomic data and radiomic features; bridges imaging ↔ molecular |
| Tumor segmentation pretraining | MAMA-MIA | Large breast DCE-MRI segmentation benchmark (1,506 expert masks) with pretrained nnU-Net weights |
| Secondary / external validation | I-SPY1 | Smaller but relevant neoadjuvant breast DCE-MRI dataset with segmentations and molecular/follow-up data |
| Independent longitudinal calibration check | QIN-BREAST-DCE-MRI | Small (~10–20) but fully longitudinal (4 timepoints); use to sanity-check calibration, not to train |
| Observation / measurement-noise model | RIDER Breast MRI | Test-retest repeat scans; calibrates the observation likelihood used in Bayesian update (`07`) and SBI |
| Drug-sensitivity priors (α) | GDSC / DepMap / CCLE | Cell-line drug response by subtype; sets biologically grounded priors on drug-sensitivity parameters |
| Molecular graph pretraining | TCGA-BRCA, METABRIC (+ GENIE, CPTAC-BRCA) | Mutation/expression/pathway relationships and subtype biology; GENIE/CPTAC add genomic and proteogenomic depth |

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

The broader BreastDCEDL release (~2,070 patients across I-SPY1/I-SPY2/Duke) also exposes the **multi-phase DCE series**: its "Full" version preserves all acquired timepoints (3–12, typically ~7), while a lighter version keeps three (pre / early-post / late-post). This is the source for the deterministic DCE kinetic maps and the `delivery(x)` term (`04_…`). Note the temporal resolution is ~60–120 s, so kinetics are semi-quantitative (curve shape / AUC), not full pharmacokinetic rate constants.

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

### ACRIN-6698 (I-SPY2 DWI substudy)

Serial diffusion-weighted MRI acquired alongside DCE across I-SPY2 timepoints. This is a priority addition, not optional: the mechanistic state variable `N(x,t)` is tumor cell density, which is derived from ADC, so ACRIN-6698 supplies the signal the simulator is actually fit against.

Use it for:

```text
ADC-derived tumor cellularity maps for calibration
DWI features for the response encoder
spatial initial condition refinement
```

### Duke-Breast-Cancer-MRI

922 single-timepoint (pre-operative) cases with rich clinical, pathology, treatment, outcome, and genomic data plus radiomic features. Not longitudinal, so not for calibration, but valuable as a large encoder-pretraining and imaging↔molecular bridge cohort.

### QIN-BREAST-DCE-MRI and RIDER Breast MRI

```text
QIN-BREAST-DCE-MRI: fully longitudinal (4 timepoints) but small;
  use as an independent calibration sanity-check, not for training.
RIDER Breast MRI: test-retest repeat scans; use to estimate the
  measurement-noise / observation model for Bayesian update and SBI.
```

### GDSC / DepMap / CCLE

Cell-line drug-sensitivity panels. Not patient data, but they provide subtype-conditioned priors over the drug-sensitivity parameters (`α`) used by the residual-prior design in `06_ai_personalization_parameter_amortizer.md`.

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

### 3. Public 3D MRI foundation models (preferred image encoder)

Prefer adopting an off-the-shelf 3D MRI foundation model with open weights — **Triad**, **Decipher-MR**, or **MRI-CORE** — as the amortizer's image encoder, kept frozen (or lightly fine-tuned) with only a small task head trained on top. This is the **adopt-don't-build** stance: segmentation and generic imaging representation are commodity public SOTA, so bespoke encoder pretraining is a fallback, not the plan.

Expected role:

```text
frozen 3D image encoder feeding the parameter amortizer
representation reused across pCR/subtype auxiliary tasks
```

**MedicalNet 3D ResNet** remains a simpler, smaller baseline (pretrained 3D ResNet weights, not breast-specific) for comparison or low-resource settings.

Note: these foundation models are trained largely on single-volume structural MRI, so they do not model DCE contrast dynamics — those are supplied separately as deterministic kinetic-map channels (`04_…`), not learned.

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

Rather than a slow per-patient optimization, prefer **amortized calibration via simulation-based inference**: train a neural posterior estimator on unlimited synthetic `(θ, trajectory)` pairs, then apply it to each real longitudinal case to get an uncertainty-aware posterior over parameters in one pass. These posteriors become the targets for Stage E. See family A in `06_ai_personalization_parameter_amortizer.md`.

### Stage E: AI parameter amortization

Train an AI model to map baseline multimodal data to the fitted mechanistic parameter distributions. Because fully-multimodal cases with longitudinal outcomes number only in the hundreds, this stage follows the data-scarcity curriculum rather than naive supervised training.

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
or sampled parameter ensembles (from Stage D's SBI calibration)
```

Training strategy (full detail in `06_ai_personalization_parameter_amortizer.md`):

```text
self-supervised encoder pretraining on all unlabeled DCE-MRI
synthetic pretraining on simulated dynamics, then fine-tune on real cases
end-to-end fine-tuning through the differentiable simulator
biology-informed residual priors, modality dropout, hierarchical pooling
gated on sim-to-real and posterior-coverage checks
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
posterior coverage / simulation-based calibration (SBC) of the amortizer
sim-to-real check: synthetic-vs-real trajectory two-sample test
do learned priors beat a generic population prior on a held-out cohort?
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
- The binding constraint is not "breast MRI" in general but the **intersection** of modalities in the same patients: longitudinal MRI ∩ molecular ∩ outcome (∩ patient-reported outcomes) exists for only low hundreds of partial cases. This is why the amortizer relies on the simulation-based / self-supervised curriculum rather than naive supervised training.
- The curriculum's main risk is the **sim-to-real gap**: synthetic pretraining helps only if simulated trajectories resemble real ones. This must be checked (discriminator / two-sample tests, held-out-cohort validation, posterior-coverage calibration) before the synthetic priors are trusted.
- MRI acquisition protocols and timepoint definitions vary by cohort.
- Treatment regimen details may be incomplete or simplified in public datasets.
- pCR labels are endpoints and do not always describe spatial residual disease.
- Patient-reported toxicity data have no good public match and will likely require app-native collection; the toxicity twin should be treated as a later-phase feature.
- Public cohorts may not represent all populations equally.

## Source references

- I-SPY2 TCIA: https://www.cancerimagingarchive.net/collection/ispy2/
- ACRIN-6698 (I-SPY2 DWI substudy) TCIA: https://www.cancerimagingarchive.net/collection/acrin-6698/
- Duke-Breast-Cancer-MRI TCIA: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/
- QIN-BREAST-DCE-MRI TCIA: https://www.cancerimagingarchive.net/collection/qin-breast-dce-mri/
- RIDER Breast MRI TCIA: https://www.cancerimagingarchive.net/collection/rider-breast-mri/
- BreastDCEDL_ISPY2 TCIA: https://www.cancerimagingarchive.net/analysis-result/breastdcedl_ispy2/
- BreastDCEDL Nature Scientific Data: https://www.nature.com/articles/s41597-026-06589-6
- BreastDCEDL GitHub: https://github.com/naomifridman/BreastDCEDL
- MAMA-MIA GitHub: https://github.com/LidiaGarrucho/MAMA-MIA
- MAMA-MIA Scientific Data: https://www.nature.com/articles/s41597-025-04707-4
- TNBC_DigitalTwins: https://github.com/cchristenson2/TNBC_DigitalTwins
- MedicalNet: https://github.com/Tencent/MedicalNet
- MONAI Model Zoo: https://project-monai.github.io/model-zoo.html
- GDSC (drug sensitivity): https://www.cancerrxgene.org/
- DepMap / CCLE: https://depmap.org/portal/
- AACR Project GENIE: https://www.aacr.org/professionals/research/aacr-project-genie/
- CPTAC (proteogenomics): https://proteomics.cancer.gov/programs/cptac
- sbi (simulation-based inference): https://github.com/sbi-dev/sbi
- BayesFlow (amortized Bayesian inference): https://github.com/stefanradev93/BayesFlow
- SwinUNETR self-supervised pretraining (Tang et al., CVPR 2022): https://arxiv.org/abs/2111.14791
- MONAI self-supervised pretraining tutorial: https://github.com/Project-MONAI/tutorials/tree/main/self_supervised_pretraining
- Models Genesis (restoration-based 3D SSL): https://arxiv.org/abs/2004.07882
- Triad (3D MRI foundation model): https://arxiv.org/abs/2502.14064
- Decipher-MR (3D MRI vision-language foundation model): https://www.nature.com/articles/s41746-026-02596-4
- MRI-CORE (MRI foundation model): https://arxiv.org/abs/2506.12186
