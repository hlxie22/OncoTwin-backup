# V1 MRI preprocessing lane

This lane creates cached MRI feature rows for the table-first V1 prior stack. It
is intentionally separate from prior/eval code: segmentation and image-derived
features write JSONL artifacts that can be reviewed, QCed, and merged into the
V1 cohort later.

## Main artifacts

```text
data/curated/nnunet_inputs/
data/curated/segmentations/mamamia_nnunet/
data/curated/features/mri_features.jsonl
```

## Workflow

Prepare MAMA-MIA/nnU-Net inputs from a JSONL manifest of curated NIfTI files:

```bash
python3 scripts/mri_preprocessing/prepare_mamamia_nnunet_inputs.py \
  --manifest data/curated/manifests/mri_cases.jsonl \
  --output-dir data/curated/nnunet_inputs \
  --prepared-manifest data/curated/manifests/mamamia_prepared_inputs.jsonl
```

Run pretrained MAMA-MIA/nnU-Net inference after installing nnU-Net and setting
`nnUNet_results`:

```bash
bash scripts/mri_preprocessing/run_mamamia_inference.sh \
  data/curated/nnunet_inputs \
  data/curated/segmentations/mamamia_nnunet
```

Extract cached features from mask/intensity metadata:

```bash
python3 scripts/mri_preprocessing/extract_mri_features.py \
  --input data/curated/features/mri_feature_metadata.jsonl \
  --output data/curated/features/mri_features.raw.jsonl
```

Apply fail-closed MRI QC gates before merging features into the V1 cohort:

```bash
python3 scripts/mri_preprocessing/qc_mri_features.py \
  --input data/curated/features/mri_features.raw.jsonl \
  --output data/curated/features/mri_features.jsonl
```

## Feature metadata input

The lightweight extractor expects metadata that can be produced by a local image
reader or a reviewed spreadsheet. Useful fields include:

```text
case_id
source_image
source_mask
mask_voxels
voxel_volume_ml or voxel_spacing_mm
image_voxels
connected_component_count
enhancement_values or enhancement_mean/enhancement_std/low_enhancement_fraction
segmentation_qc
registration_qc
warnings
```

Rows with missing/empty masks, implausible tumor volumes, invalid voxel geometry,
or failed QC are marked `mri_feature_status=failed` and `layer4_feature_policy=report_only`.
Low-quality but nonfailed rows are marked `layer4_feature_policy=uncertainty_only`.
