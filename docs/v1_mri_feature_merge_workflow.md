# V1 MRI feature merge workflow

This workflow is the first V1-D2 bridge between cached MRI-derived features and
the prior-stack cohort consumed by:

```text
evals/prior_stack/v1_real_data_eval.py
evals/prior_stack/v1_uncertainty_calibration_eval.py
evals/prior_stack/run_v1_eval_suite.py
```

The merge step does not run segmentation or image preprocessing. It assumes that
an approved upstream process has already produced one case-level MRI feature row
per case.

## Input feature table

The preferred feature table path is:

```text
data/curated/features/mri_features.jsonl
```

Each row should include a stable `case_id` or `patient_id`. Useful fields are:

```text
source_image
source_mask
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
warnings
```

`segmentation_qc` should use one of:

```text
high
medium
low
failed
unknown
```

## Merge command

Merge cached features into the V1 prior eval cohort:

```bash
python3 scripts/merge_mri_features_into_v1_cohort.py \
  --cohort data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.jsonl \
  --mri-features data/curated/features/mri_features.jsonl \
  --output data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort_with_mri.jsonl
```

The default behavior preserves all cohort rows. Cases without an MRI feature row
receive a small `mri_features` object with:

```json
{
  "mri_feature_status": "missing"
}
```

To create a feature-complete analysis subset, explicitly pass:

```bash
python3 scripts/merge_mri_features_into_v1_cohort.py \
  --cohort data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort.jsonl \
  --mri-features data/curated/features/mri_features.jsonl \
  --output data/processed/v1_prior_stack/ispy2_v1_prior_eval_cohort_with_mri.complete.jsonl \
  --drop-missing-features
```

## Fail-closed MRI QC behavior

Rows with empty masks, non-positive tumor volumes, or `segmentation_qc=failed`
are not allowed to contribute numeric MRI features by default. The merge keeps
provenance, QC labels, and warnings so reports can explain why MRI was ignored,
