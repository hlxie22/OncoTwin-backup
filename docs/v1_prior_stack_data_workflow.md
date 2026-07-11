# V1 prior-stack data workflow

This workflow prepares the longitudinal cohort consumed by:

```text
evals/prior_stack/v1_real_data_eval.py
evals/prior_stack/v1_uncertainty_calibration_eval.py
evals/prior_stack/run_v1_eval_suite.py
```

The repository does not commit downloaded imaging, clinical tables, manifests, PHI, credentials, or generated cohorts. Local data are written under `data/v1_prior_stack/`, which is ignored by git.

## Supported starting points

### BreastDCEDL_ISPY2 / I-SPY2 family

The primary V1 source family is I-SPY2-derived breast DCE-MRI data. Use BreastDCEDL_ISPY2 for standardized pretreatment DCE-MRI, tumor annotations, and harmonized clinicopathologic metadata when those artifacts are useful for context fields. Use I-SPY2/TCIA-derived longitudinal measurement or functional tumor volume exports for the actual eval cohort, because the current V1 real-data eval requires baseline and held-out final tumor volumes, with an optional early follow-up volume.

This repository-side workflow does not assume that every I-SPY2 artifact is directly downloadable without user action. When files are available through a public URL, pass that URL explicitly to `download_v1_eval_data.py`. When access requires a TCIA/NBIA manifest, Data Retriever, institutional approval, or a portal login, use the approved portal/tooling first and then stage the local files with `--local-path`.

Expected useful local artifacts are one or more CSV/JSON/JSONL tables containing:

```text
case_id or patient_id
baseline/follow-up/final tumor volume measurements
relative day or timepoint label for each measurement
subtype or receptor status where available
treatment context or regimen/category where available
pathology, biomarkers, MRI feature, and QC fields where available
```

Do not convert pCR labels into tumor volumes. If the source does not include longitudinal tumor volumes or functional tumor volumes, it can support context enrichment, but it cannot by itself run the real-data V1 prior-layer eval.

### Generic longitudinal tumor-volume table

Use `generic-longitudinal-table` for a manually prepared table from any approved source. The builder accepts either:

```text
one row per case with baseline_volume_ml, final_volume_ml, baseline_day, final_day
```

or:

```text
one row per case/timepoint with case_id, day, tumor_volume_ml
```

The loader accepts `.json`, `.jsonl`, or `.csv` cohorts after normalization.

## Stage raw files

Copy a manually downloaded file or directory into the local staging area:

```bash
python3 scripts/data/download_v1_eval_data.py   --source breastdcedl-ispy2   --local-path /path/to/downloaded/ispy2_measurements.csv
```

Preserve a TCIA/NBIA manifest in the staging area:

```bash
python3 scripts/data/download_v1_eval_data.py   --source breastdcedl-ispy2   --manifest /path/to/manifest.tcia
```

Download only explicit public URLs that you have verified are appropriate for the needed table:

```bash
python3 scripts/data/download_v1_eval_data.py   --source breastdcedl-ispy2   --url https://example.org/path/to/public_measurements.csv
```

If the staged directory contains multiple tables, pass the exact measurement table to the builder with `--measurements`.

## Build the normalized cohort

Build from a staged measurement table:

```bash
python3 scripts/data/build_v1_eval_cohort.py   --source breastdcedl-ispy2   --measurements data/v1_prior_stack/raw/breastdcedl-ispy2/ispy2_measurements.csv   --clinical data/v1_prior_stack/raw/breastdcedl-ispy2/ispy2_clinical.csv   --default-treatment-context "neoadjuvant chemotherapy"
```

`--default-treatment-context` is optional and should only be used when source documentation supports the treatment category but the local table omits it. It should not be used to invent a specific drug regimen.

If a long I-SPY2-style table has T0/T1/T2/T3 labels but no numeric day column, the builder can use a clearly labeled nominal-day approximation:

```bash
python3 scripts/data/build_v1_eval_cohort.py   --source breastdcedl-ispy2   --measurements data/v1_prior_stack/raw/breastdcedl-ispy2/ispy2_measurements.csv   --use-nominal-ispy2-days   --default-treatment-context "neoadjuvant chemotherapy"
```

Prefer true relative days whenever the source provides them.

The default output is:

```text
data/v1_prior_stack/processed/v1_eval_cohort.jsonl
```

## Validate the cohort

Validate the generated cohort against the existing eval loader and require at least one V1-A in-scope TNBC + chemotherapy case:

```bash
python3 scripts/data/validate_v1_eval_cohort.py   --cohort data/v1_prior_stack/processed/v1_eval_cohort.jsonl   --require-in-scope
```

The validator rejects demo, synthetic, simulated, toy, and fixture data by default because `v1_real_data_eval.load_real_cohort` rejects them by default.

## Demo smoke check

A tiny synthetic cohort is available only to smoke-check wiring:

```bash
python3 scripts/data/build_v1_eval_cohort.py --write-demo-synthetic
python3 scripts/data/validate_v1_eval_cohort.py   --cohort data/v1_prior_stack/demo/v1_eval_cohort.synthetic.jsonl   --allow-demo-data   --require-in-scope
```

Never use this path for real validation or performance claims.

## Run V1 evals

Run the real-data prior-layer eval only:

```bash
python3 -m evals.prior_stack.v1_real_data_eval   --cohort data/v1_prior_stack/processed/v1_eval_cohort.jsonl   --report evals/reports/v1_real_data_prior_layer_eval.md
```

Run the uncertainty calibration eval only:

```bash
python3 -m evals.prior_stack.v1_uncertainty_calibration_eval   --cohort data/v1_prior_stack/processed/v1_eval_cohort.jsonl   --report evals/reports/v1_uncertainty_calibration.md
```

Run the full V1 suite:

```bash
python3 -m evals.prior_stack.run_v1_eval_suite   --cohort data/v1_prior_stack/processed/v1_eval_cohort.jsonl   --report evals/reports/v1_eval_suite.md
```

The suite also writes a machine-readable summary next to the Markdown report by default, for example `evals/reports/v1_eval_suite.summary.json`. When present, cohort-builder sidecars named like `v1_eval_cohort.summary.json` and `v1_eval_cohort.exclusions.jsonl` are automatically included in the suite report; use `--cohort-summary`, `--exclusions`, or `--summary` to override those paths.

For a quick synthetic smoke check only:

```bash
python3 -m evals.prior_stack.run_v1_eval_suite   --cohort data/v1_prior_stack/demo/v1_eval_cohort.synthetic.jsonl   --allow-demo-data   --n-samples 200   --report evals/reports/v1_eval_suite.synthetic_smoke.md
```

## Cohort fields

The normalized cohort keeps missing context fields as JSON null when possible and fails only when the current eval loader cannot run. Required fields are:

```text
case_id
baseline_day
baseline_volume_ml
final_day
final_volume_ml
```

Optional paired fields are:

```text
early_day
early_volume_ml
```

Useful context fields include:

```text
subtype, disease_context, cancer_subtype
treatment_context, treatment_regimen, regimen_name, schedule_type
er_status, pr_status, her2_status, hr_status
grade, ki67_percent, brca_status, hrd_status
volume_ml, functional_tumor_volume_ml, longest_diameter_cm
enhancement_std, segmentation_qc, registration_qc
```
