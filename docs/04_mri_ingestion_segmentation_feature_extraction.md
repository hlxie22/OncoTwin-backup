# MRI Ingestion, Segmentation, and Feature Extraction

## Purpose

The imaging pipeline converts raw MRI data into the spatial and quantitative inputs needed by the digital twin. It should produce:

```text
tumor mask
tumor volume
longest diameter
enhancement features
spatial heterogeneity features
quality-control flags
initial tumor-cell density map
```

This is the first major implementation feature because the mechanistic simulator needs a tumor geometry.

## Inputs

The pipeline should support three input levels.

### Level 1: report-only mode

```text
longest tumor diameter
optional second/third dimensions
radiology-reported response measurements
```

This mode creates an approximate ellipsoid or spherical tumor model. It is useful for demos but not the full digital twin.

### Level 2: single MRI mode

```text
baseline DCE-MRI
optional tumor mask
pathology metadata
```

This mode creates a spatial baseline twin but has limited calibration until follow-up data arrive.

### Level 3: longitudinal MRI mode

```text
baseline MRI
early-treatment MRI
mid-treatment MRI
presurgery MRI
masks or segmentations for each timepoint
```

This is the strongest mode and supports true Bayesian updating.

## DICOM/NIfTI ingestion

Implementation steps:

```text
1. Accept DICOM series or NIfTI volume.
2. Identify DCE-MRI phases and acquisition order.
3. Validate modality, orientation, spacing, dimensions, and metadata.
4. Convert DICOM to NIfTI if needed.
5. Store raw and processed volumes separately.
6. Create an imaging-timepoint record.
```

Suggested tools:

```text
dcm2niix for conversion
SimpleITK for image IO and resampling
pydicom for metadata checks
MONAI transforms for model preprocessing
```

## Preprocessing

Preprocessing should standardize the image enough for segmentation and feature extraction.

Steps:

```text
1. Bias-field correction if needed.
2. Resample to a standard voxel spacing.
3. Reorient to a consistent coordinate system.
4. Register DCE phases within a scan.
5. Register follow-up scans to baseline if longitudinal data exist.
6. Normalize intensities.
7. Crop to breast or tumor region.
8. Generate quality-control images.
```

Recommended output structure:

```text
processed/
  case_id/
    T0/
      dce_phase_0000.nii.gz
      dce_phase_0001.nii.gz
      dce_phase_0002.nii.gz
      subtraction.nii.gz
      qc_thumbnail.png
    T1/
      ...
```

## Tumor segmentation

### Initial backbone

Use the MAMA-MIA pretrained nnU-Net as the initial tumor segmentation backbone because it is breast DCE-MRI specific and trained with expert segmentations.

### Model input

Depending on the available scan format, the segmentation input can be:

```text
first post-contrast image
subtraction image
multi-phase DCE stack
cropped breast volume
```

A practical first version should support:

```text
input: preprocessed 3D DCE image or subtraction volume
output: binary tumor mask
```

### Segmentation output schema

```typescript
type TumorSegmentationOutput = {
  caseId: string;
  timepointId: string;
  modelName: string;
  modelVersion: string;
  tumorMaskUri: string;
  tumorVolumeMl: number;
  longestDiameterCm: number;
  confidenceScore: number;
  qcFlags: string[];
  createdAt: string;
};
```

## Segmentation confidence

The app must not treat every mask as equally reliable. Add confidence using:

```text
softmax/probability entropy
model ensemble variance
test-time augmentation variance
shape plausibility checks
volume outlier checks
image-quality checks
```

Example QC flags:

```text
low_contrast
motion_artifact_possible
tumor_mask_too_small
mask_fragmented
tumor_near_boundary
out_of_distribution_spacing
manual_review_recommended
```

## Tumor-volume extraction

Compute tumor volume from the binary mask:

```text
volume_ml = voxel_count × voxel_spacing_x × voxel_spacing_y × voxel_spacing_z / 1000
```

Also compute:

```text
longest diameter
bounding-box dimensions
surface area
sphericity
compactness
tumor centroid
residual-risk region coordinates
```

## Functional tumor volume and enhancement features

For DCE-MRI, compute enhancement features inside the tumor mask:

```text
pre-contrast intensity
post-contrast intensity
subtraction intensity
early enhancement slope
wash-in proxy
wash-out proxy if phases allow
percent enhancing volume
heterogeneity of enhancement
low-enhancement or necrotic fraction
```

These features become mechanistic proxies:

| Imaging feature | Mechanistic interpretation |
|---|---|
| High enhancement | stronger perfusion / drug delivery proxy |
| Low enhancement region | necrosis or poor delivery proxy |
| Heterogeneous enhancement | heterogeneous sensitivity or delivery |
| Irregular boundary | higher invasion/diffusion prior |
| Early shrinkage | higher drug sensitivity posterior |

## DCE kinetic maps and the delivery term

The enhancement features above also have a *voxel-wise* form that the simulator consumes directly. From the multi-phase DCE series, compute per-voxel kinetic maps:

```text
wash-in slope     early enhancement rate
peak enhancement  maximum signal increase
washout           late-phase decline, if phases allow
normalized AUC    area under the DCE time course, normalized to the tumor maximum
```

The **normalized-AUC map provides the simulator's `delivery(x)` term** (see `05_mechanistic_tumor_simulator.md`): voxels that enhance more are treated as receiving more drug. This follows the mechanistic-model literature and is a **deterministic computation — no training required**. The same maps can be passed as extra input channels to the parameter amortizer's image encoder (family D in `06_ai_personalization_parameter_amortizer.md`).

Caveat — temporal resolution. Public breast DCE is sampled at ~60–120 s with ~3–12 phases. That is enough for semi-quantitative curve shape (wash-in / plateau / washout) and AUC, but **not** for full pharmacokinetic (Tofts-style) rate constants. Treat `delivery(x)` as a semi-quantitative perfusion/delivery proxy, and normalize across cohorts because phase counts and timing vary.

## Initial tumor-cell density map

The mechanistic simulator needs `N(x,0)`, an initial tumor-cell density field. A simple first version:

```text
N(x,0) = θ inside tumor mask
N(x,0) = 0 outside tumor mask
```

A better version uses imaging features:

```text
N(x,0) = θ × normalized_enhancement(x)
```

If ADC/DWI is available, use an inverse ADC cellularity proxy:

```text
higher cellularity proxy → higher initial N(x,0)
```

## Longitudinal registration

For digital-twin updates, the system must compare tumor state across timepoints.

Steps:

```text
1. Register T1/T2/T3 scans to T0.
2. Transform masks into baseline coordinates.
3. Compute voxelwise residual or shrinkage maps.
4. Track tumor centroid shift and residual regions.
5. Store both native-space and registered-space masks.
```

Registration should be quality-controlled. If longitudinal registration is poor, the system should fall back to volume-only updating rather than spatial residual maps.

## Feature vector schema

```typescript
type ImagingFeatureVector = {
  volumeMl: number;
  longestDiameterCm: number;
  surfaceArea: number;
  sphericity: number;
  compactness: number;
  enhancementMean: number;
  enhancementStd: number;
  enhancementEntropy: number;
  percentEnhancingVolume: number;
  lowEnhancementFraction: number;
  boundaryIrregularity: number;
  centroid: [number, number, number];
  voxelSpacing: [number, number, number];
};
```

## Service pseudocode

```python
def process_mri_timepoint(case_id, timepoint_id, raw_uri):
    nifti = convert_to_nifti(raw_uri)
    preprocessed = preprocess_mri(nifti)
    segmentation = segment_tumor(preprocessed)
    qc = assess_segmentation_quality(preprocessed, segmentation.mask)
    features = extract_imaging_features(preprocessed, segmentation.mask)
    density_map = initialize_cell_density(segmentation.mask, features)

    save_outputs(
        case_id=case_id,
        timepoint_id=timepoint_id,
        preprocessed=preprocessed,
        mask=segmentation.mask,
        features=features,
        density_map=density_map,
        qc=qc,
    )

    return {
        "tumor_mask_uri": segmentation.mask_uri,
        "tumor_volume_ml": features.volume_ml,
        "longest_diameter_cm": features.longest_diameter_cm,
        "confidence": qc.confidence_score,
        "qc_flags": qc.flags,
    }
```

## UI output

The imaging feature should show:

```text
Tumor volume: 18.4 mL
Longest diameter: 3.2 cm
Segmentation confidence: moderate
Image quality notes: possible motion artifact
Spatial twin mode: enabled
```

If confidence is low:

```text
The tumor segmentation is uncertain. The simulation will use wider uncertainty bands and should be interpreted cautiously.
```

## Implementation priorities

1. Support NIfTI inputs first.
2. Integrate MAMA-MIA nnU-Net inference.
3. Compute tumor volume and basic features.
4. Add confidence/QC flags.
5. Add longitudinal registration.
6. Add DCE enhancement features and per-voxel kinetic maps (normalized-AUC delivery term).
7. Add ADC/DWI cellularity proxy if available.
