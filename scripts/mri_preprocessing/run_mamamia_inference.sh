#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-data/curated/nnunet_inputs}"
OUTPUT_DIR="${2:-data/curated/segmentations/mamamia_nnunet}"
if [ "$#" -ge 2 ]; then
  shift 2
elif [ "$#" -eq 1 ]; then
  shift 1
fi

DATASET_ID="${MAMAMIA_NNUNET_DATASET_ID:-101}"
CONFIGURATION="${MAMAMIA_NNUNET_CONFIGURATION:-3d_fullres}"

if ! command -v nnUNetv2_predict >/dev/null 2>&1; then
  echo "nnUNetv2_predict is not available on PATH" >&2
  exit 2
fi

if [ -z "${nnUNet_results:-}" ]; then
  echo "nnUNet_results is not set" >&2
  exit 2
fi

if [ ! -d "$INPUT_DIR" ]; then
  echo "input directory not found: $INPUT_DIR" >&2
  exit 2
fi

if find "$INPUT_DIR" -maxdepth 1 -type f ! -name '*.nii.gz' | grep -q .; then
  echo "all MAMA-MIA nnU-Net input files must be compressed NIfTI (*.nii.gz)" >&2
  exit 2
fi

if ! find "$INPUT_DIR" -maxdepth 1 -type f -name '*.nii.gz' | grep -q .; then
  echo "no compressed NIfTI inputs found in $INPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
if [ ! -w "$OUTPUT_DIR" ]; then
  echo "output directory is not writable: $OUTPUT_DIR" >&2
  exit 2
fi

exec nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -d "$DATASET_ID" \
  -c "$CONFIGURATION" \
  "$@"
