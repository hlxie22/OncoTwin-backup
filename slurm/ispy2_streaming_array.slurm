#!/usr/bin/env bash
#SBATCH --job-name=ispy2-stream
#SBATCH --output=logs/slurm/ispy2-stream-%A_%a.out
#SBATCH --error=logs/slurm/ispy2-stream-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs/slurm

if [ -n "${SETUP_SCRIPT:-}" ]; then
  # shellcheck source=/dev/null
  source "$SETUP_SCRIPT"
fi

MANIFEST="${MANIFEST:-data/ispy2_streaming/manifests/ispy2_derived_manifest.tsv}"
SHARD_DIR="${SHARD_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/ispy2_streaming}"
MIN_FREE_SPACE_GB="${MIN_FREE_SPACE_GB:-50}"
MAX_ACTIVE_SCRATCH_GB="${MAX_ACTIVE_SCRATCH_GB:-75}"
KEEP_FAILED_DICOM="${KEEP_FAILED_DICOM:-0}"
FORCE="${FORCE:-0}"
PATIENT_COUNT="${PATIENT_COUNT:-}"
WORKER_COUNT="${WORKER_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script must be submitted as a Slurm array job." >&2
  exit 2
fi

if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_case_index() {
  local case_index="$1"
  local cmd=(
    python3 scripts/data/ispy2_streaming_pipeline.py process
    --manifest "$MANIFEST"
  )

  if [ -n "$SHARD_DIR" ]; then
    local shard_manifest
    shard_manifest="$(printf '%s/shard_%04d.tsv' "$SHARD_DIR" "$case_index")"
    if [ ! -f "$shard_manifest" ]; then
      echo "Missing shard manifest: $shard_manifest" >&2
      return 2
    fi
    cmd+=(--shard-manifest "$shard_manifest")
  else
    cmd+=(--case-index "$case_index")
  fi

  cmd+=(
    --output-root "$OUTPUT_ROOT"
    --worker-id "$SLURM_ARRAY_TASK_ID"
    --min-free-space-gb "$MIN_FREE_SPACE_GB"
    --max-active-scratch-gb "$MAX_ACTIVE_SCRATCH_GB"
  )

  if [ "$KEEP_FAILED_DICOM" = "1" ]; then
    cmd+=(--keep-failed-dicom)
  fi
  if [ "$FORCE" = "1" ]; then
    cmd+=(--force)
  fi

  printf 'Running:'
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

failures=0

if [ -n "$SHARD_DIR" ]; then
  if ! run_case_index "$SLURM_ARRAY_TASK_ID"; then
    failures=$((failures + 1))
  fi
elif [ -n "$PATIENT_COUNT" ]; then
  if ! [[ "$PATIENT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "PATIENT_COUNT must be a positive integer, got: $PATIENT_COUNT" >&2
    exit 2
  fi
  if ! [[ "$WORKER_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORKER_COUNT must be a positive integer, got: $WORKER_COUNT" >&2
    exit 2
  fi

  echo "Worker ${SLURM_ARRAY_TASK_ID}/${WORKER_COUNT} processing case indices ${SLURM_ARRAY_TASK_ID}, $((SLURM_ARRAY_TASK_ID + WORKER_COUNT)), ... up to ${PATIENT_COUNT}"

  case_index="$SLURM_ARRAY_TASK_ID"
  while [ "$case_index" -le "$PATIENT_COUNT" ]; do
    if ! run_case_index "$case_index"; then
      echo "Case index failed: $case_index" >&2
      failures=$((failures + 1))
    fi
    case_index=$((case_index + WORKER_COUNT))
  done
else
  if ! run_case_index "$SLURM_ARRAY_TASK_ID"; then
    failures=$((failures + 1))
  fi
fi

if [ "$failures" -gt 0 ]; then
  echo "Worker ${SLURM_ARRAY_TASK_ID} completed with ${failures} failed case(s)." >&2
  exit 1
fi
