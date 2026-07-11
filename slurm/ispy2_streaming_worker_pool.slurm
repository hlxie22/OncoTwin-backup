#!/usr/bin/env bash
#SBATCH --job-name=ispy2_stream_worker
#SBATCH --output=logs/ispy2_stream_worker_%A_%a.out
#SBATCH --error=logs/ispy2_stream_worker_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs

if [ -n "${SETUP_SCRIPT:-}" ]; then
  source "$SETUP_SCRIPT"
fi

: "${MANIFEST:?MANIFEST is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${TOTAL_PATIENTS:?TOTAL_PATIENTS is required}"
: "${NUM_WORKERS:?NUM_WORKERS is required}"

WORKER_ID="${SLURM_ARRAY_TASK_ID}"

i="$WORKER_ID"
while [ "$i" -le "$TOTAL_PATIENTS" ]; do
  echo "Worker ${WORKER_ID}/${NUM_WORKERS}: processing case-index ${i}"

  python3 scripts/data/ispy2_streaming_pipeline.py process \
    --manifest "$MANIFEST" \
    --case-index "$i" \
    --output-root "$OUTPUT_ROOT" \
    --worker-id "${WORKER_ID}" \
    --min-free-space-gb "${MIN_FREE_SPACE_GB:-50}" \
    --max-active-scratch-gb "${MAX_ACTIVE_SCRATCH_GB:-75}"

  i=$((i + NUM_WORKERS))
done
