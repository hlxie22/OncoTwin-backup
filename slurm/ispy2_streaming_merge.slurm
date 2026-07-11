#!/usr/bin/env bash
#SBATCH --job-name=ispy2-stream-merge
#SBATCH --output=logs/slurm/ispy2-stream-merge-%j.out
#SBATCH --error=logs/slurm/ispy2-stream-merge-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

mkdir -p logs/slurm

if [ -n "${SETUP_SCRIPT:-}" ]; then
  # shellcheck source=/dev/null
  source "$SETUP_SCRIPT"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-data/ispy2_streaming}"
CLINICAL="${CLINICAL:-}"
DEFAULT_TREATMENT_CONTEXT="${DEFAULT_TREATMENT_CONTEXT:-neoadjuvant chemotherapy}"
V1_OUTPUT="${V1_OUTPUT:-data/v1_prior_stack/processed/v1_eval_cohort.streamed.jsonl}"
RUN_V1_BUILD="${RUN_V1_BUILD:-1}"

python3 scripts/data/ispy2_streaming_pipeline.py merge-features \
  --output-root "$OUTPUT_ROOT"

LONG_CMD=(
  python3 scripts/data/ispy2_streaming_pipeline.py build-longitudinal
  --output-root "$OUTPUT_ROOT"
  --default-treatment-context "$DEFAULT_TREATMENT_CONTEXT"
)
if [ -n "$CLINICAL" ]; then
  LONG_CMD+=(--clinical "$CLINICAL")
fi
"${LONG_CMD[@]}"

if [ "$RUN_V1_BUILD" = "1" ]; then
  python3 scripts/data/build_v1_eval_cohort.py \
    --source generic-longitudinal-table \
    --measurements "$OUTPUT_ROOT/features/ispy2_streamed_longitudinal_features.csv" \
    --output "$V1_OUTPUT"

  python3 scripts/data/validate_v1_eval_cohort.py \
    --cohort "$V1_OUTPUT" \
    --require-in-scope
fi
