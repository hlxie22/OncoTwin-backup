#!/usr/bin/env bash
#SBATCH --job-name=v1-data-build
#SBATCH --output=logs/slurm/v1-data-build-%A_%a.out
#SBATCH --error=logs/slurm/v1-data-build-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs/slurm

# Optional environment setup. Override with SETUP_SCRIPT=/path/to/setup.sh.
# Example setup script contents:
#   module purge
#   module load python
#   source ~/.bashrc
#   conda activate oncotwin-repomix
if [ -n "${SETUP_SCRIPT:-}" ]; then
  # shellcheck source=/dev/null
  source "$SETUP_SCRIPT"
fi

TASK_FILE="${TASK_FILE:-configs/v1_data_tasks.tsv}"
DATA_ROOT="${DATA_ROOT:-data/v1_prior_stack}"
N_SAMPLES="${N_SAMPLES:-2000}"
SEED_BASE="${SEED_BASE:-2026}"
MIN_CASES="${MIN_CASES:-1}"
OVERWRITE="${OVERWRITE:-0}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "This script must be submitted as a Slurm array job." >&2
  echo "Use: bash slurm/launch_v1_data_pipeline.sh --task-file $TASK_FILE --parallelism 4" >&2
  exit 2
fi

if [ ! -f "$TASK_FILE" ]; then
  echo "Missing task file: $TASK_FILE" >&2
  exit 2
fi

# Skip the header row. Array index 1 maps to TSV data row 2.
TASK_LINE="$(awk -v line="$((SLURM_ARRAY_TASK_ID + 1))" 'NR == line {print}' "$TASK_FILE")"
if [ -z "$TASK_LINE" ]; then
  echo "No task line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

IFS=$'\t' read -r SOURCE MEASUREMENTS CLINICAL LOCAL_PATHS URLS EXPECTED_FILES DEFAULT_TREATMENT USE_NOMINAL <<< "$TASK_LINE"

if [ -z "${SOURCE:-}" ] || [ "$SOURCE" = "-" ]; then
  echo "Missing source in task line" >&2
  exit 2
fi

if [ -z "${MEASUREMENTS:-}" ] || [ "$MEASUREMENTS" = "-" ]; then
  echo "Missing measurements path in task line" >&2
  exit 2
fi

TASK_NAME="${SOURCE}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="$DATA_ROOT/processed/slurm/$TASK_NAME"
COHORT="$OUT_DIR/v1_eval_cohort.jsonl"
REPORT_DIR="evals/reports/slurm/$TASK_NAME"
SUITE_REPORT="$REPORT_DIR/v1_eval_suite.md"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

echo "Task: $TASK_NAME"
echo "Source: $SOURCE"
echo "Measurements: $MEASUREMENTS"
echo "Clinical: ${CLINICAL:-}"
echo "Output cohort: $COHORT"
echo "Report: $SUITE_REPORT"

DOWNLOAD_CMD=(python3 scripts/data/download_v1_eval_data.py --source "$SOURCE" --output-root "$DATA_ROOT")

if [ "$OVERWRITE" = "1" ]; then
  DOWNLOAD_CMD+=(--overwrite)
fi

if [ -n "${LOCAL_PATHS:-}" ] && [ "$LOCAL_PATHS" != "-" ]; then
  IFS=',' read -ra LOCAL_ARRAY <<< "$LOCAL_PATHS"
  for path in "${LOCAL_ARRAY[@]}"; do
    DOWNLOAD_CMD+=(--local-path "$path")
  done
fi

if [ -n "${URLS:-}" ] && [ "$URLS" != "-" ]; then
  IFS=',' read -ra URL_ARRAY <<< "$URLS"
  for url in "${URL_ARRAY[@]}"; do
    DOWNLOAD_CMD+=(--url "$url")
  done
fi

if [ -n "${EXPECTED_FILES:-}" ] && [ "$EXPECTED_FILES" != "-" ]; then
  IFS=',' read -ra EXPECTED_ARRAY <<< "$EXPECTED_FILES"
  for expected in "${EXPECTED_ARRAY[@]}"; do
    DOWNLOAD_CMD+=(--expected-file "$expected")
  done
fi

echo "Running staging/download step..."
printf '  %q' "${DOWNLOAD_CMD[@]}"
echo
"${DOWNLOAD_CMD[@]}"

BUILD_CMD=(
  python3 scripts/data/build_v1_eval_cohort.py
  --source "$SOURCE"
  --measurements "$MEASUREMENTS"
  --output "$COHORT"
)

if [ -n "${CLINICAL:-}" ] && [ "$CLINICAL" != "-" ]; then
  BUILD_CMD+=(--clinical "$CLINICAL")
fi

if [ -n "${DEFAULT_TREATMENT:-}" ] && [ "$DEFAULT_TREATMENT" != "-" ]; then
  BUILD_CMD+=(--default-treatment-context "$DEFAULT_TREATMENT")
fi

if [ "${USE_NOMINAL:-0}" = "1" ]; then
  BUILD_CMD+=(--use-nominal-ispy2-days)
fi

echo "Running cohort build step..."
printf '  %q' "${BUILD_CMD[@]}"
echo
"${BUILD_CMD[@]}"

echo "Running repository validator..."
python3 scripts/data/validate_v1_eval_cohort.py \
  --cohort "$COHORT" \
  --require-in-scope \
  --min-cases "$MIN_CASES"

echo "Running extra strict guard for duplicates and early timepoints..."
python3 - "$COHORT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

cohort = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in cohort.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

ids = [str(row.get("case_id", "")).strip() for row in rows]
missing = [i for i, case_id in enumerate(ids, start=1) if not case_id]
dupes = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)

errors = []
if missing:
    errors.append(f"missing case_id rows: {missing[:10]}")
if dupes:
    errors.append("duplicate case_id values: " + ", ".join(dupes[:20]))

for row in rows:
    case_id = row.get("case_id", "<unknown>")
    baseline_day = float(row["baseline_day"])
    final_day = float(row["final_day"])
    baseline_volume = float(row["baseline_volume_ml"])
    final_volume = float(row["final_volume_ml"])

    if baseline_volume <= 0:
        errors.append(f"{case_id}: baseline_volume_ml must be positive")
    if final_volume <= 0:
        errors.append(f"{case_id}: final_volume_ml must be positive")
    if final_day <= baseline_day:
        errors.append(f"{case_id}: final_day must be after baseline_day")

    has_early_day = row.get("early_day") not in (None, "")
    has_early_volume = row.get("early_volume_ml") not in (None, "")
    if has_early_day != has_early_volume:
        errors.append(f"{case_id}: early_day and early_volume_ml must be paired")
    if has_early_day and has_early_volume:
        early_day = float(row["early_day"])
        early_volume = float(row["early_volume_ml"])
        if early_volume <= 0:
            errors.append(f"{case_id}: early_volume_ml must be positive")
        if not (baseline_day < early_day < final_day):
            errors.append(f"{case_id}: early_day must be between baseline_day and final_day")

if errors:
    print("Strict cohort validation failed:", file=sys.stderr)
    for error in errors[:50]:
        print(f"- {error}", file=sys.stderr)
    raise SystemExit(2)

print(f"Strict cohort validation passed: {len(rows)} rows")
PY

echo "Running V1 eval suite for this shard..."
python3 -m evals.prior_stack.run_v1_eval_suite \
  --cohort "$COHORT" \
  --report "$SUITE_REPORT" \
  --n-samples "$N_SAMPLES" \
  --seed "$((SEED_BASE + SLURM_ARRAY_TASK_ID))"

echo "Done: $TASK_NAME"
echo "Cohort: $COHORT"
echo "Report: $SUITE_REPORT"
