#!/usr/bin/env bash
#SBATCH --job-name=v1-merge-eval
#SBATCH --output=logs/slurm/v1-merge-eval-%j.out
#SBATCH --error=logs/slurm/v1-merge-eval-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

mkdir -p logs/slurm

# Optional environment setup. Override with SETUP_SCRIPT=/path/to/setup.sh.
if [ -n "${SETUP_SCRIPT:-}" ]; then
  # shellcheck source=/dev/null
  source "$SETUP_SCRIPT"
fi

DATA_ROOT="${DATA_ROOT:-data/v1_prior_stack}"
N_SAMPLES="${N_SAMPLES:-2000}"
SEED="${SEED:-2026}"
MIN_CASES="${MIN_CASES:-1}"

MERGED_DIR="$DATA_ROOT/processed/merged"
MERGED_COHORT="$MERGED_DIR/v1_eval_cohort.merged.jsonl"
MERGED_REPORT="evals/reports/v1_eval_suite.merged.md"

mkdir -p "$MERGED_DIR" evals/reports

echo "Merging shard cohorts..."
mapfile -d '' COHORT_FILES < <(find "$DATA_ROOT/processed/slurm" -name 'v1_eval_cohort.jsonl' -type f -print0 | sort -z)

if [ "${#COHORT_FILES[@]}" -eq 0 ]; then
  echo "No shard cohorts found under $DATA_ROOT/processed/slurm" >&2
  exit 2
fi

: > "$MERGED_COHORT"
for cohort in "${COHORT_FILES[@]}"; do
  cat "$cohort" >> "$MERGED_COHORT"
done

echo "Checking merged cohort for duplicate case IDs..."
python3 - "$MERGED_COHORT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
ids = [str(row.get("case_id", "")).strip() for row in rows]
dupes = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
missing = sum(1 for case_id in ids if not case_id)

if missing or dupes:
    if missing:
        print(f"Missing case_id count: {missing}", file=sys.stderr)
    if dupes:
        print("Duplicate case_id values: " + ", ".join(dupes[:50]), file=sys.stderr)
    raise SystemExit(2)

print(f"Merged rows: {len(rows)}")
PY

python3 scripts/data/validate_v1_eval_cohort.py \
  --cohort "$MERGED_COHORT" \
  --require-in-scope \
  --min-cases "$MIN_CASES"

python3 -m evals.prior_stack.run_v1_eval_suite \
  --cohort "$MERGED_COHORT" \
  --report "$MERGED_REPORT" \
  --n-samples "$N_SAMPLES" \
  --seed "$SEED"

echo "Merged cohort: $MERGED_COHORT"
echo "Merged report: $MERGED_REPORT"
