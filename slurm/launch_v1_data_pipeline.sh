#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash slurm/launch_v1_data_pipeline.sh --task-file configs/v1_data_tasks.tsv --parallelism 4 [options]

Required:
  --task-file PATH        TSV with a header plus one data source per row.
  --parallelism N         Max number of Slurm array tasks running at once.

Optional:
  --data-root PATH        Default: data/v1_prior_stack
  --n-samples N           Default: 2000
  --seed-base N           Default: 2026
  --min-cases N           Default: 1
  --setup-script PATH     Optional shell script sourced inside each Slurm job.
  --array-script PATH     Default: slurm/v1_data_build_array.slurm
  --merge-script PATH     Default: slurm/v1_merge_eval.slurm
  --no-merge              Submit only the array job, no dependent merge/eval job.
  --overwrite             Pass overwrite behavior to staging step.
  --partition NAME        Add --partition to sbatch.
  --account NAME          Add --account to sbatch.
  --qos NAME              Add --qos to sbatch.
  --time HH:MM:SS         Override Slurm time for both jobs.
  --mem SIZE              Override Slurm memory for both jobs, e.g. 16G.
  --cpus N                Override CPUs per task for the array job.
  --merge-cpus N          Override CPUs for merge job. Default: 2.
  --dry-run               Print sbatch commands without submitting.
  -h, --help              Show this help.

Task TSV columns, tab-separated:
  source measurements clinical local_paths urls expected_files default_treatment_context use_nominal_ispy2_days

Use '-' for blank values. Comma-separate multiple local_paths, urls, or expected_files.
USAGE
}

TASK_FILE=""
PARALLELISM=""
DATA_ROOT="data/v1_prior_stack"
N_SAMPLES="2000"
SEED_BASE="2026"
MIN_CASES="1"
SETUP_SCRIPT=""
ARRAY_SCRIPT="slurm/v1_data_build_array.slurm"
MERGE_SCRIPT="slurm/v1_merge_eval.slurm"
DO_MERGE="1"
OVERWRITE="0"
PARTITION=""
ACCOUNT=""
QOS=""
TIME_LIMIT=""
MEMORY=""
CPUS=""
MERGE_CPUS="2"
DRY_RUN="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --task-file) TASK_FILE="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --min-cases) MIN_CASES="$2"; shift 2 ;;
    --setup-script) SETUP_SCRIPT="$2"; shift 2 ;;
    --array-script) ARRAY_SCRIPT="$2"; shift 2 ;;
    --merge-script) MERGE_SCRIPT="$2"; shift 2 ;;
    --no-merge) DO_MERGE="0"; shift ;;
    --overwrite) OVERWRITE="1"; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --merge-cpus) MERGE_CPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$TASK_FILE" ]; then
  echo "Missing required --task-file" >&2
  usage >&2
  exit 2
fi

if [ -z "$PARALLELISM" ]; then
  echo "Missing required --parallelism" >&2
  usage >&2
  exit 2
fi

if ! [[ "$PARALLELISM" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallelism must be a positive integer" >&2
  exit 2
fi

if [ ! -f "$TASK_FILE" ]; then
  echo "Task file not found: $TASK_FILE" >&2
  exit 2
fi

if [ ! -f "$ARRAY_SCRIPT" ]; then
  echo "Array script not found: $ARRAY_SCRIPT" >&2
  exit 2
fi

if [ "$DO_MERGE" = "1" ] && [ ! -f "$MERGE_SCRIPT" ]; then
  echo "Merge script not found: $MERGE_SCRIPT" >&2
  exit 2
fi

TASK_COUNT="$(awk 'NR > 1 && $0 !~ /^[[:space:]]*$/ {count++} END {print count+0}' "$TASK_FILE")"
if [ "$TASK_COUNT" -lt 1 ]; then
  echo "Task file has no data rows: $TASK_FILE" >&2
  exit 2
fi

mkdir -p logs/slurm

SBATCH_COMMON=()
if [ -n "$PARTITION" ]; then SBATCH_COMMON+=(--partition "$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then SBATCH_COMMON+=(--account "$ACCOUNT"); fi
if [ -n "$QOS" ]; then SBATCH_COMMON+=(--qos "$QOS"); fi
if [ -n "$TIME_LIMIT" ]; then SBATCH_COMMON+=(--time "$TIME_LIMIT"); fi
if [ -n "$MEMORY" ]; then SBATCH_COMMON+=(--mem "$MEMORY"); fi

ARRAY_SBATCH=(
  sbatch --parsable
  "${SBATCH_COMMON[@]}"
  --array "1-${TASK_COUNT}%${PARALLELISM}"
  --export "ALL,TASK_FILE=${TASK_FILE},DATA_ROOT=${DATA_ROOT},N_SAMPLES=${N_SAMPLES},SEED_BASE=${SEED_BASE},MIN_CASES=${MIN_CASES},OVERWRITE=${OVERWRITE},SETUP_SCRIPT=${SETUP_SCRIPT}"
)
if [ -n "$CPUS" ]; then ARRAY_SBATCH+=(--cpus-per-task "$CPUS"); fi
ARRAY_SBATCH+=("$ARRAY_SCRIPT")

printf 'Array sbatch command:'
printf ' %q' "${ARRAY_SBATCH[@]}"
echo

echo "Task rows: $TASK_COUNT"
echo "Max concurrent array tasks: $PARALLELISM"

if [ "$DRY_RUN" = "1" ]; then
  if [ "$DO_MERGE" = "1" ]; then
    MERGE_DRY=(
      sbatch --parsable
      "${SBATCH_COMMON[@]}"
      --dependency 'afterok:<ARRAY_JOB_ID>'
      --cpus-per-task "$MERGE_CPUS"
      --export "ALL,DATA_ROOT=${DATA_ROOT},N_SAMPLES=${N_SAMPLES},SEED=${SEED_BASE},MIN_CASES=${MIN_CASES},SETUP_SCRIPT=${SETUP_SCRIPT}"
      "$MERGE_SCRIPT"
    )
    printf 'Merge sbatch command:'
    printf ' %q' "${MERGE_DRY[@]}"
    echo
  fi
  exit 0
fi

ARRAY_JOB_ID="$(${ARRAY_SBATCH[@]})"
echo "Submitted array job: $ARRAY_JOB_ID"

if [ "$DO_MERGE" = "1" ]; then
  MERGE_SBATCH=(
    sbatch --parsable
    "${SBATCH_COMMON[@]}"
    --dependency "afterok:${ARRAY_JOB_ID}"
    --cpus-per-task "$MERGE_CPUS"
    --export "ALL,DATA_ROOT=${DATA_ROOT},N_SAMPLES=${N_SAMPLES},SEED=${SEED_BASE},MIN_CASES=${MIN_CASES},SETUP_SCRIPT=${SETUP_SCRIPT}"
    "$MERGE_SCRIPT"
  )
  printf 'Merge sbatch command:'
  printf ' %q' "${MERGE_SBATCH[@]}"
  echo
  MERGE_JOB_ID="$(${MERGE_SBATCH[@]})"
  echo "Submitted merge/eval job: $MERGE_JOB_ID"
fi
