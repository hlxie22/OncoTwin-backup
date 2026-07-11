#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash slurm/launch_ispy2_streaming_pipeline.sh --manifest data/ispy2_streaming/manifests/ispy2_derived_manifest.tsv --parallelism 8 [options]

Required:
  --manifest PATH       Derived manifest produced by scripts/data/ispy2_streaming_pipeline.py inventory.
  --parallelism N       Number of Slurm worker tasks. Each worker processes a strided subset of patients.

Optional:
  --output-root PATH    Default: data/ispy2_streaming
  --setup-script PATH   Optional shell script sourced inside each Slurm job.
  --array-script PATH   Default: slurm/ispy2_streaming_array.slurm
  --merge-script PATH   Default: slurm/ispy2_streaming_merge.slurm
  --no-merge            Submit only the extraction array.
  --partition NAME      Add --partition to sbatch.
  --account NAME        Add --account to sbatch.
  --qos NAME            Add --qos to sbatch.
  --time HH:MM:SS       Override Slurm time for both jobs.
  --mem SIZE            Override Slurm memory for extraction tasks, e.g. 8G.
  --cpus N              Override CPUs per extraction task. Default script value is 2.
  --min-free-space-gb N Default: 50.
  --max-active-scratch-gb N Default: 75.
  --keep-failed-dicom   Keep failed scratch payloads for debugging. Off by default.
  --force               Reprocess completed patients.
  --clinical PATH       Optional clinical/context table for longitudinal build.
  --default-treatment-context TEXT
                         Default: neoadjuvant chemotherapy.
  --v1-output PATH      Default: data/v1_prior_stack/processed/v1_eval_cohort.streamed.jsonl
  --no-v1-build         Merge features and longitudinal CSV only; do not build V1 JSONL.
  --dry-run             Print sbatch commands without submitting.
  -h, --help            Show this help.
USAGE
}

MANIFEST=""
PARALLELISM=""
OUTPUT_ROOT="data/ispy2_streaming"
SETUP_SCRIPT=""
ARRAY_SCRIPT="slurm/ispy2_streaming_array.slurm"
MERGE_SCRIPT="slurm/ispy2_streaming_merge.slurm"
DO_MERGE="1"
PARTITION=""
ACCOUNT=""
QOS=""
TIME_LIMIT=""
MEMORY=""
CPUS=""
MIN_FREE_SPACE_GB="50"
MAX_ACTIVE_SCRATCH_GB="75"
KEEP_FAILED_DICOM="0"
FORCE="0"
CLINICAL=""
DEFAULT_TREATMENT_CONTEXT="neoadjuvant chemotherapy"
V1_OUTPUT="data/v1_prior_stack/processed/v1_eval_cohort.streamed.jsonl"
RUN_V1_BUILD="1"
DRY_RUN="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --setup-script) SETUP_SCRIPT="$2"; shift 2 ;;
    --array-script) ARRAY_SCRIPT="$2"; shift 2 ;;
    --merge-script) MERGE_SCRIPT="$2"; shift 2 ;;
    --no-merge) DO_MERGE="0"; shift ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --min-free-space-gb) MIN_FREE_SPACE_GB="$2"; shift 2 ;;
    --max-active-scratch-gb) MAX_ACTIVE_SCRATCH_GB="$2"; shift 2 ;;
    --keep-failed-dicom) KEEP_FAILED_DICOM="1"; shift ;;
    --force) FORCE="1"; shift ;;
    --clinical) CLINICAL="$2"; shift 2 ;;
    --default-treatment-context) DEFAULT_TREATMENT_CONTEXT="$2"; shift 2 ;;
    --v1-output) V1_OUTPUT="$2"; shift 2 ;;
    --no-v1-build) RUN_V1_BUILD="0"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$MANIFEST" ]; then
  echo "Missing required --manifest" >&2
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
if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
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

PATIENT_COUNT="$(python3 - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
delimiter = '\t' if path.suffix.lower() in {'.tsv', '.txt'} else ','
with path.open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle, delimiter=delimiter))
ids = sorted({(row.get('case_id') or row.get('patient_id') or '').strip() for row in rows})
ids = [case_id for case_id in ids if case_id]
print(len(ids))
PY
)"

if [ "$PATIENT_COUNT" -lt 1 ]; then
  echo "No case_id values found in manifest: $MANIFEST" >&2
  exit 2
fi

SHARD_DIR="$OUTPUT_ROOT/manifests/launch_shards"
rm -rf "$SHARD_DIR"
mkdir -p "$SHARD_DIR"
python3 - "$MANIFEST" "$SHARD_DIR" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

manifest = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
delimiter = "\t" if manifest.suffix.lower() in {".tsv", ".txt"} else ","
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter=delimiter))
if not rows:
    raise SystemExit("Manifest has no rows: %s" % manifest)
fieldnames = list(rows[0].keys())
by_case = defaultdict(list)
for row in rows:
    case_id = (row.get("case_id") or row.get("patient_id") or row.get("subject_id") or "").strip()
    if case_id:
        by_case[case_id].append(row)
if not by_case:
    raise SystemExit("No case_id/patient_id/subject_id values found in %s" % manifest)
for index, case_id in enumerate(sorted(by_case), start=1):
    path = shard_dir / ("shard_%04d.tsv" % index)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(by_case[case_id])
PY

mkdir -p logs/slurm

SBATCH_COMMON=()
if [ -n "$PARTITION" ]; then SBATCH_COMMON+=(--partition "$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then SBATCH_COMMON+=(--account "$ACCOUNT"); fi
if [ -n "$QOS" ]; then SBATCH_COMMON+=(--qos "$QOS"); fi
if [ -n "$TIME_LIMIT" ]; then SBATCH_COMMON+=(--time "$TIME_LIMIT"); fi
if [ -n "$MEMORY" ]; then SBATCH_COMMON+=(--mem "$MEMORY"); fi

WORKER_COUNT="$PARALLELISM"
if [ "$WORKER_COUNT" -gt "$PATIENT_COUNT" ]; then
  WORKER_COUNT="$PATIENT_COUNT"
fi

ARRAY_SBATCH=(
  sbatch --parsable
  "${SBATCH_COMMON[@]}"
  --array "1-${PATIENT_COUNT}%${WORKER_COUNT}"
  --export "ALL,MANIFEST=${MANIFEST},SHARD_DIR=${SHARD_DIR},OUTPUT_ROOT=${OUTPUT_ROOT},PATIENT_COUNT=${PATIENT_COUNT},WORKER_COUNT=${WORKER_COUNT},MIN_FREE_SPACE_GB=${MIN_FREE_SPACE_GB},MAX_ACTIVE_SCRATCH_GB=${MAX_ACTIVE_SCRATCH_GB},KEEP_FAILED_DICOM=${KEEP_FAILED_DICOM},FORCE=${FORCE},SETUP_SCRIPT=${SETUP_SCRIPT}"
)
if [ -n "$CPUS" ]; then ARRAY_SBATCH+=(--cpus-per-task "$CPUS"); fi
ARRAY_SBATCH+=("$ARRAY_SCRIPT")

printf 'Array sbatch command:'
printf ' %q' "${ARRAY_SBATCH[@]}"
echo

echo "Patients: $PATIENT_COUNT"
echo "Max concurrent array tasks: $WORKER_COUNT"
echo "Array task count: $PATIENT_COUNT"
echo "Shard manifests: $SHARD_DIR"

if [ "$DRY_RUN" = "1" ]; then
  if [ "$DO_MERGE" = "1" ]; then
    MERGE_DRY=(
      sbatch --parsable
      "${SBATCH_COMMON[@]}"
      --dependency 'afterok:<ARRAY_JOB_ID>'
      --export "ALL,OUTPUT_ROOT=${OUTPUT_ROOT},CLINICAL=${CLINICAL},DEFAULT_TREATMENT_CONTEXT=${DEFAULT_TREATMENT_CONTEXT},V1_OUTPUT=${V1_OUTPUT},RUN_V1_BUILD=${RUN_V1_BUILD},SETUP_SCRIPT=${SETUP_SCRIPT}"
      "$MERGE_SCRIPT"
    )
    printf 'Merge sbatch command:'
    printf ' %q' "${MERGE_DRY[@]}"
    echo
  fi
  exit 0
fi

ARRAY_JOB_ID="$("${ARRAY_SBATCH[@]}")"
echo "Submitted extraction array job: $ARRAY_JOB_ID"

if [ "$DO_MERGE" = "1" ]; then
  MERGE_SBATCH=(
    sbatch --parsable
    "${SBATCH_COMMON[@]}"
    --dependency "afterok:${ARRAY_JOB_ID}"
    --export "ALL,OUTPUT_ROOT=${OUTPUT_ROOT},CLINICAL=${CLINICAL},DEFAULT_TREATMENT_CONTEXT=${DEFAULT_TREATMENT_CONTEXT},V1_OUTPUT=${V1_OUTPUT},RUN_V1_BUILD=${RUN_V1_BUILD},SETUP_SCRIPT=${SETUP_SCRIPT}"
    "$MERGE_SCRIPT"
  )
  printf 'Merge sbatch command:'
  printf ' %q' "${MERGE_SBATCH[@]}"
  echo
  MERGE_JOB_ID="$("${MERGE_SBATCH[@]}")"
  echo "Submitted merge job: $MERGE_JOB_ID"
fi
