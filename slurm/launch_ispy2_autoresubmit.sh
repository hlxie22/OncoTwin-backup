#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash slurm/launch_ispy2_autoresubmit.sh \
    --manifest data/ispy2_streaming/manifests/ispy2_derived_manifest.tsv \
    --parallelism 8 \
    --time 05:55:00

This submits a tiny Slurm controller job and exits immediately.
The controller job submits extraction arrays until all cases are complete.

Options:
  --manifest PATH
  --parallelism N
  --output-root PATH                  Default: data/ispy2_streaming
  --setup-script PATH
  --array-script PATH                 Default: slurm/ispy2_streaming_array.slurm
  --merge-script PATH                 Default: slurm/ispy2_streaming_merge.slurm
  --controller-script PATH            Default: slurm/ispy2_autoresubmit_controller.slurm
  --time HH:MM:SS                     Default: 05:55:00
  --partition NAME
  --account NAME
  --qos NAME
  --mem SIZE                          Default: 8G
  --cpus N                            Default: 2
  --min-free-space-gb N               Default: 50
  --max-active-scratch-gb N           Default: 75
  --keep-failed-dicom
  --clinical PATH
  --default-treatment-context TEXT    Default: neoadjuvant chemotherapy
  --v1-output PATH                    Default: data/v1_prior_stack/processed/v1_eval_cohort.streamed.jsonl
  --no-v1-build
  --no-merge
  --max-rounds N                      Default: 100
  --no-progress-limit N               Default: 3
USAGE
}

MANIFEST=""
PARALLELISM=""
OUTPUT_ROOT="data/ispy2_streaming"
SETUP_SCRIPT=""
ARRAY_SCRIPT="slurm/ispy2_streaming_array.slurm"
MERGE_SCRIPT="slurm/ispy2_streaming_merge.slurm"
CONTROLLER_SCRIPT="slurm/ispy2_autoresubmit_controller.slurm"
TIME_LIMIT="05:55:00"
PARTITION=""
ACCOUNT=""
QOS=""
MEMORY="8G"
CPUS="2"
MIN_FREE_SPACE_GB="50"
MAX_ACTIVE_SCRATCH_GB="75"
KEEP_FAILED_DICOM="0"
DO_MERGE="1"
CLINICAL=""
DEFAULT_TREATMENT_CONTEXT="neoadjuvant chemotherapy"
V1_OUTPUT="data/v1_prior_stack/processed/v1_eval_cohort.streamed.jsonl"
RUN_V1_BUILD="1"
MAX_ROUNDS="100"
NO_PROGRESS_LIMIT="3"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --setup-script) SETUP_SCRIPT="$2"; shift 2 ;;
    --array-script) ARRAY_SCRIPT="$2"; shift 2 ;;
    --merge-script) MERGE_SCRIPT="$2"; shift 2 ;;
    --controller-script) CONTROLLER_SCRIPT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --min-free-space-gb) MIN_FREE_SPACE_GB="$2"; shift 2 ;;
    --max-active-scratch-gb) MAX_ACTIVE_SCRATCH_GB="$2"; shift 2 ;;
    --keep-failed-dicom) KEEP_FAILED_DICOM="1"; shift ;;
    --clinical) CLINICAL="$2"; shift 2 ;;
    --default-treatment-context) DEFAULT_TREATMENT_CONTEXT="$2"; shift 2 ;;
    --v1-output) V1_OUTPUT="$2"; shift 2 ;;
    --no-v1-build) RUN_V1_BUILD="0"; shift ;;
    --no-merge) DO_MERGE="0"; shift ;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
    --no-progress-limit) NO_PROGRESS_LIMIT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$MANIFEST" ] || [ -z "$PARALLELISM" ]; then
  usage >&2
  exit 2
fi

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi

if [ ! -f "$CONTROLLER_SCRIPT" ]; then
  echo "Controller script not found: $CONTROLLER_SCRIPT" >&2
  exit 2
fi

mkdir -p logs/slurm

RUN_DIR="$(pwd)"

CTRL_SBATCH=(sbatch --parsable --time 00:10:00 --mem 1G --cpus-per-task 1)
if [ -n "$PARTITION" ]; then CTRL_SBATCH+=(--partition "$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then CTRL_SBATCH+=(--account "$ACCOUNT"); fi
if [ -n "$QOS" ]; then CTRL_SBATCH+=(--qos "$QOS"); fi

CTRL_SBATCH+=(
  --export "ALL,RUN_DIR=${RUN_DIR},MANIFEST=${MANIFEST},PARALLELISM=${PARALLELISM},OUTPUT_ROOT=${OUTPUT_ROOT},SETUP_SCRIPT=${SETUP_SCRIPT},ARRAY_SCRIPT=${ARRAY_SCRIPT},MERGE_SCRIPT=${MERGE_SCRIPT},CONTROLLER_SCRIPT=${CONTROLLER_SCRIPT},PARTITION=${PARTITION},ACCOUNT=${ACCOUNT},QOS=${QOS},TIME_LIMIT=${TIME_LIMIT},MEMORY=${MEMORY},CPUS=${CPUS},MIN_FREE_SPACE_GB=${MIN_FREE_SPACE_GB},MAX_ACTIVE_SCRATCH_GB=${MAX_ACTIVE_SCRATCH_GB},KEEP_FAILED_DICOM=${KEEP_FAILED_DICOM},DO_MERGE=${DO_MERGE},CLINICAL=${CLINICAL},DEFAULT_TREATMENT_CONTEXT=${DEFAULT_TREATMENT_CONTEXT},V1_OUTPUT=${V1_OUTPUT},RUN_V1_BUILD=${RUN_V1_BUILD},ROUND=1,MAX_ROUNDS=${MAX_ROUNDS},PREVIOUS_COMPLETE=-1,NO_PROGRESS_ROUNDS=0,NO_PROGRESS_LIMIT=${NO_PROGRESS_LIMIT}"
  "$CONTROLLER_SCRIPT"
)

printf 'Submitting first controller:'
printf ' %q' "${CTRL_SBATCH[@]}"
echo

job_id="$("${CTRL_SBATCH[@]}")"
echo "Submitted first controller: ${job_id}"
echo "This launcher is done. Slurm dependencies will handle the rest."
