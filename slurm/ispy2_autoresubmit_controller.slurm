#!/usr/bin/env bash
#SBATCH --job-name=ispy2-watch
#SBATCH --output=logs/slurm/ispy2-watch-%j.out
#SBATCH --error=logs/slurm/ispy2-watch-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

RUN_DIR="${RUN_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$RUN_DIR"

MANIFEST="${MANIFEST:?missing MANIFEST}"
PARALLELISM="${PARALLELISM:?missing PARALLELISM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/ispy2_streaming}"
SETUP_SCRIPT="${SETUP_SCRIPT:-}"
ARRAY_SCRIPT="${ARRAY_SCRIPT:-slurm/ispy2_streaming_array.slurm}"
MERGE_SCRIPT="${MERGE_SCRIPT:-slurm/ispy2_streaming_merge.slurm}"
CONTROLLER_SCRIPT="${CONTROLLER_SCRIPT:-slurm/ispy2_autoresubmit_controller.slurm}"

PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
TIME_LIMIT="${TIME_LIMIT:-05:55:00}"
MEMORY="${MEMORY:-8G}"
CPUS="${CPUS:-2}"

MIN_FREE_SPACE_GB="${MIN_FREE_SPACE_GB:-50}"
MAX_ACTIVE_SCRATCH_GB="${MAX_ACTIVE_SCRATCH_GB:-75}"
KEEP_FAILED_DICOM="${KEEP_FAILED_DICOM:-0}"

DO_MERGE="${DO_MERGE:-1}"
CLINICAL="${CLINICAL:-}"
DEFAULT_TREATMENT_CONTEXT="${DEFAULT_TREATMENT_CONTEXT:-neoadjuvant chemotherapy}"
V1_OUTPUT="${V1_OUTPUT:-data/v1_prior_stack/processed/v1_eval_cohort.streamed.jsonl}"
RUN_V1_BUILD="${RUN_V1_BUILD:-1}"

ROUND="${ROUND:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-100}"
PREVIOUS_COMPLETE="${PREVIOUS_COMPLETE:--1}"
NO_PROGRESS_ROUNDS="${NO_PROGRESS_ROUNDS:-0}"
NO_PROGRESS_LIMIT="${NO_PROGRESS_LIMIT:-3}"

mkdir -p logs/slurm "$OUTPUT_ROOT/logs"

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi

if [ ! -f "$ARRAY_SCRIPT" ]; then
  echo "Array script not found: $ARRAY_SCRIPT" >&2
  exit 2
fi

progress_file="$(mktemp)"
python3 - "$MANIFEST" "$OUTPUT_ROOT" > "$progress_file" <<'PY'
import csv
import json
import re
import shlex
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
output_root = Path(sys.argv[2])

case_id_fields = (
    "case_id",
    "patient_id",
    "subject_id",
    "patientid",
    "dicom_patient_id",
    "submitter_case_id",
    "PatientID",
)

delimiter = "\t" if manifest.suffix.lower() in {".tsv", ".txt"} else ","
with manifest.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter=delimiter))

def first_case_id(row):
    for field in case_id_fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value).strip()
    return ""

def slug(value):
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:180] or "unknown"

case_ids = sorted({first_case_id(row) for row in rows})
case_ids = [case_id for case_id in case_ids if case_id]

complete = []
incomplete = []
failed_or_partial = []
missing_or_killed = []

for case_id in case_ids:
    status_path = output_root / "logs" / "patients" / f"{slug(case_id)}.json"
    status = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}

    is_complete = (
        status.get("download_status")
        == status.get("extract_status")
        == status.get("feature_status")
        == "complete"
    )

    if is_complete:
        complete.append(case_id)
    else:
        incomplete.append(case_id)
        if status_path.exists():
            failed_or_partial.append(case_id)
        else:
            missing_or_killed.append(case_id)

print(f"TOTAL={len(case_ids)}")
print(f"COMPLETE={len(complete)}")
print(f"INCOMPLETE={len(incomplete)}")
print(f"FAILED_OR_PARTIAL={len(failed_or_partial)}")
print(f"MISSING_OR_KILLED={len(missing_or_killed)}")
print("FIRST_INCOMPLETE=" + shlex.quote(" ".join(incomplete[:20])))
PY

# shellcheck source=/dev/null
source "$progress_file"
rm -f "$progress_file"

echo "Round ${ROUND}: complete=${COMPLETE}/${TOTAL}, incomplete=${INCOMPLETE}, failed_or_partial=${FAILED_OR_PARTIAL}, missing_or_killed=${MISSING_OR_KILLED}"
if [ -n "${FIRST_INCOMPLETE:-}" ]; then
  echo "First incomplete cases: ${FIRST_INCOMPLETE}"
fi

if [ "$TOTAL" -lt 1 ]; then
  echo "No cases found in manifest." >&2
  exit 2
fi

if [ "$INCOMPLETE" -eq 0 ]; then
  echo "All extraction cases complete."

  if [ "$DO_MERGE" = "1" ]; then
    MERGE_SBATCH=(sbatch --parsable)
    if [ -n "$PARTITION" ]; then MERGE_SBATCH+=(--partition "$PARTITION"); fi
    if [ -n "$ACCOUNT" ]; then MERGE_SBATCH+=(--account "$ACCOUNT"); fi
    if [ -n "$QOS" ]; then MERGE_SBATCH+=(--qos "$QOS"); fi

    MERGE_SBATCH+=(
      --export "ALL,RUN_DIR=${RUN_DIR},OUTPUT_ROOT=${OUTPUT_ROOT},CLINICAL=${CLINICAL},DEFAULT_TREATMENT_CONTEXT=${DEFAULT_TREATMENT_CONTEXT},V1_OUTPUT=${V1_OUTPUT},RUN_V1_BUILD=${RUN_V1_BUILD},SETUP_SCRIPT=${SETUP_SCRIPT}"
      "$MERGE_SCRIPT"
    )

    printf 'Submitting merge job:'
    printf ' %q' "${MERGE_SBATCH[@]}"
    echo
    merge_job_id="$("${MERGE_SBATCH[@]}")"
    echo "Submitted merge job: ${merge_job_id}"
  fi

  exit 0
fi

if [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
  echo "Reached MAX_ROUNDS=${MAX_ROUNDS}; stopping with ${INCOMPLETE} incomplete cases." >&2
  exit 1
fi

if [ "$COMPLETE" -eq "$PREVIOUS_COMPLETE" ]; then
  NO_PROGRESS_ROUNDS=$((NO_PROGRESS_ROUNDS + 1))
else
  NO_PROGRESS_ROUNDS=0
fi

if [ "$NO_PROGRESS_ROUNDS" -ge "$NO_PROGRESS_LIMIT" ]; then
  echo "No new completed cases after ${NO_PROGRESS_ROUNDS} controller round(s); stopping." >&2
  echo "Inspect ${OUTPUT_ROOT}/logs/errors and ${OUTPUT_ROOT}/logs/patients before resubmitting." >&2
  exit 1
fi

WORKER_COUNT="$PARALLELISM"
if [ "$WORKER_COUNT" -gt "$TOTAL" ]; then
  WORKER_COUNT="$TOTAL"
fi

ARRAY_SBATCH=(sbatch --parsable)
if [ -n "$PARTITION" ]; then ARRAY_SBATCH+=(--partition "$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then ARRAY_SBATCH+=(--account "$ACCOUNT"); fi
if [ -n "$QOS" ]; then ARRAY_SBATCH+=(--qos "$QOS"); fi
if [ -n "$TIME_LIMIT" ]; then ARRAY_SBATCH+=(--time "$TIME_LIMIT"); fi
if [ -n "$MEMORY" ]; then ARRAY_SBATCH+=(--mem "$MEMORY"); fi
if [ -n "$CPUS" ]; then ARRAY_SBATCH+=(--cpus-per-task "$CPUS"); fi

ARRAY_SBATCH+=(
  --array "1-${WORKER_COUNT}"
  --export "ALL,RUN_DIR=${RUN_DIR},MANIFEST=${MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT},PATIENT_COUNT=${TOTAL},WORKER_COUNT=${WORKER_COUNT},MIN_FREE_SPACE_GB=${MIN_FREE_SPACE_GB},MAX_ACTIVE_SCRATCH_GB=${MAX_ACTIVE_SCRATCH_GB},KEEP_FAILED_DICOM=${KEEP_FAILED_DICOM},FORCE=0,SETUP_SCRIPT=${SETUP_SCRIPT}"
  "$ARRAY_SCRIPT"
)

printf 'Submitting extraction array:'
printf ' %q' "${ARRAY_SBATCH[@]}"
echo

array_job_id="$("${ARRAY_SBATCH[@]}")"
array_job_id="${array_job_id%%;*}"
echo "Submitted extraction array: ${array_job_id}"

NEXT_ROUND=$((ROUND + 1))

CTRL_SBATCH=(sbatch --parsable)
if [ -n "$PARTITION" ]; then CTRL_SBATCH+=(--partition "$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then CTRL_SBATCH+=(--account "$ACCOUNT"); fi
if [ -n "$QOS" ]; then CTRL_SBATCH+=(--qos "$QOS"); fi

CTRL_SBATCH+=(
  --dependency "afterany:${array_job_id}"
  --export "ALL,RUN_DIR=${RUN_DIR},MANIFEST=${MANIFEST},PARALLELISM=${PARALLELISM},OUTPUT_ROOT=${OUTPUT_ROOT},SETUP_SCRIPT=${SETUP_SCRIPT},ARRAY_SCRIPT=${ARRAY_SCRIPT},MERGE_SCRIPT=${MERGE_SCRIPT},CONTROLLER_SCRIPT=${CONTROLLER_SCRIPT},PARTITION=${PARTITION},ACCOUNT=${ACCOUNT},QOS=${QOS},TIME_LIMIT=${TIME_LIMIT},MEMORY=${MEMORY},CPUS=${CPUS},MIN_FREE_SPACE_GB=${MIN_FREE_SPACE_GB},MAX_ACTIVE_SCRATCH_GB=${MAX_ACTIVE_SCRATCH_GB},KEEP_FAILED_DICOM=${KEEP_FAILED_DICOM},DO_MERGE=${DO_MERGE},CLINICAL=${CLINICAL},DEFAULT_TREATMENT_CONTEXT=${DEFAULT_TREATMENT_CONTEXT},V1_OUTPUT=${V1_OUTPUT},RUN_V1_BUILD=${RUN_V1_BUILD},ROUND=${NEXT_ROUND},MAX_ROUNDS=${MAX_ROUNDS},PREVIOUS_COMPLETE=${COMPLETE},NO_PROGRESS_ROUNDS=${NO_PROGRESS_ROUNDS},NO_PROGRESS_LIMIT=${NO_PROGRESS_LIMIT}"
  "$CONTROLLER_SCRIPT"
)

printf 'Submitting next controller:'
printf ' %q' "${CTRL_SBATCH[@]}"
echo

next_controller_job_id="$("${CTRL_SBATCH[@]}")"
echo "Submitted next controller: ${next_controller_job_id}"
