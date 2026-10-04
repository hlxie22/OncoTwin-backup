#!/usr/bin/env bash

# Run a command strictly enough to detect failure, but never propagate that
# failure out of this wrapper. This protects an interactive Slurm allocation.
set +e

if [[ $# -eq 0 ]]; then
  echo "usage: safe_run.sh COMMAND [ARGS...]"
  exit 0
fi

echo
echo "=============================================================="
echo " OncoTwin allocation-safe runner"
echo " Host: $(hostname)"
echo " Command: $*"
echo "=============================================================="
echo

"$@"
RC=$?

echo
echo "=============================================================="
if [[ $RC -eq 0 ]]; then
  echo " COMMAND_STATUS=PASS"
else
  echo " COMMAND_STATUS=FAIL"
fi
echo " CHILD_EXIT_CODE=$RC"
echo " Allocation-safe wrapper returning 0."
echo "=============================================================="
echo

exit 0
