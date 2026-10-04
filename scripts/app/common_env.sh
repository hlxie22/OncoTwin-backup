#!/usr/bin/env bash

# Shared runtime contract for all OncoTwin application scripts.
#
# The frozen research environment is deliberately NOT modified or selected
# here. Application work uses the dedicated oncotwin-app environment.

ONCOTWIN_APP_ENV="${ONCOTWIN_APP_ENV:-/home/henryxie/orcd/scratch/.conda/envs/oncotwin-app}"

if [[ ! -d "$ONCOTWIN_APP_ENV" ]]; then
  echo "ERROR: OncoTwin app environment not found:"
  echo "  $ONCOTWIN_APP_ENV"
  return 20 2>/dev/null || exit 20
fi

if [[ ! -x "$ONCOTWIN_APP_ENV/bin/python" ]]; then
  echo "ERROR: app Python not found:"
  echo "  $ONCOTWIN_APP_ENV/bin/python"
  return 21 2>/dev/null || exit 21
fi

# Make app Python, Node, npm, and their shared libraries authoritative for
# application scripts even when the Slurm shell did not activate Conda.
export PATH="$ONCOTWIN_APP_ENV/bin:$PATH"

# ORCD compute nodes expose an older system libstdc++. Conda's ICU/SQLite
# stack requires the newer C++ ABI shipped in the app environment.
case ":${LD_LIBRARY_PATH:-}:" in
  *":$ONCOTWIN_APP_ENV/lib:"*) ;;
  *)
    export LD_LIBRARY_PATH="$ONCOTWIN_APP_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
esac

export ONCOTWIN_PYTHON="${ONCOTWIN_PYTHON:-$ONCOTWIN_APP_ENV/bin/python}"
PY="$ONCOTWIN_PYTHON"
export PY
