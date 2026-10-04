#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source scripts/app/common_env.sh

if [[ -n "${ONCOTWIN_DEMO_FIXTURE_ID:-}" ]]; then
  echo "CP4 deterministic fixture mode: language workflows use fixtures; evidence uses the frozen snapshot; forecast remains real."
elif [[ -n "${ONCOTWIN_GROQ_API_KEY:-}" || -n "${ONCOTWIN_GOOGLE_API_KEY:-}" ]]; then
  export ONCOTWIN_EXTRACTOR_PROVIDER="${ONCOTWIN_EXTRACTOR_PROVIDER:-router}"
  export ONCOTWIN_EVIDENCE_MODE="${ONCOTWIN_EVIDENCE_MODE:-live}"
  echo "CP4 live intelligence enabled. Extractor provider: $ONCOTWIN_EXTRACTOR_PROVIDER"
else
  export ONCOTWIN_EXTRACTOR_PROVIDER="${ONCOTWIN_EXTRACTOR_PROVIDER:-rules}"
  echo "No Groq/Gemini key detected. Core app + deterministic extraction will run; live CP4 language workflows return a recoverable unavailable state."
fi

bash scripts/app/run_cp3_dev.sh
