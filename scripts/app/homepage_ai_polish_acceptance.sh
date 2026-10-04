#!/usr/bin/env bash
set -euo pipefail
REPO="${ONCOTWIN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PAGE="$REPO/apps/oncotwin-web/app/page.tsx"
CSS="$REPO/apps/oncotwin-web/app/globals.css"
WEB="$REPO/apps/oncotwin-web"
echo "[HOME] repository=$REPO"
echo "[HOME] 1/3 source contract"
python3 - "$PAGE" "$CSS" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).read_text()
c = Path(sys.argv[2]).read_text()
checks = {
  "hero_outcome": "Understand how your cancer is changing—not just what each report says." in p,
  "language_ai": "LANGUAGE AI" in p,
  "specialized_ml": "SPECIALIZED ML MODEL" in p and "specially trained machine-learning model" in p,
  "journey_value": "How OncoTwin helps as things change" in p and p.count("WHAT YOU GET") >= 1,
  "how_it_helps_anchor": 'href="#journey"' in p and 'id="journey"' in p,
  "no_metastatic_research_pill": "Metastatic breast cancer" not in p and "research prototype" not in p.lower(),
  "new_css": "ONCOTWIN HOMEPAGE AI/JOURNEY POLISH START" in c,
}
failed = [k for k,v in checks.items() if not v]
print({"checks": checks, "failed": failed, "status": "PASS" if not failed else "FAIL"})
if failed:
    raise SystemExit(1)
PY

echo "[HOME] 2/3 production build"
(
  cd "$WEB"
  npm run build
)

echo "[HOME] 3/3 build route check"
test -f "$WEB/.next/BUILD_ID"
grep -q "Understand how your cancer is changing" "$PAGE"
echo "HOMEPAGE_AI_POLISH_PASS"
