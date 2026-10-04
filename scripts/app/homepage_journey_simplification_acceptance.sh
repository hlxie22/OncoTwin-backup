#!/usr/bin/env bash
set -euo pipefail
REPO="${ONCOTWIN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PAGE="$REPO/apps/oncotwin-web/app/page.tsx"
CSS="$REPO/apps/oncotwin-web/app/globals.css"
WEB="$REPO/apps/oncotwin-web"

echo "[HOME-SIMPLIFY] repository=$REPO"
echo "[HOME-SIMPLIFY] 1/3 source contract"
python3 - "$PAGE" "$CSS" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).read_text()
c = Path(sys.argv[2]).read_text()
checks = {
  "hero_outcome": "Understand how your cancer is changing—not just what each report says." in p,
  "progression_training_explicit": "machine-learning model specially trained to predict cancer progression" in p,
  "journey_model_label": "CANCER PROGRESSION MODEL" in p,
  "journey_is_concise": "Different kinds of AI help at different moments in your cancer journey." in p,
  "hero_anchor_works": 'href="#journey"' in p and 'id="journey"' in p,
  "no_top_right_links": "landing-nav-actions" not in p and "landing-nav-link" not in p,
  "no_redundant_ai_visual": "TWO TYPES OF AI, DIFFERENT JOBS" not in p and "intelligence-visual" not in p,
  "no_bottom_cta": "ONE EVOLVING PICTURE" not in p and "landing-bottom-cta" not in p,
  "no_research_prototype_copy": "research prototype" not in p.lower(),
  "css_marker": "ONCOTWIN HOMEPAGE AI/JOURNEY POLISH START" in c,
}
failed = [k for k, v in checks.items() if not v]
print({"checks": checks, "failed": failed, "status": "PASS" if not failed else "FAIL"})
if failed:
    raise SystemExit(1)
PY

echo "[HOME-SIMPLIFY] 2/3 production build"
(
  cd "$WEB"
  npm run build
)

echo "[HOME-SIMPLIFY] 3/3 build route check"
test -f "$WEB/.next/BUILD_ID"
grep -q "CANCER PROGRESSION MODEL" "$PAGE"
grep -q "specially trained to predict cancer progression" "$PAGE"
echo "HOMEPAGE_JOURNEY_SIMPLIFICATION_PASS"
