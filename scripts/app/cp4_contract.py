from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
api = ROOT / "services/oncotwin-api/src/oncotwin_api"
web = ROOT / "apps/oncotwin-web"
checks = {}

required = [
    api / "llm_router.py",
    api / "intelligence_schemas.py",
    api / "intelligence_service.py",
    api / "evidence_service.py",
    api / "export_service.py",
    web / "app/app/prepare/page.tsx",
]
checks["cp4_files_present"] = all(p.is_file() for p in required)

main = (api / "main.py").read_text()
checks["record_explanation_routes"] = '/explanation")' in main
checks["what_changed_route"] = 'intelligence/what-changed' in main
checks["questions_route"] = 'intelligence/questions' in main
checks["visit_brief_route"] = 'intelligence/visit-brief' in main
checks["evidence_route"] = '@app.post("/v1/patients/{patient_id}/evidence")' in main
checks["export_route"] = '@app.post("/v1/patients/{patient_id}/export")' in main

router = (api / "llm_router.py").read_text()
checks["provider_abstraction"] = "class GroqProvider" in router and "class GeminiProvider" in router and "def run_structured" in router
checks["fixture_fail_closed"] = "has no deterministic workflow fixture" in router
checks["trial_eligibility_guard"] = "asserted eligibility" in router
checks["forecast_not_in_router"] = "selected_logits" not in router and "locked_alpha" not in router

# Pages must not call provider endpoints or SDKs directly.
page_text = "\n".join(p.read_text(errors="ignore") for p in web.rglob("*.tsx"))
checks["no_provider_calls_in_pages"] = all(x not in page_text for x in ["api.groq.com", "generativelanguage.googleapis.com", "OpenAI("])

update = (web / "app/app/update/page.tsx").read_text()
checks["what_changed_integrated"] = "what-changed" in update and "Prepare for your visit" in update
records = (web / "app/app/records/page.tsx").read_text()
checks["record_explanation_integrated"] = "Explain this report" in records
prepare = (web / "app/app/prepare/page.tsx").read_text()
checks["visit_prepare_integrated"] = "QUESTIONS" in prepare and "EVIDENCE & TRIALS" in prepare and "Create shareable PDF" in prepare

print(json.dumps(checks, indent=2, sort_keys=True))
failed = [k for k, v in checks.items() if not v]
if failed:
    raise SystemExit("CP4 contract failures: " + ", ".join(failed))
print("CP4_SOURCE_CONTRACT_PASS")
