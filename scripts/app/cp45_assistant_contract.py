from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
api = ROOT / "services/oncotwin-api/src/oncotwin_api"
web = ROOT / "apps/oncotwin-web"

assistant_service = (api / "assistant_service.py").read_text()
demo = (api / "demo_assistant.py").read_text()
router = (api / "llm_router.py").read_text()
main = (api / "main.py").read_text()
overview = (web / "app/app/page.tsx").read_text()
update = (web / "app/app/update/page.tsx").read_text()
component = (web / "components/AskOncoTwin.tsx").read_text()
types = (web / "lib/types.ts").read_text()

checks = {
    "assistant_files_present": all((api / x).is_file() for x in ["assistant_schemas.py", "assistant_service.py", "demo_assistant.py"]) and (web / "components/AskOncoTwin.tsx").is_file(),
    "same_api_contract": '/assistant/messages' in main and '@app.get("/v1/patients/{patient_id}/assistant")' in main,
    "demo_fails_closed_before_router": "if fixture_mode_enabled():" in assistant_service and assistant_service.index("if fixture_mode_enabled():") < assistant_service.index("run_structured("),
    "demo_exact_lookup": "lookup_question(stage, request.question)" in assistant_service and "normalize_question" in demo,
    "demo_no_semantic_matcher": (
        "def normalize_question" in demo
        and "normalize_question(authored) == wanted" in demo
        and all(
            marker not in demo.lower()
            for marker in [
                "import rapidfuzz",
                "from rapidfuzz",
                "levenshtein(",
                "cosine_similarity(",
                "sentence_transformer",
                "sentence-transformer",
                "get_close_matches(",
                "fuzz.ratio(",
                "fuzz.partial_ratio(",
            ]
        )
    ),
    "live_uses_shared_router": 'task="patient_assistant"' in assistant_service and '"patient_assistant"' in router,
    "live_grounded_context": all(x in assistant_service for x in ["verified_patient_state", "verified_facts", "forecast_context", "source_catalog"]),
    "live_history_supported": "recent_conversation" in assistant_service and "history" in component,
    "treatment_boundary": "Do not tell the patient to start, stop, switch, or choose a treatment" in assistant_service,
    "forecast_boundary": "Do not calculate new prognosis values" in assistant_service,
    "suggested_questions": "SUGGESTED_QUESTIONS" in demo and "suggested_questions" in component,
    "freeform_input": "textarea" in component and "Ask a question" in component,
    "source_viewer": "SourceViewer" in component,
    "overview_integrated": "AskOncoTwin" in overview,
    "update_integrated": "AskOncoTwin" in update,
    "typed_contract": all(x in types for x in ["AssistantBootstrap", "AssistantAnswer", "AssistantTurn"]),
    "no_provider_calls_in_ui": all(x not in component for x in ["api.groq.com", "generativelanguage.googleapis.com", "OpenAI("]),
    "nuanced_demo_coverage": all(q in demo for q in [
        "Did the neutropenia or ribociclib dose reduction cause the progression?",
        "Can you tell which lesion caused the forecast to drop?",
        "Why can AI explain this result but the prediction model can't use it?",
        "Was the model already expecting this progression?",
    ]),
}

print(json.dumps(checks, indent=2, sort_keys=True))
failed = [k for k, v in checks.items() if not v]
if failed:
    raise SystemExit("CP4.5 assistant contract failures: " + ", ".join(failed))
print("CP45_ASSISTANT_CONTRACT_PASS")
