from __future__ import annotations

from types import SimpleNamespace

from oncotwin_api.demo_assistant import lookup_question, render_answer, suggested_questions
from oncotwin_api.assistant_schemas import AssistantAnswerLLM, AssistantAskRequest
from oncotwin_api.assistant_service import assistant_stage


def test_stage_detection():
    assert assistant_stage({"latest_scan": {"assessment": "stable"}, "genomics": []}) == "baseline"
    assert assistant_stage({"latest_scan": {"assessment": "progression"}, "genomics": []}) == "after_progression"
    assert assistant_stage({"latest_scan": {"assessment": "progression"}, "genomics": [{"alteration": "ESR1 D538G"}]}) == "after_esr1"


def test_demo_lookup_is_authored_not_semantic():
    authored, row = lookup_question("after_progression", "Why did my outlook change?")
    assert authored == "Why did my outlook change?"
    assert row is not None
    authored2, row2 = lookup_question("after_progression", "why did my outlook change")
    assert authored2 == authored and row2 is not None
    assert lookup_question("after_progression", "tell me why things got worse please") == (None, None)


def test_demo_dynamic_forecast_injection():
    _, row = lookup_question("after_progression", "Why did my outlook change?")
    assert row is not None
    rendered = render_answer(row.answer, {
        "pre_pfs_6m_display": "72%",
        "selected_pfs_6m_display": "49%",
        "delta_6m_pp_display": "-23",
    })
    assert "72%" in rendered
    assert "49%" in rendered
    assert "-23 percentage points" in rendered


def test_suggestions_are_broad_and_nuanced_questions_stay_freeform():
    suggestions = suggested_questions("after_progression")
    assert len(suggestions) == 4
    assert "What changed on my newest scan?" in suggestions
    assert "Can you tell which lesion caused the forecast to drop?" not in suggestions
    assert lookup_question("after_progression", "Can you tell which lesion caused the forecast to drop?")[1] is not None


def test_esr1_boundary_is_explicit():
    _, row = lookup_question("after_esr1", "Does the ESR1 result change my OncoTwin forecast?")
    assert row is not None
    text = row.answer.lower()
    assert "not in the current deployed forecasting model" in text
    assert "rather than inventing a genomic input" in text


def test_live_response_schema_accepts_grounded_shape():
    value = AssistantAnswerLLM.model_validate({
        "answer": "The verified record shows a change on the newest scan.",
        "suggested_followups": ["What changed on the scan?"],
        "source_keys": ["fact:1"],
        "uncertainty_note": "This is a research-model estimate.",
    })
    assert value.source_keys == ["fact:1"]
    req = AssistantAskRequest(question="What changed?", history=[])
    assert req.question == "What changed?"
