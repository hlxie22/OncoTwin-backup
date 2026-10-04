from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import oncotwin_api.llm_router as router
from oncotwin_api.intelligence_schemas import RecordExplanationLLM, TrialRelevanceLLM
from oncotwin_api.models import Base, GeneratedArtifact, ProviderCooldown


class _FakeProvider:
    def __init__(self, calls, fail_first=False, unsafe_trial=False):
        self.calls = calls
        self.fail_first = fail_first
        self.unsafe_trial = unsafe_trial

    def generate_json(self, *, model, system, payload, schema, reasoning):
        self.calls.append(model)
        if self.fail_first and len(self.calls) == 1:
            raise router.ProviderError("quota", rate_limited=True, retry_after_seconds=60)
        if self.unsafe_trial:
            raw = {
                "relevance_summary": "You are eligible for this trial.",
                "matching_features": [],
                "questions_to_confirm": [],
                "cautions": [],
                "source_ids": ["NCT00000000"],
            }
        else:
            raw = {"summary": "A grounded explanation.", "key_points": ["Grounded point"], "source_keys": []}
        return router.ProviderResponse(raw, "fake", model, 1, {})


def _db():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return Session(engine)


def test_router_falls_back_after_rate_limit_and_caches(monkeypatch):
    monkeypatch.delenv("ONCOTWIN_DEMO_FIXTURE_ID", raising=False)
    calls = []
    fake = _FakeProvider(calls, fail_first=True)
    monkeypatch.setattr(router, "_provider", lambda name: fake)
    with _db() as db:
        result = router.run_structured(
            db,
            task="record_explanation",
            payload={"text": "grounded"},
            response_model=RecordExplanationLLM,
            cache_basis={"document_sha256": "abc"},
        )
        assert result.value.summary == "A grounded explanation."
        assert len(calls) == 2
        assert db.scalar(select(ProviderCooldown)) is not None
        cached = router.run_structured(
            db,
            task="record_explanation",
            payload={"text": "grounded"},
            response_model=RecordExplanationLLM,
            cache_basis={"document_sha256": "abc"},
        )
        assert cached.cache_hit is True
        assert len(calls) == 2
        assert db.scalar(select(GeneratedArtifact)) is not None


def test_trial_eligibility_assertion_is_rejected(monkeypatch):
    monkeypatch.delenv("ONCOTWIN_DEMO_FIXTURE_ID", raising=False)
    fake = _FakeProvider([], unsafe_trial=True)
    monkeypatch.setattr(router, "_provider", lambda name: fake)
    with _db() as db:
        try:
            router.run_structured(
                db,
                task="trial_relevance",
                payload={"trial": "NCT00000000"},
                response_model=TrialRelevanceLLM,
                cache_basis={"trial": "NCT00000000"},
            )
        except router.RouterUnavailable as exc:
            assert "eligibility" in str(exc).lower() or "failed" in str(exc).lower()
        else:
            raise AssertionError("unsafe eligibility assertion should fail closed")
