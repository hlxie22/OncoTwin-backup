from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GroundingRef(BaseModel):
    key: str
    kind: Literal["record", "literature", "trial"]
    label: str
    document_id: str | None = None
    document_name: str | None = None
    page_number: int | None = None
    source_snippet: str | None = None
    source_id: str | None = None
    url: str | None = None


class RecordExplanationLLM(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list, max_length=8)
    source_keys: list[str] = Field(default_factory=list, max_length=12)


class WhatChangedLLM(BaseModel):
    headline: str
    scan_summary: str
    state_change_summary: str
    outlook_context: str
    questions_to_consider: list[str] = Field(default_factory=list, max_length=5)
    source_keys: list[str] = Field(default_factory=list, max_length=12)


class VisitQuestionItem(BaseModel):
    question: str
    why_this_may_be_useful: str
    source_keys: list[str] = Field(default_factory=list, max_length=8)


class VisitQuestionsLLM(BaseModel):
    intro: str
    questions: list[VisitQuestionItem] = Field(default_factory=list, min_length=1, max_length=8)


class VisitBriefLLM(BaseModel):
    headline: str
    current_picture: list[str] = Field(default_factory=list, max_length=8)
    recent_changes: list[str] = Field(default_factory=list, max_length=8)
    uncertainties: list[str] = Field(default_factory=list, max_length=8)
    discussion_points: list[str] = Field(default_factory=list, max_length=8)
    source_keys: list[str] = Field(default_factory=list, max_length=20)


class EvidenceSummaryLLM(BaseModel):
    plain_language_summary: str
    why_it_may_be_relevant: str
    limitations: str
    source_ids: list[str] = Field(default_factory=list, max_length=6)


class TrialRelevanceLLM(BaseModel):
    relevance_summary: str
    matching_features: list[str] = Field(default_factory=list, max_length=8)
    questions_to_confirm: list[str] = Field(default_factory=list, max_length=8)
    cautions: list[str] = Field(default_factory=list, max_length=5)
    source_ids: list[str] = Field(default_factory=list, max_length=6)


class ArtifactEnvelope(BaseModel):
    id: str | None = None
    artifact_type: str
    status: str = "COMPLETED"
    cache_hit: bool = False
    patient_id: str | None = None
    document_id: str | None = None
    state_hash: str | None = None
    provider: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    generated_at: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)
    grounding: list[GroundingRef] = Field(default_factory=list)


class WhatChangedRequest(BaseModel):
    before_state_hash: str
    document_id: str | None = None


class IntelligenceGenerateRequest(BaseModel):
    refresh: bool = False


class EvidenceGenerateRequest(BaseModel):
    refresh: bool = False
    max_literature: int = Field(default=4, ge=1, le=8)
    max_trials: int = Field(default=4, ge=1, le=8)


class ExportRequest(BaseModel):
    include_evidence: bool = True
    include_forecast: bool = True


class ProviderIdentity(BaseModel):
    provider: str
    model_name: str
    tier: str
    prompt_version: str
    schema_version: str
