from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DemoAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    display_name: str


class PatientCreate(BaseModel):
    display_name: str = "My cancer journey"
    disease_pack: str = "mbc_v2"


class PatientOut(BaseModel):
    id: str
    display_name: str
    disease_pack: str


class ExtractedFact(BaseModel):
    fact_type: Literal[
        "diagnosis",
        "er_status",
        "pr_status",
        "her2_status",
        "treatment",
        "disease_site",
        "scan_assessment",
        "scan_modality",
        "genomic_assay",
        "genomic_alteration",
        "lab",
    ]
    value: str
    effective_date: str | None = None
    unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_page: int = Field(ge=1)
    source_snippet: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractionPayload(BaseModel):
    document_type: Literal["pathology", "radiology", "genomics", "treatment", "laboratory", "mixed", "other"]
    facts: list[ExtractedFact]


class CandidateDecision(BaseModel):
    decision: Literal["accept", "reject", "edit"]
    edited_value: str | None = None
    edited_effective_date: str | None = None
    note: str | None = None


class CandidateFactOut(BaseModel):
    id: str
    document_id: str
    document_name: str
    fact_type: str
    value: str
    effective_date: str | None
    confidence: float
    source_page: int
    source_snippet: str
    review_reason: str | None
    status: str


class SourceOut(BaseModel):
    document_id: str
    document_name: str
    page_number: int
    source_snippet: str


class TimelineEventOut(BaseModel):
    id: str
    event_type: str
    event_date: str | None
    title: str
    summary: str
    source: SourceOut | None = None


class StateResponse(BaseModel):
    snapshot_id: str | None
    state_hash: str | None
    state: dict[str, Any]
    sources: dict[str, SourceOut]


class DocumentOut(BaseModel):
    id: str
    filename: str
    document_type: str | None
    status: str
    created_at: str
    processing_error: str | None = None
