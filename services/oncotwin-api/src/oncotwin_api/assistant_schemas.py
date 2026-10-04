from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .intelligence_schemas import GroundingRef


class AssistantTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class AssistantAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[AssistantTurn] = Field(default_factory=list, max_length=12)


class AssistantAnswerLLM(BaseModel):
    answer: str = Field(min_length=1, max_length=9000)
    suggested_followups: list[str] = Field(default_factory=list, max_length=4)
    source_keys: list[str] = Field(default_factory=list, max_length=12)
    uncertainty_note: str | None = Field(default=None, max_length=1500)


class AssistantBootstrapResponse(BaseModel):
    stage: Literal["baseline", "after_progression", "after_esr1"]
    mode: Literal["fixture", "llm"]
    greeting: str
    suggested_questions: list[str] = Field(default_factory=list, max_length=6)


class AssistantAnswerResponse(BaseModel):
    stage: Literal["baseline", "after_progression", "after_esr1"]
    mode: Literal["fixture", "llm"]
    matched_question: bool = True
    answer: str
    suggested_questions: list[str] = Field(default_factory=list, max_length=6)
    suggested_followups: list[str] = Field(default_factory=list, max_length=4)
    grounding: list[GroundingRef] = Field(default_factory=list)
    forecast_refs: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
    model_name: str | None = None
    uncertainty_note: str | None = None
