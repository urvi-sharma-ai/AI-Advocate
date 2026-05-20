from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class LegalDocument(BaseModel):
    """A source statute file described in the manifest."""

    id: str
    act_title: str
    domain: Literal["Finance", "Land"]
    source_path: str
    relative_file: str
    act_version: str
    effective_date: date | None = None
    supersedes: str | None = None


class Chunk(BaseModel):
    """Indexed text slice with mandatory section-context prefix in ``text``."""

    id: str
    text: str
    document_id: str
    domain: Literal["Finance", "Land"]
    act_title: str
    act_version: str
    source_path: str
    chunk_index: int
    page_start: int
    page_end: int
    section_number: str | None = None
    section_title: str | None = None
    section_label: str | None = None
    subsection_label: str | None = None


class Citation(BaseModel):
    """A reference line shown to the user."""

    act_title: str
    act_version: str
    section_label: str | None = None
    pages: str
    source_path: str


PredictedDomain = Literal["Finance", "Land", "All"]
TemporalScope = Literal["current", "legacy", "unspecified"]


class RewriteResult(BaseModel):
    """Structured rewrite + routing prediction from the LLM."""

    rewritten_question: str
    predicted_domain: PredictedDomain
    temporal_scope: TemporalScope = "unspecified"
    reasoning: str = Field(default="", description="Short rationale for debugging / UI")


class Answer(BaseModel):
    """Assistant response grounded in statute chunks."""

    text: str
    citations: list[Citation]
    grounded: bool
    act_versions_used: list[str] = Field(default_factory=list)
    temporal_scope_applied: TemporalScope = "unspecified"
    routing_reasoning: str = ""
