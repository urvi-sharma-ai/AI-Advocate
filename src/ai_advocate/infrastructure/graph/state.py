from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class RAGGraphState(TypedDict, total=False):
    question: str
    rewritten_question: str
    predicted_domain: str
    temporal_scope: str
    domain_filter_override: str | None
    reference_date: str
    routing_reasoning: str

    documents: list[dict[str, Any]]
    generation: str
    citations: list[dict[str, Any]]
    grounded: bool

    retrieval_attempts: int
    grade: str

    # Returned to UI
    act_versions_used: NotRequired[list[str]]
    temporal_scope_applied: NotRequired[str]
