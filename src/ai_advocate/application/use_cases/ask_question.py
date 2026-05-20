from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ai_advocate.domain.models import Answer, Citation


@dataclass(frozen=True)
class AskParams:
    question: str
    domain_filter_override: str | None = None
    reference_date: date | None = None


class AskQuestionUseCase:
    def __init__(self, *, graph) -> None:
        self._graph = graph

    def run(self, params: AskParams) -> Answer:
        initial_state = {
            "question": params.question,
            "domain_filter_override": params.domain_filter_override,
            "reference_date": (params.reference_date or date.today()).isoformat(),
            "retrieval_attempts": 0,
        }

        out = self._graph.invoke(initial_state)
        citations = [Citation.model_validate(c) for c in (out.get("citations") or [])]
        return Answer(
            text=str(out.get("generation") or ""),
            citations=citations,
            grounded=bool(out.get("grounded", False)),
            act_versions_used=list(out.get("act_versions_used") or []),
            temporal_scope_applied=str(out.get("temporal_scope_applied") or "unspecified"),
            routing_reasoning=str(out.get("routing_reasoning") or ""),
        )
