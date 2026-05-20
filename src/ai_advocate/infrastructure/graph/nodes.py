from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ai_advocate.domain.models import Chunk, Citation, RewriteResult
from ai_advocate.infrastructure.graph.filters import build_chroma_where
from ai_advocate.infrastructure.graph.state import RAGGraphState


class GraphRuntime:
    """Callable dependencies for LangGraph nodes (DI-friendly)."""

    def __init__(
        self,
        *,
        llm_invoke_json: Callable[[str, str], str],
        llm_invoke_messages: Callable[[str, str], str],
        embed_query: Callable[[str], list[float]],
        similarity_search: Callable[..., list[Chunk]],
    ) -> None:
        self.llm_invoke_json = llm_invoke_json
        self.llm_invoke_messages = llm_invoke_messages
        self.embed_query = embed_query
        self.similarity_search = similarity_search


def _parse_rewrite(content: str) -> RewriteResult:
    data = json.loads(content)
    return RewriteResult.model_validate(data)


def _parse_grade(content: str) -> bool:
    data = json.loads(content)
    return bool(data.get("relevant", False))


def rewrite_query(state: RAGGraphState, rt: GraphRuntime) -> dict[str, Any]:
    ref = state.get("reference_date") or ""
    system = (
        "You are a routing assistant for Indian statute QA. "
        "Return ONLY valid JSON with keys: "
        "rewritten_question (string), predicted_domain (Finance|Land|All), "
        "temporal_scope (current|legacy|unspecified), reasoning (short string).\n"
        "Domain rules: Land = real estate/property/RERA/registration; "
        "Finance = tax, company law, SEBI, forex/FEMA.\n"
        "Temporal rules: legacy ONLY when clearly about pre-2026 income tax disputes "
        "or the pre-2026 Income Tax Act.\n"
        "current for FY/returns/'this year' tax questions under the new regime.\n"
        "Use All if the question is genuinely cross-domain."
    )
    user = f"Reference date (ISO): {ref}\nUser question:\n{state['question']}\n"
    raw = rt.llm_invoke_json(system, user)
    rr = _parse_rewrite(raw)
    return {
        "rewritten_question": rr.rewritten_question,
        "predicted_domain": rr.predicted_domain,
        "temporal_scope": rr.temporal_scope,
        "routing_reasoning": rr.reasoning,
        "temporal_scope_applied": rr.temporal_scope,
    }


def retrieve(state: RAGGraphState, rt: GraphRuntime, *, k: int = 6) -> dict[str, Any]:
    where = build_chroma_where(
        predicted_domain=str(state.get("predicted_domain", "All")),
        domain_filter_override=state.get("domain_filter_override"),
        temporal_scope=str(state.get("temporal_scope", "unspecified")),
    )
    q = state.get("rewritten_question") or state["question"]
    emb = rt.embed_query(q)
    docs = rt.similarity_search(emb, k=k, filters=where)
    return {"documents": [chunk.model_dump(mode="json") for chunk in docs]}


def grade_documents(state: RAGGraphState, rt: GraphRuntime) -> dict[str, Any]:
    docs = state.get("documents") or []
    if not docs:
        attempts = int(state.get("retrieval_attempts") or 0) + 1
        return {"grade": "not_relevant", "retrieval_attempts": attempts}

    ctx = "\n\n".join(f"[{i}] {d.get('text', '')}" for i, d in enumerate(docs[:6]))
    system = (
        "You grade whether retrieved statute excerpts help answer the user's question. "
        'Return JSON: {"relevant": true|false}. '
        "If excerpts are off-topic or wrong act era for the question, use false."
    )
    user = (
        f"Question:\n{state['question']}\n\nRewritten:\n{state.get('rewritten_question', '')}\n\n"
        f"Excerpts:\n{ctx}\n"
    )
    raw = rt.llm_invoke_json(system, user)
    relevant = _parse_grade(raw)

    attempts = int(state.get("retrieval_attempts") or 0)
    if relevant:
        return {"grade": "relevant", "retrieval_attempts": attempts}
    return {"grade": "not_relevant", "retrieval_attempts": attempts + 1}


def generate(state: RAGGraphState, rt: GraphRuntime) -> dict[str, Any]:
    docs = state.get("documents") or []
    temporal = str(state.get("temporal_scope", "unspecified"))

    if temporal == "legacy" and not docs:
        msg = (
            "I cannot answer from the **Income Tax Act, 1961** because that act is not present "
            "in this project’s indexed corpus yet. Add the 1961 PDF to `data/Finance/`, update "
            "`data/manifest.yaml`, and re-run ingestion. I have **not** substituted the Income "
            "Tax Act, 2025 text as a stand-in for 1961 law."
        )
        return {
            "generation": msg,
            "citations": [],
            "grounded": False,
            "act_versions_used": [],
        }

    if not docs:
        msg = (
            "I could not find relevant excerpts in the provided statutes for this question. "
            "Try rephrasing with an act name or a more specific section topic."
        )
        return {"generation": msg, "citations": [], "grounded": False, "act_versions_used": []}

    ctx = []
    for i, d in enumerate(docs[:6], start=1):
        meta = (
            f"{d.get('act_title')} ({d.get('act_version')}) "
            f"pages {d.get('page_start')}-{d.get('page_end')}"
        )
        if d.get("section_label"):
            meta += f", {d.get('section_label')}"
        ctx.append(f"### Source {i}: {meta}\n{d.get('text', '')}")
    context = "\n\n".join(ctx)

    system = (
        "You are a careful legal research assistant. "
        "Answer ONLY using the provided statute excerpts. "
        "Every factual claim must be traceable to the excerpts. "
        "Quote or paraphrase with inline source numbers "
        "like [Source 1]. If excerpts conflict or are insufficient, say so explicitly.\n"
        "This is not legal advice.\n"
        "After the answer, the app will show structured citations separately — "
        "still cite sources inline."
    )
    user = f"Question:\n{state['question']}\n\nStatute excerpts:\n{context}\n"
    answer = rt.llm_invoke_messages(system, user)

    acts = sorted({str(d.get("act_version")) for d in docs if d.get("act_version")})
    cites: list[dict[str, Any]] = []
    for d in docs[:6]:
        cites.append(
            Citation(
                act_title=str(d.get("act_title", "")),
                act_version=str(d.get("act_version", "")),
                section_label=d.get("section_label"),
                pages=f"{d.get('page_start')}-{d.get('page_end')}",
                source_path=str(d.get("source_path", "")),
            ).model_dump(mode="json")
        )

    return {
        "generation": answer.strip(),
        "citations": cites,
        "grounded": True,
        "act_versions_used": acts,
    }


def route_after_grade(state: RAGGraphState) -> str:
    if state.get("grade") == "relevant":
        return "generate"
    if int(state.get("retrieval_attempts") or 0) < 2:
        return "rewrite"
    return "generate"
