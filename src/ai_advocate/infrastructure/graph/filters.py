from __future__ import annotations

from typing import Any


def build_chroma_where(
    *,
    predicted_domain: str,
    domain_filter_override: str | None,
    temporal_scope: str,
) -> dict[str, Any] | None:
    """
    Compose Chroma ``where`` from routing state.

    Temporal policy:
    - legacy -> only Income Tax Act 1961 chunks (corpus may be empty)
    - current -> exclude IT_1961 chunks (prefer IT_2025 and other evergreen acts)
    - unspecified -> no act_version filtering
    """
    clauses: list[dict[str, Any]] = []

    domain = None
    override = (domain_filter_override or "").strip()
    if override in ("Finance", "Land"):
        domain = override
    elif predicted_domain in ("Finance", "Land"):
        domain = predicted_domain
    if domain in ("Finance", "Land"):
        clauses.append({"domain": domain})

    if temporal_scope == "legacy":
        clauses.append({"act_version": "IT_1961"})
    elif temporal_scope == "current":
        clauses.append({"act_version": {"$ne": "IT_1961"}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
