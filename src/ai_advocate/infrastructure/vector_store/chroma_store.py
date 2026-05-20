from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_advocate.infrastructure.sqlite_compat import ensure_sqlite_fts5

ensure_sqlite_fts5()

import chromadb  # noqa: E402 — must load after sqlite3 FTS5 shim

from ai_advocate.domain.models import Chunk  # noqa: E402


def chunk_metadata_dict(chunk: Chunk) -> dict[str, str]:
    """Chroma's SQLite-backed filtering behaves reliably when primitives/strings dominate."""
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "domain": chunk.domain,
        "act_title": chunk.act_title,
        "act_version": chunk.act_version,
        "source_path": chunk.source_path,
        "chunk_index": str(chunk.chunk_index),
        "page_start": str(chunk.page_start),
        "page_end": str(chunk.page_end),
        "section_number": chunk.section_number or "",
        "section_title": chunk.section_title or "",
        "section_label": chunk.section_label or "",
        "subsection_label": chunk.subsection_label or "",
    }


class ChromaVectorStore:
    """Persistence adapter for statute chunks + embeddings."""

    def __init__(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [chunk_metadata_dict(c) for c in chunks]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 6,
        filters: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        where = _normalize_where(filters)
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)
        metas = (result.get("metadatas") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        out: list[Chunk] = []
        for md, text in zip(metas, docs, strict=False):
            if md is None or text is None:
                continue
            out.append(_chunk_from_metadata(md, text))
        return out

    def count(self) -> int:
        return int(self._collection.count())


def _normalize_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    # Allow callers to pass Chroma-native dicts directly.
    return filters


def _chunk_from_metadata(md: dict[str, Any], document: str) -> Chunk:
    sec_no = md.get("section_number") or None
    if sec_no == "":
        sec_no = None
    sec_title = md.get("section_title") or None
    if sec_title == "":
        sec_title = None
    sec_label = md.get("section_label") or None
    if sec_label == "":
        sec_label = None

    return Chunk(
        id=str(md.get("chunk_id")),
        text=document,
        document_id=str(md["document_id"]),
        domain=md["domain"],  # type: ignore[arg-type]
        act_title=str(md["act_title"]),
        act_version=str(md["act_version"]),
        source_path=str(md["source_path"]),
        chunk_index=int(md.get("chunk_index") or 0),
        page_start=int(md.get("page_start") or 1),
        page_end=int(md.get("page_end") or 1),
        section_number=str(sec_no) if sec_no else None,
        section_title=str(sec_title) if sec_title else None,
        section_label=str(sec_label) if sec_label else None,
        subsection_label=str(md.get("subsection_label") or "") or None,
    )
