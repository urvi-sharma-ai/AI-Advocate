from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ai_advocate.domain.models import Chunk, LegalDocument


@runtime_checkable
class EmbedderPort(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return embedding vector for a single query."""
        ...


@runtime_checkable
class VectorStorePort(Protocol):
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunks with embeddings (idempotent on chunk id)."""
        ...

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 6,
        filters: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Return nearest chunks respecting optional metadata filters."""
        ...

    def count(self) -> int:
        """Number of vectors in the collection."""
        ...


@runtime_checkable
class LLMPort(Protocol):
    def invoke_json(self, system: str, user: str) -> str:
        """Return model output expected to be parseable JSON."""
        ...

    def invoke_messages(self, system: str, user: str) -> str:
        """Return free-form text."""
        ...


@runtime_checkable
class DocumentLoaderPort(Protocol):
    def load_pdf(self, path: str, doc: LegalDocument) -> list[tuple[int, str]]:
        """Return (1-based page number, text) for each page."""
        ...
