from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_advocate.application.ports import EmbedderPort, VectorStorePort
from ai_advocate.domain.models import Chunk
from ai_advocate.infrastructure.pdf.chunker import chunk_document
from ai_advocate.infrastructure.pdf.loader import load_pdf_pages
from ai_advocate.infrastructure.pdf.manifest import load_manifest_yaml


@dataclass(frozen=True)
class IngestResult:
    documents_indexed: int
    chunks_indexed: int


class IngestDocumentsUseCase:
    def __init__(
        self,
        *,
        data_dir: Path,
        manifest_path: Path,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._data_dir = data_dir
        self._manifest_path = manifest_path
        self._embedder = embedder
        self._vector_store = vector_store

    def run(self) -> IngestResult:
        docs = load_manifest_yaml(self._manifest_path, self._data_dir)
        all_chunks: list[Chunk] = []

        for doc in docs:
            pages = load_pdf_pages(doc)
            chunks = chunk_document(doc, pages)
            all_chunks.extend(chunks)

        embeddings = self._embedder.embed([c.text for c in all_chunks])
        self._vector_store.upsert_chunks(all_chunks, embeddings)

        return IngestResult(documents_indexed=len(docs), chunks_indexed=len(all_chunks))
