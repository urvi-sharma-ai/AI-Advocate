from __future__ import annotations

from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedderAdapter:
    """Implements ``EmbedderPort`` using OpenAI embeddings."""

    def __init__(self, *, api_key: str, model: str, batch_size: int = 100) -> None:
        self._emb = OpenAIEmbeddings(api_key=api_key, model=model)
        self._batch_size = max(1, batch_size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(self._emb.embed_documents(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._emb.embed_query(text)
