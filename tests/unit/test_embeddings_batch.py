from __future__ import annotations

from unittest.mock import patch

from ai_advocate.infrastructure.llm.embeddings import OpenAIEmbedderAdapter


def test_embed_batches_respect_batch_size() -> None:
    calls: list[int] = []

    def fake_embed_documents(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return [[0.0] * 4 for _ in texts]

    adapter = OpenAIEmbedderAdapter(api_key="x", model="text-embedding-3-small", batch_size=10)
    with patch.object(adapter, "_emb") as emb:
        emb.embed_documents.side_effect = fake_embed_documents
        result = adapter.embed(["x"] * 25)
        assert len(result) == 25
        assert calls == [10, 10, 5]
