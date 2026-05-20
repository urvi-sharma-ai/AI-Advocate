from __future__ import annotations

from functools import cached_property

from ai_advocate.application.use_cases.ask_question import AskQuestionUseCase
from ai_advocate.application.use_cases.ingest_documents import IngestDocumentsUseCase
from ai_advocate.infrastructure.config import Settings
from ai_advocate.infrastructure.graph.nodes import GraphRuntime
from ai_advocate.infrastructure.graph.rag_graph import build_rag_graph
from ai_advocate.infrastructure.llm.embeddings import OpenAIEmbedderAdapter
from ai_advocate.infrastructure.llm.openai_client import LangChainLLMAdapter
from ai_advocate.infrastructure.vector_store.chroma_store import ChromaVectorStore


class Container:
    """Composition root. Keeps Streamlit and CLIs out of infrastructure details."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def chroma(self) -> ChromaVectorStore:
        return ChromaVectorStore(
            persist_directory=self.settings.chroma_persist_dir,
            collection_name=self.settings.chroma_collection,
        )

    @cached_property
    def embedder(self) -> OpenAIEmbedderAdapter:
        api_key = self.settings.require_api_key()
        return OpenAIEmbedderAdapter(
            api_key=api_key,
            model=self.settings.openai_embedding_model,
            batch_size=self.settings.embedding_batch_size,
        )

    @cached_property
    def llm(self) -> LangChainLLMAdapter:
        api_key = self.settings.require_api_key()
        return LangChainLLMAdapter(api_key=api_key, model=self.settings.openai_chat_model)

    @cached_property
    def _graph_runtime(self) -> GraphRuntime:
        return GraphRuntime(
            llm_invoke_json=self.llm.invoke_json,
            llm_invoke_messages=self.llm.invoke_messages,
            embed_query=self.embedder.embed_query,
            similarity_search=self.chroma.similarity_search,
        )

    @cached_property
    def rag_graph(self):
        return build_rag_graph(self._graph_runtime)

    def ingest_use_case(self) -> IngestDocumentsUseCase:
        return IngestDocumentsUseCase(
            data_dir=self.settings.data_dir,
            manifest_path=self.settings.resolved_manifest(),
            embedder=self.embedder,
            vector_store=self.chroma,
        )

    def ask_use_case(self) -> AskQuestionUseCase:
        return AskQuestionUseCase(graph=self.rag_graph)
