# Learning notes: AI Advocate

This repo is intentionally structured to teach you *how* RAG systems are built, not just to ship a demo.

## 1) Follow one question end-to-end

User types a question in Streamlit: `src/ai_advocate/presentation/streamlit_app.py`.

That calls the application layer:

- `AskQuestionUseCase` in `src/ai_advocate/application/use_cases/ask_question.py`

The use case invokes the LangGraph workflow:

- Graph builder: `src/ai_advocate/infrastructure/graph/rag_graph.py`
- Nodes: `src/ai_advocate/infrastructure/graph/nodes.py`

Retrieval uses Chroma:

- Adapter: `src/ai_advocate/infrastructure/vector_store/chroma_store.py`

## 2) Why “section-context chunking” matters (Income Tax Act 2025)

Legal statutes often have long sections. The **Income Tax Act, 2025** consolidates topics into fewer, larger sections.

If you chunk purely by length, the model may see an isolated paragraph and lose the fact that it came from *Section 393*.

In this project, every chunk’s text is prefixed like:

```text
[S393] Deduction of tax at source
---
{chunk_body}
```

Implementation:

- Chunker: `src/ai_advocate/infrastructure/pdf/chunker.py`
- Test: `tests/unit/test_chunker_section_context.py`

## 3) Automated routing: Finance vs Land

Instead of relying only on a UI domain filter, the `rewrite_query` node predicts the domain and then the retrieve node applies a Chroma `where` filter.

Implementation:

- Routing prompt + JSON parsing: `rewrite_query` in `src/ai_advocate/infrastructure/graph/nodes.py`
- Filter composition: `src/ai_advocate/infrastructure/graph/filters.py`

Try asking:

- “Registration of a flat” (should route to Land)
- “TDS on salary 2026” (should route to Finance)

## 4) Temporal routing: 2025 vs legacy Income Tax

The manifest assigns an `act_version` and an `effective_date`.

In v1, we only have `IT_2025` in the corpus. If you ask a clearly legacy question (e.g. “AY 2018”), the agent should refuse rather than silently using `IT_2025`.

To extend this:

1. Add the legacy PDF to `data/Finance/Income_Tax_Act_1961.pdf`
2. Uncomment the legacy entry in `data/manifest.yaml`
3. Re-run ingestion

## Exercises

1. Add a new statute PDF + manifest entry.
2. Tune `_CHAR_SOFT_LIMIT` in the chunker and observe retrieval quality.
3. Add an evaluation set: a few questions with expected sections.
