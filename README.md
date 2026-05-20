# AI Advocate

A learning-focused legal RAG project over a small corpus of Indian statutes in `data/`, using:

- LangGraph for an agentic RAG workflow (rewrite → retrieve → grade → generate)
- ChromaDB for local vector storage
- Streamlit for a simple chat UI

This is not legal advice.

## Setup

```bash
uv sync
cp .env.example .env
# add OPENAI_API_KEY=...
```

## Ingest the PDFs (build the vector DB)

```bash
uv sync
uv run ai-advocate-ingest
```

### Troubleshooting: `sqlite3.OperationalError: no such module: fts5`

ChromaDB needs SQLite **FTS5**. The Python from `uv` often ships without it on macOS. This project depends on `pysqlite3` and swaps it in automatically before Chroma starts.

If ingest still fails after a previous crash, delete the partial DB and retry:

```bash
rm -rf .chroma
uv run ai-advocate-ingest
```

This reads `data/manifest.yaml`, loads PDFs in `data/`, chunks them with **section-context prefixes** (critical for the Income Tax Act 2025 mega-sections), embeds the chunks, and stores them in Chroma.

## Run the app

```bash
uv run streamlit run src/ai_advocate/presentation/streamlit_app.py
```

## Project layout (clean architecture)

- `src/ai_advocate/domain/`: core models (documents, chunks, citations)
- `src/ai_advocate/application/`: ports + use cases
- `src/ai_advocate/infrastructure/`: adapters (PDF loader/chunker, Chroma, OpenAI, LangGraph)
- `src/ai_advocate/presentation/`: Streamlit UI

If ingest fails with `max_tokens_per_request`, lower `EMBEDDING_BATCH_SIZE` in `.env` (default 100 chunks per API call).

## Notes

- The manifest uses official act names and an `act_version` field (e.g. `IT_2025`).
- Legacy Income Tax Act 1961 is **not in the corpus yet**. If a user asks a legacy question, the agent will refuse instead of hallucinating from `IT_2025`.
