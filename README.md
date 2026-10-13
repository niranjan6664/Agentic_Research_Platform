# Agentic Research & Decision Intelligence Platform

An AI research assistant that retrieves evidence from your documents (and, from Milestone 2, permitted
web sources), compares alternatives, and produces transparent, citation-backed recommendations.

**Status: Milestone 1 of 4 complete** (ingestion, vector indexing, RAG retrieval, citations).

| Milestone | Scope | Status |
|---|---|---|
| 1 | Ingestion, chunking, embeddings, FAISS index, retrieval, citations, REST API | ✅ done |
| 2 | LangGraph agent workflow, tools, LLM providers | ⏳ |
| 3 | Decision comparison, evaluation harness, extended tests | ⏳ |
| 4 | Streamlit UI, deployment, demo scenario | ⏳ |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env              # placeholders only; never commit .env
```
The first run with `EMBEDDING_BACKEND=sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB, free).
For a fully offline smoke test set `EMBEDDING_BACKEND=hashing` (lexical only, **not semantic**; for tests/demos).

## Run
```bash
uvicorn app.api.main:app --reload     # API docs at http://localhost:8000/docs
pytest                                 # 37 tests, no network or model download needed
```

## API (Milestone 1)
| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload PDF/TXT/MD (`file`, optional comma-separated `tags`) |
| GET / DELETE | `/documents`, `/documents/{id}` | List / remove indexed documents |
| POST | `/search` | Semantic search with filters (`doc_ids`, `filenames`, `tags`, `page_min/max`) |
| GET | `/health`, `/config` | Status; config with secrets reduced to booleans |

Every search hit is a **citation** (`S1`, `S2`, ...) carrying filename, page, score and the exact retrieved text.
If nothing clears `MIN_SCORE`, the response sets `insufficient_evidence: true` instead of guessing.

## Design notes and limits
- Re-uploading identical content is idempotent (content-hash document IDs).
- The index records which embedder built it and refuses to load with a different one.
- Exact FAISS search (`IndexFlatIP`); fine to ~100k chunks. Metadata filters are applied exactly, before top-k.
- Scanned PDFs (no text layer) and encrypted PDFs are rejected with a clear message; OCR is not included.
- `MIN_SCORE` is embedder-specific and should be tuned on the evaluation set (Milestone 3).

## Costs
Everything in Milestone 1 is free and local. Paid services appear only if you opt in later
(OpenAI/Anthropic LLMs, Tavily search).
