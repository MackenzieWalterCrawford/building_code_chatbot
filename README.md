# NYC Building Code RAG

Ask natural-language questions about the **2022 NYC Building Code** and get plain-language answers with exact section citations, confidence scores, and source text.

## Repo layout

```
backend/    FastAPI service + RAG pipeline (ingest → vectorstore → retrieve → generate → api)
frontend/   React app (not yet scaffolded) — talks to the backend over HTTP
```

## Architecture

```
PDF files (backend/data/raw/)
      │
      ▼
 ingest.py  ──────────────────────────────────────────────────
  • pymupdf text extraction                                    │
  • Section-aware chunking (§NNNN.N boundaries)               │
  • Metadata: chapter, section_number, title, edition         │
  • Output: backend/data/processed/*.json                     │
      │                                                        │
      ▼                                                        │
 vectorstore.py                                               │
  • OpenAI text-embedding-3-large                             │
  • Weaviate collection: NYCBuildingCode                      │
  • Idempotent upsert (content-hash based)                    │
      │                                                        │
      ▼                                                        │
 retrieve.py                                                   │
  • Hybrid search: dense vector (α) + BM25 (1-α)             │
  • Metadata filtering by chapter                             │
  • Returns top-k RetrievedChunk objects                      │
      │                                                        │
      ▼                                                        │
 generate.py                                                   │
  • LLM (GPT-4o or Claude) grounded on retrieved text        │
  • Structured output: answer, cited_sections, sufficient     │
  • Confidence heuristic from retrieval scores                │
      │                                                        │
      ▼                                                        │
 api.py  (FastAPI)                                             │
  GET  /health                                                │
  GET  /chapters      → chapter list, for a UI filter         │
  POST /ask           → answer + sources + citations          │
      │                                                        │
      ▼                                                        │
 logger.py                                                     │
  SQLite: logs/interactions.db ──────────────────────────────
```

React (`frontend/`) calls `api.py` over HTTP — see `frontend/README.md` for
the request/response contract.

## Setup

### 1. Prerequisites

- Docker + Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key (required for embeddings + LLM unless using Anthropic)

### 2. Place the PDFs

Chapter PDFs are already in `backend/data/pdfs/`. Symlinks to `backend/data/raw/`
are created by the ingest script. If you want to ingest ALL Building Code chapters,
run this from `backend/`:

```bash
cd backend
for f in data/pdfs/2022BC_Chapter*.pdf; do
  ln -sf "../../$f" data/raw/"$(basename $f)"
done
```

### 3. Configure environment

```bash
cd backend
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for embeddings |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `ANTHROPIC_API_KEY` | — | Required if using Anthropic |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate endpoint |
| `CHUNK_TARGET_TOKENS` | `768` | Target tokens per chunk |
| `RETRIEVAL_ALPHA` | `0.75` | Hybrid search balance |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Allowed origins for the React frontend |

### 4. Install Python dependencies (local dev)

```bash
cd backend
pip install -r requirements.txt
```

### 5. Run the ingestion pipeline

Run from `backend/`:

```bash
# Single chapter first (for testing)
python src/ingest.py --file data/raw/2022BC_Chapter16_StructuralDesignWBwm.pdf

# All Building Code chapters
python src/ingest.py

# Load into Weaviate (Weaviate must be running)
python src/vectorstore.py
```

### 6. One-command Docker run

```bash
docker-compose up --build
```

This starts:
- **Weaviate** on port 8080
- **FastAPI** on port 8000  →  http://localhost:8000/docs

**First run:** after containers start, run ingestion inside the api container:
```bash
docker-compose exec api python src/ingest.py
docker-compose exec api python src/vectorstore.py
```

## Usage

The React frontend isn't scaffolded yet (see `frontend/README.md`). Until then,
interact with the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum floor live load for office occupancies?"}'
```

Or use the built-in Swagger UI at http://localhost:8000/docs.

Example questions:
- *What is the minimum floor live load for office occupancies?*
- *What are the fire resistance requirements for Type I-A construction?*
- *What is the maximum travel distance to an exit in a sprinklered assembly occupancy?*

## Local development

```bash
# Start Weaviate only
docker-compose up weaviate -d

# Run FastAPI locally (from backend/) — PYTHONPATH=src is required because
# api.py imports its sibling modules (generate, retrieve, vectorstore) as
# flat imports; Docker sets this via ENV PYTHONPATH=/app/src.
cd backend
PYTHONPATH=src uvicorn src.api:app --reload --port 8000
```

## Running tests

```bash
cd backend
pytest tests/ -v
```

## Known limitations

- PDFs must be text-readable (not scanned images); the 2022 WBwm PDFs are.
- Section detection depends on the NYC BC numbering format (`NNNN.N`); tables and figures are not structured.
- Confidence scores are heuristic, not calibrated probabilities.
- Does not cover the Electrical Code, Mechanical Code, or other NYC codes (only the Building Code).
- This is a reference tool; always verify with a licensed professional.
