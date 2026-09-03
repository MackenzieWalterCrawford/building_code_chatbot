# NYC Building Code RAG

Ask natural-language questions about the **2022 NYC Building Code** and get plain-language answers with exact section citations, confidence scores, and source text.

## Architecture

```
PDF files (data/raw/)
      │
      ▼
 ingest.py  ──────────────────────────────────────────────────
  • pymupdf text extraction                                    │
  • Section-aware chunking (§NNNN.N boundaries)               │
  • Metadata: chapter, section_number, title, edition         │
  • Output: data/processed/*.json                             │
      │                                                        │
      ▼                                                        │
 vectorstore.py                                               │
  • OpenAI text-embedding-3-large                             │
  • Weaviate collection: NYCBuildingCode                      │
  • Idempotent upsert                                         │
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
  • Extracts §citations from answer                           │
  • Confidence heuristic from retrieval scores               │
      │                                                        │
      ▼                                                        │
 api.py  (FastAPI)     app.py  (Streamlit)                    │
  POST /ask              UI: question → answer + sources      │
      │                                                        │
      ▼                                                        │
 logger.py                                                     │
  SQLite: logs/interactions.db ──────────────────────────────
```

## Setup

### 1. Prerequisites

- Docker + Docker Compose
- Python 3.11+ (for local development)
- OpenAI API key (required for embeddings + LLM unless using Anthropic)

### 2. Place the PDFs

Chapter PDFs are already in `data/pdfs/`. Symlinks to `data/raw/` are created by the ingest script. If you want to ingest ALL Building Code chapters:

```bash
for f in data/pdfs/2022BC_Chapter*.pdf; do
  ln -sf "../../$f" data/raw/"$(basename $f)"
done
```

### 3. Configure environment

```bash
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

### 4. Install Python dependencies (local dev)

```bash
pip install -r requirements.txt
```

### 5. Run the ingestion pipeline

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
- **Streamlit** on port 8501  →  http://localhost:8501

**First run:** after containers start, run ingestion inside the api container:
```bash
docker-compose exec api python src/ingest.py
docker-compose exec api python src/vectorstore.py
```

## Usage

Open http://localhost:8501, type a question, and press **Ask**.

Example questions:
- *What is the minimum floor live load for office occupancies?*
- *What are the fire resistance requirements for Type I-A construction?*
- *What is the maximum travel distance to an exit in a sprinklered assembly occupancy?*

## Local development

```bash
# Start Weaviate only
docker-compose up weaviate -d

# Run FastAPI locally
uvicorn src.api:app --reload --port 8000

# Run Streamlit locally
streamlit run src/app.py
```

## Running tests

```bash
pytest tests/ -v
```

## Known limitations

- PDFs must be text-readable (not scanned images); the 2022 WBwm PDFs are.
- Section detection depends on the NYC BC numbering format (`NNNN.N`); tables and figures are not structured.
- Confidence scores are heuristic, not calibrated probabilities.
- Does not cover the Electrical Code, Mechanical Code, or other NYC codes (only the Building Code).
- This is a reference tool; always verify with a licensed professional.
