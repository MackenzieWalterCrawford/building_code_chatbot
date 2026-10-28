# NYC Building Code RAG

Ask natural-language questions about the **2022 NYC Building Code** and get plain-language answers with exact section citations, confidence scores, and source text.

<img width="835" height="656" alt="image" src="https://github.com/user-attachments/assets/b7f61f5d-7ff3-4735-b0da-fbc02dbbb87d" />

<img width="1915" height="967" alt="image" src="https://github.com/user-attachments/assets/2da3061b-d6a7-44ee-a149-0b3b839efa8e" />



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
  • pymupdf text extraction                                    
  • Section-aware chunking (§NNNN.N boundaries)               
  • Metadata: chapter, section_number, title, edition         
  • Output: backend/data/processed/*.json                     
      │                                                        
      ▼                                                        
 vectorstore.py                                               
  • OpenAI text-embedding-3-large                             
  • Weaviate collection: NYCBuildingCode                      
  • Idempotent upsert (content-hash based)                    
      │                                                        
      ▼                                                        
 retrieve.py                                                   
  • Hybrid search: dense vector (α) + BM25 (1-α)             
  • Metadata filtering by chapter                             
  • Returns top-k RetrievedChunk objects                      
      │                                                        
      ▼                                                        
 generate.py                                                   
  • LLM (GPT-4o or Claude) grounded on retrieved text        
  • Structured output: answer, cited_sections, sufficient     
  • Confidence heuristic from retrieval scores                
      │                                                        
      ▼                                                        
 api.py  (FastAPI)                                             
  GET  /health                                                
  GET  /chapters      → chapter list, for a UI filter         
  POST /ask           → answer + sources + citations          
      │                                                        
      ▼                                                        
 logger.py                                                     
  SQLite: logs/interactions.db ──────────────────────────────
```

React (`frontend/`) calls `api.py` over HTTP — see `frontend/README.md` for
the request/response contract.

## Chunking strategy

The Building Code is a legal document with its own numbering, so chunks follow
that structure rather than a fixed window size. This is **structure-aware
(hierarchical) chunking** with size-based fallbacks (`backend/src/ingest.py`):

1. **One chunk per numbered section.** Section boundaries are detected from the
   code's numbering (`CHAPTER N`, `NNNN`, `NNNN.N`, `NNNN.N.N`, `NNNN.N.N.N`),
   so a provision is never cut in half and every chunk can be cited by section number.
2. **Merge small sections.** Sections under `CHUNK_MIN_TOKENS` (100) are merged
   into their parent section to avoid context-poor fragments.
3. **Split oversized sections.** Sections over `CHUNK_MAX_TOKENS` (1024) are split,
   aiming for `CHUNK_TARGET_TOKENS` (768) per piece: first on paragraph
   boundaries, then on sentences, and as a last resort by token count.
4. **Metadata on every chunk.** `code_name`, `edition`, `source_file`, `chapter`,
   `chapter_title`, `section_number`, `section_title`, `parent_section`,
   `page_start`, `page_end`, `token_count`, `split_index`. This powers citations
   and the chapter filter.

Trade-offs: chunk sizes vary, there is no overlap between chunks, and
cross-references ("see Section 1607.5") are not followed at retrieval time.
Text is extracted per page with PyMuPDF; tables and figures are not structured.

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

## Evaluation

`backend/eval/` contains a labeled question set (36 questions: 31 answerable, 5 that should be
refused) and a harness that measures retrieval and answer quality. See
[`backend/eval/README.md`](backend/eval/README.md) for the metric definitions and caveats.

```bash
cd backend
python eval/run_eval.py --retrieval                  # dense vs BM25 vs hybrid sweep
python eval/run_eval.py --answers --judge            # citations, abstention, LLM-as-judge
```

Results (2026-10-12; `text-embedding-3-large`, GPT-4o as generator and judge):

**Retrieval** — share of questions where a gold section appears in the top *k* results

| alpha | mode | hit@1 | hit@3 | hit@5 | hit@8 | MRR |
|---|---|---|---|---|---|---|
| 0.0 | BM25 only | 0.52 | 0.74 | 0.81 | 0.87 | 0.641 |
| 0.25 | hybrid | 0.71 | 0.84 | 0.94 | 0.94 | 0.781 |
| 0.5 | hybrid | 0.74 | 0.90 | 0.94 | 1.00 | 0.841 |
| 0.75 | hybrid (default) | 0.87 | 1.00 | 1.00 | 1.00 | 0.930 |
| 1.0 | dense only | 0.87 | 1.00 | 1.00 | 1.00 | 0.935 |

**Answers** (alpha 0.75, top_k 8)

| metric | value |
|---|---|
| answered when answerable | 31/31 |
| refused when out of scope | 5/5 |
| citation recall / precision | 0.98 / 0.87 |
| citations grounded in retrieved text | 0.99 |
| judged faithful | 0.97 |
| judged correct | 0.84 |
| median latency (retrieve + generate) | 2.5 s |

Read these with the sample size in mind: 31 answerable questions separates BM25 from dense/hybrid
clearly, but dense (α=1.0) and hybrid (α=0.75) are indistinguishable here, so this set does not
show hybrid retrieval outperforming dense. The `correct` score is judge-dependent and varied
between runs in which questions failed; most misses were the judge penalizing omitted secondary
details from the reference answer. The gold set was drafted with LLM assistance and should be
spot-checked.

## Known limitations

- PDFs must be text-readable (not scanned images); the 2022 WBwm PDFs are.
- Section detection depends on the NYC BC numbering format (`NNNN.N`); tables and figures are not structured.
- Confidence scores are heuristic, not calibrated probabilities.
- Does not cover the Electrical Code, Mechanical Code, or other NYC codes (only the Building Code).
- This is a reference tool; always verify with a licensed professional.
