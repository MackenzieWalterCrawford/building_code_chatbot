# Backend

FastAPI + RAG pipeline for the NYC Building Code chatbot. Six scripts, each
one phase of the pipeline, run in this order:

```
ingest.py  →  vectorstore.py  →  retrieve.py  →  generate.py  →  api.py  →  logger.py
(offline, run once)  (offline)      (online, per-request)                 (per-request)
```

`ingest.py` and `vectorstore.py` are run manually (or via Docker exec) to
build the knowledge base. `api.py` is the long-running service; it calls
`retrieve.py` and `generate.py` on every request and `logger.py` records the
result.

## End-to-end flow

```mermaid
flowchart TD
    subgraph offline["Offline: build the knowledge base"]
        PDF["Chapter PDFs\nbackend/data/raw/*.pdf"] --> ING["ingest.py\npymupdf extraction +\nsection-aware chunking"]
        ING --> JSON["backend/data/processed/*.json\n(Chunk records)"]
        JSON --> VEC["vectorstore.py\nembed + upsert"]
        VEC --> WEAV[("Weaviate\nNYCBuildingCode collection")]
    end

    subgraph online["Online: answer a question"]
        Q["POST /ask\n{question, top_k, alpha, chapter_filter}"] --> API["api.py"]
        API --> RET["retrieve.py\nhybrid search (vector + BM25)"]
        RET -- "query embedding" --> WEAV
        WEAV -- "top-k RetrievedChunk" --> RET
        RET --> GEN["generate.py\nLLM call, grounded on chunks"]
        GEN --> STRUCT["StructuredAnswer\nanswer, cited_sections, sufficient"]
        STRUCT --> CONF["confidence heuristic\n(retrieval scores + citation count)"]
        CONF --> RESP["AskResponse\nanswer, sources, confidence, insufficient"]
        RESP --> API
        API --> LOG["logger.py\nSQLite: logs/interactions.db"]
        API --> CLIENT["Response → caller"]
    end
```

## 1. `ingest.py` — PDF → section-aware chunks

Turns a raw Building Code chapter PDF into a list of `Chunk` records, one
per code section (e.g. `§1607.1`), written to
`data/processed/<stem>.json`.

```mermaid
flowchart TD
    A["extract_pages()\npymupdf, reading order"] --> B["pages_to_text()\nstrip watermarks, page numbers,\nheader/footer lines"]
    B --> C["detect_chapter()\nfind 'CHAPTER N' + title"]
    B --> D["extract_sections()\nregex match NNNN.N headings\n→ list of _SectionSpan"]
    C --> E
    D --> E["spans_to_chunks()"]
    E --> F{"span ≥ MIN_TOKENS?"}
    F -- yes --> G["emit as chunk\n(split if > MAX_TOKENS)"]
    F -- no --> H{"descendant of\nprevious chunk?"}
    H -- yes --> I["merge backward\ninto previous chunk"]
    H -- no --> J{"next span is its\ndescendant?"}
    J -- yes --> K["hold as forward prefix,\nmerge into next span"]
    J -- no --> G
    G --> L["Chunk{chunk_id, chapter,\nsection_number, section_title,\ntext, pages, token_count, ...}"]
    I --> L
    K --> L
    L --> M["data/processed/<stem>.json"]
```

Splitting an oversized section falls through progressively finer
boundaries so nothing silently exceeds `CHUNK_MAX_TOKENS`:

```
_split_on_paragraphs()  →  _split_on_sentences()  →  _hard_split_tokens()
   (\n\n boundaries)        (sentence boundaries)      (raw token window,
                                                         e.g. a dense table)
```

Run it:

```bash
python src/ingest.py --file data/raw/2022BC_Chapter16_StructuralDesignWBwm.pdf
python src/ingest.py                 # all data/raw/2022BC_Chapter*.pdf
```

## 2. `vectorstore.py` — embed + upsert into Weaviate

Reads the processed JSON, embeds chunk text with OpenAI, and upserts into
the `NYCBuildingCode` Weaviate collection.

```mermaid
flowchart TD
    A["data/processed/*.json"] --> B["ensure_collection()\ncreate schema if missing,\nbackfill content_hash if needed"]
    B --> C["load_chunks()"]
    C --> D["hash each chunk's text\n(sha256)"]
    D --> E{"chunk_id exists in\nstore with same hash?"}
    E -- yes --> F["skip\n(unchanged)"]
    E -- no --> G["embed_texts()\nOpenAI text-embedding-3-large,\nbatched, retried w/ backoff"]
    G --> H["batch upsert\n(uuid5 of chunk_id)"]
    H --> I[("Weaviate: NYCBuildingCode")]
```

Idempotency is **content-based**, not just ID-based: a chunk is only
skipped if both its `chunk_id` *and* its content hash already match what's
in the store. Editing the chunking logic in `ingest.py` and re-running
produces the same `chunk_id`s but different text — which this correctly
detects as "needs re-embedding" rather than leaving a stale vector in
place.

Run it (Weaviate must be reachable):

```bash
python src/vectorstore.py
```

## 3. `retrieve.py` — hybrid search

Given a question, returns the top-k most relevant chunks by blending dense
vector similarity with BM25 keyword search.

```mermaid
flowchart LR
    Q["query string"] --> EMB["embed_texts([query])\n(same model as ingestion)"]
    Q --> BM25["BM25 keyword score"]
    EMB --> HYBRID["Weaviate hybrid search\nscore = α·vector + (1-α)·BM25\n(RELATIVE_SCORE fusion)"]
    BM25 --> HYBRID
    FILTER["optional chapter_filter"] --> HYBRID
    HYBRID --> TOPK["top_k RetrievedChunk\n(section, text, pages, score)"]
```

`alpha` tunes the blend: `1.0` = pure vector similarity, `0.0` = pure
keyword match. Default `0.75` (config via `RETRIEVAL_ALPHA`).

## 4. `generate.py` — grounded answer generation

Builds a context block from the retrieved chunks, asks an LLM (OpenAI or
Anthropic — switchable via `LLM_PROVIDER`) for a **structured** answer, and
derives a confidence score.

```mermaid
flowchart TD
    CH["retrieved chunks"] --> CTX["_build_context()\n'§NNNN.N — title\\ntext' per chunk\n+ the question"]
    CTX --> LLM["_llm_call()"]
    LLM -->|LLM_PROVIDER=openai| OAI["OpenAI chat.completions.parse()"]
    LLM -->|LLM_PROVIDER=anthropic| ANT["Anthropic messages.parse()"]
    OAI --> SA["StructuredAnswer\n{answer, cited_sections, sufficient}"]
    ANT --> SA
    SA --> CONF["estimate_confidence()\navg(top-3 scores)\n+0.1 if ≥2 sections cited\n-0.3 if not sufficient"]
    CONF --> RES["GenerationResult\n{answer, cited_sections,\nconfidence, insufficient}"]
```

Both providers are forced into the same `StructuredAnswer` schema
(`answer`, `cited_sections`, `sufficient`), so the system never has to
regex citations or string-match an "insufficient" phrase out of free-form
prose — and if the LLM decides the retrieved sections are inadequate, the
answer is replaced with a fixed `INSUFFICIENT_MESSAGE` and confidence takes
a `-0.3` penalty.

## 5. `api.py` — FastAPI service

Wires the above into a long-running HTTP service. One shared Weaviate
client, created at startup (`lifespan`) and closed at shutdown.

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /chapters` | Distinct chapters in the store, for a UI filter dropdown |
| `POST /ask` | `{question, top_k?, alpha?, chapter_filter?}` → `{answer, sources, cited_sections, confidence, insufficient}` |

`POST /ask` request lifecycle:

```mermaid
sequenceDiagram
    participant C as Client
    participant A as api.py
    participant R as retrieve.py
    participant W as Weaviate
    participant G as generate.py
    participant L as logger.py

    C->>A: POST /ask {question, top_k, alpha, chapter_filter}
    A->>R: retrieve(question, top_k, alpha, chapter_filter)
    R->>W: hybrid query (vector + BM25)
    W-->>R: top-k chunks + scores
    R-->>A: list[RetrievedChunk]
    A->>G: generate(question, chunks)
    G-->>A: GenerationResult (answer, citations, confidence)
    A->>L: log_interaction(question, chunks, answer, ...)
    A-->>C: AskResponse {answer, sources, cited_sections, confidence, insufficient}
```

## 6. `logger.py` — interaction logging

Every `/ask` call is appended as one row to `logs/interactions.db`
(SQLite), capturing the question, retrieved chunks (as JSON), the answer,
cited sections, confidence, and latency. Logging failures are caught and
warned, not raised — a logging hiccup should never fail the user's request.

## Configuration

All scripts read from `backend/.env` (see `.env.example`). The variables
that affect this pipeline:

| Variable | Used by | Default |
|---|---|---|
| `OPENAI_API_KEY` | `vectorstore.py`, `generate.py` | — |
| `OPENAI_EMBEDDING_MODEL` | `vectorstore.py`, `retrieve.py` | `text-embedding-3-large` |
| `LLM_PROVIDER` | `generate.py` | `openai` |
| `OPENAI_LLM_MODEL` / `ANTHROPIC_MODEL` | `generate.py` | `gpt-4o` / `claude-sonnet-4-6` |
| `WEAVIATE_URL`, `WEAVIATE_API_KEY` | `vectorstore.py`, `retrieve.py`, `api.py` | `http://localhost:8080` |
| `CHUNK_TARGET_TOKENS` / `MAX` / `MIN` | `ingest.py` | `768` / `1024` / `100` |
| `RETRIEVAL_TOP_K`, `RETRIEVAL_ALPHA` | `retrieve.py` | `8`, `0.75` |
| `CORS_ORIGINS` | `api.py` | `http://localhost:5173,http://localhost:3000` |
| `LOG_DIR` | `logger.py` | `./logs` |

## Running tests

```bash
cd backend
pytest tests/ -v
```

Currently covers `ingest.py` (`tests/test_ingest.py`); `vectorstore.py`,
`retrieve.py`, `generate.py`, and `api.py` have no dedicated tests yet.
