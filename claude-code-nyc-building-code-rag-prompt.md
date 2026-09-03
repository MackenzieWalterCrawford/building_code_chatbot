# Claude Code Task: NYC Building Code RAG Application

## Objective
Build an end-to-end Retrieval-Augmented Generation (RAG) application that lets architects ask natural-language questions about the **NYC Building Code (2022 edition)** and get accurate, plain-language answers **with the exact code section cited** for every claim. The LLM must ground answers only in retrieved code text — never invent rules.

## Non-negotiable requirements
- Accept natural-language questions.
- Retrieve relevant context from a vector database (hybrid: vector + BM25).
- Generate answers via an LLM API, grounded ONLY in retrieved chunks.
- **Every answer displays the source section number(s)** (e.g., §1607.1) and a confidence signal.
- If no relevant section is found, say so — do NOT guess.
- Display a disclaimer: this is a reference tool, not a substitute for a NY-licensed professional.
- Log every interaction (question, retrieved chunks, answer) for later analysis.
- Deployable with a public URL + comprehensive README.

## Tech stack
- **Language:** Python 3.11+
- **Vector DB:** Weaviate (local via Docker for dev; hybrid search + metadata filtering)
- **Embeddings:** OpenAI `text-embedding-3-large` (make the provider swappable)
- **LLM:** OpenAI or Anthropic API (make it configurable via env var)
- **Backend:** FastAPI
- **Frontend:** React
- **Config:** `.env` for all keys/secrets; never hard-code

## Data source
NYC Construction Codes are published as PDFs by the NYC Dept. of Buildings:
https://www.nyc.gov/site/buildings/codes/nyc-code.page
Start with ONLY the 2022 **Building Code** PDF. Assume the PDF is placed in `./data/raw/`. Do not attempt to scrape or download automatically — read from the local file.

## Build in this order (each phase must work before the next)

### Phase 1 — Ingestion & section-aware chunking (MOST IMPORTANT)
- Parse the Building Code PDF into structured records preserving hierarchy: `chapter → section → subsection`.
- **Chunk by code section, NOT by fixed character count.** Keep each numbered section intact so a retrieved chunk equals one citable unit.
- Attach metadata to every chunk: `{ code_name, chapter, section_number, section_title, edition }`.
- Output intermediate structured JSON to `./data/processed/` so parsing can be inspected independently.
- Make chunk size configurable; support merging very short subsections and splitting oversized ones (target 512–1024 tokens, test both).

### Phase 2 — Vector store
- Define a Weaviate schema/collection with the metadata fields above and a text field.
- Write an idempotent loader that embeds chunks and upserts them.

### Phase 3 — Retrieval
- Implement **hybrid search** (dense vector + BM25) returning top-k chunks WITH their metadata.
- Support metadata filtering (e.g., restrict to a chapter).
- Expose an `alpha`/weighting param to tune lexical vs semantic balance.

### Phase 4 — Generation
- Prompt the LLM to answer ONLY from retrieved chunks, in plain language, and to cite the exact `section_number`(s).
- If retrieved context is insufficient, respond that the answer isn't in the retrieved code sections.
- Return: answer text, cited section numbers, the raw retrieved chunks, and a confidence score.

### Phase 5 — API + UI
- FastAPI endpoint: `POST /ask` → `{ question }` returns `{ answer, sources[], confidence }`.
- Streamlit UI: question box → answer + expandable cited sections + confidence + disclaimer.

### Phase 6 — Ops
- Log every interaction to a local file or SQLite (`./logs/`).
- Dockerfile + docker-compose (app + Weaviate) for a one-command local run.
- README: setup, env vars, how to add the PDF, run instructions, architecture diagram, and known limitations.

## Project structure
```
nyc-code-rag/
├── data/{raw,processed}/
├── src/
│   ├── ingest.py        # Phase 1
│   ├── vectorstore.py   # Phase 2
│   ├── retrieve.py      # Phase 3
│   ├── generate.py      # Phase 4
│   ├── api.py           # Phase 5 (FastAPI)
│   └── app.py           # Phase 5 (Streamlit)
├── logs/
├── tests/
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Working style
- Build and verify **one phase at a time**; show me the output of each phase before moving on.
- **Prove the full pipeline on a single chapter first**, then scale to the whole Building Code.
- Write small unit tests for the parser and retriever.
- Keep code clean and commented. Ask before adding any dependency not listed above.

## Definition of done
End-to-end: I place the Building Code PDF in `data/raw/`, run docker-compose, open the Streamlit UI, ask "What is the minimum required occupant load for..." and get a plain-language answer that cites the exact section number(s), with a confidence score, sources shown, and the interaction logged.
