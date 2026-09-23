# Frontend (React) — not yet scaffolded

This will replace the retired Streamlit UI (`app.py`) as the client for the
FastAPI backend in `../backend`.

## Backend contract

Base URL: `http://localhost:8000` (see `backend/.env`'s `CORS_ORIGINS` — add
this app's dev server origin there, e.g. `http://localhost:5173` for Vite).

- `GET /health` → `{"status": "ok"}`
- `GET /chapters` → `[{"chapter": "16", "chapter_title": "Structural Design"}, ...]`
  for a chapter-filter dropdown.
- `POST /ask`
  - Request: `{"question": str, "top_k"?: int, "alpha"?: float, "chapter_filter"?: str}`
  - Response: `{"answer": str, "sources": SourceChunk[], "cited_sections": str[], "confidence": float, "insufficient": bool}`
  - `SourceChunk`: `{section_number, section_title, chapter, chapter_title, text, page_start, page_end, score}`

See `backend/src/api.py` for the authoritative request/response models.
