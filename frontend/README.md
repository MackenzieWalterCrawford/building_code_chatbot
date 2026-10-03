# Frontend (React)

Vite + React + TypeScript app. Replaces the retired Streamlit UI (`app.py`)
as the client for the FastAPI backend in `../backend`. Currently just a
question box wired to `POST /ask` — no chapter filter, sources list, or
confidence badge yet.

## Run locally

```bash
npm install
cp .env.example .env.local   # defaults to http://localhost:8000, edit if needed
npm run dev
```

Opens on `http://localhost:5173`. Make sure the backend is running (see
`../backend/README.md` or the root `README.md`) and that `5173` is listed in
the backend's `CORS_ORIGINS`.

## Backend contract

Base URL: `VITE_API_URL` (`.env.local`), default `http://localhost:8000`.

- `GET /health` → `{"status": "ok"}`
- `GET /chapters` → `[{"chapter": "16", "chapter_title": "Structural Design"}, ...]`
  for a future chapter-filter dropdown.
- `POST /ask`
  - Request: `{"question": str, "top_k"?: int, "alpha"?: float, "chapter_filter"?: str}`
  - Response: `{"answer": str, "sources": SourceChunk[], "cited_sections": str[], "confidence": float, "insufficient": bool}`
  - `SourceChunk`: `{section_number, section_title, chapter, chapter_title, text, page_start, page_end, score}`

See `backend/src/api.py` for the authoritative request/response models, and
`src/api.ts` for the typed client used here.
