"""
Phase 5: FastAPI backend.

POST /ask   { "question": "..." }
            → { "answer": "...", "sources": [...], "confidence": 0.87 }

GET  /health
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import weaviate
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Sibling modules (generate, retrieve, ...) are imported as top-level names. Docker
# gets this via PYTHONPATH=/app/src; put src/ on the path here so the Vercel
# service (entrypoint src.api:app) resolves them too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import generate
from logger import log_interaction
from retrieve import retrieve
from vectorstore import COLLECTION_NAME, get_weaviate_client

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weaviate client (shared across requests)
# ---------------------------------------------------------------------------

_wv_client: Optional[weaviate.WeaviateClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _wv_client
    _wv_client = get_weaviate_client()
    log.info("Weaviate client connected.")
    yield
    if _wv_client:
        _wv_client.close()
        log.info("Weaviate client closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NYC Building Code RAG",
    description="Ask natural-language questions about the 2022 NYC Building Code.",
    version="1.0.0",
    lifespan=lifespan,
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    top_k: int = Field(default=8, ge=1, le=20)
    alpha: float = Field(default=0.75, ge=0.0, le=1.0)
    chapter_filter: Optional[str] = Field(default=None)


class SourceChunk(BaseModel):
    section_number: str
    section_title: str
    chapter: str
    chapter_title: str
    text: str
    page_start: int
    page_end: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    cited_sections: list[str]
    confidence: float
    insufficient: bool


class ChapterInfo(BaseModel):
    chapter: str
    chapter_title: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/chapters", response_model=list[ChapterInfo])
def chapters():
    """List distinct chapters in the store, for a frontend filter dropdown."""
    collection = _wv_client.collections.get(COLLECTION_NAME)
    seen: dict[str, str] = {}
    for obj in collection.iterator(return_properties=["chapter", "chapter_title"]):
        seen.setdefault(obj.properties["chapter"], obj.properties["chapter_title"])

    def sort_key(chapter: str) -> tuple[int, str]:
        return (0, f"{int(chapter):04d}") if chapter.isdigit() else (1, chapter)

    return [
        ChapterInfo(chapter=c, chapter_title=t)
        for c, t in sorted(seen.items(), key=lambda item: sort_key(item[0]))
    ]


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.perf_counter()

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(
        query=req.question,
        top_k=req.top_k,
        alpha=req.alpha,
        chapter_filter=req.chapter_filter,
        client=_wv_client,
    )

    result = generate(question=req.question, chunks=chunks)

    elapsed = round(time.perf_counter() - t0, 3)

    log_interaction(
        question=req.question,
        retrieved_chunks=[c.__dict__ for c in chunks],
        answer=result.answer,
        cited_sections=result.cited_sections,
        confidence=result.confidence,
        latency_s=elapsed,
    )

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                section_number=c.section_number,
                section_title=c.section_title,
                chapter=c.chapter,
                chapter_title=c.chapter_title,
                text=c.text,
                page_start=c.page_start,
                page_end=c.page_end,
                score=c.score,
            )
            for c in chunks
        ],
        cited_sections=result.cited_sections,
        confidence=result.confidence,
        insufficient=result.insufficient,
    )


# Vercel forwards the original request path (no prefix stripping), so the public
# routes live under /api. The bare paths stay for local/docker use and curl.
app.include_router(router, prefix="/api")
app.include_router(router, include_in_schema=False)
