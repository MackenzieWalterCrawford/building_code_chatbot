"""
Phase 3: Hybrid retrieval (dense vector + BM25) from Weaviate.

Key params:
  alpha   — 1.0 = pure vector, 0.0 = pure BM25  (default 0.75)
  top_k   — number of chunks to return           (default 8)
  chapter — optional metadata filter
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv

from vectorstore import COLLECTION_NAME, embed_texts, get_weaviate_client

load_dotenv()

DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
DEFAULT_ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.75"))


@dataclass
class RetrievedChunk:
    chunk_id: str
    section_number: str
    section_title: str
    chapter: str
    chapter_title: str
    text: str
    page_start: int
    page_end: int
    score: float
    source_file: str
    edition: str = "2022"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    chapter_filter: Optional[str] = None,
    client: Optional[weaviate.WeaviateClient] = None,
) -> list[RetrievedChunk]:
    """
    Hybrid search: dense vector + BM25.
    alpha=1.0 → pure vector, alpha=0.0 → pure keyword.
    """
    owns_client = client is None
    if owns_client:
        client = get_weaviate_client()

    try:
        # Embed the query
        query_vector = embed_texts([query])[0]

        collection = client.collections.get(COLLECTION_NAME)

        # Build optional where filter
        filters = None
        if chapter_filter:
            filters = wvc.query.Filter.by_property("chapter").equal(chapter_filter)

        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=top_k,
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(score=True),
            return_properties=[
                "chunk_id", "section_number", "section_title",
                "chapter", "chapter_title", "text",
                "page_start", "page_end", "source_file", "edition",
            ],
        )

        results: list[RetrievedChunk] = []
        for obj in response.objects:
            p = obj.properties
            score = obj.metadata.score if obj.metadata else 0.0
            results.append(RetrievedChunk(
                chunk_id=p.get("chunk_id", ""),
                section_number=p.get("section_number", ""),
                section_title=p.get("section_title", ""),
                chapter=p.get("chapter", ""),
                chapter_title=p.get("chapter_title", ""),
                text=p.get("text", ""),
                page_start=p.get("page_start", 0),
                page_end=p.get("page_end", 0),
                score=float(score),
                source_file=p.get("source_file", ""),
                edition=p.get("edition", "2022"),
            ))

        return results

    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "minimum floor live load for office buildings"
    print(f"Query: {query}\n")
    results = retrieve(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"[{i}] §{r.section_number} — {r.section_title}  (score: {r.score:.4f})")
        print(f"     Chapter {r.chapter}: {r.chapter_title}")
        print(f"     {r.text[:200]}...")
        print()
