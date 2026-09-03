"""
Phase 2: Weaviate vector store — schema definition and idempotent loader.

Collection: NYCBuildingCode
Fields: chunk_id, code_name, edition, source_file, chapter, chapter_title,
        section_number, section_title, parent_section, page_start, page_end,
        token_count, split_index, text (vectorized)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

COLLECTION_NAME = "NYCBuildingCode"
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

_openai_client: Optional[OpenAI] = None


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Return a connected Weaviate client."""
    if WEAVIATE_API_KEY:
        auth = weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        return weaviate.connect_to_custom(
            http_host=WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0],
            http_port=int(WEAVIATE_URL.split(":")[-1]) if ":" in WEAVIATE_URL else 8080,
            http_secure=WEAVIATE_URL.startswith("https"),
            grpc_host=WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0],
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=auth,
        )
    return weaviate.connect_to_local(
        host=WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0],
        port=int(WEAVIATE_URL.split(":")[-1]) if ":" in WEAVIATE_URL else 8080,
    )


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    """Create the collection if it doesn't exist (idempotent)."""
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    client.collections.create(
        name=COLLECTION_NAME,
        description="NYC Building Code 2022 — section-level chunks",
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(name="chunk_id",       data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="code_name",      data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="edition",        data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="source_file",    data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="chapter",        data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="chapter_title",  data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="section_number", data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="section_title",  data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="parent_section", data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="page_start",     data_type=wvc.config.DataType.INT,   skip_vectorization=True),
            wvc.config.Property(name="page_end",       data_type=wvc.config.DataType.INT,   skip_vectorization=True),
            wvc.config.Property(name="token_count",    data_type=wvc.config.DataType.INT,   skip_vectorization=True),
            wvc.config.Property(name="split_index",    data_type=wvc.config.DataType.INT,   skip_vectorization=True),
            wvc.config.Property(name="text",           data_type=wvc.config.DataType.TEXT,  skip_vectorization=False),
        ],
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts using OpenAI in batches."""
    client = _get_openai()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend([r.embedding for r in resp.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")
    return vectors


def load_chunks(chunks: list[dict], client: weaviate.WeaviateClient) -> None:
    """
    Upsert chunks into Weaviate.  Idempotent: existing chunk_ids are skipped.
    """
    collection = client.collections.get(COLLECTION_NAME)

    # Fetch existing chunk_ids to avoid re-embedding
    existing_ids: set[str] = set()
    for obj in collection.iterator(return_properties=["chunk_id"]):
        existing_ids.add(obj.properties["chunk_id"])
    print(f"  Existing objects in store: {len(existing_ids)}")

    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"  New chunks to embed and upsert: {len(new_chunks)}")
    if not new_chunks:
        return

    texts = [c["text"] for c in new_chunks]
    vectors = embed_texts(texts)

    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(new_chunks, vectors):
            props = {k: v for k, v in chunk.items() if k != "chunk_id"}
            batch.add_object(properties=props, vector=vector, uuid=weaviate.util.generate_uuid5(chunk["chunk_id"]))

    print(f"  Upserted {len(new_chunks)} chunks.")


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    processed_dir = Path("data/processed")
    json_files = sorted(processed_dir.glob("2022BC_Chapter*.json"))
    if not json_files:
        print("No processed JSON files found. Run ingest.py first.")
        sys.exit(1)

    all_chunks: list[dict] = []
    for f in json_files:
        all_chunks.extend(json.loads(f.read_text()))
    print(f"Loaded {len(all_chunks)} chunks from {len(json_files)} files")

    wv = get_weaviate_client()
    ensure_collection(wv)
    load_chunks(all_chunks, wv)
    wv.close()
    print("Done.")
