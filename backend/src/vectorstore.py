"""
Phase 2: Weaviate vector store — schema definition and idempotent loader.

Collection: NYCBuildingCode
Fields: chunk_id, code_name, edition, source_file, chapter, chapter_title,
        section_number, section_title, parent_section, page_start, page_end,
        token_count, split_index, content_hash, text (vectorized)
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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


def _split_host_port(url: str, default_port: int) -> tuple[str, int]:
    """Split a 'http(s)://host[:port]' URL into (host, port).

    Cloud-style URLs (e.g. Weaviate Cloud) commonly have no explicit port —
    checking `":" in url` to decide that is wrong because the "http://" /
    "https://" scheme itself always contains a colon, so that check is
    always true and int() blows up parsing the bare hostname as a port.
    """
    hostpart = url.replace("http://", "").replace("https://", "").split("/")[0]
    if ":" in hostpart:
        host, port_str = hostpart.rsplit(":", 1)
        return host, int(port_str)
    return hostpart, default_port


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Return a connected Weaviate client."""
    if WEAVIATE_API_KEY:
        auth = weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
        host, http_port = _split_host_port(WEAVIATE_URL, default_port=443)
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=WEAVIATE_URL.startswith("https"),
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=WEAVIATE_URL.startswith("https"),
            auth_credentials=auth,
        )
    host, port = _split_host_port(WEAVIATE_URL, default_port=8080)
    return weaviate.connect_to_local(host=host, port=port)


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    """Create the collection if it doesn't exist (idempotent).

    Also backfills the `content_hash` property onto a collection that was
    created before that field existed, so content-based idempotency in
    `load_chunks` works even against a store from an older version of this
    script.
    """
    if client.collections.exists(COLLECTION_NAME):
        collection = client.collections.get(COLLECTION_NAME)
        existing_props = {p.name for p in collection.config.get().properties}
        if "content_hash" not in existing_props:
            collection.config.add_property(
                wvc.config.Property(name="content_hash", data_type=wvc.config.DataType.TEXT, skip_vectorization=True)
            )
            print(f"Collection '{COLLECTION_NAME}' already exists — added missing 'content_hash' property.")
        else:
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
            wvc.config.Property(name="content_hash",   data_type=wvc.config.DataType.TEXT,  skip_vectorization=True),
            wvc.config.Property(name="text",           data_type=wvc.config.DataType.TEXT,  skip_vectorization=False),
        ],
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def _content_hash(text: str) -> str:
    """Stable content fingerprint used to detect edited chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_RETRYABLE_OPENAI_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_OPENAI_ERRORS),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_batch(client: OpenAI, batch: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    return [r.embedding for r in resp.data]


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts using OpenAI in batches.

    Transient failures (rate limits, timeouts, connection drops, 5xx) are
    retried with exponential backoff per batch; other errors (e.g. a bad
    request) fail immediately rather than burning through retries.
    """
    client = _get_openai()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(_embed_batch(client, batch))
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")
    return vectors


def load_chunks(chunks: list[dict], client: weaviate.WeaviateClient) -> None:
    """
    Upsert chunks into Weaviate.  Idempotent by content: a chunk is skipped
    only if its chunk_id already exists AND its text is unchanged, per a
    content hash — not just a chunk_id match. That way, editing the
    chunking logic in ingest.py (same chunk_id, different text) causes the
    chunk to be re-embedded instead of silently leaving a stale vector/text
    in the store. A chunk_id with no matching hash in the store (new,
    edited, or backfilled from a pre-content-hash collection) is treated
    as needing (re-)embedding.
    """
    collection = client.collections.get(COLLECTION_NAME)

    # Fetch existing chunk_id -> content_hash so unchanged chunks (skip)
    # can be told apart from new or edited ones (re-embed).
    existing_hashes: dict[str, str] = {}
    for obj in collection.iterator(return_properties=["chunk_id", "content_hash"]):
        existing_hashes[obj.properties["chunk_id"]] = obj.properties.get("content_hash") or ""
    print(f"  Existing objects in store: {len(existing_hashes)}")

    hashed_chunks = [{**c, "content_hash": _content_hash(c["text"])} for c in chunks]
    new_chunks = [c for c in hashed_chunks if existing_hashes.get(c["chunk_id"]) != c["content_hash"]]
    print(f"  New or changed chunks to embed and upsert: {len(new_chunks)}")
    if not new_chunks:
        return

    texts = [c["text"] for c in new_chunks]
    vectors = embed_texts(texts)

    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(new_chunks, vectors):
            batch.add_object(properties=chunk, vector=vector, uuid=weaviate.util.generate_uuid5(chunk["chunk_id"]))

    failed = collection.batch.failed_objects
    if failed:
        for f in failed[:10]:
            bad_id = f.object_.properties.get("chunk_id", "?") if f.object_ else "?"
            print(f"  FAILED to upsert {bad_id}: {f.message}")
        if len(failed) > 10:
            print(f"  ...and {len(failed) - 10} more failures.")
        raise RuntimeError(
            f"{len(failed)}/{len(new_chunks)} chunks failed to upsert into Weaviate "
            "(see messages above). Fix the underlying data/schema issue and re-run — "
            "already-succeeded chunks will be skipped next time."
        )

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

    with get_weaviate_client() as wv:
        ensure_collection(wv)
        load_chunks(all_chunks, wv)
    print("Done.")
