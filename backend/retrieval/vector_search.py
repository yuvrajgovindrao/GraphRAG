"""
Qdrant vector search for chunk retrieval.
Handles embedding storage and similarity search.
Supports both local embedded storage (zero-docker) and cloud Qdrant.
"""

from __future__ import annotations

import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from backend.config import Settings
from backend.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """
    Factory for Qdrant client.
    Uses local disk embedded storage by default (zero Docker required).
    Falls back to cloud/remote cluster if qdrant_url is provided.
    """
    if settings.qdrant_url:
        logger.info("Connecting to remote Qdrant at %s", settings.qdrant_url)
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )
    else:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        logger.info("Using local embedded Qdrant store at %s", settings.qdrant_path)
        return QdrantClient(path=str(settings.qdrant_path))


def ensure_collection(client: QdrantClient, dimension: int) -> None:
    """Create the chunks collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s' with dim=%d", COLLECTION_NAME, dimension)
    else:
        logger.info("Qdrant collection '%s' already exists", COLLECTION_NAME)


async def index_chunks(
    client: QdrantClient,
    llm: BaseLLMProvider,
    chunks: list[dict],
) -> None:
    """Embed chunks and upsert into Qdrant."""
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = await llm.embed(texts)

    points = [
        PointStruct(
            id=c["id"],
            vector=emb,
            payload={
                "doc_id": c["doc_id"],
                "text": c["text"],
                "page_number": c.get("page_number"),
                "chunk_index": c.get("chunk_index", 0),
                "filename": c.get("source_filename", ""),
            },
        )
        for c, emb in zip(chunks, embeddings)
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    logger.info("Indexed %d chunks into Qdrant", len(chunks))


async def search_similar(
    client: QdrantClient,
    llm: BaseLLMProvider,
    query: str,
    top_k: int = 5,
    doc_id_filter: str | None = None,
) -> list[dict]:
    """Embed query and search for similar chunks."""
    query_embedding = (await llm.embed([query]))[0]

    search_filter = None
    if doc_id_filter:
        search_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    return [
        {
            "chunk_id": str(hit.id),
            "doc_id": hit.payload.get("doc_id", ""),
            "text": hit.payload.get("text", ""),
            "filename": hit.payload.get("filename", ""),
            "page_number": hit.payload.get("page_number"),
            "score": hit.score,
        }
        for hit in results.points
    ]


def delete_doc_vectors(client: QdrantClient, doc_id: str) -> None:
    """Delete all vectors for a given document."""
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    logger.info("Deleted vectors for doc_id=%s", doc_id)
