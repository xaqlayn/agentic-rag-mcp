"""Qdrant hybrid search: dense (embedding) + sparse (BM25-style keyword) vectors fused with RRF."""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from . import embeddings
from .config import settings
from .textproc import sparse_doc, sparse_query

DENSE = "dense"
SPARSE = "bm25"
_NAMESPACE = uuid.UUID("6f1d3b1e-2c55-4c0e-9d8e-5b0f4f3a7e11")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            location=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
            check_compatibility=False,
        )
    return _client


def collection_dim() -> int | None:
    """Vector size of the existing collection, or None if it does not exist."""
    client = get_client()
    if not client.collection_exists(settings.collection):
        return None
    vectors = client.get_collection(settings.collection).config.params.vectors
    return vectors[DENSE].size if isinstance(vectors, dict) and DENSE in vectors else None


def ensure_collection(dim: int, recreate: bool = False) -> None:
    client = get_client()
    name = settings.collection
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        # IDF modifier: Qdrant computes inverse document frequency itself -> BM25-like scoring
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.create_payload_index(name, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD)


def delete_source(source: str) -> None:
    get_client().delete(
        collection_name=settings.collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
            )
        ),
    )


def upsert_chunks(source: str, chunks: list[str], dense_vectors: list[list[float]]) -> None:
    """Store all chunks of one source file. Point IDs are stable, so re-ingesting overwrites."""
    points = []
    for i, (text, vector) in enumerate(zip(chunks, dense_vectors)):
        indices, values = sparse_doc(text)
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, f"{source}#{i}")),
                vector={DENSE: vector, SPARSE: models.SparseVector(indices=indices, values=values)},
                payload={"source": source, "chunk": i, "text": text},
            )
        )
    client = get_client()
    for start in range(0, len(points), 64):
        client.upsert(collection_name=settings.collection, points=points[start : start + 64], wait=True)


def hybrid_search(query: str, k: int | None = None) -> list[dict]:
    """Keyword + semantic search, fused with Reciprocal Rank Fusion."""
    k = k or settings.top_k
    prefetch = [
        models.Prefetch(query=embeddings.embed_query(query), using=DENSE, limit=settings.prefetch_k)
    ]
    indices, values = sparse_query(query)
    if indices:
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(indices=indices, values=values),
                using=SPARSE,
                limit=settings.prefetch_k,
            )
        )
    result = get_client().query_points(
        collection_name=settings.collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [
        {
            "source": p.payload.get("source"),
            "chunk": p.payload.get("chunk"),
            "text": p.payload.get("text", ""),
            "score": p.score,
        }
        for p in result.points
    ]


def count() -> int:
    client = get_client()
    if not client.collection_exists(settings.collection):
        return 0
    return client.count(settings.collection, exact=True).count
