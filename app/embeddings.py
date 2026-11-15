"""Embeddings via llama.cpp's llama-server (OpenAI-compatible /v1/embeddings endpoint)."""
from __future__ import annotations

import time

import httpx

from .config import settings

_client = httpx.Client(timeout=120.0)


def _post(texts: list[str]) -> list[list[float]]:
    url = settings.embed_base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {settings.embed_api_key}"} if settings.embed_api_key else {}
    payload = {"model": settings.embed_model, "input": texts}

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = _client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d.get("index", 0))
            vectors = [d["embedding"] for d in data]
            if vectors and isinstance(vectors[0][0], list):
                raise RuntimeError(
                    "The server returned one vector per token. Start llama-server with "
                    "--embeddings and a pooling mode (e.g. --pooling mean)."
                )
            if len(vectors) != len(texts):
                raise RuntimeError(f"Sent {len(texts)} texts but got {len(vectors)} embeddings back.")
            return vectors
        except RuntimeError:
            raise
        except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Embedding server error at {url}: {last_error}")


def embed_documents(texts: list[str]) -> list[list[float]]:
    prefix = settings.embed_doc_prefix
    out: list[list[float]] = []
    for start in range(0, len(texts), settings.embed_batch):
        batch = [prefix + t for t in texts[start : start + settings.embed_batch]]
        out.extend(_post(batch))
    return out


def embed_query(text: str) -> list[float]:
    return _post([settings.embed_query_prefix + text])[0]


def embedding_dim() -> int:
    return len(embed_query("dimension probe"))
