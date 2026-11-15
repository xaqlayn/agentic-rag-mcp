"""Central configuration. Everything comes from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # real environment variables always win over .env


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # --- MCP server (what Unsloth Studio connects to) ---
    mcp_transport: str  # streamable-http | sse | stdio
    mcp_host: str  # 127.0.0.1 = this machine only; 0.0.0.0 = reachable from the network (Docker)
    mcp_port: int
    mcp_path: str  # URL path of the endpoint, e.g. /mcp
    mcp_auth_token: str  # if set, clients must send  Authorization: Bearer <token>

    # --- Embeddings: llama.cpp llama-server started with --embeddings ---
    embed_base_url: str
    embed_api_key: str
    embed_model: str
    embed_query_prefix: str  # e.g. "search_query: " for nomic-embed
    embed_doc_prefix: str  # e.g. "search_document: " for nomic-embed
    embed_batch: int

    # --- Qdrant ---
    qdrant_url: str  # http://host:6333  (":memory:" is used by the tests)
    qdrant_api_key: str | None
    collection: str

    # --- Data locations ---
    docs_dir: Path
    tables_dir: Path
    db_path: Path

    # --- Tool behaviour ---
    timezone: str
    top_k: int  # passages returned by search_docs
    prefetch_k: int  # candidates per search branch before fusion
    chunk_chars: int
    chunk_overlap: int
    sql_max_rows: int
    sql_timeout: float


def load_settings() -> Settings:
    return Settings(
        mcp_transport=_str("MCP_TRANSPORT", "streamable-http"),
        mcp_host=_str("MCP_HOST", "127.0.0.1"),
        mcp_port=_int("MCP_PORT", 8000),
        mcp_path=_str("MCP_PATH", "/mcp"),
        mcp_auth_token=_str("MCP_AUTH_TOKEN", ""),
        embed_base_url=_str("EMBED_BASE_URL", "http://localhost:8081/v1"),
        embed_api_key=_str("EMBED_API_KEY", ""),
        embed_model=_str("EMBED_MODEL", "embedding"),
        embed_query_prefix=_str("EMBED_QUERY_PREFIX", ""),
        embed_doc_prefix=_str("EMBED_DOC_PREFIX", ""),
        embed_batch=_int("EMBED_BATCH", 16),
        qdrant_url=_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=_str("QDRANT_API_KEY") or None,
        collection=_str("QDRANT_COLLECTION", "company_docs"),
        docs_dir=Path(_str("DOCS_DIR", str(ROOT / "data" / "docs"))),
        tables_dir=Path(_str("TABLES_DIR", str(ROOT / "data" / "tables"))),
        db_path=Path(_str("DB_PATH", str(ROOT / "state" / "company.db"))),
        timezone=_str("TIMEZONE", "Europe/Berlin"),
        top_k=_int("TOP_K", 5),
        prefetch_k=_int("PREFETCH_K", 20),
        chunk_chars=_int("CHUNK_CHARS", 900),
        chunk_overlap=_int("CHUNK_OVERLAP", 120),
        sql_max_rows=_int("SQL_MAX_ROWS", 50),
        sql_timeout=_float("SQL_TIMEOUT", 5.0),
    )


settings = load_settings()
