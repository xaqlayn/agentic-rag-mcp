"""Load data: text documents -> Qdrant (hybrid index), CSV files -> SQLite.

    python -m app.ingest                # documents + tables
    python -m app.ingest --recreate     # drop and rebuild the Qdrant collection first
    python -m app.ingest --docs-only
    python -m app.ingest --tables-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import embeddings, tables, vectorstore
from .config import settings
from .textproc import chunk_text

DOC_SUFFIXES = {".txt", ".md", ".pdf"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # optional dependency, only needed for PDFs
        except ImportError:
            raise SystemExit(f"{path.name}: PDF support is not installed. Run: pip install pypdf") from None

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_docs(recreate: bool = False) -> tuple[int, int]:
    """Returns (number of files, number of chunks)."""
    docs_dir = settings.docs_dir
    files = sorted(p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in DOC_SUFFIXES)
    if not files:
        print(f"No .txt/.md/.pdf files found in {docs_dir}")
        return 0, 0

    dim = embeddings.embedding_dim()
    existing = None if recreate else vectorstore.collection_dim()
    if existing is not None and existing != dim:
        raise SystemExit(
            f"The collection has {existing}-dimensional vectors but the embedding model returns {dim}. "
            "You changed the embedding model - run again with --recreate."
        )
    vectorstore.ensure_collection(dim, recreate=recreate)

    total_chunks = 0
    for path in files:
        source = path.relative_to(docs_dir).as_posix()
        chunks = chunk_text(read_document(path), settings.chunk_chars, settings.chunk_overlap)
        vectorstore.delete_source(source)  # drop stale chunks if the file got shorter
        if not chunks:
            print(f"  {source}: no text found, skipped")
            continue
        vectorstore.upsert_chunks(source, chunks, embeddings.embed_documents(chunks))
        total_chunks += len(chunks)
        print(f"  {source}: {len(chunks)} chunks")
    return len(files), total_chunks


def ingest_tables() -> list[str]:
    names = tables.build_db(force=True)
    print(f"  tables: {', '.join(names) if names else '(none)'}")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load documents and tables for the agent.")
    parser.add_argument("--recreate", action="store_true", help="drop the Qdrant collection first")
    parser.add_argument("--docs-only", action="store_true")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.tables_only:
        print(f"Documents from {settings.docs_dir}")
        files, chunks = ingest_docs(recreate=args.recreate)
        print(f"-> {files} files, {chunks} chunks; collection now holds {vectorstore.count()} points")
    if not args.docs_only:
        print(f"Tables from {settings.tables_dir}")
        ingest_tables()
    return 0


if __name__ == "__main__":
    sys.exit(main())
