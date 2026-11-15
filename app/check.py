"""Connection check.   python -m app.check

Tests every piece separately so you see exactly which one is misconfigured: embedding server,
Qdrant, the SQLite tables, and the MCP server itself (called like Unsloth Studio would call it).
Run it after the server is up, e.g.  docker compose exec mcp python -m app.check
"""
from __future__ import annotations

import asyncio
import os
import sys

from . import embeddings, tables, vectorstore
from .config import settings

_failures = 0


def report(ok: bool, name: str, detail: str = "", hint: str = "") -> None:
    global _failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        _failures += 1
        if hint:
            print(f"       hint: {hint}")


def _leaf(exc: BaseException) -> BaseException:
    """Unwrap ExceptionGroups (the MCP client raises them) to the actual cause."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


def _probe(url: str) -> tuple[str, str] | None:
    """Plain HTTP request first, so a failure can say *why*: server down, wrong token, wrong path."""
    import httpx

    headers = {"Accept": "application/json, text/event-stream"}
    if settings.mcp_auth_token:
        headers["Authorization"] = f"Bearer {settings.mcp_auth_token}"
    try:
        response = httpx.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=headers, timeout=5)
    except httpx.HTTPError as exc:
        return (
            f"no connection ({type(exc).__name__})",
            "is the server running (python -m app.server / docker compose up -d)? check MCP_PORT",
        )
    if response.status_code in (401, 403):
        return f"HTTP {response.status_code}: token rejected", "MCP_AUTH_TOKEN must be identical on the server and the client"
    if response.status_code == 404:
        return "HTTP 404: nothing at this path", "check MCP_PATH (default /mcp)"
    return None


async def _check_mcp(url: str) -> list[tuple[bool, str, str, str]]:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    results: list[tuple[bool, str, str, str]] = []
    headers = {"Authorization": f"Bearer {settings.mcp_auth_token}"} if settings.mcp_auth_token else {}
    async with httpx2.AsyncClient(headers=headers, timeout=30) as http:
        async with Client(streamable_http_client(url, http_client=http)) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            expected = ["calculator", "get_current_time", "query_database", "search_docs"]
            results.append((names == expected, "MCP tools listed", ", ".join(names), f"expected {expected}"))

            calc = await client.call_tool("calculator", {"expression": "12*7"})
            text = calc.content[0].text if calc.content else ""
            results.append((text == "84", "MCP tool call (calculator)", f"12*7 -> {text!r}", ""))

            search = await client.call_tool("search_docs", {"query": "test"})
            text = search.content[0].text if search.content else ""
            ok = not text.startswith("Search failed")
            first = text.splitlines()[0][:70] if text else ""
            results.append((ok, "MCP tool call (search_docs)", first, "run `python -m app.ingest` and check embeddings/Qdrant above"))
    return results


def main() -> int:
    # 1. embeddings
    try:
        dim = embeddings.embedding_dim()
        report(True, "Embedding server", f"{settings.embed_base_url} -> {dim} dimensions")
    except Exception as exc:  # noqa: BLE001
        report(False, "Embedding server", str(exc), "is llama-server running with --embeddings? check EMBED_BASE_URL")

    # 2. qdrant
    try:
        vectorstore.get_client().get_collections()
        report(True, "Qdrant", f"{settings.qdrant_url}, collection '{settings.collection}' holds {vectorstore.count()} chunks")
    except Exception as exc:  # noqa: BLE001
        report(False, "Qdrant", str(exc), "is the qdrant container up? check QDRANT_URL")

    # 3. database
    try:
        names = tables.build_db()
        report(True, "Database tables", ", ".join(names) if names else "no CSV files found in " + str(settings.tables_dir))
    except Exception as exc:  # noqa: BLE001
        report(False, "Database tables", str(exc))

    # 4. the MCP server, called like an MCP client (Unsloth Studio) would
    url = os.getenv("MCP_CHECK_URL", f"http://127.0.0.1:{settings.mcp_port}{settings.mcp_path}")
    problem = _probe(url)
    if problem:
        report(False, "MCP server", f"{url} -> {problem[0]}", problem[1])
    else:
        try:
            for ok, name, detail, hint in asyncio.run(_check_mcp(url)):
                report(ok, name, detail, hint)
        except Exception as exc:  # noqa: BLE001
            cause = _leaf(exc)
            report(False, "MCP server", f"{url} -> {type(cause).__name__}: {cause}", "see the server log for details")

    print("\nAll checks passed." if _failures == 0 else f"\n{_failures} check(s) failed.")
    return 0 if _failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
