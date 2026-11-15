"""MCP server: exposes the four tools to any MCP client (e.g. the Unsloth Studio chat).

The LLM in the chat UI is the agent: it sees the tool descriptions, decides which tool to call,
reads the result, and repeats until it can answer. This server only provides the tools.

    python -m app.server                            # HTTP server (MCP_HOST / MCP_PORT / MCP_PATH)
    python -m app.server --transport sse            # legacy SSE transport, if a client needs it
    python -m app.server --transport stdio          # for clients that launch the server themselves
"""
from __future__ import annotations

import argparse
import hmac
import sys
from typing import Annotated

import uvicorn
from mcp.server.mcpserver import MCPServer
from pydantic import Field
from starlette.responses import PlainTextResponse

from . import tables, tools
from .config import settings

INSTRUCTIONS = """Company knowledge tools. Use them instead of answering company questions from memory.
- search_docs: what the company's text documents say.
- query_database: facts and numbers in the company's tables.
- calculator: all arithmetic (also sums over values you retrieved).
- get_current_time: today's date; call it before answering anything relative to today.
Never invent facts. If a search finds nothing useful, retry with different keywords, then say so.
Name your sources: document file names for text, table names for database results."""


def build_server() -> MCPServer:
    """Build the database from the CSVs (if they changed) and register the tools."""
    tables.build_db()
    schema = tables.describe_schema(sample_rows=1)
    mcp = MCPServer("company-knowledge", instructions=INSTRUCTIONS)

    @mcp.tool(
        name="search_docs",
        description=(
            "Search the company's text documents (policies, manuals, reports, notes) with hybrid "
            "keyword + semantic search. Returns the most relevant passages with their source file. "
            "Use it for questions about what the documents say. If the passages do not contain the "
            "answer, search again with different wording. Not for numbers that live in database tables."
        ),
    )
    def search_docs(
        query: Annotated[str, Field(description="What to look for. Use specific keywords, names and numbers, e.g. 'vacation carry-over deadline'.")],
    ) -> str:
        return tools.search_docs(query)

    @mcp.tool(
        name="query_database",
        description=(
            "Run one read-only SQLite SELECT query on the company database. Use it for counts, "
            "totals, averages, lists and lookups in structured data. Use GROUP BY / aggregates "
            "instead of fetching many rows. Only the tables and columns below exist.\n\n"
            "Database schema:\n" + schema
        ),
    )
    def query_database(
        sql: Annotated[str, Field(description="One read-only SQLite SELECT statement.")],
    ) -> str:
        return tools.query_database(sql)

    @mcp.tool(
        name="calculator",
        description=(
            "Evaluate a math expression exactly. Use it for ANY arithmetic instead of calculating "
            "in your head. Supports + - * / // % ** ( ) and sqrt, abs, round, min, max, log, "
            "log10, exp, floor, ceil, sin, cos, tan, pi, e."
        ),
    )
    def calculator(
        expression: Annotated[str, Field(description="Math expression, e.g. '(120 * 8) / 3' or 'sqrt(2) * 10'.")],
    ) -> str:
        return tools.calculator(expression)

    @mcp.tool(
        name="get_current_time",
        description=(
            "Get the current date, weekday and time. Use it for questions about today/now and to "
            "resolve relative dates such as 'last month', 'this week' or 'in 30 days'."
        ),
    )
    def get_current_time(
        timezone: Annotated[str, Field(description="IANA timezone like 'Europe/Berlin' or 'UTC'. Leave empty for the default.")] = "",
    ) -> str:
        return tools.get_current_time(timezone)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request) -> PlainTextResponse:  # used by the Docker healthcheck
        return PlainTextResponse("ok")

    return mcp


class BearerAuthMiddleware:
    """Pure ASGI middleware: every HTTP request needs  Authorization: Bearer <token>  (except /health)."""

    def __init__(self, app, token: str, exempt_paths: tuple[str, ...] = ("/health",)):
        self.app = app
        self.token = token.encode()
        self.exempt_paths = set(exempt_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] not in self.exempt_paths:
            header = dict(scope["headers"]).get(b"authorization", b"")
            supplied = header[7:] if header[:7].lower() == b"bearer " else b""
            if not hmac.compare_digest(supplied, self.token):
                body = b'{"error": "unauthorized"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def create_app(
    transport: str | None = None,
    path: str | None = None,
    token: str | None = None,
    host: str | None = None,
):
    """The ASGI app (Starlette) for the HTTP transports. Arguments default to the settings."""
    transport = transport or settings.mcp_transport
    path = path or settings.mcp_path
    token = settings.mcp_auth_token if token is None else token
    host = host or settings.mcp_host

    mcp = build_server()
    if transport == "sse":
        app = mcp.sse_app(host=host)
    else:
        # stateless + JSON responses: simplest for clients, survives server restarts
        app = mcp.streamable_http_app(streamable_http_path=path, json_response=True, stateless_http=True, host=host)
    if token:
        app.add_middleware(BearerAuthMiddleware, token=token)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Company knowledge MCP server")
    parser.add_argument("--transport", choices=["streamable-http", "sse", "stdio"], default=settings.mcp_transport)
    args = parser.parse_args(argv)

    if args.transport == "stdio":
        build_server().run(transport="stdio")
        return 0

    loopback = settings.mcp_host in ("127.0.0.1", "localhost", "::1")
    if not settings.mcp_auth_token and not loopback:
        print(
            "WARNING: MCP_AUTH_TOKEN is empty and the server listens on the network - "
            "anyone who can reach it can use the tools. Set MCP_AUTH_TOKEN.",
            file=sys.stderr,
        )
    endpoint = "/sse" if args.transport == "sse" else settings.mcp_path
    print(f"MCP server ({args.transport}) on http://{settings.mcp_host}:{settings.mcp_port}{endpoint}", file=sys.stderr)
    uvicorn.run(create_app(args.transport), host=settings.mcp_host, port=settings.mcp_port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
