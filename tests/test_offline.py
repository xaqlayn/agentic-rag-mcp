"""Offline tests: no embedding server, Qdrant server or LLM needed.

    pip install pytest
    pytest -q

Qdrant runs in its in-memory mode and embeddings are faked. The MCP server itself is tested with
the real MCP client, in-process and over real HTTP (including the bearer-token check).
"""
import asyncio
import contextlib
import math
import os
import socket
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
_TMP = Path(tempfile.mkdtemp())
os.environ.update(
    {
        "QDRANT_URL": ":memory:",
        "DB_PATH": str(_TMP / "company.db"),
        "DOCS_DIR": str(PROJECT / "data" / "docs"),
        "TABLES_DIR": str(PROJECT / "data" / "tables"),
        "MCP_AUTH_TOKEN": "",
    }
)

import httpx  # noqa: E402
import httpx2  # noqa: E402
import pytest  # noqa: E402
import uvicorn  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from app import embeddings, ingest, server, tables, textproc, tools, vectorstore  # noqa: E402

TOKEN = "test-token"
EXPECTED_TOOLS = ["calculator", "get_current_time", "query_database", "search_docs"]


# ---------------------------------------------------------------------------- test doubles

DIM = 64


def fake_embed(texts):
    """Deterministic bag-of-words embedding (hashed into 64 dims, L2-normalised)."""
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for token in textproc.tokenize(text):
            vec[textproc._index(token) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "_post", fake_embed)


@pytest.fixture(scope="module")
def built_db():
    return tables.build_db(force=True)


@pytest.fixture(scope="module")
def indexed():
    patch = pytest.MonkeyPatch()
    patch.setattr(embeddings, "_post", fake_embed)
    ingest.ingest_docs(recreate=True)
    yield
    patch.undo()


# ------------------------------------------------------------------------------ calculator


@pytest.mark.parametrize(
    "expr,expected",
    [("2+3*4", "14"), ("sqrt(16)", "4"), ("2^10", "1024"), ("(120*8)/3", "320"), ("round(10/3, 2)", "3.33"), ("-5 + abs(-2)", "-3")],
)
def test_calculator_ok(expr, expected):
    assert tools.calculator(expr) == expected


@pytest.mark.parametrize(
    "expr",
    ["1/0", "__import__('os').system('id')", "open('x')", "9**9**9", "10**10000", "a + 1", "", "[1,2]", "lambda: 1"],
)
def test_calculator_rejects(expr):
    assert tools.calculator(expr).startswith("Error")


# ------------------------------------------------------------------------------------ time


def test_time_tool():
    assert "Europe/Berlin" in tools.get_current_time()
    assert "UTC" in tools.get_current_time("UTC")
    assert "Unknown timezone" in tools.get_current_time("Mars/Olympus")


# --------------------------------------------------------------------------------- database


def test_csv_import_and_types(built_db):
    assert set(built_db) == {"employees", "orders"}
    con = sqlite3.connect(os.environ["DB_PATH"])
    types = {r[1]: r[2] for r in con.execute("PRAGMA table_info(orders)")}
    con.close()
    assert types["quantity"] == "INTEGER" and types["unit_price"] == "REAL" and types["customer"] == "TEXT"


def test_csv_import_semicolon_zip_and_umlauts(tmp_path):
    csv_file = tmp_path / "Kunden Liste.csv"
    csv_file.write_bytes("Name;PLZ;Umsatz\nMüller;01234;12.5\nSchröder;83022;7\n".encode("cp1252"))
    con = sqlite3.connect(":memory:")
    name = tables._import_csv(con, csv_file, set())
    assert name == "kunden_liste"
    assert con.execute("SELECT plz, umsatz, name FROM kunden_liste ORDER BY name").fetchall() == [
        ("01234", 12.5, "Müller"),
        ("83022", 7.0, "Schröder"),
    ]


def test_sql_select_works(built_db):
    out = tools.query_database("SELECT region, SUM(quantity*unit_price) AS revenue FROM orders GROUP BY region ORDER BY region")
    assert "Bavaria" in out and "Hamburg" in out and "NRW" in out


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "PRAGMA table_info(orders)",
        "ATTACH DATABASE 'x.db' AS x",
        "WITH x AS (SELECT 1) DELETE FROM orders",
        "SELECT * FROM _meta",
        "",
    ],
)
def test_sql_rejects_everything_but_reads(built_db, sql):
    assert tools.query_database(sql).startswith("Error")
    assert tools.query_database("SELECT COUNT(*) FROM orders").endswith("20")  # still intact


def test_sql_cte_and_row_limit(built_db):
    assert "Error" not in tools.query_database("WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t")
    recursive = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<500) SELECT x FROM c"
    assert "only the first" in tools.query_database(recursive)


def test_schema_description(built_db):
    schema = tables.describe_schema()
    assert "Table orders (20 rows)" in schema and "unit_price REAL" in schema and "_meta" not in schema


# ------------------------------------------------------------------------ chunking / sparse


def test_chunking():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))
    chunks = textproc.chunk_text(text, size=500, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) < 700 for c in chunks)
    joined = " ".join(chunks)
    assert all(f"Paragraph {i}." in joined for i in range(12))
    assert textproc.chunk_text("   ") == []
    assert len(textproc.chunk_text("x" * 2500, size=900, overlap=0)) == 3  # hard split of one giant token


def test_sparse_vectors_overlap_on_shared_terms():
    di, dv = textproc.sparse_doc("Passwords must have at least 14 characters")
    qi, qv = textproc.sparse_query("password characters")  # no stemming: 'password' != 'passwords'
    assert set(di) & set(qi) == {textproc._index("characters")}
    assert len(di) == len(dv) and all(v > 0 for v in dv)


# --------------------------------------------------------------------------- hybrid search


def test_ingest_and_hybrid_search():
    files, chunks = ingest.ingest_docs(recreate=True)
    assert files == 3 and chunks >= 3
    again = vectorstore.count()
    ingest.ingest_docs()  # idempotent: same IDs, no duplicates
    assert vectorstore.count() == again

    hits = vectorstore.hybrid_search("carry-over deadline March 31", 3)
    assert hits and hits[0]["source"] == "vacation_policy.md"
    hits = vectorstore.hybrid_search("phishing email forward security", 3)
    assert hits[0]["source"] == "it_security.md"
    assert "[1] source:" in tools.search_docs("probation period")


# ----------------------------------------------------------------- MCP server, in-process


def run(coro):
    return asyncio.run(coro)


async def call(client, name, **args):
    result = await client.call_tool(name, args)
    return result.content[0].text


def test_mcp_lists_tools_with_descriptions(built_db):
    async def go():
        async with Client(server.build_server()) as client:
            listed = (await client.list_tools()).tools
            return client.instructions, {t.name: t for t in listed}

    instructions, listed = run(go())
    assert sorted(listed) == EXPECTED_TOOLS
    assert "Table orders (20 rows)" in listed["query_database"].description  # schema is in the description
    assert "Never invent facts" in instructions
    schema = listed["calculator"].input_schema if hasattr(listed["calculator"], "input_schema") else listed["calculator"].inputSchema
    assert schema["required"] == ["expression"]


def test_mcp_tools_work(built_db, indexed):
    async def go():
        async with Client(server.build_server()) as client:
            return {
                "calc": await call(client, "calculator", expression="12*7"),
                "bad_calc": await call(client, "calculator", expression="__import__('os')"),
                "time": await call(client, "get_current_time"),
                "sql": await call(client, "query_database", sql="SELECT COUNT(*) AS n FROM employees"),
                "bad_sql": await call(client, "query_database", sql="DROP TABLE orders"),
                "docs": await call(client, "search_docs", query="carry-over deadline March 31"),
            }

    out = run(go())
    assert out["calc"] == "84"
    assert out["bad_calc"].startswith("Error")
    assert "Europe/Berlin" in out["time"]
    assert out["sql"].splitlines()[-1] == "10"
    assert out["bad_sql"].startswith("Error")
    assert "vacation_policy.md" in out["docs"]


# ----------------------------------------------------------- MCP server over real HTTP + auth


def start_uvicorn(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=uv.run, daemon=True).start()
    while not uv.started:
        time.sleep(0.05)
    return uv, port


@contextlib.asynccontextmanager
async def connect(url, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx2.AsyncClient(headers=headers, timeout=30) as http:
        async with Client(streamable_http_client(url, http_client=http)) as client:
            yield client


@pytest.fixture(scope="module")
def secured_server(built_db, indexed):
    uv, port = start_uvicorn(server.create_app(token=TOKEN))
    yield f"http://127.0.0.1:{port}"
    uv.should_exit = True


@pytest.fixture(scope="module")
def open_server_at_root(built_db):
    uv, port = start_uvicorn(server.create_app(path="/", token=""))
    yield f"http://127.0.0.1:{port}"
    uv.should_exit = True


def test_http_with_token_works(secured_server):
    async def go():
        async with connect(secured_server + "/mcp", TOKEN) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            return names, await call(client, "calculator", expression="6*7"), await call(client, "search_docs", query="phishing")

    names, calc, docs = run(go())
    assert names == EXPECTED_TOOLS and calc == "42" and "it_security.md" in docs


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_http_rejects_missing_or_wrong_token(secured_server, token):
    async def go():
        async with connect(secured_server + "/mcp", token) as client:
            await client.list_tools()

    with pytest.raises(BaseException):  # noqa: B017 - the client raises an ExceptionGroup around the 401
        run(go())


def test_http_raw_status_codes(secured_server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    headers = {"Accept": "application/json, text/event-stream"}
    assert httpx.post(secured_server + "/mcp", json=body, headers=headers).status_code == 401
    assert httpx.post(secured_server + "/mcp", json=body, headers={**headers, "Authorization": "Basic abc"}).status_code == 401
    assert httpx.get(secured_server + "/health").text == "ok"  # health check needs no token


def test_http_endpoint_at_root_path(open_server_at_root):
    async def go():
        async with connect(open_server_at_root + "/") as client:
            return sorted(t.name for t in (await client.list_tools()).tools)

    assert run(go()) == EXPECTED_TOOLS
