"""Structured data: CSV files -> SQLite database, queried read-only by the agent.

Safety layers for agent-written SQL (all of them apply):
  1. only statements starting with SELECT / WITH are accepted
  2. the database file is opened read-only (mode=ro) with PRAGMA query_only=ON
  3. a SQLite authorizer denies everything except reading (no ATTACH, PRAGMA, writes, ...)
  4. row limit, output size limit and a time limit
"""
from __future__ import annotations

import codecs
import csv
import os
import re
import sqlite3
import time
from pathlib import Path

from .config import settings

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
_LEADING_ZERO = re.compile(r"^[+-]?0\d+")  # "007", "01234" -> keep as text (ids, ZIP codes)
_META = "_meta"


class SqlError(Exception):
    """A query was rejected or failed. The message is safe to show to the LLM."""


# ----------------------------------------------------------------------------- CSV import


def _detect_encoding(path: Path) -> str:
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    try:
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"  # typical for CSVs exported from German Excel


def _sniff(path: Path, encoding: str) -> type[csv.Dialect]:
    with path.open("r", encoding=encoding, errors="replace", newline="") as f:
        sample = f.read(65536)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")  # type: ignore[return-value]
    except csv.Error:
        return csv.excel


def _clean_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = "c_" + cleaned
    return cleaned


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        n = seen.get(name, 0) + 1
        seen[name] = n
        out.append(name if n == 1 else f"{name}_{n}")
    return out


def _is_int(v: str) -> bool:
    return bool(_INT.match(v)) and not _LEADING_ZERO.match(v) and len(v.lstrip("+-")) <= 18


def _is_float(v: str) -> bool:
    return bool(_FLOAT.match(v)) and not _LEADING_ZERO.match(v)


def _import_csv(con: sqlite3.Connection, path: Path, used_names: set[str] | None = None) -> str | None:
    """Load one CSV into a new table. Returns the table name (None for an empty file)."""
    encoding = _detect_encoding(path)
    dialect = _sniff(path, encoding)

    def open_reader():
        f = path.open("r", encoding=encoding, errors="replace", newline="")
        return f, csv.reader(f, dialect)

    # pass 1: header + column type inference
    f, reader = open_reader()
    with f:
        header = next(reader, None)
        if not header:
            return None
        columns = _unique([_clean_name(h, f"col{i + 1}") for i, h in enumerate(header)])
        n = len(columns)
        can_int = [True] * n
        can_float = [True] * n
        seen = [False] * n
        for row in reader:
            for i in range(n):
                v = row[i].strip() if i < len(row) else ""
                if not v:
                    continue
                seen[i] = True
                if can_int[i] and not _is_int(v):
                    can_int[i] = False
                if can_float[i] and not _is_float(v):
                    can_float[i] = False
    types = [
        "INTEGER" if seen[i] and can_int[i] else "REAL" if seen[i] and can_float[i] else "TEXT"
        for i in range(n)
    ]

    table = _clean_name(path.stem, "table")
    if used_names is not None:
        base, k = table, 1
        while table in used_names or table == _META:
            k += 1
            table = f"{base}_{k}"
        used_names.add(table)

    column_sql = ", ".join(f'"{c}" {t}' for c, t in zip(columns, types))
    con.execute(f'CREATE TABLE "{table}" ({column_sql})')
    insert = f'INSERT INTO "{table}" VALUES ({",".join("?" * n)})'

    def convert(i: int, raw: str):
        v = raw.strip()
        if not v:
            return None
        if types[i] == "INTEGER":
            return int(v)
        if types[i] == "REAL":
            return float(v)
        return raw

    # pass 2: insert
    f, reader = open_reader()
    with f:
        next(reader, None)
        batch: list[tuple] = []
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            batch.append(tuple(convert(i, row[i] if i < len(row) else "") for i in range(n)))
            if len(batch) >= 5000:
                con.executemany(insert, batch)
                batch.clear()
        if batch:
            con.executemany(insert, batch)
    return table


def _csv_files() -> list[Path]:
    return sorted(settings.tables_dir.glob("*.csv")) if settings.tables_dir.is_dir() else []


def _signature(files: list[Path]) -> str:
    return "|".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files)


def build_db(force: bool = False) -> list[str]:
    """(Re)build the SQLite file from the CSVs if they changed. Returns the table names."""
    files = _csv_files()
    signature = _signature(files)
    db = settings.db_path
    db.parent.mkdir(parents=True, exist_ok=True)

    if db.exists() and not force:
        try:
            con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
            try:
                row = con.execute(f"SELECT signature FROM {_META}").fetchone()
            finally:
                con.close()
            if row and row[0] == signature:
                return list_tables()
        except sqlite3.Error:
            pass  # unreadable or old format -> rebuild

    tmp = db.with_name(db.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        used: set[str] = set()
        for path in files:
            _import_csv(con, path, used)
        con.execute(f"CREATE TABLE {_META} (signature TEXT)")
        con.execute(f"INSERT INTO {_META} VALUES (?)", (signature,))
        con.commit()
    finally:
        con.close()
    os.replace(tmp, db)
    return list_tables()


# ------------------------------------------------------------------------ read-only access


def _ro_connect() -> sqlite3.Connection:
    db = settings.db_path
    if not db.exists():
        raise SqlError("The database has not been built yet (no CSV files loaded).")
    con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
    con.execute("PRAGMA query_only=ON")
    return con


def list_tables() -> list[str]:
    con = _ro_connect()
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '\\_%' ESCAPE '\\' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def describe_schema(sample_rows: int = 2) -> str:
    """Human/LLM-readable schema with a couple of example rows per table."""
    try:
        con = _ro_connect()
    except SqlError:
        return "(no tables loaded)"
    try:
        lines = []
        for table in list_tables():
            cols = con.execute(f'PRAGMA table_info("{table}")').fetchall()
            n_rows = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            lines.append(f"Table {table} ({n_rows} rows): " + ", ".join(f"{c[1]} {c[2]}" for c in cols))
            for row in con.execute(f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}'):
                lines.append("  example row: " + str(row))
        return "\n".join(lines) if lines else "(no tables loaded)"
    finally:
        con.close()


def _authorizer(action, arg1, arg2, db_name, trigger):
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        getattr(sqlite3, "SQLITE_RECURSIVE", -1),
    }
    if action == sqlite3.SQLITE_READ and arg1 == _META:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def run_select(sql: str) -> tuple[list[str], list[tuple], bool]:
    """Run one read-only query. Returns (column names, rows, truncated)."""
    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise SqlError("Empty query.")
    first_word = statement.split(None, 1)[0].lower()
    if first_word not in ("select", "with"):
        raise SqlError("Only SELECT queries are allowed.")

    con = _ro_connect()
    deadline = time.monotonic() + settings.sql_timeout
    con.set_authorizer(_authorizer)
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cursor = con.execute(statement)
        rows = cursor.fetchmany(settings.sql_max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except sqlite3.Error as exc:
        raise SqlError(f"SQL error: {exc}") from exc
    finally:
        con.close()

    truncated = len(rows) > settings.sql_max_rows
    return columns, rows[: settings.sql_max_rows], truncated


def format_result(columns: list[str], rows: list[tuple], truncated: bool, max_chars: int = 8000) -> str:
    if not columns:
        return "Query ran but returned no columns."
    if not rows:
        return "No rows returned.\ncolumns: " + " | ".join(columns)
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (output cut off)"
    if truncated:
        text += f"\n(only the first {len(rows)} rows are shown - refine the query with WHERE / GROUP BY / LIMIT)"
    return text
