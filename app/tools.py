"""The tool logic: search_docs, query_database, calculator, get_current_time.

Plain Python functions that return strings and never raise - errors come back as text so the LLM
can read them and correct itself (e.g. fix a bad SQL query). server.py exposes them over MCP.
"""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from . import tables, vectorstore
from .config import settings

# ------------------------------------------------------------------------------ search_docs


def search_docs(query: str) -> str:
    try:
        hits = vectorstore.hybrid_search(query, settings.top_k)
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the LLM as text
        return f"Search failed: {exc}. (Has the document index been created with `python -m app.ingest`?)"
    if not hits:
        return "No matching passages found. Try different keywords."
    blocks = [f"[{i}] source: {h['source']} (part {h['chunk']})\n{h['text']}" for i, h in enumerate(hits, 1)]
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- query_database


def query_database(sql: str) -> str:
    try:
        columns, rows, truncated = tables.run_select(sql)
    except tables.SqlError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"
    return tables.format_result(columns, rows, truncated)


# ------------------------------------------------------------------------------- calculator

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str):
    """Evaluate arithmetic without eval(): only numbers, operators, and whitelisted functions."""
    expression = expression.strip().replace("^", "**")
    if not expression:
        raise ValueError("empty expression")
    if len(expression) > 300:
        raise ValueError("expression too long")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow):
                if isinstance(left, int) and isinstance(right, int) and right > 0:
                    if left.bit_length() * right > 10_000:
                        raise ValueError("number too large")
                elif abs(right) > 1000:
                    raise ValueError("exponent too large")
            return _BINARY[type(node.op)](left, right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("unsupported expression")

    return ev(ast.parse(expression, mode="eval"))


def calculator(expression: str) -> str:
    try:
        value = safe_eval(expression)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


# --------------------------------------------------------------------------------------- time


def get_current_time(timezone: str = "") -> str:
    name = timezone.strip() or settings.timezone
    try:
        now = datetime.now(ZoneInfo(name))
    except Exception:  # noqa: BLE001
        return f"Unknown timezone '{name}'. Use an IANA name such as Europe/Berlin or UTC."
    return f"{now:%A}, {now:%Y-%m-%d} {now:%H:%M:%S} ({name}, UTC{now:%z}) | ISO: {now.isoformat(timespec='seconds')}"
