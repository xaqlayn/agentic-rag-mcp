"""Text helpers: chunking, tokenizing and BM25-style sparse vectors for keyword search."""
from __future__ import annotations

import re
import zlib
from collections import Counter

# ----------------------------------------------------------------------------- chunking


def _split_long(paragraph: str, size: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries (hard split as last resort)."""
    pieces: list[str] = []
    cur = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > size:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sentence[:size])
            sentence = sentence[size:]
        if not cur:
            cur = sentence
        elif len(cur) + 1 + len(sentence) <= size:
            cur += " " + sentence
        else:
            pieces.append(cur)
            cur = sentence
    if cur:
        pieces.append(cur)
    return pieces


def _tail(text: str, n: int) -> str:
    """Last n characters of text, starting at a word boundary."""
    tail = text[-n:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    """Paragraph-aware chunking. Chunks are ~size characters (soft limit) and overlap a little."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        units.extend([para] if len(para) <= size else _split_long(para, size))

    chunks: list[str] = []
    cur = ""
    for unit in units:
        if not cur:
            cur = unit
        elif len(cur) + 2 + len(unit) <= size:
            cur = f"{cur}\n\n{unit}"
        else:
            chunks.append(cur)
            cur = f"{_tail(cur, overlap)}\n\n{unit}" if overlap > 0 else unit
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------- sparse (keyword) side

_WORD = re.compile(r"\w+", re.UNICODE)

_STOP = set(
    """
    a an and are as at be but by for if in into is it no not of on or such that the their then
    there these they this to was will with
    der die das den dem des ein eine einer einen einem eines und oder aber ist sind war waren
    wird werden wurde nicht von zu mit auf für im am an als bei auch es sich dass wie
    """.split()
)

# BM25 term-frequency saturation. IDF is applied by Qdrant (Modifier.IDF on the sparse vector).
_K1 = 1.2
_B = 0.75
_AVG_LEN = 120.0  # assumed average chunk length in tokens


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in _STOP]


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))  # unsigned 32-bit, fits Qdrant sparse indices


def sparse_doc(text: str) -> tuple[list[int], list[float]]:
    """Sparse vector for a document chunk: BM25 term weights (without IDF)."""
    tokens = tokenize(text)
    doc_len = max(len(tokens), 1)
    acc: dict[int, float] = {}
    for token, tf in Counter(tokens).items():
        weight = tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * doc_len / _AVG_LEN))
        idx = _index(token)
        acc[idx] = acc.get(idx, 0.0) + weight
    return list(acc.keys()), list(acc.values())


def sparse_query(text: str) -> tuple[list[int], list[float]]:
    """Sparse vector for a query: weight 1 per distinct term."""
    acc = {_index(t): 1.0 for t in set(tokenize(text))}
    return list(acc.keys()), list(acc.values())
