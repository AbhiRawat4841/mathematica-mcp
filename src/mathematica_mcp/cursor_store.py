"""Continuation-cursor store for oversized tool output (plan §3.7).

When a tool's response exceeds the character cap, the full text is stored here
keyed by a content-derived token and the tool returns a first page plus a
``next_cursor``. The client fetches the remainder by passing ``cursor=`` back to
the same tool. Bounded, in-memory, thread-safe; cursors are best-effort and may
expire under memory pressure.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any

_MAX_ENTRIES = 32
_lock = threading.Lock()
_store: OrderedDict[str, str] = OrderedDict()


def default_page_size() -> int:
    """Hard output cap in characters (env MATHEMATICA_MAX_OUTPUT_CHARS, default 4000)."""
    try:
        return max(500, int(os.environ.get("MATHEMATICA_MAX_OUTPUT_CHARS", "4000")))
    except (TypeError, ValueError):
        return 4000


def _token(text: str) -> str:
    return "cur" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def store(text: str) -> str:
    """Store *text* and return its cursor token (content-derived, so identical
    text reuses the same slot)."""
    token = _token(text)
    with _lock:
        _store[token] = text
        _store.move_to_end(token)
        while len(_store) > _MAX_ENTRIES:
            _store.popitem(last=False)
    return token


def page(cursor: str, page_size: int | None = None) -> dict[str, Any] | None:
    """Return the page at *cursor* (``token`` or ``token:offset``), or None if the
    cursor is unknown/expired."""
    size = page_size or default_page_size()
    token, _, off = cursor.partition(":")
    try:
        offset = int(off) if off else 0
    except ValueError:
        return None
    with _lock:
        text = _store.get(token)
        if text is not None:
            _store.move_to_end(token)
    if text is None:
        return None
    chunk = text[offset : offset + size]
    nxt = offset + size
    return {
        "chunk": chunk,
        "offset": offset,
        "total_length": len(text),
        "next_cursor": f"{token}:{nxt}" if nxt < len(text) else None,
    }


def paginate(text: str, page_size: int | None = None) -> tuple[str, dict[str, Any] | None]:
    """If *text* fits in one page, return (text, None). Otherwise store it and
    return (first_page, cursor_info) where cursor_info has next_cursor/total."""
    size = page_size or default_page_size()
    if len(text) <= size:
        return text, None
    token = store(text)
    return text[:size], {
        "next_cursor": f"{token}:{size}",
        "total_length": len(text),
        "returned": size,
    }


def clear() -> None:
    with _lock:
        _store.clear()
