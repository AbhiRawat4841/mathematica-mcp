"""Continuation-cursor store (plan §3.7): paginate, fetch, reassemble, evict."""

from __future__ import annotations

import json

import pytest

from mathematica_mcp import cursor_store as cs
from mathematica_mcp import server as srv


@pytest.fixture(autouse=True)
def _clear_store():
    cs.clear()
    yield
    cs.clear()


def test_small_text_is_not_paginated():
    first, info = cs.paginate("short", page_size=100)
    assert first == "short"
    assert info is None


def test_large_text_paginates_and_reassembles():
    text = "".join(str(i % 10) for i in range(10_000))
    first, info = cs.paginate(text, page_size=4000)
    assert info is not None
    assert first == text[:4000]

    collected = first
    cursor = info["next_cursor"]
    guard = 0
    while cursor and guard < 100:
        pg = cs.page(cursor, page_size=4000)
        assert pg is not None
        collected += pg["chunk"]
        cursor = pg["next_cursor"]
        guard += 1

    assert collected == text


def test_unknown_cursor_returns_none():
    assert cs.page("curdeadbeef:0") is None


def test_bad_offset_returns_none():
    tok = cs.store("hello world")
    assert cs.page(f"{tok}:notanumber") is None


def test_lru_eviction():
    tokens = [cs.store(f"payload-number-{i}-" + "x" * 50) for i in range(cs._MAX_ENTRIES + 5)]
    # The earliest few should have been evicted.
    assert cs.page(tokens[0]) is None
    assert cs.page(tokens[-1]) is not None


def test_default_page_size_env(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_MAX_OUTPUT_CHARS", "1234")
    assert cs.default_page_size() == 1234
    monkeypatch.setenv("MATHEMATICA_MAX_OUTPUT_CHARS", "bogus")
    assert cs.default_page_size() == 4000


# ---- lean-tool integration -------------------------------------------------


async def test_evaluate_cursor_unknown():
    out = json.loads(await srv.evaluate(cursor="curmissing:0"))
    assert out["success"] is False


async def test_evaluate_truncates_and_pages(monkeypatch):
    big = "R" * 12_000

    async def fake_execute(*a, **k):
        return big

    monkeypatch.setenv("MATHEMATICA_MAX_OUTPUT_CHARS", "4000")
    monkeypatch.setattr(srv, "execute_code", fake_execute)

    out = json.loads(await srv.evaluate(code="x"))
    assert out["truncated"] is True
    assert out["total_length"] == 12_000
    assert out["preview"] == big[:4000]

    # Fetch the remainder via the returned cursor.
    rest = ""
    cursor = out["next_cursor"]
    while cursor:
        page = json.loads(await srv.evaluate(cursor=cursor))
        rest += page["chunk"]
        cursor = page["next_cursor"]
    assert out["preview"] + rest == big


# ---- kernel/vars/cells pagination (Codex finding: 31KB kernel-state responses
# went to the agent unpaginated; only evaluate/read_notebook_file were capped) --


async def test_kernel_state_truncates_and_pages(monkeypatch):
    big = json.dumps({"success": True, "blob": "K" * 12_000})

    async def fake_state(*a, **k):
        return big

    monkeypatch.setenv("MATHEMATICA_MAX_OUTPUT_CHARS", "4000")
    monkeypatch.setattr(srv, "get_kernel_state", fake_state)

    out = json.loads(await srv.kernel(action="state"))
    assert out["truncated"] is True
    assert out["total_length"] == len(big)

    rest = ""
    cursor = out["next_cursor"]
    while cursor:
        page = json.loads(await srv.kernel(cursor=cursor))
        rest += page["chunk"]
        cursor = page["next_cursor"]
    assert out["preview"] + rest == big


async def test_vars_list_truncates(monkeypatch):
    big = "V" * 9_000

    async def fake_list(*a, **k):
        return big

    monkeypatch.setenv("MATHEMATICA_MAX_OUTPUT_CHARS", "4000")
    monkeypatch.setattr(srv, "list_variables", fake_list)

    out = json.loads(await srv.vars(action="list"))
    assert out["truncated"] is True
    assert out["next_cursor"]


async def test_cells_list_truncates(monkeypatch):
    big = "C" * 9_000

    async def fake_cells(*a, **k):
        return big

    monkeypatch.setenv("MATHEMATICA_MAX_OUTPUT_CHARS", "4000")
    monkeypatch.setattr(srv, "get_cells", fake_cells)

    out = json.loads(await srv.cells(action="list"))
    assert out["truncated"] is True
    assert out["next_cursor"]


async def test_kernel_small_output_passes_through(monkeypatch):
    small = json.dumps({"success": True, "count": 2})

    async def fake_pkgs(*a, **k):
        return small

    monkeypatch.setattr(srv, "list_loaded_packages", fake_pkgs)
    out = json.loads(await srv.kernel(action="packages"))
    assert out == {"success": True, "count": 2}
