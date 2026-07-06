"""Opt-in screenshot cache (notebook/cell scope).

Two layers under test:
- connection.send_command bumps the notebook mutation epoch after a successful
  notebook-mutating command, and only then;
- the lean screenshot(cache=True) tool serves notebook/cell PNGs from an
  epoch-keyed LRU, calling through only on a miss.

All mocked, no Mathematica required.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp.server.fastmcp import Image

import mathematica_mcp.cache as cache
from mathematica_mcp import server as srv
from mathematica_mcp.connection import MathematicaConnection


@pytest.fixture
def reset_cache():
    cache._notebook_epoch = 0
    cache._screenshot_cache.clear()
    yield
    cache._notebook_epoch = 0
    cache._screenshot_cache.clear()


# ---------------------------------------------------------------------------
# Epoch bump at the send_command chokepoint
# ---------------------------------------------------------------------------


class _ReplaySocket:
    """Minimal socket that replays one framed JSON response."""

    def __init__(self, frame: bytes):
        self._frame = frame
        self.closed = False

    def settimeout(self, _timeout):
        pass

    def sendall(self, _data):
        pass

    def recv(self, _bufsize):
        frame, self._frame = self._frame, b""
        return frame

    def getpeername(self):
        return ("localhost", 9881)

    def close(self):
        self.closed = True


def _conn_for(response: dict) -> MathematicaConnection:
    conn = MathematicaConnection()
    conn._socket = _ReplaySocket(json.dumps(response).encode() + b"\n")
    return conn


def test_mutating_command_bumps_epoch(reset_cache):
    before = cache.get_notebook_epoch()
    _conn_for({"status": "ok", "result": {"success": True}}).send_command("write_cell", {})
    assert cache.get_notebook_epoch() == before + 1


def test_batch_commands_bumps_epoch(reset_cache):
    before = cache.get_notebook_epoch()
    _conn_for({"status": "ok", "result": {"success": True}}).send_command("batch_commands", {})
    assert cache.get_notebook_epoch() == before + 1


def test_ping_does_not_bump(reset_cache):
    before = cache.get_notebook_epoch()
    _conn_for({"status": "ok", "result": {"pong": True}}).send_command("ping", {})
    assert cache.get_notebook_epoch() == before


def test_execute_code_does_not_bump(reset_cache):
    before = cache.get_notebook_epoch()
    _conn_for({"status": "ok", "result": {"success": True}}).send_command("execute_code", {})
    assert cache.get_notebook_epoch() == before


def test_read_with_state_delta_does_not_bump(reset_cache):
    # get_notebook_info carries a state_delta but is a read; keying off the
    # command name (not state_delta presence) keeps it from bumping.
    before = cache.get_notebook_epoch()
    _conn_for({"status": "ok", "result": {"success": True}, "state_delta": {"notebook": "nb"}}).send_command(
        "get_notebook_info", {}
    )
    assert cache.get_notebook_epoch() == before


def test_failed_mutating_command_does_not_bump(reset_cache):
    before = cache.get_notebook_epoch()
    with pytest.raises(RuntimeError):
        _conn_for({"status": "error", "message": "boom"}).send_command("write_cell", {})
    assert cache.get_notebook_epoch() == before


# ---------------------------------------------------------------------------
# Screenshot cache behaviour through the lean tool
# ---------------------------------------------------------------------------


def _counter_fake(monkeypatch, name: str) -> dict:
    """Replace srv.<name> with an async producer that returns a distinct PNG
    each call (so a stale hit is detectable) and counts invocations."""
    state = {"calls": 0}

    async def fake(*_args, **_kwargs):
        state["calls"] += 1
        return Image(data=b"PNG" + bytes([state["calls"]]), format="png")

    monkeypatch.setattr(srv, name, fake)
    return state


def test_cache_hit_returns_identical_bytes_without_calling_through(reset_cache, monkeypatch):
    # Explicit notebook id is a stable cache target (see the focused-bypass tests
    # below for why notebook=None is not).
    state = _counter_fake(monkeypatch, "screenshot_notebook")
    img1 = asyncio.run(srv.screenshot(scope="notebook", notebook="nbA", cache=True))
    img2 = asyncio.run(srv.screenshot(scope="notebook", notebook="nbA", cache=True))
    assert state["calls"] == 1
    assert img1.data == img2.data


def test_cell_scope_caches(reset_cache, monkeypatch):
    state = _counter_fake(monkeypatch, "screenshot_cell")
    img1 = asyncio.run(srv.screenshot(scope="cell", cell_id="c1", cache=True))
    img2 = asyncio.run(srv.screenshot(scope="cell", cell_id="c1", cache=True))
    assert state["calls"] == 1
    assert img1.data == img2.data


def test_mutation_forces_fresh_capture(reset_cache, monkeypatch):
    state = _counter_fake(monkeypatch, "screenshot_notebook")
    img1 = asyncio.run(srv.screenshot(scope="notebook", notebook="nbA", cache=True))
    cache.bump_notebook_epoch()
    img2 = asyncio.run(srv.screenshot(scope="notebook", notebook="nbA", cache=True))
    assert state["calls"] == 2
    assert img1.data != img2.data


def test_cache_false_never_touches_cache(reset_cache, monkeypatch):
    state = _counter_fake(monkeypatch, "screenshot_notebook")
    asyncio.run(srv.screenshot(scope="notebook"))
    asyncio.run(srv.screenshot(scope="notebook"))
    assert state["calls"] == 2
    assert len(cache._screenshot_cache) == 0


def test_focused_notebook_bypasses_cache(reset_cache, monkeypatch):
    # notebook=None + session_id=None resolves the "focused" notebook via
    # $MCPActiveNotebook, which reads silently repoint without bumping the epoch.
    # The unstable identity must never be cached: call through every time.
    state = _counter_fake(monkeypatch, "screenshot_notebook")
    img1 = asyncio.run(srv.screenshot(scope="notebook", cache=True))
    img2 = asyncio.run(srv.screenshot(scope="notebook", cache=True))
    assert state["calls"] == 2
    assert len(cache._screenshot_cache) == 0
    assert img1.data != img2.data


def test_session_pinned_notebook_caches(reset_cache, monkeypatch):
    # A session_id pins a stable notebook (resolveNotebook returns it directly),
    # so caching stays on even with notebook=None.
    state = _counter_fake(monkeypatch, "screenshot_notebook")
    img1 = asyncio.run(srv.screenshot(scope="notebook", session_id="s1", cache=True))
    img2 = asyncio.run(srv.screenshot(scope="notebook", session_id="s1", cache=True))
    assert state["calls"] == 1
    assert img1.data == img2.data


def test_distinct_notebooks_do_not_collide(reset_cache, monkeypatch):
    # Two explicit notebook ids get distinct keys: one never serves the other's
    # PNG, and each caches independently.
    state = _counter_fake(monkeypatch, "screenshot_notebook")
    a = asyncio.run(srv.screenshot(scope="notebook", notebook="nbA", cache=True))
    b = asyncio.run(srv.screenshot(scope="notebook", notebook="nbB", cache=True))
    assert state["calls"] == 2
    assert a.data != b.data
    a_again = asyncio.run(srv.screenshot(scope="notebook", notebook="nbA", cache=True))
    assert state["calls"] == 2  # nbA hit, not a call-through
    assert a_again.data == a.data


# ---------------------------------------------------------------------------
# LRU mechanics
# ---------------------------------------------------------------------------


def test_lru_eviction_at_capacity(reset_cache):
    cap = cache._MAX_SCREENSHOT_ENTRIES
    for i in range(cap + 3):
        cache.put_cached_screenshot((i,), bytes([i % 256]))
    assert len(cache._screenshot_cache) == cap
    assert cache.get_cached_screenshot((0,)) is None  # oldest evicted
    assert cache.get_cached_screenshot((cap + 2,)) is not None  # newest kept


def test_lru_touch_on_get_keeps_entry_alive(reset_cache):
    cap = cache._MAX_SCREENSHOT_ENTRIES
    for i in range(cap):
        cache.put_cached_screenshot((i,), bytes([i]))
    cache.get_cached_screenshot((0,))  # promote oldest to most-recent
    cache.put_cached_screenshot((99,), b"z")  # evicts key 1, not key 0
    assert cache.get_cached_screenshot((0,)) is not None
    assert cache.get_cached_screenshot((1,)) is None
