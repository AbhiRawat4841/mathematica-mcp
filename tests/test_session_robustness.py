"""Regression tests for session.py robustness (Domain A). No live kernel.

Each test FAILS against the pre-fix code:

  A1 warm cache-hit rasterization: a query-cache hit with a raster-cache miss
     must rasterize the CACHED output expression via evaluate_wl (warm funnel),
     not spawn a cold wolframscript re-run of the user's code. Only placeholder
     text ("-Graphics-", ...) may fall back to the cold re-run, which is counted.
  A2 health-check ping race: get_kernel_session must not ping outside
     _session_eval_lock; if the lock is held an eval is in flight => alive =>
     skip the ping entirely.
  A3 process-exit cleanup: a shutdown handler is registered and close is
     idempotent.
"""

from __future__ import annotations

import re
import threading

import pytest

import mathematica_mcp.cache as C
import mathematica_mcp.lazy_wolfram_tools as LW
from mathematica_mcp import session as S


class _FakeSession:
    """Returns a fixed value from evaluate(), recording calls."""

    def __init__(self, ret=1):
        self._ret = ret
        self.calls = 0

    def evaluate(self, expr):
        self.calls += 1
        return self._ret

    def terminate(self):
        pass


@pytest.fixture(autouse=True)
def _clean_caches():
    S.clear_raster_cache()
    yield
    S.clear_raster_cache()


# ---- A1: warm cache-hit rasterization ---------------------------------------


def _cached_graphics_entry(text: str):
    return {
        "success": True,
        "output": text,
        "output_inputform": text,
        "output_fullform": "",
        "output_tex": "",
        "warnings": [],
        "timing_ms": 5,
        "execution_method": "wolframclient",
    }


def test_cache_hit_rasterizes_cached_expression_warm(monkeypatch):
    graphics_text = "Graphics[{Disk[]}]"
    monkeypatch.setattr(C._query_cache, "get", lambda *a, **k: _cached_graphics_entry(graphics_text))

    seen = {}

    def fake_evaluate_wl(code, timeout=60):
        seen["code"] = code
        m = re.search(r'Export\["([^"]+)"', code)
        assert m is not None, f"no Export path in generated WL: {code[:200]}"
        with open(m.group(1), "wb") as f:
            f.write(b"fake png bytes")
        return S.WLResult(text="success", success=True, execution_method="wolframclient")

    monkeypatch.setattr(S, "evaluate_wl", fake_evaluate_wl)

    def _no_cold(*a, **k):
        raise AssertionError("cold wolframscript re-run must not happen for rasterizable cached text")

    monkeypatch.setattr(S, "_rasterize_via_wolframscript", _no_cold)

    result = S.execute_in_kernel("Plot[Sin[x], {x, 0, 1}] (* warm-raster test *)")

    assert result["from_cache"] is True
    assert result["is_graphics"] is True
    assert result["image_path"]
    # The warm snippet rasterizes the cached expression text, not the user code.
    code = seen["code"]
    assert "Rasterize" in code
    assert "Export[" in code
    assert "ToExpression" in code
    assert "MCPScratch`" in code
    assert graphics_text in code
    assert "Plot[Sin[x]" not in code  # never re-runs the original code warm


def test_cache_hit_placeholder_falls_back_cold(monkeypatch):
    monkeypatch.setattr(C._query_cache, "get", lambda *a, **k: _cached_graphics_entry("-Graphics-"))

    def _no_warm(*a, **k):
        raise AssertionError("placeholder text is not rasterizable warm")

    monkeypatch.setattr(S, "evaluate_wl", _no_warm)

    cold_calls = []

    def fake_cold(code, image_size=500):
        cold_calls.append(code)
        return "/tmp/fake_raster.png"

    monkeypatch.setattr(S, "_rasterize_via_wolframscript", fake_cold)
    # _put_cached_raster would reject a nonexistent path only on the read side;
    # keep the raster cache out of the assertion by checking the response.
    result = S.execute_in_kernel("Plot[Cos[x], {x, 0, 1}] (* placeholder test *)")

    assert result["from_cache"] is True
    assert result["is_graphics"] is True
    assert cold_calls and "Plot[Cos[x]" in cold_calls[0]


def test_cold_rasterize_is_counted(monkeypatch):
    """_rasterize_via_wolframscript spawns a cold subprocess: it must be counted."""

    class _Proc:
        returncode = 0
        stdout = "not_graphics"
        stderr = ""

    monkeypatch.setattr(LW, "_find_wolframscript", lambda: "/usr/bin/wolframscript")
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: _Proc())

    S.reset_cold_execution_count()
    S._rasterize_via_wolframscript("Graphics[{}]")
    assert S.cold_execution_count() == 1


# ---- A2: health-check ping honors the eval lock ------------------------------


def test_ping_skipped_when_eval_lock_held(monkeypatch):
    fake = _FakeSession()
    monkeypatch.setattr(S, "_use_wolframscript", False)
    monkeypatch.setattr(S, "_kernel_session", fake)
    monkeypatch.setattr(S, "_last_kernel_health_check", 0.0)  # stale => ping due

    assert S._session_eval_lock.acquire(blocking=False)
    try:
        got = S.get_kernel_session()
    finally:
        S._session_eval_lock.release()

    assert got is fake
    # Pre-fix: pinged concurrently with the in-flight eval (calls == 1).
    assert fake.calls == 0


def test_ping_runs_under_eval_lock_when_free(monkeypatch):
    class _LockCheckingSession:
        def __init__(self):
            self.pinged_under_lock = None

        def evaluate(self, expr):
            self.pinged_under_lock = S._session_eval_lock.locked()
            return 1

        def terminate(self):
            pass

    fake = _LockCheckingSession()
    monkeypatch.setattr(S, "_use_wolframscript", False)
    monkeypatch.setattr(S, "_kernel_session", fake)
    monkeypatch.setattr(S, "_last_kernel_health_check", 0.0)

    got = S.get_kernel_session()

    assert got is fake
    assert fake.pinged_under_lock is True  # pre-fix: pinged with the lock free
    assert not S._session_eval_lock.locked()  # released afterwards


def test_failed_ping_releases_lock(monkeypatch):
    class _DeadSession:
        def __init__(self):
            self.terminated = False

        def evaluate(self, expr):
            raise RuntimeError("WSTP link dead")

        def terminate(self):
            self.terminated = True

    dead = _DeadSession()
    monkeypatch.setattr(S, "_use_wolframscript", False)
    monkeypatch.setattr(S, "_kernel_session", dead)
    monkeypatch.setattr(S, "_last_kernel_health_check", 0.0)
    monkeypatch.delenv("MATHEMATICA_KERNEL_PATH", raising=False)
    monkeypatch.setattr(S, "find_wolfram_kernel", lambda: None)  # no recreate

    got = S.get_kernel_session()

    assert got is None
    assert dead.terminated
    assert not S._session_eval_lock.locked()


# ---- A3: process-exit cleanup ------------------------------------------------


def test_shutdown_handler_registered():
    # Registered via threading._register_atexit (runs before non-daemon thread
    # join, unlike plain atexit) — entries are functools.partial wrappers.
    registry = getattr(threading, "_threading_atexits", [])
    assert any(getattr(cb, "func", cb) is S._shutdown_at_exit for cb in registry)


def test_shutdown_at_exit_closes_kernel_and_is_idempotent(monkeypatch):
    class _FakeKernel:
        def __init__(self):
            self.terminated = 0

        def terminate(self):
            self.terminated += 1

    fake = _FakeKernel()
    monkeypatch.setattr(S, "_use_wolframscript", False)
    monkeypatch.setattr(S, "_kernel_session", fake)

    S._shutdown_at_exit()
    assert fake.terminated == 1
    assert S._kernel_session is None

    S._shutdown_at_exit()  # double invocation is a no-op
    assert fake.terminated == 1
    assert S._kernel_session is None
