"""Regression tests for kernel lifecycle & concurrency (Domain A).

Each test is written to FAIL against the pre-fix code:

  A1 warm-path timeout: execute_in_kernel must bound the kernel-side eval with
     TimeConstrained and map the sentinel to a timed_out dict (not report it as a
     successful "$Aborted" string result).
  A2 startup reaping / cold-fallback recovery: the idle reaper must not terminate
     a session mid-startup, and close_kernel_session() must clear the permanent
     cold-fallback flag so a restart retries the warm session.
  A3 activity stamping: idle is measured from eval completion, not eval start, so
     a long single eval isn't reaped immediately after finishing.

Uses the fake-session pattern from tests/test_warm_funnel.py with no live kernel.
"""

from __future__ import annotations

import pytest

import mathematica_mcp.cache as C
from mathematica_mcp import session as S


class _FakeSession:
    """Returns a fixed value from evaluate(), recording the last expr passed."""

    def __init__(self, ret):
        self._ret = ret
        self.calls = 0
        self.last_expr = None

    def evaluate(self, expr):
        self.calls += 1
        self.last_expr = expr
        return self._ret


class _FakeKernel:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    """Force cache misses / no-op puts so execute_in_kernel always hits the kernel."""
    monkeypatch.setattr(C._query_cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(C._query_cache, "put", lambda *a, **k: None)


# ---- A1: warm-path timeout is honored --------------------------------------


def test_execute_in_kernel_warm_timeout_returns_timed_out(monkeypatch):
    # The kernel returns the TimeConstrained sentinel string on timeout.
    fake = _FakeSession("$Aborted")
    monkeypatch.setattr(S, "get_kernel_session", lambda: fake)
    monkeypatch.setattr(S, "_use_wolframscript", False)

    result = S.execute_in_kernel("Pause[999]", render_graphics=False, timeout=7)

    # Pre-fix ignored `timeout` and reported the sentinel as a success string.
    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["error"] == "timeout"
    assert result["execution_method"] == "wolframclient"
    # The timeout param must actually reach the kernel-side guard.
    code = str(fake.last_expr)
    assert "TimeConstrained" in code
    assert "7" in code


# ---- A2b: reaper does not reap during startup or mid-eval ------------------


def test_reaper_skips_during_startup(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "100")
    fake = _FakeKernel()
    monkeypatch.setattr(S, "_kernel_session", fake)
    monkeypatch.setattr(S, "_last_activity", 0.0)
    # raising=False so this also runs against pre-fix code (which lacked the flag).
    monkeypatch.setattr(S, "_session_starting", True, raising=False)

    assert S._maybe_reap_idle_kernel(now=10_000.0) is False
    assert fake.terminated is False


def test_reaper_skips_when_eval_in_flight(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "100")
    fake = _FakeKernel()
    monkeypatch.setattr(S, "_kernel_session", fake)
    monkeypatch.setattr(S, "_last_activity", 0.0)

    assert S._session_eval_lock.acquire(blocking=False)
    try:
        assert S._maybe_reap_idle_kernel(now=10_000.0) is False
        assert fake.terminated is False
    finally:
        S._session_eval_lock.release()


# ---- A2c: close_kernel_session recovers from permanent cold fallback --------


def test_close_resets_cold_fallback(monkeypatch):
    # The stuck state: fell into cold fallback, kernel already gone.
    monkeypatch.setattr(S, "_use_wolframscript", True)
    monkeypatch.setattr(S, "_kernel_session", None)

    S.close_kernel_session()

    # Pre-fix left _use_wolframscript=True forever -> permanent cold fallback.
    assert S._use_wolframscript is False


# ---- A3: activity stamped on eval completion, not just start ---------------


def test_activity_stamped_on_warm_evaluate_wl(monkeypatch):
    fake = _FakeSession("42")
    # Mocked get_kernel_session does NOT stamp activity, so only an end-of-eval
    # stamp can move _last_activity off its start value.
    monkeypatch.setattr(S, "get_kernel_session", lambda: fake)
    monkeypatch.setattr(S, "_last_activity", 0.0)

    result = S.evaluate_wl("6*7")

    assert result.success is True
    assert result.execution_method == "wolframclient"
    assert S._last_activity > 0.0


def test_activity_stamped_on_execute_in_kernel_success(monkeypatch):
    fake = _FakeSession(
        {
            "inputform": "42",
            "fullform": "",
            "tex": "",
            "messages": "{}",
            "is_graphics": False,
            "image_path": "",
        }
    )
    monkeypatch.setattr(S, "get_kernel_session", lambda: fake)
    monkeypatch.setattr(S, "_use_wolframscript", False)
    monkeypatch.setattr(S, "_last_activity", 0.0)

    result = S.execute_in_kernel("6*7", render_graphics=False)

    assert result["success"] is True
    assert result["execution_method"] == "wolframclient"
    assert S._last_activity > 0.0
