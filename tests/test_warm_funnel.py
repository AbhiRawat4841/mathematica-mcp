"""P1 warm-funnel: evaluate_wl warm/cold routing, cold counter, parity.

Mocked tests run everywhere; the live tests (``wolfram_runtime``) assert that the
warm persistent session and the cold wolframscript subprocess parse to the same
result, and that the warm happy path spawns zero cold subprocesses.
"""

from __future__ import annotations

import json
import subprocess
import types

import pytest

from mathematica_mcp import session as S


def _kernel_version_ok(version: str | None) -> bool:
    """A live kernel reports a positive version float; 14.x stays supported.

    Replaces the old ``startswith("15")`` check, which wrongly rejected the still
    supported 14.x kernels (docs/roadmaps/v1.0-lean-plan-v2.md).
    """
    if version is None:
        return False
    try:
        return float(version) >= 13.0
    except (TypeError, ValueError):
        return False


@pytest.fixture(autouse=True)
def _reset_cold_counter():
    S.reset_cold_execution_count()
    yield
    S.reset_cold_execution_count()


class _FakeSession:
    def __init__(self, ret):
        self._ret = ret
        self.calls = 0

    def evaluate(self, _expr):
        self.calls += 1
        return self._ret


# ---- warm path -------------------------------------------------------------


def test_warm_success(monkeypatch):
    fake = _FakeSession("<|success -> True, x -> 3|>")
    monkeypatch.setattr(S, "get_kernel_session", lambda: fake)

    result = S.evaluate_wl("<|x -> 3|>")

    assert result.success is True
    assert result.execution_method == "wolframclient"
    assert result.text == "<|success -> True, x -> 3|>"
    assert S.cold_execution_count() == 0  # warm path spawns no cold subprocess
    assert fake.calls == 1


def test_warm_timeout_maps_to_timed_out(monkeypatch):
    monkeypatch.setattr(S, "get_kernel_session", lambda: _FakeSession("$Aborted"))

    result = S.evaluate_wl("Pause[999]", timeout=1)

    assert result.success is False
    assert result.timed_out is True
    assert result.execution_method == "wolframclient"
    assert S.cold_execution_count() == 0


def test_warm_exception_falls_back_to_cold(monkeypatch):
    class _Boom(_FakeSession):
        def evaluate(self, _expr):
            raise RuntimeError("kernel died")

    monkeypatch.setattr(S, "get_kernel_session", lambda: _Boom(None))
    monkeypatch.setattr("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", lambda: "/fake/wolframscript")
    monkeypatch.setattr(
        S.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="<|success -> True|>", stderr=""),
    )

    result = S.evaluate_wl("<|x -> 1|>")

    assert result.success is True
    assert result.execution_method == "wolframscript"
    assert S.cold_execution_count() == 1  # counted the cold fallback


# ---- cold path -------------------------------------------------------------


def test_cold_when_no_session(monkeypatch):
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    monkeypatch.setattr("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", lambda: "/fake/wolframscript")
    monkeypatch.setattr(
        S.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="42", stderr=""),
    )

    result = S.evaluate_wl("6*7")

    assert result.success is True
    assert result.text == "42"
    assert result.execution_method == "wolframscript"
    assert S.cold_execution_count() == 1


def test_cold_nonzero_exit_is_failure(monkeypatch):
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    monkeypatch.setattr("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", lambda: "/fake/wolframscript")
    monkeypatch.setattr(
        S.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="Syntax error"),
    )

    result = S.evaluate_wl("Integrate[")

    assert result.success is False
    assert "Syntax error" in result.error


def test_cold_timeout(monkeypatch):
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    monkeypatch.setattr("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", lambda: "/fake/wolframscript")

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="wolframscript", timeout=1)

    monkeypatch.setattr(S.subprocess, "run", _raise)

    result = S.evaluate_wl("Pause[999]", timeout=1)

    assert result.success is False
    assert result.timed_out is True


def test_cold_no_wolframscript(monkeypatch):
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    monkeypatch.setattr("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", lambda: None)

    result = S.evaluate_wl("1+1")

    assert result.success is False
    assert result.execution_method == "none"
    assert S.cold_execution_count() == 0  # never spawned


# ---- live parity -----------------------------------------------------------


# ---- idle kernel shutdown --------------------------------------------------


class _FakeKernel:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


def test_idle_reaper_closes_idle_kernel(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "100")
    fake = _FakeKernel()
    monkeypatch.setattr(S, "_kernel_session", fake)
    monkeypatch.setattr(S, "_last_activity", 0.0)

    assert S._maybe_reap_idle_kernel(now=10_000.0) is True
    assert fake.terminated is True
    assert S._kernel_session is None


def test_idle_reaper_keeps_active_kernel(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "100")
    fake = _FakeKernel()
    monkeypatch.setattr(S, "_kernel_session", fake)
    now = S.time.monotonic()
    monkeypatch.setattr(S, "_last_activity", now)

    assert S._maybe_reap_idle_kernel(now=now + 1) is False
    assert fake.terminated is False


def test_idle_reaper_disabled_when_timeout_zero(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "0")
    fake = _FakeKernel()
    monkeypatch.setattr(S, "_kernel_session", fake)
    monkeypatch.setattr(S, "_last_activity", 0.0)

    assert S._maybe_reap_idle_kernel(now=10_000.0) is False
    assert fake.terminated is False


def test_idle_reaper_skips_when_eval_in_flight(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "100")
    fake = _FakeKernel()
    monkeypatch.setattr(S, "_kernel_session", fake)
    monkeypatch.setattr(S, "_last_activity", 0.0)

    acquired = S._session_eval_lock.acquire(blocking=False)
    assert acquired
    try:
        # A held eval lock means an evaluation is running -> not idle.
        assert S._maybe_reap_idle_kernel(now=10_000.0) is False
        assert fake.terminated is False
    finally:
        S._session_eval_lock.release()


def test_idle_timeout_env_parsing(monkeypatch):
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "42")
    assert S._kernel_idle_timeout() == 42.0
    monkeypatch.setenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", "bogus")
    assert S._kernel_idle_timeout() == S._DEFAULT_IDLE_TIMEOUT
    monkeypatch.delenv("MATHEMATICA_KERNEL_IDLE_TIMEOUT", raising=False)
    assert S._kernel_idle_timeout() == S._DEFAULT_IDLE_TIMEOUT


# ---- live parity -----------------------------------------------------------


def test_kernel_version_ok_accepts_supported_versions():
    """Regression for the version gate below: 14.x must be accepted (the old
    ``startswith("15")`` check rejected every supported 14.x kernel)."""
    assert _kernel_version_ok("15.1")
    assert _kernel_version_ok("14.2")
    assert _kernel_version_ok("13.0")
    assert not _kernel_version_ok("12.9")
    assert not _kernel_version_ok("")
    assert not _kernel_version_ok(None)


@pytest.mark.wolfram_runtime
def test_live_symbol_index_builds_warm(require_wolfram_runtime):
    """The symbol index must build on the warm session (no cold subprocess) and
    record the running kernel's own version (>= 13; 14.x stays supported)."""
    from mathematica_mcp import symbol_index as SI

    session = S.get_kernel_session()
    if session is None:
        pytest.skip("no persistent kernel session available")

    SI.invalidate()
    S.reset_cold_execution_count()

    assert SI._build_index_sync() is True

    version = SI.get_version()
    assert _kernel_version_ok(version), f"unexpected kernel version {version!r}"
    # The stored version must equal what the kernel itself reports (warm eval).
    probe = S.evaluate_wl("$VersionNumber", timeout=30)
    assert probe.success
    assert float(probe.text) == float(version)

    assert "Integrate" in SI.search("Integrate")
    assert S.cold_execution_count() == 0


@pytest.mark.wolfram_runtime
async def test_live_migrated_tools_are_warm(require_wolfram_runtime):
    """The migrated cold tools must run on the warm session end-to-end: every
    response carries execution_method='wolframclient' and no cold subprocess is
    spawned."""
    from mathematica_mcp import lazy_wolfram_tools as L
    from mathematica_mcp.server import _parse_wolfram_association as P

    session = S.get_kernel_session()
    if session is None:
        pytest.skip("no persistent kernel session available")

    S.reset_cold_execution_count()

    # Each response must be warm AND structured: the JSON-export path must parse,
    # never the {"raw": ..., "parse_error": true} regex fallback (Codex finding).
    const = json.loads(await L.get_constant("Pi", parse_wolfram_association=P))
    assert const.get("execution_method") == "wolframclient"
    assert "parse_error" not in const, const.get("raw", "")[:200]
    assert const.get("success") is True

    conv = json.loads(await L.convert_units("5 meters", "feet", parse_wolfram_association=P))
    assert conv.get("execution_method") == "wolframclient"
    assert "parse_error" not in conv, conv.get("raw", "")[:200]

    state = json.loads(await L.get_kernel_state(parse_wolfram_association=P))
    assert state.get("execution_method") == "wolframclient"
    assert "parse_error" not in state, state.get("raw", "")[:200]
    assert isinstance(state.get("global_symbol_count"), int)
    assert isinstance(state.get("memory_in_use_mb"), (int, float))

    pkgs = json.loads(await L.list_loaded_packages(parse_wolfram_association=P))
    assert pkgs.get("execution_method") == "wolframclient"
    assert "parse_error" not in pkgs, pkgs.get("raw", "")[:200]
    assert isinstance(pkgs.get("packages"), list)

    assert S.cold_execution_count() == 0


@pytest.mark.wolfram_runtime
def test_live_warm_matches_cold(require_wolfram_runtime):
    """The warm session and a cold wolframscript must parse to the same dict, and
    the warm path must not spawn a cold subprocess."""
    import subprocess as sp

    from mathematica_mcp.server import _parse_wolfram_association

    code = 'Module[{c}, c = ToExpression["Pi"]; <|"success" -> True, "name" -> "Pi", "numeric" -> ToString[N[c, 15]]|>]'

    session = S.get_kernel_session()
    if session is None:
        pytest.skip("no persistent kernel session available")

    S.reset_cold_execution_count()
    warm = S.evaluate_wl(code)
    assert warm.execution_method == "wolframclient"
    assert S.cold_execution_count() == 0

    cold_stdout = sp.run(["wolframscript", "-code", code], capture_output=True, text=True, timeout=60).stdout.strip()

    assert _parse_wolfram_association(warm.text) == _parse_wolfram_association(cold_stdout)
