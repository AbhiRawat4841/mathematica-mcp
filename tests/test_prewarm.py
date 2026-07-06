"""Background kernel prewarm + creation-lock race safety.

All tests use fakes; none boot a real Wolfram kernel. The race test injects a
fake ``wolframclient`` into ``sys.modules`` so ``get_kernel_session`` exercises
its real creation path without a live kernel.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
import types

import pytest

from mathematica_mcp import session as S


@pytest.fixture
def _reset_session():
    """Snapshot and restore the session module globals the create path mutates."""
    names = (
        "_kernel_session",
        "_use_wolframscript",
        "_session_starting",
        "_last_kernel_health_check",
        "_reaper_thread",
        "_last_activity",
        "_boot_retry_after",
    )
    saved = {n: getattr(S, n) for n in names}
    S._kernel_session = None
    S._use_wolframscript = False
    S._session_starting = False
    S._last_kernel_health_check = 0.0
    S._reaper_thread = None
    S._boot_retry_after = 0.0
    try:
        yield
    finally:
        for n, v in saved.items():
            setattr(S, n, v)


# ---- creation-lock race ----------------------------------------------------


def test_get_kernel_session_race_boots_one_kernel(monkeypatch, _reset_session):
    """Two concurrent get_kernel_session() calls must boot exactly ONE kernel and
    both receive the same session (no leaked license seat)."""
    instantiations: list[str] = []
    count_lock = threading.Lock()

    class _FakeWLSession:
        def __init__(self, kernel_path):
            with count_lock:
                instantiations.append(kernel_path)
            time.sleep(0.1)  # widen the window so both threads race the create path

        def start(self):
            time.sleep(0.1)

        def evaluate(self, _expr):
            return 2

        def terminate(self):
            pass

    fake_pkg = types.ModuleType("wolframclient")
    fake_eval = types.ModuleType("wolframclient.evaluation")
    fake_eval.WolframLanguageSession = _FakeWLSession
    fake_lang = types.ModuleType("wolframclient.language")
    fake_lang.wlexpr = lambda s: s
    fake_pkg.evaluation = fake_eval
    fake_pkg.language = fake_lang
    monkeypatch.setitem(sys.modules, "wolframclient", fake_pkg)
    monkeypatch.setitem(sys.modules, "wolframclient.evaluation", fake_eval)
    monkeypatch.setitem(sys.modules, "wolframclient.language", fake_lang)

    monkeypatch.setenv("MATHEMATICA_KERNEL_PATH", "/fake/kernel")
    monkeypatch.setattr(S, "_start_idle_reaper", lambda: None)  # keep no reaper thread alive

    results: list[object] = []
    results_lock = threading.Lock()

    def _call():
        session = S.get_kernel_session()
        with results_lock:
            results.append(session)

    threads = [threading.Thread(target=_call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5.0)

    assert len(instantiations) == 1, instantiations
    assert results[0] is results[1]
    assert results[0] is not None


# ---- prewarm gating --------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "no", "FALSE", "No", "off", "OFF"])
def test_prewarm_disabled_returns_none(monkeypatch, _reset_session, value):
    monkeypatch.setenv("MATHEMATICA_PREWARM", value)
    calls: list[int] = []
    monkeypatch.setattr(S, "get_kernel_session", lambda: calls.append(1))

    assert S.prewarm_kernel() is None
    assert calls == []  # never spawned the boot thread


def test_prewarm_skips_when_session_exists(monkeypatch, _reset_session):
    monkeypatch.delenv("MATHEMATICA_PREWARM", raising=False)  # default on
    S._kernel_session = object()  # pretend a session already exists
    calls: list[int] = []
    monkeypatch.setattr(S, "get_kernel_session", lambda: calls.append(1))

    assert S.prewarm_kernel() is None
    assert calls == []


def test_prewarm_skips_when_cold_forced(monkeypatch, _reset_session):
    monkeypatch.delenv("MATHEMATICA_PREWARM", raising=False)
    S._use_wolframscript = True
    calls: list[int] = []
    monkeypatch.setattr(S, "get_kernel_session", lambda: calls.append(1))

    assert S.prewarm_kernel() is None
    assert calls == []


def test_prewarm_spawns_thread_and_boots(monkeypatch, _reset_session):
    monkeypatch.delenv("MATHEMATICA_PREWARM", raising=False)  # default on
    booted = threading.Event()
    monkeypatch.setattr(S, "get_kernel_session", lambda: booted.set())

    thread = S.prewarm_kernel()

    assert thread is not None
    assert thread.name == "kernel-prewarm"
    assert thread.daemon is True
    thread.join(5.0)
    assert booted.is_set()


def test_prewarm_swallows_boot_failure(monkeypatch, _reset_session):
    monkeypatch.delenv("MATHEMATICA_PREWARM", raising=False)

    def _boom():
        raise RuntimeError("kernel exploded")

    monkeypatch.setattr(S, "get_kernel_session", _boom)

    thread = S.prewarm_kernel()  # must not raise into the caller
    assert thread is not None
    thread.join(5.0)  # the boot thread swallows the exception and exits cleanly
    assert not thread.is_alive()


# ---- import safety ---------------------------------------------------------


def test_main_calls_prewarm(monkeypatch):
    """server.main() must trigger prewarm before serving."""
    from mathematica_mcp import server

    calls: list[int] = []
    monkeypatch.setattr("mathematica_mcp.session.prewarm_kernel", lambda: calls.append(1))
    monkeypatch.setattr(server.mcp, "run", lambda: None)

    server.main()

    assert calls == [1]


def test_importing_server_does_not_prewarm(monkeypatch):
    """Importing server (as tests and schema dumps do) must NOT prewarm; the call
    lives only in main(). Reloading re-runs every top-level statement, so a stray
    module-level prewarm would bump the counter."""
    from mathematica_mcp import server

    calls: list[int] = []
    monkeypatch.setattr("mathematica_mcp.session.prewarm_kernel", lambda: calls.append(1))

    importlib.reload(server)

    assert calls == []


# ---- transient boot-failure cooldown (P1) ----------------------------------


def _install_fake_wl(monkeypatch, *, fail_starts=0):
    """Install a fake wolframclient into sys.modules so get_kernel_session runs
    its real create path with no live kernel. start() raises RuntimeError for the
    first ``fail_starts`` calls (transient boot failure), then succeeds. Returns
    (instantiations, terminated) trackers."""
    instantiations: list[str] = []
    terminated: list[str] = []
    budget = [fail_starts]

    class _FakeWLSession:
        def __init__(self, kernel_path):
            instantiations.append(kernel_path)
            self._kp = kernel_path

        def start(self):
            if budget[0] > 0:
                budget[0] -= 1
                raise RuntimeError("transient license contention at startup")

        def evaluate(self, _expr):
            return 2

        def terminate(self):
            terminated.append(self._kp)

    fake_pkg = types.ModuleType("wolframclient")
    fake_eval = types.ModuleType("wolframclient.evaluation")
    fake_eval.WolframLanguageSession = _FakeWLSession
    fake_lang = types.ModuleType("wolframclient.language")
    fake_lang.wlexpr = lambda s: s
    fake_pkg.evaluation = fake_eval
    fake_pkg.language = fake_lang
    monkeypatch.setitem(sys.modules, "wolframclient", fake_pkg)
    monkeypatch.setitem(sys.modules, "wolframclient.evaluation", fake_eval)
    monkeypatch.setitem(sys.modules, "wolframclient.language", fake_lang)
    monkeypatch.setenv("MATHEMATICA_KERNEL_PATH", "/fake/kernel")
    monkeypatch.setattr(S, "_start_idle_reaper", lambda: None)  # no reaper thread
    return instantiations, terminated


def test_transient_boot_failure_arms_cooldown_not_permanent_cold(monkeypatch, _reset_session):
    """A failed boot must NOT latch permanent-cold; it arms a cooldown and cleans
    up the half-started session (terminated + nulled)."""
    instantiations, terminated = _install_fake_wl(monkeypatch, fail_starts=1)

    assert S.get_kernel_session() is None  # start() raised
    assert S._use_wolframscript is False  # NOT permanent-cold
    assert S._boot_retry_after > time.monotonic()  # cooldown armed
    assert S._kernel_session is None  # half-started session nulled
    assert terminated == ["/fake/kernel"]  # ... and terminated
    assert len(instantiations) == 1


def test_cooldown_skips_boot_retry_fast(monkeypatch, _reset_session):
    """A call within the cooldown returns None without instantiating a new session."""
    instantiations, _ = _install_fake_wl(monkeypatch, fail_starts=1)

    assert S.get_kernel_session() is None  # first boot fails -> cooldown
    assert len(instantiations) == 1
    assert S.get_kernel_session() is None  # still in cooldown
    assert len(instantiations) == 1  # no new WolframLanguageSession


def test_retry_after_cooldown_recovers_warm(monkeypatch, _reset_session):
    """After the cooldown expires the next call retries the boot; a now-healthy
    fake succeeds and the session is warm again."""
    instantiations, _ = _install_fake_wl(monkeypatch, fail_starts=1)

    assert S.get_kernel_session() is None  # transient failure
    assert len(instantiations) == 1
    S._boot_retry_after = 0.0  # simulate cooldown expiry

    session = S.get_kernel_session()  # start() budget spent -> succeeds
    assert session is not None
    assert len(instantiations) == 2  # rebooted exactly once
    assert S._use_wolframscript is False


def test_close_kernel_session_resets_cooldown(monkeypatch, _reset_session):
    """An explicit close/restart is an explicit retry request: it clears the cooldown."""
    _install_fake_wl(monkeypatch, fail_starts=1)

    assert S.get_kernel_session() is None
    assert S._boot_retry_after > time.monotonic()

    S.close_kernel_session()
    assert S._boot_retry_after == 0.0
