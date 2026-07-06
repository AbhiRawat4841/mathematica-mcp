"""Addon kernel as the MIDDLE fallback rung in evaluate_wl.

When the warm persistent session is not ready (booting or cold-forced) AND a call
opts in via ``allow_addon_fallback=True``, evaluate_wl routes through the already
connected addon kernel (~30ms) instead of a cold ``wolframscript`` subprocess.
Mocked tests run everywhere; the live test (``wolfram_runtime``) exercises the
real addon and only runs when it is connected.
"""

from __future__ import annotations

import json
import types

import pytest

from mathematica_mcp import session as S

_CONN_PATH = "mathematica_mcp.connection.get_mathematica_connection"


@pytest.fixture(autouse=True)
def _reset_counters():
    S.reset_cold_execution_count()
    S.reset_addon_execution_count()
    yield
    S.reset_cold_execution_count()
    S.reset_addon_execution_count()


class _FakeSession:
    def __init__(self, ret):
        self._ret = ret

    def evaluate(self, _expr):
        return self._ret


class _FakeConnection:
    """Stands in for MathematicaConnection: records sends, returns a canned resp."""

    def __init__(self, resp, *, connected=True, raise_on_send=False):
        self._resp = resp
        self._connected = connected
        self.raise_on_send = raise_on_send
        self.sent: list[tuple[str, dict]] = []
        # Socket-timeout arg passed to send_command, one entry per send. Distinct
        # from params["timeout"] (the addon's kernel-side limit).
        self.socket_timeouts: list = []

    def is_connected(self):
        return self._connected

    def send_command(self, command, params=None, timeout=None):
        self.sent.append((command, params or {}))
        self.socket_timeouts.append(timeout)
        if self.raise_on_send:
            raise ConnectionError("addon socket died")
        return self._resp


def _addon_resp(inputform, *, success=True, truncated=False):
    """Build a canned execute_code response matching the real addon shape.

    ``inputform`` is what cmdExecuteCode puts in ``output_inputform``: the
    InputForm rendering of the wrapper's String result, i.e. a quoted+escaped
    string literal. json.dumps(raw) is a faithful stand-in for WL's InputForm
    quoting over the escape subset we support (verified against the live addon).
    """
    resp = {
        "success": success,
        "output": inputform,
        "output_inputform": inputform,
        "output_fullform": "",
        "output_tex": "",
        "messages": [],
        "warnings": [],
        "has_errors": False,
        "has_warnings": False,
        "timing_ms": 1,
        "execution_method": "addon",
    }
    if truncated:
        resp["truncated"] = True
    return resp


def _install_cold_sentinel(monkeypatch, stdout="COLD"):
    """Wire the cold wolframscript path to a sentinel stdout."""
    monkeypatch.setattr("mathematica_mcp.lazy_wolfram_tools._find_wolframscript", lambda: "/fake/wolframscript")
    monkeypatch.setattr(
        S.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )


# ---- warm ready: addon never consulted -------------------------------------


def test_warm_ready_never_consults_addon(monkeypatch):
    monkeypatch.setattr(S, "_warm_session_ready", lambda: True)
    monkeypatch.setattr(S, "get_kernel_session", lambda: _FakeSession("<|ok -> True|>"))

    def _boom():
        raise AssertionError("addon must not be consulted when the warm session is ready")

    monkeypatch.setattr(_CONN_PATH, _boom)

    result = S.evaluate_wl("1 + 1", allow_addon_fallback=True)

    assert result.success is True
    assert result.execution_method == "wolframclient"
    assert S.addon_execution_count() == 0


# ---- addon rung fires ------------------------------------------------------


def test_addon_rung_success_roundtrips_payload(monkeypatch):
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)  # not reached on addon hit

    # A JSON-ish payload with the escapes that matter: quotes, a backslash, and
    # embedded newline + tab. It must round-trip byte-for-byte.
    raw = 'line1\nline2\ttab "quoted" back\\slash {"json":true}'
    conn = _FakeConnection(_addon_resp(json.dumps(raw)))
    monkeypatch.setattr(_CONN_PATH, lambda: conn)

    result = S.evaluate_wl("SomeMath[]", timeout=30, allow_addon_fallback=True)

    assert result.success is True
    assert result.execution_method == "addon"
    assert result.text == raw
    assert S.addon_execution_count() == 1
    assert S.cold_execution_count() == 0
    # The addon received our parity wrapper (same one the warm path uses) with a
    # margin on the addon-side timeout. timeout=30 is exactly the cap, so
    # addon_timeout=30: wrapper limit 30, params timeout 35, socket timeout 45.
    command, params = conn.sent[0]
    assert command == "execute_code"
    assert "TimeConstrained" in params["code"] and "PageWidth -> Infinity" in params["code"]
    assert "TimeConstrained[(\nSomeMath[]\n), 30, $Aborted]" in params["code"]
    assert params["timeout"] == 35
    assert conn.socket_timeouts[0] == 45


def test_addon_rung_aborted_maps_to_timed_out(monkeypatch):
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    conn = _FakeConnection(_addon_resp(json.dumps("$Aborted")))
    monkeypatch.setattr(_CONN_PATH, lambda: conn)

    result = S.evaluate_wl("Pause[999]", timeout=1, allow_addon_fallback=True)

    assert result.success is False
    assert result.timed_out is True
    assert result.execution_method == "addon"
    assert result.text == ""
    assert S.addon_execution_count() == 1


# ---- timeout clamping to the cap -------------------------------------------


def test_addon_rung_clamps_timeout_when_over_cap(monkeypatch):
    # Caller asks for 120s; the rung runs on the user's front-end kernel, so the
    # kernel-side limit is clamped to ADDON_RUNG_TIMEOUT_CAP (30). params timeout
    # = cap+5 = 35, socket timeout = cap+15 = 45.
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    conn = _FakeConnection(_addon_resp(json.dumps("ok")))
    monkeypatch.setattr(_CONN_PATH, lambda: conn)

    result = S.evaluate_wl("SomeMath[]", timeout=120, allow_addon_fallback=True)

    assert result.success is True
    assert result.execution_method == "addon"
    _, params = conn.sent[0]
    assert "TimeConstrained[(\nSomeMath[]\n), 30, $Aborted]" in params["code"]
    assert params["timeout"] == 35
    assert conn.socket_timeouts[0] == 45


def test_addon_rung_passes_timeout_through_when_under_cap(monkeypatch):
    # Caller asks for 10s (under the cap): used verbatim. params timeout = 15,
    # socket timeout = 25.
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    conn = _FakeConnection(_addon_resp(json.dumps("ok")))
    monkeypatch.setattr(_CONN_PATH, lambda: conn)

    result = S.evaluate_wl("SomeMath[]", timeout=10, allow_addon_fallback=True)

    assert result.success is True
    _, params = conn.sent[0]
    assert "TimeConstrained[(\nSomeMath[]\n), 10, $Aborted]" in params["code"]
    assert params["timeout"] == 15
    assert conn.socket_timeouts[0] == 25


def test_aborted_over_cap_falls_through_to_cold(monkeypatch):
    # $Aborted at the cap only proves the job is too slow for the user's front-end
    # kernel, not that it exceeds the caller's 120s budget -> fall through to cold
    # (which gets the full budget). No addon execution is counted.
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    conn = _FakeConnection(_addon_resp(json.dumps("$Aborted")))
    monkeypatch.setattr(_CONN_PATH, lambda: conn)
    _install_cold_sentinel(monkeypatch, stdout="COLD")

    result = S.evaluate_wl("Slow[]", timeout=120, allow_addon_fallback=True)

    assert result.execution_method == "wolframscript"
    assert result.text == "COLD"
    assert S.addon_execution_count() == 0
    assert S.cold_execution_count() == 1


def test_aborted_within_cap_is_genuine_timeout(monkeypatch):
    # Caller's 20s is within the cap, so $Aborted is a real timeout: keep the
    # timed-out WLResult and never fall through to cold.
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    conn = _FakeConnection(_addon_resp(json.dumps("$Aborted")))
    monkeypatch.setattr(_CONN_PATH, lambda: conn)
    _install_cold_sentinel(monkeypatch, stdout="COLD")

    result = S.evaluate_wl("Slow[]", timeout=20, allow_addon_fallback=True)

    assert result.success is False
    assert result.timed_out is True
    assert result.execution_method == "addon"
    assert result.text == ""
    assert S.addon_execution_count() == 1
    assert S.cold_execution_count() == 0


# ---- fall through to cold --------------------------------------------------


def test_truncated_falls_through_to_cold(monkeypatch):
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    conn = _FakeConnection(_addon_resp(json.dumps("partial payload"), truncated=True))
    monkeypatch.setattr(_CONN_PATH, lambda: conn)
    _install_cold_sentinel(monkeypatch)

    result = S.evaluate_wl("Big[]", allow_addon_fallback=True)

    assert result.execution_method == "wolframscript"
    assert result.text == "COLD"
    assert S.addon_execution_count() == 0
    assert S.cold_execution_count() == 1


@pytest.mark.parametrize(
    "conn_factory",
    [
        pytest.param(lambda: _FakeConnection(None, connected=False), id="not_connected"),
        pytest.param(lambda: _FakeConnection(None), id="send_returns_none"),
        pytest.param(lambda: _FakeConnection(None, raise_on_send=True), id="send_raises"),
        pytest.param(lambda: _FakeConnection(_addon_resp(json.dumps("x"), success=False)), id="addon_failed"),
        pytest.param(lambda: _FakeConnection(_addon_resp("$Failed")), id="unquoted_output"),
        pytest.param(lambda: _FakeConnection(_addon_resp('"a\\[Alpha]"')), id="undecodable_escape"),
    ],
)
def test_bad_addon_response_falls_through_to_cold(monkeypatch, conn_factory):
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)
    monkeypatch.setattr(_CONN_PATH, lambda: conn_factory())
    _install_cold_sentinel(monkeypatch, stdout="42")

    result = S.evaluate_wl("6 * 7", allow_addon_fallback=True)

    assert result.execution_method == "wolframscript"
    assert result.text == "42"
    assert S.addon_execution_count() == 0
    assert S.cold_execution_count() == 1


def test_connection_lookup_raises_falls_through_to_cold(monkeypatch):
    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)

    def _raise():
        raise ConnectionError("could not connect to addon")

    monkeypatch.setattr(_CONN_PATH, _raise)
    _install_cold_sentinel(monkeypatch, stdout="42")

    result = S.evaluate_wl("6 * 7", allow_addon_fallback=True)

    assert result.execution_method == "wolframscript"
    assert S.addon_execution_count() == 0
    assert S.cold_execution_count() == 1


def test_default_opt_out_goes_straight_to_cold(monkeypatch):
    # No _warm_session_ready patch: opt-out must skip the addon regardless of it.
    monkeypatch.setattr(S, "get_kernel_session", lambda: None)

    def _boom():
        raise AssertionError("addon must not be consulted without allow_addon_fallback")

    monkeypatch.setattr(_CONN_PATH, _boom)
    _install_cold_sentinel(monkeypatch, stdout="42")

    result = S.evaluate_wl("6 * 7")  # allow_addon_fallback defaults False

    assert result.execution_method == "wolframscript"
    assert result.text == "42"
    assert S.addon_execution_count() == 0
    assert S.cold_execution_count() == 1


# ---- live parity -----------------------------------------------------------


@pytest.mark.wolfram_runtime
async def test_live_addon_rung_get_constant(monkeypatch):
    """With the warm session forced not-ready, an opted-in pure-math tool routes
    through the real addon kernel and returns the correct value.

    Forces readiness False via monkeypatch only (never touches the real
    _kernel_session global), so no persistent kernel is booted or orphaned.
    Timing-loose: other agents share the addon concurrently.
    """
    from mathematica_mcp import lazy_wolfram_tools as L
    from mathematica_mcp.server import _parse_wolfram_association as P

    try:
        from mathematica_mcp.connection import get_mathematica_connection

        conn = get_mathematica_connection()
    except Exception:
        pytest.skip("addon not connected")
    if not conn.is_connected():
        pytest.skip("addon not connected")

    monkeypatch.setattr(S, "_warm_session_ready", lambda: False)

    const = json.loads(await L.get_constant("Pi", parse_wolfram_association=P))

    assert const.get("success") is True
    assert const.get("execution_method") == "addon"
    assert abs(float(const["numeric"]) - 3.141592653589793) < 1e-6
    assert S.addon_execution_count() >= 1
