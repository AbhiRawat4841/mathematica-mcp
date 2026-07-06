"""P3 guidance v2: retry_with in the analyzer, error analysis on all evaluate
paths, and survival through compact response filtering."""

from __future__ import annotations

import json
from pathlib import Path

from mathematica_mcp import connection as C
from mathematica_mcp import server as srv
from mathematica_mcp.error_analyzer import analyze_messages
from mathematica_mcp.response_filter import _filter_response


def test_retry_with_never_canned():
    # Codex correction: retry_with must be contextual or absent — the old canned
    # UnitConvert example was not a runnable fix for the user's expression. The
    # guidance survives in suggested_fix instead.
    a = analyze_messages([{"tag": "UnitConvert::compat", "text": "...", "type": "error"}])
    assert a["retry_with"] is None
    assert any("QuantityMagnitude" in an.get("suggested_fix", "") for an in a["analyses"])


def test_retry_with_none_for_unknown_error():
    a = analyze_messages([{"tag": "Foo::bar", "text": "x", "type": "error"}])
    assert a["retry_with"] is None


def test_messages_from_warnings_extracts_tags():
    msgs = srv._messages_from_warnings(["{Power::infy, General::stop}"])
    tags = {m["tag"] for m in msgs}
    assert "Power::infy" in tags
    assert "General::stop" in tags


def test_messages_from_warnings_empty():
    assert srv._messages_from_warnings([]) == []
    assert srv._messages_from_warnings(None) == []


def test_attach_error_analysis_on_kernel_path():
    resp = {"success": True, "warnings": ["{Divide::infy}"]}
    srv._attach_error_analysis(resp)
    assert "error_analysis" in resp
    # No canned retry_with; the analysis still carries the suggested fix.
    assert resp["error_analysis"]["retry_with"] is None
    assert resp["error_analysis"]["recommendations"]


def test_attach_error_analysis_is_idempotent():
    resp = {"error_analysis": {"sentinel": 1}, "warnings": ["{Power::infy}"]}
    srv._attach_error_analysis(resp)
    assert resp["error_analysis"] == {"sentinel": 1}


def test_attach_error_analysis_noop_when_clean():
    resp = {"success": True, "warnings": []}
    srv._attach_error_analysis(resp)
    assert "error_analysis" not in resp


def test_error_analysis_survives_compact_filter():
    resp = {
        "success": True,
        "output": "x",
        "error_analysis": {"retry_with": "QuantityMagnitude[...]", "should_retry": True},
    }
    filtered = _filter_response(resp, "compact")
    assert "error_analysis" in filtered
    assert filtered["error_analysis"]["retry_with"] == "QuantityMagnitude[...]"


# ---- state deltas ----------------------------------------------------------


class _MockSock:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._sent = False

    def settimeout(self, *a):
        pass

    def sendall(self, *a):
        pass

    def recv(self, _n: int) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._payload


def test_connection_surfaces_state_delta():
    envelope = (
        json.dumps(
            {
                "status": "success",
                "result": {"pong": True},
                "state_delta": {"notebook": None, "cell_count": 3, "kernel_busy": False},
            }
        ).encode()
        + b"\n"
    )
    conn = C.MathematicaConnection()
    conn._socket = _MockSock(envelope)
    out = conn.send_command("ping")
    assert out["pong"] is True
    assert out["state_delta"]["cell_count"] == 3


def test_addon_source_appends_state_delta():
    wl = (Path(__file__).resolve().parents[1] / "addon" / "MathematicaMCP.wl").read_text()
    assert "mcpStateDelta[]" in wl
    assert '"state_delta" -> mcpStateDelta[]' in wl
    for key in ("notebook", "cell_count", "kernel_busy"):
        assert key in wl
