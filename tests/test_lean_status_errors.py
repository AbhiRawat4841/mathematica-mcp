"""Lean status/error-analysis correctness (Domain C). Mocked — no kernel.

Covers:
- C1: benign $MessageList warnings default to type "warning", not "error", so
  analyze_messages reports severity="warning" and does not flip should_retry.
- C2: status() must not let get_feature_status's unconditional success=True
  overwrite a disconnected/error status from get_mathematica_status.
- C3: guide("errors") describes retry_with as conditionally available.
"""

from __future__ import annotations

import json

from mathematica_mcp import server as srv
from mathematica_mcp.error_analyzer import analyze_messages

# ---- C1: message severity defaults ----------------------------------------


def test_benign_warning_extracted_as_warning():
    # Solve::ratnz is a benign "succeeded with warnings" message, not a hard error.
    msgs = srv._messages_from_warnings(
        ["Solve::ratnz: Solve was unable to solve the system with inexact coefficients."]
    )
    assert len(msgs) == 1
    assert msgs[0]["type"] == "warning"


def test_benign_warning_no_error_severity_no_retry():
    msgs = srv._messages_from_warnings(["Solve::ratnz: warning text"])
    analysis = analyze_messages(msgs)
    assert analysis["severity"] == "warning"
    assert analysis["errors"] == 0
    assert analysis["should_retry"] is False


def test_hard_error_family_still_error():
    # Syntax is a known hard-error family — must stay type "error".
    msgs = srv._messages_from_warnings(["Syntax::sntxi: Incomplete expression."])
    assert msgs[0]["type"] == "error"
    assert analyze_messages(msgs)["severity"] == "error"


# ---- C2: status() merge precedence ----------------------------------------


async def test_status_disconnected_does_not_surface_success(monkeypatch):
    async def fake_math_status():
        return json.dumps({"connection_mode": "disconnected", "error": "No connection available: boom"})

    async def fake_feature_status():
        return json.dumps({"success": True, "features": {"profile": "lean"}})

    monkeypatch.setattr(srv, "get_mathematica_status", fake_math_status)
    monkeypatch.setattr(srv, "get_feature_status", fake_feature_status)

    merged = json.loads(await srv.status())
    assert merged.get("success") is not True
    assert merged["connection_mode"] == "disconnected"
    assert "error" in merged
    assert merged["features"] == {"profile": "lean"}


async def test_status_connected_preserves_fields(monkeypatch):
    async def fake_math_status():
        return json.dumps(
            {
                "success": True,
                "connection_mode": "addon",
                "warm_path": {"cold_executions": 0},
                "protocol_version": 3,
            }
        )

    async def fake_feature_status():
        return json.dumps({"success": True, "features": {"profile": "lean"}})

    monkeypatch.setattr(srv, "get_mathematica_status", fake_math_status)
    monkeypatch.setattr(srv, "get_feature_status", fake_feature_status)

    merged = json.loads(await srv.status())
    assert merged["connection_mode"] == "addon"
    assert merged["warm_path"] == {"cold_executions": 0}
    assert merged["protocol_version"] == 3
    assert merged["features"] == {"profile": "lean"}


# ---- C3: guide wording -----------------------------------------------------


async def test_guide_errors_softens_retry_with():
    guidance = json.loads(await srv.guide("errors"))["guidance"]
    assert "when available" in guidance
    assert "retry_with" in guidance
