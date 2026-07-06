"""Live end-to-end smoke for the lean consolidated tools. Headless paths use the
warm kernel; the notebook roundtrip drives the frontend with a throwaway
notebook that is always closed. Skipped when the runtime/addon is unavailable.
"""

from __future__ import annotations

import json
import socket

import pytest

from mathematica_mcp import server as srv
from mathematica_mcp import session as S
from mathematica_mcp.connection import DEFAULT_HOST, DEFAULT_PORT


def _addon_reachable() -> bool:
    try:
        with socket.create_connection((DEFAULT_HOST, DEFAULT_PORT), timeout=2.0):
            return True
    except OSError:
        return False


@pytest.mark.wolfram_runtime
async def test_live_evaluate_kernel_is_warm(require_wolfram_runtime):
    if S.get_kernel_session() is None:
        pytest.skip("no persistent kernel session")
    S.reset_cold_execution_count()
    out = json.loads(await srv.evaluate(code="Integrate[x^2, x]", target="kernel"))
    text = json.dumps(out)
    assert "x^3" in text or "3" in text
    assert S.cold_execution_count() == 0


@pytest.mark.wolfram_runtime
async def test_live_status_reports_warm_path(require_wolfram_runtime):
    out = json.loads(await srv.status())
    assert "warm_path" in out
    assert "cold_executions" in out["warm_path"]
    assert "idle_timeout_seconds" in out["warm_path"]
    # profile/features come from get_feature_status (nested under "features")
    assert "features" in out and out["features"].get("profile")


@pytest.mark.wolfram_runtime
async def test_live_verify_derivation_warm(require_wolfram_runtime):
    if S.get_kernel_session() is None:
        pytest.skip("no persistent kernel session")
    out = json.loads(await srv.verify_derivation(["Sin[x]^2 + Cos[x]^2", "1"]))
    assert out["success"] is True
    assert out.get("execution_method") == "wolframclient"


async def test_live_read_notebook_file_no_kernel():
    """read_notebook_file parses a .nb with the Python-native backend (no kernel).

    Uses Integration.nb — the only TRACKED notebook fixture (root *.nb is
    gitignored, so anything else is absent on a clean checkout)."""
    out = await srv.read_notebook_file("Integration.nb", mode="outline")
    assert isinstance(out, str) and len(out) > 0
    assert "error" not in out.lower() or "cell" in out.lower()


@pytest.mark.needs_live_addon
@pytest.mark.wolfram_runtime
async def test_live_notebook_fe_roundtrip():
    if not _addon_reachable():
        pytest.skip(f"no addon on {DEFAULT_HOST}:{DEFAULT_PORT}")

    created = json.loads(await srv.notebooks(action="create", title="LeanLiveThrowaway"))
    nb = created.get("id")
    assert created.get("created") is True and nb

    try:
        wrote = json.loads(await srv.edit_cells(action="write", content="2 + 2", notebook=nb))
        assert wrote.get("error") is None

        listed = json.loads(await srv.cells(action="list", notebook=nb))
        assert listed.get("error") is None

        shot = await srv.screenshot(scope="notebook", notebook=nb)
        assert getattr(shot, "data", None)  # non-empty PNG bytes
    finally:
        closed = json.loads(await srv.notebooks(action="close", notebook=nb))
        assert closed.get("closed") is True
