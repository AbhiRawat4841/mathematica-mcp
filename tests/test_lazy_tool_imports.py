from __future__ import annotations

import asyncio
import importlib
import sys


def test_lazy_wolfram_tools_not_imported_with_server():
    sys.modules.pop("mathematica_mcp.lazy_wolfram_tools", None)
    server = importlib.reload(importlib.import_module("mathematica_mcp.server"))

    assert "mathematica_mcp.lazy_wolfram_tools" not in sys.modules
    assert hasattr(server, "wolfram_alpha")


def test_lazy_wolfram_tools_imported_on_first_use(monkeypatch):
    sys.modules.pop("mathematica_mcp.lazy_wolfram_tools", None)
    server = importlib.reload(importlib.import_module("mathematica_mcp.server"))

    monkeypatch.setattr(
        server,
        "_try_addon_command",
        lambda command, params=None: {"success": True, "trace": []},
    )

    result = asyncio.run(server.trace_evaluation("x"))

    assert "mathematica_mcp.lazy_wolfram_tools" in sys.modules
    assert '"success": true' in result.lower()
