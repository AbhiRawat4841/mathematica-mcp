from __future__ import annotations

import asyncio
import importlib
import shutil
import sys


def test_lazy_wolfram_tools_not_imported_with_server():
    sys.modules.pop("mathematica_mcp.lazy_wolfram_tools", None)
    server = importlib.reload(importlib.import_module("mathematica_mcp.server"))

    assert "mathematica_mcp.lazy_wolfram_tools" not in sys.modules
    assert hasattr(server, "wolfram_alpha")


def test_lazy_wolfram_tools_imported_on_first_use(monkeypatch):
    sys.modules.pop("mathematica_mcp.lazy_wolfram_tools", None)
    # Also clear the package's cached submodule reference
    pkg = sys.modules.get("mathematica_mcp")
    if pkg and hasattr(pkg, "lazy_wolfram_tools"):
        delattr(pkg, "lazy_wolfram_tools")
    server = importlib.reload(importlib.import_module("mathematica_mcp.server"))

    monkeypatch.setattr(
        server,
        "_try_addon_command",
        lambda command, params=None, timeout=None: {"success": True, "trace": []},
    )

    result = asyncio.run(server.trace_evaluation("x"))

    assert "mathematica_mcp.lazy_wolfram_tools" in sys.modules
    assert '"success": true' in result.lower()


# ============================================================================
# Phase 4: wolframscript discovery caching tests
# ============================================================================


def test_find_wolframscript_caches_result(monkeypatch):
    """Repeated calls to _find_wolframscript invoke shutil.which once."""
    from mathematica_mcp.lazy_wolfram_tools import (
        _clear_wolframscript_cache,
        _find_wolframscript,
    )

    _clear_wolframscript_cache()

    call_count = {"n": 0}
    original_which = shutil.which

    def counting_which(name):
        call_count["n"] += 1
        return original_which(name)

    monkeypatch.setattr(shutil, "which", counting_which)

    # Call twice
    result1 = _find_wolframscript()
    result2 = _find_wolframscript()

    assert result1 == result2
    assert call_count["n"] == 1, f"shutil.which called {call_count['n']} times, expected 1"

    _clear_wolframscript_cache()


def test_clear_wolframscript_cache_forces_new_lookup(monkeypatch):
    """After clearing cache, a fresh shutil.which call is made."""
    from mathematica_mcp.lazy_wolfram_tools import (
        _clear_wolframscript_cache,
        _find_wolframscript,
    )

    _clear_wolframscript_cache()

    call_count = {"n": 0}
    original_which = shutil.which

    def counting_which(name):
        call_count["n"] += 1
        return original_which(name)

    monkeypatch.setattr(shutil, "which", counting_which)

    _find_wolframscript()
    assert call_count["n"] == 1

    _clear_wolframscript_cache()
    _find_wolframscript()
    assert call_count["n"] == 2

    _clear_wolframscript_cache()


def test_missing_binary_returns_none(monkeypatch):
    """When wolframscript is not on PATH, _find_wolframscript returns None."""
    from mathematica_mcp.lazy_wolfram_tools import (
        _clear_wolframscript_cache,
        _find_wolframscript,
    )

    _clear_wolframscript_cache()
    monkeypatch.setattr(shutil, "which", lambda _: None)

    result = _find_wolframscript()
    assert result is None

    _clear_wolframscript_cache()
