from __future__ import annotations

import asyncio
import json

from mathematica_mcp import server


def test_get_notebooks_uses_to_thread(monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(server, "_try_addon_command", lambda command, params=None, timeout=None: [])

    result = asyncio.run(server.get_notebooks())

    assert json.loads(result) == []
    assert calls


def test_execute_code_cli_uses_to_thread(monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        server,
        "_try_addon_command",
        lambda command, params=None, timeout=None: {
            "success": True,
            "output": "2",
            "output_inputform": "2",
            "output_fullform": "",
            "output_tex": "",
        },
    )

    result = asyncio.run(server.execute_code(code="1+1", output_target="cli"))

    assert json.loads(result)["output_inputform"] == "2"
    assert calls


def test_read_notebook_content_uses_to_thread(monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(server.read_notebook_content("Integration.nb"))

    payload = json.loads(result)
    assert payload["success"] is True
    assert "_load_cached_notebook" in calls


def test_select_cell_uses_to_thread(monkeypatch):
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(server, "_try_addon_command", lambda command, params=None, timeout=None: {"selected": True})

    result = asyncio.run(server.select_cell("123"))

    assert json.loads(result)["selected"] is True
    assert calls


def test_get_mathematica_status_fallback_does_not_crash(monkeypatch):
    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        server,
        "_try_addon_command",
        lambda command, params=None, timeout=None: {"success": False, "error": "addon down"},
    )
    monkeypatch.setattr(server, "get_kernel_session", lambda: None)

    result = asyncio.run(server.get_mathematica_status())
    payload = json.loads(result)

    assert payload["connection_mode"] == "disconnected"
    assert "No connection available" in payload["error"]


def test_execute_code_cli_forwards_timeout_to_addon(monkeypatch):
    """execute_code(timeout=X) should pass timeout=X+10 to _try_addon_command."""
    captured = {}

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fake_addon(command, params=None, timeout=None):
        captured["command"] = command
        captured["timeout"] = timeout
        captured["params_timeout"] = (params or {}).get("timeout")
        return {
            "success": True,
            "output": "ok",
            "output_inputform": "ok",
            "output_fullform": "",
            "output_tex": "",
        }

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(server, "_try_addon_command", fake_addon)

    asyncio.run(server.execute_code(code="1+1", output_target="cli", timeout=120))

    # Socket timeout should be timeout + 10
    assert captured["timeout"] == 130.0
    # The params dict should carry the original timeout for the addon
    assert captured["params_timeout"] == 120


def test_execute_code_default_timeout_is_300(monkeypatch):
    """execute_code default timeout should be 300, not 60."""
    captured = {}

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fake_addon(command, params=None, timeout=None):
        captured["params_timeout"] = (params or {}).get("timeout")
        captured["socket_timeout"] = timeout
        return {
            "success": True,
            "output": "ok",
            "output_inputform": "ok",
            "output_fullform": "",
            "output_tex": "",
        }

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(server, "_try_addon_command", fake_addon)

    asyncio.run(server.execute_code(code="1+1", output_target="cli"))

    assert captured["params_timeout"] == 300
    assert captured["socket_timeout"] == 310.0


def test_execute_code_notebook_forwards_timeout_to_addon(monkeypatch):
    """Notebook path should forward timeout in params and use timeout+10 for socket."""
    captured = {}

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fake_addon(command, params=None, timeout=None):
        captured["command"] = command
        captured["socket_timeout"] = timeout
        captured["params_timeout"] = (params or {}).get("timeout")
        return {
            "success": True,
            "mode": "kernel",
            "notebook_id": "nb1",
            "cell_id": "c1",
            "cell_id_numeric": 1,
            "timing_ms": 100,
            "created_notebook": False,
            "timed_out": False,
            "messages": [],
            "has_errors": False,
            "has_warnings": False,
            "message_count": 0,
            "output_preview": "ok",
            "is_graphics": False,
        }

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(server, "_try_addon_command", fake_addon)

    asyncio.run(server.execute_code(code="1+1", output_target="notebook", timeout=200))

    assert captured["command"] == "execute_code_notebook"
    assert captured["params_timeout"] == 200
    assert captured["socket_timeout"] == 210.0


def test_notebook_timeout_does_not_fall_through_to_cli(monkeypatch):
    """A timed_out notebook result must be returned directly, not re-executed via CLI."""
    call_log = []

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fake_addon(command, params=None, timeout=None):
        call_log.append(command)
        if command == "execute_code_notebook":
            return {
                "success": False,
                "timed_out": True,
                "mode": "kernel",
                "notebook_id": "nb1",
                "cell_id": "c1",
                "cell_id_numeric": 1,
                "timing_ms": 300000,
                "created_notebook": False,
                "messages": [],
                "has_errors": False,
                "has_warnings": False,
                "message_count": 0,
                "output_preview": "$Aborted",
                "is_graphics": False,
            }
        # If CLI fallback is triggered, this would be called
        return {"success": True, "output": "should not happen"}

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(server, "_try_addon_command", fake_addon)

    result = asyncio.run(server.execute_code(code="Pause[9999]", output_target="notebook", timeout=300))
    payload = json.loads(result)

    # Must return timeout status, not re-execute
    assert payload["status"] == "timeout"
    assert payload["timed_out"] is True
    assert payload["evaluated"] is False
    # Only one addon call should have been made (no CLI fallback)
    assert call_log == ["execute_code_notebook"]


def test_notebook_frontend_timeout_does_not_fall_through_to_cli(monkeypatch):
    """A timed_out frontend notebook result must be returned directly."""
    call_log = []

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def fake_addon(command, params=None, timeout=None):
        call_log.append(command)
        if command == "execute_code_notebook":
            return {
                "success": False,
                "timed_out": True,
                "mode": "frontend",
                "notebook_id": "nb1",
                "cell_id": "c1",
                "cell_id_numeric": 1,
                "waited_seconds": 305,
                "created_notebook": False,
                "messages": [],
                "has_errors": False,
                "has_warnings": False,
                "message_count": 0,
                "output_preview": "$Aborted",
            }
        return {"success": True, "output": "should not happen"}

    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(server, "_try_addon_command", fake_addon)

    result = asyncio.run(
        server.execute_code(code="Pause[9999]", output_target="notebook", mode="frontend", timeout=300)
    )
    payload = json.loads(result)

    assert payload["status"] == "timeout"
    assert payload["timed_out"] is True
    assert call_log == ["execute_code_notebook"]
    # Frontend waited_seconds should be converted to timing_ms
    assert payload["timing_ms"] == 305000
