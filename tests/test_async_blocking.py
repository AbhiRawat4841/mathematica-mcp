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
    monkeypatch.setattr(server, "_try_addon_command", lambda command, params=None: [])

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
        lambda command, params=None: {
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
    monkeypatch.setattr(server, "_try_addon_command", lambda command, params=None: {"selected": True})

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
        lambda command, params=None: {"success": False, "error": "addon down"},
    )
    monkeypatch.setattr(server, "get_kernel_session", lambda: None)

    result = asyncio.run(server.get_mathematica_status())
    payload = json.loads(result)

    assert payload["connection_mode"] == "disconnected"
    assert "No connection available" in payload["error"]
