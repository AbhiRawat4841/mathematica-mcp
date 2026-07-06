"""Lean-tool dispatch: each consolidated tool routes to the right internal with
the right arguments, and required-param guards fire. Mocked (no kernel)."""

from __future__ import annotations

import json

import pytest

from mathematica_mcp import server as srv


def _record(monkeypatch, name: str, ret: str = "{}"):
    """Patch srv.<name> with an async recorder; return the calls list."""
    calls: list[tuple] = []

    async def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return ret

    monkeypatch.setattr(srv, name, fake)
    return calls


# ---- notebooks -------------------------------------------------------------


async def test_notebooks_list(monkeypatch):
    calls = _record(monkeypatch, "get_notebooks")
    await srv.notebooks(action="list")
    assert len(calls) == 1


async def test_notebooks_create_passes_title(monkeypatch):
    calls = _record(monkeypatch, "create_notebook")
    await srv.notebooks(action="create", title="My NB")
    assert calls[0][0][0] == "My NB"


async def test_notebooks_open_requires_path():
    out = json.loads(await srv.notebooks(action="open"))
    assert out["success"] is False


async def test_notebooks_save_defaults_format(monkeypatch):
    calls = _record(monkeypatch, "save_notebook")
    await srv.notebooks(action="save")
    assert calls[0][0][2] == "Notebook"  # (notebook, path, format, session_id)


# ---- cells / edit_cells ----------------------------------------------------


async def test_cells_read_requires_cell_id():
    out = json.loads(await srv.cells(action="read"))
    assert out["success"] is False


async def test_cells_read_dispatches(monkeypatch):
    calls = _record(monkeypatch, "get_cell_content")
    await srv.cells(action="read", cell_id="c1")
    assert calls[0][0][0] == "c1"


async def test_edit_cells_write_requires_content():
    out = json.loads(await srv.edit_cells(action="write"))
    assert out["success"] is False


async def test_edit_cells_delete_dispatches(monkeypatch):
    calls = _record(monkeypatch, "delete_cell")
    await srv.edit_cells(action="delete", cell_id="c9")
    assert calls[0][0][0] == "c9"


# ---- evaluate --------------------------------------------------------------


async def test_evaluate_kernel_uses_cli(monkeypatch):
    calls = _record(monkeypatch, "execute_code")
    await srv.evaluate(code="1+1", target="kernel")
    assert calls[0][1]["output_target"] == "cli"


async def test_evaluate_notebook_uses_notebook(monkeypatch):
    calls = _record(monkeypatch, "execute_code")
    await srv.evaluate(code="Plot[x,{x,0,1}]", target="notebook")
    assert calls[0][1]["output_target"] == "notebook"


async def test_evaluate_dry_run_checks_syntax(monkeypatch):
    calls = _record(monkeypatch, "check_syntax")
    await srv.evaluate(code="1+", dry_run=True)
    assert calls[0][0][0] == "1+"


async def test_evaluate_file_runs_script(monkeypatch):
    calls = _record(monkeypatch, "run_script")
    await srv.evaluate(file="/tmp/x.wl")
    assert calls[0][0][0] == "/tmp/x.wl"


async def test_evaluate_cell_requires_cell_id():
    out = json.loads(await srv.evaluate(target="cell"))
    assert out["success"] is False


async def test_evaluate_requires_code():
    out = json.loads(await srv.evaluate())
    assert out["success"] is False


# ---- kernel / vars ---------------------------------------------------------


async def test_kernel_inspect_dispatches(monkeypatch):
    calls = _record(monkeypatch, "get_expression_info")
    await srv.kernel(action="inspect", expression="Integrate")
    assert calls[0][0][0] == "Integrate"


async def test_kernel_load_package_requires_package():
    out = json.loads(await srv.kernel(action="load_package"))
    assert out["success"] is False


async def test_vars_bare_clear_rejected_clear_all_wipes(monkeypatch):
    # Data-loss guard: bare clear errors instead of wiping; clear_all wipes explicitly.
    calls = _record(monkeypatch, "clear_variables")
    out = json.loads(await srv.vars(action="clear"))
    assert out["success"] is False
    assert calls == []
    await srv.vars(action="clear_all")
    assert calls[0][1]["clear_all"] is True


async def test_vars_set_requires_name_and_value():
    out = json.loads(await srv.vars(action="set", name="x"))
    assert out["success"] is False


# ---- screenshot / status / guide / batch / read_notebook_file --------------


async def test_screenshot_cell_requires_cell_id():
    with pytest.raises(ValueError):
        await srv.screenshot(scope="cell")


async def test_screenshot_expression_requires_expression():
    with pytest.raises(ValueError):
        await srv.screenshot(scope="expression")


async def test_status_merges_status_and_features(monkeypatch):
    async def fake_status():
        return json.dumps({"connection_mode": "addon"})

    async def fake_features():
        return json.dumps({"profile": "lean"})

    monkeypatch.setattr(srv, "get_mathematica_status", fake_status)
    monkeypatch.setattr(srv, "get_feature_status", fake_features)
    out = json.loads(await srv.status())
    assert out["connection_mode"] == "addon"
    assert out["profile"] == "lean"


async def test_guide_returns_topic_content():
    out = json.loads(await srv.guide(topic="workflow"))
    assert out["topic"] == "workflow"
    assert "evaluate" in out["guidance"]


async def test_batch_passthrough(monkeypatch):
    calls = _record(monkeypatch, "batch_commands")
    ops = [{"command": "ping", "params": {}}]
    await srv.batch(ops)
    assert calls[0][0][0] == ops


async def test_read_notebook_file_maps_mode(monkeypatch):
    calls = _record(monkeypatch, "read_notebook")
    await srv.read_notebook_file("/tmp/x.nb", mode="outline")
    assert calls[0][1]["output_format"] == "outline"


async def test_notebooks_create_forwards_show_chatbar(monkeypatch):
    calls = _record(monkeypatch, "create_notebook")
    await srv.notebooks(action="create", title="Chatty", show_chatbar=True)
    assert calls[-1][0] == ("Chatty", None, True)


async def test_create_notebook_sends_show_chatbar_to_addon(monkeypatch):
    payloads = []

    async def fake_addon(command, params=None, **k):
        payloads.append((command, params))
        return {"success": True, "id": "nb1"}

    monkeypatch.setattr(srv, "_addon_result", fake_addon)
    await srv.create_notebook("T", show_chatbar=True)
    command, params = payloads[-1]
    assert command == "create_notebook"
    assert params["show_chatbar"] is True

    await srv.create_notebook("T")
    assert payloads[-1][1]["show_chatbar"] is False
