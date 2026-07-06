"""Domain C hardening (v1.0): schema data-loss guards, contextual next_step
routing, non-obvious followups, honest retry_with, and profile-aware prompts.

Every test here fails against the pre-fix code. Mocked — no kernel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mathematica_mcp import server as srv
from mathematica_mcp.config import FeatureFlags
from mathematica_mcp.error_analyzer import ERROR_PATTERNS, analyze_messages

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _record(monkeypatch, name: str, ret: str = "{}"):
    calls: list[tuple] = []

    async def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return ret

    monkeypatch.setattr(srv, name, fake)
    return calls


# ---- vars: bare clear must not wipe the session -----------------------------


async def test_vars_bare_clear_errors_with_next_step_and_no_wipe(monkeypatch):
    calls = _record(monkeypatch, "clear_variables")
    out = json.loads(await srv.vars(action="clear"))
    assert out["success"] is False
    assert "clear requires name= or pattern=" in out["error"]
    assert "clear_all" in out["next_step"]
    assert calls == []  # clear_variables must NOT have been called


async def test_vars_clear_all_wipes_explicitly(monkeypatch):
    calls = _record(monkeypatch, "clear_variables")
    await srv.vars(action="clear_all")
    assert calls[0][1]["clear_all"] is True


async def test_vars_clear_with_name_still_clears(monkeypatch):
    calls = _record(monkeypatch, "clear_variables")
    await srv.vars(action="clear", name="x")
    assert calls[0][1]["names"] == ["x"]
    assert calls[0][1]["clear_all"] is False


# ---- evaluate: ambiguous combos rejected, note attached ----------------------


async def test_evaluate_code_plus_file_rejected():
    out = json.loads(await srv.evaluate(code="1+1", file="/tmp/x.wl"))
    assert out["success"] is False
    assert "code" in out["error"] and "file" in out["error"]


async def test_evaluate_dry_run_plus_file_rejected():
    out = json.loads(await srv.evaluate(file="/tmp/x.wl", dry_run=True))
    assert out["success"] is False
    assert "dry_run" in out["error"]


async def test_evaluate_cell_nondefault_timeout_gets_note(monkeypatch):
    _record(monkeypatch, "evaluate_cell", ret=json.dumps({"success": True}))
    out = json.loads(await srv.evaluate(target="cell", cell_id="c1", timeout=500))
    assert "timeout" in out["note"]


async def test_evaluate_cell_default_timeout_no_note(monkeypatch):
    _record(monkeypatch, "evaluate_cell", ret=json.dumps({"success": True}))
    out = json.loads(await srv.evaluate(target="cell", cell_id="c1"))
    assert "note" not in out


async def test_evaluate_file_nondefault_timeout_gets_note(monkeypatch):
    _record(monkeypatch, "run_script", ret=json.dumps({"success": True}))
    out = json.loads(await srv.evaluate(file="/tmp/x.wl", timeout=10))
    assert "timeout" in out["note"]


# ---- notebooks: per-action format validation ---------------------------------


async def test_notebooks_save_rejects_markdown(monkeypatch):
    calls = _record(monkeypatch, "save_notebook")
    out = json.loads(await srv.notebooks(action="save", format="Markdown"))
    assert out["success"] is False
    assert "Notebook|PDF|HTML|TeX" in out["error"]
    assert calls == []


async def test_notebooks_export_rejects_notebook_format(monkeypatch):
    calls = _record(monkeypatch, "export_notebook")
    out = json.loads(await srv.notebooks(action="export", path="/tmp/x.pdf", format="Notebook"))
    assert out["success"] is False
    assert "PDF|HTML|TeX|Markdown" in out["error"]
    assert calls == []


async def test_notebooks_export_valid_format_passes(monkeypatch):
    calls = _record(monkeypatch, "export_notebook")
    await srv.notebooks(action="export", path="/tmp/x.md", format="Markdown")
    assert calls[0][0][2] == "Markdown"  # (path, notebook, format, session_id)


# ---- corrective errors: example call + cursor next_step ----------------------


async def test_lean_bad_includes_example_call():
    out = json.loads(await srv.vars(action="bogus"))
    assert out["success"] is False
    assert "e.g. vars(action='list')" in out["error"]


async def test_unknown_cursor_has_next_step():
    out = json.loads(await srv.evaluate(cursor="curmissing:0"))
    assert out["success"] is False
    assert "without cursor=" in out["next_step"]


# ---- followups: create / dry_run / has_errors --------------------------------


async def test_notebooks_create_success_has_followup(monkeypatch):
    _record(monkeypatch, "create_notebook", ret=json.dumps({"success": True, "id": "nb1"}))
    out = json.loads(await srv.notebooks(action="create", title="T"))
    assert out["available_followups"] == ["evaluate(code=..., target='notebook') to run code in it"]


async def test_notebooks_create_failure_has_no_followup(monkeypatch):
    _record(monkeypatch, "create_notebook", ret=json.dumps({"error": "no frontend"}))
    out = json.loads(await srv.notebooks(action="create", title="T"))
    assert "available_followups" not in out


async def test_evaluate_dry_run_success_has_followup(monkeypatch):
    _record(monkeypatch, "check_syntax", ret=json.dumps({"success": True, "valid": True}))
    out = json.loads(await srv.evaluate(code="1+1", dry_run=True))
    assert out["available_followups"] == ["re-run without dry_run to execute"]


def _finalize(response: dict) -> dict:
    return json.loads(
        srv._finalize_execute_response(
            response,
            route_variant="compute",
            execution_path="addon_cli",
            fell_back=False,
            start_time=time.monotonic(),
        )
    )


_ERROR_MESSAGES = [{"tag": "Part::partw", "text": "Part 5 does not exist", "type": "error"}]


def test_error_response_next_step_and_followup_lean(monkeypatch):
    monkeypatch.setattr(srv, "FEATURES", FeatureFlags.from_env("lean"))
    out = _finalize({"success": True, "code": "x[[5]]", "output": "", "messages": list(_ERROR_MESSAGES)})
    ea = out["error_analysis"]
    assert ea["retry_with"] is None
    assert "kernel(action='messages')" in ea["next_step"]
    assert "guide(topic='errors')" in ea["next_step"]
    assert out["available_followups"] == ["kernel(action='messages')"]


def test_error_response_next_step_classic_wording():
    # conftest pins the in-process suite to the full/classic profile.
    out = _finalize({"success": True, "code": "x[[5]]", "output": "", "messages": list(_ERROR_MESSAGES)})
    assert "get_messages()" in out["error_analysis"]["next_step"]
    assert out["available_followups"] == ["get_messages()"]


# ---- error_analyzer: retry_with honesty --------------------------------------


def test_no_pattern_ships_canned_retry_with():
    """A static retry_with can never substitute the user's failing input, so no
    pattern may define one. Pre-fix, UnitConvert::compat and Divide::infy did."""
    offenders = [tag for tag, info in ERROR_PATTERNS.items() if "retry_with" in info]
    assert offenders == []


def test_analyzer_never_emits_retry_with_for_any_known_tag():
    for tag in ERROR_PATTERNS:
        result = analyze_messages([{"tag": tag, "text": "boom", "type": "error"}])
        assert result["retry_with"] is None, tag


# ---- C1: cold/regex sites migrated to the warm JSON-first funnel -------------


def _fake_wl(monkeypatch, payload: dict):
    """Patch session.evaluate_wl with a warm fake returning *payload* as JSON;
    return the list of WL code strings it received."""
    import mathematica_mcp.session as session_mod

    calls: list[str] = []

    def fake_evaluate_wl(code: str, timeout: int = 60):
        calls.append(code)
        return session_mod.WLResult(text=json.dumps(payload), success=True, execution_method="wolframclient")

    monkeypatch.setattr(session_mod, "evaluate_wl", fake_evaluate_wl)
    return calls


def test_hydrate_usage_uses_warm_funnel_and_caches(monkeypatch):
    import mathematica_mcp.symbol_index as idx

    calls = _fake_wl(monkeypatch, {"FooBar": "FooBar[x] does foo."})
    monkeypatch.setattr(idx, "get_cached_metadata", lambda s: None)
    cached: dict = {}
    monkeypatch.setattr(idx, "cache_metadata", lambda s, m: cached.update({s: m}))

    out = srv._hydrate_usage(["FooBar"])
    assert out == {"FooBar": "FooBar[x] does foo."}
    assert cached["FooBar"]["usage"] == "FooBar[x] does foo."
    # JSON-first: the code went through the _json_wl ExportString wrapper.
    assert calls and "ExportString" in calls[0]


def test_symbol_lookup_fallback_uses_warm_funnel(monkeypatch):
    import mathematica_mcp.symbol_index as idx

    monkeypatch.setattr(idx, "search", lambda q, max_results=20: [])
    calls = _fake_wl(
        monkeypatch,
        {"success": True, "query": "Foo", "matches": [{"symbol": "FooBar", "usage": "u"}]},
    )

    out = srv._lookup_symbols_in_kernel("Foo")
    assert out["success"] is True
    assert out["candidates"] == [{"symbol": "FooBar", "usage": "u"}]
    assert calls and "ExportString" in calls[0]


async def test_convert_notebook_latex_uses_warm_funnel(monkeypatch, tmp_path):
    nb = tmp_path / "t.nb"
    nb.write_text("Notebook[{}]")
    monkeypatch.setattr(srv, "_load_cached_notebook", lambda path, truncation_threshold=25000: object())
    calls = _fake_wl(monkeypatch, {"success": True, "format": "latex", "content": "\\\\documentclass"})

    out = json.loads(await srv.convert_notebook(str(nb), output_format="latex"))
    assert out["success"] is True
    assert out["content"] == "\\\\documentclass"
    assert out["execution_method"] == "wolframclient"
    assert calls and "ExportString" in calls[0]


# ---- prompts: lean profile speaks lean vocabulary -----------------------------

_PROMPT_SCRIPT = """
import json
from mathematica_mcp import server as srv
print(json.dumps({
    "calculate": srv.calculate("1+1"),
    "notebook": srv.notebook("Plot[x,{x,0,1}]"),
    "new_notebook": srv.new_notebook("t", "T"),
    "interactive": srv.interactive("Manipulate[x,{x,0,1}]"),
    "quickstart": srv.quickstart(),
    "mathematica_expert": srv.mathematica_expert("hi"),
}))
"""


def test_lean_prompts_contain_no_classic_vocabulary():
    env = {k: v for k, v in os.environ.items() if not k.startswith("MATHEMATICA_")}
    env["MATHEMATICA_PROFILE"] = "lean"
    env["PYTHONPATH"] = str(SRC_ROOT)
    out = subprocess.run(
        [sys.executable, "-c", _PROMPT_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    ).stdout
    prompts = json.loads(out)
    for name, text in prompts.items():
        for forbidden in ("execute_code", "style=", "create_notebook(", "get_session_brief"):
            assert forbidden not in text, f"{name} contains classic vocabulary {forbidden!r}"
    assert "evaluate(code" in prompts["calculate"]
    assert "target='notebook'" in prompts["notebook"] or 'target="notebook"' in prompts["notebook"]


def test_classic_prompts_keep_legacy_text():
    # In-process profile is full (conftest): wiring must keep the classic strings.
    assert "style='compute'" in srv.calculate("1+1")
    assert "style='notebook'" in srv.notebook("t")
    assert "create_notebook()" in srv.new_notebook("t")
    assert "style='interactive'" in srv.interactive("t")
    assert 'execute_code(style="notebook")' in srv.quickstart()
