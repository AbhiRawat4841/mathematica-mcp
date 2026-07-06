"""Domain B: optional repository/symbol tools route through the warm funnel (mocked).

The 9 former cold ``subprocess.run wolframscript`` sites in
optional_repository_tools.py and optional_symbol_tools.py now go through
``lazy_wolfram_tools._run_wl_parsed`` -> ``session.evaluate_wl``: warm-first,
execution_method attached, user text escaped via ``_wl_string``, and no
``subprocess`` usage left in either module.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from types import SimpleNamespace

import pytest

from mathematica_mcp import optional_repository_tools as R
from mathematica_mcp import optional_symbol_tools as Y
from mathematica_mcp import session as S
from mathematica_mcp.lazy_wolfram_tools import _wl_string


def _parser(text):
    # Stand-in for _parse_wolfram_association; wiring is under test, not parsing.
    return {"success": True, "raw": text}


def _tool_fns(register, **kwargs):
    from mcp.server.fastmcp import FastMCP

    m = FastMCP("test-optional-warm")
    register(m, **kwargs)
    return {name: tool.fn for name, tool in m._tool_manager._tools.items()}


def _symbol_kwargs(lookup=None):
    return {
        "lookup_symbols_in_kernel": lookup or (lambda q: {"success": True, "candidates": [], "system_only": False}),
        "hydrate_usage": lambda syms: {s: f"{s}[x] does things" for s in syms},
        "extract_short_description": lambda usage: usage,
        "extract_example_signature": lambda usage, sym: "",
        "rank_candidates": lambda q, cands: [dict(c, _score=100) for c in cands],
        "parse_wolfram_association": _parser,
        "execute_code": None,  # only needed for auto_execute paths, unused here
    }


class _Calls:
    def __init__(self):
        self.codes: list[str] = []
        self.payload: dict = {"success": True}


@pytest.fixture
def warm_eval(monkeypatch):
    """Intercept the funnel boundary: session.evaluate_wl returns a warm result.

    Any subprocess.run during the call is a regression back to the cold path.
    """
    calls = _Calls()

    def fake_evaluate_wl(code, timeout=60):
        calls.codes.append(code)
        return SimpleNamespace(
            text=json.dumps(calls.payload),
            success=True,
            execution_method="wolframclient",
            error="",
            timed_out=False,
        )

    def no_subprocess(*args, **kwargs):
        raise AssertionError(f"cold subprocess spawned: {args!r}")

    monkeypatch.setattr(S, "evaluate_wl", fake_evaluate_wl, raising=False)
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    return calls


def test_no_subprocess_in_migrated_modules():
    # Fails pre-migration: both modules used subprocess.run(wolframscript, ...).
    for mod in (R, Y):
        src = inspect.getsource(mod)
        assert "import subprocess" not in src, mod.__name__
        assert "subprocess.run" not in src, mod.__name__


# ---------------------------------------------------------------- repository


REPO_CASES = [
    (
        R.register_function_repository_tools,
        "search_function_repository",
        ("plot",),
        {"success": True, "query": "plot", "count": 1, "results": [{"name": "F"}]},
        ["query", "count", "results"],
    ),
    (
        R.register_function_repository_tools,
        "get_function_repository_info",
        ("BirdSay",),
        {"success": True, "name": "BirdSay", "description": "d"},
        ["name", "description"],
    ),
    (
        R.register_function_repository_tools,
        "load_resource_function",
        ("BirdSay",),
        {"success": True, "function": "BirdSay", "loaded": True, "usage": "u", "message": "m"},
        ["function", "loaded", "usage", "message"],
    ),
    (
        R.register_data_repository_tools,
        "search_data_repository",
        ("iris",),
        {"success": True, "query": "iris", "count": 0, "datasets": []},
        ["query", "count", "datasets"],
    ),
    (
        R.register_data_repository_tools,
        "get_dataset_info",
        ("Fisher's Irises",),
        {"success": True, "name": "Fisher's Irises", "description": "d", "keywords": []},
        ["name", "description", "keywords"],
    ),
    (
        R.register_data_repository_tools,
        "load_dataset",
        ("Fisher's Irises", 5),
        {"success": True, "name": "Fisher's Irises", "loaded": True, "type": "Dataset", "sample": "..."},
        ["name", "loaded", "type", "sample"],
    ),
]


@pytest.mark.parametrize("register, tool, args, payload, keys", REPO_CASES, ids=[c[1] for c in REPO_CASES])
async def test_repository_tool_is_warm_and_shape_preserved(warm_eval, register, tool, args, payload, keys):
    fns = _tool_fns(register, parse_wolfram_association=_parser)
    warm_eval.payload = payload
    out = json.loads(await fns[tool](*args))
    assert out["success"] is True
    for key in keys:
        assert out[key] == payload[key]
    assert out["execution_method"] == "wolframclient"
    assert len(warm_eval.codes) == 1  # exactly one trip through the warm funnel


async def test_funnel_error_shape(monkeypatch):
    def fail_evaluate_wl(code, timeout=60):
        return SimpleNamespace(text="", success=False, execution_method="none", error="no kernel", timed_out=False)

    monkeypatch.setattr(S, "evaluate_wl", fail_evaluate_wl, raising=False)
    fns = _tool_fns(R.register_function_repository_tools, parse_wolfram_association=_parser)
    out = json.loads(await fns["search_function_repository"]("plot"))
    assert out["success"] is False
    assert out["error"] == "no kernel"
    assert out["execution_method"] == "none"


async def test_user_text_escaped_into_wl_literal(warm_eval):
    # search_data_repository interpolated the raw query pre-fix (WL injection).
    fns = _tool_fns(R.register_data_repository_tools, parse_wolfram_association=_parser)
    evil = 'x"; Quit[]; "'
    warm_eval.payload = {"success": True, "query": evil, "count": 0, "datasets": []}
    await fns["search_data_repository"](evil)
    code = warm_eval.codes[0]
    assert _wl_string(evil) in code  # escaped literal, not raw interpolation
    assert '-> "x"; Quit[]; ""' not in code


# -------------------------------------------------------------------- symbol


async def test_get_symbol_info_warm_scratch_blocked(warm_eval, monkeypatch):
    from mathematica_mcp import symbol_index

    cached: dict = {}
    monkeypatch.setattr(symbol_index, "get_cached_metadata", lambda s: None)
    monkeypatch.setattr(symbol_index, "cache_metadata", lambda s, m: cached.update({s: m}))

    fns = _tool_fns(Y.register_symbol_lookup_tools, **_symbol_kwargs())
    warm_eval.payload = {
        "success": True,
        "symbol": "Sin",
        "usage": "Sin[z] gives the sine of z.",
        "attributes": ["Listable", "Protected"],
        "options_count": 0,
        "options": [],
        "related_symbols": ["Cos"],
        "context": "System`",
    }
    out = json.loads(await fns["get_symbol_info"]("Sin"))
    assert out["success"] is True
    assert out["symbol"] == "Sin"
    assert out["attributes"] == ["Listable", "Protected"]
    assert out["execution_method"] == "wolframclient"
    # ToExpression on caller text is isolated from the shared kernel's Global`.
    assert "MCPScratch`" in warm_eval.codes[0]
    # Cached metadata must not freeze the execution method of the first lookup.
    assert "execution_method" not in cached["Sin"]


async def test_suggest_similar_functions_fallback_warm(warm_eval, monkeypatch):
    from mathematica_mcp import symbol_index

    monkeypatch.setattr(symbol_index, "search", lambda q, max_results=20: [])
    fns = _tool_fns(Y.register_symbol_lookup_tools, **_symbol_kwargs())
    warm_eval.payload = {"success": True, "query": "fourier", "matches": []}
    out = json.loads(await fns["suggest_similar_functions"]("fourier"))
    assert out["success"] is True
    assert out["matches"] == []
    assert out["execution_method"] == "wolframclient"


async def test_resolve_function_global_lookup_warm(warm_eval):
    fns = _tool_fns(Y.register_symbol_lookup_tools, **_symbol_kwargs())
    warm_eval.payload = {"success": True, "names": ["myPlotHelper"]}
    out = json.loads(await fns["resolve_function"]("myPlotHelper", auto_execute=False))
    assert out["status"] == "resolved"
    assert out["resolved_symbol"] == "myPlotHelper"
    assert len(warm_eval.codes) == 1
    assert "Names[" in warm_eval.codes[0]


# ------------------------------------------------- live (marked, not run in mocked CI)


@pytest.mark.wolfram_runtime
@pytest.mark.needs_network
async def test_live_search_function_repository(require_wolfram_runtime):
    from mathematica_mcp.server import _parse_wolfram_association

    fns = _tool_fns(R.register_function_repository_tools, parse_wolfram_association=_parse_wolfram_association)
    out = json.loads(await fns["search_function_repository"]("BirdSay", max_results=2))
    assert out["success"] is True
    assert "results" in out
    assert "execution_method" in out


@pytest.mark.wolfram_runtime
@pytest.mark.needs_network
async def test_live_get_symbol_info(require_wolfram_runtime):
    from mathematica_mcp import symbol_index
    from mathematica_mcp.server import _parse_wolfram_association

    was_cached = symbol_index.get_cached_metadata("Sin")
    kwargs = _symbol_kwargs()
    kwargs["parse_wolfram_association"] = _parse_wolfram_association
    fns = _tool_fns(Y.register_symbol_lookup_tools, **kwargs)
    out = json.loads(await fns["get_symbol_info"]("Sin"))
    assert out["success"] is True
    if was_cached is None:  # fresh lookup went through the funnel
        assert "execution_method" in out


async def test_get_symbol_info_wl_guards_undefined_usage(warm_eval, monkeypatch):
    """Codex finding: get_symbol_info's WL did ToString[sym::usage] unguarded, so an
    undefined symbol echoed the literal 'Sym::usage' as usage AND cached it."""
    from mathematica_mcp import symbol_index

    monkeypatch.setattr(symbol_index, "get_cached_metadata", lambda s: None)
    monkeypatch.setattr(symbol_index, "cache_metadata", lambda s, m: None)

    fns = _tool_fns(Y.register_symbol_lookup_tools, **_symbol_kwargs())
    warm_eval.payload = {"success": True, "symbol": "NoSuchSymbolDefinitely", "usage": ""}
    await fns["get_symbol_info"]("NoSuchSymbolDefinitely")

    code = warm_eval.codes[-1]
    assert "If[StringQ[u], u" in code, "usage must be StringQ-guarded"
    assert "ToString[sym::usage]" not in code
