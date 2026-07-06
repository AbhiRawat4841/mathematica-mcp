"""WL composition safety & flagship correctness (Domain B).

Covers:
- B1: ``_wl_string`` escapes backslash then double-quote and returns a quoted literal,
  so user strings can't break out of a WL string literal (injection).
- B2: ``verify_derivation`` emits ``ExportString[..., "JSON"]`` and its Python side
  parses via ``json.loads`` (not the fragile OutputForm-Association regex).
- B3: ``verify_derivation`` evaluates user steps inside a scratch context (``_scratch_block``).

The string-level tests need no kernel. The ``wolfram_runtime`` tests DO need a live
kernel and are meant to be run by a human, not in this phase.
"""

from __future__ import annotations

import json

import pytest

from mathematica_mcp import lazy_wolfram_tools as L
from mathematica_mcp import server as srv
from mathematica_mcp import session as S
from mathematica_mcp.session import WLResult


# ---------------------------------------------------------------------------
# B1 — _wl_string escaping
# ---------------------------------------------------------------------------
def test_wl_string_escapes_quote_and_backslash():
    # Plain text is just quoted.
    assert L._wl_string("Meters") == '"Meters"'
    # Embedded double-quotes are escaped, wrapped in outer quotes.
    assert L._wl_string('Quantity[1, "Meters"]') == '"Quantity[1, \\"Meters\\"]"'
    # Backslash is escaped BEFORE the quote so it can't consume the quote-escape.
    assert L._wl_string(chr(92)) == '"' + chr(92) * 2 + '"'
    assert L._wl_string('a\\"b') == '"a\\\\\\"b"'


# ---------------------------------------------------------------------------
# B2/B3 — verify_derivation generates escaped, JSON-exporting, scratch-scoped WL
# ---------------------------------------------------------------------------
async def _capture_verify_wl(monkeypatch, steps):
    """Run verify_derivation with a stubbed kernel that records the generated WL."""
    captured = {}

    def _fake_evaluate_wl(code, timeout=60):
        captured["code"] = code
        return WLResult(
            text='{"steps": [], "all_valid": true, "valid_count": 0, "total_steps": 1}',
            success=True,
            execution_method="wolframclient",
        )

    monkeypatch.setattr(S, "evaluate_wl", _fake_evaluate_wl)
    out = await L.verify_derivation(steps, parse_wolfram_association=lambda t: {})
    return captured["code"], json.loads(out)


async def test_verify_derivation_wl_escapes_quoted_step(monkeypatch):
    # Live-confirmed mangling input: quoted-string steps must survive intact.
    code, _ = await _capture_verify_wl(
        monkeypatch,
        ['Quantity[100, "Centimeters"]', 'Quantity[1, "Meters"]'],
    )
    # The escaped literal is present ...
    assert 'Quantity[100, \\"Centimeters\\"]' in code
    assert 'Quantity[1, \\"Meters\\"]' in code
    # ... and the old mangling (bare unescaped quotes closing the WL string) is gone.
    assert '"Quantity[100, "Centimeters"]"' not in code


async def test_verify_derivation_wl_exports_json_and_scopes(monkeypatch):
    code, out = await _capture_verify_wl(monkeypatch, ["x + x", "2 x"])
    # B2: the Module's final expression is a JSON export (not a bare Association).
    marker = 'ExportString[results, "JSON"]'
    assert marker in code
    # Nothing but whitespace and closing brackets follows the JSON export,
    # i.e. it really is the last thing the Module computes.
    tail = code[code.rindex(marker) + len(marker) :]
    assert set(tail) <= set(" \n\t]"), f"unexpected tail after JSON export: {tail!r}"
    # B3: user steps evaluate inside a throwaway context.
    assert 'Block[{$Context = "MCPScratch`"' in code
    # B2 Python side: json.loads path parses cleanly (no "Could not parse").
    assert out["success"] is True
    assert "Could not parse" not in out["report"]


async def test_verify_derivation_json_loads_beats_regex(monkeypatch):
    """A JSON payload with an expression-valued field parses via json.loads, which the
    old OutputForm-Association regex could not handle."""

    def _fake_evaluate_wl(code, timeout=60):
        payload = {
            "steps": [
                {
                    "from": 1,
                    "to": 2,
                    "expr_from": "Integrate[x^2 Sin[x], {x,0,Pi}]",
                    "expr_to": "-4 + Pi^2",
                    "valid": True,
                    "simplified": "-4 + Pi^2",
                }
            ],
            "all_valid": True,
            "valid_count": 1,
            "total_steps": 1,
        }
        return WLResult(text=json.dumps(payload), success=True, execution_method="wolframclient")

    monkeypatch.setattr(S, "evaluate_wl", _fake_evaluate_wl)

    def _explode(_text):  # regex fallback must NOT be needed
        raise AssertionError("parse_wolfram_association should not be called for JSON output")

    out = json.loads(
        await L.verify_derivation(
            ["Integrate[x^2 Sin[x], {x,0,Pi}]", "-4 + Pi^2"],
            parse_wolfram_association=_explode,
        )
    )
    assert out["success"] is True
    assert out["raw_data"]["all_valid"] is True
    assert "Could not parse" not in out["report"]
    assert "✓ VALID" in out["report"]


# ---------------------------------------------------------------------------
# _run_wl_parsed / _json_wl — JSON-first parsing for the migrated cold tools
# (kernel state/packages/load_package, cloud + graphics tools). Regression for
# the Codex finding that kernel(action="state"|"packages") returned
# {"parse_error": true, "raw": ...} instead of structured fields.
# ---------------------------------------------------------------------------
def test_json_wl_wraps_code_with_json_export_and_sanitizer():
    wrapped = L._json_wl('<|"a" -> 1|>')
    # Primary attempt: plain JSON export of the Association.
    assert 'ExportString[mcpRes, "JSON", "Compact" -> True]' in wrapped
    # Fallback: explicit recursive sanitizer (NOT ReplaceAll — it visits heads and
    # stringifies the Association head symbol itself, breaking the export).
    # Keys are sanitized too: an association KEYED by EntityProperty[...] would
    # otherwise still fail the JSON export (Codex finding).
    assert "mcpSan[x_Association] := Association[KeyValueMap[" in wrapped
    assert "If[StringQ[#1], #1, ToString[#1, InputForm]]" in wrapped
    assert 'ExportString[mcpSan[mcpRes], "JSON", "Compact" -> True]' in wrapped
    assert "/." not in wrapped  # no ReplaceAll anywhere in the wrapper
    # The caller's code is embedded verbatim.
    assert '<|"a" -> 1|>' in wrapped


async def test_run_wl_parsed_failed_result_is_not_success(monkeypatch):
    """A whole-result $Failed sanitizes to the JSON string "$Failed"; it must be
    labeled a failure, not {"success": true, "result": "$Failed"}."""

    def _fake_evaluate_wl(code, timeout=60):
        return WLResult(text='"$Failed"', success=True, execution_method="wolframclient")

    monkeypatch.setattr(S, "evaluate_wl", _fake_evaluate_wl)
    out = json.loads(await L._run_wl_parsed("$Failed", lambda t: {}))
    assert out["success"] is False
    assert out["result"] == "$Failed"


async def test_run_wl_parsed_json_first_no_parse_error(monkeypatch):
    captured = {}

    def _fake_evaluate_wl(code, timeout=60):
        captured["code"] = code
        return WLResult(
            text='{"success": true, "memory_in_use_mb": 123.4, "global_symbol_count": 7}',
            success=True,
            execution_method="wolframclient",
        )

    monkeypatch.setattr(S, "evaluate_wl", _fake_evaluate_wl)

    def _explode(_text):
        raise AssertionError("regex fallback must not run for JSON output")

    out = json.loads(await L._run_wl_parsed('<|"x" -> 1|>', _explode))
    # Structured fields survive; no raw/parse_error fallback.
    assert out["success"] is True
    assert out["memory_in_use_mb"] == 123.4
    assert "parse_error" not in out
    assert "raw" not in out
    assert out["execution_method"] == "wolframclient"
    # The kernel-side code was JSON-wrapped.
    assert 'ExportString[mcpRes, "JSON"' in captured["code"]


async def test_run_wl_parsed_falls_back_to_regex_on_non_json(monkeypatch):
    """If the kernel-side JSON export failed (last-resort raw Association), the
    old regex parser still runs so behavior degrades no worse than before."""

    def _fake_evaluate_wl(code, timeout=60):
        return WLResult(
            text="<|success -> True, x -> 3|>",
            success=True,
            execution_method="wolframclient",
        )

    monkeypatch.setattr(S, "evaluate_wl", _fake_evaluate_wl)
    called = {}

    def _regex(text):
        called["text"] = text
        return {"success": True, "raw": text, "parse_error": True}

    out = json.loads(await L._run_wl_parsed('<|"x" -> 3|>', _regex))
    assert called["text"] == "<|success -> True, x -> 3|>"
    assert out["parse_error"] is True


# ---------------------------------------------------------------------------
# LIVE — requires a real kernel. DO NOT run in this phase; a human runs these.
# ---------------------------------------------------------------------------
@pytest.mark.wolfram_runtime
async def test_live_kernel_state_is_structured(require_wolfram_runtime):
    if S.get_kernel_session() is None:
        pytest.skip("no persistent kernel session")
    state = json.loads(await srv.get_kernel_state())
    assert state.get("success") is True
    assert "parse_error" not in state
    assert "raw" not in state
    assert isinstance(state.get("global_symbol_count"), int)
    assert isinstance(state.get("memory_in_use_mb"), (int, float))


@pytest.mark.wolfram_runtime
async def test_live_list_loaded_packages_is_structured(require_wolfram_runtime):
    if S.get_kernel_session() is None:
        pytest.skip("no persistent kernel session")
    pkgs = json.loads(await srv.list_loaded_packages())
    assert pkgs.get("success") is True
    assert "parse_error" not in pkgs
    assert isinstance(pkgs.get("packages"), list)
    assert isinstance(pkgs.get("count"), int)


@pytest.mark.wolfram_runtime
async def test_live_non_json_safe_association_sanitizes(require_wolfram_runtime):
    """Adversarial regression: Quantity/Rational/Entity-rule values (the
    entity_lookup shape) must sanitize to structured JSON — no parse_error.
    The old ReplaceAll-based fallback corrupted the Association head and could
    never succeed."""
    if S.get_kernel_session() is None:
        pytest.skip("no persistent kernel session")
    nasty = (
        '<|"success" -> True, "q" -> Quantity[1.5, "Meters"], "r" -> 1/3, '
        '"l" -> {1, Pi}, "nested" -> <|"h" -> Hold[2 + 2]|>, '
        '"rules" -> {EntityProperty["City", "Population"] -> 123}, '
        '"keyed" -> <|EntityProperty["City", "Population"] -> 42|>|>'
    )
    from mathematica_mcp.server import _parse_wolfram_association as P

    out = json.loads(await L._run_wl_parsed(nasty, P))
    assert out.get("success") is True
    assert "parse_error" not in out, out.get("raw", "")[:200]
    assert out["q"] == 'Quantity[1.5, "Meters"]'
    assert out["r"] == "1/3"
    assert out["l"] == [1, "Pi"]
    assert out["nested"] == {"h": "Hold[2 + 2]"}
    assert out["rules"] == {'EntityProperty["City", "Population"]': 123}
    # Non-string ASSOCIATION KEYS must sanitize too (previously exported $Failed).
    assert out["keyed"] == {'EntityProperty["City", "Population"]': 42}


@pytest.mark.wolfram_runtime
async def test_live_verify_readme_example_parses(require_wolfram_runtime):
    if S.get_kernel_session() is None:
        pytest.skip("no persistent kernel session")
    out = json.loads(await srv.verify_derivation(["Integrate[x^2 Sin[x], {x, 0, Pi}]", "-4 + Pi^2"]))
    assert out["success"] is True
    assert "Could not parse" not in out["report"]
    assert out["raw_data"]["all_valid"] is True


@pytest.mark.wolfram_runtime
async def test_live_verify_quoted_step_not_mangled(require_wolfram_runtime):
    if S.get_kernel_session() is None:
        pytest.skip("no persistent kernel session")
    out = json.loads(await srv.verify_derivation(['Quantity[100, "Centimeters"]', 'Quantity[1, "Meters"]']))
    assert out["success"] is True
    assert "Could not parse" not in out["report"]
    # 100 cm == 1 m, so the single step must verify as valid.
    assert out["raw_data"]["all_valid"] is True
