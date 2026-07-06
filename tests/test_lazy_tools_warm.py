"""P1: migrated lazy tools route through the warm session (mocked).

Confirms the cold-`wolframscript` tools now go through session.evaluate_wl (warm),
attach execution_method, and spawn no cold subprocess when the session is up.
"""

from __future__ import annotations

import json

import pytest

from mathematica_mcp import lazy_wolfram_tools as L
from mathematica_mcp import session as S


@pytest.fixture(autouse=True)
def _reset_cold_counter():
    S.reset_cold_execution_count()
    yield
    S.reset_cold_execution_count()


class _FakeSession:
    def evaluate(self, _expr):
        return "<|success -> True, ok -> True|>"


def _parser(text):
    # Stand-in for _parse_wolfram_association; we only assert wiring, not parsing.
    return {"success": True, "raw": text}


@pytest.fixture
def warm(monkeypatch):
    monkeypatch.setattr(S, "get_kernel_session", lambda: _FakeSession())


@pytest.mark.parametrize(
    "call",
    [
        lambda: L.get_constant("Pi", parse_wolfram_association=_parser),
        lambda: L.convert_units("5 m", "ft", parse_wolfram_association=_parser),
        lambda: L.list_loaded_packages(parse_wolfram_association=_parser),
        lambda: L.get_kernel_state(parse_wolfram_association=_parser),
        lambda: L.inspect_graphics("Graphics[Disk[]]", parse_wolfram_association=_parser),
    ],
)
async def test_migrated_tool_is_warm(warm, call):
    result = json.loads(await call())
    assert result["execution_method"] == "wolframclient"
    assert S.cold_execution_count() == 0


async def test_verify_derivation_is_warm(warm, monkeypatch):
    # verify_derivation's WL returns a steps assoc; use a parser that yields it.
    monkeypatch.setattr(
        S,
        "get_kernel_session",
        lambda: type("F", (), {"evaluate": lambda self, e: '<|"steps" -> {}, "all_valid" -> True|>'})(),
    )
    monkeypatch.setattr(
        "mathematica_mcp.server._parse_wolfram_association",
        _parser,
        raising=False,
    )
    out = json.loads(
        await L.verify_derivation(["x+x", "2 x"], parse_wolfram_association=lambda t: {"steps": [], "all_valid": True})
    )
    assert out["success"] is True
    assert out["execution_method"] == "wolframclient"
    assert S.cold_execution_count() == 0


async def test_interpret_natural_language_warm_json(monkeypatch):
    # WL side returns a JSON string; evaluate_wl renders it verbatim via OutputForm.
    monkeypatch.setattr(
        S,
        "get_kernel_session",
        lambda: type("F", (), {"evaluate": lambda self, e: '{"success": true, "result": "42"}'})(),
    )
    out = json.loads(await L.interpret_natural_language("what is 6*7"))
    assert out["success"] is True
    assert out["result"] == "42"
    assert out["execution_method"] == "wolframclient"
    assert S.cold_execution_count() == 0
