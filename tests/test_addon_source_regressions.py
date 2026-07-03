from __future__ import annotations

from pathlib import Path

ADDON_SOURCE = Path(__file__).resolve().parents[1] / "addon" / "MathematicaMCP.wl"


def _addon_source() -> str:
    return ADDON_SOURCE.read_text(encoding="utf-8")


def test_addon_get_notebooks_returns_dispatcher_association():
    source = _addon_source()

    assert '"notebooks" -> Map[notebookToAssoc, nbs]' in source
    assert '"count" -> Length[nbs]' in source


def test_addon_status_handles_inactive_frontend():
    source = _addon_source()

    assert 'frontEndInfo = Quiet[Check[SystemInformation["FrontEnd"], $Failed]]' in source
    assert "AssociationQ[frontEndInfo]" in source
    assert "If[!ListQ[nbs], nbs = {}]" in source


def test_addon_notebook_creation_rejects_failed_frontend_document():
    source = _addon_source()

    assert "If[!validNotebookQ[nb]," in source
    assert "Failed to create notebook. Mathematica front end may not be active." in source
    assert "No valid notebook available. Mathematica front end may not be active." in source


def test_addon_auth_token_is_string_before_string_length():
    source = _addon_source()

    assert 'If[!StringQ[$MCPAuthToken], $MCPAuthToken = ""];' in source
    assert "StringQ[$MCPAuthToken] && StringLength[$MCPAuthToken] > 0" in source


def test_addon_multiline_code_parses_as_single_held_expression():
    source = _addon_source()

    assert "parseHeldCode[code_String]" in source
    assert 'ToExpression["(" <> code <> ")", InputForm, HoldComplete]' in source
    assert "ToExpression[code, InputForm, HoldComplete]" not in source


def test_addon_frontend_evaluation_waits_for_observed_start():
    source = _addon_source()

    assert 'FrontEndTokenExecute[nb, "EvaluateCells"]' in source
    assert "observedEvaluation" in source
