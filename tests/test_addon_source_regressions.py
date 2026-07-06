from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
ADDON_SOURCE = _REPO_ROOT / "addon" / "MathematicaMCP.wl"
CONNECTION_SOURCE = _REPO_ROOT / "src" / "mathematica_mcp" / "connection.py"


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


def test_addon_state_delta_gated_to_notebook_commands():
    source = _addon_source()

    match = re.search(r"\$MCPStateDeltaCommands\s*=\s*\{(.*?)\};", source, re.DOTALL)
    assert match, "$MCPStateDeltaCommands list missing from addon source"
    commands = re.findall(r'"([^"]+)"', match.group(1))
    assert "write_cell" in commands
    assert "execute_code" not in commands
    # The delta is attached only for allowlisted commands, not unconditionally.
    assert "MemberQ[$MCPStateDeltaCommands, command]" in source


def test_addon_state_delta_kernel_busy_is_honest():
    source = _addon_source()

    # The selected-notebook branch reads the real front-end evaluation state.
    assert '"kernel_busy" -> TrueQ[Quiet[CurrentValue[nb, Evaluating]]]' in source
    # Hardcoded False survives only in the no-notebook and error-fallback branches.
    assert source.count('"kernel_busy" -> False') == 2


def test_protocol_version_3_in_lockstep():
    assert "$MCPProtocolVersion = 3" in _addon_source()
    assert "ADDON_PROTOCOL_VERSION = 3" in CONNECTION_SOURCE.read_text(encoding="utf-8")
