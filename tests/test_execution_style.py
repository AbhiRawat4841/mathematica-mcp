"""Tests for the execution style parameter and resolver."""

from __future__ import annotations

import asyncio
import json

import pytest

from mathematica_mcp import server as srv
from mathematica_mcp.server import _is_interactive_code, _normalize_response_detail, _resolve_execution_params


class TestResolveExecutionParams:
    """Tests for the pure _resolve_execution_params() resolver."""

    def test_compute_style(self):
        ot, mode = _resolve_execution_params("compute", None, None, "cli")
        assert ot == "cli"
        assert mode == "kernel"

    def test_notebook_style(self):
        ot, mode = _resolve_execution_params("notebook", None, None, "cli")
        assert ot == "notebook"
        assert mode == "kernel"

    def test_interactive_style(self):
        ot, mode = _resolve_execution_params("interactive", None, None, "cli")
        assert ot == "notebook"
        assert mode == "frontend"

    def test_explicit_output_target_overrides_style(self):
        ot, mode = _resolve_execution_params("compute", "notebook", None, "cli")
        assert ot == "notebook"
        assert mode == "kernel"

    def test_explicit_mode_overrides_style(self):
        ot, mode = _resolve_execution_params("notebook", None, "frontend", "cli")
        assert ot == "notebook"
        assert mode == "frontend"

    def test_cli_normalizes_mode_to_kernel(self):
        """CLI path ignores mode entirely, so resolver canonicalizes to kernel."""
        ot, mode = _resolve_execution_params("compute", None, "frontend", "cli")
        assert ot == "cli"
        assert mode == "kernel"

    def test_cli_normalizes_mode_explicit_output_target(self):
        """Even with explicit output_target=cli and mode=frontend, mode is normalized."""
        ot, mode = _resolve_execution_params(None, "cli", "frontend", "notebook")
        assert ot == "cli"
        assert mode == "kernel"

    def test_no_style_no_explicit_uses_profile_default_cli(self):
        ot, mode = _resolve_execution_params(None, None, None, "cli")
        assert ot == "cli"
        assert mode == "kernel"

    def test_no_style_no_explicit_uses_profile_default_notebook(self):
        ot, mode = _resolve_execution_params(None, None, None, "notebook")
        assert ot == "notebook"
        assert mode == "kernel"

    def test_no_style_explicit_output_target(self):
        ot, mode = _resolve_execution_params(None, "notebook", "frontend", "cli")
        assert ot == "notebook"
        assert mode == "frontend"

    def test_invalid_style_fast(self):
        with pytest.raises(ValueError, match="Unknown style 'fast'"):
            _resolve_execution_params("fast", None, None, "cli")

    def test_invalid_style_live(self):
        with pytest.raises(ValueError, match="Unknown style 'live'"):
            _resolve_execution_params("live", None, None, "cli")

    def test_invalid_style_inline(self):
        with pytest.raises(ValueError, match="Unknown style 'inline'"):
            _resolve_execution_params("inline", None, None, "cli")

    def test_style_none_is_valid(self):
        """style=None should not raise."""
        ot, mode = _resolve_execution_params(None, None, None, "cli")
        assert ot == "cli"
        assert mode == "kernel"

    def test_all_explicit_overrides_style_completely(self):
        """When both output_target and mode are explicit, style defaults are irrelevant."""
        ot, mode = _resolve_execution_params("interactive", "cli", "kernel", "notebook")
        assert ot == "cli"
        assert mode == "kernel"


# ---------------------------------------------------------------------------
# response_detail parameter
# ---------------------------------------------------------------------------


class TestResponseDetail:
    def test_aliases_normalize(self):
        assert _normalize_response_detail("short") == "compact"
        assert _normalize_response_detail("medium") == "standard"
        assert _normalize_response_detail("long") == "verbose"

    def test_invalid_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown response_detail 'tiny'"):
            _normalize_response_detail("tiny")

    def test_default_detail_is_standard(self):
        import inspect

        from mathematica_mcp.server import execute_code

        sig = inspect.signature(execute_code)
        assert sig.parameters["response_detail"].default == "standard"

    def test_execute_code_schema_exposes_style_and_response_detail_aliases(self):
        from mathematica_mcp.server import mcp

        async def _load_schema():
            tools = await mcp.list_tools()
            for tool in tools:
                if tool.name == "execute_code":
                    return tool.inputSchema
            raise AssertionError("execute_code tool not found")

        schema = asyncio.run(_load_schema())
        properties = schema["properties"]

        assert "style" in properties
        assert "response_detail" in properties
        assert set(properties["response_detail"]["enum"]) == {
            "compact",
            "standard",
            "verbose",
            "diagnostic",
            "short",
            "medium",
            "long",
        }


class TestInteractiveDetection:
    """Word-anchored detection of interactive heads for auto-routing."""

    @pytest.mark.parametrize(
        "code",
        [
            "Manipulate[Plot[Sin[n x], {x, 0, 10}], {n, 1, 5}]",
            "Column[{Slider[Dynamic[x]], Dynamic[x]}]",  # Dynamic nested in code
            "DynamicModule [{x = 0}, Slider[Dynamic[x]]]",  # whitespace before [
            "Animate[Plot[Sin[a x], {x, 0, 6}], {a, 1, 5}]",
            "ListAnimate[Table[Plot[Sin[k x], {x, 0, 6}], {k, 1, 5}]]",
        ],
    )
    def test_positives(self, code):
        assert _is_interactive_code(code) is True

    @pytest.mark.parametrize(
        "code",
        [
            "1 + 1",
            "Integrate[x^2, x]",
            "ManipulateData[foo]",  # symbol name, not a call to Manipulate
            "ManipulatePlot[Sin[x], {x, 0, 1}]",  # ManipulatePlot-like name must NOT match
            "myDynamicVar + 1",  # Dynamic mid-identifier must NOT match
        ],
    )
    def test_negatives(self, code):
        assert _is_interactive_code(code) is False

    def test_string_literal_is_acceptable_false_positive(self):
        # A head name inside a string matches the regex; frontend's pending
        # contract makes this harmless (documented tradeoff).
        assert _is_interactive_code('Print["Manipulate[x]"]') is True


def _fake_addon_pending(captured: dict):
    async def fake_addon_result(command, params=None, timeout=None):
        captured["command"] = command
        captured["params"] = params
        return {
            "success": True,
            "status": "evaluation_pending",
            "notebook_id": "nb-1",
            "cell_id": "c-1",
            "waited_seconds": 0.2,
        }

    return fake_addon_result


_MANIP = "Manipulate[Plot[Sin[n x], {x, 0, 10}], {n, 1, 5}]"


class TestInteractiveAutoRouting:
    """execute_code routes interactive code to frontend when style/mode unset."""

    async def test_manipulate_no_style_routes_frontend(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(srv, "_addon_result", _fake_addon_pending(captured))
        out = await srv.execute_code(_MANIP, output_target="notebook")
        assert captured["command"] == "execute_code_notebook"
        assert captured["params"]["mode"] == "frontend"
        data = json.loads(out)
        assert data.get("auto_routed")

    async def test_explicit_kernel_mode_wins(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(srv, "_addon_result", _fake_addon_pending(captured))
        out = await srv.execute_code(_MANIP, output_target="notebook", mode="kernel")
        assert captured["params"]["mode"] == "kernel"
        assert "auto_routed" not in json.loads(out)

    async def test_explicit_style_notebook_wins(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(srv, "_addon_result", _fake_addon_pending(captured))
        await srv.execute_code(_MANIP, style="notebook")
        assert captured["params"]["mode"] == "kernel"

    async def test_plain_code_stays_kernel(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(srv, "_addon_result", _fake_addon_pending(captured))
        await srv.execute_code("1 + 1", output_target="notebook")
        assert captured["params"]["mode"] == "kernel"

    async def test_lean_evaluate_inherits_routing(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(srv, "_addon_result", _fake_addon_pending(captured))
        await srv.evaluate(code=_MANIP, target="notebook")
        assert captured["params"]["mode"] == "frontend"


class TestResponseDetailFrozenKeys:
    def test_standard_does_not_modify_frozen_keys(self):
        """Verify that response_detail='standard' preserves all REQUIRED_SUCCESS_KEYS."""
        import json
        import time

        from mathematica_mcp.server import _finalize_execute_response

        required = {"success", "output", "output_inputform", "output_fullform", "warnings", "timing_ms"}
        response = {k: "test" for k in required}
        result_json = _finalize_execute_response(
            response,
            route_variant="compute",
            execution_path="addon_cli",
            fell_back=False,
            start_time=time.monotonic(),
            response_detail="standard",
        )
        parsed = json.loads(result_json)
        missing = required - set(parsed.keys())
        assert not missing, f"Missing keys in standard response: {missing}"
