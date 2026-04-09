"""Tests for the execution style parameter and resolver."""

from __future__ import annotations

import asyncio

import pytest

from mathematica_mcp.server import _normalize_response_detail, _resolve_execution_params


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
