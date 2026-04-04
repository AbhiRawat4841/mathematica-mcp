"""Tests for the execution style parameter and resolver."""

from __future__ import annotations

import pytest

from mathematica_mcp.server import _resolve_execution_params


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
