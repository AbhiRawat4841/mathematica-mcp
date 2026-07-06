"""Lean-profile guidance: instructions and prompt builders must use lean vocabulary."""

from __future__ import annotations

import pytest

from mathematica_mcp.config import (
    PROFILE_FEATURE_DEFAULTS,
    PROFILE_TOOL_GROUPS,
    FeatureFlags,
)
from mathematica_mcp.guidance import (
    build_claude_command,
    build_claude_hint,
    build_codex_guidance,
    build_mathematica_expert_prompt,
    build_prompt_calculate,
    build_prompt_interactive,
    build_prompt_new_notebook,
    build_prompt_notebook,
    build_prompt_quickstart,
    build_server_instructions,
)

# Tool names / vocabulary that do not exist in the lean 12-tool surface.
FORBIDDEN_IN_LEAN = [
    "execute_code",
    "style=",
    "style='",
    "get_session_brief",
    "get_computation_journal",
    "create_notebook",
    "write_cell",
    "response_detail",
    "not exposed",
]


def _flags(profile: str) -> FeatureFlags:
    d = PROFILE_FEATURE_DEFAULTS[profile]
    return FeatureFlags(
        profile=profile,
        tool_groups=PROFILE_TOOL_GROUPS[profile],
        function_repository=d["function_repository"],
        data_repository=d["data_repository"],
        async_computation=d["async_computation"],
        symbol_lookup=d["symbol_lookup"],
        math_aliases=d["math_aliases"],
        expression_cache=d["expression_cache"],
        telemetry=d["telemetry"],
        cache_tools=False,
        routing_memory="off",
        routing_action="off",
        default_output_target="cli" if profile in ("math", "lean") else "notebook",
    )


LEAN = _flags("lean")
CLASSIC = _flags("classic")


def _assert_lean_clean(text: str) -> None:
    for forbidden in FORBIDDEN_IN_LEAN:
        assert forbidden not in text, f"lean guidance mentions absent vocabulary: {forbidden!r}"


class TestLeanServerInstructions:
    def test_no_absent_tools(self):
        _assert_lean_clean(build_server_instructions(LEAN))

    def test_mentions_lean_tools(self):
        text = build_server_instructions(LEAN)
        for name in ("evaluate", "notebooks", "verify_derivation", "guide"):
            assert name in text

    def test_stays_compact(self):
        assert len(build_server_instructions(LEAN)) <= 1600

    def test_classic_unchanged_vocabulary(self):
        text = build_server_instructions(CLASSIC)
        assert "execute_code" in text
        assert 'style="compute"' in text


class TestLeanExpertPrompt:
    def test_no_absent_tools(self):
        _assert_lean_clean(build_mathematica_expert_prompt("integrate x^2", features=LEAN))

    def test_routes_to_lean_tools(self):
        text = build_mathematica_expert_prompt("integrate x^2", features=LEAN)
        for name in ("evaluate", "notebooks", "verify_derivation", "guide", "status()", 'kernel(action="messages")'):
            assert name in text

    def test_classic_keeps_legacy_routing(self):
        text = build_mathematica_expert_prompt("integrate x^2", features=CLASSIC)
        assert "execute_code" in text
        assert "get_session_brief" in text


class TestLeanClientGuidance:
    @pytest.mark.parametrize("builder", [build_claude_hint, build_claude_command, build_codex_guidance])
    def test_no_absent_tools(self, builder):
        _assert_lean_clean(builder(LEAN))

    @pytest.mark.parametrize("builder", [build_claude_hint, build_claude_command, build_codex_guidance])
    def test_mentions_evaluate(self, builder):
        assert "evaluate" in builder(LEAN)


class TestPromptBuilders:
    def test_calculate(self):
        lean = build_prompt_calculate(LEAN, "1+1")
        classic = build_prompt_calculate(CLASSIC, "1+1")
        _assert_lean_clean(lean)
        assert "evaluate(code)" in lean and "1+1" in lean
        assert "style='compute'" in classic and "1+1" in classic

    def test_notebook(self):
        lean = build_prompt_notebook(LEAN, "plot it")
        classic = build_prompt_notebook(CLASSIC, "plot it")
        _assert_lean_clean(lean)
        assert "target='notebook'" in lean and "plot it" in lean
        assert "style='notebook'" in classic

    def test_new_notebook(self):
        lean = build_prompt_new_notebook(LEAN, "plot it", title="T")
        classic = build_prompt_new_notebook(CLASSIC, "plot it", title="T")
        _assert_lean_clean(lean)
        assert "notebooks(action='create'" in lean and "'T'" in lean
        assert "create_notebook()" in classic and "style='notebook'" in classic

    def test_interactive(self):
        lean = build_prompt_interactive(LEAN, "Manipulate a slider")
        classic = build_prompt_interactive(CLASSIC, "Manipulate a slider")
        _assert_lean_clean(lean)
        assert "target='notebook'" in lean
        assert "style='interactive'" in classic

    def test_quickstart(self):
        lean = build_prompt_quickstart(LEAN)
        classic = build_prompt_quickstart(CLASSIC)
        _assert_lean_clean(lean)
        for name in ("evaluate", "notebooks", "verify_derivation", "screenshot"):
            assert name in lean
        assert 'execute_code(style="notebook")' in classic
        assert "create_notebook" in classic


def test_lean_default_output_target_is_cli():
    # F3: lean evaluate defaults to target="kernel" (cli); guidance must agree.
    assert FeatureFlags.from_env(profile_override="lean").default_output_target == "cli"
    assert FeatureFlags.from_env(profile_override="classic").default_output_target == "notebook"
