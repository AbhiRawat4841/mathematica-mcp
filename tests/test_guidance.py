from __future__ import annotations

import importlib


def _reload_server(profile: str | None = None):
    if profile is None:
        import os

        os.environ.pop("MATHEMATICA_PROFILE", None)
    else:
        import os

        os.environ["MATHEMATICA_PROFILE"] = profile

    config = importlib.reload(importlib.import_module("mathematica_mcp.config"))
    server = importlib.reload(importlib.import_module("mathematica_mcp.server"))
    return config, server


def _reload_guidance(profile: str):
    import os

    os.environ["MATHEMATICA_PROFILE"] = profile
    config = importlib.reload(importlib.import_module("mathematica_mcp.config"))
    guidance = importlib.reload(importlib.import_module("mathematica_mcp.guidance"))
    features = config.FeatureFlags.from_env()
    return guidance, features


def test_math_prompt_is_compute_first():
    _, server = _reload_server("math")

    prompt = server.mathematica_expert("integrate x^2")

    assert "PROFILE: `math`" in prompt
    assert 'style="compute"' in prompt
    assert "create_notebook" not in prompt


def test_notebook_prompt_includes_notebook_antipattern():
    _, server = _reload_server("notebook")

    prompt = server.mathematica_expert("plot sin x")

    assert "PROFILE: `notebook`" in prompt
    assert "NEVER: `create_notebook` -> `write_cell` -> `evaluate_cell`" in prompt
    assert 'execute_code(code, style="notebook")' in prompt
    assert "reuses the active notebook" in prompt
    assert 'sync="none"' in prompt
    assert "screenshot_notebook()" in prompt


def test_profile_aware_docstrings_are_applied():
    _, server = _reload_server("math")

    assert "[PRIMARY]" in (server.execute_code.__doc__ or "")
    assert 'style="compute"' in (server.execute_code.__doc__ or "")
    assert "[ADVANCED]" in (server.write_cell.__doc__ or "")
    assert "[LEGACY]" in (server.read_notebook_content.__doc__ or "")


def test_math_profile_hint_omits_notebook_flows():
    guidance, features = _reload_guidance("math")

    hint = guidance.build_claude_hint(features)

    assert 'style="compute"' in hint
    assert 'style="notebook"' not in hint
    assert 'style="interactive"' not in hint


def test_notebook_profile_hint_includes_all_styles():
    guidance, features = _reload_guidance("notebook")

    hint = guidance.build_claude_hint(features)

    assert 'style="compute"' in hint
    assert 'style="notebook"' in hint
    assert 'style="interactive"' in hint


def test_math_profile_codex_guidance_omits_notebook_flows():
    guidance, features = _reload_guidance("math")

    codex = guidance.build_codex_guidance(features)

    assert 'style="compute"' in codex
    assert 'style="notebook"' not in codex
    assert 'style="interactive"' not in codex
    assert "create_notebook" not in codex
    assert "read_notebook()" not in codex
    assert "LIVE WINDOW" not in codex
    assert "Notebook tools are not exposed" in codex
    assert 'sync="none"' not in codex
    assert "## Key concept" not in codex
    assert "\n\nThis profile is compute-first." in codex


def test_notebook_profile_codex_guidance_includes_all_styles():
    guidance, features = _reload_guidance("notebook")

    codex = guidance.build_codex_guidance(features)

    assert 'style="compute"' in codex
    assert 'style="notebook"' in codex
    assert 'style="interactive"' in codex
    assert "read_notebook()" in codex
    assert 'response_detail="compact"' not in codex
    assert "get_session_brief()" not in codex
    assert 'sync="none"' not in codex


def test_math_profile_command_omits_notebook_flows():
    guidance, features = _reload_guidance("math")

    command = guidance.build_claude_command(features)

    assert 'style="compute"' in command
    assert 'style="notebook"' not in command
    assert 'style="interactive"' not in command
    assert "create_notebook" not in command
    assert 'sync="none"' not in command


def test_server_instructions_include_quick_defaults_and_recovery_defaults():
    guidance, features = _reload_guidance("full")

    instructions = guidance.build_server_instructions(features)

    assert 'response_detail="compact"' in instructions
    assert "get_session_brief()" in instructions
    assert "get_computation_journal()" in instructions
    assert 'sync="none"' in instructions
    assert "reuses the active notebook" in instructions


def test_math_profile_server_instructions_omit_notebook_guidance():
    guidance, features = _reload_guidance("math")

    instructions = guidance.build_server_instructions(features)

    assert "Notebook tools are not exposed" in instructions
    assert "LIVE WINDOW" not in instructions
    assert "create_notebook" not in instructions
    assert 'style="notebook"' not in instructions
    assert 'style="interactive"' not in instructions
    assert 'sync="none"' not in instructions


def test_claude_hint_is_additive_and_lean():
    guidance, features = _reload_guidance("full")

    hint = guidance.build_claude_hint(features)

    assert 'response_detail="compact"' not in hint
    assert "get_session_brief()" not in hint
    assert 'sync="none"' not in hint
    assert len(hint.split()) < 180


def test_guidance_word_budgets_stay_reasonable():
    guidance, full_features = _reload_guidance("full")
    _, math_features = _reload_guidance("math")

    assert len(guidance.build_server_instructions(full_features).split()) < 380
    assert len(guidance.build_codex_guidance(full_features).split()) < 260
    assert len(guidance.build_codex_guidance(math_features).split()) < 140


def test_always_on_combined_budgets_stay_reasonable():
    guidance, full_features = _reload_guidance("full")

    server = guidance.build_server_instructions(full_features)
    codex = guidance.build_codex_guidance(full_features)
    hint = guidance.build_claude_hint(full_features)

    assert len((server + "\n" + codex).split()) < 600
    assert len((server + "\n" + hint).split()) < 500


# ---------------------------------------------------------------------------
# Session brief
# ---------------------------------------------------------------------------


def test_session_brief_includes_profile_and_connection():
    guidance, features = _reload_guidance("full")
    brief = guidance.build_session_brief(features, connection_mode="addon")
    assert "full" in brief
    assert "addon" in brief


def test_session_brief_recent_errors_shown():
    guidance, features = _reload_guidance("full")
    brief = guidance.build_session_brief(features, recent_errors=["Syntax::sntxi", "Part::partw"])
    assert "Syntax" in brief
    assert "Part" in brief


def test_session_brief_no_errors_says_none():
    guidance, features = _reload_guidance("full")
    brief = guidance.build_session_brief(features, recent_errors=[])
    assert "none" in brief.lower()


def test_session_brief_routing_hints_shown():
    guidance, features = _reload_guidance("full")
    brief = guidance.build_session_brief(features, routing_hints=["Plot timeout 40%"])
    assert "Plot timeout" in brief


def test_session_brief_no_hints_omits_section():
    guidance, features = _reload_guidance("full")
    brief = guidance.build_session_brief(features, routing_hints=[])
    assert "Routing" not in brief


def test_session_brief_compact_length():
    guidance, features = _reload_guidance("full")
    brief = guidance.build_session_brief(
        features,
        connection_mode="addon",
        recent_errors=["Syntax"],
        routing_hints=["Plot timeout 40%"],
    )
    assert len(brief) < 600
