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


def test_notebook_profile_codex_guidance_includes_all_styles():
    guidance, features = _reload_guidance("notebook")

    codex = guidance.build_codex_guidance(features)

    assert 'style="compute"' in codex
    assert 'style="notebook"' in codex
    assert 'style="interactive"' in codex


def test_math_profile_command_omits_notebook_flows():
    guidance, features = _reload_guidance("math")

    command = guidance.build_claude_command(features)

    assert 'style="compute"' in command
    assert 'style="notebook"' not in command
    assert 'style="interactive"' not in command
