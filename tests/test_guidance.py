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


def test_math_prompt_is_compute_first():
    _, server = _reload_server("math")

    prompt = server.mathematica_expert("integrate x^2")

    assert "PROFILE: `math`" in prompt
    assert 'execute_code(..., output_target="cli", mode="kernel")' in prompt
    assert "create_notebook" not in prompt


def test_notebook_prompt_includes_notebook_antipattern():
    _, server = _reload_server("notebook")

    prompt = server.mathematica_expert("plot sin x")

    assert "PROFILE: `notebook`" in prompt
    assert "NEVER: `create_notebook` -> `write_cell` -> `evaluate_cell`" in prompt
    assert 'execute_code(code, output_target="notebook")' in prompt


def test_profile_aware_docstrings_are_applied():
    _, server = _reload_server("math")

    assert "[PRIMARY]" in (server.execute_code.__doc__ or "")
    assert "Profile default when `output_target` is omitted: `cli`." in (
        server.execute_code.__doc__ or ""
    )
    assert "[ADVANCED]" in (server.write_cell.__doc__ or "")
    assert "[LEGACY]" in (server.read_notebook_content.__doc__ or "")
