"""G4' registration parity (generated, not hardcoded).

classic must expose exactly the current full surface; lean must expose exactly
the 12 consolidated tools; MATHEMATICA_TOOLSETS must add opt-in extras to lean.
The classic==full check is generated at runtime, so it can never drift from the
real registry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

_SCRIPT = """
import asyncio, json
from mathematica_mcp.server import mcp
print(json.dumps(sorted(t.name for t in asyncio.run(mcp.list_tools()))))
"""

LEAN_TOOLS = {
    "status",
    "notebooks",
    "cells",
    "edit_cells",
    "evaluate",
    "screenshot",
    "verify_derivation",
    "kernel",
    "vars",
    "read_notebook_file",
    "guide",
    "batch",
}


def _names(profile: str, extra_env: dict[str, str] | None = None) -> set[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MATHEMATICA_")}
    env["MATHEMATICA_PROFILE"] = profile
    env["PYTHONPATH"] = str(SRC_ROOT)
    if extra_env:
        env.update(extra_env)
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    ).stdout
    return set(json.loads(out))


def test_classic_equals_full_surface():
    """classic is the byte-identical current surface — generated, not hardcoded."""
    assert _names("classic") == _names("full")


def test_classic_is_the_current_82():
    assert len(_names("classic")) == 82


def test_lean_registers_exactly_the_twelve():
    assert _names("lean") == LEAN_TOOLS


def test_lean_within_tool_budget():
    assert len(_names("lean")) <= 18


def test_verify_derivation_present_in_lean_and_classic():
    # The moat tool is shared: it must appear in both surfaces.
    assert "verify_derivation" in _names("lean")
    assert "verify_derivation" in _names("classic")


def test_toolset_adds_data_io_to_lean():
    base = _names("lean")
    extended = _names("lean", {"MATHEMATICA_TOOLSETS": "data_io"})
    assert "import_data" in extended
    assert "import_data" not in base
    assert extended >= LEAN_TOOLS  # extras add to, never replace, the lean core


def test_toolset_adds_symbols_feature_to_lean():
    extended = _names("lean", {"MATHEMATICA_TOOLSETS": "symbols"})
    assert "resolve_function" in extended


def test_toolsets_do_not_affect_classic():
    # Toolset opt-ins are lean-only; classic already has everything.
    assert _names("classic", {"MATHEMATICA_TOOLSETS": "data_io,symbols"}) == _names("classic")
