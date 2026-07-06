"""Release correctness: the package __version__ must match pyproject.toml."""

from __future__ import annotations

from pathlib import Path

import tomllib

import mathematica_mcp


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert mathematica_mcp.__version__ == declared
