"""Release correctness: the package __version__ must match pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

import mathematica_mcp


def _pyproject_version(text: str) -> str:
    # tomllib is stdlib only on 3.11+; the project supports 3.10, so parse the
    # one line we need instead of importing a TOML parser.
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m is not None, "no version line in pyproject.toml"
    return m.group(1)


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert mathematica_mcp.__version__ == _pyproject_version(pyproject.read_text())
